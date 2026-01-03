import os
import sys
from typing import Dict, List, Sequence, Any

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from src.exception import MyException
from src.ingestion.extractor import DocumentExtractor
from src.ingestion.loaders import DocumentLoader
from src.logger import logging
from src.preprocessing.clean_normalize import DocumentNormalizationAndCleaning
from src.preprocessing.chunking import DocumentChunker
from src.rag import prompts
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RerankMMRRetriever
from src.utils.main_utils import (
    num_tokens_from_string,
    load_configs,
    build_context,
    extract_sources,
)
from src.vectorstore.faiss_store import FaissVectorStore


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1) Load + clean + chunk documents
    2) Build vector store
    3) Retrieve -> rerank -> MMR
    4) Choose Stuff vs Map-Reduce prompting and query LLM
    """

    def __init__(self, config_dir: str = "configs"):
        self.config = load_configs(config_dir)

        gen_cfg = self.config.get("generation", {})
        llm_kwargs = {
            "model": gen_cfg.get("llm_model"),
            "temperature": gen_cfg.get("temperature"),
        }
        # Add max_output_tokens if specified (some models support this)
        max_tokens = gen_cfg.get("max_output_tokens")
        if max_tokens:
            llm_kwargs["num_predict"] = max_tokens  # Ollama uses num_predict for max tokens
        self.llm = ChatOllama(**llm_kwargs)

        retr_cfg = self.config.get("retrieval", {})
        self.reranker = CrossEncoderReranker(
            model_name=retr_cfg.get("reranker_model")
        )

        self.vector_store = None
        self.retriever = None

    # ----------------------------
    # Data preparation
    # ----------------------------
    def prepare_vector_store(self) -> None:
        """Load documents, clean, chunk, and build FAISS vector store."""
        try:
            docs_cfg = self.config.get("documents", [])
            if not docs_cfg:
                raise MyException("No documents configured for processing.", sys)
            
            chunk_cfg = self.config.get("chunking", {})
            target_chunk_size = chunk_cfg.get("target_chunk_size")
            chunk_overlap = chunk_cfg.get("chunk_overlap")
            
            logging.info("Starting vector store preparation with %d document(s)", len(docs_cfg))

            # Initialize processing components
            loader = DocumentLoader()
            extractor = DocumentExtractor()
            cleaner = DocumentNormalizationAndCleaning()
            chunker = DocumentChunker()

            # Process each document
            all_chunks = []
            for idx, doc_info in enumerate(docs_cfg, 1):
                if not doc_info.get("enabled", True):
                    logging.info("Skipping disabled document: %s", doc_info.get("path", "unknown"))
                    continue
                
                path = doc_info["path"]
                logging.info("[%d/%d] Processing document: %s", idx, len(docs_cfg), path)

                try:
                    # Pipeline: load -> extract -> clean -> chunk
                    loaded = loader.load_document(path)
                    extracted = extractor.extract_document_info(loaded, path)
                    cleaned = cleaner.initialize_document_normalizer(extracted)
                    chunks = chunker.chunk_document(cleaned, target_chunk_size, chunk_overlap)
                    
                    logging.info("Generated %d chunks from document: %s", len(chunks), path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logging.error("Failed to process document %s: %s", path, e)
                    raise MyException(f"Error processing document {path}: {e}", sys)

            if not all_chunks:
                raise MyException("No chunks generated; check document config and ensure documents are enabled.", sys)

            # Create vector store and retriever
            logging.info("Creating vector store with %d total chunks...", len(all_chunks))
            self.vector_store = FaissVectorStore().create_vector_store(all_chunks)
            self.retriever = RerankMMRRetriever(self.vector_store, self.reranker)
            logging.info("Vector store prepared successfully with %d chunks", len(all_chunks))
        except Exception as e:
            logging.exception("Failed to prepare vector store: %s", e)
            raise MyException(e, sys)

    # ----------------------------
    # Retrieval + Routing
    # ----------------------------
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if self.retriever is None:
            raise MyException("Retriever not initialized. Call prepare_vector_store().", sys)

        query_preview = query[:100] if len(query) > 100 else query
        logging.info("Retrieving documents for query: %s", query_preview)
        
        retr_cfg = self.config.get("retrieval", {})
        retrieve_kwargs = {}
        
        # Add retrieval parameters if present in config
        optional_keys = ["lambda_mult", "initial_k", "rerank_k", "mmr_k", "initial_pct", "rerank_pct", "mmr_pct", "min_chunk"]
        for key in optional_keys:
            if key in retr_cfg:
                retrieve_kwargs[key] = retr_cfg[key]

        documents = self.retriever.retrieve(query, **retrieve_kwargs)
        logging.info("Retrieved %d documents for query", len(documents))
        return documents

    def answer(self, query: str) -> str:
        """Retrieve context and generate an answer with grounding + citations."""
        result = self.answer_with_sources(query)
        return result.get("answer", "")

    def answer_with_sources(self, query: str) -> Dict[str, Any]:
        """Retrieve context and generate an answer with sources."""
        try:
            query_preview = query[:100] if len(query) > 100 else query
            logging.info("Generating answer for query: %s", query_preview)
            
            documents = self.retrieve(query)
            if not documents:
                logging.warning("No documents retrieved for query: %s", query)
                return {
                    "answer": "I don't have enough information to answer this question based on the provided documents.",
                    "sources": []
                }

            gen_cfg = self.config.get("generation", {})
            token_limit = gen_cfg.get("stuff_context_token_limit")
            
            total_tokens = sum(num_tokens_from_string(doc.page_content) for doc in documents)
            logging.info("Total context tokens: %d (limit: %d)", total_tokens, token_limit)
            
            # Choose strategy based on token count
            if total_tokens <= token_limit:
                answer = self._answer_with_stuff(query, documents)
            else:
                answer = self._answer_with_map_reduce(query, documents)
            
            sources = extract_sources(documents, answer_text=answer)
            logging.info("Answer generated successfully (length: %d chars, sources: %d)", len(answer), len(sources))
            
            return {"answer": answer, "sources": sources}
        except Exception as e:
            logging.exception("Failed to generate answer: %s", e)
            raise MyException(e, sys)

    # ----------------------------
    # Prompting strategies
    # ----------------------------
    def _answer_with_stuff(self, query: str, docs: Sequence[Document]) -> str:
        """Generate answer using Stuff strategy (all context in one prompt)."""
        # Build context without citations for clean answer
        context_str = build_context(docs, include_citations=False)
        
        # Build the user prompt
        prompt_tpl = prompts.build_stuff_prompt()
        user_prompt = prompt_tpl.format(
            context=context_str,
            question=query
        )
        
        # Use proper message format for ChatOllama
        messages = [
            SystemMessage(content=prompts.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logging.info("Using Stuff strategy with %d docs", len(docs))
        logging.info("Context length: %d characters", len(context_str))
        # Sanitize context preview for logging to avoid Unicode encoding issues
        try:
            context_preview = context_str[:300].encode('ascii', errors='replace').decode('ascii')
            logging.debug("Context preview (first 300 chars): %s", context_preview)
        except Exception:
            logging.debug("Context preview: [contains non-ASCII characters, length: %d]", len(context_str))
        
        response = self.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        logging.info("Generated answer length: %d characters", len(answer))
        # Sanitize answer preview for logging
        try:
            answer_preview = answer[:200].encode('ascii', errors='replace').decode('ascii') if answer else "Empty"
            logging.debug("Answer preview: %s", answer_preview)
        except Exception:
            logging.debug("Answer preview: [contains non-ASCII characters]")
        return answer

    def _answer_with_map_reduce(self, query: str, docs: Sequence[Document]) -> str:
        """Generate answer using Map-Reduce strategy (process each doc, then combine)."""
        map_tpl = prompts.build_map_prompt()
        map_outputs = []
        logging.info("Using Map-Reduce strategy with %d docs", len(docs))
        
        # Map: process each document individually (without citations in context)

        # Build the whole chunks context
        context_str = build_context(docs, include_citations=False)
        logging.info("Total context length for Map-Reduce: %d characters", len(context_str))
        # Map: process all doc together
        map_user_prompt = map_tpl.format(
            context=context_str,
            question=query
        )
        map_messages = [
            SystemMessage(content=prompts.SYSTEM_PROMPT),
            HumanMessage(content=map_user_prompt)
        ]
        res = self.llm.invoke(map_messages)
        map_res = getattr(res, "content", str(res))
        # for idx, doc in enumerate(docs, 1):
        #     ctx = doc.page_content  # Use clean content without citations
        #     map_user_prompt = map_tpl.format(
        #         context=ctx,
        #         question=query
        #     )
            
        #     map_messages = [
        #         SystemMessage(content=prompts.SYSTEM_PROMPT),
        #         HumanMessage(content=map_user_prompt)
        #     ]
            
        #     res = self.llm.invoke(map_messages)
        #     map_output = getattr(res, "content", str(res))
        #     map_outputs.append(map_output)
        #     logging.debug("Map output %d/%d: %d chars", idx, len(docs), len(map_output))

        # Reduce: combine all map outputs
        reduce_tpl = prompts.build_reduce_prompt()
        reduce_user_prompt = reduce_tpl.format(
            # map_summaries="\n\n".join(map_outputs),
            map_summaries=map_res,
            question=query,
        )
        
        reduce_messages = [
            SystemMessage(content=prompts.SYSTEM_PROMPT),
            HumanMessage(content=reduce_user_prompt)
        ]
        
        reduced = self.llm.invoke(reduce_messages)
        answer = getattr(reduced, "content", str(reduced))
        logging.info("Map-Reduce answer generated: %d characters", len(answer))
        return answer


