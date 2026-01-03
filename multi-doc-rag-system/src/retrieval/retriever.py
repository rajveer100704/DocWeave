import sys
from typing import List, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.exception import MyException
from src.logger import logging
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.main_utils import compute_k, count_documents


class RerankMMRRetriever:
    """
    Retrieve -> rerank -> diversify (MMR) pipeline built around FAISS.

    Usage:
        retriever = RerankMMRRetriever(vector_store, reranker)
        docs = retriever.retrieve("query")
    """

    def __init__(
        self,
        vector_store: FAISS,
        reranker: CrossEncoderReranker,
        embedder: Embeddings | None = None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.embedder = embedder or getattr(vector_store, "embedding_function", None)

        if self.embedder is None:
            raise MyException("No embedding function available for MMR.", sys)

    def retrieve(
        self,
        query: str,
        *,
        initial_pct: float | None = None,
        rerank_pct: float | None = None,
        mmr_pct: float | None = None,
        lambda_mult: float = 0.5,
        min_chunk: int | None = None,
    ) -> List[Document]:
        """
        Run vector search -> rerank -> MMR over the reranked set.

        Args:
            query: Search query.
            initial_pct: Percentage of total chunks to fetch from vector search.
            rerank_pct: Percentage of initial_k to keep after reranking.
            mmr_pct: Percentage of rerank_k to keep after MMR.
            lambda_mult: Trade-off for MMR (1.0 = purely relevance).
            min_chunk: If total chunks <= min_chunk, skip rerank/MMR and return all.
        """
        try:
            total_docs = count_documents(self.vector_store)
            logging.info("Total documents in vector store: %d", total_docs)
            
            if total_docs == 0:
                logging.warning("Vector store is empty. No documents to retrieve.")
                return []
            
            # Short-circuit for small corpora
            if min_chunk is not None and total_docs <= min_chunk:
                logging.info(
                    "Total documents (%d) <= min_chunk (%d). Skipping rerank/MMR.",
                    total_docs,
                    min_chunk,
                )
                return self.vector_store.similarity_search(query, k=total_docs)
            
            initial_k_final = compute_k(
                total=total_docs,
                pct=initial_pct,
                upper_bound=total_docs,
            )
            rerank_k_final = compute_k(
                total=initial_k_final,
                pct=rerank_pct,
                upper_bound=initial_k_final,
            )
            mmr_k_final = compute_k(
                total=rerank_k_final,
                pct=mmr_pct,
                upper_bound=rerank_k_final,
            )

            if initial_k_final <= 0:
                logging.warning("Computed initial_k is 0. No documents will be retrieved.")
                return []
            
            if rerank_k_final <= 0:
                logging.warning("Computed rerank_k is 0. Adjusting to use at least 1 document.")
                rerank_k_final = min(1, initial_k_final)
            
            if mmr_k_final <= 0:
                logging.warning("Computed mmr_k is 0. Adjusting to use at least 1 document.")
                mmr_k_final = min(1, rerank_k_final)

            initial_docs = self.vector_store.similarity_search(query, k=initial_k_final)
            logging.info(
                "Initial vector search returned %d docs (k=%d)",
                len(initial_docs),
                initial_k_final,
            )

            reranked_docs = self.reranker.rerank(
                query, initial_docs, top_k=rerank_k_final
            )
            logging.info(
                "Reranked docs down to %d (rerank_k=%d)",
                len(reranked_docs),
                rerank_k_final,
            )

            diversified_docs = self._apply_mmr(
                query, reranked_docs, k=mmr_k_final, lambda_mult=lambda_mult
            )
            logging.info(
                "MMR selected %d docs (mmr_k=%d)", len(diversified_docs), mmr_k_final
            )
            return diversified_docs
        except Exception as e:
            raise MyException(e, sys)

    def _apply_mmr(
        self, query: str, candidates: Sequence[Document], k: int, lambda_mult: float
    ) -> List[Document]:
        """Apply maximal marginal relevance over reranked candidates."""
        if not candidates or k <= 0:
            return []

        query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        doc_vecs = [
            np.array(vec, dtype=np.float32)
            for vec in self.embedder.embed_documents(
                [doc.page_content for doc in candidates]
            )
        ]

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        while remaining and len(selected) < k:
            if not selected:
                # pick best relevance to query
                chosen = max(remaining, key=lambda idx: cosine(query_vec, doc_vecs[idx]))
            else:
                chosen = max(
                    remaining,
                    key=lambda idx: lambda_mult * cosine(query_vec, doc_vecs[idx])
                    - (1 - lambda_mult)
                    * max(
                        cosine(doc_vecs[idx], doc_vecs[sel_idx])
                        for sel_idx in selected
                    ),
                )
            selected.append(chosen)
            remaining.remove(chosen)

        return [candidates[idx] for idx in selected]

