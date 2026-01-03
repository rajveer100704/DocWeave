import math
import os
import re
import sys
from typing import Dict, List, Sequence

import yaml
import tiktoken
from langchain_core.documents import Document

from src.exception import MyException
from src.logger import logging


def num_tokens_from_string(text: str, model_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a common encoding if model_name is not directly supported
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e


def load_configs(config_dir: str) -> Dict:
    """Load all configuration files from the specified directory."""
    try:
        configs = {}
        for name in ["ingestion", "chunking", "retrieval", "generation", "pipeline"]:
            path = os.path.join(config_dir, f"{name}.yaml")
            if os.path.exists(path):
                configs.update(read_yaml_file(path) or {})
            else:
                logging.warning("Config file missing: %s", path)
        return configs
    except Exception as e:
        raise MyException(e, sys) from e


def compute_k(*, total: int, pct: float | None, upper_bound: int) -> int:
    """
    Convert a percentage to an integer k, clamped to available docs.

    - Uses ceil to avoid losing small fractions.
    - Ensures the value is >= 0 and <= upper_bound.
    """
    if total <= 0 or upper_bound <= 0:
        return 0

    if pct is None:
        return 0

    calculated = int(math.ceil(total * pct))
    return max(0, min(calculated, upper_bound))


def count_documents(vector_store) -> int:
    """
    Safely count documents in a FAISS store.
    """
    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        try:
            return len(ids)
        except Exception:
            pass

    docstore = getattr(vector_store, "docstore", None)
    if docstore is not None and hasattr(docstore, "_dict"):
        try:
            return len(docstore._dict)
        except Exception:
            pass
    return 0


# ----------------------------
# RAG Pipeline Helper Functions
# ----------------------------

def build_context(docs: Sequence[Document], include_citations: bool = False) -> str:
    """Concatenate documents into a single context string."""
    # Note: include_citations parameter kept for API compatibility but not implemented
    # Citations are handled separately via extract_sources()
    parts = [doc.page_content for doc in docs]
    return "\n\n".join(parts)


def _build_highlight_pattern(answer_text: str) -> re.Pattern | None:
    """
    Build a regex pattern for highlightable keywords from the answer text.
    Keeps unique words longer than 3 characters.
    """
    if not answer_text:
        return None
    words = re.findall(r"\b\w+\b", answer_text.lower())
    keywords = [w for w in words if len(w) > 3]
    deduped = list(dict.fromkeys(keywords))
    if not deduped:
        return None
    escaped = [re.escape(w) for w in deduped[:50]]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def highlight_overlap(text: str, answer_text: str) -> str:
    """
    Highlight sentences (not individual words) from the source chunk that overlap
    with words in the answer. Returns HTML with <mark> wrapped sentences.
    """
    pattern = _build_highlight_pattern(answer_text)
    if not pattern:
        return text

    # Split into sentences while retaining punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    highlighted = []
    for sent in sentences:
        if pattern.search(sent):
            highlighted.append(f"<mark>{sent}</mark>")
        else:
            highlighted.append(sent)
    return " ".join(highlighted)


def extract_sources(docs: Sequence[Document], answer_text: str | None = None) -> List[Dict[str, str]]:
    """
    Extract source information from documents with deduplication.
    Returns filename-only paths and includes the full chunk plus a highlighted version.
    """
    sources = []
    seen_sources = set()
    
    for doc in docs:
        meta = doc.metadata or {}
        source_path = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        
        # Normalize path to show only filename, not full path
        if source_path and source_path != "unknown":
            normalized_path = os.path.basename(source_path)
        else:
            normalized_path = "unknown"
        
        # Create a unique key for deduplication
        source_key = f"{normalized_path}|{page}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            
            # Format page info
            page_info = f"Page {page}" if page != "N/A" else "N/A"
            
            chunk_text = doc.page_content
            highlighted_chunk = highlight_overlap(chunk_text, answer_text) if answer_text else chunk_text
            
            sources.append({
                "path": normalized_path,
                "page_info": page_info,
                "snippet": chunk_text,
                "chunk": chunk_text,
                "highlighted_chunk": highlighted_chunk,
            })
    
    return sources

