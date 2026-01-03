import sys
from typing import List, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.exception import MyException
from src.logger import logging


class CrossEncoderReranker:
    """
    Lightweight cross-encoder based reranker.

    Performs re-ranking on a list of retrieved documents using a
    sentence-transformers CrossEncoder model. Returns documents ordered
    by the cross-encoder relevance score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        try:
            logging.info("Loading cross-encoder reranker model: %s", model_name)
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            raise MyException(e, sys)

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int | None = None
    ) -> List[Document]:
        """
        Score and reorder candidate documents.

        Args:
            query: User query string.
            documents: Candidate documents to score.
            top_k: Optional cap on number of documents to return.

        Returns:
            Documents ordered by cross-encoder score (highest first).
        """
        if not documents:
            return []

        try:
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)
            scored = sorted(
                zip(documents, scores), key=lambda item: item[1], reverse=True
            )
            if top_k is not None:
                scored = scored[:top_k]
            reranked_docs = [doc for doc, _ in scored]
            logging.debug(
                "Reranked %d documents, returning %d", len(documents), len(reranked_docs)
            )
            return reranked_docs
        except Exception as e:
            raise MyException(e, sys)

