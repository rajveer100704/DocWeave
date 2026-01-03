import sys
from langchain_ollama.embeddings import OllamaEmbeddings
from src.logger import logging
from src.exception import MyException


class OllamaEmbedder:
    """
    Thin wrapper around LangChain's OllamaEmbeddings to delay initialization
    until it is actually needed.
    """

    def __init__(self, model_name: str = "embeddinggemma"):
        # Model name should eventually come from configuration/constants.
        self.model_name = model_name
        self._embedder = None

    def get_embedder(self) -> OllamaEmbeddings:
        """Create (once) and return the Ollama embedding model."""
        if self._embedder is None:
            try:
                logging.info("Initializing the Ollama embedder.")
                self._embedder = OllamaEmbeddings(model=self.model_name)
            except Exception as e:
                raise MyException(e, sys)
        return self._embedder