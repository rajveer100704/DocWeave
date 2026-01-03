from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.embedding.embedder import OllamaEmbedder
from src.exception import MyException
import sys

class FaissVectorStore:
    def __init__(self):
        """
        Initialize the Ollama Embedder
        """
        self.embedder = OllamaEmbedder().get_embedder()
        
    def create_vector_store(self, documents: list) -> FAISS:
            """This function create a FAISS vector store and return it.
            Args:
                documents (list): an list of chunk documents (dictionaries with 'text' and 'metadata')

            Raises:
                Exception: return an exception when, fails to initialise the vector store

            Returns:
                FAISS: return an vector store of FAISS
            """
            try:
                # Convert list of dictionaries to list of Document objects
                langchain_documents = []
                for i, doc in enumerate(documents):
                    if not isinstance(doc, dict):
                        raise TypeError(f"Expected a dictionary for document item {i}, but got {type(doc)}. Item: {doc}")
                    if 'text' not in doc or 'metadata' not in doc:
                        raise ValueError(f"Document item {i} is missing 'text' or 'metadata' key. Item: {doc}")
                    langchain_documents.append(Document(page_content=doc['text'], metadata=doc['metadata']))

                vector_store = FAISS.from_documents(langchain_documents, self.embedder)
                return vector_store
            except Exception as e:
                raise MyException(e, sys)