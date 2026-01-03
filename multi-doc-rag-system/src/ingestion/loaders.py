from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader, UnstructuredMarkdownLoader
from src.logger import logging
from src.exception import MyException
import sys

class DocumentLoader:
    def __init__(self):
        pass
    
    def load_document(self, document_path):
        """
        Loads a document from a given path or URL using the appropriate Langchain loader.
        Args:
            document_path (str): The path to the document file or a URL.
        Returns:
            list: A list of loaded documents.
        Raises:
            ValueError: If the document type is unsupported or the path is invalid.
        """
        logging.info(f"Attempting to load document from: {document_path}")
        try:
            if document_path.startswith(('http://', 'https://')):
                loader = WebBaseLoader(document_path)
            elif document_path.endswith('.pdf'):
                loader = PyPDFLoader(document_path)
            elif document_path.endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(document_path)
            elif document_path.endswith('.txt'):
                loader = TextLoader(document_path)
            elif document_path.endswith('.md'):
                loader = UnstructuredMarkdownLoader(document_path)
            else:
                raise MyException(f"Unsupported document type: {document_path}. Please provide a PDF, DOCX, TXT file, .MD file or a URL.", sys)

            document = loader.load()
            logging.info(f"Successfully loaded {len(document)} pages/parts from {document_path}")
            return document
        except Exception as e:
            logging.info(f"Error loading document {document_path}: {e}")
            raise MyException(f"Could not load document {document_path}. Error: {e}", sys)

