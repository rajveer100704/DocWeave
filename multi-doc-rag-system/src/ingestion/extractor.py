import os
import sys
from src.logger import logging
from src.exception import MyException

class DocumentExtractor:
    def __init__(self):
        pass

    def extract_document_info(self, document: list, doc_path: str) -> list:
        """
        Extracts text, structure, and enhanced metadata from loaded Langchain document.
        Adds 'doc_type' and ensures 'source', 'page', and 'section' are present.
        Args:
            document (list): A list of Langchain Document objects.
            doc_path (str): The document path or URL used to load the document.
        Returns:
            list: A list of dictionaries, each containing extracted info for a document part (page).
        """
        try:
            logging.info("Start the data extraction process")
            all_extracted_data = [] # Initialize an empty list to store dictionaries

            # Determine document type once based on the doc_path
            doc_type = "unknown"
            if doc_path.startswith(('http://', 'https://')):
                doc_type = "web"
            else:
                _, ext = os.path.splitext(doc_path)
                if ext:
                    doc_type = ext.lstrip('.').lower()
                if doc_type == 'doc': # Handle .doc being treated as docx
                    doc_type = 'docx'

            for i, doc in enumerate(document):
                # Extract core content for the current doc
                text_content = doc.page_content
                metadata = {}   # Initialize an empty metadata

                # Collect metadata from the data
                current_doc_info = doc.metadata.copy()
                # Add the determined doc_type to this metadata
                metadata['doc_type'] = doc_type
                # Ensure source, page, and section are present (or default)
                # For 'source', normalize to show only filename, not full path
                source_path = current_doc_info.get('source', doc_path)
                if source_path and not source_path.startswith(('http://', 'https://')):
                    # Extract just the filename from the path
                    metadata['source'] = os.path.basename(source_path)
                else:
                    metadata['source'] = source_path  # Keep URLs as-is
                # For 'page', prefer existing page from metadata, otherwise use index + 1
                metadata['page'] = current_doc_info.get('page', i) + 1
                # For 'section', use existing section from metadata, otherwise 'N/A'
                metadata['section'] = current_doc_info.get('section', 'N/A')

                # Create a dictionary for the current document part and append it to the list
                all_extracted_data.append({
                    'text': text_content,
                    'metadata': metadata
                })
                logging.info(f"Successfully, extracted text and metadata for page/part {i+1}")
            return all_extracted_data
            
        except MyException:
            logging.error("Error occur during the data extraction")

        except Exception as e:
            raise MyException(e, sys)