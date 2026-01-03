from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.main_utils import num_tokens_from_string
from langchain_core.documents import Document
from src.logger import logging
from src.exception import MyException
import sys

class DocumentChunker:
    def __init__(self):
        pass

    def structure_aware_splitter(self, extracted_doc_dict):
        """
        Performs initial splitting based on natural document boundaries.
        Args:
            extracted_doc_dict (dict): A dictionary containing 'text' and 'metadata'.
                                    'metadata' must contain 'doc_type', 'source', 'page', 'section'.
        Returns:
            list: A list of dictionaries, each representing a structurally aware chunk
                with 'text' and updated 'metadata'.
        """
        if 'text' not in extracted_doc_dict or 'metadata' not in extracted_doc_dict:
            raise MyException("Input dictionary must contain 'text' and 'metadata' keys.")
        
        try:
            raw_text = extracted_doc_dict['text']
            metadata = extracted_doc_dict['metadata']
            doc_type = metadata['doc_type']

            logging.info(f"Applying structure-aware splitting for document type: {doc_type}")

            # Define separators based on document type
            if doc_type == 'txt':
                separators = ['\n\n', '\n', ' ', ''] # Prioritize paragraphs for plain text
            # Add more conditions for other document types if needed (e.g., 'md', 'web')
            # For simplicity, using same separators for now but can be customized later.
            else:
                separators = ['\n\n', '\n', ' ', ''] # Default separators

            # Initialize RecursiveCharacterTextSplitter for initial structural chunks
            # Larger chunk_size and no overlap for initial structural split
            text_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=2000,  # Larger chunks for initial structural split
                chunk_overlap=0,
                length_function=len, # Character count for initial split
                add_start_index=True
            )

            # Split the document's text
            # The splitter expects a list of Document objects, so create one from the raw text.
            doc_for_splitting = [Document(page_content=raw_text, metadata=metadata)]
            split_documents = text_splitter.split_documents(doc_for_splitting)

            # Format the split documents into the desired dictionary structure
            formatted_chunks = []
            for i, split_doc in enumerate(split_documents):
                chunk_metadata = split_doc.metadata.copy()
                # Update chunk metadata with more specific chunk information if needed
                chunk_metadata['chunk_id'] = i
                formatted_chunks.append({
                    'text': split_doc.page_content,
                    'metadata': chunk_metadata
                })
            logging.info(f"Original text split into {len(formatted_chunks)} structural chunks.")
            return formatted_chunks
        except Exception as e:
            raise MyException(e, sys)

    def length_based_refinement(self, structural_chunks: list, target_chunk_size: int, chunk_overlap: int) -> list:
        """
        Refines a list of structural chunks by further splitting any chunk exceeding a target
        length using a RecursiveCharacterTextSplitter with token-based length function.

        Args:
            structural_chunks (list): A list of dictionaries, each containing 'text' and 'metadata',
                                    output from structure_aware_splitter.
            target_chunk_size (int): The desired maximum token length for refined chunks.
            chunk_overlap (int): The number of tokens to overlap between sub-chunks.

        Returns:
            list: A list of dictionaries, each representing a refined chunk with 'text' and 'metadata'.
        """
        try:
            refined_chunks = []

            logging.info(f"Applying length-based refinement with target_chunk_size={target_chunk_size} and chunk_overlap={chunk_overlap}.")

            for i, structural_chunk in enumerate(structural_chunks):
                text = structural_chunk['text']
                metadata = structural_chunk['metadata'].copy()
                current_chunk_tokens = num_tokens_from_string(text)

                if current_chunk_tokens > target_chunk_size:
                    print(f"  Chunk {i} (original tokens: {current_chunk_tokens}) exceeds target. Further splitting...")
                    # Create a new splitter for this sub-splitting process
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=target_chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=num_tokens_from_string, # Use token counting
                        add_start_index=True
                    )
                    # Convert the structural chunk into a Document object for the splitter
                    doc_to_split = Document(page_content=text, metadata=metadata)
                    sub_documents = sub_splitter.split_documents([doc_to_split])

                    for j, sub_doc in enumerate(sub_documents):
                        sub_chunk_metadata = sub_doc.metadata.copy()
                        # Update chunk_id to reflect it's a sub-chunk
                        sub_chunk_metadata['chunk_id'] = f"{metadata.get('chunk_id', i)}-{j}"
                        refined_chunks.append({
                            'text': sub_doc.page_content,
                            'metadata': sub_chunk_metadata
                        })
                else:
                    print(f"  Chunk {i} (tokens: {current_chunk_tokens}) is within target. Adding directly.")
                    refined_chunks.append(structural_chunk)

            logging.info(f"Total refined chunks after length-based refinement: {len(refined_chunks)}")
            return refined_chunks
        except Exception as e:
            raise MyException(e, sys)
    
    def chunk_document(self, cleaned_doc_list: list, target_chunk_size: int = 500, chunk_overlap: int = 100) -> list:
        """
        Combines structure-aware splitting and length-based refinement into a single function
        for processing cleaned documents into final chunks.

        Args:
            cleaned_doc_list (list): A list of dictionaries, each containing 'text' and 'metadata'
                                    from the cleaned documents.
            target_chunk_size (int): The desired maximum token length for refined chunks.
            chunk_overlap (int): The number of tokens to overlap between sub-chunks.

        Returns:
            list: A list of dictionaries, each representing a final chunk with 'text' and 'metadata'.
        """
        logging.info("Starting document chunking process...")
        all_final_chunks = []
        try:
            for i, extracted_doc_dict in enumerate(cleaned_doc_list):
                print(f"Processing document {i+1}/{len(cleaned_doc_list)} for chunking...")
                # Step 1: Perform structure-aware splitting
                structural_chunks = self.structure_aware_splitter(extracted_doc_dict)

                # Step 2: Perform length-based refinement
                refined_chunks = self.length_based_refinement(
                    structural_chunks,
                    target_chunk_size=target_chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_final_chunks.extend(refined_chunks)

            logging.info(f"Document chunking process completed. Generated {len(all_final_chunks)} total final chunks from all documents.")
            return all_final_chunks
        except Exception as e:
            raise MyException(e, sys)
