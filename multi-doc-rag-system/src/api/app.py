import os
import shutil
import tempfile
from enum import Enum
from hashlib import md5
from typing import List, Optional, Dict, Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.logger import logging
from src.rag.pipelines import RAGPipeline


app = FastAPI(title="RAG Service", version="1.0.0")

# Allow local dev from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    name: str
    path: str
    page_info: str
    snippet: str
    chunk: str | None = None
    highlighted_chunk: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class ProcessingStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class PipelineState:
    """
    Holds a single pipeline instance and reuses the vector store
    unless the document set changes.
    """

    def __init__(self) -> None:
        self.pipeline = RAGPipeline(config_dir="configs")
        self.current_fingerprint: Optional[str] = None
        self.tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
        self.status = ProcessingStatus.IDLE
        self.error_message: Optional[str] = None
        self.loaded_documents: List[Dict[str, str]] = []
        self.documents_config: List[dict] = []

    def _fingerprint(self, docs: List[dict]) -> str:
        """
        Create a fingerprint based on file content hash, not paths.
        This ensures same content = same fingerprint even if temp paths differ.
        """
        import hashlib
        
        key_parts = []
        for doc in sorted(docs, key=lambda x: x.get('path', '')):
            path = doc.get('path', '')
            enabled = doc.get('enabled', True)
            
            # For file paths, hash the file content
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    with open(path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    key_parts.append(f"{file_hash}|{enabled}")
                except Exception as e:
                    logging.warning(f"Could not hash file {path}: {e}, using path instead")
                    key_parts.append(f"{path}|{enabled}")
            else:
                # For URLs or non-existent files, use path
                key_parts.append(f"{path}|{enabled}")
        
        joined = "|".join(key_parts)
        return md5(joined.encode("utf-8")).hexdigest()

    async def _persist_uploads_async(self, uploads: List[UploadFile]) -> List[str]:
        """Save uploaded files asynchronously."""
        saved_paths: List[str] = []
        for upload in uploads:
            suffix = os.path.splitext(upload.filename)[-1]
            fd, path = tempfile.mkstemp(dir=self.tmp_dir, suffix=suffix)
            try:
                content = await upload.read()
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                saved_paths.append(path)
                logging.info("Saved upload %s to %s", upload.filename, path)
            except Exception as e:
                os.close(fd)
                os.remove(path)
                raise Exception(f"Failed to save {upload.filename}: {e}") from e
        return saved_paths

    def _prepare_docs_list(self, file_paths: List[str], url: Optional[str]) -> List[dict]:
        """Prepare the documents list from file paths and URL."""
        docs: List[dict] = []
        if file_paths:
            for p in file_paths:
                docs.append({"path": p, "enabled": True})
        if url:
            docs.append({"path": url, "enabled": True})
        return docs

    def _process_documents(self, docs: List[dict]) -> None:
        """Process documents and build vector store (runs in background)."""
        try:
            self.status = ProcessingStatus.PROCESSING
            self.error_message = None
            
            logging.info("Starting document processing for %d document(s)", len(docs))
            
            new_fp = self._fingerprint(docs)
            logging.info("Document fingerprint: %s (current: %s)", new_fp, self.current_fingerprint)
            
            if self.pipeline.vector_store is not None and new_fp == self.current_fingerprint:
                logging.info("Reusing existing vector store for unchanged documents.")
                self.status = ProcessingStatus.READY
                return

            # Clear old vector store and retriever to prevent context leakage
            logging.info("Clearing old vector store and creating new one...")
            self.pipeline.vector_store = None
            self.pipeline.retriever = None

            # Override documents config dynamically
            self.pipeline.config["documents"] = docs
            self.documents_config = docs
            self.pipeline.prepare_vector_store()
            self.current_fingerprint = new_fp
            self.status = ProcessingStatus.READY
            
            # Update loaded documents list for status endpoint
            self.loaded_documents = [
                {
                    "name": os.path.basename(doc.get("path", "unknown")),
                    "path": doc.get("path", "unknown")
                }
                for doc in docs
            ]
            
            logging.info("Vector store prepared successfully for %d document(s).", len(docs))
        except Exception as e:
            self.status = ProcessingStatus.ERROR
            self.error_message = str(e)
            logging.exception("Failed to process documents: %s", e)


state = PipelineState()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/load")
async def load_documents(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[str] = Form(None),
) -> dict:
    try:
        # Save uploaded files immediately (before background processing)
        file_paths = []
        if files:
            file_paths = await state._persist_uploads_async(files)
        
        docs = state._prepare_docs_list(file_paths, url)
        if not docs:
            raise HTTPException(status_code=400, detail="Provide at least one file or URL.")

        # Check if we can reuse existing vector store
        new_fp = state._fingerprint(docs)
        if state.pipeline.vector_store is not None and new_fp == state.current_fingerprint:
            return {
                "status": "ready",
                "message": "Documents already indexed. Ready for queries.",
            }

        # Start background processing
        background_tasks.add_task(state._process_documents, docs)
        return {
            "status": "processing",
            "message": "Document processing started. Check /status endpoint for progress.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed to initiate document loading: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_status() -> dict:
    """Get the current processing status."""
    status_info = {
        "status": state.status.value,
        "loaded_documents": state.loaded_documents,
    }
    if state.status == ProcessingStatus.ERROR:
        status_info["error"] = state.error_message
    elif state.status == ProcessingStatus.READY:
        status_info["message"] = "Ready for queries."
    elif state.status == ProcessingStatus.PROCESSING:
        status_info["message"] = "Processing documents..."
    return status_info


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        if state.status != ProcessingStatus.READY:
            if state.status == ProcessingStatus.PROCESSING:
                raise HTTPException(
                    status_code=503,
                    detail="Documents are still being processed. Please wait and check /status endpoint.",
                )
            elif state.status == ProcessingStatus.ERROR:
                raise HTTPException(
                    status_code=503,
                    detail=f"Pipeline is in error state: {state.error_message}",
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No documents loaded. Please load documents first using /load endpoint.",
                )
        
        # Get answer and sources from pipeline
        result = state.pipeline.answer_with_sources(payload.query)
        
        # Convert sources to the expected format
        sources = [
            Source(
                name=os.path.basename(src.get("path", "unknown")),
                path=src.get("path", "unknown"),
                page_info=src.get("page_info", "N/A"),
                snippet=src.get("snippet", "")[:200] + "..." if len(src.get("snippet", "")) > 200 else src.get("snippet", ""),
                chunk=src.get("chunk"),
                highlighted_chunk=src.get("highlighted_chunk"),
            )
            for src in result.get("sources", [])
        ]
        
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
def cleanup() -> dict:
    """Clear the indexed context."""
    try:
        state.pipeline.vector_store = None
        state.pipeline.retriever = None
        state.current_fingerprint = None
        state.documents_config = []
        state.loaded_documents = []
        state.status = ProcessingStatus.IDLE
        state.error_message = None
        
        logging.info("Pipeline cleaned up successfully")
        return {"message": "All indexed documents and context have been cleared."}
    except Exception as e:
        logging.exception("Cleanup failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class CleanupSelectedRequest(BaseModel):
    paths: List[str]


@app.post("/cleanup_selected")
def cleanup_selected(req: CleanupSelectedRequest) -> dict:
    """Remove selected indexed sources and rebuild the vector store."""
    try:
        if not state.documents_config:
            raise HTTPException(status_code=400, detail="No documents indexed to clear.")

        remaining_docs = [
            doc for doc in state.documents_config
            if doc.get("path") not in set(req.paths or [])
        ]

        # If nothing remains, fallback to full cleanup
        if not remaining_docs:
            cleanup_resp = cleanup()
            return {
                "message": "Selected sources cleared. No documents remain.",
                "status": state.status,
                "loaded_documents": state.loaded_documents,
                **cleanup_resp,
            }

        state.pipeline.vector_store = None
        state.pipeline.retriever = None
        state.current_fingerprint = None

        state._process_documents(remaining_docs)

        return {
            "message": "Selected sources cleared and context rebuilt.",
            "status": state.status,
            "loaded_documents": state.loaded_documents,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Partial cleanup failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
