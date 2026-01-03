import streamlit as st
import tempfile
import uuid
import os
import shutil
import logging
import time # Import time for placeholder simulation
from typing import List, Optional, Dict, Any
import html

# --- IMPORTANT ASSUMPTION ---
# The following imports are based on the code provided by the user.
# It is assumed that 'src.rag.pipelines' and 'src.exception' are available
# in the environment where this Streamlit app will be executed.
# We will use placeholder classes for local execution visibility but preserve the imports.
try:
    from src.rag.pipelines import RAGPipeline
    from src.exception import MyException
except ImportError:
    # Placeholder classes for local testing/visibility when 'src' is not available
    class RAGPipeline:
        def __init__(self, config_dir):
            self.config = {"documents": []}
            self.vector_store = None
            self.retriever = None
            self.tmp_dir = None
        def prepare_vector_store(self):
            # Simulate indexing time
            time.sleep(2)
            if not self.config.get("documents"):
                raise Exception("No documents configured.")
            # Simulate setting up the vector store
            self.vector_store = True 
            self.retriever = True
        def answer_with_sources(self, query):
            if not self.vector_store:
                raise Exception("Vector store not ready.")
            
            # --- STRUCTURED MARKDOWN RESPONSE (as requested) ---
            markdown_answer = f"""
            ## Detailed Analysis for Query: {query}

            The RAG model has processed your request and generated the following comprehensive response based on the indexed documents.

            ### **1. Key Summary Points**

            * The **Vector Store** is the core component, responsible for managing high-dimensional vector embeddings.
            * **Document Processing** involves chunking, cleaning, and metadata extraction before the final embedding step.
            * The system supports retrieval from PDF, DOCX, TXT, and URL sources, ensuring broad compatibility.
            * All data handling adheres to the configured security protocols (e.g., encryption at rest).

            ### **2. Retrieval Strategy**

            The system employs a Hybrid Retrieval strategy:

            1.  **Semantic Search:** Uses the query vector to find the most relevant document chunks based on meaning.
            2.  **Keyword Matching:** Supplements semantic search by identifying exact keyword matches for highly specific queries.

            This combined approach maximizes both relevance and precision.

            ### **3. Conclusion**

            Based on the indexed context, the information provided above is grounded and validated against the original sources.

            """
            return {
                "answer": markdown_answer,
                "sources": [
                    {"path": "file1.pdf", "snippet": "The quick brown fox jumps <mark>over the lazy dog</mark> (page 1)."},
                    {"path": "report/data/file2.txt", "snippet": "A key finding is that the <mark>Simulated RAG architecture is robust</mark> and handles concurrent document processing efficiently."},
                ]
            }
    class MyException(Exception):
        pass


# --- Configuration and Initial Setup ---
st.set_page_config(
    page_title="Q&A RAG App by Jeet Majumder",
    page_icon="static/images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "status" not in st.session_state:
    st.session_state["status"] = "idle"
if "loaded_documents" not in st.session_state:
    st.session_state["loaded_documents"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

# Set up logging to avoid verbose output in Streamlit
logging.basicConfig(level=logging.WARNING)


# --- Custom theming for an Iconic UI (Light Theme) ---
st.markdown(
    """
    <style>
    /* Main Streamlit App and Background (Light Gray/White) */
    .stApp { background-color: #f8fafc; color: #1f2937; } 
    
    /* Title and Subtitle */
    .title-text { 
        font-size: 2.8rem; 
        font-weight: 900; 
        color: #3b3b3d;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.1);
    } 
    .sub-text { color: #64748b; margin-bottom: 2rem; font-size: 1.1rem; } 
    
    /* Streamlit chat message specific styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        background-color: #ffffff; /* Pure white chat background */
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Fix Code Block Readability */
    pre, code {
        background-color: #f1f5f9 !important; /* Light slate/blue for code blocks */
        color: #1f2937 !important; /* Dark text for code */
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        overflow-x: auto;
    }

    /* Answer Card - Prominent and professional */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    .answer-title {
        color: #3b82f6; /* Blue Accent */
        font-weight: 800;
        font-size: 1.3rem;
        margin-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .answer-body {
        color: #1f2937;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Source citation styling */
    .source-citation {
        background-color: #f1f5f9; /* Light gray background */
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0.75rem;
        border-left: 4px solid #f97316; /* Amber accent */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .chunk-block {
        background-color: #e2e8f0; /* Slightly darker light gray */
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.75rem;
        color: #1f2937; /* Ensure non-highlighted text in chunks is dark */
        font-size: 0.9rem;
    }
    .chunk-block p {
        margin-bottom: 0 !important;
    }
    .chunk-block mark {
        background-color: #fde047; /* Bright yellow highlight (good contrast on light) */
        color: #1f2937;
        padding: 0.1rem 0.15rem;
        border-radius: 4px;
        font-weight: 600;
    }

    /* General text and elements */
    .stChatMessage p, .stChatMessage li, .stChatMessage h2, .stChatMessage h3, .stChatMessage h4 { 
        color: #1f2937 !important; 
    }
    .stChatMessage a { color: #3b82f6 !important; }
    
    /* Input/Widget Styling */
    .stTextInput>div>div>input, .stFileUploader>div>div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1;
        color: #1f2937;
        border-radius: 8px;
        caret-color: #059669 !important; 
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
        font-weight: 600 !important;
    }
    
    /* Primary Button (Send/Index) */
    .stButton:has(button[kind="primary"]) button {
        background-color: #059669 !important; /* Deep Green */
        color: #ffffff !important;
        border: none;
    }
    
    /* Secondary Button (Clear Chat History) - Pink/Maroon for distinct action */
    .stButton:has(button[kind="secondary"]) button {
        background-color: #fbcfe8 !important; /* Light Pink */
        border-color: #f472b6 !important;
        color: #9d174d !important; /* Dark Maroon Text */
        box-shadow: none !important;
    }

    /* Sidebar Expander Styling */
    .streamlit-expanderHeader {
        background-color: #dbeafe; /* Light Blue */
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        color: #1e40af !important; /* Dark Blue Text */
        font-weight: 700;
    }
    
    /* Sidebar Input Labels */
    .st-emotion-cache-10ohe8c label { /* Target Streamlit label */
        color: #4b5563 !important;
    }

    /* FIX 1: Sidebar long text wrapping/breaking inside expander */
    .stSidebar .streamlit-expanderContent div[data-testid^="stMarkdownContainer"] p, 
    .stSidebar .streamlit-expanderContent div[data-testid^="stMarkdownContainer"] strong {
        word-wrap: break-word;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Core Logic Functions ---

def cleanup_resources():
    """Clears the RAG pipeline instance and removes temporary files."""
    with st.spinner("Clearing resources..."):
        try:
            # Remove pipeline state and temporary files
            if "pipeline" in st.session_state and st.session_state.pipeline:
                st.session_state.pipeline.vector_store = None
                st.session_state.pipeline.retriever = None
                del st.session_state.pipeline
            st.session_state.pipeline = None

            tmp = st.session_state.pop("tmp_dir", None)
            if tmp and os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            
            # Recreate tmp dir for next upload session
            st.session_state.tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")

            st.session_state["status"] = "idle"
            st.session_state["loaded_documents"] = []
            st.session_state["messages"] = [] # Also clear chat history on full cleanup
            st.success("All indexed documents and chat history cleared.")
            st.rerun()

        except Exception as e:
            logging.exception("Cleanup failed: %s", e)
            st.error(f"Cleanup failed: {e}")


def display_name_from_path(p: str) -> str:
    """Return a human-friendly name for a path or URL, removing a UUID prefix if present.

    If the basename is prefixed with a 32-character UUID hex and an underscore,
    the function strips that prefix and returns the original filename.
    """
    b = os.path.basename(p)
    parts = b.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 32:
        return parts[1]
    return b


def sidebar():
    st.sidebar.title("⚙️ RAG System Controls")

    # --- 1. Document Upload + Indexing ---
    st.sidebar.header("📚 Load Documents")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF/DOCX/TXT/MD documents",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True
    )

    url_input = st.sidebar.text_input("Or provide a URL (optional):", placeholder="https://example.com/report.pdf")

    if st.sidebar.button("🚀 Index Documents", type="primary", use_container_width=True):
        if not uploaded_files and not url_input.strip():
            st.sidebar.warning("Please upload at least one file or provide a URL.")
            return
        
        with st.spinner("Starting document indexing... this may take a moment."):
            
            # Ensure we have a pipeline instance
            if st.session_state.pipeline is None:
                st.session_state.pipeline = RAGPipeline(config_dir="configs")

            pipeline: RAGPipeline = st.session_state.pipeline

            try:
                # Save uploaded files to a temporary directory and build docs list
                file_paths: List[str] = []
                if uploaded_files:
                    for f in uploaded_files:
                        # Preserve original filename but make it unique by prefixing a UUID
                        basename = os.path.basename(f.name)
                        path = os.path.join(st.session_state.tmp_dir, f"{uuid.uuid4().hex}_{basename}")
                        with open(path, "wb") as out:
                            out.write(f.read())
                        file_paths.append(path)

                docs: List[dict] = []
                if file_paths:
                    # Use the original filename for display by removing the UUID prefix when present
                    docs.extend({"path": p, "enabled": True, "name": display_name_from_path(p)} for p in file_paths)
                if url_input.strip():
                    docs.append({"path": url_input.strip(), "enabled": True, "name": url_input.strip()})

                # Override pipeline documents config and run preparation synchronously
                pipeline.config["documents"] = docs
                st.session_state["status"] = "processing"
                pipeline.prepare_vector_store()

                st.sidebar.success("Documents indexed successfully!")
                st.session_state["status"] = "ready"
                st.session_state["loaded_documents"] = [
                    {"name": d["name"], "path": d["path"]} for d in docs
                ]

            except Exception as e:
                logging.exception("Indexing failed: %s", e)
                st.sidebar.error(f"Indexing failed: {e}")
                st.session_state["status"] = "error"
        st.rerun() # Rerun to update status display


    st.sidebar.markdown("---")
    
    # --- 2. Current Status ---
    st.sidebar.header("Current Context Status")
    
    status_msg = st.session_state.get("status", "idle")
    loaded_docs = st.session_state.get("loaded_documents", [])
    
    if status_msg == "ready":
        st.sidebar.success(f"Context Ready: {len(loaded_docs)} source(s) indexed.")
    elif status_msg == "processing":
        st.sidebar.info("Indexing in Progress...")
    elif status_msg == "error":
        st.sidebar.error("Error State. Check logs.")
    else:
        st.sidebar.warning("Idle. No documents loaded.")
        
    if loaded_docs:
        with st.sidebar.expander(f"Indexed Sources ({len(loaded_docs)})", expanded=True):
            for d in loaded_docs:
                doc_name = d.get("name", d.get("path", "Unknown Source"))
                st.caption(f"📃 {doc_name}")

    st.sidebar.markdown("---")

    # --- 3. Cleanup ---
    st.sidebar.header("Context Management")

    if st.sidebar.button("🔥 Clear ALL Indexed Documents", type="secondary", use_container_width=True):
        cleanup_resources()
    

def chat_area():
    st.markdown('<div class="title-text">🪐 Multi-Doc RAG Q&A Interface</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Ask questions grounded in your indexed documents.</div>', unsafe_allow_html=True)

    rag_ready = st.session_state["status"] == "ready"

    # Display Chat History using st.chat_message
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            response_content: Dict[str, Any] = message["content"]
            
            answer_text = response_content.get("answer", "No answer generated.")
            sources: List[Dict[str, str]] = response_content.get("sources", [])
            
            with st.chat_message("assistant", avatar="🤖"):
                # 1. Display the structured answer with enhanced styling
                st.markdown(
                    """<div class="answer-card"><div class="answer-title">🤖 AI Response</div><div class="answer-body">""", 
                    unsafe_allow_html=True
                )
                # st.markdown handles the Markdown structure (H2, H3, lists) in answer_text
                st.markdown(answer_text, unsafe_allow_html=False)
                st.markdown("""</div></div>""", unsafe_allow_html=True)
                
                # 2. Display Sources
                if sources:
                    with st.expander(f"📚 Sources Cited ({len(sources)} citation{'s' if len(sources) > 1 else ''})", expanded=False):
                        for i, source in enumerate(sources):
                            source_path = source.get('path', 'N/A')
                            page_number = source.get('page_info')
                            snippet = source.get('snippet', '') # This is the highlighted text from the original code
                            source_name = display_name_from_path(source_path)

                            # Format the source citation block
                            st.markdown(
                                f"""
                                <div class="source-citation">
                                    <p style="margin-bottom: 0.5rem; color: #f97316; font-weight: 700; font-size: 1.05rem;">
                                        {i+1}. Source: {source_name}
                                    </p>
                                    <p style="margin-bottom: 0.5rem; color: #94a3b8; font-size: 0.9rem;">
                                        📍 Location: <span style="color: #434547;">{page_number}</span>
                                    </p>
                                    <p style="margin-bottom: 0.25rem; color: #4b5563; font-size: 0.85rem; font-family: monospace;">
                                        📄 Path: <code>{source_name}</code>
                                    </p>
                                    {f'<div class="chunk-block"><p style="font-style: italic; color: #4b5563;">Snippet:</p><div class="chunk-body">{snippet}</div></div>' if snippet else ''}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(
                        '<p style="color: #64748b; font-size: 0.9rem; font-style: italic; margin-top: 1rem;">No specific sources were cited for this response.</p>',
                        unsafe_allow_html=True
                    )


    # Chat Input Area 
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about the indexed documents", 
            placeholder="e.g., Summarize the key findings from the uploaded report.", 
            disabled=not rag_ready,
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([1, 6])
        
        with col1:
            submitted = st.form_submit_button("Send", type="primary", disabled=not rag_ready, use_container_width=True)
        
        with col2:
            # Styled Secondary Button for Clearing History
            if st.form_submit_button("Clear Chat History", type="secondary"): 
                 st.session_state["messages"] = []
                 st.rerun() 
        
        # FIX 2: Store query and force rerun to display user message immediately
        if submitted and query.strip():
            st.session_state["messages"].append({"role": "user", "content": query.strip()})
            st.session_state["pending_query"] = query.strip()
            st.rerun()


    # FIX 2: Process the query in a separate block after the initial rerun
    if st.session_state["pending_query"]:
        query_to_process = st.session_state["pending_query"]
        
        # This spinner blocks the thread while the LLM processes, but the previous rerun
        # ensures the user's message is already visible in the chat history.
        with st.spinner(f"Retrieving information for: \"{query_to_process}\"…"):
            try:
                pipeline: RAGPipeline = st.session_state.get("pipeline")
                
                if not pipeline:
                    error_result = {"answer": "Error: Pipeline is not initialized. Please index documents first.", "sources": []}
                    st.session_state["messages"].append({"role": "assistant", "content": error_result})
                else:
                    # Run the RAG query
                    result = pipeline.answer_with_sources(query_to_process)
                    
                    # Store the structured result
                    st.session_state["messages"].append({"role": "assistant", "content": result})
                
            except MyException as me:
                error_result = {"answer": f"Failed to fetch answer (MyException): {me}", "sources": []}
                st.session_state["messages"].append({"role": "assistant", "content": error_result})
            except Exception as e:
                logging.exception("Query failed: %s", e)
                error_result = {"answer": f"Failed to fetch answer: {e}", "sources": []}
                st.session_state["messages"].append({"role": "assistant", "content": error_result})

        # Cleanup and final rerun to update the chat history with the response/error
        st.session_state["pending_query"] = None
        st.rerun()


def main():
    sidebar()
    chat_area()


if __name__ == "__main__":
    main()