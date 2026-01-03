# 📚 Multi-Document RAG Pipeline

This project implements a **production-oriented Retrieval-Augmented Generation (RAG) system** that can answer questions over **multiple document types** such as PDFs, DOCX, TXT, Markdown files, and web URLs.

The system is designed to be:

- **Accurate** (grounded answers only)
- **Scalable** (handles long documents efficiently)
- **Modular** (each pipeline stage is clearly separated)
- **Deployable** (works locally and on Streamlit Cloud)

---

## 🔍 What Problem Does This Solve?

Large Language Models (LLMs) do not inherently know your private or domain-specific documents.

This RAG pipeline:

1. Ingests your documents  
2. Converts them into searchable vectors  
3. Retrieves the most relevant content for a query  
4. Uses an LLM **only on retrieved context**  
5. Produces grounded, reliable answers with sources  

---

## 🧠 High-Level Architecture

The system consists of **four major layers**:

1. Data Ingestion & Preprocessing  
2. Vector Indexing & Retrieval  
3. Answer Generation (Stuff / Map-Reduce)  
4. User Interface (Streamlit with Embedded Backend)  

---

## 🔁 RAG Pipeline Flowchart

Below is the conceptual flow of the RAG pipeline used in this project.

<img width="1312" height="720" alt="RAG Pipeline Flowchart" src="https://github.com/user-attachments/assets/bf0a0054-bdce-400c-8029-c2f06d8141a1" />

---

## 🔄 Flow Description

1. User uploads documents or provides URLs  
2. Documents are cleaned, normalized, and chunked  
3. Chunks are embedded and stored in FAISS  
4. User asks a question  
5. Relevant chunks are retrieved, reranked, and diversified  
6. Answer is generated using Stuff or Map-Reduce  
7. Final answer and sources are returned  

---

## 🧩 Pipeline Stages Explained

### 1️⃣ Data Ingestion

Supported inputs:

- PDF, DOCX, TXT, Markdown  
- HTTP / HTTPS URLs  

Each document is extracted into:

- Clean text  
- Metadata (source, document type, page/section info)  

---

### 2️⃣ Structure-Aware Cleaning & Normalization

**Type-specific cleaning**

- Markdown → formatting removal  
- Web → HTML/script stripping  
- PDF/DOCX → layout artifact cleanup  

**Global normalization**

- Unicode normalization  
- Whitespace and newline cleanup  
- Consistent paragraph formatting  

---

### 3️⃣ Adaptive Chunking Strategy

Chunking is **document-length aware**, not file-type dependent:

- Small documents → larger chunks  
- Large documents → smaller overlapping chunks  

This balances:

- Retrieval accuracy  
- LLM context constraints  
- Processing latency  

---

### 4️⃣ Embeddings & Vector Store

- Embedding models: Ollama / Gemma-based embeddings  
- Vector store: FAISS (Approximate Nearest Neighbor)  

Each chunk is embedded and indexed for efficient similarity search.

---

### 5️⃣ Retrieval, Reranking & Diversification

Retrieval follows a **three-stage strategy**:

1. Vector similarity search  
2. Cross-encoder reranking  
3. MMR (Maximal Marginal Relevance)  

This ensures:

- High semantic relevance  
- Low redundancy  
- Better topic coverage  

---

## 🧠 Answer Generation Strategies

The system dynamically chooses between two strategies.

### 🟢 Stuff Strategy (Small Context)

Used when:

- Few chunks are retrieved  
- Total context is small  

Process:

- All chunks combined into one context  
- Single LLM call  
- Direct, concise answer  

Benefits:

- Fast  
- Simple  
- Low cost  

---

### 🔵 Map-Reduce Strategy (Large Context)

Used when:

- Many chunks are retrieved  
- Document is long or complex  

Map Phase:

- Each chunk processed independently  
- Relevant facts extracted  
- Partial relevance preserved  

Reduce Phase:

- Map outputs combined  
- Duplicates removed  
- Final grounded answer synthesized  

Benefits:

- Scales to large documents  
- Better factual accuracy  
- Reduced hallucination  

---

## 🧪 Evaluation Philosophy

Evaluation focuses on:

- Retrieval Recall@K  
- Answer Faithfulness  
- Semantic Relevance  
- End-to-End Latency  

Future extensions include:

- Automated RAG benchmarks  
- Regression testing  
- Confidence scoring  

---

## 🖥️ User Interface

- Built using **Streamlit**  
- Backend logic embedded for simplicity  
- Supports:
  - Local execution  
  - Cloud deployment  

---

## 📂 Project Structure

    DocWeave/
    ├── src/
    │   ├── ingestion/
    │   ├── preprocessing/
    │   ├── embedding/
    │   ├── vectorstore/
    │   ├── retrieval/
    │   ├── rag/
    │   ├── api/
    │   └── ui/
    ├── configs/
    ├── static/
    │   └── images/
    ├── architecture.md
    ├── evaluation.md
    ├── demo.md
    └── README.md

This structure ensures:

- High relevance  
- Low redundancy  
- Good topic coverage  

---

## 🧠 Prompting Design

The system uses **four carefully designed prompts**:

- System Prompt – enforces grounding and behavior  
- Map Prompt – extracts relevant facts per chunk  
- Reduce Prompt – synthesizes final answer  
- Stuff Prompt – direct QA for small contexts  

All prompts are:

- Strict about context usage  
- Anti-hallucination  
- Concise and deterministic  

---

## 🤖 LLM & Model Choices

### LLM (Answer Generation)

- Ollama Models  
  - LLaMA 3.2  
  - embedding-gemma  

Chosen for:

- High speed  
- Free local inference  

### Embeddings

- Ollama embeddings  
- No external API cost  

### Reranking

- Cross-Encoder (ms-marco-MiniLM-L-6-v2)  
- Improves semantic precision  

---

## 🚀 Deployment Notes

- Ollama used **locally only**  
- Cloud deployment uses:
  - Groq for LLM  
  - HuggingFace embeddings  
- Secrets managed via Streamlit Cloud settings  

---

## ✅ Key Design Decisions

- No hallucinated answers  
- Chunk-aware retrieval  
- Balanced speed vs accuracy  
- Simple deployment  
- Modular, extensible codebase  

---

## 📌 Future Improvements

- Query-aware retrieval tuning  
- Answer confidence scoring  
- Multi-turn conversational memory  
- Citation scoring and highlighting  
- Automated evaluation metrics  

---

## 🏁 Summary

This project demonstrates a **real-world, production-ready RAG pipeline** that balances:

- Correctness  
- Performance  
- Simplicity  
- Scalability  

Suitable for:

- Document Q&A systems  
- Internal knowledge bases  
- Research assistants  
- Enterprise document search

