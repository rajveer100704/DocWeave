# 📚 Multi-Document RAG Pipeline

This project implements a **production-oriented Retrieval-Augmented Generation (RAG) system** that can answer questions over **multiple document types** such as PDFs, DOCX, TXT, Markdown files, and web URLs.

The system is designed to be:

* **Accurate** (grounded answers only)
* **Scalable** (handles long documents efficiently)
* **Modular** (each pipeline stage is clearly separated)
* **Deployable** (works locally and on Streamlit Cloud)

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

The system consists of  **four major layers** :

1. **Data Ingestion & Preprocessing**
2. **Vector Indexing & Retrieval**
3. **Answer Generation (Stuff / Map-Reduce)**
4. **User Interface (Streamlit with Embedded Backend)**

---

## 🔁 RAG Pipeline Flowchart

Below is the conceptual flow of the RAG pipeline used in this project.

<img width="1312" height="720" alt="Screenshot 2026-01-03 132341" src="https://github.com/user-attachments/assets/bf0a0054-bdce-400c-8029-c2f06d8141a1" />


### Flow Description

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

- **Embedding models:** Ollama / Gemma-based embeddings
- **Vector store:** FAISS (Approximate Nearest Neighbor)

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

DocWeave dynamically selects the generation strategy:

### 🟢 Stuff Strategy (Small Context)
- Single LLM call
- Fast and cost-efficient
- Used when retrieved context fits comfortably

### 🔵 Map-Reduce Strategy (Large Context)
- Chunk-level fact extraction (Map)
- Deduplicated synthesis (Reduce)
- Scales well for long documents

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

```text
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

This ensures:

* high relevance
* low redundancy
* good coverage of the topic

---

## 🧠 Answer Generation Strategies

The system dynamically chooses between two strategies.

---

### 🟢 Stuff Strategy (Small Context)

Used when:

* few chunks are retrieved
* total context is small

**Process:**

* All chunks are combined into one context
* Single LLM call
* Direct, concise answer

**Benefits:**

* Fast
* Simple
* Low cost

---

### 🔵 Map-Reduce Strategy (Large Context)

Used when:

* many chunks are retrieved
* document is long or complex

#### Map Phase

* Each chunk is processed **independently**
* Relevant facts are extracted
* Partial relevance is preserved

#### Reduce Phase

* All map outputs are combined
* Duplicates are removed
* Final grounded answer is synthesized

**Benefits:**

* Scales to large documents
* Better factual accuracy
* Reduced hallucination

---

## 🧠 Prompting Design

The system uses  **three carefully designed prompts** :

* **System Prompt:** Enforces grounding and behavior
* **Map Prompt:** Extracts relevant facts per chunk
* **Reduce Prompt:** Synthesizes a final answer
* **Stuff Prompt:** Direct QA for small contexts

All prompts are:

* strict about context usage
* anti-hallucination
* concise and deterministic

---

## 🤖 LLM & Model Choices

### LLM (Answer Generation)

* **OlIama Models**
  * LLama 3.2
  * embeddinggemma
* Chosen for:
  * high speed
  * free access

### Embeddings

* Ollama Embeddings
* Local inference (no API cost)

### Reranking

* Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* Improves semantic precision

---

## 🖥️ User Interface

* Built using **Streamlit**
* Backend logic is **embedded**
* Works on:
  * local machine
  * Streamlit Cloud

**User flow:**

1. Upload documents
2. Index documents
3. Ask a question
4. View answer + sources
5. Clear context if needed

---

## 🚀 Deployment Notes

* Ollama is used **only locally**
* Cloud deployment uses:
  * Groq for LLM
  * HuggingFace embeddings
* Secrets managed via Streamlit Cloud settings

---

## ✅ Key Design Decisions

* No hallucinated answers
* Chunk-aware retrieval
* Balanced speed vs accuracy
* Simple deployment
* Modular and extensible codebase

---

## 📌 Future Improvements

* Query-aware retrieval tuning
* Answer confidence scoring
* Multi-turn conversational memory
* Citation scoring and highlighting
* Evaluation metrics (faithfulness, recall)

---

## 🏁 Summary

This project demonstrates a **real-world, production-ready RAG pipeline** that balances:

* correctness
* performance
* simplicity
* scalability

It is suitable for:

* document Q&A systems
* internal knowledge bases
* research assistants
* enterprise document search
