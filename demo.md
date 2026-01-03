\# DocWeave – Demo \& Usage Guide



This document demonstrates how to run DocWeave locally and query

multiple documents using the RAG pipeline.



---



\## 1. Setup Instructions



\### Clone the Repository

```bash

git clone https://github.com/rajveer100704/DocWeave.git

cd DocWeave



## 2. Create Virtual Environment



python -m venv .venv

source .venv/bin/activate   # Linux/Mac

.venv\\Scripts\\activate      # Windows


## 3. Install Dependencies

 pip install -r requirements.txt



## 4. Environment Configuration



&nbsp;Create a .env file using the template:



&nbsp; cp .env.example .env



&nbsp;Add your API key and model configuration.

## 5. Ingest Documents



&nbsp;Place documents inside a folder (PDF, TXT, MD supported).



&nbsp; python -m src.ingestion.extractor --input ./documents





&nbsp;This will:



&nbsp; Load documents

&nbsp; Chunk text

&nbsp; Generate embeddings

&nbsp; Store vectors in FAISS

## 6. Run the Application

&nbsp; Option A: Streamlit UI

&nbsp;  streamlit run src/ui/streamlit\_app.py



&nbsp; Option B: API Mode

&nbsp;  python -m src.api.app

## 7. Sample Queries



&nbsp; Query 1 : What are the key responsibilities of an AI Engineer Intern?

&nbsp; Query 2 : Summarize the architecture of the RAG system.

&nbsp; Query 3 : Which document discusses evaluation metrics?

## 8. Expected Behavior



&nbsp; Retrieved chunks are grounded in source documents

&nbsp; Responses include relevant contextual information

&nbsp; Hallucinations are reduced through metadata-aware retrieval


## 9. Notes

  Chunk size and top-k retrieval can be tuned via config files

&nbsp; The system is modular and supports backend swapping 

