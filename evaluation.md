\# Evaluation Metrics – RAG System



This document defines how the quality and reliability of the

Retrieval-Augmented Generation (RAG) system are evaluated.



---



\## 1. Retrieval Metrics



\### Recall@K

Measures whether the correct document chunk appears in the top-K retrieved results.



\- High Recall@K ⇒ good document coverage

\- Evaluated using labeled QA pairs



---



\### Precision@K

Measures how many retrieved chunks are actually relevant.



Helps identify noisy retrieval.



---



\## 2. Generation Metrics



\### Faithfulness

Checks whether generated answers are grounded in retrieved context.



\- Penalizes hallucinations

\- Can be evaluated via:

&nbsp; - Human review

&nbsp; - LLM-as-a-judge scoring



---



\### Answer Relevance

Measures alignment between user query and generated response.



---



\## 3. End-to-End Metrics



\### Exact Match (EM)

Used when answers are deterministic.



---



\### Semantic Similarity

Uses embedding similarity between generated and reference answers.



---



\## 4. Latency Metrics



\- Retrieval latency

\- Generation latency

\- End-to-end response time



Important for production readiness.



---



\## 5. Future Evaluation Enhancements



\- RAGAS-based evaluation

\- Automated regression tests

\- Benchmark datasets (domain-specific)



