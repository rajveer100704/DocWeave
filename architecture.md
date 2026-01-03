---



\## Architecture Diagram



User Query

│

▼

┌───────────────┐

│ Query Encoder │

└───────┬───────┘

│

▼

┌─────────────────────┐

│ Vector Store (ANN) │

│ • similarity search│

└───────┬─────────────┘

│ Top-K chunks

▼

┌─────────────────────┐

│ Context Assembler │

│ • metadata filter │

│ • prompt builder │

└───────┬─────────────┘

│

▼

┌─────────────────────┐

│ LLM Generator │

│ • grounded output │

└───────┬─────────────┘

│

▼
Final Answer



\*\*Design Notes\*\*

\- Retrieval is isolated from generation for easy experimentation

\- Metadata-aware filtering reduces hallucinations

\- Embedding + LLM layers are fully swappable



