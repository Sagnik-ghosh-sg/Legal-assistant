# Indian Legal AI Assistant

A local, privacy-preserving AI assistant for Indian Law, built using quantized GGUF large language models (LLMs) (such as Gemma 2B) via `llama-cpp-python`, LangChain, ChromaDB, and HuggingFace sentence-transformer embeddings.

It answers legal questions with structured reasoning and citation, simulates courtroom strategies, and suggests required legal documents for case preparation. All data and computation run locally—no internet required.

---

## Features

- Legal Q&A with structured section-wise responses
- Courtroom Strategy Mode – Simulates arguments, counters, and legal tactics
- Document Planning Mode – Recommends legal documents and procedures
- Retrieval-Augmented Generation (RAG) using actual Indian legal texts
- Structured responses including:
  - Applicable Law or Section
  - Legal Interpretation
  - Reasoned Conclusion
  - Citation of Source
  - Simplified Explanation
- Fully offline and private (local LLM and vector search)
- Can be extended to multilingual use (Indian regional languages)

---

## Tech Stack

| Layer               | Tool                                |
|---------------------|-------------------------------------|
| LLM Inference       | `llama-cpp-python` (GGUF models)    |
| Framework           | LangChain                           |
| Embeddings          | HuggingFace Sentence Transformers   |
| Vector Database     | ChromaDB                            |
| Document Loaders    | PyPDFLoader, TextLoader, BeautifulSoup |
| Optional Interface  | Streamlit                           |

---

## Directory Structure

```bash
legal-ai-assistant/
│
├── data/                   # Legal text sources (IPC, CrPC, etc.)
├── embeddings/             # Chroma vector store files
├── models/                 # GGUF LLM model files (e.g., gemma-2b.Q4_K_M.gguf)
├── main.py                 # Main assistant logic
├── requirements.txt
├── LICENSE
└── README.md
## CONTACT 
NAME: SAGNIK GHOSH
EMAIL:sagnikghosh2112003@gmail.com