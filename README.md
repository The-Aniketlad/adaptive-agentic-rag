# 🚀 Agentic RAG vs. Naive RAG — From Scratch to Production

A comprehensive, production-grade implementation comparing **Naive RAG** vs. **Agentic RAG (LangGraph)** on multi-hop question answering ([HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)) using 100% free-tier local and cloud resources.

---

## 📌 Project Overview
Retrieval-Augmented Generation (RAG) is powerful, but standard (Naive) RAG often struggles with complex multi-hop queries, ambiguous prompts, and hallucinations. 

This project explores and benchmarks:
1. **Naive RAG Baseline**: Query Vectorization $\rightarrow$ Vector DB Retrieval $\rightarrow$ Prompt Augmentation $\rightarrow$ LLM Generation.
2. **Agentic RAG (LangGraph)**: Adaptive Query Routing, Multi-hop Query Decomposition, Retrieval Grading & Self-Correction, Hallucination Verification, and Fallback Web Search.

---

## 🛠️ Tech Stack & Dependencies

| Category | Tools / Libraries | Purpose |
|---|---|---|
| **Agent Orchestration** | `langgraph`, `langchain`, `langchain-community` | Graph-based state machine for adaptive agent workflows |
| **LLMs (Free Tier)** | `langchain-groq` (Llama-3), `langchain-google-genai` (Gemini) | Blazing fast inference for reasoning, routing & grading |
| **Embeddings & Vector Store** | `sentence-transformers` (`BAAI/bge-small-en-v1.5`), `chromadb` | Local embeddings & persistent vector storage (lightweight on RAM/GPU) |
| **Evaluation & Dataset** | `datasets` (HotpotQA), `rank_bm25`, `pandas`, `tqdm` | Benchmark corpus, exact match (EM), F1 score, and LLM-as-a-judge |
| **Observability & Search** | `arize-phoenix`, `tavily-python` | Trace agent decision loops & fallback live web search |
| **UI & App** | `streamlit` | Interactive interactive demo for agent visual traces |

---

## 📂 Project Structure (Planned)
```
agentic-rag/
├── data/                  # HotpotQA corpus & evaluation splits
├── src/
│   ├── ingest.py          # Document chunking & ChromaDB vector ingestion
│   ├── naive_rag.py       # Naive RAG baseline pipeline
│   ├── agent.py           # LangGraph Agentic RAG workflow & state machine
│   ├── eval.py            # Evaluation suite (EM, F1, Judge metrics)
│   └── llm.py             # LLM configurations, fallbacks & caching
├── app.py                 # Streamlit UI demo
├── .env.example           # Template for environment variables
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation & benchmark results
```

---

## 📝 Build Progress & Changelog

- [x] **Project Initialization**: Architecture roadmap finalized and GitHub documentation initialized.
- [ ] **Phase 0 - Setup**: Python virtual environment created, dependencies installed, API keys configured.
- [ ] **Phase 1 - Data & Baseline**: HotpotQA corpus ingestion with ChromaDB & Naive RAG baseline eval.
- [ ] **Phase 2 - Agentic RAG**: LangGraph graph implementation (Router, Grader, Rewriter, Decomposer, Web Fallback).
- [ ] **Phase 3 - Evaluation & Benchmarking**: Comparative analysis, ablations, and Phoenix tracing.
- [ ] **Phase 4 - UI & Deployment**: Streamlit web application & final showcase.

---
*Built from scratch with ❤️ by [Aniket Lad](https://github.com/) & Lucy.*
