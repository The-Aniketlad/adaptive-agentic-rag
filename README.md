# 🚀 Adaptive Agentic RAG vs. Naive RAG — From Scratch to Production

A comprehensive, production-grade benchmarking of **Naive RAG** vs. **Agentic RAG (LangGraph)** on multi-hop question answering ([HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)) built 100% with local vector storage and free-tier LLMs.

---

## 📊 Benchmark Results (Head-to-Head on HotpotQA)

| Metric / Question Type | 🥉 Naive RAG Baseline | 🥇 LangGraph Agentic RAG | Impact / Gain |
|---|:---:|:---:|:---:|
| **Bridge (Multi-Hop) F1 Score** | **58.47%** | **68.19%** | **🚀 +9.72% Boost** |
| **Overall F1 Score** | **65.52%** | **68.52%** | **📈 +3.00% Overall Gain** |
| **Exact Match (EM %)** | 56.00% | 52.00% | *More descriptive answers* |
| **Average Latency** | 3.85s / query | 25.76s / query | *Trade-off: Iterative self-correction* |

> **Key Finding:** Standard Naive RAG fails on multi-hop bridge questions because dense vector search misses intermediate entities. LangGraph Agentic RAG uses query decomposition and retrieval grading to bridge the reasoning gap, yielding a **+9.72% F1 score improvement**.

---

## 🧠 System Architecture (LangGraph State Machine)

```
                       [User Question]
                              │
                              ▼
                     [1. Query Router]
                    /         │       \
                   /          │        \
         (Chit-Chat)    (Multi-Hop)    (Standard)
             │                │             │
             ▼                ▼             ▼
       [Direct Gen]    [2. Decomposer]   [3. ChromaDB Retriever]
             │                │             │
             │                └─────────────┘
             │                       │
             │                       ▼
             │               [4. Retrieval Grader]
             │              /                     \
             │        (Relevant)             (Irrelevant)
             │            │                       │
             │            ▼                       ▼
             │     [5. Generator]       [6. Query Rewriter]
             │            │                       │
             │            ▼             (Loop <= 2) ──► Back to [3. Retrieve]
             │   [7. Grounding Check]   (Loop > 2)  ──► [8. Tavily Web Fallback]
             │            │
             └────────────┴─────────────► [Final Output]
```

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration & State Machine:** `langgraph` (v0.2+), `langchain`
- **LLM Inference:** `langchain-groq` (Qwen-3.8 / Llama-3), `langchain-google-genai` (Gemini 3.8 Flash)
- **Local Embeddings:** `sentence-transformers` (`BAAI/bge-small-en-v1.5`, 384 dimensions)
- **Vector Storage:** `chromadb` (Persistent, Cosine similarity metric)
- **Dataset & Benchmark:** `datasets` (HotpotQA distractor split, 4,937 Wikipedia context passages)
- **Web Fallback:** `tavily-python` (live search fallback)
- **Interactive UI:** `streamlit`

---

## 📂 Repository Structure

```
├── data/
│   ├── chroma_db/            # Local persistent ChromaDB vector store (4,937 docs)
│   ├── hotpotqa_eval_500.json# 500 evaluation questions split
│   └── results/              # Detailed CSV benchmark results per run
├── src/
│   ├── llm.py                # Multi-provider LLM factory (Groq & Gemini)
│   ├── ingest.py             # HotpotQA loader & ChromaDB indexing pipeline
│   ├── naive_rag.py          # Naive RAG baseline implementation
│   ├── agent_state.py        # LangGraph AgentState TypedDict & Pydantic schemas
│   ├── agent.py              # LangGraph Agentic RAG state machine
│   └── eval.py               # Official HotpotQA Exact Match & F1 benchmark runner
├── app.py                    # Streamlit interactive UI demo
├── requirements.txt          # Production dependencies
├── .env.example              # Environment variables template
└── README.md                 # Project documentation & benchmark analysis
```

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/The-Aniketlad/adaptive-agentic-rag.git
cd adaptive-agentic-rag

python -m venv .venv
# On Windows Git Bash:
source .venv/Scripts/activate

pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Build Local Vector Database
```bash
python src/ingest.py
```

### 4. Run Benchmarks
```bash
# Run Naive RAG vs Agentic RAG benchmark
python src/eval.py --num_samples 25 --pipeline all
```

### 5. Launch Interactive Web Demo
```bash
streamlit run app.py
```

---
