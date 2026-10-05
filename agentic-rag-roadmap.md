# Roadmap: Agentic RAG vs. Naive RAG (4 Weeks)

**Goal:** Build a LangGraph agentic RAG system (query routing, decomposition, retrieval grading, self-correction) and benchmark it against a naive RAG baseline on HotpotQA, using only free tools.

**Hardware notes (8 GB RAM, GTX 1050 3 GB):** run embeddings and the vector store locally, use free LLM APIs (Groq / Gemini) for the agent, and avoid Docker.

---

## Phase 0: Setup (Day 1)

### Install on Windows
1. **Python 3.11** from python.org (tick "Add to PATH"). LangGraph and the ML libraries are most stable on 3.10-3.12.
2. **VS Code** plus the Python extension.
3. **Git** from git-scm.com, and a free GitHub account.

### Create the project
Run in a terminal inside VS Code:

```bash
mkdir agentic-rag && cd agentic-rag
python -m venv .venv
.venv\Scripts\activate
pip install langgraph langchain langchain-community langchain-groq langchain-google-genai
pip install chromadb sentence-transformers rank_bm25 datasets
pip install arize-phoenix openinference-instrumentation-langchain
pip install tavily-python python-dotenv streamlit pandas tqdm
```

### Get free API keys
Put them in a `.env` file and add `.env` to `.gitignore`.

| Service | URL | Use |
|---|---|---|
| Groq | console.groq.com | Fast Llama models. Small model for grading/routing, larger one for final answers |
| Google AI Studio | aistudio.google.com | Gemini free tier, backup when Groq rate-limits |
| Tavily | tavily.com | Free web search tool for the fallback step |

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

### Suggested structure
```
agentic-rag/
  data/            # downloaded corpus + eval set
  src/
    ingest.py      # build the vector index
    naive_rag.py   # baseline
    agent.py       # LangGraph agentic version
    eval.py        # benchmark runner
    llm.py         # model setup + caching
  app.py           # Streamlit demo
  README.md
  .env
```

---

## Week 1: Data + Naive Baseline

- [ ] **Dataset:** load HotpotQA (distractor setting) from Hugging Face with `datasets`. Take ~500 questions for eval. Pool their context paragraphs into one corpus (~5-10k passages) to stay light on RAM.
- [ ] **Ingest:** embed with `BAAI/bge-small-en-v1.5` (via `sentence-transformers`) and store in a local Chroma collection. Persist to disk so you only build it once.
- [ ] **Naive RAG:** retrieve top-5, stuff into a prompt, generate. No tricks. This is the baseline.
- [ ] **Eval script:** compute Exact Match and F1 against gold answers (HotpotQA's standard metrics), plus an LLM-as-judge correctness score. Save results to a CSV per run.
- [ ] **Cache LLM calls** (SQLite cache via LangChain) so reruns don't burn free quota.

**Deliverable:** baseline numbers in a table.

---

## Week 2: Build the Agent (LangGraph)

### Graph design
```
question -> router -> [direct answer | retrieve]
retrieve -> grader -> (relevant?) -> generate -> hallucination check -> answer
                   \-> (not relevant) -> rewrite query -> retrieve (max 2-3 loops)
                   \-> (still failing) -> web search (Tavily) -> generate
multi-hop questions -> decomposer -> sub-questions -> retrieve each -> synthesize
```

### Build order (test each on ~20 questions before adding the next)
- [ ] State definition (`TypedDict`) and the retrieve / generate nodes
- [ ] Retrieval grader (LLM returns relevant yes/no per chunk)
- [ ] Query rewriter and the conditional loop with a hard iteration cap
- [ ] Decomposer for multi-hop questions
- [ ] Web search fallback
- [ ] Answer-grounding check

**Tip:** use structured output (Pydantic schemas) for the router and graders so parsing never breaks.

---

## Week 3: Evaluation and Analysis

- [ ] Run both systems on the same 500 questions; compare EM, F1, and judge score.
- [ ] **Ablations:** add one component at a time (grader only, + rewriter, + decomposer, + web fallback) and tabulate the gain from each.
- [ ] **Efficiency metrics:** average LLM calls per question, latency, tokens. Report the cost/accuracy tradeoff honestly.
- [ ] **Tracing:** run Phoenix locally (`python -m phoenix.server.main serve`) to inspect agent runs and capture screenshots for the README.
- [ ] **Failure analysis:** categorize 15-20 failures (retrieval miss, bad decomposition, over-looping, etc.).

**Deliverable:** results table, ablation table, failure analysis.

---

## Week 4: Polish and Ship

- [ ] **Streamlit demo** (`app.py`) showing the answer, the agent's steps, and retrieved sources.
- [ ] **Deploy** free on Streamlit Community Cloud or Hugging Face Spaces. Use a small prebuilt index; keep API keys in the platform's secrets.
- [ ] **README:** architecture diagram, results table, ablation table, failure analysis, setup instructions, short demo GIF.
- [ ] **Resume bullet** with real numbers.

Example:

> Built a LangGraph agentic RAG system (query routing, decomposition, retrieval grading, self-correction) that improved multi-hop QA F1 from X to Y over a naive baseline on HotpotQA; ablated each component and traced runs with Phoenix.

---

## Watch-outs

- **Rate limits:** do dev runs on 20-50 questions; run the full 500 only at the end. Use caching and exponential backoff.
- **RAM:** keep embedding batch sizes small, and don't run Phoenix, Streamlit, and a big eval simultaneously.
- **Scope:** finish the full pipeline end to end before polishing any single node.

---

## Stretch Goals (after the core is done)

- Add tools beyond retrieval (SQL tool, Python sandbox) for a more "agentic" system.
- Build trajectory-level evals (tool-call correctness, step efficiency, loop detection) and run them in GitHub Actions.
- Swap in a fine-tuned embedding model and report before/after retrieval metrics.
