import os
import sys
import json
import time
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.naive_rag import NaiveRAG
from src.agent import AgenticRAG

# Page Configuration
st.set_page_config(
    page_title="Agentic RAG vs Naive RAG Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .sub-q-tag {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
EVAL_SET_PATH = DATA_DIR / "hotpotqa_eval_500.json"

@st.cache_resource
def load_pipelines():
    naive = NaiveRAG(top_k=5)
    agent = AgenticRAG(top_k=5)
    return naive, agent

naive_rag, agentic_rag = load_pipelines()

# Sidebar
st.sidebar.title("⚙️ RAG Settings")
pipeline_choice = st.sidebar.radio(
    "Choose System Pipeline:",
    ["LangGraph Agentic RAG", "Naive RAG Baseline", "Side-by-Side Comparison ⚔️"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Benchmark Summary")
st.sidebar.markdown("""
- **HotpotQA Corpus:** 4,937 Passages
- **Multi-Hop Gain:** **+9.72% F1** (58.4% ➔ 68.2%)
- **Embedder:** `BAAI/bge-small-en-v1.5`
- **LLMs:** Gemini 3.8 Flash & Groq Qwen
""")

# Main Header
st.markdown('<div class="main-title">🚀 Adaptive Agentic RAG vs. Naive RAG</div>', unsafe_allow_html=True)
st.caption("Benchmark & live trace visualization on multi-hop question answering (HotpotQA).")

# Sample questions selector
sample_questions = [
    "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
    "Custom Question"
]

selected_sample = st.selectbox("💡 Choose a sample question or type your own:", sample_questions)

if selected_sample == "Custom Question":
    user_query = st.text_input("Enter your question:", placeholder="Ask anything about the indexed corpus...")
else:
    user_query = st.text_input("Enter your question:", value=selected_sample)

run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

if run_button and user_query:
    st.markdown("---")
    
    if pipeline_choice == "Side-by-Side Comparison ⚔️":
        col1, col2 = st.columns(2)
        
        # 1. Naive RAG Run
        with col1:
            st.subheader("🥉 Naive RAG Baseline")
            t0 = time.time()
            with st.spinner("Running Naive Retrieval & Generation..."):
                naive_res = naive_rag.generate_answer(user_query)
            t_naive = time.time() - t0
            
            st.success(f"**Answer:** {naive_res['answer']}")
            st.caption(f"⏱️ Latency: {t_naive:.2f}s | 📚 Retrieved: {len(naive_res['retrieved_docs'])} docs")
            
            with st.expander("📄 View Retrieved Passages (Top-5)"):
                for i, doc in enumerate(naive_res['retrieved_docs'], 1):
                    st.markdown(f"**Passage {i}:**")
                    st.text(doc)
                    
        # 2. Agentic RAG Run
        with col2:
            st.subheader("🥇 LangGraph Agentic RAG")
            t0 = time.time()
            with st.spinner("Running Agentic State Machine (Routing ➔ Decomposing ➔ Grading)..."):
                agent_res = agentic_rag.generate_answer(user_query)
            t_agent = time.time() - t0
            
            st.success(f"**Answer:** {agent_res['answer']}")
            st.caption(f"⏱️ Latency: {t_agent:.2f}s | 🔄 Loops: {agent_res['loop_step']}")
            
            if agent_res.get('sub_questions'):
                st.markdown("**🔀 Decomposed Sub-Questions:**")
                for sq in agent_res['sub_questions']:
                    st.markdown(f"<span class='sub-q-tag'>🔍 {sq}</span>", unsafe_allow_html=True)
                    
            with st.expander("📄 View Agent Context Documents"):
                for i, doc in enumerate(agent_res['retrieved_docs'], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc)

    elif pipeline_choice == "LangGraph Agentic RAG":
        st.subheader("🥇 LangGraph Agentic RAG Workflow")
        t0 = time.time()
        with st.spinner("Executing Graph Nodes..."):
            res = agentic_rag.generate_answer(user_query)
        latency = time.time() - t0
        
        st.success(f"### **Answer:** {res['answer']}")
        
        # Agent execution breakdown
        st.markdown("#### 🧠 Agent Internal State & Steps")
        c1, c2, c3 = st.columns(3)
        c1.metric("Execution Latency", f"{latency:.2f}s")
        c2.metric("Total Context Docs", len(res['retrieved_docs']))
        c3.metric("Rewriting Loops", res['loop_step'])
        
        if res.get('sub_questions'):
            st.markdown("##### 🔀 Query Decomposition")
            for i, sq in enumerate(res['sub_questions'], 1):
                st.info(f"**Sub-Query {i}:** {sq}")
                
        with st.expander("📚 View Retrieved & Graded Context Passages"):
            for i, doc in enumerate(res['retrieved_docs'], 1):
                st.markdown(f"**Passage {i}:**")
                st.text(doc)

    else:
        st.subheader("🥉 Naive RAG Baseline")
        t0 = time.time()
        with st.spinner("Retrieving & Generating..."):
            res = naive_rag.generate_answer(user_query)
        latency = time.time() - t0
        
        st.success(f"### **Answer:** {res['answer']}")
        st.metric("Latency", f"{latency:.2f}s")
        
        with st.expander("📚 View Top-5 Retrieved Passages"):
            for i, doc in enumerate(res['retrieved_docs'], 1):
                st.markdown(f"**Passage {i}:**")
                st.text(doc)
