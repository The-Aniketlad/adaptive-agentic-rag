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

# Page Configuration - Clean & Professional
st.set_page_config(
    page_title="Agentic RAG Benchmark Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean, Modern Developer & Research Dashboard
st.markdown("""
<style>
    /* Global Font & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    
    .result-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.5rem;
    }
    
    .answer-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        font-size: 1rem;
        color: #F8FAFC;
        margin-bottom: 0.75rem;
    }
    
    .tag {
        display: inline-block;
        background-color: #1E293B;
        color: #93C5FD;
        border: 1px solid #3B82F6;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    
    .metric-item {
        background-color: #0B1120;
        border: 1px solid #1E293B;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        flex: 1;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 0.2rem;
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

# Sidebar - Settings & Configuration
st.sidebar.markdown("### Configuration")
pipeline_choice = st.sidebar.radio(
    "Evaluation Mode",
    ["Side-by-Side Comparison", "LangGraph Agentic RAG", "Naive RAG Baseline"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Benchmark Specifications")
st.sidebar.markdown("""
- **Dataset:** HotpotQA (Distractor Setting)
- **Indexed Passages:** 4,937 Wikipedia Passages
- **Embedding Model:** BAAI/bge-small-en-v1.5 (384-dim)
- **Primary LLM:** Gemini 3.8 Flash / Groq Qwen-3.8
- **Evaluation Metrics:** Exact Match (EM) & Token F1
""")

# Main Content Area
st.markdown('<div class="app-header">Adaptive Agentic RAG vs. Naive RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Benchmarking multi-hop retrieval, self-correction, and query decomposition.</div>', unsafe_allow_html=True)

sample_questions = [
    "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
    "Custom Question"
]

selected_sample = st.selectbox("Select Test Query", sample_questions)

if selected_sample == "Custom Question":
    user_query = st.text_input("Enter Query", placeholder="Enter your factual question...")
else:
    user_query = st.text_input("Enter Query", value=selected_sample)

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_button = st.button("Run Evaluation", type="primary", use_container_width=True)

if run_button and user_query:
    st.markdown("---")
    
    if pipeline_choice == "Side-by-Side Comparison":
        col1, col2 = st.columns(2)
        
        # 1. Naive RAG Execution
        with col1:
            st.markdown('<div class="section-title">Naive RAG Baseline</div>', unsafe_allow_html=True)
            t0 = time.time()
            with st.spinner("Processing Naive Pipeline..."):
                naive_res = naive_rag.generate_answer(user_query)
            t_naive = time.time() - t0
            
            st.markdown(f'<div class="answer-box"><strong>Answer:</strong> {naive_res["answer"]}</div>', unsafe_allow_html=True)
            
            # Metrics
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{t_naive:.2f}s</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Retrieved Passages</div>
                    <div class="metric-value">{len(naive_res['retrieved_docs'])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(naive_res['retrieved_docs'], 1):
                    st.markdown(f"**Passage {i}**")
                    st.text(doc)
                    
        # 2. Agentic RAG Execution
        with col2:
            st.markdown('<div class="section-title">LangGraph Agentic RAG</div>', unsafe_allow_html=True)
            t0 = time.time()
            with st.spinner("Executing State Machine (Route ➔ Decompose ➔ Grade ➔ Synthesize)..."):
                agent_res = agentic_rag.generate_answer(user_query)
            t_agent = time.time() - t0
            
            st.markdown(f'<div class="answer-box"><strong>Answer:</strong> {agent_res["answer"]}</div>', unsafe_allow_html=True)
            
            # Metrics
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{t_agent:.2f}s</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Documents</div>
                    <div class="metric-value">{len(agent_res['retrieved_docs'])}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Correction Loops</div>
                    <div class="metric-value">{agent_res['loop_step']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if agent_res.get('sub_questions'):
                st.markdown("<br><strong>Decomposed Sub-Queries:</strong>", unsafe_allow_html=True)
                for sq in agent_res['sub_questions']:
                    st.markdown(f'<span class="tag">{sq}</span>', unsafe_allow_html=True)
                    
            with st.expander("Agent Context & Graded Documents"):
                for i, doc in enumerate(agent_res['retrieved_docs'], 1):
                    st.markdown(f"**Document {i}**")
                    st.text(doc)

    elif pipeline_choice == "LangGraph Agentic RAG":
        st.markdown('<div class="section-title">LangGraph Agentic Execution Trace</div>', unsafe_allow_html=True)
        t0 = time.time()
        with st.spinner("Processing State Machine..."):
            res = agentic_rag.generate_answer(user_query)
        latency = time.time() - t0
        
        st.markdown(f'<div class="answer-box"><strong>Final Generated Answer:</strong><br>{res["answer"]}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-label">Execution Time</div>
                <div class="metric-value">{latency:.2f}s</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Context Documents</div>
                <div class="metric-value">{len(res['retrieved_docs'])}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Self-Correction Loops</div>
                <div class="metric-value">{res['loop_step']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if res.get('sub_questions'):
            st.markdown("<br><strong>Query Decomposition Breakdown:</strong>", unsafe_allow_html=True)
            for i, sq in enumerate(res['sub_questions'], 1):
                st.markdown(f'<span class="tag">Sub-query {i}: {sq}</span>', unsafe_allow_html=True)
                
        with st.expander("Inspected Context Passages"):
            for i, doc in enumerate(res['retrieved_docs'], 1):
                st.markdown(f"**Passage {i}**")
                st.text(doc)

    else:
        st.markdown('<div class="section-title">Naive RAG Baseline Execution</div>', unsafe_allow_html=True)
        t0 = time.time()
        with st.spinner("Executing Retrieval & Generation..."):
            res = naive_rag.generate_answer(user_query)
        latency = time.time() - t0
        
        st.markdown(f'<div class="answer-box"><strong>Answer:</strong> {res["answer"]}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-label">Latency</div>
                <div class="metric-value">{latency:.2f}s</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Retrieved Documents</div>
                <div class="metric-value">{len(res['retrieved_docs'])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Top-5 Retrieved Context Passages"):
            for i, doc in enumerate(res['retrieved_docs'], 1):
                st.markdown(f"**Passage {i}**")
                st.text(doc)
