import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from src.llm import get_primary_llm, get_fallback_llm
from src.agent_state import (
    AgentState, 
    RouteQuery, 
    GradeDocuments, 
    DecomposeQuery, 
    GradeHallucination
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "hotpotqa_corpus"

class AgenticRAG:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        
        # 1. Local Vector Store & Embedder
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # 2. LLMs
        self.llm = get_primary_llm(temperature=0.0)
        self.structured_llm = get_primary_llm(temperature=0.0)
        
        # 3. Web Search (Tavily)
        tavily_key = os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
        
        # 4. Build LangGraph Workflow
        self.graph = self._build_graph()

    # ---------------------------------------------------------
    # Helper: ChromaDB Retrieval
    # ---------------------------------------------------------
    def _query_chroma(self, query: str, k: int = 5) -> List[str]:
        query_emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=[query_emb], n_results=k)
        return res["documents"][0] if res["documents"] else []

    # ---------------------------------------------------------
    # Graph Nodes
    # ---------------------------------------------------------

    def route_question(self, state: AgentState) -> Dict[str, Any]:
        """Classify question into: direct, decomposer, or vectorstore."""
        question = state["question"]
        return {"current_query": question, "loop_step": 0}

    def decompose_node(self, state: AgentState) -> Dict[str, Any]:
        """Decomposes a complex multi-hop question into 2 sub-queries and retrieves both."""
        question = state["question"]
        prompt = ChatPromptTemplate.from_template(
            """You are an expert query decomposer. Break down this complex multi-hop question into 2 simple, independent search sub-questions.
Original Question: {question}"""
        )
        decomposer_chain = prompt | self.structured_llm.with_structured_output(DecomposeQuery)
        
        try:
            res = decomposer_chain.invoke({"question": question})
            sub_qs = res.sub_questions
        except Exception:
            sub_qs = [question]
            
        all_docs = []
        for sq in sub_qs:
            docs = self._query_chroma(sq, k=4)
            all_docs.extend(docs)
            
        # Also query the original question to make sure we don't miss any direct matches
        all_docs.extend(self._query_chroma(question, k=3))
            
        # Deduplicate while preserving order
        unique_docs = list(dict.fromkeys(all_docs))
        return {"sub_questions": sub_qs, "documents": unique_docs}

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Standard retrieval node from ChromaDB."""
        query = state.get("current_query") or state["question"]
        docs = self._query_chroma(query, k=self.top_k)
        return {"documents": docs}

    def grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """Grades retrieved documents for relevance to the user question."""
        question = state["question"]
        docs = state.get("documents", [])
        
        if not docs:
            return {"web_search_needed": True}
            
        prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of retrieved documents to a user question.
Context:
{context}

Question: {question}

Do the documents contain factual keywords/information to help answer the question? Answer 'yes' or 'no'."""
        )
        grader_chain = prompt | self.structured_llm.with_structured_output(GradeDocuments)
        
        try:
            res = grader_chain.invoke({"context": "\n\n".join(docs[:4]), "question": question})
            is_relevant = res.binary_score == "yes"
        except Exception:
            is_relevant = True
            
        return {"web_search_needed": not is_relevant}

    def rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Rewrites the query into better search keywords when retrieval was insufficient."""
        question = state["question"]
        loop_step = state.get("loop_step", 0) + 1
        
        prompt = ChatPromptTemplate.from_template(
            """Look at the input question and formulate an improved, keyword-rich search query optimized for vector retrieval.
Initial Question: {question}
Improved Search Query (provide only the query string):"""
        )
        chain = prompt | self.llm
        res = chain.invoke({"question": question})
        rewritten = res.content if isinstance(res.content, str) else str(res.content)
        
        return {"current_query": rewritten.strip(), "loop_step": loop_step}

    def web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Fallback web search using Tavily API."""
        query = state.get("current_query") or state["question"]
        docs = state.get("documents", [])
        
        if self.tavily:
            try:
                search_res = self.tavily.search(query=query, max_results=3)
                web_docs = [f"Web Source ({r.get('url')}):\n{r.get('content')}" for r in search_res.get("results", [])]
                docs.extend(web_docs)
            except Exception:
                pass
                
        return {"documents": docs, "web_search_needed": False}

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizes final factual answer from all gathered documents."""
        question = state["question"]
        docs = state.get("documents", [])
        context_str = "\n\n---\n\n".join(docs)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a precise, factual question-answering assistant.
Read the context passages carefully and answer the target question directly.
Pay close attention to what the question is specifically asking for (e.g., person name, government position/title, date, location, or yes/no).

Context Passages:
{context}

Target Question: {question}

Provide the exact, concise answer (keep it as short and direct as possible, typically 1 to 5 words):"""
        )
        chain = prompt | self.llm
        res = chain.invoke({"context": context_str, "question": question})
        answer = res.content if isinstance(res.content, str) else str(res.content)
        return {"generation": answer.strip()}

    def direct_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Direct answer for chit-chat."""
        prompt = ChatPromptTemplate.from_template("Respond kindly and concisely to: {question}")
        chain = prompt | self.llm
        res = chain.invoke({"question": state["question"]})
        return {"generation": str(res.content).strip()}

    # ---------------------------------------------------------
    # Conditional Routing Logic
    # ---------------------------------------------------------

    def _decide_route(self, state: AgentState) -> str:
        question = state["question"]
        prompt = ChatPromptTemplate.from_template(
            "Route question to 'direct', 'decomposer' (multi-hop/bridge with multiple entities), or 'vectorstore': {question}"
        )
        try:
            decision = (prompt | self.structured_llm.with_structured_output(RouteQuery)).invoke({"question": question})
            return decision.destination
        except Exception:
            return "decomposer" if " and " in question or "who" in question.lower() else "vectorstore"

    def _decide_after_grading(self, state: AgentState) -> str:
        if not state.get("web_search_needed", False):
            return "generate"
        if state.get("loop_step", 0) >= 2:
            return "web_search"
        return "rewrite"

    # ---------------------------------------------------------
    # Graph Construction
    # ---------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("route_node", self.route_question)
        workflow.add_node("decompose", self.decompose_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_docs", self.grade_documents_node)
        workflow.add_node("rewrite", self.rewrite_query_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("direct", self.direct_answer_node)
        
        # Set Entry Point
        workflow.set_entry_point("route_node")
        
        # Conditional Edge from Router
        workflow.add_conditional_edges(
            "route_node",
            self._decide_route,
            {
                "direct": "direct",
                "decomposer": "decompose",
                "vectorstore": "retrieve"
            }
        )
        
        # Edges to Grader
        workflow.add_edge("decompose", "grade_docs")
        workflow.add_edge("retrieve", "grade_docs")
        
        # Conditional Edge from Grader
        workflow.add_conditional_edges(
            "grade_docs",
            self._decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite",
                "web_search": "web_search"
            }
        )
        
        # Loop back from Rewrite to Retrieve
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("web_search", "generate")
        
        # End nodes
        workflow.add_edge("generate", END)
        workflow.add_edge("direct", END)
        
        return workflow.compile()

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Executes the full LangGraph agentic workflow."""
        inputs: AgentState = {
            "question": question,
            "current_query": question,
            "sub_questions": [],
            "documents": [],
            "generation": "",
            "loop_step": 0,
            "web_search_needed": False,
            "grounded": True
        }
        
        final_state = self.graph.invoke(inputs)
        return {
            "question": question,
            "answer": final_state.get("generation", ""),
            "sub_questions": final_state.get("sub_questions", []),
            "retrieved_docs": final_state.get("documents", []),
            "loop_step": final_state.get("loop_step", 0)
        }

def interactive_demo():
    print("=" * 65)
    print("🚀 LANGGRAPH AGENTIC RAG (Testing On Real HotpotQA)")
    print("=" * 65)
    
    agent = AgenticRAG(top_k=5)
    
    # Test on the exact question Naive RAG failed on!
    test_question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    
    print(f"\n❓ Question: {test_question}")
    print("🎯 Gold Answer: Chief of Protocol")
    print("⏳ Running Agentic State Machine (Routing -> Decomposing -> Grading)...")
    
    res = agent.generate_answer(test_question)
    print(f"\n🤖 Agent Answer: {res['answer']}")
    if res['sub_questions']:
        print(f"🔀 Decomposed Sub-Questions: {res['sub_questions']}")
    print(f"📚 Total Context Documents Gathered: {len(res['retrieved_docs'])}")
    print(f"🔄 Rewriting Loops: {res['loop_step']}")
    print("=" * 65)

if __name__ == "__main__":
    interactive_demo()
