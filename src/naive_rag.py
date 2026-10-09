import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_primary_llm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "hotpotqa_corpus"
EVAL_SET_PATH = DATA_DIR / "hotpotqa_eval_500.json"

class NaiveRAG:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        
        # 1. Connect to local ChromaDB
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        
        # 2. Local Embedding Model
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # 3. LLM (Groq / Gemini)
        self.llm = get_primary_llm(temperature=0.0)
        
        # 4. Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a concise question-answering assistant. Answer the question based ONLY on the provided context passages.
If the context does not contain enough facts to answer, respond with 'Insufficient context.'

Context Passages:
{context}

Question: {question}

Provide the exact, concise answer (keep it as short and direct as possible):"""
        )
        
        self.chain = self.prompt | self.llm

    def retrieve(self, query: str) -> List[str]:
        """
        Embeds the query and fetches top_k most similar passages from ChromaDB.
        """
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        
        documents = results["documents"][0] if results["documents"] else []
        return documents

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Runs the full Naive RAG pipeline: Retrieve -> Stuff -> Generate.
        """
        retrieved_docs = self.retrieve(question)
        context_str = "\n\n---\n\n".join(retrieved_docs)
        
        response = self.chain.invoke({
            "context": context_str,
            "question": question
        })
        
        # Extract text from response
        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in answer_text]).strip()
        else:
            answer_text = str(answer_text).strip()
            
        return {
            "question": question,
            "answer": answer_text,
            "retrieved_docs": retrieved_docs
        }

def interactive_demo():
    print("=" * 65)
    print("🧠 NAIVE RAG BASELINE (Testing on Real HotpotQA Corpus)")
    print("=" * 65)
    
    rag = NaiveRAG(top_k=5)
    
    # Load first 3 real HotpotQA questions from our indexed dataset
    if EVAL_SET_PATH.exists():
        with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)[:3]
    else:
        eval_data = [
            {"question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes"}
        ]
        
    for item in eval_data:
        q = item["question"]
        gold_ans = item.get("answer", "N/A")
        print(f"\n❓ Question: {q}")
        print(f"🎯 Ground Truth (Gold Answer): {gold_ans}")
        
        result = rag.generate_answer(q)
        print(f"🤖 Naive RAG Answer: {result['answer']}")
        print(f"📚 Top-1 Retrieved Title: {result['retrieved_docs'][0].splitlines()[0] if result['retrieved_docs'] else 'None'}")
        print("-" * 65)

if __name__ == "__main__":
    interactive_demo()
