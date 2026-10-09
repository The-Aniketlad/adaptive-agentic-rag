import os
import sys
import re
import string
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.naive_rag import NaiveRAG
from src.agent import AgenticRAG

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_SET_PATH = DATA_DIR / "hotpotqa_eval_500.json"
RESULTS_DIR = DATA_DIR / "results"

# ---------------------------------------------------------
# Official HotpotQA / SQuAD Evaluation Normalization & Metrics
# ---------------------------------------------------------

def normalize_text(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))

def compute_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
        
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
        
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# ---------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------

def run_benchmark(pipeline_name: str, pipeline_runner, eval_data: List[Dict[str, Any]], output_csv: Path) -> Dict[str, Any]:
    print(f"\n🚀 Running Benchmark on {len(eval_data)} questions for: [{pipeline_name}]...")
    
    results = []
    
    for item in tqdm(eval_data, desc=f"Evaluating {pipeline_name}"):
        q_id = item["id"]
        question = item["question"]
        gold_answer = item["answer"]
        q_type = item.get("type", "unknown")
        
        start_time = time.time()
        try:
            res = pipeline_runner.generate_answer(question)
            prediction = res.get("answer", "")
        except Exception as e:
            prediction = f"ERROR: {str(e)}"
        latency = round(time.time() - start_time, 3)
        
        em = compute_exact_match(prediction, gold_answer)
        f1 = compute_f1_score(prediction, gold_answer)
        
        results.append({
            "id": q_id,
            "question": question,
            "type": q_type,
            "gold_answer": gold_answer,
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "latency_sec": latency
        })
        
    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    avg_em = df["em"].mean() * 100
    avg_f1 = df["f1"].mean() * 100
    avg_latency = df["latency_sec"].mean()
    
    breakdown = df.groupby("type")[["em", "f1"]].mean() * 100
    
    print("\n" + "=" * 60)
    print(f"📊 BENCHMARK RESULTS: {pipeline_name}")
    print("=" * 60)
    print(f"Total Questions:   {len(df)}")
    print(f"Exact Match (EM):  {avg_em:.2f}%")
    print(f"F1 Score:          {avg_f1:.2f}%")
    print(f"Avg Latency:       {avg_latency:.2f}s per question")
    print("\n📈 Type Breakdown:")
    print(breakdown.to_string())
    print(f"\n💾 Saved full detailed results to: {output_csv}")
    print("=" * 60)
    
    return {
        "pipeline": pipeline_name,
        "total": len(df),
        "em": avg_em,
        "f1": avg_f1,
        "latency": avg_latency
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipelines on HotpotQA")
    parser.add_argument("--num_samples", type=int, default=25, help="Number of questions to evaluate")
    parser.add_argument("--pipeline", type=str, choices=["naive", "agent", "all"], default="agent", help="Pipeline to evaluate")
    args = parser.parse_args()
    
    if not EVAL_SET_PATH.exists():
        print(f"❌ Evaluation dataset not found at {EVAL_SET_PATH}. Run 'python src/ingest.py' first.")
        return
        
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        full_eval_data = json.load(f)
        
    eval_subset = full_eval_data[:args.num_samples]
    
    if args.pipeline in ["naive", "all"]:
        naive_pipeline = NaiveRAG(top_k=5)
        output_csv = RESULTS_DIR / f"benchmark_naive_rag_{args.num_samples}q.csv"
        run_benchmark("Naive RAG Baseline", naive_pipeline, eval_subset, output_csv)

    if args.pipeline in ["agent", "all"]:
        agent_pipeline = AgenticRAG(top_k=5)
        output_csv = RESULTS_DIR / f"benchmark_agentic_rag_{args.num_samples}q.csv"
        run_benchmark("LangGraph Agentic RAG", agent_pipeline, eval_subset, output_csv)

if __name__ == "__main__":
    main()
