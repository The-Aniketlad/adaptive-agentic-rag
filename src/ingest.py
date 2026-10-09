import os
import json
from pathlib import Path
from datasets import load_dataset
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH = DATA_DIR / "chroma_db"
EVAL_SET_PATH = DATA_DIR / "hotpotqa_eval_500.json"
COLLECTION_NAME = "hotpotqa_corpus"

def load_and_prepare_hotpotqa(num_samples: int = 500):
    """
    Loads HotpotQA (distractor setting) from Hugging Face.
    Extracts eval samples and pools all unique context paragraphs into a corpus.
    """
    print(f"📥 Loading HotpotQA dataset from Hugging Face (taking {num_samples} samples)...")
    
    # Load validation split of HotpotQA distractor
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    subset = dataset.select(range(min(num_samples, len(dataset))))
    
    eval_data = []
    unique_passages = {} # title_sentence_idx -> text
    
    print("🔄 Processing questions and extracting context passages...")
    for item in tqdm(subset, desc="Extracting"):
        q_id = item["id"]
        question = item["question"]
        answer = item["answer"]
        question_type = item["type"] # 'comparison' or 'bridge'
        
        # Supporting facts
        supporting_facts = {
            (title, sent_id) for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
        }
        
        # Context contains 10 paragraphs (titles + sentences list)
        context_titles = item["context"]["title"]
        context_sentences = item["context"]["sentences"]
        
        gold_passages = []
        
        for title, sentences in zip(context_titles, context_sentences):
            full_text = f"Title: {title}\n" + " ".join(sentences)
            doc_id = f"doc_{hash(title) & 0xffffffff}_{len(full_text)}"
            
            # Check if this paragraph contains supporting facts
            is_gold = any(title == sf_title for sf_title, _ in supporting_facts)
            if is_gold:
                gold_passages.append(full_text)
                
            if doc_id not in unique_passages:
                unique_passages[doc_id] = {
                    "id": doc_id,
                    "title": title,
                    "text": full_text
                }
                
        eval_data.append({
            "id": q_id,
            "question": question,
            "answer": answer,
            "type": question_type,
            "gold_passages": gold_passages
        })
        
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation split to json
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(eval_data)} eval questions to: {EVAL_SET_PATH}")
    print(f"📚 Total unique context passages collected: {len(unique_passages)}")
    
    return list(unique_passages.values())

def build_vector_store(passages, batch_size: int = 64):
    """
    Embeds passages using local BGE-small-en-v1.5 and stores in ChromaDB.
    """
    print(f"\n⚡ Initializing ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    
    # Reset collection if exists to avoid stale data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Cosine similarity for dense retrieval
    )
    
    print("🤖 Loading embedding model: BAAI/bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print(f"🚀 Embedding and indexing {len(passages)} passages in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(passages), batch_size), desc="Indexing ChromaDB"):
        batch = passages[i:i + batch_size]
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        metadatas = [{"title": doc["title"]} for doc in batch]
        
        # Generate dense embeddings (384-dimensions)
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
    print(f"\n🎉 Successfully indexed {collection.count()} passages into ChromaDB!")

def main():
    passages = load_and_prepare_hotpotqa(num_samples=500)
    build_vector_store(passages)
    print("\n✅ Data ingestion & Vector indexing completed successfully!")

if __name__ == "__main__":
    main()
