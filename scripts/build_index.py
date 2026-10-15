"""
Vector Index Construction & BM25 Lexical Index for CIT Knowledge AI
Builds persistent ChromaDB collections for:
1. Baseline Model (all-MiniLM-L6-v2)
2. Fine-tuned CIT Model (cit-minilm-l6-v2)
Also builds BM25 lexical index for hybrid search experimentation.
"""

import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
CHROMA_PERSIST_DIR = os.path.join(VECTOR_STORE_DIR, "chroma")
BM25_CORPUS_PATH = os.path.join(VECTOR_STORE_DIR, "bm25_corpus.json")

CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
BASELINE_MODEL_DIR = os.path.join(BASE_DIR, "models", "baseline", "all-MiniLM-L6-v2")
FINETUNED_MODEL_DIR = os.path.join(BASE_DIR, "models", "finetuned", "cit-minilm-l6-v2")

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

def tokenize_for_bm25(text: str) -> list:
    import re
    return re.findall(r"\b\w+\b", text.lower())

def build_indices(reset: bool = True):
    print("=" * 70)
    print("BUILDING VECTOR INDICES AND LEXICAL BM25 INDEX")
    print(f"Chroma persistence path: {CHROMA_PERSIST_DIR}")
    print("=" * 70)

    # 1. Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks for indexing.")

    # 2. Setup ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        for col_name in ["cit_baseline", "cit_finetuned"]:
            try:
                client.delete_collection(col_name)
                print(f"Reset collection: {col_name}")
            except Exception:
                pass

    # 3. Build Baseline Collection
    print(f"\n--- Loading Baseline Model from {BASELINE_MODEL_DIR} ---")
    baseline_model = SentenceTransformer(BASELINE_MODEL_DIR)
    baseline_col = client.get_or_create_collection(
        name="cit_baseline",
        metadata={"description": "Baseline pretrained all-MiniLM-L6-v2 embeddings"}
    )

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        # ChromaDB requires primitive metadata types (str, int, float, bool)
        meta = {
            "chunk_id": str(c["chunk_id"]),
            "original_document_id": str(c["original_document_id"]),
            "document_title": str(c["document_title"]),
            "category": str(c["category"]),
            "source_url": str(c.get("source_url") or "https://cit.edu.in/"),
            "academic_year": str(c.get("academic_year") or "N/A"),
            "section_heading": str(c.get("section_heading") or "General"),
            "page_number": int(c.get("page_number")) if c.get("page_number") is not None else -1,
            "word_count": int(c.get("word_count", 0)),
            "content_hash": str(c.get("content_hash", ""))
        }
        metadatas.append(meta)

    print("Generating baseline embeddings...")
    baseline_embeddings = baseline_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    baseline_col.add(
        ids=ids,
        embeddings=baseline_embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    print(f"Indexed {baseline_col.count()} records into 'cit_baseline'.")

    # 4. Build Fine-tuned Collection
    print(f"\n--- Loading Fine-tuned Model from {FINETUNED_MODEL_DIR} ---")
    finetuned_model = SentenceTransformer(FINETUNED_MODEL_DIR)
    finetuned_col = client.get_or_create_collection(
        name="cit_finetuned",
        metadata={"description": "Fine-tuned CIT-domain MiniLM-L6-v2 embeddings"}
    )

    print("Generating fine-tuned embeddings...")
    finetuned_embeddings = finetuned_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    finetuned_col.add(
        ids=ids,
        embeddings=finetuned_embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    print(f"Indexed {finetuned_col.count()} records into 'cit_finetuned'.")

    # 5. Build BM25 Lexical Index
    print("\n--- Building BM25 Lexical Index ---")
    bm25_docs = []
    for c in chunks:
        tokens = tokenize_for_bm25(c["text"])
        bm25_docs.append({
            "chunk_id": c["chunk_id"],
            "tokens": tokens,
            "metadata": {
                "chunk_id": c["chunk_id"],
                "document_title": c["document_title"],
                "category": c["category"],
                "source_url": c.get("source_url") or "https://cit.edu.in/",
                "academic_year": c.get("academic_year") or "N/A",
                "section_heading": c.get("section_heading") or "General"
            },
            "text": c["text"]
        })

    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as fp:
        json.dump(bm25_docs, fp, indent=2)

    print(f"BM25 corpus saved to: {BM25_CORPUS_PATH}")
    print("=" * 70)
    print("ALL VECTOR AND LEXICAL INDICES BUILT AND PERSISTED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    build_indices(reset=True)
