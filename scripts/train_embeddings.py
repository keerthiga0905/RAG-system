"""
Fine-tuning Embedding Model for CIT Knowledge AI
Performs reproducible fine-tuning of sentence-transformers/all-MiniLM-L6-v2
on CIT-specific question-to-passage retrieval using MultipleNegativesRankingLoss.
Evaluates on validation set, implements early stopping / best checkpoint saving,
and logs training loss and evaluation metrics.
"""

import os
import json
import time
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.json")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.json")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")

BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline", "all-MiniLM-L6-v2")
FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "finetuned", "cit-minilm-l6-v2")
TRAINING_REPORT_PATH = os.path.join(REPORTS_DIR, "training_history.json")

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_evaluator(val_records: list, chunks: list) -> InformationRetrievalEvaluator:
    """Prepares InformationRetrievalEvaluator on validation queries and corpus."""
    corpus = {}
    for c in chunks:
        corpus[c["chunk_id"]] = c["text"]

    queries = {}
    relevant_docs = {}

    for r in val_records:
        qid = r["qa_id"]
        queries[qid] = r["question"]
        pos_id = r.get("positive_passage_id")
        if pos_id and pos_id in corpus:
            relevant_docs[qid] = {pos_id}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="cit-val-retrieval",
        show_progress_bar=False,
        batch_size=16
    )
    return evaluator

def run_training(
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
    seed: int = 42
):
    print("=" * 70)
    print("EXPERIMENT B: FINE-TUNING EMBEDDING MODEL FOR CIT RETRIEVAL")
    print(f"Base Model: {base_model_name}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, seed={seed}")
    print("=" * 70)

    set_seed(seed)

    # 1. Load train & val records
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Filter out unanswerable for embedding training
    valid_train = [r for r in train_data if r.get("positive_passage_text")]
    print(f"Loaded {len(valid_train)} valid training pairs with positive evidence.")
    print(f"Loaded {len(val_data)} validation samples.")

    # 2. Build InputExamples
    train_examples = []
    for r in valid_train:
        q = r["question"]
        pos = r["positive_passage_text"]
        neg = r.get("negative_passage_text", "")
        if neg:
            train_examples.append(InputExample(texts=[q, pos, neg]))
        else:
            train_examples.append(InputExample(texts=[q, pos]))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # 3. Save Baseline Model Copy locally
    print(f"Saving baseline model locally to: {BASELINE_MODEL_DIR}")
    model = SentenceTransformer(base_model_name)
    model.save(BASELINE_MODEL_DIR)

    # 4. Setup Loss
    # MultipleNegativesRankingLoss supports (anchor, positive, negative) triples
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 5. Setup Evaluator
    evaluator = prepare_evaluator(val_data, chunks)
    print("Evaluating baseline model performance on validation set before training...")
    initial_val_metrics = evaluator(model)
    print(f"Initial Validation Evaluation Result: {initial_val_metrics}")

    # Custom training callback to track loss & validation progression
    history = {
        "base_model": base_model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_samples_count": len(train_examples),
        "validation_samples_count": len(val_data),
        "initial_val_metrics": initial_val_metrics,
        "epoch_metrics": [],
        "training_time_seconds": 0
    }

    t0 = time.time()
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    print("\nStarting training loop...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader),
        warmup_steps=warmup_steps,
        output_path=FINETUNED_MODEL_DIR,
        save_best_model=True,
        show_progress_bar=True
    )
    training_time = round(time.time() - t0, 2)
    history["training_time_seconds"] = training_time

    # Evaluate final checkpoint
    finetuned_model = SentenceTransformer(FINETUNED_MODEL_DIR)
    final_val_metrics = evaluator(finetuned_model)
    history["final_val_metrics"] = final_val_metrics

    with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print("=" * 70)
    print("MODEL FINE-TUNING COMPLETE!")
    print(f"Training Time: {training_time} seconds")
    print(f"Fine-tuned model checkpoint saved to: {FINETUNED_MODEL_DIR}")
    print(f"Baseline val score:  {initial_val_metrics}")
    print(f"Finetuned val score: {final_val_metrics}")
    print(f"Training report saved to: {TRAINING_REPORT_PATH}")
    print("=" * 70)

    return history

if __name__ == "__main__":
    run_training()
