#!/usr/bin/env python3
"""
GuardianLM — ML Classifier Fine-Tuning Script
===============================================
Fine-tunes protectai/deberta-v3-base-prompt-injection-v2 on the neuralchemy
dataset for improved prompt injection detection.

Prerequisites
-------------
    python setup_dataset.py --layer mlc

Usage
-----
    cd BackEnd
    python finetune_mlc.py                        # default settings
    python finetune_mlc.py --epochs 5 --lr 2e-5   # custom
    python finetune_mlc.py --batch-size 8          # if GPU-limited

Output
------
    models/deberta-v3-prompt-injection-finetuned/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...

    mlc.py will auto-detect this directory and load the local model.

Requirements
------------
    pip install transformers datasets torch scikit-learn
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data" / "mlc_dataset"
OUTPUT_DIR     = BASE_DIR / "models" / "deberta-v3-prompt-injection-finetuned"

BASE_MODEL     = "protectai/deberta-v3-base-prompt-injection-v2"


def load_json_dataset(path: Path) -> Dataset:
    """Load a JSON array file into a HuggingFace Dataset."""
    records = json.loads(path.read_text())
    return Dataset.from_dict({
        "text":  [r["text"]  for r in records],
        "label": [r["label"] for r in records],
    })


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for the Trainer."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa on neuralchemy prompt injection dataset"
    )
    parser.add_argument("--epochs",     type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int,   default=16,   help="Batch size (default: 16)")
    parser.add_argument("--max-length", type=int,   default=512,  help="Max token length (default: 512)")
    parser.add_argument("--warmup",     type=float, default=0.1,  help="Warmup ratio (default: 0.1)")
    args = parser.parse_args()

    # ── Validate prerequisites ────────────────────────────────────────────
    train_path = DATA_DIR / "train.json"
    val_path   = DATA_DIR / "val.json"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        print(f"Run:  python setup_dataset.py --layer mlc")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("ML CLASSIFIER FINE-TUNING")
    print("=" * 60)

    print(f"\n→ Loading training data from {DATA_DIR}/")
    train_ds = load_json_dataset(train_path)
    print(f"  Train samples: {len(train_ds)}")

    val_ds = None
    if val_path.exists():
        val_ds = load_json_dataset(val_path)
        print(f"  Val samples:   {len(val_ds)}")
    else:
        # Split 10% for validation
        split = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        train_ds = split["train"]
        val_ds   = split["test"]
        print(f"  Train samples (after 90/10 split): {len(train_ds)}")
        print(f"  Val samples:                       {len(val_ds)}")

    # ── Load base model + tokenizer ───────────────────────────────────────
    print(f"\n→ Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )

    # Update label mapping for clearer inference output
    model.config.id2label = {0: "SAFE", 1: "INJECTION"}
    model.config.label2id = {"SAFE": 0, "INJECTION": 1}

    # ── Tokenize ──────────────────────────────────────────────────────────
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    print(f"→ Tokenizing (max_length={args.max_length})...")
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # ── Training arguments ────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n→ Device: {device}")
    print(f"→ Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n→ Starting fine-tuning...\n")
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\n→ Final evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\n✓ Fine-tuned model saved to {OUTPUT_DIR}/")
    print(f"  mlc.py will auto-detect this directory on next restart.")


if __name__ == "__main__":
    main()