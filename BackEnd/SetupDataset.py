#!/usr/bin/env python3
"""
GuardianLM — Dataset Setup Script
==================================
One-time script that downloads the neuralchemy/Prompt-injection-dataset from
HuggingFace and prepares cached assets for two detection layers:

  1. Similarity Analysis  → pre-computed sentence embeddings (data/sa_vectors.pt)
  2. ML Classifier        → formatted train/val splits   (data/mlc_dataset/)

Usage
-----
    cd BackEnd
    python setup_dataset.py

    # Optional: only prepare one layer
    python setup_dataset.py --layer sa
    python setup_dataset.py --layer mlc

Requirements
------------
    pip install datasets sentence-transformers torch
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
SA_CACHE     = DATA_DIR / "sa_vectors.pt"
SA_META      = DATA_DIR / "sa_metadata.json"
MLC_DIR      = DATA_DIR / "mlc_dataset"

DATASET_ID   = "neuralchemy/Prompt-injection-dataset"
SA_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Category mapping (dataset categories → SA signal categories) ──────────
# The neuralchemy dataset uses fine-grained categories.  We map them to the
# broader signal categories already used in sa.py.
_CATEGORY_MAP: dict[str, str] = {
    # Direct override / instruction manipulation
    "direct_injection":       "direct_override",
    "instruction_override":   "direct_override",
    "context_manipulation":   "direct_override",
    # Jailbreak / persona
    "jailbreak":              "jailbreak_persona",
    "role_play":              "jailbreak_persona",
    "persona_hijack":         "jailbreak_persona",
    "character_play":         "jailbreak_persona",
    # System extraction
    "system_prompt_leak":     "system_extraction",
    "information_extraction": "system_extraction",
    "prompt_extraction":      "system_extraction",
    # Payload smuggling
    "payload_smuggling":      "payload_smuggling",
    "encoding_attack":        "payload_smuggling",
    "translation_attack":     "payload_smuggling",
    # Token / delimiter injection
    "token_injection":        "token_injection",
    "delimiter_injection":    "token_injection",
    "xml_injection":          "token_injection",
    # Data exfiltration
    "data_exfiltration":      "data_exfiltration",
    "credential_extraction":  "data_exfiltration",
    # Control injection
    "control_injection":      "control_injection",
    "safety_bypass":          "control_injection",
    "filter_evasion":         "control_injection",
    # Obfuscation
    "obfuscation":            "obfuscation",
    "encoding_evasion":       "obfuscation",
    # Persona lock
    "persona_lock":           "persona_lock",
}


def _map_category(raw_cat: str) -> str:
    """Map a dataset category to the SA signal category.  Unknown → 'other'."""
    return _CATEGORY_MAP.get(raw_cat, "other")


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY ANALYSIS SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_sa():
    """
    Downloads injection samples, embeds them with MiniLM, and saves:
      - data/sa_vectors.pt   → dict with 'embeddings' tensor + 'texts' list
      - data/sa_metadata.json → category tags and stats
    """
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("SIMILARITY ANALYSIS — preparing vector library")
    print("=" * 60)

    # ── Load dataset (full config) ────────────────────────────────────────
    print(f"\n→ Loading {DATASET_ID} (full config)...")
    ds = load_dataset(DATASET_ID, "full", split="train")
    print(f"  Total samples: {len(ds)}")

    # ── Filter injection samples only ─────────────────────────────────────
    injections = ds.filter(lambda x: x["label"] == 1)
    print(f"  Injection samples: {len(injections)}")

    # ── Deduplicate by text ───────────────────────────────────────────────
    seen = set()
    unique_texts      = []
    unique_categories = []
    for row in injections:
        text = row["text"].strip()
        if text and text not in seen:
            seen.add(text)
            unique_texts.append(text)
            unique_categories.append(_map_category(row.get("category", "")))

    print(f"  Unique injection texts: {len(unique_texts)}")

    # ── Embed ─────────────────────────────────────────────────────────────
    print(f"\n→ Loading embedding model: {SA_MODEL_ID}")
    model = SentenceTransformer(SA_MODEL_ID)

    print(f"→ Encoding {len(unique_texts)} vectors (this may take a minute)...")
    embeddings = model.encode(unique_texts, convert_to_tensor=True,
                              show_progress_bar=True, batch_size=128)

    # ── Save ──────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        "embeddings":  embeddings,
        "texts":       unique_texts,
        "categories":  unique_categories,
    }, SA_CACHE)
    print(f"\n✓ Saved embeddings → {SA_CACHE}  ({SA_CACHE.stat().st_size / 1e6:.1f} MB)")

    # Metadata (human-readable stats)
    cat_counts = {}
    for c in unique_categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    meta = {
        "dataset":        DATASET_ID,
        "total_vectors":  len(unique_texts),
        "categories":     cat_counts,
        "model":          SA_MODEL_ID,
    }
    SA_META.write_text(json.dumps(meta, indent=2))
    print(f"✓ Saved metadata  → {SA_META}")

    print(f"\n  Category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<25s} {count:>5d}")


# ═══════════════════════════════════════════════════════════════════════════
# ML CLASSIFIER SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_mlc():
    """
    Prepares the dataset in the format expected by finetune_mlc.py:
      - data/mlc_dataset/train.json
      - data/mlc_dataset/val.json
    """
    from datasets import load_dataset

    print("=" * 60)
    print("ML CLASSIFIER — preparing fine-tuning dataset")
    print("=" * 60)

    # ── Load dataset (full config, with splits) ───────────────────────────
    print(f"\n→ Loading {DATASET_ID} (full config, all splits)...")
    ds = load_dataset(DATASET_ID, "full")

    MLC_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "test"]:
        if split_name not in ds:
            print(f"  ⚠ Split '{split_name}' not found, skipping")
            continue

        split = ds[split_name]
        out_name = "val.json" if split_name == "test" else "train.json"
        out_path = MLC_DIR / out_name

        records = []
        for row in split:
            records.append({
                "text":     row["text"],
                "label":    row["label"],
                "category": row.get("category", ""),
                "severity": row.get("severity", ""),
            })

        out_path.write_text(json.dumps(records, ensure_ascii=False))
        print(f"  ✓ {split_name} → {out_path}  ({len(records)} samples)")

    # ── Stats ─────────────────────────────────────────────────────────────
    if "train" in ds:
        train = ds["train"]
        n_inj  = sum(1 for r in train if r["label"] == 1)
        n_safe = sum(1 for r in train if r["label"] == 0)
        print(f"\n  Train split: {n_inj} injection + {n_safe} benign = {len(train)} total")

    print(f"\n✓ ML Classifier dataset ready at {MLC_DIR}/")
    print(f"  Run: python finetune_mlc.py")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GuardianLM — Download and prepare neuralchemy dataset"
    )
    parser.add_argument(
        "--layer", choices=["sa", "mlc", "all"], default="all",
        help="Which layer to prepare data for (default: all)"
    )
    args = parser.parse_args()

    if args.layer in ("sa", "all"):
        setup_sa()
        print()

    if args.layer in ("mlc", "all"):
        setup_mlc()
        print()

    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()