#!/usr/bin/env python3
"""Fine-tune BGE-small on Engram-shaped triplets (hard negatives + LoCoMo QA).

Phase 10 of the Engram embedding plan: replaces the PersonaChat smoke-tune
(finetune_embed.py) with a proper retrieval-objective training using:
  - (query, positive, hard_negative) triplets from mine_hard_negatives.py
  - TripletLoss (margin-based) — correct objective for retrieval with hard negatives
  - Optional fallback to MultipleNegativesRankingLoss for pairs without hard negs

Data priority:
  40–60%  Engram-shaped triplets  (engram_triplets.jsonl — from mine_hard_negatives.py)
  [future] synthetic Mockaroo / external support pairs

Acceptance gates (must ALL pass on LoCoMo recall eval after training):
  R@40  >= 93.08  (best known: rrf_k20 / B.db)
  R@5   >= 74.07
  multi_hop R@5 >= 61.57
  questions_with_evidence >= 1531

Usage:
    python finetune_embed_v2.py
    python finetune_embed_v2.py --output bge-small-engram-v2 --epochs 2
    python finetune_embed_v2.py --loss mnrl   # use MultipleNegativesRankingLoss only
    python finetune_embed_v2.py --max-triplets 10000 --epochs 1   # fast test
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base-model",    default="BAAI/bge-small-en-v1.5")
parser.add_argument("--output",        default="bge-small-engram-v2")
parser.add_argument("--triplets-file", default="engram_triplets.jsonl")
parser.add_argument("--loss",          default="triplet",
                    choices=["triplet", "mnrl"],
                    help="triplet: TripletLoss (best with hard negatives). "
                         "mnrl: MultipleNegativesRankingLoss (in-batch negatives only).")
parser.add_argument("--margin",        type=float, default=0.5,
                    help="Triplet margin (default 0.5 — cosine space).")
parser.add_argument("--max-triplets",  type=int, default=0,
                    help="Cap on training examples (0 = no cap).")
parser.add_argument("--batch-size",    type=int, default=32)
parser.add_argument("--epochs",        type=int, default=1)
parser.add_argument("--max-seq-len",   type=int, default=128)
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--val-frac",      type=float, default=0.1,
                    help="Fraction of data to hold out for validation.")
args = parser.parse_args()

random.seed(args.seed)

_DIR         = os.path.join(os.path.expanduser("~"), ".config", "preflight")
TRIPLETS_PATH = os.path.join(_DIR, args.triplets_file)

print("=" * 64)
print("  Engram embedding fine-tune  v2  (hard-negative triplets)")
print("=" * 64)
print(f"  Base model     : {args.base_model}")
print(f"  Output         : {args.output}")
print(f"  Loss           : {args.loss}")
print(f"  Triplets file  : {TRIPLETS_PATH}")
print(f"  Max triplets   : {args.max_triplets or 'unlimited'}")
print(f"  Batch size     : {args.batch_size}")
print(f"  Epochs         : {args.epochs}")
print(f"  Max seq len    : {args.max_seq_len}")
print(f"  Margin (triplet): {args.margin}")
print()

# ── Load triplets ────────────────────────────────────────────────────────────
if not os.path.exists(TRIPLETS_PATH):
    print(f"✗ Triplets file not found: {TRIPLETS_PATH}")
    print("  Run mine_hard_negatives.py first.")
    sys.exit(1)

print("Loading triplets ...")
all_data: list[dict] = []
with open(TRIPLETS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_data.append(json.loads(line))

print(f"  Loaded: {len(all_data):,} examples")

with_neg  = sum(1 for d in all_data if d.get("has_hard_neg"))
plain     = len(all_data) - with_neg
print(f"  With hard negatives : {with_neg:,}")
print(f"  Plain pairs (no neg): {plain:,}")

# Category breakdown
by_cat: dict[str, int] = {}
for d in all_data:
    c = d.get("category", "unknown")
    by_cat[c] = by_cat.get(c, 0) + 1
print(f"  By category:")
for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"    {c:<22} {n:>5}")
print()

# ── Build InputExample lists ─────────────────────────────────────────────────
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer, InputExample, losses  # noqa: E402
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator   # noqa: E402
from torch.utils.data import DataLoader                                      # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU   : {props.name}  ({props.total_memory // 1024**2} MB)")
print()

# Split by sample_id to avoid leaking the same conversation into train and val
sample_ids = list({d["sample_id"] for d in all_data})
random.shuffle(sample_ids)
n_val = max(1, int(len(sample_ids) * args.val_frac))
val_sids = set(sample_ids[:n_val])
train_sids = set(sample_ids[n_val:])

train_data = [d for d in all_data if d["sample_id"] in train_sids]
val_data   = [d for d in all_data if d["sample_id"] in val_sids]

# Cap if requested
random.shuffle(train_data)
if args.max_triplets and len(train_data) > args.max_triplets:
    train_data = train_data[:args.max_triplets]

print(f"Split (by sample_id, {len(train_sids)} train / {len(val_sids)} val conversations):")
print(f"  Train: {len(train_data):,}  |  Val: {len(val_data):,}")
print()

# ── Build training examples ───────────────────────────────────────────────────
if args.loss == "triplet":
    # TripletLoss: (anchor, positive, negative)
    # For examples with multiple hard negatives, emit one triplet per negative.
    # For plain pairs (no hard neg), emit with a random in-batch negative placeholder
    # (the loss won't use it, but DataLoader needs consistent format — we skip them).
    train_examples: list[InputExample] = []
    for d in train_data:
        q  = d["query"]
        p  = d["positive"]
        negs = d.get("hard_negatives", [])
        if not negs:
            continue  # TripletLoss requires an explicit negative — skip plain pairs
        for neg in negs:
            train_examples.append(InputExample(texts=[q, p, neg]))

    print(f"TripletLoss training examples: {len(train_examples):,}")

    if len(train_examples) < 100:
        print("  WARNING: Very few triplet examples. Falling back to MNRL.")
        args.loss = "mnrl"

if args.loss == "mnrl":
    # MultipleNegativesRankingLoss: (anchor, positive) — in-batch negatives
    # Include all pairs regardless of whether they have hard negatives.
    train_examples = []
    for d in train_data:
        train_examples.append(InputExample(texts=[d["query"], d["positive"]]))
        # Also add hard negatives as reversed pairs with a different anchor
        # This increases the pool of in-batch negatives.
        for neg in d.get("hard_negatives", [])[:2]:
            # Use the negative as its own positive with a different query style
            # Not ideal, but adds negatives into the batch rotation.
            pass  # keep it simple — just (q, p) pairs for MNRL

    print(f"MNRL training examples: {len(train_examples):,}")

print()

if not train_examples:
    print("✗ No training examples built. Check the triplets file.")
    sys.exit(1)

# ── Build validation evaluator ───────────────────────────────────────────────
val_sample = val_data[:500]
if val_sample:
    neg_pool = [d["positive"] for d in val_data]
    random.shuffle(neg_pool)
    neg_pool = neg_pool[:len(val_sample)]

    eval_s1     = [d["query"]    for d in val_sample] + [d["query"]    for d in val_sample]
    eval_s2     = [d["positive"] for d in val_sample] + neg_pool[:len(val_sample)]
    eval_scores = [1.0] * len(val_sample)              + [0.0] * len(val_sample)

    evaluator = EmbeddingSimilarityEvaluator(
        eval_s1, eval_s2, eval_scores,
        name="engram-val",
        show_progress_bar=False,
    )
else:
    evaluator = None
    print("  [WARN] No validation data — running without evaluator.")

# ── Load base model ───────────────────────────────────────────────────────────
print(f"Loading base model: {args.base_model} ...")
model = SentenceTransformer(args.base_model, device=device)
model.max_seq_length = args.max_seq_len

# ── Build loss ────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)

if args.loss == "triplet":
    from sentence_transformers.losses import TripletLoss, TripletDistanceMetric  # noqa: E402
    loss_fn = TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    print(f"Loss: TripletLoss (margin={args.margin}, COSINE)")
else:
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    print("Loss: MultipleNegativesRankingLoss (in-batch negatives)")

warmup_steps = max(10, len(train_loader) // 10)
print(f"Training: {len(train_examples):,} examples, {args.epochs} epoch(s), "
      f"batch={args.batch_size}, warmup={warmup_steps}")
print()

# ── Train ─────────────────────────────────────────────────────────────────────
t_start = time.time()
model.fit(
    train_objectives=[(train_loader, loss_fn)],
    evaluator=evaluator,
    epochs=args.epochs,
    warmup_steps=warmup_steps,
    output_path=args.output,
    save_best_model=True,
    show_progress_bar=True,
)
elapsed = time.time() - t_start

print()
print(f"Training done in {elapsed/60:.1f} min")
print(f"Model saved to: {args.output}")

# ── Quick sanity check ───────────────────────────────────────────────────────
print()
print("Sanity check ...")
loaded   = SentenceTransformer(args.output)

# Use a LoCoMo-style query/fact pair rather than generic sentences
test_q    = "When did Caroline join the LGBTQ support group?"
test_pos  = "Caroline signed up for the LGBTQ support group at her university."
test_neg1 = "Melanie painted a sunrise last weekend and loved it."
test_neg2 = "I have a dog named Max and I love hiking."

vq  = loaded.encode(test_q,    normalize_embeddings=True)
vp  = loaded.encode(test_pos,  normalize_embeddings=True)
vn1 = loaded.encode(test_neg1, normalize_embeddings=True)
vn2 = loaded.encode(test_neg2, normalize_embeddings=True)

import numpy as np  # noqa: E402

sim_pos  = float(np.dot(vq, vp))
sim_neg1 = float(np.dot(vq, vn1))
sim_neg2 = float(np.dot(vq, vn2))

print(f"  sim(q, positive)      = {sim_pos:.4f}")
print(f"  sim(q, hard_neg)      = {sim_neg1:.4f}")
print(f"  sim(q, random_neg)    = {sim_neg2:.4f}")

ok_pos  = sim_pos  > sim_neg1
ok_rand = sim_pos  > sim_neg2
print(f"  pos > hard_neg?       {'✓ YES' if ok_pos  else '✗ NO  ← WARNING'}")
print(f"  pos > random_neg?     {'✓ YES' if ok_rand else '✗ NO  ← WARNING'}")

if not ok_pos:
    print()
    print("  WARNING: Positive similarity is NOT higher than hard negative.")
    print("  The model may need more training, a larger margin, or more data.")

print()
print("=" * 64)
print(f"  Next step: benchmark this model vs B.db / K20 baseline")
print(f"  Run:")
print(f"    $env:ENGRAM_EMBED_BACKEND='sentence-transformers'")
print(f"    $env:ENGRAM_EMBED_MODEL='{os.path.join(_DIR, args.output)}'")
print(f"    $env:PREFLIGHT_RRF_K='20'")
print(f"    python recall_ablation.py --reingest --db-letter F --tag engram_v2")
print()
print("  Accept if:")
print("    R@40  >= 93.08")
print("    R@5   >= 74.07")
print("    multi_hop R@5 >= 61.57")
print("    questions_with_evidence >= 1531")
print("=" * 64)
