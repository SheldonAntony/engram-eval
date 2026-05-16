#!/usr/bin/env python3
"""finetune_embed_v3.py — Fine-tune BGE-small on clean v3 triplets.

Key improvements over v2 (finetune_embed_v2.py):
  1. Reads positive_curr ("Speaker: text") — matches DB embedding format exactly.
  2. Reads negative_curr (list of [curr] lines) — matches DB embedding format.
  3. Reduced TripletLoss margin: 0.2 (was 0.5) — less aggressive push.
  4. Max 1 negative per query by default (was 5) — fewer bad-label examples.
  5. Validation uses positive_curr format (consistent with training).
  6. Sanity check uses "Speaker: text" format matching DB embeddings.
  7. Acceptance gate printed: must beat G.db (base_st_control) on all metrics.

Usage:
    python finetune_embed_v3.py
    python finetune_embed_v3.py --margin 0.25 --max-neg 2 --epochs 2
    python finetune_embed_v3.py --loss mnrl    # safer for uncertain data
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--base-model",    default="BAAI/bge-small-en-v1.5")
parser.add_argument("--output",        default="bge-small-engram-v3")
parser.add_argument("--triplets-file", default="engram_triplets_v3.jsonl")
parser.add_argument("--loss",          default="triplet", choices=["triplet", "mnrl"],
                    help="triplet: TripletLoss (explicit negatives required); "
                         "mnrl: MultipleNegativesRankingLoss (in-batch negatives).")
parser.add_argument("--margin",        type=float, default=0.2,
                    help="TripletLoss margin (default 0.2; was 0.5 in v2).")
parser.add_argument("--max-neg",       type=int,   default=1,
                    help="Max negative_curr entries to use per query (default 1; was 5).")
parser.add_argument("--batch-size",    type=int,   default=32)
parser.add_argument("--epochs",        type=int,   default=1)
parser.add_argument("--max-seq-len",   type=int,   default=128)
parser.add_argument("--val-frac",      type=float, default=0.10)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--max-triplets",  type=int,   default=0,
                    help="Cap training examples (0 = use all).")
args = parser.parse_args()
random.seed(args.seed)

_DIR          = os.path.join(os.path.expanduser("~"), ".config", "preflight")
TRIPLETS_PATH = os.path.join(_DIR, args.triplets_file)
OUTPUT_PATH   = os.path.join(_DIR, args.output)

print("=" * 66)
print("  Engram embedding fine-tune  v3  (clean triplets)")
print("=" * 66)
print(f"  Base model      : {args.base_model}")
print(f"  Output          : {OUTPUT_PATH}")
print(f"  Loss            : {args.loss}")
print(f"  Margin (triplet): {args.margin}  (was 0.5 in v2)")
print(f"  Max neg/query   : {args.max_neg}  (was 5 in v2)")
print(f"  Batch size      : {args.batch_size}")
print(f"  Epochs          : {args.epochs}")
print(f"  Max seq len     : {args.max_seq_len}")
print()

if not os.path.exists(TRIPLETS_PATH):
    print(f"✗ Triplets file not found: {TRIPLETS_PATH}")
    print("  Run extract_engram_pairs_v3.py then mine_hard_negatives_v3.py first.")
    sys.exit(1)

# ── Load triplets ─────────────────────────────────────────────────────────────

print(f"Loading: {TRIPLETS_PATH}")
all_data: list[dict] = []
with open(TRIPLETS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_data.append(json.loads(line))

with_neg = sum(1 for d in all_data if d.get("has_hard_neg"))
print(f"  Loaded           : {len(all_data):,} examples")
print(f"  With hard neg    : {with_neg:,}  ({100*with_neg/max(len(all_data),1):.1f}%)")
print(f"  Without hard neg : {len(all_data) - with_neg:,}")

by_cat: dict[str, int] = {}
for d in all_data:
    c = d.get("category", "unknown")
    by_cat[c] = by_cat.get(c, 0) + 1
print("  By category:")
for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"    {c:<24}  {n:>5}")
print()

# ── Imports ───────────────────────────────────────────────────────────────────

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU   : {props.name}  ({props.total_memory // 1024**2} MB)")
print()

# ── Split by sample_id (avoid leakage) ───────────────────────────────────────

sample_ids = sorted({d["sample_id"] for d in all_data})
random.shuffle(sample_ids)
n_val      = max(1, int(len(sample_ids) * args.val_frac))
val_sids   = set(sample_ids[:n_val])
train_sids = set(sample_ids[n_val:])

train_data = [d for d in all_data if d["sample_id"] in train_sids]
val_data   = [d for d in all_data if d["sample_id"] in val_sids]
random.shuffle(train_data)
if args.max_triplets and len(train_data) > args.max_triplets:
    train_data = train_data[:args.max_triplets]

print(f"Split ({len(train_sids)} train / {len(val_sids)} val conversations):")
print(f"  Train: {len(train_data):,}  |  Val: {len(val_data):,}")
print()

# ── Build training examples ───────────────────────────────────────────────────

train_examples: list[InputExample] = []

if args.loss == "triplet":
    for d in train_data:
        query    = d["query"]
        pos_text = d.get("positive_curr") or d.get("positive", "")  # Speaker: text
        negs     = d.get("negative_curr", [])                        # [curr] lines
        if not negs:
            continue   # TripletLoss requires explicit negative
        for neg in negs[:args.max_neg]:
            train_examples.append(InputExample(texts=[query, pos_text, neg]))
    print(f"TripletLoss training examples: {len(train_examples):,}")
    if len(train_examples) < 50:
        print("  ⚠ Very few triplet examples — auto-switching to MNRL.")
        args.loss = "mnrl"

if args.loss == "mnrl":
    train_examples = []
    for d in train_data:
        query    = d["query"]
        pos_text = d.get("positive_curr") or d.get("positive", "")
        train_examples.append(InputExample(texts=[query, pos_text]))
    print(f"MNRL training examples: {len(train_examples):,}")

print()
if not train_examples:
    print("✗ No training examples built. Check the triplets file.")
    sys.exit(1)

# ── Validation evaluator ──────────────────────────────────────────────────────

val_pool = [d for d in val_data if d.get("has_hard_neg")][:300]
if not val_pool:
    val_pool = val_data[:300]

if val_pool:
    pos_texts = [d.get("positive_curr") or d.get("positive", "") for d in val_pool]
    neg_texts = list(pos_texts)
    random.shuffle(neg_texts)

    # (query, positive, 1.0) + (query, shuffled_positive, 0.0)
    s1      = [d["query"] for d in val_pool] + [d["query"] for d in val_pool]
    s2      = pos_texts + neg_texts[:len(val_pool)]
    scores  = [1.0] * len(val_pool) + [0.0] * len(val_pool)

    evaluator = EmbeddingSimilarityEvaluator(
        s1, s2, scores,
        name="engram-val-v3",
        show_progress_bar=False,
    )
    print(f"Validation: {len(val_pool)} positive pairs  (positive_curr format)")
else:
    evaluator = None
    print("  [WARN] No validation data available.")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\nLoading base model: {args.base_model} ...")
model = SentenceTransformer(args.base_model, device=device)
model.max_seq_length = args.max_seq_len

# ── Loss function ─────────────────────────────────────────────────────────────

train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)

if args.loss == "triplet":
    from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
    loss_fn = TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    print(f"Loss: TripletLoss  margin={args.margin}  COSINE")
else:
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    print("Loss: MultipleNegativesRankingLoss (in-batch negatives)")

warmup_steps = max(10, len(train_loader) // 10)
print(f"Training: {len(train_examples):,} examples, {args.epochs} epoch(s), "
      f"batch={args.batch_size}, warmup={warmup_steps}\n")

# ── Train ─────────────────────────────────────────────────────────────────────

t_start = time.time()
model.fit(
    train_objectives=[(train_loader, loss_fn)],
    evaluator=evaluator,
    epochs=args.epochs,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_PATH,
    save_best_model=True,
    show_progress_bar=True,
)
elapsed = time.time() - t_start
print(f"\nTraining done in {elapsed/60:.1f} min")
print(f"Model saved to: {OUTPUT_PATH}")

# ── Sanity check (uses Speaker: text format — matches DB embedding) ───────────

print("\nSanity check (Speaker: text format matching DB embedding target) ...")
import numpy as np
loaded = SentenceTransformer(OUTPUT_PATH)

# Use format exactly matching memory.store_turn_window → embed_text(curr_line)
test_q    = "When did Caroline go biking with her friends?"
test_pos  = "Caroline: Hey Mel, long time no chat! I had a wicked day out with the gang last weekend — we went biking and saw some cool stuff."
test_hrd  = "Caroline: Thanks, I'm glad we caught up! We had a blast at the Pride fest last year too."
test_rnd  = "Melanie: I painted that lake sunrise last year! It's really special to me, I think about it a lot."

vq  = loaded.encode(test_q,   normalize_embeddings=True)
vp  = loaded.encode(test_pos, normalize_embeddings=True)
vhn = loaded.encode(test_hrd, normalize_embeddings=True)
vrn = loaded.encode(test_rnd, normalize_embeddings=True)

sim_pos = float(np.dot(vq, vp))
sim_hrd = float(np.dot(vq, vhn))
sim_rnd = float(np.dot(vq, vrn))

print(f"  sim(q, positive)         = {sim_pos:.4f}")
print(f"  sim(q, hard_neg_same_ctx)= {sim_hrd:.4f}")
print(f"  sim(q, random_neg)       = {sim_rnd:.4f}")

ok_hard = sim_pos > sim_hrd
ok_rand = sim_pos > sim_rnd
print(f"  pos > hard_neg?  {'✓ YES' if ok_hard else '✗ NO  ← WARNING'}")
print(f"  pos > rnd_neg?   {'✓ YES' if ok_rand else '✗ NO  ← WARNING'}")

if not ok_hard:
    print()
    print("  WARNING: Model did not push hard negative below positive.")
    print("  Options: increase margin, add epochs, or review triplets for leakage.")

# ── Benchmark instructions ────────────────────────────────────────────────────

print()
print("=" * 66)
print("  NEXT STEP: benchmark against G.db base_st_control")
print("=" * 66)
print("  Run (use DB letter H to avoid overwriting existing DBs):")
print()
print(f"    $env:ENGRAM_EMBED_BACKEND='sentence-transformers'")
print(f"    $env:ENGRAM_EMBED_MODEL='{OUTPUT_PATH}'")
print(f"    $env:PREFLIGHT_RRF_K='20'")
print(f"    $env:PREFLIGHT_BM25_WEIGHT='1.0'")
print(f"    python recall_ablation.py --reingest --db-letter H --tag engram_v3")
print()
print("  Acceptance gate (must beat G.db base_st_control on ALL three):")
print("    R@5  >= 75.10   (G.db target, frozen floor >= 74.07)")
print("    R@40 >= 93.96   (G.db target, frozen floor >= 93.08)")
print("    multi_hop R@5 >= 61.92   (frozen floor >= 61.57)  ← HARD GATE")
print()
print("  If multi_hop R@5 regresses below G.db: reject v3, keep G.db as prod.")
print("=" * 66)
