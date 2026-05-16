#!/usr/bin/env python3
"""
Smoke fine-tune: BAAI/bge-small-en-v1.5 on PersonaChat (filtered 20k pairs).

Usage:
    python finetune_embed.py                         # smoke: 20k pairs, 1 epoch
    python finetune_embed.py --max-pairs 0 --epochs 2   # full PersonaChat, 2 epochs
    python finetune_embed.py --output bge-small-engram-v1

Output: saved SentenceTransformer model directory ready to use via
    ENGRAM_EMBED_MODEL=<output> ENGRAM_EMBED_BACKEND=sentence-transformers
"""

import argparse
import random
import time

# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
parser.add_argument("--output", default="bge-small-engram-v1")
parser.add_argument("--max-pairs", type=int, default=20_000,
                    help="Cap on training pairs (0 = no cap, use full dataset)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--max-seq-len", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

print("=" * 60)
print("  Engram embedding fine-tune")
print("=" * 60)
print(f"  Base model     : {args.base_model}")
print(f"  Output         : {args.output}")
print(f"  Max pairs      : {args.max_pairs or 'unlimited'}")
print(f"  Batch size     : {args.batch_size}")
print(f"  Epochs         : {args.epochs}")
print(f"  Max seq len    : {args.max_seq_len}")
print()

# ── Load PersonaChat ─────────────────────────────────────────────────────────

print("Loading PersonaChat ...")
from datasets import load_dataset  # noqa: E402

pc = load_dataset("bavard/personachat_truecased", split="train", trust_remote_code=True)
print(f"  Loaded {len(pc):,} rows")

# ── Build filtered (anchor, positive) pairs ──────────────────────────────────
#
# Strategy:
#   anchor   = last utterance in conversation history (the "question" side)
#   positive = one persona fact for the speaker
#
# Filters applied:
#   1. Skip anchors shorter than 8 chars (generic filler turns)
#   2. Skip anchors that are pure back-channel ("ok", "yeah", "i see", etc.)
#   3. Skip persona facts shorter than 10 chars
#   4. Only emit at most 3 pairs per row to prevent a single conv dominating

BACKCHANNELS = {
    "ok", "okay", "yeah", "yes", "no", "nope", "sure", "right",
    "wow", "cool", "great", "nice", "i see", "ah", "oh", "hm",
    "hmm", "interesting", "really", "of course", "definitely",
    "i know", "me too", "same here", "for sure", "exactly",
}

def is_backchannel(text: str) -> bool:
    t = text.strip().lower().rstrip("!.?")
    return t in BACKCHANNELS or len(t.split()) <= 1


def build_pairs(row) -> list[tuple[str, str]]:
    history = row.get("history", [])
    personas = row.get("personality", [])
    if not history or not personas:
        return []

    anchor = history[-1].strip()
    if len(anchor) < 8 or is_backchannel(anchor):
        return []

    pairs = []
    shuffled = personas[:]
    random.shuffle(shuffled)
    for fact in shuffled[:3]:          # max 3 pairs per row
        fact = fact.strip()
        if len(fact) < 10:
            continue
        pairs.append((anchor, fact))
    return pairs


print("Building training pairs ...")
t0 = time.time()
all_pairs: list[tuple[str, str]] = []
for row in pc:
    all_pairs.extend(build_pairs(row))

print(f"  Raw pairs: {len(all_pairs):,}  ({time.time()-t0:.1f}s)")

# Deduplicate
seen = set()
deduped = []
for a, p in all_pairs:
    key = (a[:80], p[:80])
    if key not in seen:
        seen.add(key)
        deduped.append((a, p))

print(f"  After dedup: {len(deduped):,}")

# Shuffle + cap
random.shuffle(deduped)
if args.max_pairs and len(deduped) > args.max_pairs:
    deduped = deduped[:args.max_pairs]
    print(f"  Capped to:   {len(deduped):,}")

# Train/val split (90/10), split by index to avoid leaking adjacent rows
split_idx = int(len(deduped) * 0.9)
train_pairs = deduped[:split_idx]
val_pairs   = deduped[split_idx:]
print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")
print()

# ── SentenceTransformer training ─────────────────────────────────────────────

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer, InputExample, losses  # noqa: E402
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:    {props.name}  ({props.total_memory // 1024**2} MB)")
print()

model = SentenceTransformer(args.base_model, device=device)
model.max_seq_length = args.max_seq_len

train_examples = [InputExample(texts=[a, p]) for a, p in train_pairs]
train_loader   = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
train_loss     = losses.MultipleNegativesRankingLoss(model)

# Validation evaluator: use a subset (up to 1k) of val pairs as similarity pairs
# We treat each (anchor, positive) as a pair with score=1.0 (positive)
# plus a random negative with score=0.0 to give the evaluator contrast
val_sample = val_pairs[:1000]
neg_positives = [p for _, p in random.sample(val_pairs, len(val_sample))]
random.shuffle(neg_positives)

eval_sentences1 = [a for a, _ in val_sample] + [a for a, _ in val_sample]
eval_sentences2 = [p for _, p in val_sample] + neg_positives
eval_scores     = [1.0] * len(val_sample)   + [0.0] * len(val_sample)

evaluator = EmbeddingSimilarityEvaluator(
    eval_sentences1, eval_sentences2, eval_scores,
    name="personachat-val",
    show_progress_bar=False,
)

warmup_steps = max(10, len(train_loader) // 10)

print(f"Training: {len(train_examples):,} pairs, {args.epochs} epoch(s), "
      f"batch={args.batch_size}, warmup={warmup_steps}")
print()

t_start = time.time()
model.fit(
    train_objectives=[(train_loader, train_loss)],
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
loaded = SentenceTransformer(args.output)
test_q    = "Where did you go to school?"
test_fact = "I studied computer science at MIT."
test_neg  = "I love hiking on weekends."

vq = loaded.encode(test_q,    normalize_embeddings=True)
vf = loaded.encode(test_fact, normalize_embeddings=True)
vn = loaded.encode(test_neg,  normalize_embeddings=True)

sim_pos = float(vq @ vf)
sim_neg = float(vq @ vn)
dim     = len(vq)

print(f"  Embedding dim         : {dim}")
print(f"  sim(question, fact)   : {sim_pos:.4f}  (should be higher)")
print(f"  sim(question, negfact): {sim_neg:.4f}  (should be lower)")
print()
if sim_pos > sim_neg:
    print("  PASS: model correctly ranks positive fact above negative")
else:
    print("  WARN: model ranks negative above positive on this example -- inspect quality")

print()
print("Done. To use with Engram benchmark:")
print(f"  $env:ENGRAM_EMBED_MODEL='{args.output}'")
print(f"  $env:ENGRAM_EMBED_BACKEND='sentence-transformers'")
print(f"  python recall_ablation.py --reingest --db-letter E --tag ft_embed_smoke")
