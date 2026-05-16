#!/usr/bin/env python3
"""audit_triplets.py — Quantify false negatives in engram_triplets.jsonl.

Reports:
  - Rate of negatives whose [curr] line overlaps the gold positive
  - Rate of negatives where ANY window line overlaps the gold positive
  - Embedding-text mismatch analysis (positive format vs DB embedding format)
  - Vague-positive rate
  - Examples by category

Usage:
    python audit_triplets.py
    python audit_triplets.py --threshold 0.30 --show-examples 10
"""
from __future__ import annotations
import argparse
import json
import os
import re

_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")

parser = argparse.ArgumentParser()
parser.add_argument("--input",         default="engram_triplets.jsonl")
parser.add_argument("--threshold",     type=float, default=0.30,
                    help="Token-recall threshold to flag a [curr]-line negative as suspicious.")
parser.add_argument("--adj-threshold", type=float, default=0.45,
                    help="Token-recall threshold to flag any adjacent window line as suspicious.")
parser.add_argument("--show-examples", type=int,   default=8)
args = parser.parse_args()

TRIPLETS_PATH = os.path.join(_DIR, args.input)

# ── helpers ──────────────────────────────────────────────────────────────────

def _norm_tokens(text: str) -> set[str]:
    t = text.lower()
    t = re.sub(r'\[(prev|curr|next)\]\s*', '', t)
    t = re.sub(r'^\w[\w ]*:\s*', '', t, flags=re.M)
    t = re.sub(r'[^\w\s]', ' ', t)
    return {w for w in t.split() if len(w) > 2}


def _extract_curr(window: str) -> str:
    """Extract the [curr] turn line, stripping the tag prefix."""
    for line in window.split("\n"):
        if line.startswith("[curr] "):
            return line[len("[curr] "):]
    return ""  # not a windowed fact


def _extract_all_lines(window: str) -> list[str]:
    """Return all [prev]/[curr]/[next] lines stripped of their tags."""
    lines = []
    for line in window.split("\n"):
        for tag in ("[prev] ", "[curr] ", "[next] "):
            if line.startswith(tag):
                lines.append(line[len(tag):])
                break
    return lines


def _token_recall(gold: str, candidate: str) -> float:
    """Fraction of gold tokens found in candidate."""
    g = _norm_tokens(gold)
    c = _norm_tokens(candidate)
    if not g:
        return 0.0
    return len(g & c) / len(g)


def _any_line_recall(positive: str, window: str, threshold: float) -> tuple[bool, float]:
    """Return (flag, best_recall) over all window lines."""
    best = 0.0
    for line in _extract_all_lines(window):
        r = _token_recall(positive, line)
        if r > best:
            best = r
    return best >= threshold, best


# ── load ─────────────────────────────────────────────────────────────────────

print(f"Loading: {TRIPLETS_PATH}")
triplets = []
with open(TRIPLETS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            triplets.append(json.loads(line))

print(f"  {len(triplets):,} triplets loaded\n")

# ── vague positive detection ──────────────────────────────────────────────────

VAGUE_PHRASES = {
    "take a look", "here's", "here is", "by the way",
    "check this out", "check it out", "look at this",
    "here you go", "i sent", "sending", "attached",
}

def _is_vague(text: str) -> bool:
    if len(text.strip()) < 40:
        return True
    t = text.lower()
    if any(p in t for p in VAGUE_PHRASES) and len(text.strip()) < 80:
        return True
    return False

# ── analysis ─────────────────────────────────────────────────────────────────

total_negs    = 0
curr_flagged  = 0   # [curr] line overlaps positive >= threshold
adj_flagged   = 0   # any window line overlaps positive >= adj_threshold
vague_pos     = 0

by_cat_total:     dict[str, int] = {}
by_cat_curr_flag: dict[str, int] = {}

flagged_examples: list[dict] = []

for t in triplets:
    cat   = t.get("category", "unknown")
    query = t["query"]
    pos   = t["positive"]
    negs  = t.get("hard_negatives", [])

    by_cat_total[cat] = by_cat_total.get(cat, 0) + 1

    if _is_vague(pos):
        vague_pos += 1

    triplet_curr_flagged = False
    ex_flagged_negs = []

    for neg in negs:
        total_negs += 1
        curr_line  = _extract_curr(neg)
        curr_rec   = _token_recall(pos, curr_line) if curr_line else 0.0
        adj_flag, adj_best = _any_line_recall(pos, neg, args.adj_threshold)

        if curr_rec >= args.threshold:
            curr_flagged += 1
            triplet_curr_flagged = True
            ex_flagged_negs.append({
                "curr": curr_line[:120],
                "curr_recall": round(curr_rec, 3),
                "adj_best":    round(adj_best, 3),
            })
        if adj_flag:
            adj_flagged += 1

    if triplet_curr_flagged:
        by_cat_curr_flag[cat] = by_cat_curr_flag.get(cat, 0) + 1
        if len(flagged_examples) < args.show_examples:
            flagged_examples.append({
                "query":    query,
                "positive": pos,
                "category": cat,
                "vague":    _is_vague(pos),
                "flagged_negs": ex_flagged_negs[:3],
            })

# ── print results ─────────────────────────────────────────────────────────────

W = 65
print("=" * W)
print("  TRIPLET AUDIT RESULTS")
print("=" * W)
print(f"  Total triplets          : {len(triplets):,}")
print(f"  Total negatives         : {total_negs:,}")
print(f"  Avg negatives/triplet   : {total_negs/max(len(triplets),1):.1f}")
print(f"  Vague positives         : {vague_pos:,}  ({100*vague_pos/max(len(triplets),1):.1f}%)")
print()
print(f"  ── FALSE-NEGATIVE RATE ──")
print(f"  [curr] overlap >= {args.threshold:.2f}  (Level-2 filter):")
print(f"    Negatives flagged   : {curr_flagged:,}  ({100*curr_flagged/max(total_negs,1):.1f}% of all negs)")
print(f"    Triplets affected   : {sum(by_cat_curr_flag.values()):,}  ({100*sum(by_cat_curr_flag.values())/max(len(triplets),1):.1f}% of triplets)")
print()
print(f"  Any-line overlap >= {args.adj_threshold:.2f}  (Level-3 filter / adjacent gold):")
print(f"    Negatives flagged   : {adj_flagged:,}  ({100*adj_flagged/max(total_negs,1):.1f}% of all negs)")
print()
print(f"  By category  (triplets with ≥1 flagged [curr]-line negative):")
print(f"    {'CATEGORY':<22}  {'FLAGGED':>8}  {'TOTAL':>8}  {'RATE':>7}")
print("    " + "─" * 48)
for cat in sorted(by_cat_total):
    nf = by_cat_curr_flag.get(cat, 0)
    nt = by_cat_total[cat]
    print(f"    {cat:<22}  {nf:>8}  {nt:>8}  {100*nf/max(nt,1):>6.1f}%")

if flagged_examples:
    print()
    print(f"  EXAMPLE FALSE-NEGATIVE TRIPLETS (threshold={args.threshold})")
    print("  " + "─" * 62)
    for ex in flagged_examples:
        print(f"\n  [{ex['category']}]  vague_pos={ex['vague']}")
        print(f"  Q  : {ex['query'][:90]}")
        print(f"  Pos: {ex['positive'][:90]}")
        for i, fn in enumerate(ex["flagged_negs"]):
            print(f"  Neg[curr]{i+1}: {fn['curr'][:90]}")
            print(f"    curr_recall={fn['curr_recall']:.3f}  adj_best={fn['adj_best']:.3f}")

print()
print("=" * W)
print("  EMBEDDING TEXT MISMATCH ANALYSIS")
print("=" * W)
print("  What the DB embeds at inference time:")
print("    embed_text('Speaker: text')     ← curr_line with speaker prefix")
print()
print("  What v2 triplets train on:")
print("    positive : raw turn text        ← NO speaker prefix")
print("    negative : full 3-turn window   ← [prev][curr][next] with tags")
print()
print("  Mismatches:")
print("    Positive: missing speaker prefix → minor, consistent shift")
print("    Negative: full window instead of [curr] line → MAJOR mismatch")
print("      - Window embeds 3 speakers' text concatenated")
print("      - Model at inference time only ever embeds one curr_line")
print("      - Training signal is computed on a distribution the retriever")
print("        never encounters → guaranteed embedding space corruption")
print()
print("  Fix:")
print("    positive_curr = 'Speaker: text'    (matches DB embedding)")
print("    negative_curr = extracted [curr] line   (matches DB embedding)")
print("=" * W)
