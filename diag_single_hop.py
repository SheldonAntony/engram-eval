#!/usr/bin/env python3
"""Diagnose single-hop recall failures in the v12 pipeline.

Single-hop R@5 is stuck at 56.18% across all versions.
This script classifies each failure by where the gold fact was lost.

Usage:
    python diag_single_hop.py [--tag v12_gbm21feat]

Output:
    - Summary table of failure categories
    - Per-question detail for each failure
    - Recommendation for next step
"""

import argparse
import json
import os
import sys

_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_SCRIPTS_DIR = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="v12_gbm21feat",
                    help="Result tag to analyze (default: v12_gbm21feat)")
args = parser.parse_args()

RESULT_PATH = os.path.join(_PREFLIGHT_DIR, f"locomo_recall_{args.tag}.json")
with open(RESULT_PATH) as f:
    data = json.load(f)

per_q = data.get("per_question", [])
single_hop = [q for q in per_q if q.get("category") == "single_hop"]
with_ev = [q for q in single_hop if q.get("has_evidence")]
no_ev = [q for q in single_hop if not q.get("has_evidence")]
fails_r5 = [q for q in with_ev if not q.get("hit@5")]

# Classify
pool_miss = []
deep_rerank = []
close_rerank = []
for q in fails_r5:
    rrf = q.get("gold_rrf_rank_best")
    if rrf is None or rrf >= 200:
        pool_miss.append(q)
    elif rrf >= 40:
        deep_rerank.append(q)
    else:
        close_rerank.append(q)

# Summary
print("=" * 70)
print("  SINGLE-HOP RECALL FAILURE DIAGNOSTIC")
print(f"  Tag: {args.tag}")
print("=" * 70)
print(f"\n  Single-hop total:            {len(single_hop)}")
print(f"  With evidence:               {len(with_ev)}")
print(f"  No evidence (unmappable):    {len(no_ev)}")
print(f"  Failures at R@5:             {len(fails_r5)}")
print()
print(f"  Categories:")
print(f"    Pool miss (rrf >= 200):    {len(pool_miss)}")
print(f"    Deep rerank (rrf 40-199):  {len(deep_rerank)}")
print(f"    Close rerank (rrf < 40):   {len(close_rerank)}")
print()

# Detailed: each failure
if fails_r5:
    print("─" * 70)
    print("  PER-QUESTION FAILURES")
    print("─" * 70)
    for q in sorted(fails_r5, key=lambda x: x.get("gold_rrf_rank_best", 9999)):
        rrf = q.get("gold_rrf_rank_best", "?")
        cos = q.get("gold_cos_rank_best", "?")
        if rrf is None or rrf == "?" or rrf >= 200:
            cat = "POOL_MISS"
        elif rrf >= 40:
            cat = "DEEP_RERANK"
        else:
            cat = "CLOSE_RERANK"
        print(f"\n  [{cat}] RRF={rrf} Cos={cos}")
        print(f"    Q: {q['question'][:100]}")
        print(f"    Ev: {q.get('evidence', [])}")

print()
print("─" * 70)
print("  INTERPRETATION")
print("─" * 70)
if pool_miss:
    print(f"\n  {len(pool_miss)} pool misses — gold fact never reached top-200 RRF pool.")
    print("  Fix: Need a dedicated single-hop lexical or embedding channel.")
if deep_rerank:
    print(f"\n  {len(deep_rerank)} deep rerank failures — gold fact in pool (rrf 40-199)")
    print("  but pushed out of top-5 by GBM + CE + coverage guards.")
    print("  Fix: GBM or CE actively scoring this fact below distractors.")
    print("  Try: Adding single_hop feature flag to GBM, or CE training data.")
if close_rerank:
    print(f"\n  {len(close_rerank)} close rerank failures — gold fact was in RRF top-40")
    print("  but demoted below rank 5 by the full reranking pipeline.")
    print("  Fix: Coverage guard or GBM alpha tuning may help.")
    print("  These are the easiest to fix — reranker is overriding a strong RRF signal.")

print(f"\n  Overall: {len(fails_r5)} single-hop failures at R@5.")
if deep_rerank or close_rerank:
    print("  Since most have rrf_rank < 200, this is NOT a pool ceiling problem.")
    print("  The reranker (GBM + CE) is demoting correct single-hop evidence.")
    print("  Recommended: Add a GBM feature for single-hop question type.")
