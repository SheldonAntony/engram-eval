#!/usr/bin/env python3
"""Category-level recall failure analysis.

Reads any locomo_recall_*.json result file and produces a detailed breakdown
of which questions failed and why — used to decide where the next training
data investment should go.

Usage:
    python analyze_category_failures.py                          # compare best vs E
    python analyze_category_failures.py --a rrf_k20 --b engram_v2   # compare two runs
    python analyze_category_failures.py --a rrf_k20 --show-fails    # list failing Qs
"""

from __future__ import annotations
import argparse
import json
import os

_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")

parser = argparse.ArgumentParser()
parser.add_argument("--a", default="rrf_k20",       help="Tag for run A (baseline to compare).")
parser.add_argument("--b", default="ft_embed_smoke", help="Tag for run B (new model).")
parser.add_argument("--show-fails", action="store_true",
                    help="Print questions where A hit but B missed at @5.")
parser.add_argument("--k", type=int, default=5, help="K for hit@K analysis.")
args = parser.parse_args()


# ── Load helper ──────────────────────────────────────────────────────────────

def _load(tag: str) -> dict | None:
    path = os.path.join(_DIR, f"locomo_recall_{tag}.json")
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return None
    return json.load(open(path, encoding="utf-8"))


def _hit_k(q: dict, k: int) -> bool:
    return bool(q.get(f"hit@{k}"))


def _gold_rank(q: dict) -> int:
    return int(q.get("gold_rrf_rank_best") or 9999)


# ── Category stats helper ─────────────────────────────────────────────────────

def category_stats(questions: list[dict], k: int) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        if cat not in stats:
            stats[cat] = {"total": 0, "hit": 0, "rank_sum": 0, "no_ev": 0}
        s = stats[cat]
        s["total"] += 1
        if not q.get("has_evidence"):
            s["no_ev"] += 1
            continue
        if _hit_k(q, k):
            s["hit"] += 1
        s["rank_sum"] += _gold_rank(q)
    return stats


def print_cat_table(stats: dict[str, dict], k: int, label: str):
    print(f"\n  [{label}]  Recall@{k} by category:")
    print(f"    {'CATEGORY':<16} {'HIT':>6} {'TOTAL':>6} {'R@%d'%k:>8} {'AVG_RANK':>10}")
    print("    " + "─" * 52)
    cats = sorted(stats.keys())
    for cat in cats:
        s     = stats[cat]
        total = s["total"] - s["no_ev"]   # only countable questions
        if total == 0:
            continue
        pct      = 100.0 * s["hit"] / total
        avg_rank = s["rank_sum"] / total if total else 0
        print(f"    {cat:<16} {s['hit']:>6} {total:>6} {pct:>8.2f}  {avg_rank:>10.1f}")


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nComparing:  A = {args.a}   B = {args.b}   (k={args.k})")
print("─" * 60)

data_a = _load(args.a)
data_b = _load(args.b)

if data_a is None:
    print("Cannot load run A. Aborting.")
    raise SystemExit(1)

qs_a = data_a.get("per_question", [])
print(f"\nRun A  ({args.a}):  {len(qs_a)} questions")
stats_a = category_stats(qs_a, args.k)
print_cat_table(stats_a, args.k, args.a)
rk_a = data_a.get("recall_at_k", {})
print(f"  Overall R@{args.k}: {rk_a.get(str(args.k), rk_a.get(args.k, 0)):.2f}")

if data_b is not None:
    qs_b = data_b.get("per_question", [])
    print(f"\nRun B  ({args.b}):  {len(qs_b)} questions")
    stats_b = category_stats(qs_b, args.k)
    print_cat_table(stats_b, args.k, args.b)
    rk_b = data_b.get("recall_at_k", {})
    print(f"  Overall R@{args.k}: {rk_b.get(str(args.k), rk_b.get(args.k, 0)):.2f}")

    # ── Delta table ──────────────────────────────────────────────────────────
    print(f"\nDelta  (B - A) per category  @ k={args.k}:")
    print(f"  {'CATEGORY':<16} {'Δ HIT':>8} {'Δ R@%d'%args.k:>8}  NOTE")
    print("  " + "─" * 52)
    for cat in sorted(set(stats_a) | set(stats_b)):
        sa = stats_a.get(cat, {"hit": 0, "total": 0, "no_ev": 0})
        sb = stats_b.get(cat, {"hit": 0, "total": 0, "no_ev": 0})
        total_a = sa["total"] - sa["no_ev"]
        total_b = sb["total"] - sb["no_ev"]
        pct_a = 100.0 * sa["hit"] / total_a if total_a else 0
        pct_b = 100.0 * sb["hit"] / total_b if total_b else 0
        d_hit = sb["hit"] - sa["hit"]
        d_pct = pct_b - pct_a
        note  = "REGRESSION" if d_pct < -1.0 else ("IMPROVE" if d_pct > 1.0 else "~same")
        sign  = "+" if d_hit >= 0 else ""
        print(f"  {cat:<16} {sign}{d_hit:>7} {d_pct:>+8.2f}%  {note}")

    # ── Regression analysis: questions A hit but B missed ───────────────────
    if args.show_fails:
        # Build question → hit_k map for both runs
        hit_a = {q["question"]: _hit_k(q, args.k) for q in qs_a if q.get("has_evidence")}
        hit_b = {q["question"]: _hit_k(q, args.k) for q in qs_b if q.get("has_evidence")}

        regressions = [
            q for q in qs_a
            if q.get("has_evidence")
            and _hit_k(q, args.k)
            and not hit_b.get(q["question"], True)
        ]
        improvements = [
            q for q in qs_b
            if q.get("has_evidence")
            and _hit_k(q, args.k)
            and not hit_a.get(q["question"], True)
        ]

        print(f"\n── REGRESSIONS (A hit@{args.k} but B missed) ─────────────────")
        if regressions:
            for q in regressions:
                cat  = q.get("category", "?")
                rank = _gold_rank(q)
                print(f"  [{cat:>12}]  rank={rank:>4}  {q['question'][:90]}")
        else:
            print("  None.")

        print(f"\n── IMPROVEMENTS (B hit@{args.k} but A missed) ────────────────")
        if improvements:
            for q in improvements:
                cat  = q.get("category", "?")
                print(f"  [{cat:>12}]  {q['question'][:90]}")
        else:
            print("  None.")

# ── Training recommendation ───────────────────────────────────────────────────
print("\n── TRAINING RECOMMENDATIONS ─────────────────────────────────────")
if data_b is not None:
    cats_by_delta = []
    for cat in sorted(set(stats_a) | set(stats_b)):
        sa = stats_a.get(cat, {"hit": 0, "total": 0, "no_ev": 0})
        sb = stats_b.get(cat, {"hit": 0, "total": 0, "no_ev": 0})
        ta = sa["total"] - sa["no_ev"]
        tb = sb["total"] - sb["no_ev"]
        pa = 100.0 * sa["hit"] / ta if ta else 0
        pb = 100.0 * sb["hit"] / tb if tb else 0
        cats_by_delta.append((cat, pb - pa, sb["hit"] - sa["hit"]))
    cats_by_delta.sort(key=lambda x: x[1])

    print(f"  Categories most in need of more training data (B worst vs A):")
    for cat, delta, d_hit in cats_by_delta:
        if delta < 0:
            print(f"    {cat:<16}  {delta:>+.2f}%  ({d_hit:+d} questions) ← add training data")
        elif delta < 1.0:
            print(f"    {cat:<16}  {delta:>+.2f}%  (~same)")
else:
    print("  Load both runs (--a and --b) for recommendations.")

print()
