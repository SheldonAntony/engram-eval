#!/usr/bin/env python3
"""LongMemEval retrieval benchmark for the Preflight memory system.

Measures Session Recall@K and MRR@K on the oracle split (only evidence sessions
in the haystack — cleanest apples-to-apples retrieval test).  No LLM / OpenAI
key required; this evaluates the *retrieval* stage only.

Downloads the dataset automatically on first run (~8 MB).

Usage:
    python bench_longmemeval.py [--limit N] [--top-k K] [--split {oracle,s,m}]

    --limit N   Evaluate only first N non-abstention instances (default: all)
    --top-k K   Retrieve top-K sessions per question (default: 5)
    --split     oracle (default) | s (LongMemEval_S, ~40 filler sessions) |
                m (LongMemEval_M, ~500 filler sessions — very slow)

Run with the venv python so real fastembed is available:
    ~/.config/opencode/.venv/bin/python bench_longmemeval.py
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# ── Add opencode scripts dir so memory.py / utils.py / extractor.py load ──────
# Derive from __file__ so this works both on Windows native and under WSL
# (where Path.home() points to the Linux home, not the Windows one).
_PREFLIGHT_DIR = Path(__file__).resolve().parent          # …/.config/preflight
_SCRIPTS = _PREFLIGHT_DIR.parent / "opencode"             # …/.config/opencode
sys.path.insert(0, str(_SCRIPTS))

# ── Dataset URLs ───────────────────────────────────────────────────────────────
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
_SPLITS = {
    "oracle": "longmemeval_oracle.json",
    "s":      "longmemeval_s_cleaned.json",
    "m":      "longmemeval_m_cleaned.json",
}
_CACHE_DIR = _PREFLIGHT_DIR / ".longmemeval_cache"


def _download(split: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = _SPLITS[split]
    dest = _CACHE_DIR / fname
    if dest.exists():
        print(f"[cache] {dest}")
        return dest
    url = f"{_HF_BASE}/{fname}"
    print(f"Downloading LongMemEval {split} split (~8-50 MB)...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        print(f"\nERROR: download failed — {e}")
        print("You can manually download from:")
        print(f"  {url}")
        print(f"and place the file at: {dest}")
        sys.exit(1)
    print(f"Saved to {dest}\n")
    return dest


def _session_text(turns: list) -> str:
    """Flatten a list of turn dicts into a single string."""
    parts = []
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _recall_at_k(retrieved_sids: list, answer_sids: set, k: int) -> int:
    return int(bool(set(retrieved_sids[:k]) & answer_sids))


def _mrr(retrieved_sids: list, answer_sids: set) -> float:
    for rank, sid in enumerate(retrieved_sids, 1):
        if sid in answer_sids:
            return 1.0 / rank
    return 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval retrieval benchmark for Preflight memory"
    )
    parser.add_argument("--limit",  type=int, default=None,
                        help="Evaluate only first N non-abstention instances")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Retrieve top-K sessions (default: 5)")
    parser.add_argument("--split",  choices=["oracle", "s", "m"], default="oracle",
                        help="Dataset split (default: oracle)")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_path = _download(args.split)
    instances: list = json.loads(data_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(instances)} instances from {args.split} split.")

    # Skip abstention instances — they have no ground-truth evidence sessions.
    non_abs = [
        inst for inst in instances
        if not inst["question_id"].endswith("_abs")
        and inst.get("answer_session_ids")
    ]
    print(f"  Non-abstention instances with answers: {len(non_abs)}")

    if args.limit:
        non_abs = non_abs[: args.limit]
        print(f"  Limited to first {args.limit}.")

    # ── Import memory AFTER path is configured ────────────────────────────────
    import memory as mem  # noqa: PLC0415

    # Redirect to a throw-away DB so the production DB is untouched.
    _bench_db = tempfile.mktemp(suffix="_longmemeval.db")
    mem.DB_PATH = _bench_db
    print(f"\nBench DB: {_bench_db}")

    top_k = args.top_k
    ks_to_eval = sorted({k for k in [1, 3, 5] if k <= top_k} | {top_k})

    # ── Evaluation loop ───────────────────────────────────────────────────────
    print(f"\nRunning retrieval eval (top_k={top_k})...\n")
    records: list[dict] = []
    t_start = time.time()

    for i, inst in enumerate(non_abs):
        qid        = inst["question_id"]
        q_type     = inst["question_type"]
        question   = inst["question"]
        answer_sids = set(inst["answer_session_ids"])
        haystack_sids     = inst["haystack_session_ids"]
        haystack_sessions = inst["haystack_sessions"]

        # Use question_id as project_id — fully isolated in SQLite.
        project_id = f"lme_{qid}"

        # ── Indexing ──────────────────────────────────────────────────────────
        content_to_sid: dict[str, str] = {}
        for sid, turns in zip(haystack_sids, haystack_sessions):
            text = _session_text(turns)
            content_to_sid[text] = sid
            # session_id stored in DB = the dataset's session_id (for later lookup).
            mem.store_fact(project_id, str(sid), text, "note")

        # ── Retrieval ─────────────────────────────────────────────────────────
        retrieved_contents = mem.retrieve_facts(
            project_id,
            "bench_query",
            question,
            top_n=top_k,
            threshold=0.0,
            max_tokens=500_000,   # disable token budget during benchmarking
        )

        # Map returned content strings back to their dataset session_ids.
        retrieved_sids: list[str] = [
            content_to_sid[c] for c in retrieved_contents if c in content_to_sid
        ]

        # ── Per-instance metrics ──────────────────────────────────────────────
        rr = _mrr(retrieved_sids, answer_sids)
        recall_at = {k: _recall_at_k(retrieved_sids, answer_sids, k) for k in ks_to_eval}

        records.append({
            "qid":    qid,
            "type":   q_type,
            "rr":     rr,
            "recall": recall_at,
            "n_haystack": len(haystack_sids),
            "n_answer":   len(answer_sids),
            "n_retrieved": len(retrieved_sids),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(non_abs):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(non_abs) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1:>4}/{len(non_abs)}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"rate={rate:.1f}q/s")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    for suffix in ["", "-wal", "-shm"]:
        try:
            os.unlink(_bench_db + suffix)
        except OSError:
            pass

    # ── Aggregate results ─────────────────────────────────────────────────────
    n = len(records)
    if n == 0:
        print("No records — nothing to report.")
        return

    print(f"\n{'='*60}")
    print(f"  LongMemEval ({args.split} split) — Preflight Retrieval")
    print(f"  N={n}  top_k={top_k}  time={total_time:.0f}s ({total_time/n:.1f}s/q)")
    print(f"{'='*60}")

    # Overall
    print(f"\n  {'Metric':<22} {'Score':>7}")
    print(f"  {'-'*30}")
    for k in ks_to_eval:
        val = sum(r["recall"][k] for r in records) / n
        print(f"  {'Recall@'+str(k):<22} {val:>7.3f}")
    overall_mrr = sum(r["rr"] for r in records) / n
    print(f"  {'MRR@'+str(top_k):<22} {overall_mrr:>7.3f}")

    # Per question-type breakdown
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)

    print(f"\n  {'Question type':<32} {'N':>4}  {'R@1':>6}  {'R@3':>6}  {'MRR':>6}")
    print(f"  {'-'*54}")
    for q_type, recs in sorted(by_type.items()):
        nt  = len(recs)
        r1  = sum(r["recall"].get(1, 0) for r in recs) / nt
        r3  = sum(r["recall"].get(3, 0) for r in recs) / nt
        mrr = sum(r["rr"] for r in recs) / nt
        print(f"  {q_type:<32} {nt:>4}  {r1:>6.3f}  {r3:>6.3f}  {mrr:>6.3f}")

    # Published baselines from the LongMemEval paper (Table 2, session level,
    # oracle split — Recall@1 / MRR reported in the paper).
    # Source: Wu et al., 2024. https://arxiv.org/abs/2410.10813
    print(f"\n  Published baselines (paper, oracle split, session granularity):")
    print(f"  {'Method':<28} {'R@1':>6}  {'MRR':>6}")
    print(f"  {'-'*42}")
    print(f"  {'flat-BM25':<28} {'~0.52':>6}  {'~0.57':>6}")
    print(f"  {'flat-Contriever':<28} {'~0.60':>6}  {'~0.65':>6}")
    print(f"  {'flat-GTE-Qwen2-7B':<28} {'~0.78':>6}  {'~0.82':>6}")
    print(f"  (Note: exact numbers vary by subset; see paper Table 2)")
    print(f"{'='*60}")

    # Save full results to JSON for further analysis
    out_path = Path(__file__).parent / f"bench_results_{args.split}.json"
    out_path.write_text(json.dumps({
        "split":   args.split,
        "top_k":   top_k,
        "n":       n,
        "overall": {
            **{f"recall@{k}": sum(r["recall"][k] for r in records) / n for k in ks_to_eval},
            f"mrr@{top_k}": overall_mrr,
        },
        "by_type": {
            q_type: {
                "n":   len(recs),
                **{f"recall@{k}": sum(r["recall"].get(k, 0) for r in recs) / len(recs)
                   for k in ks_to_eval},
                f"mrr@{top_k}": sum(r["rr"] for r in recs) / len(recs),
            }
            for q_type, recs in by_type.items()
        },
        "instances": records,
    }, indent=2), encoding="utf-8")
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
