#!/usr/bin/env python3
"""Grid-search RRF signal weights + composite scoring weights on LoCoMo data.

Usage:
    # Quick RRF weight search (30 questions)
    python tune_weights.py --rrf-only --n 30

    # Full composite weight search (slower)
    python tune_weights.py --full --n 30

Results saved to: ~/.config/preflight/tune_results.json
"""

from __future__ import annotations
import json, os, sys, time, sqlite3
from pathlib import Path

# Ensure user site-packages is on the path (for fastembed).
_user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if str(_user_site) not in sys.path:
    sys.path.insert(0, str(_user_site))

_PREFLIGHT = Path.home() / ".config" / "preflight"
_SCRIPTS   = Path.home() / ".config" / "opencode"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_PREFLIGHT))

import memory as mem
mem.DB_PATH = str(_PREFLIGHT / "locomo_eval_B.db")

import eval_locomo as ev

# Warm up: call retrieve_facts once to load all models (embedding, CE, NER).
# This ~30s cost is paid once, not per query.
print("Warming up models (first call loads embedding + CE + NER)...", flush=True)
t0 = time.time()
mem.retrieve_facts("locomo_0", "", "What is this conversation about?", top_n=5, threshold=0.0)
print(f"  Warm-up done in {time.time()-t0:.1f}s", flush=True)

# ── Grids ───────────────────────────────────────────────────────────────────
RRF_TEST_COMBOS = [
    {"vec": 1.0, "bm25": 1.0, "entity": 1.0},  # baseline (equal weights)
    {"vec": 1.0, "bm25": 1.2, "entity": 0.8},  # boost BM25, reduce entity
    {"vec": 1.2, "bm25": 1.0, "entity": 0.8},  # boost dense, reduce entity
    {"vec": 1.5, "bm25": 0.8, "entity": 0.5},  # aggressive dense
    {"vec": 1.0, "bm25": 1.5, "entity": 0.5},  # aggressive BM25
]
COMPOSITE_TEST_COMBOS = [
    {"w_rrf": 0.60, "w_recency": 0.12, "w_staleness": 0.08, "w_session_rec": 0.12, "w_freq": 0.08},  # default
    {"w_rrf": 0.50, "w_recency": 0.15, "w_staleness": 0.10, "w_session_rec": 0.15, "w_freq": 0.10},  # lower RRF
    {"w_rrf": 0.70, "w_recency": 0.09, "w_staleness": 0.06, "w_session_rec": 0.09, "w_freq": 0.06},  # higher RRF
    {"w_rrf": 0.65, "w_recency": 0.10, "w_staleness": 0.08, "w_session_rec": 0.10, "w_freq": 0.07},  # slight RRF boost
    {"w_rrf": 0.55, "w_recency": 0.14, "w_staleness": 0.09, "w_session_rec": 0.14, "w_freq": 0.08},  # slight RRF reduction
]


def recall_at_k(ranked_fids, gold_fids, k):
    return int(bool(set(ranked_fids[:k]) & set(gold_fids)))

def load_questions(samples, max_n=30):
    """Yield (pid, question, gold_fids) for up to max_n scorable QAs total."""
    conn = sqlite3.connect(mem.DB_PATH)
    seen = 0
    for ci, conv in enumerate(samples):
        sid_str = str(conv.get("sample_id", ci))
        pid = f"locomo_{sid_str}"
        qa_list = list(ev.iter_qa(conv))
        if not qa_list:
            continue
        rows = conn.execute(
            "SELECT id, content FROM facts WHERE project_id = ? AND superseded_at IS NULL",
            (pid,),
        ).fetchall()
        content_to_ids: dict[str, set] = {}
        for fid, content in rows:
            for tag in ("[prev] ", "[next] "):
                for line in content.split("\n"):
                    if line.startswith(tag):
                        content_to_ids.setdefault(line[len(tag):], set()).add(fid)
            for tag in ("[curr] ",):
                for line in content.split("\n"):
                    if line.startswith(tag):
                        content_to_ids.setdefault(line[len(tag):], set()).add(fid)
        for qa in qa_list:
            gold_fids: set[int] = set()
            for d in qa.get("evidence", []):
                for _sn, _date, turns in ev.iter_sessions(conv.get("conversation", {})):
                    for turn in turns:
                        if turn.get("dia_id") == d:
                            content = f"{turn.get('speaker','?')}: {turn.get('text','')}"
                            gold_fids.update(content_to_ids.get(content, set()))
            if gold_fids:
                yield pid, qa["question"], gold_fids
                seen += 1
                if seen >= max_n:
                    conn.close()
                    return
    conn.close()

def evaluate(questions, w_rrf=None, w_recency=None, w_staleness=None,
             w_session_rec=None, w_freq=None, rrf_weights: dict | None = None,
             threshold=0.0):
    """Set weights on the live memory module, run queries, return recall."""
    if w_rrf is not None:
        mem._W_COMP_RRF = w_rrf
        mem._W_COMP_RECENCY = w_recency or 0.12
        mem._W_COMP_STALENESS = w_staleness or 0.08
        mem._W_COMP_SESSION_REC = w_session_rec or 0.12
        mem._W_COMP_FREQ = w_freq or 0.08
    if rrf_weights is not None:
        mem._RRF_W.clear()
        mem._RRF_W.update(rrf_weights)
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    for pid, question, gold_fids in questions:
        result = mem.retrieve_facts(pid, "", question, top_n=40, threshold=threshold, include_budget_info=True)
        ranked = result.get("all_ranked_fids", [])
        for k in [1, 3, 5]:
            hits[k] += recall_at_k(ranked, gold_fids, k)
        total += 1
    n = max(total, 1)
    return {k: v / n for k, v in hits.items()}, total

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--n", type=int, default=30, help="Max questions per conv")
    parser.add_argument("--fast", action="store_true", help="Skip CE reranking for speed (threshold=0.3)")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    samples = json.loads((_PREFLIGHT / "locomo10.json").read_text())
    questions = list(load_questions(samples, max_n=args.n))
    print(f"  {len(questions)} questions loaded", flush=True)

    eval_kw = dict(threshold=0.3) if args.fast else dict(threshold=0.0)

    best_r3 = 0.0
    best_config = {}
    results: list[dict] = []

    def save_intermediate():
        out = _PREFLIGHT / "tune_results.json"
        out.write_text(json.dumps({"best": best_config, "all": results}, indent=2))

    if args.rrf_only or args.full:
        print(f"Testing {len(RRF_TEST_COMBOS)} RRF weight combos...", flush=True)
        for i, rrf_w in enumerate(RRF_TEST_COMBOS):
            t1 = time.time()
            rec, n = evaluate(questions, rrf_weights=rrf_w, **eval_kw)
            dt = time.time() - t1
            r3 = rec[3]
            results.append({"type": "rrf", "weights": rrf_w, **rec, "n": n})
            dt_per_q = dt / max(n, 1)
            tag = "BEST" if r3 > best_r3 else ""
            if r3 > best_r3:
                best_r3 = r3
                best_config = {"type": "rrf", "weights": rrf_w, **rec}
            save_intermediate()
            print(f"  [{i+1}/{len(RRF_TEST_COMBOS)}] RRF={rrf_w}  R@3={r3:.4f}  [{dt:.0f}s, {dt_per_q:.1f}s/q]  {tag}", flush=True)

    if args.full:
        print(f"Testing {len(COMPOSITE_TEST_COMBOS)} composite weight combos...", flush=True)
        restore_rrf = mem._RRF_W.copy()
        restore_composite = {k: getattr(mem, k) for k in
                            ["_W_COMP_RRF", "_W_COMP_RECENCY", "_W_COMP_STALENESS", "_W_COMP_SESSION_REC", "_W_COMP_FREQ"]}
        for i, cw in enumerate(COMPOSITE_TEST_COMBOS):
            # Reset RRF weights to default for composite tests
            mem._RRF_W.clear()
            mem._RRF_W.update({"vec": 1.0, "bm25": 1.0, "entity": 1.0})
            t1 = time.time()
            rec, n = evaluate(questions, **cw)
            dt = time.time() - t1
            r3 = rec[3]
            results.append({"type": "composite", **cw, **rec, "n": n})
            tag = "BEST" if r3 > best_r3 else ""
            if r3 > best_r3:
                best_r3 = r3
                best_config = {"type": "composite", **cw, **rec}
            save_intermediate()
            print(f"  [{i+1}/{len(COMPOSITE_TEST_COMBOS)}] COMP={cw}  R@3={r3:.4f}  [{dt:.0f}s]  {tag}", flush=True)

    out = _PREFLIGHT / "tune_results.json"
    out.write_text(json.dumps({"best": best_config, "all": results}, indent=2))
    print(f"\nDone. Best R@3: {best_r3:.4f}")
    print(f"Config: {json.dumps(best_config, indent=2)}")
    print(f"Results -> {out}")

if __name__ == "__main__":
    main()
