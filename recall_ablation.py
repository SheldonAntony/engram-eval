#!/usr/bin/env python3
"""Fast Recall@K ablation runner.

Reuses the existing locomo_eval_B.db — no re-ingestion (saves ~780 seconds).
Runs run_recall_eval() only with the flags below, saves a tagged result JSON,
and prints a comparison table against the known baseline.

Usage:
    python recall_ablation.py                          # baseline (all flags off)
    python recall_ablation.py --tag stopwords          # + manual label

Override experiment flags via environment variables (one at a time):
    $env:PREFLIGHT_USE_STOPWORDS="1"; python recall_ablation.py
    $env:PREFLIGHT_BM25_WEIGHT="0.75"; python recall_ablation.py
    $env:PREFLIGHT_BM25_WEIGHT="0.5";  python recall_ablation.py
    $env:PREFLIGHT_USE_CE="1";         python recall_ablation.py
    $env:PREFLIGHT_SPEAKER_BOOST="1";  python recall_ablation.py

LLM extractor ablation (requires Ollama + qwen2.5:1.5b):
    $env:PREFLIGHT_USE_LLM_EXTRACTOR="1"; python recall_ablation.py --reingest --db-letter C

Results are saved to: locomo_recall_<tag>.json
The standard locomo_recall_results.json is also updated (always = latest run).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_SCRIPTS_DIR   = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

DATA_CACHE = os.path.join(_PREFLIGHT_DIR, "locomo10.json")

# Known-good baseline for comparison
_BASELINE = {
    "R@1": 47.35, "R@3": 65.90, "R@5": 73.87,
    "R@10": 81.78, "R@40": 92.62,
}

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Recall@K ablation — no ingestion step.")
parser.add_argument("--tag", default="", help="Label for output file suffix.")
parser.add_argument("--reingest", action="store_true",
                    help="Re-ingest all conversations into a new DB before eval. "
                         "Required when testing storage-side changes (e.g. LLM extractor).")
parser.add_argument("--db-letter", default="B", metavar="LETTER",
                    help="DB suffix letter (default: B). Use C, D, … for LLM extractor runs.")
args = parser.parse_args()

DB_PATH = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{args.db_letter}.db")

# Auto-generate tag from active flags if not specified manually
_sw  = os.environ.get("PREFLIGHT_USE_STOPWORDS",     "0")
_bw  = os.environ.get("PREFLIGHT_BM25_WEIGHT",       "1.0")
_ce  = os.environ.get("PREFLIGHT_USE_CE",             "0")
_sp  = os.environ.get("PREFLIGHT_SPEAKER_BOOST",     "0")
_llm = os.environ.get("PREFLIGHT_USE_LLM_EXTRACTOR", "0")
_rk  = os.environ.get("PREFLIGHT_RRF_K",             "60")

if args.tag:
    tag = args.tag
else:
    tag = f"sw{_sw}_bm{_bw.replace('.','p')}_ce{_ce}_sp{_sp}_llm{_llm}_k{_rk}"

out_path = os.path.join(_PREFLIGHT_DIR, f"locomo_recall_{tag}.json")

# ── Import eval_locomo AFTER env vars are in place (flags read at import) ─────
import eval_locomo as ev  # noqa: E402

# ── Header ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  RECALL@K ABLATION")
print("=" * 60)
print(f"  Tag                    : {tag}")
print(f"  _USE_BM25_STOPWORDS    : {ev._USE_BM25_STOPWORDS}")
print(f"  _BM25_RRF_WEIGHT       : {ev._BM25_RRF_WEIGHT}")
print(f"  _USE_CE_IN_RECALL_EVAL : {ev._USE_CE_IN_RECALL_EVAL}")
print(f"  _USE_EVAL_SPEAKER_BOOST: {ev._USE_EVAL_SPEAKER_BOOST}")
print(f"  _RRF_K                 : {ev._RRF_K}")
print(f"  _USE_DERIVED_BM25      : {ev._USE_DERIVED_BM25}")
print(f"  _POOL_A_SIZE           : {ev._POOL_A_SIZE}")
print(f"  LLM extractor          : {_llm == '1'}")
print(f"  LLM workers            : {os.environ.get('PREFLIGHT_LLM_WORKERS', '4')}")
print(f"  store_turn             : False (benchmark mode)")
print(f"  Reingest               : {args.reingest}")
print(f"  DB path                : {DB_PATH}")
print(f"  DB exists              : {os.path.exists(DB_PATH)}")
print()

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading dataset from {DATA_CACHE} ...")
samples = json.load(open(DATA_CACHE, encoding="utf-8"))
if isinstance(samples, dict):
    samples = list(samples.values())
print(f"  {len(samples)} conversations loaded.\n")

# ── Optional re-ingestion ─────────────────────────────────────────────────────
if args.reingest:
    import memory as _mem  # noqa: E402
    print(f"Re-ingesting into {DB_PATH} ...")
    _mem.DB_PATH = DB_PATH
    _mem._compacted_this_process = False
    # Remove stale DB and WAL files
    for _suffix in ("", "-wal", "-shm"):
        try:
            os.remove(DB_PATH + _suffix)
        except OSError:
            pass
    _t_ingest = time.time()
    _stats = ev.ingest(samples, _mem, mode="B")
    print(f"  Done in {time.time()-_t_ingest:.1f}s  "
          f"turns={_stats['total_turns']}  kw-facts={_stats['kw_facts']}")
    # Report llm_atomic counts if any
    import sqlite3 as _sq
    _con = _sq.connect(DB_PATH)
    _llm_cnt = _con.execute(
        "SELECT COUNT(*) FROM facts WHERE fact_type='llm_atomic'"
    ).fetchone()[0]
    _con.close()
    print(f"  llm_atomic facts stored: {_llm_cnt}\n")
elif not os.path.exists(DB_PATH):
    print(f"ERROR: {DB_PATH} not found.")
    print("Run 'python eval_locomo.py' first, or use --reingest to create a fresh DB.")
    sys.exit(1)

# ── Run recall eval ────────────────────────────────────────────────────────────
t0 = time.time()
result = ev.run_recall_eval(samples, DB_PATH)
elapsed = time.time() - t0

# ── Save tagged results ────────────────────────────────────────────────────────
result["_ablation"] = {
    "tag": tag,
    "USE_BM25_STOPWORDS":        ev._USE_BM25_STOPWORDS,
    "BM25_RRF_WEIGHT":           ev._BM25_RRF_WEIGHT,
    "USE_CE_IN_RECALL_EVAL":     ev._USE_CE_IN_RECALL_EVAL,
    "USE_EVAL_SPEAKER_BOOST":    ev._USE_EVAL_SPEAKER_BOOST,
    "USE_LLM_EXTRACTOR":         _llm == "1",
    "reingest":                  args.reingest,
    "db_letter":                 args.db_letter,
    "elapsed_s":                 round(elapsed, 1),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

# ── Comparison table ───────────────────────────────────────────────────────────
rk = result.get("recall_at_k", {})
print(f"\nElapsed: {elapsed:.1f}s")
print(f"Tagged result  -> {out_path}")
print()
print(f"{'Metric':<8} {'Baseline':>10} {'Now':>10} {'Delta':>9}  {'Status'}")
print("-" * 54)
for k, bk in [("1","R@1"),("3","R@3"),("5","R@5"),("10","R@10"),("40","R@40")]:
    now = rk.get(k, 0.0)
    bl  = _BASELINE[bk]
    delta = now - bl
    status = "OK" if delta >= -0.05 else ("WARN" if delta >= -0.5 else "REGRESS")
    print(f"  R@{k:<4} {bl:>10.2f} {now:>10.2f} {delta:>+9.2f}  {status}")

cat5 = result.get("recall_at_5_by_category", {})
if cat5:
    print()
    print("  R@5 by category:")
    labels = {"single_hop": "Single-hop", "multi_hop": "Multi-hop",
              "temporal": "Temporal", "open_domain": "Open-domain"}
    for k, lbl in labels.items():
        v = cat5.get(k)
        if v is not None:
            print(f"    {lbl:<14}: {v:.2f}%")
