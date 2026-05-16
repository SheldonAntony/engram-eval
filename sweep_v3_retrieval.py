#!/usr/bin/env python3
"""sweep_v3_retrieval.py — Retrieval-only ablation sweep on existing H.db.

Runs recall_ablation.py with different env-var knobs via subprocess so every
run gets a fresh Python import (flags are read at module-import time).

No reingest. H.db (sentence-transformers bge-small-engram-v3) must exist.

Usage:
    python sweep_v3_retrieval.py
    python sweep_v3_retrieval.py --configs k15,bm125,derived   # subset

After all runs, prints a comparison table against G.db acceptance gates.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_MODEL = os.path.join(_DIR, "bge-small-engram-v3")

# ── Acceptance gates (must beat G.db on ALL three) ───────────────────────────
GATES = {
    "R@5":          75.10,
    "R@40":         93.96,
    "multi_hop_r5": 61.92,
}
GATED_KEYS = {"R@5": ("1","5"), "R@40": ("1","40"), "multi_hop_r5": None}

# ── G.db reference numbers ────────────────────────────────────────────────────
GDB = {
    "R@1":  48.82, "R@3":  67.81, "R@5":  75.10,
    "R@10": 83.90, "R@40": 93.96,
    "multi_hop": 61.92, "open_domain": 82.97,
    "temporal":  74.29, "single_hop":  45.45,
}

# ── v3 baseline reference (H.db K=20 BM25=1.0) ───────────────────────────────
V3_TAG = "engram_v3"

# ── Sweep configurations ──────────────────────────────────────────────────────
# Each entry: (tag, env-var overrides dict)
# Base env for ALL runs: sentence-transformers backend + v3 model.
# Only the retrieval knobs change.
CONFIGS: list[tuple[str, dict[str, str]]] = [
    # RRF K sweep
    ("v3_k15",   {"PREFLIGHT_RRF_K": "15"}),
    ("v3_k25",   {"PREFLIGHT_RRF_K": "25"}),
    ("v3_k30",   {"PREFLIGHT_RRF_K": "30"}),
    ("v3_k40",   {"PREFLIGHT_RRF_K": "40"}),
    ("v3_k50",   {"PREFLIGHT_RRF_K": "50"}),
    # BM25 weight sweep (K=20)
    ("v3_bm075", {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_BM25_WEIGHT": "0.75"}),
    ("v3_bm150", {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_BM25_WEIGHT": "1.50"}),
    ("v3_bm200", {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_BM25_WEIGHT": "2.00"}),
    # Derived BM25 (WordNet expansion already in H.db facts_derived_fts)
    ("v3_derived",     {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_USE_DERIVED_BM25": "1"}),
    ("v3_derived_k15", {"PREFLIGHT_RRF_K": "15", "PREFLIGHT_USE_DERIVED_BM25": "1"}),
    # Stopwords
    ("v3_sw",    {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_USE_STOPWORDS": "1"}),
    # Combinations of best candidates (will add based on early results)
    ("v3_bm150_k15", {"PREFLIGHT_RRF_K": "15", "PREFLIGHT_BM25_WEIGHT": "1.50"}),
    ("v3_bm200_k15", {"PREFLIGHT_RRF_K": "15", "PREFLIGHT_BM25_WEIGHT": "2.00"}),
    ("v3_derived_bm150", {"PREFLIGHT_RRF_K": "20", "PREFLIGHT_BM25_WEIGHT": "1.50", "PREFLIGHT_USE_DERIVED_BM25": "1"}),
]

parser = argparse.ArgumentParser()
parser.add_argument("--configs", default="", help="Comma-separated subset of tags to run (default: all).")
parser.add_argument("--dry-run", action="store_true", help="Print configs without running.")
args = parser.parse_args()

if args.configs:
    allowed = set(args.configs.split(","))
    configs = [(t, e) for t, e in CONFIGS if t in allowed]
else:
    configs = CONFIGS

# ── Helper ────────────────────────────────────────────────────────────────────

def _load_result(tag: str) -> dict | None:
    path = os.path.join(_DIR, f"locomo_recall_{tag}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return None


def _print_gates(tag: str, result: dict):
    rk  = result.get("recall_at_k", {})
    cat = result.get("recall_at_5_by_category", {})
    r5  = float(rk.get("5", 0))
    r40 = float(rk.get("40", 0))
    mh5 = float(cat.get("multi_hop", 0))
    od5 = float(cat.get("open_domain", 0))
    tmp = float(cat.get("temporal", 0))
    sh5 = float(cat.get("single_hop", 0))

    g5  = "PASS" if r5  >= GATES["R@5"]          else "FAIL"
    g40 = "PASS" if r40 >= GATES["R@40"]          else "FAIL"
    gmh = "PASS" if mh5 >= GATES["multi_hop_r5"]  else "FAIL"
    all_pass = g5 == g40 == gmh == "PASS"

    print(f"\n  [{tag}]")
    print(f"    R@5  : {r5:6.2f}  (G.db: {GDB['R@5']:.2f})  [{g5}]")
    print(f"    R@40 : {r40:6.2f}  (G.db: {GDB['R@40']:.2f})  [{g40}]")
    print(f"    MH@5 : {mh5:6.2f}  (G.db: {GDB['multi_hop']:.2f})  [{gmh}]")
    print(f"    OD@5 : {od5:6.2f}  (G.db: {GDB['open_domain']:.2f})")
    print(f"    TMP@5: {tmp:6.2f}  (G.db: {GDB['temporal']:.2f})")
    print(f"    SH@5 : {sh5:6.2f}  (G.db: {GDB['single_hop']:.2f})")
    if all_pass:
        print(f"  *** ALL GATES PASSED — CANDIDATE FOR PROMOTION ***")
    return all_pass


# ── Base env for all H.db runs ────────────────────────────────────────────────
BASE_ENV = {
    **os.environ,
    "ENGRAM_EMBED_BACKEND": "sentence-transformers",
    "ENGRAM_EMBED_MODEL":   _MODEL,
    "PREFLIGHT_RRF_K":      "20",       # default matching v3 baseline
    "PREFLIGHT_BM25_WEIGHT": "1.0",
    "PREFLIGHT_USE_STOPWORDS":    "0",
    "PREFLIGHT_USE_DERIVED_BM25": "0",
    "PREFLIGHT_USE_CE":           "0",
    "PREFLIGHT_SPEAKER_BOOST":    "0",
}

# ── Print plan ────────────────────────────────────────────────────────────────
print("=" * 66)
print("  v3 Retrieval Sweep  —  H.db (no reingest)")
print("=" * 66)
print(f"  DB          : locomo_eval_H.db")
print(f"  Model       : bge-small-engram-v3  (sentence-transformers)")
print(f"  Configs     : {len(configs)}")
print(f"  Gates       : R@5>={GATES['R@5']}  R@40>={GATES['R@40']}  MH@5>={GATES['multi_hop_r5']}")
print()

# Load v3 baseline for reference
v3_base = _load_result(V3_TAG)
if v3_base:
    print("  v3 baseline (K=20, BM25=1.0):")
    _print_gates(V3_TAG, v3_base)
print()
print(f"  Running {len(configs)} configurations ...")
print("-" * 66)

if args.dry_run:
    for tag, overrides in configs:
        env_str = "  ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"  {tag:<22} {env_str}")
    sys.exit(0)

# ── Run sweep ─────────────────────────────────────────────────────────────────
passed_all: list[str] = []
results: list[tuple[str, dict]] = []
t_total = time.time()

for i, (tag, overrides) in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Running: {tag}  {overrides}")

    # Check if already done (skip if result file is newer than H.db)
    out_path = os.path.join(_DIR, f"locomo_recall_{tag}.json")
    hdb_path = os.path.join(_DIR, "locomo_eval_H.db")
    if os.path.exists(out_path):
        # Already have result — check its metadata to see if it used the same DB+model
        try:
            existing = json.load(open(out_path, encoding="utf-8"))
            abl = existing.get("_ablation", {})
            if (abl.get("db_letter") == "H"
                    and abl.get("embed_backend") == "sentence-transformers"
                    and not abl.get("reingest", True)):
                print(f"  → Skipping: result already exists for H.db/{tag}")
                results.append((tag, existing))
                if _print_gates(tag, existing):
                    passed_all.append(tag)
                continue
        except Exception:
            pass

    # Build env
    run_env = {**BASE_ENV, **overrides}

    t0 = time.time()
    cmd = [
        sys.executable,
        os.path.join(_DIR, "recall_ablation.py"),
        "--db-letter", "H",
        "--tag", tag,
    ]
    proc = subprocess.run(cmd, env=run_env, cwd=_DIR)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ✗ FAILED (exit {proc.returncode})  {elapsed:.0f}s")
        continue

    result = _load_result(tag)
    if result:
        results.append((tag, result))
        passed = _print_gates(tag, result)
        if passed:
            passed_all.append(tag)
    else:
        print(f"  ✗ Result file not found after run")

print(f"\n{'='*66}")
print(f"  SWEEP COMPLETE  ({time.time()-t_total:.0f}s total)")
print(f"{'='*66}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'TAG':<24} {'R@5':>7} {'R@40':>7} {'MH@5':>7} {'OD@5':>7} {'TMP@5':>7}  GATES")
print("-" * 74)

# v3 baseline first
if v3_base:
    rk  = v3_base.get("recall_at_k", {})
    cat = v3_base.get("recall_at_5_by_category", {})
    print(f"  {'engram_v3 (baseline)':<22} "
          f"{float(rk.get('5',0)):7.2f} "
          f"{float(rk.get('40',0)):7.2f} "
          f"{float(cat.get('multi_hop',0)):7.2f} "
          f"{float(cat.get('open_domain',0)):7.2f} "
          f"{float(cat.get('temporal',0)):7.2f}  "
          f"{'ALL PASS' if float(rk.get('5',0))>=GATES['R@5'] and float(rk.get('40',0))>=GATES['R@40'] else 'FAIL'}")

for tag, result in sorted(results, key=lambda x: -float(x[1].get("recall_at_k", {}).get("5", 0))):
    rk  = result.get("recall_at_k", {})
    cat = result.get("recall_at_5_by_category", {})
    r5  = float(rk.get("5", 0))
    r40 = float(rk.get("40", 0))
    mh5 = float(cat.get("multi_hop", 0))
    od5 = float(cat.get("open_domain", 0))
    tmp = float(cat.get("temporal", 0))
    g5  = r5  >= GATES["R@5"]
    g40 = r40 >= GATES["R@40"]
    gmh = mh5 >= GATES["multi_hop_r5"]
    gate_str = "ALL PASS" if (g5 and g40 and gmh) else (
        " ".join(filter(None, [
            "" if g5 else "R@5-FAIL",
            "" if g40 else "R@40-FAIL",
            "" if gmh else "MH-FAIL",
        ]))
    )
    mark = " ★" if (g5 and g40 and gmh) else ""
    print(f"  {tag:<24} {r5:7.2f} {r40:7.2f} {mh5:7.2f} {od5:7.2f} {tmp:7.2f}  {gate_str}{mark}")

print(f"\n  G.db reference:    "
      f"{GDB['R@5']:7.2f} {GDB['R@40']:7.2f} {GDB['multi_hop']:7.2f} "
      f"{GDB['open_domain']:7.2f} {GDB['temporal']:7.2f}")

if passed_all:
    print(f"\n  ★ Configs passing all gates: {', '.join(passed_all)}")
else:
    print(f"\n  No config passed all gates in this sweep.")
    print(f"  Next step: proceed to dual-model fusion script or v4a training.")
