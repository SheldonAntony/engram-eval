#!/usr/bin/env python3
"""Freeze and compare ALL locomo recall benchmark results.

Reads every locomo_recall_*.json in the preflight dir, prints a comparison
table sorted by R@40, and writes baseline_frozen.json with the canonical
numbers we must beat with the new fine-tuned model.

Usage:
    python baseline_summary.py
"""

import json
import os

_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")

# ── Collect all result files ─────────────────────────────────────────────────
results = []

for fname in sorted(os.listdir(_DIR)):
    if not (fname.startswith("locomo_recall_") and fname.endswith(".json")):
        continue
    tag = fname[len("locomo_recall_"):-len(".json")]
    path = os.path.join(_DIR, fname)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Could not read {fname}: {exc}")
        continue

    rk  = data.get("recall_at_k", {})
    cat = data.get("recall_at_5_by_category", {})
    abl = data.get("_ablation", {})

    def _k(key):
        return rk.get(str(key), rk.get(key, 0.0)) or 0.0

    results.append({
        "tag":     tag,
        "R@1":     _k(1),
        "R@3":     _k(3),
        "R@5":     _k(5),
        "R@10":    _k(10),
        "R@40":    _k(40),
        "sh_r5":   cat.get("single_hop",  0.0),
        "mh_r5":   cat.get("multi_hop",   0.0),
        "tmp_r5":  cat.get("temporal",    0.0),
        "od_r5":   cat.get("open_domain", 0.0),
        "n_ev":    data.get("questions_with_evidence", 0),
        "n_tot":   data.get("questions_total", 0),
        "backend": abl.get("embed_backend", "fastembed"),
        "model":   abl.get("embed_model",   "default"),
        "db":      abl.get("db_letter",     "B"),
        "reingest": abl.get("reingest",     False),
        "elapsed_s": abl.get("elapsed_s",  0),
    })

results.sort(key=lambda x: x["R@40"], reverse=True)

# ── Print table ──────────────────────────────────────────────────────────────
SEP = "─" * 136
HDR = (f"{'TAG':<28} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'R@40':>6} │ "
       f"{'SH':>6} {'MH':>6} {'TMP':>6} {'OD':>6} │ "
       f"{'N_EV':>5} {'DB':>2} {'REINGEST':>8} {'BACKEND':<24} MODEL")

print(f"\n{SEP}")
print(HDR)
print(SEP)

best = results[0] if results else None

for r in results:
    marker = " ◄ BEST" if r is best else ""
    print(
        f"{r['tag']:<28} {r['R@1']:>6.2f} {r['R@3']:>6.2f} {r['R@5']:>6.2f} "
        f"{r['R@10']:>6.2f} {r['R@40']:>6.2f} │ "
        f"{r['sh_r5']:>6.2f} {r['mh_r5']:>6.2f} {r['tmp_r5']:>6.2f} {r['od_r5']:>6.2f} │ "
        f"{r['n_ev']:>5} {r['db']:>2} {'Y' if r['reingest'] else 'N':>8} "
        f"{r['backend']:<24} {r['model']}{marker}"
    )

print(SEP)

# ── Deltas relative to fastembed baseline ────────────────────────────────────
base = next((r for r in results if r["tag"] == "baseline"), None)
if base:
    print(f"\nDeltas vs baseline (tag=baseline):")
    print(f"  {'TAG':<28} {'ΔR@1':>8} {'ΔR@3':>8} {'ΔR@5':>8} {'ΔR@40':>8}")
    print("  " + "─" * 60)
    for r in results:
        if r["tag"] == "baseline":
            continue
        print(f"  {r['tag']:<28} "
              f"{r['R@1'] - base['R@1']:>+8.2f} "
              f"{r['R@3'] - base['R@3']:>+8.2f} "
              f"{r['R@5'] - base['R@5']:>+8.2f} "
              f"{r['R@40'] - base['R@40']:>+8.2f}")

# ── Frozen canonical targets ─────────────────────────────────────────────────
TARGET = {
    "R@5":  74.07,
    "R@40": 93.08,
    "source": "rrf_k20  (B.db fastembed, RRF K=20, no reingest)",
    "note":   "New fine-tuned model MUST beat or match both to be accepted",
}

frozen = {
    "frozen_at": "2026-05-11",
    "target_to_beat": TARGET,
    "category_targets": {
        "multi_hop":   61.57,
        "temporal":    74.38,
        "single_hop":  46.07,
        "open_domain": 81.09,
        "note":        "All R@5 by category from rrf_k20 (best run). Fine-tuned model must not regress MH.",
    },
    "acceptance_criteria": {
        "R@40":        ">= 93.08",
        "R@5":         ">= 74.07",
        "multi_hop_r5": ">= 61.57  (E.db failed here: 58.01 — primary regression)",
        "n_ev":        ">= 1531   (E.db dropped to 1522 — evidence count must not decrease)",
    },
    "all_runs": results,
}

frozen_path = os.path.join(_DIR, "baseline_frozen.json")
with open(frozen_path, "w", encoding="utf-8") as f:
    json.dump(frozen, f, indent=2)

print(f"\n✓ Targets to beat:")
print(f"  R@5  >= {TARGET['R@5']}   (from {TARGET['source']})")
print(f"  R@40 >= {TARGET['R@40']}   (from {TARGET['source']})")
print(f"  multi_hop R@5 >= 61.57  (E.db failed: PersonaChat damaged multi-hop)")
print(f"  questions_with_evidence >= 1531  (E.db had 1522 — evidence count must not drop)")
print(f"\n✓ Frozen to: {frozen_path}")
