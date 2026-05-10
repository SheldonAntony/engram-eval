#!/usr/bin/env python3
"""Re-run only the F1 evaluation on the existing Mode B DB (no re-ingestion)."""
import sys, os, json, time

_SCRIPTS_DIR   = os.path.join(os.path.expanduser("~"), ".config", "opencode")
_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

# Must set up utils before importing eval_locomo
try:
    import utils as _u; _u.embed_text("test")
except Exception:
    import hashlib, types as _t
    _s = _t.ModuleType("utils")
    def _e(text): h=hashlib.sha256(text.encode()).digest(); v=[b/255.0 for b in h[:32]]; n=sum(x*x for x in v)**0.5; return [x/n for x in v] if n else v
    def _c(a,b): d=sum(x*y for x,y in zip(a,b)); na=sum(x*x for x in a)**0.5; nb=sum(x*x for x in b)**0.5; return d/(na*nb) if na and nb else 0.0
    _s.embed_text=_e; _s.cosine_similarity=_c; sys.modules["utils"]=_s

import eval_locomo as ev

db_path = os.path.join(_PREFLIGHT_DIR, "locomo_eval_B.db")
if not os.path.exists(db_path):
    print(f"ERROR: DB not found at {db_path}")
    print("Run eval_locomo.py first to ingest data.")
    sys.exit(1)

print("Loading dataset...")
samples = ev.download_dataset()
print(f"  {len(samples)} conversations loaded")

print(f"\nRunning F1 evaluation on existing DB: {db_path}")
t0 = time.time()
results = ev.evaluate(samples, None, db_path=db_path)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

print("\n" + "="*60)
print("  F1 RESULTS (extract_answer fix)")
print("="*60)
print(f"Overall F1    : {results['overall_f1']}%")
for cat, val in results['by_category'].items():
    print(f"  {cat:<14}: {val}%")

print("\n--- Sample predictions ---")
for q in results['per_question'][:15]:
    print(f"Q: {q['question']}")
    print(f"GT: {q['ground_truth']}")
    print(f"PRED: {q['prediction']}")
    print(f"F1: {q['f1']}")
    print("---")
