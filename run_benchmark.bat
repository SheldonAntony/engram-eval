@echo off
set ENGRAM_EMBED_BACKEND=sentence-transformers
set ENGRAM_EMBED_MODEL=C:\Users\Sheldon Antony\.config\preflight\bge-small-engram-v3
set PREFLIGHT_USE_LLM_EXTRACTOR=1
set PREFLIGHT_LLM_WORKERS=4
set PREFLIGHT_RRF_K=15
set PREFLIGHT_USE_DERIVED_BM25=1
cd /d "C:\Users\Sheldon Antony\.config\preflight"
python recall_ablation.py --reingest --db-letter I --tag v3_llm_atomic_derived_k15 > benchmark_run.log 2>&1
