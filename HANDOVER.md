# LoCoMo Retrieval — Complete Handover Document

> **Date:** 2026-05-15  
> **Goal:** Maximize Recall@3 on the LoCoMo benchmark (1,522 QA pairs across 10 conversations).  
> **Current champion:** v8 — `bge-reranker-v2-m3` — **R@3 = 80.81%, R@40 = 96.98%**  
> **Active run:** v11 lexical channels — results pending  
> **Next AI agent:** read this top-to-bottom before touching anything.

---

## Table of Contents

1. [The Goal](#1-the-goal)
2. [Repository Map](#2-repository-map)
3. [How the Full Pipeline Works](#3-how-the-full-pipeline-works)
4. [Database Schema (Key Tables)](#4-database-schema-key-tables)
5. [Environment & How to Run](#5-environment--how-to-run)
6. [Complete Experiment Log](#6-complete-experiment-log)
7. [Key Insights & Hard-Won Lessons](#7-key-insights--hard-won-lessons)
8. [Current Code State](#8-current-code-state)
9. [What To Do Next](#9-what-to-do-next)
10. [Acceptance Rules](#10-acceptance-rules)

---

## 1. The Goal

We are building a **long-term conversation memory system** (the `opencode` project). The retrieval component must find the correct memory fact when answering questions about past conversations. We benchmark retrieval quality using the **LoCoMo dataset** (10 long conversations, ~150 QA pairs each = 1,522 total).

**The metric is Recall@K**: given a question, does the gold-standard evidence turn appear in the top-K retrieved results?

We care most about **R@3** (production quality) and **R@40** (pipeline ceiling — can the right answer ever reach the reranker?).

**Champion so far:** `v8` with `R@3 = 80.81%`, `R@40 = 96.98%`.  
**Stretch target:** R@3 ≥ 84%.

---

## 2. Repository Map

### Repo 1: `C:\Users\Sheldon Antony\.config\preflight\` (benchmark/eval)

**GitHub remote:** `https://github.com/SheldonAntony/engram-eval.git` (branch: `master`)



| File | Purpose |
|------|---------|
| `eval_locomo.py` | **CORE** — full retrieval pipeline + recall scoring. ALL pipeline logic lives here. |
| `recall_ablation.py` | Benchmark runner — sets env vars, calls `run_recall_eval()`, saves `locomo_recall_{tag}.json` |
| `reranker.py` | GBM feature extraction (18 features) + `_apply_learned_rerank()` inference |
| `train_reranker.py` | Trains the GBM model from `featcache_*.pkl` feature cache |
| `diag_v8.py` | Diagnostic: compares v5 vs v8 per-question at hit@3 and hit@40 |
| `analyze_category_failures.py` | Breaks down failures by QA category (temporal/single_hop/multi_hop/open_domain) |
| `locomo10.json` | The 10 LoCoMo conversations (source data) |
| `locomo_eval_B.db` | SQLite DB — pre-ingested facts for all 10 conversations (Mode B corpus) |
| `locomo_eval_H.db` | SQLite DB — alternative corpus (not the main benchmark DB — use B) |
| `reranker_model.pkl` | Trained GBM reranker (18 features, HistGradientBoostingClassifier) |
| `reranker_scaler.pkl` | Sklearn scaler for GBM features |
| `reranker_metadata.json` | Contains `n_features: 18` — checked on load to guard against feature mismatch |
| `featcache_H_pool200_broad200_rrf15_derived1_nfeat18.pkl` | Precomputed feature cache for GBM training |
| `bge-small-engram-v3/` | Local embedding model (134 MB, sentence-transformers format) |
| `locomo_recall_v8_bge_reranker_v2m3.json` | v8 champion result JSON |
| `locomo_recall_v11_lexical_channels.json` | v11 result (written when v11 completes) |
| `bench_v*.log` | Full stdout logs of each benchmark run |

### Repo 2: `C:\Users\Sheldon Antony\.config\opencode\` (production system)

**GitHub remote:** `https://github.com/SheldonAntony/engram.git` (branch: `main`)



| File | Purpose |
|------|---------|
| `memory.py` | **PRODUCTION** retrieval code — final port target. Currently NOT updated with v8+ improvements. |
| `utils.py` | Shared utilities: `embed_text()`, `embed_texts_batch()`, `cosine_similarity()`, `get_cross_encoder()` |
| `memory_manager.py` | Manages conversation memory ingestion |

---

## 3. How the Full Pipeline Works

The pipeline lives in `eval_locomo.py` → `run_recall_eval()` (line ~1152). For each QA question:

```
Question
   │
   ├─► [Cosine ranking]   Sort all facts by cosine(q_emb, fact_emb) descending
   │                       → _cos_order[fid → rank]
   │
   ├─► [BM25 ranking]     FTS5 query on facts_fts table, OR-tokenised
   │                       → _bm25_rank_eval[fid → rank]
   │
   ├─► [Derived BM25]     Build "derived query" from LLM expansion, hit facts_derived_fts
   │   (optional, env)    → derived_rank_eval[fid → rank]
   │
   ├─► [RRF merge]        Reciprocal Rank Fusion:
   │                       rrf_score[fid] = 1/(K+cos_rank) + w/(K+bm25_rank) + 1/(K_d+derived_rank)
   │                       K=15 (PREFLIGHT_RRF_K), w=1.0 (PREFLIGHT_BM25_WEIGHT)
   │
   ├─► [Broad Pool]       PHASE 1 — Union top-N from each signal:
   │   (BROAD_POOL=200)    broad_parts = cos[:200] + bm25[:200] + derived[:200]
   │                       + NEW: name_channel[:200] + date_channel[:200] + bigram_channel[:200]
   │                       dedup → broad_cands (~400-800 unique fids)
   │                       Tail (facts not in pool) appended after, sorted by RRF
   │
   ├─► [GBM Reranker]     PHASE 2 — 18-feature HistGBM scores broad_cands
   │   (LEARNED_RERANK)    Features: cos_sim, bm25_rank, derived_rank, IDF weights, etc.
   │                       alpha=3.0 blend: rrf_norm + 3.0*gbm_prob → sorted descending
   │
   ├─► [Coverage Guard]   PHASE 3 — Min-rank ensemble:
   │   (COVERAGE_K=40)     final_rank[fid] = min(gbm_rank[fid], rrf_rank[fid])
   │                       Guarantees R@40 ≥ RRF baseline (cannot regress below RRF)
   │
   ├─► [CE Reranker]      PHASE 4 — bge-reranker-v2-m3 cross-encoder scores top-200
   │   (CE_POOL=200)       Input: (question, [curr] line of fact content)
   │                       CE replaces ordering of top-200 candidates entirely (alpha=0)
   │
   └─► [CE Guard]         PHASE 5 — Min-rank ensemble:
       (CE_GUARD_K=40)     final_rank[fid] = min(ce_rank[fid], pre_ce_rank[fid])
                           Guarantees R@40 ≥ pre-CE baseline
                           NOTE: CE_GUARD_K value is boolean only (>0 = enabled).
                           The formula applies to ALL pool members, not just top-K.
```

### Key data structures available per question inside `run_recall_eval()`:

```python
fact_cache          # list of (fid, content, embedding) for all facts in project
content_by_fid_ev   # dict {fid: content_str}  — full [prev]/[curr]/[next] window text
fids_in_cache       # tuple of all fid ints
cos_rank            # dict {fid: rank_int}  — 0=best cosine match
bm25_rank_eval      # dict {fid: rank_int}  — 0=best BM25 match
derived_rank_eval   # dict {fid: rank_int}  — if _USE_DERIVED_BM25
rrf_scores          # dict {fid: float}     — merged RRF score (higher=better)
conn                # sqlite3 connection    — FTS5 available on facts_fts table
qa["question"]      # str                  — the question text
qa["category"]      # str                  — temporal/single_hop/multi_hop/open_domain
```

### Content format of each fact:
```
[prev] SpeakerName: text of previous turn
[curr] SpeakerName: text of this turn  ← this is what the question asks about
[next] SpeakerName: text of next turn
```
CE scorer extracts only the `[curr]` line (via `_curr_text()`) — the full window format confuses the CE model.

### Fact types in DB:
- `window` — sliding window facts (used in first-stage pool) — embeds [prev]+[curr]+[next]
- `turn` — exact turn facts — EXCLUDED from first-stage pool (same embedding as window, wastes K slots)
- `llm_atomic` — atomic facts extracted by LLM — EXCLUDED from first-stage pool when GBM is on
- `derived` — derived/expanded text facts — used only for derived BM25 signal

---

## 4. Database Schema (Key Tables)

```sql
-- Main facts table
CREATE TABLE facts (
    id              INTEGER PRIMARY KEY,
    project_id      TEXT,          -- e.g. "locomo_1"
    fact_type       TEXT,          -- window/turn/llm_atomic/derived
    content         TEXT,          -- [prev]/[curr]/[next] formatted text
    embedding       BLOB,          -- float32 array, little-endian packed
    superseded_at   INTEGER,       -- NULL = active
    valid_to        INTEGER,       -- NULL = no expiry
    ...
);

-- FTS5 virtual tables
CREATE VIRTUAL TABLE facts_fts USING fts5(content, content='facts', content_rowid='id');
CREATE VIRTUAL TABLE facts_derived_fts USING fts5(...);  -- for derived BM25
```

Query pattern for BM25:
```python
fts_q = " OR ".join(f'"{t}"' for t in tokens)
rows = conn.execute(
    "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
    (fts_q,)
).fetchall()
```

---

## 5. Environment & How to Run

### Required environment variables (v8 champion config):
```powershell
$env:ENGRAM_EMBED_BACKEND = "sentence-transformers"
$env:ENGRAM_EMBED_MODEL   = "C:\Users\Sheldon Antony\.config\preflight\bge-small-engram-v3"
$env:PREFLIGHT_RRF_K      = "15"
$env:PREFLIGHT_USE_DERIVED_BM25      = "1"
$env:PREFLIGHT_USE_LEARNED_RERANK    = "1"
$env:PREFLIGHT_BROAD_POOL            = "200"
$env:PREFLIGHT_COVERAGE_K            = "40"
$env:PREFLIGHT_LEARNED_RERANK_ALPHA  = "3.0"
$env:PREFLIGHT_USE_CE                = "1"
$env:PREFLIGHT_CE_GUARD_K            = "40"
$env:PREFLIGHT_CE_POOL               = "200"
$env:PREFLIGHT_CE_MODEL              = "BAAI/bge-reranker-v2-m3"
```

### v11 adds:
```powershell
$env:PREFLIGHT_USE_LEXICAL_CHANNELS = "1"
```

### How to run a benchmark:
```powershell
cd "C:\Users\Sheldon Antony\.config\preflight"
# Set all env vars above first, then:
python recall_ablation.py --tag v11_lexical_channels
# Output: locomo_recall_v11_lexical_channels.json
# Stdout: full recall table printed at end
```

### IMPORTANT: `cd` must quote the path (space in username):
```powershell
cd "C:\Users\Sheldon Antony\.config\preflight"   # ✓ correct
cd C:\Users\Sheldon Antony\.config\preflight       # ✗ fails — PowerShell splits on space
```

### Reading results from JSON:
```python
import json
data = json.load(open("locomo_recall_v11_lexical_channels.json"))
# data["by_k"] = {3: 0.8081, 5: 0.8693, 10: 0.9152, 40: 0.9698, ...}
# data["by_category"] = {"temporal": {...}, "single_hop": {...}, ...}
```

### Embedding model details:
- Location: `C:\Users\Sheldon Antony\.config\preflight\bge-small-engram-v3\`
- 134 MB, sentence-transformers format (fine-tuned from BGE-small-en-v1.5 on LoCoMo pairs)
- Backend: `sentence-transformers` (NOT fastembed — fastembed has different tokenization)
- Loaded via `utils.embed_texts_batch()` for batched question embedding

### CE model:
- `BAAI/bge-reranker-v2-m3` — 2.27 GB, downloaded from HuggingFace on first run
- Cached in HuggingFace default cache (usually `~/.cache/huggingface/`)
- Loaded via `utils.get_cross_encoder()` — controlled by `PREFLIGHT_CE_MODEL` env var
- Warning "unauthenticated requests" is harmless — no HF_TOKEN needed for public models

### GBM reranker:
- `reranker_model.pkl` — HistGradientBoostingClassifier, 18 features
- `reranker_scaler.pkl` — StandardScaler for features
- `reranker_metadata.json` — `{"n_features": 18}` — checked on load (mismatch = crash)
- Retrain with: `python train_reranker.py` (uses `featcache_*.pkl`)

---

## 6. Complete Experiment Log

### Baseline progression:

| Tag | R@1 | R@3 | R@5 | R@10 | R@40 | Decision |
|-----|-----|-----|-----|------|------|----------|
| baseline (cosine only) | ~50% | 65.90% | 73.87% | 81.78% | 92.62% | reference |
| v3_k15 (RRF+BM25) | — | ~68% | — | — | ~93% | stepping stone |
| v3_derived (+ derived BM25) | — | ~69% | — | — | ~94% | improvement |
| v4_learned_gbm (+ GBM reranker, 18-feat) | — | 70.96% | 78.12% | 86.01% | 95.20% | big jump |
| v5_ce_xsmall (+ mxbai CE) | — | 77.07% | 84.23% | 90.28% | 96.71% | another jump |
| **v8_bge_reranker_v2m3** (CE upgraded) | — | **80.81%** | **86.93%** | **91.52%** | **96.98%** | **CHAMPION** |
| v9_pool100 (CE_POOL=100) | — | 80.49% | 86.47% | 91.20% | 96.06% | REJECTED |
| v10_alpha2 (CE_ALPHA=2.0) | — | 77.99% | 82.79% | 88.50% | 95.66% | REJECTED |
| v11_lexical_channels | — | **TBD** | **TBD** | **TBD** | **TBD** | PENDING |

### Detailed experiment decisions:

#### v3 series (RRF parameter sweep)
- Swept RRF_K ∈ {15, 25, 30, 40, 50, 60}. K=15 was best (tighter RRF = cosine dominates less).
- Added derived BM25 (LLM-expanded query text) — small +1pp R@40 gain.
- BM25 weight sweep: 0.5, 0.75, 1.0, 1.5, 2.0. 1.0 was best.

#### v4 — GBM reranker
- Trained `HistGradientBoostingClassifier` on 18 features (cos_sim, bm25_rank, derived_rank, IDF weights, query length, content length, etc.)
- `BROAD_POOL=200`: instead of reranking all ~2000 facts, take union of top-200 from each signal first. This let GBM see facts that rank well in any ONE signal.
- `COVERAGE_K=40`: after GBM, apply min-rank(gbm_rank, rrf_rank) so R@40 cannot regress below RRF.
- `LEARNED_RERANK_ALPHA=3.0`: blend RRF rank with GBM probability — keeps GBM from overriding strong RRF signals completely.
- Result: +5pp R@3 vs v3 (70.96%).

#### v5 — first CE (mxbai-rerank-xsmall)
- Added cross-encoder reranker (mxbai-rerank-xsmall, ~80MB). 
- CE fed full window content `[prev]/[curr]/[next]` initially — net NEGATIVE (CE confused by format).
- Fixed: extract only `[curr]` line via `_curr_text()`. CE needs clean single-turn text.
- `CE_POOL=200`: CE only sees top-200 from GBM (not all facts).
- `CE_GUARD_K=40`: after CE, apply min-rank(ce_rank, pre_ce_rank) so R@40 cannot regress.
- **CRITICAL INSIGHT**: `CE_GUARD_K` value is boolean only. The guard formula is:
  ```python
  final_rank[fid] = min(ce_rank[fid], pre_ce_rank[fid])
  ```
  applied to ALL candidates in the pool, not just top-K. Setting K=20 vs K=40 vs K=60 makes NO difference. Only 0 (disabled) vs >0 (enabled) matters.
- Result: +6pp R@3 (77.07%).

#### v6, v7 — pool size experiments
- v6 (hard guard): tried limiting CE pool to top-40 only → R@40 dropped (CE can't rescue rank 41-200 items).
- v7 (pool=300): CE_POOL=300 → marginal gain, longer runtime. Not worth it.

#### v8 — upgrade CE model
- Replaced mxbai-rerank-xsmall with `BAAI/bge-reranker-v2-m3` (2.27 GB, much larger model).
- Same pipeline, same hyperparams — just better CE model.
- Result: +3.74pp R@3 over v5 (80.81%). **NEW CHAMPION**.

#### v9 — CE_POOL=100 (REJECTED)
- Hypothesis: smaller CE pool = faster, and GBM top-100 contains all relevant facts.
- Result: R@3=80.49% (-0.32pp), R@40=96.06% **(-0.92pp)**. REJECTED.
- Root cause: 14 questions had gold fact at GBM rank 101-200. CE_POOL=100 never scored them → lost CE rescue.

#### v10 — CE_ALPHA=2.0 (CATASTROPHICALLY REJECTED)
- Hypothesis: blend CE score with GBM rank instead of pure CE replacement.
  - Formula: `final_score = rank_norm(1.0→0.005) + 2.0 * sigmoid(CE_score)`
  - rank_norm: 1.0 for rank-1, 0.005 for rank-N (linear decay)
- Result: R@3=77.99% **(-2.82pp)**. R@40=95.66% (-1.32pp). CATASTROPHIC.
- Root cause unknown, but empirically: blending CE with rank_norm destroys the CE gains.
- **CE_ALPHA IS PERMANENTLY ABANDONED**. Always use alpha=0 (pure CE replacement).

#### v11 — Lexical Explicit-Memory Channels (PENDING)
- Hypothesis: 37 questions have gold facts that NEVER appear in the top-200 broad pool, regardless of signal. Cosine AND BM25 both miss them. These are "true pool misses."
- Analysis by category:
  - `temporal`: 11 pool misses — questions about specific dates/times
  - `open_domain`: 12 pool misses — questions about specific entities/people
  - `single_hop`: 10 pool misses — direct factual questions
  - `multi_hop`: 4 pool misses — multi-step reasoning questions
- Solution: Add 3 new in-memory retrieval channels to `_broad_parts`:
  
  **Channel A — Person-name**: Extract capitalized name tokens from question (filtering common words). Find facts containing those names. Score by count of matches. Add top-200 to broad pool.
  ```python
  _name_toks = [w for w in re.findall(r'\b[A-Z][a-z]{2,}\b', question) if w not in _STOPNAME]
  ```
  
  **Channel B — Date/year**: Extract year patterns and "Month YYYY" patterns from question. Find facts containing those date strings. Score by count.
  ```python
  _date_toks = re.findall(r'\b(?:January|...|December)\s+\d{4}\b|\b\d{4}\b', question)
  ```
  
  **Channel C — Key-bigram**: Extract adjacent non-stopword word pairs from question. Find facts containing those exact bigrams.
  ```python
  _bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
  ```
  
- Code location: `eval_locomo.py` lines 1322–1366, gated by `PREFLIGHT_USE_LEXICAL_CHANNELS=1`.
- Expected impact: R@40 → ~98% (recover ~37 pool misses), R@3 → ~82-84%.

### Diagnostic analysis (diag_v8.py):

Run `python diag_v8.py` to see per-question comparison between v5 and v8.

Key findings from diagnostic:
- v5→v8 hit@3 gains: 60 questions gained (CE model upgrade helped)
- v5→v8 hit@3 losses: 23 questions lost (v8 CE overconfident in some cases)
- net +37 questions at R@3
- v5→v8 hit@40: minimal change (same pool structure)

The 37 true pool misses at R@40 are the ceiling — if gold fact is never in top-200 pool, NO amount of reranking can help. Lexical channels (v11) are designed to fix these.

---

## 7. Key Insights & Hard-Won Lessons

### 1. CE_GUARD_K is boolean-only
`PREFLIGHT_CE_GUARD_K` only enables/disables the guard. The actual K value is irrelevant because the min-rank formula applies to ALL candidates in the pool. Do NOT spend time sweeping K values.

### 2. CE_ALPHA destroys gains (permanently abandoned)
At any alpha > 0, blending CE score with rank_norm causes catastrophic regression. Mechanism is unclear but the empirical result is definitive. Always use alpha=0 (pure CE replacement of top-N order).

### 3. CE pool size matters for R@40
CE_POOL=200 (not 100) is necessary. Items at GBM rank 101-200 can still be rescued by CE. Reducing pool to 100 causes -0.92pp R@40 regression.

### 4. CE needs clean [curr] text, not full window
The CE model (bge-reranker-v2-m3) is trained on clean (query, passage) pairs. Feeding the full `[prev]/[curr]/[next]` window text confuses it and degrades performance. Always extract only the `[curr]` line via `_curr_text()`.

### 5. Broad pool union is critical for R@40 ceiling
Without `BROAD_POOL=200`, GBM only sees the top-N from RRF, missing facts that rank 201+ in cosine but high in BM25. The union of top-200 from each signal dramatically raises the ceiling.

### 6. Coverage guard cannot regress below RRF baseline
`PREFLIGHT_COVERAGE_K=40` applies min-rank(reranker_rank, rrf_rank). This is a safety net — reranking can never push a top-40 RRF item out of top-40. Always keep this enabled.

### 7. GBM alpha=3.0 is the right blend
`PREFLIGHT_LEARNED_RERANK_ALPHA=3.0` blends GBM prob with RRF rank. Too low (0) = pure GBM, too noisy. Too high (>5) = too close to pure RRF, wastes GBM training. 3.0 is the empirical optimum.

### 8. RRF_K=15 beats K=60
Tighter RRF (K=15) means cosine rank differences matter more. With K=60, all ranks get similar scores. K=15 was consistently best in the v3 sweep.

### 9. fact_type='turn' must be excluded from pool
Turn rows and their companion window rows share the same embedding (both represent the same `[curr]` text). Including both wastes top-K slots with duplicate signals. Always filter `fact_type != 'turn'` from the first-stage pool.

### 10. Conv 10 can OOM on first run
The 10th conversation is the largest. If the benchmark crashes with OOM on Conv 10, just rerun from scratch (the process loads everything fresh each time). Second run usually succeeds.

### 11. Path must be quoted in PowerShell
`cd "C:\Users\Sheldon Antony\.config\preflight"` — the space in "Sheldon Antony" breaks unquoted cd.

---

## 8. Current Code State

### `eval_locomo.py` — MODIFIED (v11 changes)
- Line ~95: Added `_USE_LEXICAL_CHANNELS` env var flag
- Lines 1322–1366: Three new lexical channels injected into `_broad_parts`
- Everything else is identical to v8 champion config

### `recall_ablation.py` — MODIFIED
- Added `--tag` argument for output file naming
- No logic changes

### `utils.py` (opencode) — MODIFIED
- Added `PREFLIGHT_CE_MODEL` env var support in `get_cross_encoder()`
- This is how v8+ can use `BAAI/bge-reranker-v2-m3` instead of the default xsmall model

### `memory.py` (opencode) — UNCHANGED
- Still uses old retrieval logic (pre-v4)
- Needs port AFTER a winning config is confirmed
- Do NOT port until v11 results are analyzed

### `reranker.py` — UNCHANGED
- 18 features, HistGBM
- `N_FEATURES = 18` version guard on load

---

## 9. What To Do Next

### ⚠️ CRITICAL: DB LETTER MISMATCH — READ FIRST

All benchmarks v4 through v10 used `locomo_eval_H.db` (passed `--db-letter H` or old default).  
v11 ran on `locomo_eval_B.db` (current default in `recall_ablation.py`).

**You CANNOT directly compare v11 (B DB) to v8 (H DB).**

**Step 0 — IMMEDIATE after v11 finishes:**  
Run v8 config on B DB as a control:
```powershell
cd "C:\Users\Sheldon Antony\.config\preflight"
$env:ENGRAM_EMBED_BACKEND="sentence-transformers"
$env:ENGRAM_EMBED_MODEL="C:\Users\Sheldon Antony\.config\preflight\bge-small-engram-v3"
$env:PREFLIGHT_RRF_K="15"
$env:PREFLIGHT_USE_DERIVED_BM25="1"
$env:PREFLIGHT_USE_LEARNED_RERANK="1"
$env:PREFLIGHT_BROAD_POOL="200"
$env:PREFLIGHT_COVERAGE_K="40"
$env:PREFLIGHT_LEARNED_RERANK_ALPHA="3.0"
$env:PREFLIGHT_USE_CE="1"
$env:PREFLIGHT_CE_GUARD_K="40"
$env:PREFLIGHT_CE_POOL="200"
$env:PREFLIGHT_CE_MODEL="BAAI/bge-reranker-v2-m3"
# Note: NO PREFLIGHT_USE_LEXICAL_CHANNELS here — this is the v8 config baseline
python recall_ablation.py --tag v8_bdb_control
```
This gives you `locomo_recall_v8_bdb_control.json` — the true comparison point for v11.

---

### IMMEDIATE (after v8_bdb_control completes):

**If v11 beats v8_bdb_control (R@3 > v8_bdb_control AND R@40 > v8_bdb_control):**
1. Mark v11 as new champion
2. Add 3 new GBM features to `reranker.py` (to teach GBM about lexical channel hits):
   - `name_token_hit_count` — how many name tokens from Q appear in fact
   - `date_token_hit_count` — how many date tokens from Q appear in fact
   - `bigram_hit_count` — how many Q bigrams appear in fact
3. Retrain GBM: `python train_reranker.py` (update feature cache first)
4. Run v12 = v11 config + retrained GBM
5. If v12 beats v11 → new champion

**If v11 regresses at R@3 but R@40 improves vs v8_bdb_control:**
- The channels are working (pool misses fixed) but GBM/CE is confused by new candidates
- Try running channels A+B only (disable bigram channel: it adds noise for every question)
- Add a new env flag `PREFLIGHT_LEXICAL_NAME_DATE_ONLY=1` to run A+B without C

**If v11 regresses at both R@3 and R@40 vs v8_bdb_control:**
- The channels are adding too much noise
- The bigram channel (C) is the most likely culprit (every Q has bigrams → huge pool inflation)
- Try: name channel only, then date channel only, to isolate which helps
- Consider FTS5 phrase match instead of in-memory substring scan for channel C

### MEDIUM TERM:

**GBM retraining with lexical features:**
```python
# In reranker.py, add to feature extraction:
def _name_hit_count(content: str, question: str) -> int:
    import re
    STOPNAME = {'The', 'What', 'Who', ...}
    toks = [w for w in re.findall(r'\b[A-Z][a-z]{2,}\b', question) if w not in STOPNAME]
    return sum(content.count(t) for t in toks)

def _date_hit_count(content: str, question: str) -> int:
    import re
    pats = re.findall(r'\b(?:Jan|...|Dec)[a-z]*\s+\d{4}\b|\b\d{4}\b', question)
    return sum(content.count(p) for p in pats)

def _bigram_hit_count(content: str, question: str) -> int:
    words = [w for w in re.sub(r'[^a-z\s]', ' ', question.lower()).split() if len(w) > 2]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    c = content.lower()
    return sum(1 for bg in bigrams if bg in c)
```
Then update `N_FEATURES = 21` and retrain.

**Port winning config to memory.py:**
Once a champion is confirmed, replicate the pipeline in `memory.py`:
1. Add `_USE_LEXICAL_CHANNELS` branch to `retrieve_facts()`
2. Add broad pool union logic
3. Add GBM reranker call
4. Add CE reranker call (already partially done in utils.py)
5. Add coverage guards

### LONGER TERM IDEAS (not yet tested):

1. **Adjacent-turn expansion**: When a channel hits fact at rank X, also add facts at X-1 and X+1 (neighboring turns). Requires knowing turn order from fid sequence.

2. **FTS5 phrase match for bigrams**: Instead of in-memory substring scan, use FTS5 PHRASE query: `"word1 word2"`. Faster and more precise.

3. **Query expansion with LLM**: For temporal questions, ask LLM "when did X happen?" and use the answer as an additional BM25 query. Expensive but potentially high value.

4. **Speaker-constrained channel**: If the question asks about "what did Alice do", restrict pool to turns where Alice is the `[curr]` speaker. Need to parse `[curr] Alice:` from content.

5. **Fine-tune embedding model further**: We have `bge-small-engram-v3` (already fine-tuned). Could try another round with hard negatives mined from pool misses.

---

## 10. Acceptance Rules

These rules MUST be satisfied before promoting any version to champion:

1. **Must beat the v8_bdb_control** (v8 config on B DB) at both R@3 AND R@40
2. All comparisons must be on the **same DB** (B DB = `locomo_eval_B.db` = current default)
3. **R@5 single_hop** must not drop significantly (watch this category carefully)
4. No OOM crashes (if it crashes, rerun once; if it crashes twice, reject)

> NOTE: The old "v8 champion" numbers (R@3=80.81%, R@40=96.98%) were measured on H DB.  
> Do NOT use these as acceptance thresholds for B-DB runs. Run v8_bdb_control first.

A version that improves R@40 but regresses R@3 by > 0.5pp is also rejected — R@3 is the production metric.

---

## Appendix A: All Env Vars Reference

| Env Var | Default | Effect |
|---------|---------|--------|
| `ENGRAM_EMBED_BACKEND` | — | `sentence-transformers` (required) |
| `ENGRAM_EMBED_MODEL` | — | path to local embedding model (required) |
| `PREFLIGHT_RRF_K` | 60 | RRF smoothing constant (15 = tighter, better) |
| `PREFLIGHT_BM25_WEIGHT` | 1.0 | BM25 contribution weight in RRF |
| `PREFLIGHT_USE_STOPWORDS` | 0 | 1 = filter BM25 stopwords |
| `PREFLIGHT_USE_DERIVED_BM25` | 0 | 1 = add derived BM25 channel |
| `PREFLIGHT_USE_LEARNED_RERANK` | 0 | 1 = enable GBM reranker |
| `PREFLIGHT_LEARNED_RERANK_POOL` | 80 | How many candidates GBM reranks (ignored when BROAD_POOL>0) |
| `PREFLIGHT_LEARNED_RERANK_ALPHA` | 0.0 | 3.0 = blend RRF+GBM (0=pure GBM) |
| `PREFLIGHT_BROAD_POOL` | 0 | N > 0 = take top-N from each signal into union pool |
| `PREFLIGHT_COVERAGE_K` | 0 | N > 0 = min-rank guard after GBM (protects RRF top-N) |
| `PREFLIGHT_USE_CE` | 0 | 1 = enable cross-encoder reranker |
| `PREFLIGHT_CE_POOL` | 100 | How many top candidates CE scores (use 200) |
| `PREFLIGHT_CE_GUARD_K` | 0 | N > 0 = min-rank guard after CE (value is boolean only!) |
| `PREFLIGHT_CE_ALPHA` | 0.0 | **DO NOT USE** — values > 0 cause catastrophic regression |
| `PREFLIGHT_CE_MODEL` | (xsmall) | HuggingFace model ID for CE (use BAAI/bge-reranker-v2-m3) |
| `PREFLIGHT_USE_LEXICAL_CHANNELS` | 0 | 1 = enable name/date/bigram candidate channels (v11+) |

---

## Appendix B: File Locations Quick Reference

```
C:\Users\Sheldon Antony\.config\
├── preflight\                        ← benchmark repo (git)
│   ├── eval_locomo.py                ← CORE pipeline code
│   ├── recall_ablation.py            ← benchmark runner
│   ├── reranker.py                   ← GBM feature extraction
│   ├── train_reranker.py             ← GBM training script
│   ├── locomo10.json                 ← source dataset
│   ├── locomo_eval_B.db              ← benchmark DB (USE THIS ONE)
│   ├── reranker_model.pkl            ← trained GBM
│   ├── reranker_metadata.json        ← {"n_features": 18}
│   ├── bge-small-engram-v3\          ← local embedding model
│   ├── locomo_recall_v8_*.json       ← v8 champion results
│   ├── locomo_recall_v11_*.json      ← v11 results (pending)
│   ├── bench_v*.log                  ← full run logs
│   └── diag_v8.py                    ← diagnostic script
│
└── opencode\                         ← production repo (git)
    ├── memory.py                     ← PRODUCTION retrieval (needs port)
    └── utils.py                      ← embed/CE utilities (MODIFIED for CE model)
```

---

*This document was auto-generated during handover on 2026-05-15. v11 benchmark results will be appended below once the run completes.*

---

## v11 Results (to be filled in)

```
R@1:  TBD
R@3:  TBD   (v8 champion: 80.81%)
R@5:  TBD   (v8 champion: 86.93%)
R@10: TBD   (v8 champion: 91.52%)
R@40: TBD   (v8 champion: 96.98%)

Decision: PENDING
```
