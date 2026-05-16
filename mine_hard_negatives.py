#!/usr/bin/env python3
"""Mine hard negatives from the best existing retrieval DB (B.db / fastembed).

For each (query, positive) pair in engram_pairs_locomo.jsonl:
  1. Retrieves top-N facts from B.db using the current retriever (fastembed RRF).
  2. Marks retrieved facts that overlap with the gold positive as "gold" (skip).
  3. Collects the top-ranked non-gold facts as hard negatives.
  4. Writes JSONL triplets: {query, positive, hard_negatives: [...], category, ...}

Output: engram_triplets.jsonl
Usage:
    python mine_hard_negatives.py
    python mine_hard_negatives.py --db-letter B --top-n 40 --max-neg 5
"""

from __future__ import annotations
import argparse
import json
import os
import re
import struct
import sqlite3
import sys
import time

_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_SCRIPTS_DIR   = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

# Force fastembed backend so we match the vectors stored in B.db.
# Must be set BEFORE importing utils (which reads env at import time).
os.environ["ENGRAM_EMBED_BACKEND"] = "fastembed"
os.environ["ENGRAM_EMBED_MODEL"]   = "BAAI/bge-small-en-v1.5"

import numpy as np  # for fast batch cosine similarity

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--db-letter", default="B",
                    help="DB suffix letter to mine from (default: B = best fastembed DB).")
parser.add_argument("--top-n", type=int, default=40,
                    help="How many facts to retrieve per query.")
parser.add_argument("--max-neg", type=int, default=5,
                    help="Max hard negatives to keep per query.")
parser.add_argument("--gold-threshold", type=float, default=0.65,
                    help="Token-recall threshold to consider a retrieved fact as gold.")
parser.add_argument("--input",  default="engram_pairs_locomo.jsonl")
parser.add_argument("--output", default="engram_triplets.jsonl")
parser.add_argument("--rrf-k", type=int, default=20,
                    help="RRF K parameter (should match the best known setting).")
parser.add_argument("--bm25-weight", type=float, default=1.0)
args = parser.parse_args()

DB_PATH  = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{args.db_letter}.db")
IN_PATH  = os.path.join(_PREFLIGHT_DIR, args.input)
OUT_PATH = os.path.join(_PREFLIGHT_DIR, args.output)

# ── Load embedding backend (must be fastembed to match B.db vectors) ─────────
# Do NOT set ENGRAM_EMBED_BACKEND — default is fastembed, which matches B.db.
print("Loading fastembed embedding model (must match B.db vectors) ...")
try:
    from utils import embed_text as _embed, cosine_similarity as _cos
    _ = _embed("warmup")  # trigger lazy load
    print("  ✓ fastembed ready")
except Exception as exc:
    print(f"  ✗ Failed to load embedding model: {exc}")
    sys.exit(1)

if not os.path.exists(DB_PATH):
    print(f"  ✗ DB not found: {DB_PATH}")
    sys.exit(1)

print(f"  DB: {DB_PATH}")


# ── Gold-matching helper ──────────────────────────────────────────────────────

def _norm_tokens(text: str) -> set[str]:
    """Lowercase, strip tags and speaker labels, tokenize."""
    t = text.lower()
    t = re.sub(r'\[(prev|curr|next)\]\s*', '', t)       # strip window tags
    t = re.sub(r'^\w[\w ]*:\s*', '', t, flags=re.M)      # strip "Speaker: " per line
    t = re.sub(r'[^\w\s]', ' ', t)
    return set(t.split())


def _is_gold(gold_text: str, retrieved_content: str, threshold: float) -> bool:
    """True if retrieved_content substantially overlaps the gold turn text.

    Uses token-recall: what fraction of gold tokens appear in retrieved?
    This handles window facts ([prev]/[curr]/[next]) correctly — the gold
    single-turn text is *contained* in the window, so recall is high.
    """
    gt = _norm_tokens(gold_text)
    rc = _norm_tokens(retrieved_content)
    if not gt:
        return False
    recall = len(gt & rc) / len(gt)
    return recall >= threshold


# ── Full-corpus RRF retrieval (mirrors eval_locomo._eval_retrieve) ────────────

BM25_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "how", "did", "does",
    "was", "were", "are", "the", "and", "for", "with", "from",
})


def retrieve(db_path: str, project_id: str, query: str, top_n: int) -> list[dict]:
    """Return top_n facts for project sorted by RRF(cosine, BM25)."""
    q_emb = _embed(query)
    conn  = sqlite3.connect(db_path)
    rows  = conn.execute(
        """SELECT id, content, embedding FROM facts
           WHERE project_id = ?
             AND superseded_at IS NULL
             AND fact_type != 'turn'
             AND (valid_to IS NULL OR valid_to > unixepoch())""",
        (project_id,),
    ).fetchall()

    if not rows:
        conn.close()
        return []

    fact_cache: list[tuple[int, str, list]] = []
    for fid, content, blob in rows:
        if blob is None:
            continue
        if isinstance(blob, (bytes, bytearray)):
            n   = len(blob) // 4
            emb = list(struct.unpack(f"{n}f", blob))
        else:
            try:
                emb = json.loads(blob)
            except Exception:
                continue
        fact_cache.append((fid, content, emb))

    if not fact_cache:
        conn.close()
        return []

    n_facts = len(fact_cache)
    # Fast batch cosine similarity via numpy (avoids per-fact Python loop)
    q_arr   = np.array(q_emb, dtype=np.float32)
    fids_ordered = [fid for fid, _, _e in fact_cache]
    emb_matrix   = np.array([emb for _, _, emb in fact_cache], dtype=np.float32)
    cos_scores   = emb_matrix @ q_arr          # dot product (vecs pre-normalized)
    ranked_idx   = np.argsort(-cos_scores)     # descending
    cos_rank     = {fids_ordered[idx]: int(rank) for rank, idx in enumerate(ranked_idx)}

    bm25_rank: dict[int, int] = {}
    try:
        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
        tokens = [t for t in safe.split() if len(t) > 2 and t.lower() not in BM25_STOPWORDS]
        if tokens:
            fts_q = " OR ".join(f'"{t}"' for t in tokens)
            all_fids = {fid for fid, _, _e in fact_cache}
            bm_rows  = conn.execute(
                "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                (fts_q,),
            ).fetchall()
            rank = 0
            for (bfid,) in bm_rows:
                if bfid in all_fids:
                    bm25_rank[bfid] = rank
                    rank += 1
    except Exception:
        pass
    conn.close()

    rrf: dict[int, float] = {}
    for fid, _, _e in fact_cache:
        s = 1.0 / (args.rrf_k + cos_rank.get(fid, n_facts))
        if fid in bm25_rank:
            s += args.bm25_weight / (args.rrf_k + bm25_rank[fid])
        rrf[fid] = s

    content_by_fid = {fid: content for fid, content, _e in fact_cache}
    sorted_fids = sorted(rrf, key=rrf.__getitem__, reverse=True)
    return [{"id": fid, "content": content_by_fid[fid], "score": rrf[fid]}
            for fid in sorted_fids[:top_n]]


# ── Get all project IDs from DB ───────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
all_pids = {r[0] for r in conn.execute("SELECT DISTINCT project_id FROM facts").fetchall()}
conn.close()
print(f"  {len(all_pids)} projects in DB: {sorted(all_pids)[:5]} ...")

# ── Load pairs ────────────────────────────────────────────────────────────────
print(f"\nLoading pairs from {IN_PATH} ...")
pairs: list[dict] = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
print(f"  {len(pairs):,} pairs loaded")

# ── Mine hard negatives ───────────────────────────────────────────────────────
print(f"\nMining hard negatives (top-{args.top_n}, max {args.max_neg} per query) ...")
print(f"  DB: {DB_PATH}")
print(f"  Gold threshold: {args.gold_threshold}")

triplets:     list[dict] = []
no_project:   int = 0
no_retrieved: int = 0
no_negatives: int = 0
t0 = time.time()

for i, pair in enumerate(pairs):
    if i % 200 == 0:
        elapsed = time.time() - t0
        print(f"  [{i:>5}/{len(pairs)}] triplets so far: {len(triplets):,}  ({elapsed:.0f}s)")

    sid = pair.get("sample_id", "")
    pid = f"locomo_{sid}"

    if pid not in all_pids:
        no_project += 1
        continue

    query    = pair["query"]
    positive = pair["positive"]

    retrieved = retrieve(DB_PATH, pid, query, args.top_n)

    if not retrieved:
        no_retrieved += 1
        continue

    # Classify each retrieved fact as gold or hard negative
    hard_negs: list[str] = []
    for r in retrieved:
        if _is_gold(positive, r["content"], args.gold_threshold):
            continue  # this is the gold — skip
        hard_negs.append(r["content"])
        if len(hard_negs) >= args.max_neg:
            break

    if not hard_negs:
        no_negatives += 1
        # Still emit as a plain pair (no hard neg) so training doesn't lose this example
        triplets.append({
            "query":          query,
            "positive":       positive,
            "hard_negatives": [],
            "category":       pair.get("category", ""),
            "sample_id":      sid,
            "source":         pair.get("source", ""),
            "has_hard_neg":   False,
        })
        continue

    triplets.append({
        "query":          query,
        "positive":       positive,
        "hard_negatives": hard_negs,
        "category":       pair.get("category", ""),
        "sample_id":      sid,
        "source":         pair.get("source", ""),
        "has_hard_neg":   True,
    })

elapsed = time.time() - t0

# ── Write ─────────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

with_neg = sum(1 for t in triplets if t["has_hard_neg"])

print(f"\nDone in {elapsed:.1f}s")
print(f"  Total triplets written : {len(triplets):,}")
print(f"  With hard negatives    : {with_neg:,}  ({100*with_neg/max(len(triplets),1):.1f}%)")
print(f"  Without hard negatives : {len(triplets) - with_neg:,}")
print(f"  Skipped — no project   : {no_project:,}")
print(f"  Skipped — no retrieved : {no_retrieved:,}")
print(f"  Gold-only (no neg)     : {no_negatives:,}")
print(f"\n→ {OUT_PATH}")
