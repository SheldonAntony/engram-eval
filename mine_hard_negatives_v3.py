#!/usr/bin/env python3
"""mine_hard_negatives_v3.py — Mine clean hard negatives for triplet training.

Key improvements over v2 (mine_hard_negatives.py):
  1. Forces fastembed backend before any imports — matches B.db vectors exactly.
  2. Extracts the [curr] line from retrieved windows as the training negative text,
     matching the representation the DB actually embeds at inference time.
  3. Multi-level false-negative filtering:
       Level 2: [curr]-line token recall vs positive_curr >= --curr-threshold
                  → rejects turns whose core content matches the gold answer.
       Level 3: any window line token recall vs positive_curr >= --adj-threshold
                  → rejects windows that are adjacent to the gold turn (prev/next).
  4. Reads engram_pairs_v3.jsonl (has positive_curr and dia_id fields).
  5. Caps negatives at --max-neg (default 2, not 5).
  6. Exports negative_curr: list[str]  — the [curr] lines to use for training.
     (hard_negatives: full windows also kept for debugging.)

Output: engram_triplets_v3.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
import struct
import sys
import time

# ── MUST set env vars before any other import ─────────────────────────────────
os.environ["ENGRAM_EMBED_BACKEND"] = "fastembed"
os.environ["ENGRAM_EMBED_MODEL"]   = "BAAI/bge-small-en-v1.5"

_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_OPENCODE_DIR  = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _OPENCODE_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--db-letter",      default="B",
                    help="DB letter for B.db (fastembed vectors).")
parser.add_argument("--input",          default="engram_pairs_v3.jsonl")
parser.add_argument("--output",         default="engram_triplets_v3.jsonl")
parser.add_argument("--top-n",          type=int,   default=40)
parser.add_argument("--max-neg",        type=int,   default=2,
                    help="Max clean hard negatives per query (default 2).")
parser.add_argument("--curr-threshold", type=float, default=0.30,
                    help="[curr]-line token recall threshold for false-neg detection.")
parser.add_argument("--adj-threshold",  type=float, default=0.45,
                    help="Any window line token recall threshold for adjacent-gold detection.")
parser.add_argument("--rrf-k",          type=int,   default=20)
parser.add_argument("--bm25-weight",    type=float, default=1.0)
parser.add_argument("--debug",          action="store_true")
args = parser.parse_args()

DB_PATH  = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{args.db_letter}.db")
IN_PATH  = os.path.join(_PREFLIGHT_DIR, args.input)
OUT_PATH = os.path.join(_PREFLIGHT_DIR, args.output)

# ── Load embedding backend ────────────────────────────────────────────────────

print("Loading fastembed embedding model ...")
try:
    from utils import embed_text as _embed
    _ = _embed("warmup")
    print("  ✓ fastembed ready")
except Exception as exc:
    print(f"  ✗ Failed to load fastembed: {exc}")
    sys.exit(1)

if not os.path.exists(DB_PATH):
    print(f"✗ DB not found: {DB_PATH}")
    sys.exit(1)
print(f"  DB: {DB_PATH}\n")

# ── Token helpers ─────────────────────────────────────────────────────────────

def _norm_tokens(text: str) -> set[str]:
    t = text.lower()
    t = re.sub(r'\[(prev|curr|next)\]\s*', '', t)
    t = re.sub(r'^\w[\w ]*:\s*', '', t, flags=re.M)
    t = re.sub(r'[^\w\s]', ' ', t)
    return {w for w in t.split() if len(w) > 2}


def _token_recall(gold: str, candidate: str) -> float:
    """Fraction of gold tokens found in candidate."""
    g = _norm_tokens(gold)
    c = _norm_tokens(candidate)
    if not g:
        return 0.0
    return len(g & c) / len(g)


def _extract_curr_line(window: str) -> str:
    """Extract [curr] line text, stripping the '[curr] ' prefix."""
    for line in window.split("\n"):
        if line.startswith("[curr] "):
            return line[len("[curr] "):]
    return ""   # not a standard window fact


def _any_line_is_gold(positive: str, window: str, threshold: float) -> bool:
    """True if any [prev]/[curr]/[next] line has token recall >= threshold."""
    for line in window.split("\n"):
        for tag in ("[prev] ", "[curr] ", "[next] "):
            if line.startswith(tag):
                stripped = line[len(tag):]
                if _token_recall(positive, stripped) >= threshold:
                    return True
                break
    return False


# ── BM25 stop-words (mirrors mine_hard_negatives.py) ─────────────────────────

BM25_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "how", "did", "does",
    "was", "were", "are", "the", "and", "for", "with", "from",
})

# ── RRF retrieval ─────────────────────────────────────────────────────────────

def retrieve(project_id: str, query: str, top_n: int) -> list[dict]:
    """Return top_n facts for project sorted by RRF(cosine, BM25)."""
    q_emb = _embed(query)
    conn  = sqlite3.connect(DB_PATH)
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

    n_facts    = len(fact_cache)
    q_arr      = np.array(q_emb, dtype=np.float32)
    fids_list  = [fid for fid, _, _ in fact_cache]
    emb_matrix = np.array([emb for _, _, emb in fact_cache], dtype=np.float32)
    cos_scores = emb_matrix @ q_arr
    ranked_idx = np.argsort(-cos_scores)
    cos_rank   = {fids_list[idx]: int(r) for r, idx in enumerate(ranked_idx)}

    bm25_rank: dict[int, int] = {}
    try:
        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
        tokens = [t for t in safe.split() if len(t) > 2 and t.lower() not in BM25_STOPWORDS]
        if tokens:
            fts_q   = " OR ".join(f'"{t}"' for t in tokens)
            all_fids = {fid for fid, _, _ in fact_cache}
            bm_rows = conn.execute(
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

    content_by_fid = {fid: content for fid, content, _ in fact_cache}
    rrf: dict[int, float] = {}
    for fid, _, _ in fact_cache:
        s = 1.0 / (args.rrf_k + cos_rank.get(fid, n_facts))
        if fid in bm25_rank:
            s += args.bm25_weight / (args.rrf_k + bm25_rank[fid])
        rrf[fid] = s

    sorted_fids = sorted(rrf, key=rrf.__getitem__, reverse=True)
    return [{"id": fid, "content": content_by_fid[fid], "score": rrf[fid]}
            for fid in sorted_fids[:top_n]]


# ── Load project IDs ──────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)
all_pids = {r[0] for r in conn.execute("SELECT DISTINCT project_id FROM facts").fetchall()}
conn.close()
print(f"  {len(all_pids)} projects in DB")

# ── Load pairs ────────────────────────────────────────────────────────────────

print(f"Loading pairs from {IN_PATH} ...")
pairs: list[dict] = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
print(f"  {len(pairs):,} pairs loaded\n")

# ── Mine ─────────────────────────────────────────────────────────────────────

print(f"Mining hard negatives  (top-{args.top_n}, max {args.max_neg}/query)")
print(f"  curr_threshold = {args.curr_threshold}")
print(f"  adj_threshold  = {args.adj_threshold}\n")

triplets:      list[dict] = []
no_project     = 0
no_retrieved   = 0
no_negatives   = 0
rejected_curr  = 0
rejected_adj   = 0
t0 = time.time()

for i, pair in enumerate(pairs):
    if i % 200 == 0:
        elapsed = time.time() - t0
        print(f"  [{i:>5}/{len(pairs)}]  triplets={len(triplets):,}  "
              f"no_neg={no_negatives}  ({elapsed:.0f}s)")

    sid = pair.get("sample_id", "")
    pid = f"locomo_{sid}"
    if pid not in all_pids:
        no_project += 1
        continue

    query         = pair["query"]
    positive      = pair.get("positive", "")
    positive_curr = pair.get("positive_curr") or positive   # Speaker: text

    retrieved = retrieve(pid, query, args.top_n)
    if not retrieved:
        no_retrieved += 1
        continue

    hard_negs_curr:   list[str] = []   # [curr] line — USE FOR TRAINING
    hard_negs_window: list[str] = []   # full window — debug only

    for r in retrieved:
        if len(hard_negs_curr) >= args.max_neg:
            break

        content   = r["content"]
        curr_line = _extract_curr_line(content)

        if not curr_line:
            # Not a standard window fact; skip
            continue

        # Level 2: [curr]-line token recall vs positive_curr
        curr_rec = _token_recall(positive_curr, curr_line)
        if curr_rec >= args.curr_threshold:
            rejected_curr += 1
            if args.debug:
                print(f"    REJECT[L2 curr_rec={curr_rec:.2f}] {curr_line[:70]}")
            continue

        # Level 3: any window line (prev/curr/next) overlaps positive
        if _any_line_is_gold(positive_curr, content, args.adj_threshold):
            rejected_adj += 1
            if args.debug:
                print(f"    REJECT[L3 adj_gold] {curr_line[:70]}")
            continue

        hard_negs_curr.append(curr_line)
        hard_negs_window.append(content)

    has_neg = len(hard_negs_curr) > 0
    if not has_neg:
        no_negatives += 1

    triplets.append({
        "query":          query,
        "positive":       positive,
        "positive_curr":  positive_curr,
        "negative_curr":  hard_negs_curr,         # [curr] lines — for training
        "hard_negatives": hard_negs_window,        # full windows — debug only
        "category":       pair.get("category", ""),
        "sample_id":      sid,
        "source":         pair.get("source", ""),
        "has_hard_neg":   has_neg,
    })

elapsed = time.time() - t0

# ── Write ─────────────────────────────────────────────────────────────────────

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

with_neg  = sum(1 for t in triplets if t["has_hard_neg"])
by_cat: dict[str, int] = {}
for t in triplets:
    c = t.get("category", "unknown")
    by_cat[c] = by_cat.get(c, 0) + 1

print(f"\nDone in {elapsed:.1f}s")
print(f"  Total triplets written   : {len(triplets):,}")
print(f"  With clean hard neg      : {with_neg:,}  ({100*with_neg/max(len(triplets),1):.1f}%)")
print(f"  Without hard neg         : {len(triplets) - with_neg:,}")
print()
print(f"  Rejected [L2 curr overlap]: {rejected_curr:,}")
print(f"  Rejected [L3 adj  overlap]: {rejected_adj:,}")
print(f"  Skipped  no project       : {no_project:,}")
print(f"  Skipped  no retrieved     : {no_retrieved:,}")
print(f"  No clean neg found        : {no_negatives:,}")
print()
print("  By category:")
for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"    {c:<24}  {n:>5}")
print(f"\n→ {OUT_PATH}")
