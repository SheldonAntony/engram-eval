#!/usr/bin/env python3
"""Oracle rank audit for LoCoMo retrieval.

For every QA pair with evidence, computes where the gold fact lands under each
retrieval signal and reports TRUE any-signal@K recall (set-union membership
check — always >= any individual signal).

Usage (H.db winner):
    $env:PREFLIGHT_RRF_K="15"; $env:PREFLIGHT_USE_DERIVED_BM25="1"
    python oracle_audit.py --db-letter H

I.db with atomic rerank signal:
    python oracle_audit.py --db-letter I --include-atomic
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
import struct
import sys
import time

_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
_SCRIPTS_DIR   = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--db-letter", default="H", metavar="LETTER")
parser.add_argument("--include-atomic", action="store_true",
                    help="Add atomic-rerank signal (needs llm_atomic facts in DB).")
parser.add_argument("--save", default="", help="Save per-question data to this JSON file.")
args = parser.parse_args()
DB_PATH    = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{args.db_letter}.db")
DATA_CACHE = os.path.join(_PREFLIGHT_DIR, "locomo10.json")

# ── Env flags (matching eval_locomo.py) ──────────────────────────────────────
_RRF_K            = int(os.environ.get("PREFLIGHT_RRF_K", "60"))
_BM25_RRF_WEIGHT  = float(os.environ.get("PREFLIGHT_BM25_WEIGHT", "1.0"))
_USE_DERIVED_BM25 = os.environ.get("PREFLIGHT_USE_DERIVED_BM25", "0") == "1"
_INCLUDE_ATOMIC   = args.include_atomic or os.environ.get("PREFLIGHT_ORACLE_INCLUDE_ATOMIC", "0") == "1"
_ATOMIC_ALPHA     = float(os.environ.get("PREFLIGHT_LLM_ATOMIC_ALPHA", "0.15"))
_ATOMIC_POOL      = int(os.environ.get("PREFLIGHT_LLM_ATOMIC_POOL", "40"))
_ATOMIC_RRF_K     = int(os.environ.get("PREFLIGHT_ATOMIC_K", "15"))

# ── Embedding ─────────────────────────────────────────────────────────────────
import utils as _utils  # noqa: E402
from utils import embed_texts_batch as _ub, cosine_similarity as _cs  # noqa: E402

# ── Helpers (copied from eval_locomo.py to avoid import-time side-effects) ────

def _decode_blob(blob) -> list | None:
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))
    try:
        return json.loads(blob)
    except Exception:
        return None


def iter_sessions(conv: dict):
    """Yield (session_num, date_str, turns) from session_N / session_N_date_time keys."""
    import re as _re
    nums = sorted(
        int(k.split("_")[1])
        for k in conv.keys()
        if _re.match(r"^session_\d+$", k)
    )
    for n in nums:
        turns    = conv.get(f"session_{n}", [])
        date_str = conv.get(f"session_{n}_date_time", "")
        yield n, str(date_str), turns


_CAT_NAMES = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain"}
_SKIP_CATS = {5}  # adversarial

def iter_qa(sample: dict):
    """Match eval_locomo.py::iter_qa exactly (category field, evidence list)."""
    for qa in sample.get("qa", []):
        try:
            cat = int(qa.get("category", 0))
        except (ValueError, TypeError):
            cat = 0
        if cat in _SKIP_CATS:
            continue
        if not qa.get("question"):
            continue
        raw_evidence = qa.get("evidence", []) or []
        yield {
            "question": str(qa["question"]),
            "evidence": [str(d) for d in raw_evidence if d is not None],
            "cat_name":  _CAT_NAMES.get(cat, str(cat)),
        }


# ── Oracle rank audit ─────────────────────────────────────────────────────────

KS = [1, 3, 5, 10, 20, 40, 100, 200, 500]


def _bm25_ranks(conn, question: str, fids_set: set) -> dict[int, int]:
    """Return {fid: rank} for FTS5 BM25 retrieval over the given fid set."""
    try:
        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in question)
        tokens = [t for t in safe.split() if len(t) > 2]
        if not tokens or not fids_set:
            return {}
        fts_q  = " OR ".join(f'"{t}"' for t in tokens)
        rows   = conn.execute(
            "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
            (fts_q,),
        ).fetchall()
        result, rank = {}, 0
        for (fid,) in rows:
            if fid in fids_set:
                result[fid] = rank
                rank += 1
        return result
    except Exception:
        return {}


def _derived_bm25_ranks(conn, question: str, fids_set: set) -> dict[int, int]:
    """Return {fid: rank} for derived FTS5 BM25 (WordNet expansion)."""
    try:
        from memory import _build_derived_text as _bdt  # noqa: PLC0415
        derived_q = _bdt(question)
        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in derived_q)
        tokens = [t for t in safe.split() if len(t) > 2]
        if not tokens or not fids_set:
            return {}
        fts_q  = " OR ".join(f'"{t}"' for t in tokens)
        rows   = conn.execute(
            "SELECT rowid FROM facts_derived_fts"
            " WHERE facts_derived_fts MATCH ? ORDER BY bm25(facts_derived_fts)",
            (fts_q,),
        ).fetchall()
        result, rank = {}, 0
        for (fid,) in rows:
            if fid in fids_set:
                result[fid] = rank
                rank += 1
        return result
    except Exception:
        return {}


def _best_gold_rank(sorted_fids: list[int], gold_fids: set[int]) -> int | None:
    """0-based rank of the best-placed gold fact in sorted_fids, or None."""
    for r, fid in enumerate(sorted_fids):
        if fid in gold_fids:
            return r
    return None


def _atomic_reranked(
    conn, pid: str, q_emb: list,
    rrf_sorted: list, rrf_scores: dict,
) -> list:
    """Apply llm_atomic rerank (max mode, alpha=_ATOMIC_ALPHA) to top-_ATOMIC_POOL.

    Mirrors eval_locomo.py::_atomic_rerank with score_mode='max'.
    Returns full list (reranked pool + unchanged tail).
    """
    import hashlib as _hl
    pool = rrf_sorted[:_ATOMIC_POOL]
    if not pool:
        return rrf_sorted
    placeholders = ",".join("?" * len(pool))
    content_rows = conn.execute(
        f"SELECT id, content FROM facts WHERE id IN ({placeholders})", pool
    ).fetchall()
    fid_to_turn_hash: dict[int, str] = {}
    for fid, content in content_rows:
        for ln in content.split("\n"):
            if ln.startswith("[curr] "):
                curr_line = ln[len("[curr] "):]
                fid_to_turn_hash[fid] = _hl.sha256(curr_line.encode()).hexdigest()[:16]
                break
    if not fid_to_turn_hash:
        return rrf_sorted
    hashes = list(set(fid_to_turn_hash.values()))
    ph2 = ",".join("?" * len(hashes))
    atomic_rows = conn.execute(
        f"""SELECT source_hash, embedding FROM facts
            WHERE project_id = ?
              AND fact_type = 'llm_atomic'
              AND source_hash IN ({ph2})
              AND superseded_at IS NULL
              AND (valid_to IS NULL OR valid_to > unixepoch())""",
        [pid, *hashes],
    ).fetchall()
    hash_to_max_sim: dict[str, float] = {}
    for sh, blob in atomic_rows:
        if blob is None:
            continue
        emb = _decode_blob(blob)
        if emb is None:
            continue
        sim = _cs(q_emb, emb)
        if sh not in hash_to_max_sim or sim > hash_to_max_sim[sh]:
            hash_to_max_sim[sh] = sim
    combined = {
        fid: rrf_scores.get(fid, 0.0)
              + _ATOMIC_ALPHA * hash_to_max_sim.get(fid_to_turn_hash.get(fid, ""), 0.0)
        for fid in pool
    }
    reranked_pool = sorted(pool, key=combined.__getitem__, reverse=True)
    return reranked_pool + rrf_sorted[_ATOMIC_POOL:]


def run_oracle_audit(samples: list, db_path: str) -> dict:
    import hashlib as _hl

    print(f"\n{'='*60}")
    print(f"  ORACLE RANK AUDIT")
    print(f"  DB : {db_path}")
    print(f"  RRF_K={_RRF_K}  BM25_W={_BM25_RRF_WEIGHT}  DERIVED={_USE_DERIVED_BM25}")
    print(f"{'='*60}")

    # ── Build dia_id map (same as eval_locomo.py::build_dia_id_map) ──────────
    print("\nBuilding dia_id map...", flush=True)
    dia_id_map: dict[str, dict[str, set]] = {}
    conn = sqlite3.connect(db_path)
    for ci, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", ci))
        pid     = f"locomo_{sid_str}"
        rows_all = conn.execute(
            "SELECT id, content, source_hash, fact_type FROM facts"
            " WHERE project_id = ? AND superseded_at IS NULL",
            (pid,),
        ).fetchall()
        content_to_ids: dict[str, set] = {}
        hash_to_llm_ids: dict[str, set] = {}
        for priority_tags in (("[prev] ", "[next] "), ("[curr] ",)):
            for fid, content, _sh, _ft in rows_all:
                for line in content.split("\n"):
                    for tag in priority_tags:
                        if line.startswith(tag):
                            content_to_ids.setdefault(line[len(tag):], set()).add(fid)
        for fid, content, _sh, _ft in rows_all:
            if not any("\n" + t in ("\n" + content) for t in ("[curr] ", "[prev] ", "[next] ")):
                content_to_ids.setdefault(content, set()).add(fid)
        for fid, _content, source_hash, fact_type in rows_all:
            if fact_type == "llm_atomic" and source_hash:
                hash_to_llm_ids.setdefault(source_hash, set()).add(fid)
        pid_map: dict[str, set] = {}
        for _sn, _d, turns in iter_sessions(sample.get("conversation", {})):
            for turn in turns:
                dia_id  = turn.get("dia_id")
                speaker = str(turn.get("speaker", "?"))
                text    = str(turn.get("text", ""))
                if not text.strip() or dia_id is None:
                    continue
                ckey = f"{speaker}: {text}"
                fids = content_to_ids.get(ckey)
                if fids:
                    fids = set(fids)
                    turn_hash = _hl.sha256(ckey.encode()).hexdigest()[:16]
                    if turn_hash in hash_to_llm_ids:
                        fids.update(hash_to_llm_ids[turn_hash])
                    pid_map[str(dia_id)] = fids
        dia_id_map[pid] = pid_map

    # ── Per-question audit ───────────────────────────────────────────────────
    print("Auditing questions...", flush=True)
    if _INCLUDE_ATOMIC:
        print(f"  Atomic rerank ON  alpha={_ATOMIC_ALPHA}  pool={_ATOMIC_POOL}  K={_ATOMIC_RRF_K}")
    t0 = time.time()

    # Individual signal accumulators: {signal: {K: hit_count}}
    # Each tracks the 0-based rank of the best gold fact under that signal.
    _sigs = ["cos", "bm25", "derived", "rrf"]
    if _INCLUDE_ATOMIC:
        _sigs.append("atomic")
    oracle_hits: dict[str, dict[int, int]] = {sig: {k: 0 for k in KS} for sig in _sigs}

    # True ANY-signal@K oracle: gold in union of top-K across all active signals.
    # This is guaranteed >= every individual signal — we assert that below.
    any_union_hits: dict[int, int] = {k: 0 for k in KS}

    # Pool oracle: for candidate pool size N, gold in union of top-N from each signal.
    # This is the TRUE ceiling for a selector that reads N candidates per signal.
    # pool_oracle@N >= any_union@N because pool takes top-N PER SIGNAL (not total).
    POOL_SIZES = [20, 40, 80, 100, 150, 200, 300]
    pool_oracle_hits: dict[int, int] = {n: 0 for n in POOL_SIZES}

    per_q_data: list[dict] = []

    for si, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", si))
        pid     = f"locomo_{sid_str}"
        pid_map = dia_id_map.get(pid, {})
        print(f"  Conv {si+1}/{len(samples)}: loading embeddings...", flush=True)

        # Preload window-only facts (exclude turn & llm_atomic — same as winner config)
        rows = conn.execute(
            """SELECT id, content, embedding FROM facts
               WHERE project_id = ?
                 AND superseded_at IS NULL
                 AND fact_type NOT IN ('turn', 'llm_atomic')
                 AND (valid_to IS NULL OR valid_to > unixepoch())""",
            (pid,),
        ).fetchall()
        fact_cache: list[tuple[int, str, list]] = []
        for fid, content, blob in rows:
            emb = _decode_blob(blob)
            if emb is not None:
                fact_cache.append((fid, content, emb))

        if not fact_cache:
            continue

        fids_set = {fid for fid, _, _ in fact_cache}
        n_facts  = len(fact_cache)

        # Batch-embed questions for this conversation
        qa_list = list(iter_qa(sample))
        q_texts  = [qa["question"] for qa in qa_list]
        q_embs   = _ub(q_texts)

        for qa, q_emb in zip(qa_list, q_embs):
            if q_emb is None:
                continue
            evidence = qa["evidence"]
            gold_fids: set[int] = set()
            for d in evidence:
                fids = pid_map.get(d)
                if fids:
                    gold_fids.update(fids)
            if not gold_fids:
                continue  # skip no-evidence / adversarial

            # ── Cosine ranking ───────────────────────────────────────────────
            cos_sorted = sorted(fact_cache, key=lambda x: _cs(q_emb, x[2]), reverse=True)
            cos_rank   = {fid: i for i, (fid, _, _) in enumerate(cos_sorted)}
            cos_fids   = [fid for fid, _, _ in cos_sorted]

            # ── BM25 ranking ─────────────────────────────────────────────────
            bm25_r = _bm25_ranks(conn, qa["question"], fids_set)
            # Fill non-ranked fids at tail
            bm25_sorted = sorted(fids_set, key=lambda f: bm25_r.get(f, n_facts))

            # ── Derived BM25 ─────────────────────────────────────────────────
            if _USE_DERIVED_BM25:
                derived_r = _derived_bm25_ranks(conn, qa["question"], fids_set)
            else:
                derived_r = {}
            derived_sorted = sorted(fids_set, key=lambda f: derived_r.get(f, n_facts))

            # ── RRF fusion ───────────────────────────────────────────────────
            rrf_scores: dict[int, float] = {}
            for fid, _, _ in fact_cache:
                s = 1.0 / (_RRF_K + cos_rank.get(fid, n_facts))
                if fid in bm25_r:
                    s += _BM25_RRF_WEIGHT / (_RRF_K + bm25_r[fid])
                rrf_scores[fid] = s
            if _USE_DERIVED_BM25 and derived_r:
                _K_DERIVED = 60
                for fid in rrf_scores:
                    if fid in derived_r:
                        rrf_scores[fid] += 1.0 / (_K_DERIVED + derived_r[fid])
            rrf_sorted = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

            # ── Atomic rerank (optional, I.db only) ──────────────────────────
            atomic_sorted: list[int] = []
            if _INCLUDE_ATOMIC:
                atomic_sorted = _atomic_reranked(conn, pid, q_emb, rrf_sorted, rrf_scores)

            # ── Best gold rank per individual signal ─────────────────────────
            def _best(sorted_list) -> int | None:
                for r, fid in enumerate(sorted_list):
                    if fid in gold_fids:
                        return r
                return None

            ranks: dict[str, int | None] = {
                "cos":     _best(cos_fids),
                "bm25":    _best(bm25_sorted),
                "derived": _best(derived_sorted),
                "rrf":     _best(rrf_sorted),
            }
            if _INCLUDE_ATOMIC:
                ranks["atomic"] = _best(atomic_sorted)

            # Accumulate individual signal hits
            for sig, best in ranks.items():
                for k in KS:
                    if best is not None and best < k:
                        oracle_hits[sig][k] += 1

            # ── TRUE any-signal@K oracle (set-union membership check) ─────────
            # By construction any_union@K >= max(individual@K) at every K.
            signal_lists = [cos_fids, bm25_sorted, derived_sorted, rrf_sorted]
            if _INCLUDE_ATOMIC and atomic_sorted:
                signal_lists.append(atomic_sorted)
            q_any_union: dict[int, bool] = {}
            for k in KS:
                union_topk: set[int] = set()
                for sl in signal_lists:
                    union_topk.update(sl[:k])
                hit = bool(gold_fids & union_topk)
                q_any_union[k] = hit
                if hit:
                    any_union_hits[k] += 1

            # Pool oracle: gold in union of top-N PER SIGNAL (broader than any_union@N)
            q_pool_oracle: dict[int, bool] = {}
            for n in POOL_SIZES:
                pool_union: set[int] = set()
                for sl in signal_lists:
                    pool_union.update(sl[:n])
                ph = bool(gold_fids & pool_union)
                q_pool_oracle[n] = ph
                if ph:
                    pool_oracle_hits[n] += 1

            per_q_data.append({
                "question":   qa["question"],
                "category":   qa["cat_name"],
                "gold_fids":  list(gold_fids),
                "n_gold":     len(gold_fids),
                # 1-based ranks per signal; None = gold not in pool
                "ranks":      {sig: (r + 1 if r is not None else None) for sig, r in ranks.items()},
                "any_union":  {str(k): v for k, v in q_any_union.items()},
                "pool_oracle": {str(n): v for n, v in q_pool_oracle.items()},
            })

    conn.close()
    elapsed = time.time() - t0

    total_ev = len(per_q_data)
    print(f"\n  Done in {elapsed:.1f}s — {total_ev} questions with evidence")

    # ── Sanity check: any_union@K must be >= every individual signal@K ────────
    sanity_ok = True
    for k in KS:
        union_pct = any_union_hits[k] / total_ev * 100 if total_ev else 0.0
        for sig in _sigs:
            sig_pct = oracle_hits[sig][k] / total_ev * 100 if total_ev else 0.0
            if union_pct < sig_pct - 0.01:  # 0.01pp tolerance for float rounding
                print(f"  SANITY FAIL @{k}: union={union_pct:.2f}% < {sig}={sig_pct:.2f}%")
                sanity_ok = False
    if sanity_ok:
        print("  Sanity check PASSED: any_union@K >= all individual signals at every K")

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ORACLE RECALL (% of {total_ev} evidence questions)")
    print(f"{'='*60}")
    sig_labels: dict[str, str] = {
        "cos":     "Cosine only  ",
        "bm25":    "BM25 only    ",
        "derived": "Derived BM25 ",
        "rrf":     "RRF fusion   ",
        "atomic":  "Atomic rerank",
    }

    hdr = f"{'Signal':<16}" + "".join(f"  @{k:<5}" for k in KS)
    print(hdr)
    print("-" * len(hdr))

    results: dict[str, dict] = {}
    for sig in _sigs:
        row = f"{sig_labels[sig]:<16}"
        results[sig] = {}
        for k in KS:
            pct = oracle_hits[sig][k] / total_ev * 100 if total_ev else 0.0
            row += f"  {pct:5.1f}"
            results[sig][k] = round(pct, 2)
        print(row)

    # ANY-union row (true oracle ceiling)
    any_union_results: dict[int, float] = {}
    row = f"{'ANY union    ':<16}"
    for k in KS:
        pct = any_union_hits[k] / total_ev * 100 if total_ev else 0.0
        row += f"  {pct:5.1f}"
        any_union_results[k] = round(pct, 2)
    print(row)

    # ── Pool oracle (true candidate-set ceiling) ──────────────────────────────
    print(f"\n  Pool oracle (gold in union top-N per signal = ceiling for selector with pool-N candidates):")
    pool_oracle_results: dict[int, float] = {}
    for n in POOL_SIZES:
        pct = pool_oracle_hits[n] / total_ev * 100 if total_ev else 0.0
        pool_oracle_results[n] = round(pct, 2)
        print(f"    @{n:<4}: {pct:5.1f}%")

    # ── Gap analysis: any_union vs RRF ───────────────────────────────────────
    print(f"\n  Gap analysis (any_union – RRF):")
    for k in KS:
        gap = any_union_results[k] - results["rrf"][k]
        sign = "+" if gap >= 0 else ""
        print(f"    @{k:<4}: {sign}{gap:5.1f}pp  "
              f"(any_union={any_union_results[k]:.1f}%  rrf={results['rrf'][k]:.1f}%)")

    # ── Per-category breakdown ────────────────────────────────────────────────
    print(f"\n  Per-category any_union oracle (vs RRF):")
    cats = ["single_hop", "multi_hop", "temporal", "open_domain"]
    cat_labels = {"single_hop": "Single-hop", "multi_hop": "Multi-hop",
                  "temporal": "Temporal", "open_domain": "Open-domain"}
    for cat in cats:
        cat_q = [q for q in per_q_data if q["category"] == cat]
        if not cat_q:
            continue
        n = len(cat_q)
        def _pct_rrf(k):
            return sum(1 for q in cat_q
                       if q["ranks"]["rrf"] is not None and q["ranks"]["rrf"] <= k) / n * 100
        def _pct_union(k):
            return sum(1 for q in cat_q if q["any_union"][str(k)]) / n * 100
        print(f"    {cat_labels[cat]:<14} n={n:4d}  "
              f"any_union@3={_pct_union(3):5.1f}%  rrf@3={_pct_rrf(3):5.1f}%  "
              f"any_union@40={_pct_union(40):5.1f}%  rrf@40={_pct_rrf(40):5.1f}%")

    # ── Breakdown by retrieval difficulty (using RRF and any_union) ───────────
    def _breakdown(label: str, not_in_pool_fn, rank_fn):
        not_found        = [q for q in per_q_data if not_in_pool_fn(q)]
        found_deep       = [q for q in per_q_data
                            if not not_in_pool_fn(q) and not rank_fn(q, 40)]
        in40_not3        = [q for q in per_q_data
                            if not not_in_pool_fn(q) and rank_fn(q, 40) and not rank_fn(q, 3)]
        in_top3          = [q for q in per_q_data
                            if not not_in_pool_fn(q) and rank_fn(q, 3)]
        print(f"\n  {label} breakdown:")
        print(f"    Gold NOT in pool at all  : {len(not_found):4d}  ({len(not_found)/total_ev*100:.1f}%)")
        print(f"    Gold in pool, rank >40   : {len(found_deep):4d}  ({len(found_deep)/total_ev*100:.1f}%)")
        print(f"    Gold in top-40, rank >3  : {len(in40_not3):4d}  ({len(in40_not3)/total_ev*100:.1f}%)")
        print(f"    Gold in top-3            : {len(in_top3):4d}  ({len(in_top3)/total_ev*100:.1f}%)")

    _breakdown(
        "RRF fusion   ",
        lambda q: q["ranks"]["rrf"] is None,
        lambda q, k: q["ranks"]["rrf"] is not None and q["ranks"]["rrf"] <= k,
    )
    _breakdown(
        "ANY union    ",
        lambda q: not q["any_union"][str(max(KS))],  # not found at all = not in max K union
        lambda q, k: q["any_union"][str(k)],
    )

    # ── Save per-question data ───────────────────────────────────────────────
    out = {
        "db_letter":        args.db_letter,
        "total_ev":         total_ev,
        "KS":               KS,
        "signals":          results,
        "any_union":        any_union_results,
        "pool_oracle":      pool_oracle_results,
        "per_question":     per_q_data,
        "elapsed_s":        round(elapsed, 1),
        "sanity_passed":    sanity_ok,
        "config": {
            "RRF_K":          _RRF_K,
            "BM25_W":         _BM25_RRF_WEIGHT,
            "USE_DERIVED":    _USE_DERIVED_BM25,
            "INCLUDE_ATOMIC": _INCLUDE_ATOMIC,
            "ATOMIC_ALPHA":   _ATOMIC_ALPHA,
            "ATOMIC_POOL":    _ATOMIC_POOL,
        },
    }
    save_path = args.save or os.path.join(
        _PREFLIGHT_DIR,
        f"oracle_audit_{args.db_letter}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full per-question data -> {save_path}")

    return out


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found.")
        sys.exit(1)
    print(f"Loading dataset from {DATA_CACHE} ...")
    with open(DATA_CACHE, encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = list(samples.values())
    print(f"  {len(samples)} conversations\n")

    run_oracle_audit(samples, DB_PATH)
