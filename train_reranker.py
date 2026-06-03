#!/usr/bin/env python3
"""Train the learned feature reranker for LoCoMo retrieval.

Builds a training dataset by running the full retrieval pipeline
(cosine + BM25 + derived BM25 + atomic) over locomo10.json against an
existing ingested DB, then trains a logistic regression ranker with
leave-one-conversation-out (LOOCV) cross-validation.

Saves the final model (trained on all conversations) to reranker_model.pkl
and reranker_scaler.pkl in the same directory.

Usage:
    $env:ENGRAM_EMBED_BACKEND="sentence-transformers"
    $env:ENGRAM_EMBED_MODEL="C:\\Users\\Sheldon Antony\\.config\\preflight\\bge-small-engram-v3"
    $env:PREFLIGHT_RRF_K="15"; $env:PREFLIGHT_USE_DERIVED_BM25="1"
    python train_reranker.py --db-letter H

    # With atomic signal (I.db):
    $env:PREFLIGHT_LLM_ATOMIC_ALPHA="0.15"; $env:PREFLIGHT_LLM_ATOMIC_POOL="40"
    python train_reranker.py --db-letter I --include-atomic
"""
from __future__ import annotations
import argparse
import hashlib
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
parser.add_argument("--db-letter",      default="H", metavar="LETTER")
parser.add_argument("--include-atomic", action="store_true",
                    help="Include atomic rerank boost as a training feature.")
parser.add_argument("--pool",           type=int, default=80,
                    help="RRF top-N candidates to form the training pool (default 80). Ignored when --broad-pool > 0.")
parser.add_argument("--broad-pool",     type=int, default=0,
                    help="If > 0, build training pool as union of top-N per signal (mirrors PREFLIGHT_BROAD_POOL).")
parser.add_argument("--model-type",     default="lr",
                    choices=["lr", "gbm"],
                    help="lr=LogisticRegression, gbm=GradientBoosting (default lr).")
parser.add_argument("--alpha",           type=float, default=0.0,
                    help="Soft-blend alpha for RRF+alpha*prob sort. 0=pure prob, 3=recommended.")
args = parser.parse_args()

DB_PATH      = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{args.db_letter}.db")
DATA_CACHE    = os.path.join(_PREFLIGHT_DIR, "locomo10.json")
POOL_SIZE     = args.pool
BROAD_POOL_N  = args.broad_pool  # 0 = use RRF top-POOL_SIZE; >0 = union per-signal

# ── Env flags ─────────────────────────────────────────────────────────────────
_RRF_K            = int(os.environ.get("PREFLIGHT_RRF_K",           "60"))
_RRF_K_COS        = int(os.environ.get("PREFLIGHT_RRF_K_COS",      "15"))
_RRF_K_BM25       = int(os.environ.get("PREFLIGHT_RRF_K_BM25",     "15"))
_BM25_RRF_WEIGHT  = float(os.environ.get("PREFLIGHT_BM25_WEIGHT", "1.0"))
_USE_DERIVED_BM25 = os.environ.get("PREFLIGHT_USE_DERIVED_BM25", "0") == "1"
_INCLUDE_ATOMIC   = args.include_atomic
_ATOMIC_ALPHA     = float(os.environ.get("PREFLIGHT_LLM_ATOMIC_ALPHA", "0.15"))
_ATOMIC_POOL      = int(os.environ.get("PREFLIGHT_LLM_ATOMIC_POOL", "40"))
_RECALL_KS        = [1, 3, 5, 10, 40]

# ── Imports ────────────────────────────────────────────────────────────────────
from utils import embed_texts_batch as _ub, cosine_similarity as _cs  # noqa: E402
import reranker as _rr  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

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


import re as _re  # noqa: E402
import re as _re_tok  # noqa: E402

def iter_sessions(conv: dict):
    nums = sorted(
        int(k.split("_")[1])
        for k in conv.keys()
        if _re.match(r"^session_\d+$", k)
    )
    for n in nums:
        yield n, str(conv.get(f"session_{n}_date_time", "")), conv.get(f"session_{n}", [])


_CAT_NAMES = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain"}
_SKIP_CATS = {5}

def iter_qa(sample: dict):
    for qa in sample.get("qa", []):
        try:
            cat = int(qa.get("category", 0))
        except (ValueError, TypeError):
            cat = 0
        if cat in _SKIP_CATS or not qa.get("question"):
            continue
        raw_ev = qa.get("evidence", []) or []
        yield {
            "question": str(qa["question"]),
            "evidence": [str(d) for d in raw_ev if d is not None],
            "cat_name": _CAT_NAMES.get(cat, str(cat)),
            "category": cat,
        }


def _build_dia_id_map(samples, conn):
    """project_id -> {dia_id -> set[fact_id]}."""
    dia_id_map: dict[str, dict[str, set]] = {}
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
            if not any(f"\n{t}" in f"\n{content}" for t in ("[curr] ", "[prev] ", "[next] ")):
                content_to_ids.setdefault(content, set()).add(fid)
        for fid, _c, source_hash, fact_type in rows_all:
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
                    turn_hash = hashlib.sha256(ckey.encode()).hexdigest()[:16]
                    if turn_hash in hash_to_llm_ids:
                        fids.update(hash_to_llm_ids[turn_hash])
                    pid_map[str(dia_id)] = fids
        dia_id_map[pid] = pid_map
    return dia_id_map


def _atomic_boosts_for_pool(conn, pid, pool_fids, q_emb) -> dict[int, float]:
    """Return {fid: max_atomic_cosine_sim} for facts in pool that have siblings."""
    if not pool_fids:
        return {}
    placeholders = ",".join("?" * len(pool_fids))
    content_rows = conn.execute(
        f"SELECT id, content FROM facts WHERE id IN ({placeholders})", pool_fids
    ).fetchall()
    fid_to_hash: dict[int, str] = {}
    for fid, content in content_rows:
        for ln in content.split("\n"):
            if ln.startswith("[curr] "):
                curr_line = ln[len("[curr] "):]
                fid_to_hash[fid] = hashlib.sha256(curr_line.encode()).hexdigest()[:16]
                break
    if not fid_to_hash:
        return {}
    hashes = list(set(fid_to_hash.values()))
    ph2 = ",".join("?" * len(hashes))
    atomic_rows = conn.execute(
        f"""SELECT source_hash, embedding FROM facts
            WHERE project_id = ? AND fact_type = 'llm_atomic'
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
    return {fid: hash_to_max_sim[h] for fid, h in fid_to_hash.items() if h in hash_to_max_sim}


# ── Main feature-collection loop ──────────────────────────────────────────────

def collect_features_for_conversation(
    sample: dict,
    pid_map: dict,
    pid: str,
    conn,
    fact_cache: list,
    q_embs: list,
    qa_list: list,
) -> list[dict]:
    """Return list of training examples for one conversation."""
    examples = []
    n_facts = len(fact_cache)
    fids_in_cache = tuple(fid for fid, _, _ in fact_cache)
    fids_set = set(fids_in_cache)
    emb_by_fid = {fid: emb for fid, _, emb in fact_cache}

    for qa, q_emb in zip(qa_list, q_embs):
        if q_emb is None:
            continue
        gold_fids: set[int] = set()
        for d in qa["evidence"]:
            fids = pid_map.get(d)
            if fids:
                gold_fids.update(fids)
        if not gold_fids:
            continue

        # ── Cosine ──────────────────────────────────────────────────────────
        cos_sims = {fid: _cs(q_emb, emb) for fid, _, emb in fact_cache}
        cos_sorted = sorted(fids_in_cache, key=cos_sims.__getitem__, reverse=True)
        cos_ranks = {fid: r for r, fid in enumerate(cos_sorted)}

        # ── BM25 ────────────────────────────────────────────────────────────
        bm25_ranks: dict[int, int] = {}
        try:
            safe = "".join(c if c.isalnum() or c.isspace() else " " for c in qa["question"])
            tokens = [t for t in safe.split() if len(t) > 2]
            if tokens and fids_in_cache:
                fts_q = " OR ".join(f'"{t}"' for t in tokens)
                bm_rows = conn.execute(
                    "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                    (fts_q,),
                ).fetchall()
                bm_rank = 0
                for (bfid,) in bm_rows:
                    if bfid in fids_set:
                        bm25_ranks[bfid] = bm_rank
                        bm_rank += 1
        except Exception:
            pass

        # ── RRF (per-signal K to match eval_locomo.py) ───────────────────────
        rrf_scores: dict[int, float] = {}
        for fid, _, _ in fact_cache:
            s = 1.0 / (_RRF_K_COS + cos_ranks.get(fid, n_facts))
            if fid in bm25_ranks:
                s += _BM25_RRF_WEIGHT / (_RRF_K_BM25 + bm25_ranks[fid])
            rrf_scores[fid] = s

        # ── Derived BM25 ────────────────────────────────────────────────────
        derived_ranks: dict[int, int] = {}
        if _USE_DERIVED_BM25:
            try:
                from memory import _build_derived_text as _bdt
                derived_q = _bdt(qa["question"])
                safe_d = "".join(c if c.isalnum() or c.isspace() else " " for c in derived_q)
                dtokens = [t for t in safe_d.split() if len(t) > 2]
                if dtokens and fids_in_cache:
                    dfts_q = " OR ".join(f'"{t}"' for t in dtokens)
                    dr_rows = conn.execute(
                        "SELECT rowid FROM facts_derived_fts"
                        " WHERE facts_derived_fts MATCH ? ORDER BY bm25(facts_derived_fts)",
                        (dfts_q,),
                    ).fetchall()
                    dr_rank = 0
                    for (dfid,) in dr_rows:
                        if dfid in fids_set:
                            derived_ranks[dfid] = dr_rank
                            dr_rank += 1
                    for fid in rrf_scores:
                        if fid in derived_ranks:
                            rrf_scores[fid] += 1.0 / (60 + derived_ranks[fid])
            except Exception:
                pass

        # ── Pool: broad union or RRF top-N ──────────────────────────────────────
        fids_in_cache = tuple(fid for fid, _, _ in fact_cache)
        if BROAD_POOL_N > 0:
            # Mirrors eval_locomo.py _BROAD_POOL logic: union of top-N per signal
            _cos_order  = sorted(fids_in_cache, key=lambda f: cos_ranks.get(f, n_facts))
            _bm25_order = sorted(fids_in_cache, key=lambda f: bm25_ranks.get(f, n_facts))
            _parts = _cos_order[:BROAD_POOL_N] + _bm25_order[:BROAD_POOL_N]
            if derived_ranks:
                _parts += sorted(
                    fids_in_cache, key=lambda f: derived_ranks.get(f, n_facts)
                )[:BROAD_POOL_N]
            pool = list(dict.fromkeys(_parts))
        else:
            rrf_sorted = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
            pool = rrf_sorted[:POOL_SIZE]

        # ── Atomic boosts ────────────────────────────────────────────────────
        atomic_boosts: dict[int, float] = {}
        if _INCLUDE_ATOMIC:
            atomic_boosts = _atomic_boosts_for_pool(conn, pid, pool, q_emb)

        # ── Content-based feature pre-computation ────────────────────────────
        content_by_fid_q = {fid: content for fid, content, _ in fact_cache}
        q_tokens = frozenset(
            w.lower() for w in _re_tok.findall(r'\w+', qa["question"]) if len(w) > 2
        )

        # ── Build feature rows ───────────────────────────────────────────────
        _fid_min = min(fids_in_cache) if fids_in_cache else 0
        _fid_max = max(fids_in_cache) if fids_in_cache else 1
        X = _rr.extract_features(
            pool_fids      = pool,
            rrf_scores     = rrf_scores,
            cos_scores     = cos_sims,
            cos_ranks      = cos_ranks,
            bm25_ranks     = bm25_ranks,
            derived_ranks  = derived_ranks,
            atomic_boosts  = atomic_boosts,
            n_facts        = n_facts,
            category       = qa["category"],
            question_tokens= q_tokens,
            content_by_fid = content_by_fid_q,
            question       = qa["question"],
            fid_bounds     = (_fid_min, _fid_max),
        )
        y = [1 if fid in gold_fids else 0 for fid in pool]
        examples.append({
            "pool":       pool,
            "X":          X,
            "y":          y,
            "gold":       gold_fids,
            "cat":        qa["cat_name"],
            "rrf_scores": {fid: rrf_scores[fid] for fid in pool},
        })
    return examples


# ── Recall metric ─────────────────────────────────────────────────────────────

def _recall_at_k(sorted_fids: list[int], gold_fids: set, k: int) -> bool:
    return bool(set(sorted_fids[:k]) & gold_fids)


def _report_metrics(label: str, all_examples: list[dict], model, scaler,
                    alpha: float = 0.0) -> dict:
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    hits = {k: 0 for k in _RECALL_KS}
    n_ev = 0
    for ex in all_examples:
        X_scaled = scaler.transform(ex["X"]) if scaler is not None else ex["X"]
        try:
            probs = model.predict_proba(X_scaled)[:, 1]
        except AttributeError:
            probs = model.predict(X_scaled)
        # Soft-blend: rrf_norm + alpha * prob (same logic as reranker.rerank_pool)
        if alpha > 0.0:
            import numpy as _np
            raw = _np.array([ex.get("rrf_scores", {}).get(fid, 0.0)
                             for fid in ex["pool"]], dtype=_np.float64)
            lo, hi = raw.min(), raw.max()
            rrf_norm = (raw - lo) / (hi - lo + 1e-12)
            scores = rrf_norm + alpha * probs
        else:
            scores = probs
        order = sorted(range(len(ex["pool"])), key=lambda i: scores[i], reverse=True)
        reranked = [ex["pool"][i] for i in order]
        for k in _RECALL_KS:
            if _recall_at_k(reranked, ex["gold"], k):
                hits[k] += 1
        n_ev += 1
    print(f"\n  {label}  (n={n_ev})")
    for k in _RECALL_KS:
        pct = hits[k] / n_ev * 100 if n_ev else 0.0
        print(f"    R@{k:<3}: {pct:.2f}%")
    return {k: round(hits[k] / n_ev * 100, 2) if n_ev else 0.0 for k in _RECALL_KS}


def _report_rrf_baseline(all_examples: list[dict]) -> dict:
    """Report RRF baseline (before rerank) on these examples, sorted by RRF score.
    This mirrors what eval_locomo.py does: pool sorted by RRF, then check gold in top-K.
    """
    hits = {k: 0 for k in _RECALL_KS}
    n_ev = len(all_examples)
    for ex in all_examples:
        rrf_s = ex.get("rrf_scores", {})
        rrf_sorted_pool = sorted(ex["pool"], key=lambda f: rrf_s.get(f, 0.0), reverse=True)
        for k in _RECALL_KS:
            if _recall_at_k(rrf_sorted_pool, ex["gold"], k):
                hits[k] += 1
    pool_label = BROAD_POOL_N if BROAD_POOL_N > 0 else POOL_SIZE
    print(f"\n  RRF baseline in broad pool (pool≈{pool_label}/signal, n={n_ev})")
    for k in _RECALL_KS:
        pct = hits[k] / n_ev * 100 if n_ev else 0.0
        print(f"    R@{k:<3}: {pct:.2f}%")
    return {k: round(hits[k] / n_ev * 100, 2) if n_ev else 0.0 for k in _RECALL_KS}


# ── Training ──────────────────────────────────────────────────────────────────

def _fit_model(X_all, y_all, model_type: str):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler     # noqa: PLC0415
    import numpy as np                                   # noqa: PLC0415

    X = np.vstack([ex if isinstance(ex, np.ndarray) else np.array(ex) for ex in X_all])
    y = np.concatenate([np.array(yi) for yi in y_all])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if model_type == "gbm":
        # Prefer HistGradientBoostingClassifier (10-50× faster on large datasets)
        # Falls back to GradientBoostingClassifier, then LR.
        _n_neg = int((y == 0).sum())
        _n_pos = int((y == 1).sum())
        _w_pos = max(_n_neg / max(_n_pos, 1), 1.0)
        print(f"  GBM: {_n_pos} pos / {_n_neg} neg, w_pos={_w_pos:.1f}")
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
            _sample_weight = np.where(y == 1, _w_pos, 1.0)
            model = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=10, random_state=42,
                early_stopping=False,
            )
            model.fit(X_s, y, sample_weight=_sample_weight)
            return model, scaler
        except ImportError:
            pass
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415
            _sample_weight = np.where(y == 1, _w_pos, 1.0)
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
            )
            model.fit(X_s, y, sample_weight=_sample_weight)
            return model, scaler
        except ImportError:
            print("  GBM unavailable, falling back to LR")
            model_type = "lr"
    if model_type == "lr":
        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, random_state=42
        )
    model.fit(X_s, y)
    return model, scaler


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Run recall_ablation.py --reingest first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  TRAIN LEARNED RERANKER")
    print(f"  DB: {DB_PATH}")
    print(f"  RRF_K_COS={_RRF_K_COS}  RRF_K_BM25={_RRF_K_BM25}  POOL={POOL_SIZE}  BROAD_POOL={BROAD_POOL_N}  DERIVED={_USE_DERIVED_BM25}  ATOMIC={_INCLUDE_ATOMIC}")
    print(f"  Model type: {args.model_type}")
    print(f"{'='*60}")

    with open(DATA_CACHE, encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = list(samples.values())
    print(f"\nLoaded {len(samples)} conversations")

    # ── Feature cache (skip re-collection if features already computed) ────────
    _feat_cache_key = (
        f"featcache_H_pool{POOL_SIZE}_broad{BROAD_POOL_N}"
        f"_rrf{_RRF_K}_derived{int(_USE_DERIVED_BM25)}_nfeat{_rr.N_FEATURES}.pkl"
    )
    _feat_cache_path = os.path.join(_PREFLIGHT_DIR, _feat_cache_key)
    if os.path.exists(_feat_cache_path) and not getattr(args, "no_cache", False):
        print(f"  Loading cached features from {_feat_cache_key}", flush=True)
        import pickle as _pkl
        with open(_feat_cache_path, "rb") as _fc:
            conv_examples = _pkl.load(_fc)
        print(f"  Loaded {sum(len(c) for c in conv_examples)} examples from cache")
        all_examples = [ex for conv in conv_examples for ex in conv]
        pos = sum(sum(ex["y"]) for ex in all_examples)
        print(f"  Positives: {pos}  Negatives: {sum(len(ex['y']) for ex in all_examples) - pos}")
    else:
        conn = sqlite3.connect(DB_PATH)
        dia_id_map = _build_dia_id_map(samples, conn)

        # ── Collect features per conversation ──────────────────────────────────
        print("\nCollecting features...", flush=True)
        t0 = time.time()
        conv_examples: list[list[dict]] = []  # one list per conversation

        for si, sample in enumerate(samples):
            sid_str = str(sample.get("sample_id", si))
            pid     = f"locomo_{sid_str}"
            pid_map = dia_id_map.get(pid, {})
            print(f"  Conv {si+1}/{len(samples)}: {pid}", flush=True)

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
                conv_examples.append([])
                continue

            qa_list = list(iter_qa(sample))
            q_embs  = _ub([qa["question"] for qa in qa_list])

            exs = collect_features_for_conversation(
                sample, pid_map, pid, conn, fact_cache, q_embs, qa_list
            )
            conv_examples.append(exs)
            print(f"    {len(exs)} labelled examples")

        conn.close()
        print(f"\n  Feature collection done in {time.time()-t0:.1f}s")

        all_examples = [ex for conv in conv_examples for ex in conv]
        print(f"  Total examples: {len(all_examples)}")
        pos = sum(sum(ex["y"]) for ex in all_examples)
        print(f"  Positives: {pos}  Negatives: {sum(len(ex['y']) for ex in all_examples) - pos}")

        # Save feature cache for faster re-runs
        import pickle as _pkl
        with open(_feat_cache_path, "wb") as _fc:
            _pkl.dump(conv_examples, _fc, protocol=4)
        print(f"  Feature cache saved to {_feat_cache_key}", flush=True)

    # ── RRF baseline ──────────────────────────────────────────────────────────
    _report_rrf_baseline(all_examples)

    # ── LOOCV ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Leave-one-conversation-out cross-validation")
    print(f"{'─'*60}")
    loocv_results: list[dict] = []
    for held_out in range(len(samples)):
        train_exs = [ex for ci, conv in enumerate(conv_examples) for ex in conv
                     if ci != held_out]
        test_exs  = conv_examples[held_out]
        if not train_exs or not test_exs:
            continue

        X_all = [row for ex in train_exs for row in ex["X"]]  # noqa: F841 (kept for debugging)
        y_all = [label for ex in train_exs for label in ex["y"]]  # noqa: F841
        model, scaler = _fit_model(
            [ex["X"] for ex in train_exs],
            [ex["y"] for ex in train_exs],
            args.model_type,
        )
        sid_str = str(samples[held_out].get("sample_id", held_out))
        metrics = _report_metrics(f"LOOCV fold {held_out+1} (held out: {sid_str})",
                                  test_exs, model, scaler, alpha=args.alpha)
        loocv_results.append(metrics)

    # Aggregate LOOCV
    print(f"\n  LOOCV aggregate:")
    for k in _RECALL_KS:
        avg = sum(r[k] for r in loocv_results) / len(loocv_results)
        print(f"    R@{k:<3}: {avg:.2f}%")

    # ── Train final model on all data ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Training final model on ALL conversations...")
    model_final, scaler_final = _fit_model(
        [ex["X"] for ex in all_examples],
        [ex["y"] for ex in all_examples],
        args.model_type,
    )
    _report_metrics("Final model (train-set check)", all_examples, model_final, scaler_final,
                    alpha=args.alpha)

    # Save model
    _rr.save_model(model_final, scaler_final, _PREFLIGHT_DIR)

    # Save LR feature weights if applicable
    try:
        import numpy as np
        coef = model_final.coef_[0]
        print("\n  Feature weights (logistic regression coef):")
        pairs = sorted(zip(_rr.FEATURE_NAMES, coef), key=lambda x: abs(x[1]), reverse=True)
        for name, w in pairs:
            print(f"    {name:<25} {w:+.4f}")
    except AttributeError:
        pass  # GBM

    # Save metadata
    meta = {
        "db_letter":      args.db_letter,
        "pool_size":      POOL_SIZE,
        "broad_pool_n":   BROAD_POOL_N,
        "model_type":     args.model_type,
        "include_atomic": _INCLUDE_ATOMIC,
        "rrf_k":          _RRF_K,
        "use_derived":    _USE_DERIVED_BM25,
        "n_train":        len(all_examples),
        "features":       _rr.FEATURE_NAMES,
        "n_features":     _rr.N_FEATURES,
        "loocv_avg": {
            k: round(sum(r[k] for r in loocv_results) / len(loocv_results), 2)
            for k in _RECALL_KS
        },
    }
    meta_path = os.path.join(_PREFLIGHT_DIR, "reranker_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved -> {meta_path}")


if __name__ == "__main__":
    main()
