"""Learned feature reranker for LoCoMo retrieval.

A lightweight pointwise ranker trained on retrieval signals already computed
by eval_locomo.py.  It replaces nothing in the pipeline — it just re-sorts a
candidate pool using a logistic regression model whose input features are the
raw signal scores/ranks from cosine, BM25, derived BM25, and atomic rerank.

Public API
----------
extract_features(pool_fids, rrf_scores, cos_scores, bm25_ranks,
                 derived_ranks, atomic_boosts, n_facts, category,
                 question_tokens, content_by_fid)
    -> np.ndarray  shape (len(pool_fids), N_FEATURES)

rerank_pool(pool_fids, feature_matrix, model, scaler)
    -> list[int]   same fids, re-sorted by predicted relevance

load_model(model_dir)  -> (model, scaler)
save_model(model, scaler, model_dir)
N_FEATURES           -> int
FEATURE_NAMES        -> list[str]
"""
from __future__ import annotations
import os
import pickle
import re as _re_feat
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import StandardScaler

_MODEL_FILE  = "reranker_model.pkl"
_SCALER_FILE = "reranker_scaler.pkl"

FEATURE_NAMES: list[str] = [
    "rrf_score",            # raw RRF score
    "rrf_rank_norm",        # 0-based rrf rank / pool_size
    "cos_score",            # cosine similarity
    "cos_rank_norm",        # 0-based cos rank / n_facts
    "bm25_rank_norm",       # 0-based bm25 rank / n_facts; 1.0 if not retrieved
    "bm25_found",           # 1 if BM25 returned this fact
    "derived_rank_norm",    # 0-based derived rank / n_facts; 1.0 if not retrieved
    "derived_found",        # 1 if derived BM25 returned this fact
    "atomic_boost",         # max atomic cosine sim (0 if no siblings)
    "has_atomic",           # 1 if any llm_atomic sibling exists
    "rrf_bm25_derived_sum", # cos_rank_norm + bm25_rank_norm + derived_rank_norm (smaller=better)
    "is_temporal",          # 1 if temporal question (category==3)
    "is_multihop",          # 1 if multi-hop question (category==2)
    # v2: content-based + question-type inference features
    "token_overlap",        # fraction of q-tokens found in fact [curr] text
    "speaker_in_q",         # 1 if speaker name appears in question text
    "n_signals",            # # retrieval signals that found this fact (0-3)
    "is_temporal_q",        # 1 if question has temporal keywords (inferred, no gold label)
    "is_multihop_q",        # 1 if question has multi-hop indicators (inferred)
    # v3: lexical explicit-memory channel features
    "name_token_hit_count", # name tokens from Q found in fact content
    "date_token_hit_count", # date/year tokens from Q found in fact content
    "bigram_hit_count",     # Q bigrams found in fact content
]
N_FEATURES = len(FEATURE_NAMES)

# Question-type keyword sets for automatic inference (no gold labels needed)
_TEMPORAL_Q_WORDS = frozenset({
    "when", "before", "after", "first", "last", "during",
    "since", "until", "ago", "recently", "year", "month",
    "date", "time", "latest", "earliest", "session",
})
_MULTIHOP_Q_WORDS = frozenset({
    "both", "also", "besides", "another", "other",
    "additionally", "apart", "else",
})
_WS_RE = _re_feat.compile(r'\w+')

# ── Lexical hit-count helpers (v3 features) ──────────────────────────────────
_STOPNAME = frozenset({
    'The', 'What', 'Who', 'When', 'Where', 'Why', 'How',
    'Did', 'Does', 'Was', 'Were', 'Will', 'Would', 'Could', 'Should',
})
_MONTH_WORDS = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
_DATE_RE = _re_feat.compile(r'\b' + _MONTH_WORDS + r'\s+\d{4}\b')
_YEAR_RE = _re_feat.compile(r'\b\d{4}\b')


def _get_lexical_question_features(question: str) -> tuple[list[str], list[str], list[str]]:
    """Extract name tokens, date/year tokens, and bigrams from question text."""
    # Name tokens: capitalized words not in STOPNAME
    name_toks = [w for w in _re_feat.findall(r'\b[A-Z][a-z]{2,}\b', question)
                 if w not in _STOPNAME]
    # Date/year tokens
    date_toks = list({m.group().lower() for m in _DATE_RE.finditer(question)})
    year_toks = list({m.group() for m in _YEAR_RE.finditer(question)
                      if not date_toks or m.group() not in date_toks[0]})
    # Bigrams: adjacent non-stopword pairs
    words = [w.lower() for w in _WS_RE.findall(question) if len(w) > 2]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return name_toks, date_toks + year_toks, bigrams


def _name_hit_count(fact_content: str, name_toks: list[str]) -> int:
    return sum(fact_content.count(t) for t in name_toks)


def _date_hit_count(fact_content: str, date_toks: list[str]) -> int:
    c = fact_content.lower()
    return sum(c.count(d) for d in date_toks)


def _bigram_hit_count(fact_content: str, bigrams: list[str]) -> int:
    if not bigrams:
        return 0
    c = fact_content.lower()
    return sum(1 for bg in bigrams if bg in c)


def extract_features(
    pool_fids:       list[int],
    rrf_scores:      dict[int, float],   # fid -> rrf score
    cos_scores:      dict[int, float],   # fid -> cosine similarity (pre-computed)
    cos_ranks:       dict[int, int],     # fid -> 0-based cosine rank in full fact_cache
    bm25_ranks:      dict[int, int],     # fid -> 0-based BM25 rank (omit if not retrieved)
    derived_ranks:   dict[int, int],     # fid -> 0-based derived BM25 rank
    atomic_boosts:   dict[int, float],   # fid -> max atomic cosine sim (0 if none)
    n_facts:         int,
    category:        int,                # 1=single_hop 2=multi_hop 3=temporal 4=open_domain
    question_tokens: frozenset = frozenset(),  # pre-tokenized lowercased question words
    content_by_fid:  dict | None = None,       # fid -> full content string (for overlap)
    question:        str = "",                 # raw question text for lexical feature extraction
) -> np.ndarray:
    """Return feature matrix of shape (len(pool_fids), N_FEATURES)."""
    pool_size = max(len(pool_fids), 1)
    # RRF ranks: sort pool by rrf_score to get pool-relative ranks
    pool_rrf_order = sorted(range(len(pool_fids)),
                            key=lambda i: rrf_scores.get(pool_fids[i], 0.0),
                            reverse=True)
    rrf_rank_in_pool = {pool_fids[i]: r for r, i in enumerate(pool_rrf_order)}

    # Question-type inference (same for every fid in this pool call)
    is_temporal_q = 1.0 if question_tokens & _TEMPORAL_Q_WORDS else 0.0
    is_multihop_q = 1.0 if question_tokens & _MULTIHOP_Q_WORDS else 0.0

    # Pre-compute lexical question features (v3)
    _name_toks, _date_toks, _bigrams = _get_lexical_question_features(question)

    rows = []
    for fid in pool_fids:
        rrf_score      = rrf_scores.get(fid, 0.0)
        rrf_rank_n     = rrf_rank_in_pool.get(fid, pool_size) / pool_size
        cos_score      = cos_scores.get(fid, 0.0)
        cos_rank_n     = cos_ranks.get(fid, n_facts) / max(n_facts, 1)
        bm25_found     = 1.0 if fid in bm25_ranks else 0.0
        bm25_rank_n    = bm25_ranks[fid] / max(n_facts, 1) if fid in bm25_ranks else 1.0
        derived_found  = 1.0 if fid in derived_ranks else 0.0
        derived_rank_n = derived_ranks[fid] / max(n_facts, 1) if fid in derived_ranks else 1.0
        atomic_boost   = atomic_boosts.get(fid, 0.0)
        has_atomic     = 1.0 if fid in atomic_boosts and atomic_boosts[fid] > 0.0 else 0.0
        rank_sum       = cos_rank_n + bm25_rank_n + derived_rank_n
        is_temporal    = 1.0 if category == 3 else 0.0
        is_multihop    = 1.0 if category == 2 else 0.0

        # ── Content-based features (v2) ─────────────────────────────────────────
        content = (content_by_fid or {}).get(fid, "")
        curr_text    = ""
        speaker_name = ""
        if content:
            for ln in content.split("\n"):
                if ln.startswith("[curr] "):
                    curr_text = ln[len("[curr] "):]
                    colon_idx = curr_text.find(":")
                    if colon_idx > 0:
                        speaker_name = curr_text[:colon_idx].strip()
                    break
        # Token overlap: fraction of question tokens present in [curr] text
        if curr_text and question_tokens:
            curr_tokens = frozenset(
                w.lower() for w in _WS_RE.findall(curr_text) if len(w) > 2
            )
            tok_overlap = len(question_tokens & curr_tokens) / max(len(question_tokens), 1)
        else:
            tok_overlap = 0.0
        # Speaker in question: speaker name (before colon) appears in question
        if speaker_name and question_tokens:
            spk_in_q = 1.0 if speaker_name.lower() in question_tokens else 0.0
        else:
            spk_in_q = 0.0
        # Signal agreement count: bm25 found + derived found + cos in top-25% of n_facts
        n_sigs = bm25_found + derived_found + (1.0 if cos_rank_n < 0.25 else 0.0)

        # v3 lexical features
        _nh = _name_hit_count(content, _name_toks)
        _dh = _date_hit_count(content, _date_toks)
        _bh = _bigram_hit_count(content, _bigrams)

        rows.append([
            rrf_score, rrf_rank_n, cos_score, cos_rank_n,
            bm25_rank_n, bm25_found, derived_rank_n, derived_found,
            atomic_boost, has_atomic, rank_sum,
            is_temporal, is_multihop,
            # v2 features
            tok_overlap, spk_in_q, n_sigs, is_temporal_q, is_multihop_q,
            # v3 features
            _nh, _dh, _bh,
        ])
    return np.array(rows, dtype=np.float32)


def rerank_pool(
    pool_fids:      list[int],
    feature_matrix: np.ndarray,
    model,
    scaler,
    rrf_scores:     dict[int, float] | None = None,
    alpha:          float = 0.0,
) -> list[int]:
    """Re-sort pool_fids by predicted relevance probability. Returns same fids.

    If ``rrf_scores`` and ``alpha > 0`` are supplied, the final sort key is a
    soft blend:
        score = rrf_score_norm + alpha * model_prob
    where rrf_score_norm is linearly scaled to [0, 1] across the pool.
    This preserves most of the RRF ordering while nudging confident
    predictions upward — avoids hard R@10 regressions caused by a
    pure probability sort.
    """
    if model is None or len(pool_fids) == 0:
        return pool_fids
    X = scaler.transform(feature_matrix) if scaler is not None else feature_matrix
    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        probs = model.predict(X)

    if alpha > 0.0 and rrf_scores is not None:
        raw = np.array([rrf_scores.get(fid, 0.0) for fid in pool_fids], dtype=np.float64)
        lo, hi = raw.min(), raw.max()
        rrf_norm = (raw - lo) / (hi - lo + 1e-12)
        scores = rrf_norm + alpha * probs
    else:
        scores = probs

    order = np.argsort(scores)[::-1]
    return [pool_fids[i] for i in order]


def load_model(model_dir: str):
    """Return (model, scaler) loaded from model_dir, or (None, None) if missing."""
    mp = os.path.join(model_dir, _MODEL_FILE)
    sp = os.path.join(model_dir, _SCALER_FILE)
    if not os.path.exists(mp):
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    # Validate feature count: stale models trained with a different N_FEATURES
    # will silently produce wrong predictions — reject them so the caller sees None.
    n_feat = getattr(model, "n_features_in_", None)
    if n_feat is not None and n_feat != N_FEATURES:
        print(f"  [reranker] WARNING: saved model has {n_feat} features, "
              f"current N_FEATURES={N_FEATURES}. Run train_reranker.py to retrain.")
        return None, None
    scaler = None
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler


def save_model(model, scaler, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, _MODEL_FILE), "wb") as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(os.path.join(model_dir, _SCALER_FILE), "wb") as f:
            pickle.dump(scaler, f)
    print(f"  Model saved -> {model_dir}/{_MODEL_FILE}")
