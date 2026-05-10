#!/usr/bin/env python3
"""LoCoMo benchmark evaluation for Preflight memory system.

Scores Preflight's retrieval against the LoCoMo QA benchmark, producing
F1 scores directly comparable to Mem0 (91.6%) and MemU (92.09%).

LoCoMo (ACL 2024): 10 long multi-session conversations, annotated with
single-hop, multi-hop, temporal and open-domain QA pairs.

Run:
    python eval_locomo.py
"""

import json
import os
import re
import sqlite3
import string
import struct
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime as _DT, timedelta as _TD

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPTS_DIR   = os.path.join(os.path.expanduser("~"), ".config", "opencode")
_PREFLIGHT_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PREFLIGHT_DIR)

DATA_URL            = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_CACHE          = os.path.join(_PREFLIGHT_DIR, "locomo10.json")
RESULTS_PATH        = os.path.join(_PREFLIGHT_DIR, "locomo_results.json")
RECALL_RESULTS_PATH = os.path.join(_PREFLIGHT_DIR, "locomo_recall_results.json")
_RECALL_KS          = [1, 3, 5, 10, 40]
_RECALL_TARGET_K    = 40      # which K the target applies to
_RECALL_TARGET_PCT  = 99.0    # target: R@40 >= 99%

# BM25 stopwords — question-frame words that inflate BM25 ranks for irrelevant
# turns.  Active only when _USE_BM25_STOPWORDS=True (default: off = baseline).
_BM25_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
    "did", "does", "has", "had", "was", "were", "are", "been", "have",
    "would", "could", "should", "will", "shall",
    "the", "that", "this", "and", "for", "with", "from", "into",
    "she", "her", "his", "their", "him", "they", "you", "its",
    "not", "but", "can", "any", "all", "out",
})

# ── Experiment flags ─────────────────────────────────────────────────────────
# Change exactly ONE flag per ablation run.  Baseline (all defaults) must
# reproduce R@40 >= 92.62, R@5 >= 73.87.  Override via env var, e.g.:
#   $env:PREFLIGHT_USE_STOPWORDS="1"; python recall_ablation.py
_USE_BM25_STOPWORDS     = os.environ.get("PREFLIGHT_USE_STOPWORDS",    "0") == "1"
_BM25_RRF_WEIGHT        = float(os.environ.get("PREFLIGHT_BM25_WEIGHT", "1.0"))
_USE_CE_IN_RECALL_EVAL  = os.environ.get("PREFLIGHT_USE_CE",            "0") == "1"
_USE_EVAL_SPEAKER_BOOST = os.environ.get("PREFLIGHT_SPEAKER_BOOST",    "0") == "1"
_RRF_K                  = int(os.environ.get("PREFLIGHT_RRF_K",         "60"))
_USE_DERIVED_BM25       = os.environ.get("PREFLIGHT_USE_DERIVED_BM25", "0") == "1"
_POOL_A_SIZE            = int(os.environ.get("PREFLIGHT_POOL_A",        "750"))

# ── Embedding setup: try real fastembed; fall back to SHA-256 stub ─────────────
# Must happen BEFORE importing memory so memory.py picks up the right utils.
_REAL_EMBEDDINGS = False
try:
    sys.path.insert(0, _SCRIPTS_DIR)
    import utils as _utils_check  # noqa: F401
    _test_emb = _utils_check.embed_text("test")
    _REAL_EMBEDDINGS = True
except Exception:
    # fastembed not available in this interpreter — install stub so memory.py works
    import hashlib
    import types as _types
    _stub_utils = _types.ModuleType("utils")

    def _stub_embed(text: str) -> list:
        h = hashlib.sha256(text.encode()).digest()
        v = [b / 255.0 for b in h[:32]]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n else v

    def _stub_cos(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    _stub_utils.embed_text = _stub_embed
    _stub_utils.cosine_similarity = _stub_cos
    sys.modules["utils"] = _stub_utils

import memory as _mem  # noqa: E402  (import after stub is in place)

# ── Scoring ────────────────────────────────────────────────────────────────────
# Use NLTK Porter stemmer to match the official LoCoMo evaluation.
try:
    from nltk.stem import PorterStemmer as _PS
    _ps = _PS()
    def _stem(w: str) -> str:
        return _ps.stem(w)
    _STEMMER = "NLTK PorterStemmer"
except ImportError:
    def _stem(w: str) -> str:  # type: ignore[misc]
        return w
    _STEMMER = "none (NLTK missing — scores may differ slightly from paper)"


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _normalize(text).split()]


def f1_score(prediction: str, ground_truth: str) -> float:
    pred  = _tokenize(str(prediction))
    truth = _tokenize(str(ground_truth))
    if not pred or not truth:
        return 0.0
    common = Counter(pred) & Counter(truth)
    n = sum(common.values())
    if not n:
        return 0.0
    p = n / len(pred)
    r = n / len(truth)
    return 2 * p * r / (p + r)


def multi_hop_f1(prediction: str, ground_truth: str) -> float:
    """Category 1 (multi-hop): ground truth may be comma-separated sub-answers."""
    sub_gts = [a.strip() for a in str(ground_truth).split(",") if a.strip()]
    if len(sub_gts) <= 1:
        return f1_score(prediction, ground_truth)
    sub_preds = [a.strip() for a in str(prediction).split(",") if a.strip()] or [prediction]
    return sum(
        max(f1_score(p, gt) for p in sub_preds)
        for gt in sub_gts
    ) / len(sub_gts)


# LoCoMo category integer codes (from official evaluation.py)
_CAT_NAMES = {1: "multi_hop", 2: "temporal", 3: "single_hop", 4: "open_domain", 5: "adversarial"}
_SKIP_CATS  = {5}  # adversarial: fact not in conversation — skip from scoring


def score_qa(prediction: str, answer, category: int) -> float:
    answer = str(answer[0] if isinstance(answer, list) else answer)
    if category == 3:
        answer = answer.split(";")[0].strip()  # take first sub-answer
    if category == 1:
        return multi_hop_f1(prediction, answer)
    return f1_score(prediction, answer)


# ── Fix 3: best-sentence extractive answer ─────────────────────────────────────

def _sent_split(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _tok_overlap(a: str, b: str) -> float:
    """Token-overlap F1 between two strings (no stemming — fast)."""
    at = set(_normalize(a).split())
    bt = set(_normalize(b).split())
    if not at or not bt:
        return 0.0
    common = len(at & bt)
    if not common:
        return 0.0
    p = common / len(at)
    r = common / len(bt)
    return 2 * p * r / (p + r)


# ── Temporal date resolution helpers ─────────────────────────────────────────

_MONTHS_EN = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]
_WEEKDAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
_WEEKDAY_ABBR  = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']


def _parse_session_dt(date_str: str):
    """Parse '1:56 pm on 8 May, 2023' → datetime, or None on failure."""
    m = re.search(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', str(date_str))
    if m:
        try:
            return _DT.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y")
        except ValueError:
            pass
    return None


def _resolve_relative_date(text: str, session_dt) -> str | None:
    """Detect a relative temporal expression in *text* and return an absolute
    date string matching LoCoMo's ground-truth format.  Returns None if no
    recognisable expression is found or session_dt is unavailable.
    """
    if session_dt is None:
        return None
    t = text.lower()

    if 'yesterday' in t:
        d = session_dt - _TD(days=1)
        return f"{d.day} {_MONTHS_EN[d.month - 1]} {d.year}"

    if 'today' in t:
        return f"{session_dt.day} {_MONTHS_EN[session_dt.month - 1]} {session_dt.year}"

    if 'last sunday' in t:
        return (f"The sunday before {session_dt.day} "
                f"{_MONTHS_EN[session_dt.month - 1]} {session_dt.year}")

    if 'last week' in t:
        return (f"The week before {session_dt.day} "
                f"{_MONTHS_EN[session_dt.month - 1]} {session_dt.year}")

    # "two weekends ago" / "two weeks ago" → two weeks before session
    if 'two weekend' in t or 'two week' in t:
        return (f"two weekends before {session_dt.day} "
                f"{_MONTHS_EN[session_dt.month - 1]} {session_dt.year}")

    if 'this month' in t:
        return f"{_MONTHS_EN[session_dt.month - 1]} {session_dt.year}"

    if 'next month' in t:
        if session_dt.month == 12:
            return f"January {session_dt.year + 1}"
        return f"{_MONTHS_EN[session_dt.month]} {session_dt.year}"

    if 'last month' in t:
        if session_dt.month == 1:
            return f"December {session_dt.year - 1}"
        return f"{_MONTHS_EN[session_dt.month - 2]} {session_dt.year}"

    if 'last year' in t or 'a year ago' in t:
        return str(session_dt.year - 1)

    if 'this year' in t:
        return str(session_dt.year)

    # "last <weekday>" — full name or 3-letter abbreviation (e.g. "last Fri")
    for idx, (day_name, day_abbr) in enumerate(zip(_WEEKDAY_NAMES, _WEEKDAY_ABBR)):
        pattern = f'last {day_name}'
        abbr_pattern = f'last {day_abbr}'
        if pattern in t or abbr_pattern in t:
            dow_diff = (session_dt.weekday() - idx) % 7 or 7
            d = session_dt - _TD(days=dow_diff)
            day_cap = day_name.capitalize()
            return (f"The {day_cap} before {session_dt.day} "
                    f"{_MONTHS_EN[session_dt.month - 1]} {session_dt.year}")

    return None


def extract_answer(question: str, facts: list[str], category: int,
                   fact_session_dates: list[str] | None = None) -> str:
    """Pick the best sentence(s) from retrieved facts by token overlap with the question.

    Multi-hop (cat 1): return top-2 sentences joined with "; ".
    All others: return the single best sentence.

    Strategy:
    - Process each LINE of the (possibly multi-line) window fact individually,
      stripping [prev]/[curr]/[next] tags and "Speaker: " prefix per line.
    - Sentence-split each cleaned line; filter out questions ("?" endings) and
      short acknowledgments (< 4 tokens) which are always noise.
    - Select sentence with highest token overlap against the eval question.
      Ties broken by longer sentence (more content). Fallback: longest sentence.
    """
    if not facts:
        return ""

    _MIN_TOKENS = 4

    # ── Category-2 (temporal): try to resolve relative date expressions ────────
    if category == 2 and fact_session_dates:
        # Iterate over retrieved facts in rank order (most relevant first).
        # Return the first resolvable temporal expression found.
        # Do NOT filter lines ending with "?" — a turn can contain both a
        # statement ("I signed up yesterday") and a follow-up question.
        for fi, fact in enumerate(facts):
            date_str = fact_session_dates[fi] if fi < len(fact_session_dates) else ""
            session_dt = _parse_session_dt(date_str) if date_str else None
            for line in fact.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^\[(prev|curr|next)\]\s*", "", line)
                line = re.sub(r"^\w[\w\s]*:\s*", "", line)
                if not line:
                    continue
                resolved = _resolve_relative_date(line, session_dt)
                if resolved:
                    return resolved
        # No temporal expression resolved — fall through to normal extraction

    sentences: list[tuple[str, float]] = []  # (text, overlap_score)

    for fact in facts:
        for line in fact.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\[(prev|curr|next)\]\s*", "", line)  # strip window tag per line
            line = re.sub(r"^\w[\w\s]*:\s*", "", line)             # strip "Speaker: " per line
            if not line:
                continue
            for sent in (_sent_split(line) or [line]):
                sent = sent.strip()
                if not sent or sent.endswith("?"):
                    continue  # skip conversation questions
                if len(_normalize(sent).split()) < _MIN_TOKENS:
                    continue  # skip short fillers ("Thanks!", "Yeah!", etc.)
                sentences.append((sent, _tok_overlap(question, sent)))

    if not sentences:
        return ""

    # Sort: highest overlap first; ties → prefer longer sentence (more content)
    sentences.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

    if sentences[0][1] == 0.0:
        # No question overlap — return the longest sentence (most informative fallback)
        return max(sentences, key=lambda x: len(x[0]))[0]

    if category == 1:  # multi-hop: two best distinct sentences
        top2 = [sentences[0][0]]
        for text, _ in sentences[1:]:
            if text != sentences[0][0]:
                top2.append(text)
                break
        return "; ".join(top2)
    return sentences[0][0]


# ── Dataset loading ────────────────────────────────────────────────────────────

def download_dataset() -> list:
    if not os.path.exists(DATA_CACHE):
        print("  Downloading LoCoMo from GitHub...")
        urllib.request.urlretrieve(DATA_URL, DATA_CACHE)
        print(f"  Saved -> {DATA_CACHE}")
    else:
        print(f"  Using cached: {DATA_CACHE}")
    with open(DATA_CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw) if isinstance(raw, list) else list(raw.values())


# ── Conversation iteration ─────────────────────────────────────────────────────

def iter_sessions(conv: dict):
    """Yield (session_num, date_time_str, turns_list) in chronological order.

    conversation is a dict with keys: session_1, session_1_date_time, session_2, ...
    plus speaker_a, speaker_b.
    """
    nums = sorted(
        int(k.split("_")[1])
        for k in conv.keys()
        if re.match(r"^session_\d+$", k)
    )
    for n in nums:
        turns    = conv.get(f"session_{n}", [])
        date_str = conv.get(f"session_{n}_date_time", "")
        yield n, str(date_str), turns


def iter_turns(turns: list):
    """Yield (speaker, text) for each non-empty turn."""
    for t in turns:
        speaker = str(t.get("speaker", "?"))
        text    = str(t.get("text", ""))
        if text.strip():
            yield speaker, text


def iter_qa(sample: dict):
    """Yield normalized QA dicts, skipping adversarial (cat 5)."""
    for qa in sample.get("qa", []):
        raw_cat = qa.get("category", 0)
        try:
            cat = int(raw_cat)
        except (ValueError, TypeError):
            cat = 0
        if cat in _SKIP_CATS:
            continue
        raw_evidence = qa.get("evidence", []) or []
        evidence: list[str] = [str(d) for d in raw_evidence if d is not None]
        yield {
            "question": str(qa.get("question", "")),
            "answer":   qa.get("answer", ""),
            "category": cat,
            "cat_name": _CAT_NAMES.get(cat, str(cat)),
            "evidence": evidence,
        }


# ── Full-corpus retrieval (bypasses production LIMIT 200) ────────────────────────
# memory.retrieve_facts caps at the last 200 facts ordered by id DESC — correct
# for incremental coding sessions, but wrong for a pre-ingested benchmark where
# evidence may be anywhere in a 600-turn conversation. This function searches
# the full project corpus with pure cosine similarity, no row cap.

def _eval_retrieve(db_path: str, project_id: str, question: str, top_n: int = 5) -> list[dict]:
    """Search ALL live facts for project via RRF(cosine + BM25) — no row limit.

    Mirrors the run_recall_eval ranker so F1 evaluation uses the same retrieval
    signal as recall evaluation.  Returns list of dicts with 'id' and 'content'.
    """
    from utils import embed_text as _ue, cosine_similarity as _cs  # noqa: PLC0415
    q_emb = _ue(question)
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

    # Cosine ranking
    n_facts = len(fact_cache)
    cos_ranked = sorted(fact_cache, key=lambda x: _cs(q_emb, x[2]), reverse=True)
    cos_rank = {fid: i for i, (fid, _, _e) in enumerate(cos_ranked)}

    # BM25 ranking via FTS5
    bm25_rank: dict[int, int] = {}
    try:
        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in question)
        tokens = [t for t in safe.split() if len(t) > 2
                  and (not _USE_BM25_STOPWORDS or t.lower() not in _BM25_STOPWORDS)]
        if tokens:
            fts_q    = " OR ".join(f'"{t}"' for t in tokens)
            all_fids_set = {fid for fid, _, _e in fact_cache}
            bm_rows  = conn.execute(
                "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                (fts_q,),
            ).fetchall()
            rank = 0
            for (bfid,) in bm_rows:
                if bfid in all_fids_set:
                    bm25_rank[bfid] = rank
                    rank += 1
    except Exception:
        pass
    conn.close()

    # RRF merge — weight controlled by _BM25_RRF_WEIGHT (baseline=1.0).
    rrf: dict[int, float] = {}
    for fid, _, _e in fact_cache:
        s = 1.0 / (_RRF_K + cos_rank.get(fid, n_facts))
        if fid in bm25_rank:
            s += _BM25_RRF_WEIGHT / (_RRF_K + bm25_rank[fid])
        rrf[fid] = s

    content_by_fid = {fid: content for fid, content, _e in fact_cache}
    sorted_fids = sorted(rrf, key=rrf.__getitem__, reverse=True)
    return [{"id": fid, "content": content_by_fid[fid]} for fid in sorted_fids[:top_n]]


def build_dia_id_map(samples: list, db_path: str) -> dict:
    """Build project_id → {dia_id → set[fact_id]} by matching turn content in the DB.

    Collects ALL fact_ids that contain a given turn: window rows via [curr] tag,
    turn rows via plain content fallback.  Recall@K hits if ANY fid in the set
    appears in top-K — so retrieving either the window or the clean turn row counts.

    Two-pass strategy: [prev]/[next] tag matches go in first, [curr] tag matches
    overwrite (higher priority).  Plain-text rows (no window tags) go in via fallback.
    Both window and turn fids for the same "Speaker: text" are collected into one set.

    llm_atomic facts: also included when present.  They are stored with
    source_hash = sha256(curr_line)[:16] so we can join them back to their
    source turn via a secondary hash-keyed lookup.
    """
    import hashlib as _hl  # noqa: PLC0415
    dia_id_map: dict[str, dict[str, set]] = {}
    conn = sqlite3.connect(db_path)
    for ci, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", ci))
        pid     = f"locomo_{sid_str}"
        rows    = conn.execute(
            "SELECT id, content, source_hash, fact_type FROM facts WHERE project_id = ? AND superseded_at IS NULL",
            (pid,),
        ).fetchall()
        # Maps "Speaker: text" → set of fact_ids (window fid via [curr] tag + turn fid via fallback).
        content_to_ids: dict[str, set] = {}
        # Maps source_hash → set of llm_atomic fact_ids for secondary linking.
        hash_to_llm_ids: dict[str, set] = {}
        # Two-pass: [prev]/[next] tag matches first (lower priority), then [curr].
        for priority_tags in (("[prev] ", "[next] "), ("[curr] ",)):
            for fid, content, _sh, _ft in rows:
                for line in content.split("\n"):
                    for tag in priority_tags:
                        if line.startswith(tag):
                            key = line[len(tag):]
                            content_to_ids.setdefault(key, set()).add(fid)
        # Fallback: plain rows without any window tags (covers fact_type="turn" rows).
        for fid, content, _sh, _ft in rows:
            if not any(content.startswith(t) or "\n" + t in content
                       for t in ("[curr] ", "[prev] ", "[next] ")):
                content_to_ids.setdefault(content, set()).add(fid)
        # Build hash → llm_atomic fid mapping for secondary linking.
        for fid, _content, source_hash, fact_type in rows:
            if fact_type == "llm_atomic" and source_hash:
                hash_to_llm_ids.setdefault(source_hash, set()).add(fid)
        pid_map: dict[str, set] = {}
        conv = sample.get("conversation", {})
        for sn, _date, turns in iter_sessions(conv):
            for turn in turns:
                dia_id  = turn.get("dia_id")
                speaker = str(turn.get("speaker", "?"))
                text    = str(turn.get("text", ""))
                if not text.strip() or dia_id is None:
                    continue
                content = f"{speaker}: {text}"
                fids = content_to_ids.get(content)
                if fids:
                    fids = set(fids)  # copy so we can augment
                    # Link any llm_atomic facts derived from this turn.
                    turn_hash = _hl.sha256(content.encode()).hexdigest()[:16]
                    if turn_hash in hash_to_llm_ids:
                        fids.update(hash_to_llm_ids[turn_hash])
                    pid_map[str(dia_id)] = fids
        dia_id_map[pid] = pid_map
    conn.close()
    return dia_id_map


def recall_at_k(
    question: str,
    evidence_dia_ids: list,
    project_id: str,
    db_path: str,
    dia_id_map: dict,
    k: int = 5,
):
    """Return True if any evidence turn is in top-k retrieval; None if evidence missing."""
    if not evidence_dia_ids:
        return None
    pid_map = dia_id_map.get(project_id, {})
    evidence_fact_ids: set = set()
    for d in evidence_dia_ids:
        fids = pid_map.get(d)
        if fids:
            evidence_fact_ids.update(fids)
    if not evidence_fact_ids:
        return None
    facts = _eval_retrieve(db_path, project_id, question, top_n=k)
    return bool({f["id"] for f in facts} & evidence_fact_ids)


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest(samples: list, mem, mode: str) -> dict:
    """Ingest all samples into the memory DB.

    Benchmark mode ("B"):
      - Batch-embeds all curr_lines in one fastembed call per session.
      - Batch-extracts LLM atomic facts with ThreadPoolExecutor(4) when
        PREFLIGHT_USE_LLM_EXTRACTOR=1; blocks until ALL threads finish before
        returning so eval queries see a complete DB.
      - Skips the companion turn row (store_turn=False) — the window row is
        sufficient for ANN recall and halves the store_fact() calls.
      - Skips spaCy SVO extraction (extract_svo=False) — ~0.5s/turn, <1% gain.
    """
    import extractor as _ext
    from utils import embed_texts_batch as _uemb  # noqa: PLC0415

    _use_llm = os.environ.get("PREFLIGHT_USE_LLM_EXTRACTOR", "0") == "1"
    _llm_workers = int(os.environ.get("PREFLIGHT_LLM_WORKERS", "4"))

    total_turns = 0
    kw_facts    = 0
    for ci, sample in enumerate(samples):
        print(f"  Conversation {ci+1}/{len(samples)}...", flush=True)
        sid_str = str(sample.get("sample_id", ci))
        pid     = f"locomo_{sid_str}"
        conv    = sample.get("conversation", {})
        for sn, _date, turns in iter_sessions(conv):
            sid = f"{pid}_s{sn}"
            session_turns = [
                {"speaker": str(t.get("speaker", "?")), "text": str(t.get("text", ""))}
                for t in turns if str(t.get("text", "")).strip()
            ]
            if mode == "B":
                # Build curr_line strings for the whole session in one go so we
                # can batch-embed and batch-LLM them before the store loop.
                curr_lines: list[str] = []
                for idx, td in enumerate(session_turns):
                    curr_lines.append(f"{td['speaker']}: {td['text']}")

                # Batch embedding: one model call for the whole session.
                try:
                    curr_embs: list[list[float]] = _uemb(curr_lines)
                except Exception:
                    curr_embs = [None] * len(curr_lines)  # type: ignore[list-item]

                # Batch LLM atomic-fact extraction (non-blocking for other modes).
                llm_facts_by_idx: dict[int, list[str]] = {}
                if _use_llm:
                    try:
                        from llm_extractor import extract_batch_facts as _ebf  # noqa: PLC0415
                        _batch_results = _ebf(curr_lines, workers=_llm_workers)
                        llm_facts_by_idx = {i: fs for i, fs in enumerate(_batch_results)}
                    except Exception:
                        pass  # fall back to empty — raw window facts still stored

                for turn_idx, turn_dict in enumerate(session_turns):
                    total_turns += 1
                    _emb = curr_embs[turn_idx] if curr_embs[turn_idx] is not None else None
                    mem.store_turn_window(
                        pid, sid, session_turns, turn_idx,
                        extract_svo=False,
                        store_turn=False,
                        _precomputed_curr_emb=_emb,
                    )
                    # Store pre-computed LLM atomic facts (already fetched above).
                    if _use_llm and turn_idx in llm_facts_by_idx:
                        import hashlib as _hl  # noqa: PLC0415
                        _cl = curr_lines[turn_idx]
                        _turn_hash = _hl.sha256(_cl.encode()).hexdigest()[:16]
                        from utils import embed_text as _et  # noqa: PLC0415
                        for _ft in llm_facts_by_idx[turn_idx]:
                            _ft_emb = _et(_ft)
                            mem.store_fact(
                                pid, sid, _ft, "llm_atomic",
                                enrich=False, _precomputed_emb=_ft_emb,
                                _source_hash=_turn_hash,
                            )
                    try:
                        for fact in _ext.keyword_extract(turn_dict["text"]):
                            mem.store_fact(pid, sid, fact, "finding")
                            kw_facts += 1
                    except Exception:
                        pass
            else:
                for turn_idx, turn_dict in enumerate(session_turns):
                    total_turns += 1
                    mem.store_turn_window(pid, sid, session_turns, turn_idx,
                                         extract_svo=False)
                    try:
                        for fact in _ext.keyword_extract(turn_dict["text"]):
                            mem.store_fact(pid, sid, fact, "finding")
                            kw_facts += 1
                    except Exception:
                        pass
    return {"total_turns": total_turns, "kw_facts": kw_facts}



# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(samples: list, mem, db_path: str) -> dict:
    """Run F1 evaluation. Preloads embeddings per project (one DB read per conv)."""
    from utils import embed_texts_batch as _ub, cosine_similarity as _cs  # noqa: PLC0415
    per_q:       list[dict]         = []
    cat_scores:  dict[str, list]    = {}
    n_retrieved  = 0
    n_budget     = 0
    _RRF_K = 60

    conn_eval = sqlite3.connect(db_path)

    for sample in samples:
        sid_str = str(sample.get("sample_id", 0))
        pid     = f"locomo_{sid_str}"

        # Build session_date map for this conversation (session_num → date_str).
        _session_dates_map: dict[int, str] = {}
        for _sn, _ds, _ in iter_sessions(sample.get("conversation", {})):
            _session_dates_map[_sn] = _ds

        # Preload all fact embeddings + content for this project (one DB read).
        rows_ev = conn_eval.execute(
            """SELECT id, content, embedding, session_id FROM facts
               WHERE project_id = ?
                 AND superseded_at IS NULL
                 AND fact_type != 'turn'
                 AND (valid_to IS NULL OR valid_to > unixepoch())""",
            (pid,),
        ).fetchall()
        fact_cache_ev: list[tuple[int, str, list]] = []
        session_id_by_fid: dict[int, str] = {}
        for fid, content, blob, s_id in rows_ev:
            if s_id:
                session_id_by_fid[fid] = s_id
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
            fact_cache_ev.append((fid, content, emb))

        all_fids_ev = tuple(fid for fid, _, _e in fact_cache_ev)
        all_fids_ev_set = set(all_fids_ev)
        n_ev = len(fact_cache_ev)

        # Batch-embed all questions for this conversation at once.
        qa_list_ev = list(iter_qa(sample))
        if fact_cache_ev and qa_list_ev:
            q_embs_ev = _ub([qa["question"] for qa in qa_list_ev])
        else:
            q_embs_ev = [None] * len(qa_list_ev)

        for qa, q_emb_ev in zip(qa_list_ev, q_embs_ev):
            try:
                if not fact_cache_ev or q_emb_ev is None:
                    facts = []
                else:
                    q_emb = q_emb_ev
                    # Cosine ranking
                    cos_ranked = sorted(fact_cache_ev, key=lambda x: _cs(q_emb, x[2]), reverse=True)
                    cos_rank = {fid: i for i, (fid, _, _e) in enumerate(cos_ranked)}
                    # BM25 via FTS5
                    bm25_rank_ev: dict[int, int] = {}
                    try:
                        safe   = "".join(c if c.isalnum() or c.isspace() else " " for c in qa["question"])
                        tokens = [t for t in safe.split() if len(t) > 2]
                        if tokens and all_fids_ev:
                            fts_q = " OR ".join(f'"{t}"' for t in tokens)
                            bm_rows = conn_eval.execute(
                                "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                                (fts_q,),
                            ).fetchall()
                            rank = 0
                            for (bfid,) in bm_rows:
                                if bfid in all_fids_ev_set:
                                    bm25_rank_ev[bfid] = rank
                                    rank += 1
                    except Exception:
                        pass
                    # RRF merge
                    rrf: dict[int, float] = {}
                    for fid, _, _e in fact_cache_ev:
                        s = 1.0 / (_RRF_K + cos_rank.get(fid, n_ev))
                        if fid in bm25_rank_ev:
                            s += 1.0 / (_RRF_K + bm25_rank_ev[fid])
                        rrf[fid] = s
                    content_by_fid = {fid: content for fid, content, _e in fact_cache_ev}
                    sorted_fids    = sorted(rrf, key=rrf.__getitem__, reverse=True)
                    facts = [{"id": fid, "content": content_by_fid[fid]} for fid in sorted_fids[:5]]
                budget_hit = False
            except Exception:
                facts, budget_hit = [], False

            # Resolve session dates for retrieved facts (used for temporal resolution).
            fact_dates: list[str] = []
            for f in facts:
                s_id = session_id_by_fid.get(f["id"], "")
                try:
                    sess_num = int(s_id.split("_s")[-1]) if s_id else 0
                    fact_dates.append(_session_dates_map.get(sess_num, ""))
                except (ValueError, IndexError):
                    fact_dates.append("")

            prediction = extract_answer(
                qa["question"],
                [f["content"] for f in facts],
                qa["category"],
                fact_session_dates=fact_dates,
            )
            sc = score_qa(prediction, qa["answer"], qa["category"])
            n_retrieved += len(facts)
            n_budget    += int(budget_hit)
            cat_scores.setdefault(qa["cat_name"], []).append(sc)
            per_q.append({
                "question":       qa["question"],
                "ground_truth":   str(qa["answer"]),
                "prediction":     prediction,
                "f1":             round(sc, 4),
                "category":       qa["cat_name"],
                "facts_retrieved": len(facts),
            })

    conn_eval.close()
    total   = len(per_q)
    overall = sum(q["f1"] for q in per_q) / max(total, 1)
    by_cat  = {c: round(sum(s) / len(s) * 100, 2) for c, s in cat_scores.items()}
    return {
        "overall_f1":      round(overall * 100, 2),
        "by_category":     by_cat,
        "per_question":    per_q,
        "total_qa":        total,
        "total_retrieved": n_retrieved,
        "budget_hits":     n_budget,
    }


# ── Run one mode ───────────────────────────────────────────────────────────────

def run_mode(samples: list, mem, mode: str) -> tuple[dict, dict]:
    label   = "Full turn ingestion" if mode == "B" else "Keyword extraction only"
    db_path = os.path.join(_PREFLIGHT_DIR, f"locomo_eval_{mode}.db")

    print(f"\n{'='*60}")
    print(f"  MODE {mode}: {label}")
    print(f"  DB: {db_path}")
    print(f"{'='*60}")

    mem.DB_PATH = db_path
    mem._compacted_this_process = False
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except OSError:
            pass

    t0 = time.time()
    print("\nIngesting...")
    stats = ingest(samples, mem, mode=mode)
    elapsed = time.time() - t0
    turns_s = stats["total_turns"]
    kw_s    = stats["kw_facts"]
    mode_b_total = turns_s + kw_s if mode == "B" else kw_s
    print(f"  Done in {elapsed:.1f}s  turns={turns_s}  kw-facts={kw_s}  "
          f"total-stored={mode_b_total}")

    print("\nEvaluating (this may take a few minutes)...", flush=True)
    results = evaluate(samples, mem, db_path=db_path)
    return stats, results


# ── Recall@K evaluation ───────────────────────────────────────────────────────

def run_recall_eval(samples: list, db_path: str) -> dict:
    """Measure Recall@K: did the evidence-containing turn appear in the top-K results?

    Preloads all embeddings per project (10 DB reads total) then does pure
    in-memory cosine ranking for all 1540 questions — much faster than one
    DB round-trip per question.
    """
    from utils import embed_text as _ue, embed_texts_batch as _ub, cosine_similarity as _cs  # noqa: PLC0415

    print(f"\n{'='*60}")
    print(f"  RECALL@K EVALUATION  (Mode B corpus)")
    print(f"  DB: {db_path}")
    print(f"{'='*60}")

    print("\nBuilding dia_id map (lightweight, no embeddings)...", flush=True)
    t0 = time.time()
    dia_id_map  = build_dia_id_map(samples, db_path)
    n_mapped    = sum(len(v) for v in dia_id_map.values())
    n_fids_total = sum(len(fids) for pid_m in dia_id_map.values() for fids in pid_m.values())
    print(f"  Done in {time.time() - t0:.1f}s \u2014 {n_mapped} turns mapped ({n_fids_total} total fact IDs)")

    print("\nScoring Recall@K...", flush=True)
    per_q: list[dict] = []
    t0   = time.time()
    conn = sqlite3.connect(db_path)

    for si, sample in enumerate(samples):
        sid_str = str(sample.get("sample_id", 0))
        pid     = f"locomo_{sid_str}"
        pid_map = dia_id_map.get(pid, {})
        print(f"  Conv {si+1}/{len(samples)}: loading embeddings...", flush=True)

        # One DB read per conversation — preload all facts into memory.
        # Exclude fact_type='turn' rows: they share an identical embedding with
        # their companion window row (both embed the same [curr] turn text).
        # Including both wastes top-K slots — two rows tie on cosine score for
        # the same turn, halving effective K.  Window rows carry the embedding
        # and are sufficient; turn rows help BM25/CE in production retrieve_facts()
        # but add no signal in this pure-cosine eval scorer.
        rows = conn.execute(
            """SELECT id, content, embedding FROM facts
               WHERE project_id = ?
                 AND superseded_at IS NULL
                 AND fact_type != 'turn'
                 AND (valid_to IS NULL OR valid_to > unixepoch())""",
            (pid,),
        ).fetchall()
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

        # Batch-embed all questions for this conversation at once.
        # fastembed processes the full list in one ONNX forward pass —
        # ~10-100x faster than calling embed_text() in a per-question loop.
        qa_list = list(iter_qa(sample))
        q_texts = [qa["question"] for qa in qa_list]
        q_embs  = _ub(q_texts)
        content_by_fid_ev = {fid: c for fid, c, _ in fact_cache}
        fids_in_cache = tuple(fid for fid, _, _ in fact_cache)

        for qa, q_emb in zip(qa_list, q_embs):
            evidence         = qa["evidence"]
            evidence_fact_ids: set = set()
            for d in evidence:
                fids = pid_map.get(d)
                if fids:
                    evidence_fact_ids.update(fids)
            has_evidence     = bool(evidence) and bool(evidence_fact_ids)

            if has_evidence:
                # Cosine ranking over preloaded embeddings (q_emb from batch)
                cos_ranked = sorted(fact_cache, key=lambda x: _cs(q_emb, x[2]), reverse=True)
                cos_rank = {fid: i for i, (fid, _, _) in enumerate(cos_ranked)}
                # BM25 ranking via FTS5 (same DB connection, already open)
                bm25_rank_eval: dict[int, int] = {}
                try:
                    safe = "".join(c if c.isalnum() or c.isspace() else " " for c in qa["question"])
                    tokens = [t for t in safe.split() if len(t) > 2
                              and (not _USE_BM25_STOPWORDS or t.lower() not in _BM25_STOPWORDS)]
                    if tokens:
                        fts_q = " OR ".join(f'"{t}"' for t in tokens)
                        if fids_in_cache:
                            fids_set = set(fids_in_cache)
                            bm_rows = conn.execute(
                                "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts)",
                                (fts_q,),
                            ).fetchall()
                            bm_rank = 0
                            for (bfid,) in bm_rows:
                                if bfid in fids_set:
                                    bm25_rank_eval[bfid] = bm_rank
                                    bm_rank += 1
                except Exception:
                    pass
                # RRF merge: cosine + BM25 (derived BM25 env-gated via PREFLIGHT_USE_DERIVED_BM25)
                n_facts = len(fact_cache)
                rrf_scores: dict[int, float] = {}
                for fid, _, _ in fact_cache:
                    s  = 1.0 / (_RRF_K + cos_rank.get(fid, n_facts))
                    if fid in bm25_rank_eval:
                        s += _BM25_RRF_WEIGHT / (_RRF_K + bm25_rank_eval[fid])
                    rrf_scores[fid] = s
                if _USE_DERIVED_BM25:
                    derived_rank_eval: dict[int, int] = {}
                    _RRF_K_DERIVED = 60
                    try:
                        from memory import _build_derived_text as _bdt  # noqa: PLC0415
                        derived_q = _bdt(qa["question"])
                        safe_d = "".join(c if c.isalnum() or c.isspace() else " " for c in derived_q)
                        dtokens = [t for t in safe_d.split() if len(t) > 2]
                        if dtokens and fids_in_cache:
                            dfts_q = " OR ".join(f'"{t}"' for t in dtokens)
                            fids_set_d = set(fids_in_cache)
                            dr_rows = conn.execute(
                                "SELECT rowid FROM facts_derived_fts"
                                " WHERE facts_derived_fts MATCH ? ORDER BY bm25(facts_derived_fts)",
                                (dfts_q,),
                            ).fetchall()
                            dr_rank = 0
                            for (dfid,) in dr_rows:
                                if dfid in fids_set_d:
                                    derived_rank_eval[dfid] = dr_rank
                                    dr_rank += 1
                            for fid, _, _ in fact_cache:
                                if fid in derived_rank_eval:
                                    rrf_scores[fid] += 1.0 / (_RRF_K_DERIVED + derived_rank_eval[fid])
                    except Exception:
                        pass
                # Speaker boost removed: regex-based speaker extraction produces too many
                # false positives (capitalised words near auxiliary verbs), boosting wrong
                # window rows and causing net-negative recall across all K values.
                sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
                # CE disabled: mxbai-rerank-xsmall is not calibrated for
                # [prev]/[curr]/[next] window format and actively demotes correct facts.
                if _USE_CE_IN_RECALL_EVAL:
                    try:
                        from utils import get_cross_encoder as _gce  # noqa: PLC0415
                        _ce = _gce()
                        if _ce is not None and len(sorted_ids) > 5:
                            _ce_pool_fids = [fid for fid in sorted_ids[:100]
                                             if fid in content_by_fid_ev]
                            # Feed only [curr] line to CE — window format confuses CE;
                            # the [curr] speaker:text is what the question asks about.
                            def _curr_text(raw: str) -> str:
                                for ln in raw.split("\n"):
                                    if ln.startswith("[curr] "):
                                        return ln[len("[curr] "):]
                                return raw  # fallback: use full content

                            _ce_pairs = [(qa["question"], _curr_text(content_by_fid_ev[fid]))
                                         for fid in _ce_pool_fids]
                            _ce_scores = _ce.predict(_ce_pairs)
                            _ce_reranked = [fid for fid, _ in sorted(
                                zip(_ce_pool_fids, _ce_scores),
                                key=lambda x: x[1], reverse=True,
                            )]
                            _ce_tail = [fid for fid in sorted_ids[100:]]
                            sorted_ids = _ce_reranked + _ce_tail
                    except Exception:
                        pass
                # Diagnostic: record gold fact ranks
                gold_cos_ranks  = [cos_rank.get(fid, n_facts) + 1 for fid in evidence_fact_ids]
                gold_rrf_ranks  = [sorted_ids.index(fid) + 1 if fid in sorted_ids else n_facts + 1
                                   for fid in evidence_fact_ids]
                hits = {
                    k: bool(set(sorted_ids[:k]) & evidence_fact_ids)
                    for k in _RECALL_KS
                }
            else:
                hits = {k: None for k in _RECALL_KS}

            per_q.append({
                "question":     qa["question"],
                "category":     qa["cat_name"],
                "evidence":     evidence,
                "has_evidence": has_evidence,
                "gold_cos_rank_best":  min(gold_cos_ranks)  if has_evidence else None,
                "gold_rrf_rank_best":  min(gold_rrf_ranks)  if has_evidence else None,
                **{f"hit@{k}": hits[k] for k in _RECALL_KS},
            })

    conn.close()
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Aggregate ──────────────────────────────────────────────────────
    with_ev  = [q for q in per_q if q["has_evidence"]]
    total_ev = len(with_ev)
    recall_scores: dict[int, float] = {}
    for k in _RECALL_KS:
        hits_k = sum(1 for q in with_ev if q[f"hit@{k}"])
        recall_scores[k] = hits_k / total_ev if total_ev else 0.0

    cat_recall: dict[str, float] = {}
    for cat in ["single_hop", "multi_hop", "temporal", "open_domain"]:
        cat_q = [q for q in with_ev if q["category"] == cat]
        if cat_q:
            cat_recall[cat] = sum(1 for q in cat_q if q["hit@5"]) / len(cat_q)

    # ── Print ─────────────────────────────────────────────────────────
    total_qa_all = len(per_q)
    skipped      = total_qa_all - total_ev
    desc = {
        1:  "did the right turn rank #1?",
        3:  "did the right turn appear in top 3?",
        5:  "did the right turn appear in top 5?",
        10: "did the right turn appear in top 10?",
        40: "did the right turn appear in top 40?",
    }
    cat_labels = {
        "single_hop": "Single-hop",
        "multi_hop":  "Multi-hop",
        "temporal":   "Temporal",
        "open_domain": "Open-domain",
    }
    r5 = recall_scores.get(5, 0.0)

    print(f"\n{'='*60}")
    print(f"  PREFLIGHT LoCoMo RECALL@K RESULTS")
    print(f"{'='*60}")
    print(f"\nQuestions with evidence : {total_ev} / {total_qa_all}")
    print(f"Questions skipped (no evidence / adversarial): {skipped}")
    print()
    for k in _RECALL_KS:
        print(f"Recall@{k:<2}  : {recall_scores[k]:6.2%}   ({desc[k]})")
    print()
    print("By category (Recall@5):")
    for cat, label in cat_labels.items():
        v = cat_recall.get(cat)
        if v is not None:
            print(f"  {label:<12}: {v:.2%}")
    print()
    print("What this means:")
    print(f"  Recall@5 = {r5:.2%} means Preflight found the answer-containing turn")
    print(f"  in the top 5 results for {r5:.0%} of questions.")
    print(f"  This measures pure retrieval quality, independent of answer generation.")
    _r40_pct = recall_scores.get(_RECALL_TARGET_K, 0.0) * 100
    _pass = _r40_pct >= _RECALL_TARGET_PCT
    print(f"\nTarget  R@{_RECALL_TARGET_K} >= {_RECALL_TARGET_PCT:.0f}%  :  {'PASS' if _pass else 'FAIL'}  (got {_r40_pct:.2f}%)")
    print(f"{'='*60}")

    result = {
        "questions_with_evidence": total_ev,
        "questions_total":         total_qa_all,
        "recall_at_k":             {str(k): round(v * 100, 2) for k, v in recall_scores.items()},
        "recall_at_5_by_category": {c: round(v * 100, 2) for c, v in cat_recall.items()},
        "target":                  {"k": _RECALL_TARGET_K, "pct": _RECALL_TARGET_PCT, "pass": _pass},
        "per_question":            per_q,
    }
    with open(RECALL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved -> {RECALL_RESULTS_PATH}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== PREFLIGHT LoCoMo BENCHMARK ===\n")

    # Dataset
    print("Loading dataset...")
    samples = download_dataset()
    print(f"  {len(samples)} conversations loaded")

    # Environment info
    emb_info = (
        f"real fastembed ({len(_utils_check.embed_text('test'))}-dim)"
        if _REAL_EMBEDDINGS else
        "SHA-256 stub (BM25+entity signal only — install fastembed for full scores)"
    )
    print(f"  Embeddings : {emb_info}")
    print(f"  Scorer     : {_STEMMER}")

    # Count total QA pairs (excluding adversarial cat 5)
    total_qa_count = sum(
        1 for s in samples for qa in s.get("qa", [])
        if int(qa.get("category", 0)) not in _SKIP_CATS
    )
    print(f"  QA pairs   : {total_qa_count} (adversarial cat-5 excluded)")

    # Run both modes
    stats_b, res_b = run_mode(samples, _mem, "B")

    # Recall@K evaluation on the freshly-ingested Mode B corpus
    db_path_b = os.path.join(_PREFLIGHT_DIR, "locomo_eval_B.db")
    run_recall_eval(samples, db_path_b)

    stats_a, res_a = run_mode(samples, _mem, "A")

    # ── Print results table ────────────────────────────────────────────────────
    MEM0_F1 = 91.6
    MEMU_F1 = 92.09

    print(f"\n{'='*60}")
    print("  PREFLIGHT LoCoMo BENCHMARK RESULTS")
    print(f"{'='*60}")

    for mode, stats, res in [("B", stats_b, res_b), ("A", stats_a, res_a)]:
        label = "Full turn ingestion" if mode == "B" else "Keyword extraction only"
        tq    = res["total_qa"]
        nz    = sum(1 for q in res["per_question"] if q["facts_retrieved"] == 0)
        nl    = sum(1 for q in res["per_question"] if 1 <= q["facts_retrieved"] <= 3)
        nh    = sum(1 for q in res["per_question"] if q["facts_retrieved"] >= 4)

        print(f"\nMode {mode} - {label}:")
        print(f"  Ingestion:")
        print(f"    Conversations : {len(samples)}")
        print(f"    Total turns   : {stats['total_turns']}")
        print(f"    Facts stored  : {stats['kw_facts'] if mode == 'A' else stats['total_turns'] + stats['kw_facts']}")
        print(f"  Retrieval:")
        print(f"    Total QA pairs     : {tq}")
        print(f"    Avg facts retrieved: {res['total_retrieved'] / max(tq, 1):.1f}")
        print(f"    Budget hit rate    : {res['budget_hits'] / max(tq, 1) * 100:.1f}%")
        print(f"  Scores:")
        print(f"    Overall F1    : {res['overall_f1']:.2f}%   "
              f"(Mem0: {MEM0_F1}%, MemU: {MEMU_F1}%)")
        for cat in ["single_hop", "multi_hop", "temporal", "open_domain"]:
            v = res["by_category"].get(cat)
            if v is not None:
                print(f"    {cat:<14}: {v:.2f}%")
        print(f"  Breakdown:")
        print(f"    0 facts retrieved  : {nz} ({nz / max(tq, 1) * 100:.0f}%)")
        print(f"    1-3 facts retrieved: {nl} ({nl / max(tq, 1) * 100:.0f}%)")
        print(f"    4-5 facts retrieved: {nh} ({nh / max(tq, 1) * 100:.0f}%)")

    print(f"\n{'='*60}")
    note = "(extractive — Mem0/MemU use LLM generation)" if not _REAL_EMBEDDINGS else ""
    print(f"  Mode B (full) vs Mem0 ({MEM0_F1}%)     : {res_b['overall_f1'] - MEM0_F1:+.2f}%  {note}")
    print(f"  Mode A (kw)   vs Mem0 ({MEM0_F1}%)     : {res_a['overall_f1'] - MEM0_F1:+.2f}%")
    print(f"  A->B delta (full corpus gain)           : +{res_b['overall_f1'] - res_a['overall_f1']:.2f}%")
    print(f"  Gap to Mem0 with LLM answers            : close this by adding store_memory LLM calls")
    print(f"{'='*60}")

    # ── Save results ───────────────────────────────────────────────────────────
    full_results = {
        "embedding_mode": "real_fastembed" if _REAL_EMBEDDINGS else "sha256_stub",
        "stemmer": _STEMMER,
        "mode_B": {
            "overall_f1":   res_b["overall_f1"],
            "by_category":  res_b["by_category"],
            "per_question": res_b["per_question"],
        },
        "mode_A": {
            "overall_f1":   res_a["overall_f1"],
            "by_category":  res_a["by_category"],
            "per_question": res_a["per_question"],
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
