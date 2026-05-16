#!/usr/bin/env python3
"""extract_engram_pairs_v3.py — Improved training pair extraction.

Key improvements over v2 (extract_engram_pairs.py):
  1. positive_curr: stores "Speaker: text" format — matches what the DB embeds
     at inference time (embed_text(curr_line) in memory.store_turn_window).
  2. dia_id / all_evidence_dia_ids: stored for exact gold-fact filtering
     in mine_hard_negatives_v3.py.
  3. Vague positive filtering: skips evidence turns whose text cannot support
     the question (too short, image/link references, etc.).

Output: engram_pairs_v3.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re

_DIR = os.path.join(os.path.expanduser("~"), ".config", "preflight")

parser = argparse.ArgumentParser()
parser.add_argument("--data",         default=os.path.join(_DIR, "locomo10.json"))
parser.add_argument("--output",       default=os.path.join(_DIR, "engram_pairs_v3.jsonl"))
parser.add_argument("--include-obs",  action="store_true", default=True,
                    help="Include event_summary and observation self-retrieval pairs.")
parser.add_argument("--skip-vague",   action="store_true", default=True,
                    help="Skip evidence turns whose text can't support the question.")
parser.add_argument("--min-chars",    type=int, default=20)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()
random.seed(args.seed)

CAT_NAMES = {1: "multi_hop", 2: "temporal", 3: "single_hop", 4: "open_domain"}
SKIP_CATS = {5}   # adversarial / unanswerable

# Phrases that indicate the turn is a media/link reference with no textual answer.
VAGUE_PHRASES = {
    "take a look", "here's", "here is", "by the way",
    "check this out", "check it out", "look at this", "look at that",
    "here you go", "i sent", "sending you", "attached", "click here",
    "see the photo", "see the pic", "see the image",
}


def _is_vague(text: str) -> bool:
    """Return True if this turn text cannot reliably answer a factual question."""
    t = text.strip()
    if len(t) < args.min_chars:
        return True
    tl = t.lower()
    # Image / link references that lack textual content
    if any(p in tl for p in VAGUE_PHRASES) and len(t) < 80:
        return True
    return False


def _build_dia_map(conv: dict) -> dict[str, dict]:
    """Return {dia_id: {"text": ..., "speaker": ...}} for every session turn."""
    dia_map: dict[str, dict] = {}
    for key, turns in conv.items():
        if not re.match(r'^session_\d+$', key):
            continue
        if not isinstance(turns, list):
            continue
        for turn in turns:
            did     = str(turn.get("dia_id", "")).strip()
            text    = str(turn.get("text", "")).strip()
            speaker = str(turn.get("speaker", "unknown")).strip()
            if did and text:
                dia_map[did] = {"text": text, "speaker": speaker}
    return dia_map


def _clean_obs(text: str) -> str:
    t = str(text).strip()
    t = re.sub(r'^[-•*]\s*', '', t)
    return t


# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading: {args.data}")
raw = json.load(open(args.data, encoding="utf-8"))
samples: list[dict] = list(raw.values()) if isinstance(raw, dict) else list(raw)
print(f"  {len(samples)} conversations")

# ── Extract pairs ─────────────────────────────────────────────────────────────

pairs: list[dict] = []
n_skipped_no_ev      = 0
n_skipped_cat5       = 0
n_skipped_too_short  = 0
n_skipped_vague      = 0

for sample in samples:
    sid  = sample.get("sample_id", "?")
    conv = sample.get("conversation", {})
    dia_map = _build_dia_map(conv)

    for qa in sample.get("qa", []):
        q   = str(qa.get("question", "")).strip()
        cat = int(qa.get("category", 0) or 0)
        if not q:
            continue
        if cat in SKIP_CATS:
            n_skipped_cat5 += 1
            continue

        ev_ids = qa.get("evidence", []) or []
        ev_ids = [str(e) for e in ev_ids if str(e) in dia_map]
        if not ev_ids:
            n_skipped_no_ev += 1
            continue

        # One pair per evidence turn (fine-grained)
        for dia_id in ev_ids:
            info    = dia_map[dia_id]
            text    = info["text"]
            speaker = info["speaker"]

            if len(text) < args.min_chars:
                n_skipped_too_short += 1
                continue

            if args.skip_vague and _is_vague(text):
                n_skipped_vague += 1
                continue

            pairs.append({
                "query":                q,
                "positive":             text,               # raw text — backward compat
                "positive_curr":        f"{speaker}: {text}",  # v3: matches DB embed format
                "dia_id":               dia_id,
                "all_evidence_dia_ids": ev_ids,
                "category":             CAT_NAMES.get(cat, f"cat{cat}"),
                "sample_id":            sid,
                "source":               "locomo_qa",
            })

    if not args.include_obs:
        continue

    # event_summary self-retrieval pairs
    for sess_key, ev_list in sample.get("event_summary", {}).items():
        if not isinstance(ev_list, list):
            continue
        for ev in ev_list:
            text = _clean_obs(ev)
            if len(text) < args.min_chars:
                continue
            first_sent = (re.split(r'[.!?]', text) + [""])[0].strip()
            if len(first_sent) < 8:
                first_sent = text[:80]
            pairs.append({
                "query":                first_sent,
                "positive":             text,
                "positive_curr":        text,   # no speaker for summaries
                "dia_id":               None,
                "all_evidence_dia_ids": [],
                "category":             "event_summary",
                "sample_id":            sid,
                "source":               "locomo_event",
            })

    # observation self-retrieval pairs
    for obs in sample.get("observation", []) or []:
        raw_text = obs.get("text", obs) if isinstance(obs, dict) else obs
        text = _clean_obs(raw_text)
        if len(text) < args.min_chars:
            continue
        first_sent = (re.split(r'[.!?]', text) + [""])[0].strip()
        if len(first_sent) < 8:
            first_sent = text[:80]
        pairs.append({
            "query":                first_sent,
            "positive":             text,
            "positive_curr":        text,
            "dia_id":               None,
            "all_evidence_dia_ids": [],
            "category":             "observation",
            "sample_id":            sid,
            "source":               "locomo_obs",
        })

# ── Dedup by (query, positive_curr) ──────────────────────────────────────────

seen: set[tuple] = set()
deduped: list[dict] = []
for p in pairs:
    key = (p["query"][:120], p["positive_curr"][:120])
    if key not in seen:
        seen.add(key)
        deduped.append(p)

random.shuffle(deduped)

# ── Write ─────────────────────────────────────────────────────────────────────

with open(args.output, "w", encoding="utf-8") as f:
    for p in deduped:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

by_cat: dict[str, int] = {}
by_src: dict[str, int] = {}
for p in deduped:
    by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    by_src[p["source"]]   = by_src.get(p["source"],   0) + 1

print(f"\n=== PAIRS WRITTEN: {len(deduped):,} ===")
print(f"  Skipped — no evidence : {n_skipped_no_ev}")
print(f"  Skipped — cat-5 (adv): {n_skipped_cat5}")
print(f"  Skipped — too short  : {n_skipped_too_short}")
print(f"  Skipped — vague text : {n_skipped_vague}")
print()
print("  By category:")
for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"    {c:<24}  {n:>5}")
print()
print("  By source:")
for s, n in sorted(by_src.items(), key=lambda x: -x[1]):
    print(f"    {s:<24}  {n:>5}")
print(f"\n→ {args.output}")
