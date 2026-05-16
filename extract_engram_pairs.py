#!/usr/bin/env python3
"""Extract (query, positive_text, category) pairs from LoCoMo for embedding fine-tuning.

Reads locomo10.json, maps every QA evidence dia_id to its actual conversation
turn text, and emits one JSONL line per (question, evidence_turn) pair.

Also adds event_summary and observation text as self-retrieval pairs
(these are pre-distilled facts useful for teaching the embedder what
a "memory fact" looks like vs. a conversational query).

Output: engram_pairs_locomo.jsonl
Usage:
    python extract_engram_pairs.py
    python extract_engram_pairs.py --min-chars 20 --include-obs
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min-chars", type=int, default=10,
                    help="Minimum character length for an evidence text to be included.")
parser.add_argument("--include-obs", action="store_true", default=True,
                    help="Include observation/event_summary self-retrieval pairs.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

_DIR  = os.path.join(os.path.expanduser("~"), ".config", "preflight")
DATA  = os.path.join(_DIR, "locomo10.json")
OUT   = os.path.join(_DIR, "engram_pairs_locomo.jsonl")

CAT_NAMES = {1: "multi_hop", 2: "temporal", 3: "single_hop", 4: "open_domain"}
SKIP_CATS = {5}  # adversarial — evidence not in corpus


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_dia_map(conversation: dict) -> dict[str, str]:
    """Return {dia_id: turn_text} for every session turn in this conversation."""
    dia_map: dict[str, str] = {}
    for key, turns in conversation.items():
        if not re.match(r'^session_\d+$', key):
            continue
        if not isinstance(turns, list):
            continue
        for turn in turns:
            did  = turn.get("dia_id", "")
            text = str(turn.get("text", "")).strip()
            if did and text:
                dia_map[did] = text
    return dia_map


def _clean_obs(text: str) -> str:
    """Normalize an observation/event text for use as a training example."""
    text = str(text).strip()
    # Remove leading dash/bullet
    text = re.sub(r'^[-•*]\s*', '', text)
    return text


# ── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading: {DATA}")
raw = json.load(open(DATA, encoding="utf-8"))
samples: list[dict] = list(raw.values()) if isinstance(raw, dict) else list(raw)
print(f"  {len(samples)} conversations")

# ── Extract pairs ─────────────────────────────────────────────────────────────
pairs: list[dict] = []
skipped_no_ev   = 0
skipped_cat5    = 0
skipped_too_short = 0

for sample in samples:
    sid  = sample.get("sample_id", "?")
    conv = sample.get("conversation", {})

    # Map dia_id → turn text for this conversation
    dia_map = _build_dia_map(conv)

    # ── QA pairs ────────────────────────────────────────────────────────────
    for qa in sample.get("qa", []):
        q   = str(qa.get("question", "")).strip()
        cat = int(qa.get("category", 0) or 0)

        if not q:
            continue
        if cat in SKIP_CATS:
            skipped_cat5 += 1
            continue

        ev_ids   = qa.get("evidence", []) or []
        ev_texts = [dia_map[e] for e in ev_ids if e in dia_map]

        if not ev_texts:
            skipped_no_ev += 1
            continue

        for ev in ev_texts:
            ev = ev.strip()
            if len(ev) < args.min_chars:
                skipped_too_short += 1
                continue
            pairs.append({
                "query":     q,
                "positive":  ev,
                "category":  CAT_NAMES.get(cat, f"cat{cat}"),
                "sample_id": sid,
                "source":    "locomo_qa",
            })

    # ── Event summary self-retrieval pairs ───────────────────────────────────
    if args.include_obs:
        for sess_key, ev_list in sample.get("event_summary", {}).items():
            if not isinstance(ev_list, list):
                continue
            for ev in ev_list:
                text = _clean_obs(ev)
                if len(text) < args.min_chars:
                    continue
                # Use the first sentence of the event as the query,
                # and the full text as the positive (teaches recall of detail).
                first_sent = re.split(r'[.!?]', text)[0].strip()
                if len(first_sent) < 8:
                    first_sent = text[:80]
                pairs.append({
                    "query":     first_sent,
                    "positive":  text,
                    "category":  "event_summary",
                    "sample_id": sid,
                    "source":    "locomo_event",
                })

        # ── Observation self-retrieval pairs ─────────────────────────────────
        for obs in sample.get("observation", []) or []:
            text = _clean_obs(obs.get("text", obs) if isinstance(obs, dict) else obs)
            if len(text) < args.min_chars:
                continue
            first_sent = re.split(r'[.!?]', text)[0].strip()
            if len(first_sent) < 8:
                first_sent = text[:80]
            pairs.append({
                "query":     first_sent,
                "positive":  text,
                "category":  "observation",
                "sample_id": sid,
                "source":    "locomo_obs",
            })

# ── Dedup by (query, positive) ────────────────────────────────────────────────
seen: set[tuple[str, str]] = set()
deduped: list[dict] = []
for p in pairs:
    key = (p["query"][:120], p["positive"][:120])
    if key not in seen:
        seen.add(key)
        deduped.append(p)

# Shuffle for training
random.shuffle(deduped)

# ── Write ─────────────────────────────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    for p in deduped:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
by_cat: dict[str, int] = {}
by_src: dict[str, int] = {}
for p in deduped:
    by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    by_src[p["source"]]   = by_src.get(p["source"],   0) + 1

print(f"\nPairs written: {len(deduped):,}")
print(f"  Skipped — no evidence text : {skipped_no_ev}")
print(f"  Skipped — adversarial (cat5): {skipped_cat5}")
print(f"  Skipped — too short         : {skipped_too_short}")

print(f"\nBy category:")
for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"  {c:<20} {n:>5}")

print(f"\nBy source:")
for s, n in sorted(by_src.items(), key=lambda x: -x[1]):
    print(f"  {s:<20} {n:>5}")

print(f"\n→ {OUT}")
