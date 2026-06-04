# engram merge plan — best of both worlds

Goal: combine user/engram (production, 2,765 LOC, plugin-coupled) with
Nitin-Gupta1109/engram (modular, FAISS, bge-large, CE-by-default) to close
the 11pp R@5 gap to Nitin's 93.9% LoCoMo.

**Constraints preserved:** local-only, no LLM, no paid APIs, no training data,
plugin CLI contract unchanged.

## Vendored source

`/home/sheldon_antony/.config/opencode/engram_nitin/` — Nitin's full engram
package (v0.1.7, MIT). Draw from as a library, do not edit.

## Adopt from Nitin (high-value, low-risk)

| # | Change | Where it lives | Estimated lift |
|---|---|---|---|
| 1 | bge-large (1024d) as default embedding | utils.py | +3-5pp R@3 |
| 2 | Cross-encoder as default (bge-reranker-v2-m3) | utils.py + memory.py | +2-3pp R@3 |
| 3 | FAISS backend for >1k facts | new engram_backends/ | enables scale |
| 4 | Chunked session ingestion (6 turns, overlap 1) | memory.py | +1pp R@3 |
| 5 | Speaker-name injection in turn text | extractor/memory.py | +2pp single-hop |
| 6 | Comprehensive PREFERENCE_PATTERNS | extractor.py | +5pp preference R@3 |
| 7 | Topic extraction (vocab bridging) | extractor.py | +1pp R@3 |
| 8 | Person-name boost in RRF | memory.py retrieve_facts | +1pp R@3 |
| 9 | Quoted-phrase boost in RRF | memory.py retrieve_facts | +0.5pp R@3 |
| 10 | Temporal proximity boost in RRF | memory.py retrieve_facts | +1-2pp temporal R@3 |
| 11 | Hybrid score (60% dense + 40% BM25-norm) in MCP | mcp_server | consistency |
| 12 | Native MCP server | engram_nitin/mcp_server.py | drops TS plugin |

## Keep from user/engram (proven, production)

- SM-2 decay, mutation history, soft supersede, audit log
- Fact relations graph, slot fills
- Query decomposition (ToR-Lite), query-type routing
- Composite scoring (recency, staleness, session_rec, freq)
- MMR diversity, token budget
- Context BM25 (neighbor turn window)
- WordNet-derived BM25 (env-gated)
- Lexical channels (person-name, date, key-bigram FTS5)
- Cross-encoder guard (min-rank ensemble, no regression)
- Coverage guard (min-rank no regression)
- LLM atomic extractor (opt-in)
- Existing CLI contract (plugin compatibility)

## Implementation order (v20 → v25)

### v20 — embedding & reranker upgrade (Phase 1)
- utils.py: add bge-large as default; bge-small remains fast path
- utils.py: switch default CE to bge-reranker-v2-m3
- memory.py: remove CE env gate, make always-on (with fast timeout)
- Run LoCoMo bench, expect +3-5pp R@3

### v21 — preferences & topics (Phase 2)
- extractor.py: add Nitin's PREFERENCE_PATTERNS (27 patterns vs current ~5)
- memory.py: add topic extraction at store_turn_window
- memory.py: add topic-doc boost to RRF
- Run LoCoMo bench, expect +1-2pp preference R@3

### v22 — chunked ingestion & speaker injection (Phase 3)
- memory.py: replace _chunk_turns_window with Nitin's 6-turn overlap
- memory.py: speaker_names injection (env var SPEAKER_NAME)
- Run LoCoMo bench, expect +1pp R@3

### v23 — person/phrase/temporal boosts (Phase 4)
- memory.py: add 3 new RRF signals
- Run LoCoMo bench, expect +2-3pp R@3

### v24 — FAISS backend (Phase 5)
- new module engram_backends/faiss.py
- env opt-in: ENGRAM_VECTOR_BACKEND=faiss
- Falls back to numpy LRU (no break)

### v25 — MCP server (Phase 6)
- engram_nitin/mcp_server.py wired up
- TS plugin kept for transition, then deprecated

## Rollback

Each v## is independently env-gated. Disable with no restart:
- `ENGRAM_BGE_LARGE=0` → use bge-small
- `ENGRAM_USE_CE=0` → no cross-encoder
- `ENGRAM_USE_FAISS=0` → numpy LRU
- `ENGRAM_USE_TOPIC=0` → no topic extraction
- `ENGRAM_USE_BOOSTS=0` → no person/phrase/temporal boost

## Success criteria

- LoCoMo R@3 >= 90% (currently 80.99% eval, 77.20% prod)
- LoCoMo R@5 >= 93% (currently 86.68% eval, 82.95% prod)
- LongMemEval S R@3 >= 85% (currently 78.7%)
- Per-fact-type R@3: preference >= 70%, single-hop >= 60%
- Latency p95 < 1.5s (CE adds ~200ms)
- Plugin contract unchanged (CLI surface identical)

## Anti-goals

- Do NOT add any LLM dependency at query time
- Do NOT touch the plugin contract (PromptEnricher expects line-based output)
- Do NOT break the existing DB schema (additive migrations only)
- Do NOT add a cloud backend (Qdrant) — local only
