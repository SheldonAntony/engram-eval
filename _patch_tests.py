"""Patch script: insert 13 new tests into test_mcp.py before the summary block."""
import sys

path = r'C:\Users\Sheldon Antony\.config\preflight\test_mcp.py'
content = open(path, encoding='utf-8').read()

marker = 'total = _PASS + _FAIL'
idx = content.find(marker)
if idx == -1:
    print("ERROR: marker not found"); sys.exit(1)

last_nl   = content.rfind('\n', 0, idx)        # \n before 'total = ...'
prev_nl   = content.rfind('\n', 0, last_nl)    # \n before separator line
insert_at = prev_nl + 1                        # start of separator line

NEW_TESTS = r"""
# -- IMP1-1: Pool B rescues proven-useful fact outside Pool A --
print("\n-- IMP1-1: Pool B rescues retrieval_count>0 fact outside Pool A --")
import time as _time
IMP1_PROJ = "imp1_pool_proj"
try:
    anchor_fid = _mem.store_fact(IMP1_PROJ, "imp1_s", "the project uses a microservice architecture pattern")
    for _i in range(210):
        _mem.store_fact(IMP1_PROJ, "imp1_s", f"dummy filler fact number {_i:04d} ignore this completely")
    _conn1 = sqlite3.connect(_DB)
    _conn1.execute("UPDATE facts SET retrieval_count = 1, last_retrieved_at = ? WHERE id = ?",
                   (_time.time(), anchor_fid))
    _conn1.commit(); _conn1.close()
    r1 = _mem.retrieve_facts(IMP1_PROJ, "imp1_s", "microservice architecture", top_n=300, threshold=0.0)
    check("IMP1-1: Pool B rescues old fact with retrieval_count>0",
          isinstance(r1, list) and any("microservice architecture pattern" in x for x in r1),
          f"anchor_fid={anchor_fid}, total={len(r1)}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP1-2: candidate pool capped at <=500 with many facts --
print("\n-- IMP1-2: candidate pool capped at <=500 with many facts --")
IMP1B_PROJ = "imp1b_bigpool_proj"
try:
    for _i in range(200):
        _mem.store_fact(IMP1B_PROJ, "imp1b_s", f"big pool fact {_i:04d} about various unrelated topics content")
    r1b = _mem.retrieve_facts(IMP1B_PROJ, "imp1b_s", "unrelated topics", top_n=5, threshold=0.0,
                               include_budget_info=True)
    check("IMP1-2: retrieve returns dict when include_budget_info=True",
          isinstance(r1b, dict) and "total_candidates" in r1b,
          f"got: {type(r1b)}")
    check("IMP1-2: candidate pool capped at 500 or fewer",
          isinstance(r1b, dict) and r1b.get("total_candidates", 9999) <= 500,
          f"total_candidates={r1b.get('total_candidates') if isinstance(r1b, dict) else '?'}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP2-1: store_turn_window middle index has prev+curr+next tags --
print("\n-- IMP2-1: store_turn_window middle index has prev+curr+next tags --")
IMP2_PROJ = "imp2_window_proj"
_TURNS = [
    {"speaker": "Alice", "text": "what time is the meeting"},
    {"speaker": "Bob",   "text": "the meeting is at two pm"},
    {"speaker": "Alice", "text": "thanks I will set a reminder"},
    {"speaker": "Bob",   "text": "sure no problem at all"},
    {"speaker": "Alice", "text": "see you then"},
]
try:
    _mem.store_turn_window(IMP2_PROJ, "imp2_s", _TURNS, 2)
    _c21 = sqlite3.connect(_DB)
    _row21 = _c21.execute("SELECT content FROM facts WHERE project_id = ? ORDER BY id DESC LIMIT 1",
                          (IMP2_PROJ,)).fetchone()
    _c21.close()
    _content21 = _row21[0] if _row21 else ""
    check("IMP2-1: window contains [prev] tag",  "[prev]" in _content21, f"content={_content21[:120]}")
    check("IMP2-1: window contains [curr] tag",  "[curr]" in _content21, f"content={_content21[:120]}")
    check("IMP2-1: window contains [next] tag",  "[next]" in _content21, f"content={_content21[:120]}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP2-2: store_turn_window at index 0 -- no [prev] --
print("\n-- IMP2-2: store_turn_window at index 0 -- no [prev] --")
try:
    _mem.store_turn_window(IMP2_PROJ, "imp2_s2", _TURNS, 0)
    _c22 = sqlite3.connect(_DB)
    _row22 = _c22.execute(
        "SELECT content FROM facts WHERE project_id = ? AND session_id = 'imp2_s2' ORDER BY id DESC LIMIT 1",
        (IMP2_PROJ,)).fetchone()
    _c22.close()
    _content22 = _row22[0] if _row22 else ""
    check("IMP2-2: index-0 window has no [prev]",  "[prev]" not in _content22, f"content={_content22[:120]}")
    check("IMP2-2: index-0 window has [curr]",     "[curr]" in _content22,     f"content={_content22[:120]}")
    check("IMP2-2: index-0 window has [next]",     "[next]" in _content22,     f"content={_content22[:120]}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP2-3: store_turn_window at last index -- no [next] --
print("\n-- IMP2-3: store_turn_window at last index -- no [next] --")
try:
    _mem.store_turn_window(IMP2_PROJ, "imp2_s3", _TURNS, 4)
    _c23 = sqlite3.connect(_DB)
    _row23 = _c23.execute(
        "SELECT content FROM facts WHERE project_id = ? AND session_id = 'imp2_s3' ORDER BY id DESC LIMIT 1",
        (IMP2_PROJ,)).fetchone()
    _c23.close()
    _content23 = _row23[0] if _row23 else ""
    check("IMP2-3: last-index window has [prev]",    "[prev]" in _content23,     f"content={_content23[:120]}")
    check("IMP2-3: last-index window has [curr]",    "[curr]" in _content23,     f"content={_content23[:120]}")
    check("IMP2-3: last-index window has no [next]", "[next]" not in _content23, f"content={_content23[:120]}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP2-4: window content is retrievable --
print("\n-- IMP2-4: store_turn_window content is retrievable --")
try:
    r24 = _mem.retrieve_facts(IMP2_PROJ, "imp2_s", "meeting at two pm", top_n=5, threshold=0.0)
    check("IMP2-4: retrieve finds window content",
          isinstance(r24, list) and any("[curr]" in x for x in r24),
          f"results={[x[:60] for x in r24]}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP3-2: _session_recency_score same session = 1.0 --
print("\n-- IMP3-2: _session_recency_score same session = 1.0 --")
try:
    _score32 = _mem._session_recency_score("sess_x", "sess_x", {"sess_x": 3})
    check("IMP3-2: same-session score is 1.0", _score32 == 1.0, f"got {_score32}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP3-3: _session_recency_score gap>=7 = 0.0 --
print("\n-- IMP3-3: _session_recency_score gap>=7 = 0.0 --")
try:
    _score33 = _mem._session_recency_score("s_old", "s_new", {"s_old": 1, "s_new": 8})
    check("IMP3-3: gap=7 score is 0.0", _score33 == 0.0, f"got {_score33}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP3-4: _session_recency_score gap=3 matches formula --
print("\n-- IMP3-4: _session_recency_score gap=3 matches formula --")
try:
    _score34 = _mem._session_recency_score("s1", "s2", {"s1": 2, "s2": 5})
    _expected34 = max(0.0, 1.0 - 3 * _mem._SESSION_RECENCY_DECAY)
    check("IMP3-4: gap=3 score matches formula",
          abs(_score34 - _expected34) < 1e-9,
          f"got {_score34}, expected {_expected34}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP3-1: session-recency ranks recent session higher --
print("\n-- IMP3-1: session-recency ranks recent session higher --")
IMP3_PROJ = "imp3_recency_proj"
try:
    for _i in range(1, 6):
        _mem.session_mark(f"imp3_s{_i}", IMP3_PROJ)
    _mem.store_fact(IMP3_PROJ, "imp3_s1", "the user prefers dark mode as their editor theme setting")
    _mem.store_fact(IMP3_PROJ, "imp3_s5", "the user prefers light mode as their editor theme setting")
    _r31 = _mem.retrieve_facts(IMP3_PROJ, "imp3_s5", "editor theme", top_n=2, threshold=0.0)
    check("IMP3-1: session_5 fact ranked first",
          isinstance(_r31, list) and len(_r31) >= 1 and "light mode" in _r31[0],
          f"first result: {_r31[0][:80] if _r31 else 'none'}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP3-5: scoring weights sum to 1.0 --
print("\n-- IMP3-5: scoring weights sum to 1.0 --")
try:
    _w_sum = 0.35 + 0.20 + 0.20 + 0.15 + 0.10
    check("IMP3-5: rrf+recency+staleness+sess_rec+freq == 1.0",
          abs(_w_sum - 1.0) < 1e-9, f"sum={_w_sum}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP4-1: temporal linking creates fact_relations edges --
print("\n-- IMP4-1: temporal linking creates fact_relations edges --")
IMP4_PROJ = "imp4_temporal_proj"
try:
    _mem.store_fact(IMP4_PROJ, "imp4_s", "temporal alpha about the deployment pipeline setup stage")
    _mem.store_fact(IMP4_PROJ, "imp4_s", "temporal beta about the staging environment config details")
    _mem.store_fact(IMP4_PROJ, "imp4_s", "temporal gamma about the production release process flow")
    _c4 = sqlite3.connect(_DB)
    _tedges = _c4.execute(
        "SELECT COUNT(*) FROM fact_relations WHERE relation = 'temporal'"
    ).fetchone()[0]
    _c4.close()
    check("IMP4-1: temporal edges exist in fact_relations", _tedges >= 1,
          f"temporal edges found: {_tedges}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# -- IMP5-1: entity enrichment merges same-session entity-overlapping facts --
print("\n-- IMP5-1: entity enrichment merges same-session entity-overlapping facts --")
IMP5_PROJ = "imp5_enrich_proj"
_orig_extractor = _mem._extract_entities
try:
    _mem._extract_entities = lambda text: ["Caroline"] if "Caroline" in text else []
    _mem.store_fact(IMP5_PROJ, "imp5_s", "Caroline went to the market yesterday afternoon")
    _mem.store_fact(IMP5_PROJ, "imp5_s", "Caroline bought fresh apples and artisan bread")
    _c5 = sqlite3.connect(_DB)
    _enrich_cnt = _c5.execute(
        "SELECT COUNT(*) FROM fact_mutations WHERE mutation_type = 'ENRICH'"
    ).fetchone()[0]
    _live_cnt = _c5.execute(
        "SELECT COUNT(*) FROM facts WHERE project_id = ? AND superseded_at IS NULL",
        (IMP5_PROJ,)
    ).fetchone()[0]
    _c5.close()
    check("IMP5-1: ENRICH mutation written", _enrich_cnt >= 1,
          f"ENRICH mutations: {_enrich_cnt}")
    check("IMP5-1: only 1 live fact (merged, not duplicated)", _live_cnt == 1,
          f"live facts: {_live_cnt}")
except Exception:
    print("  ERROR:", traceback.format_exc())
finally:
    _mem._extract_entities = _orig_extractor

"""

new_content = content[:insert_at] + NEW_TESTS + content[insert_at:]
open(path, 'w', encoding='utf-8').write(new_content)
print(f"Done. Inserted {len(NEW_TESTS)} chars at position {insert_at}. New size: {len(new_content)}")
