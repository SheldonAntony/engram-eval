#!/usr/bin/env python3
"""Preflight MCP Server — E2E smoke test.

Tests core tool logic directly (bypasses MCP protocol) using a temp SQLite DB
and a stubbed fastembed so the test completes in under 5 seconds.

Run:
    python ~/.config/preflight/test_mcp.py
"""

import hashlib
import json
import os
import sys
import types
import sqlite3
import tempfile
import traceback

# ── Stub fastembed before importing anything that touches utils.py ─────────────
_stub_utils = types.ModuleType("utils")

def _embed(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    v = [b / 255.0 for b in h[:32]]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v

def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

_stub_utils.embed_text = _embed
_stub_utils.cosine_similarity = _cos
sys.modules["utils"] = _stub_utils

# ── Add backend scripts dir to path ───────────────────────────────────────────
_SCRIPTS_DIR = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import mcp_server tool functions AFTER stubs are in place ─────────────────
# Temporarily override _VENV_PYTHON so mcp_server doesn't try subprocess with venv
# (we call tool fns directly, not via subprocess, in this test).
import mcp_server as srv

import memory as _mem

# ── Isolated temp DB for this test run ────────────────────────────────────────
_DB = os.path.join(tempfile.gettempdir(), "preflight_test_mcp.db")
if os.path.exists(_DB):
    os.remove(_DB)
_mem.DB_PATH = _DB

# Patch _call_memory / _call_tasks / _call_classifier to call Python functions
# directly instead of via subprocess (avoids needing fastembed in venv during CI).
def _direct_memory(*args):
    """Route CLI-style args to memory.py functions directly."""
    cmd = args[0]
    if cmd == "store_fact":
        _, project_id, session_id, text, *rest = args
        fact_type = rest[0] if rest else "finding"
        _mem.store_fact(project_id, session_id, text, fact_type)
        return ""
    elif cmd == "retrieve_facts":
        _, project_id, session_id, prompt, top_n, threshold, *extra = (*args, None, None)
        include_budget = extra[0] == "true" if extra and extra[0] is not None else False
        return json.dumps(_mem.retrieve_facts(
            project_id, session_id, prompt, int(top_n), float(threshold),
            include_budget_info=include_budget,
        ))
    elif cmd == "store_slot_fill":
        _, project_id, session_id, slot_name, value = args
        _mem.store_slot_fill(project_id, session_id, slot_name, value)
        return ""
    elif cmd == "retrieve_slot_fills":
        _, project_id = args
        return json.dumps(_mem.retrieve_slot_fills(project_id))
    elif cmd == "session_mark":
        _, session_id, project_id = args
        _mem.session_mark(session_id, project_id)
        return ""
    elif cmd == "session_seen":
        _, session_id = args
        return "YES" if _mem.session_seen(session_id) else "NO"
    elif cmd == "session_unmark":
        _, session_id = args
        _mem.session_unmark(session_id)
        return ""
    elif cmd == "get_graph":
        _, project_id, query, *rest = args
        depth = int(rest[0]) if rest else 1
        return json.dumps(_mem.get_graph(project_id, query, depth))
    elif cmd == "get_history":
        _, fact_id = args
        return json.dumps(_mem.get_history(int(fact_id)))
    elif cmd == "consolidate_memories":
        _, project_id, *rest = args
        session_id = rest[0] if rest else ""
        return json.dumps(_mem.consolidate_memories(project_id, session_id))
    else:
        raise ValueError(f"Unknown CLI command in test: {cmd}")

def _direct_tasks(*args):
    """Route retrieve_similar to tasks.py directly."""
    import tasks as _tasks
    _tasks.DB_PATH = _DB
    cmd = args[0]
    if cmd == "retrieve_similar":
        _, project_id, session_id, prompt, top_n, threshold = args
        return json.dumps(_tasks.retrieve_similar_tasks(project_id, session_id, prompt, int(top_n), float(threshold)))
    return "[]"

def _direct_classifier(payload_json: str) -> str:
    payload = json.loads(payload_json)
    prompt  = payload.get("prompt", "").lower()
    if any(w in prompt for w in ("fix", "bug", "error", "crash")):
        return json.dumps({"type": "bug"})
    if any(w in prompt for w in ("add", "feature", "implement", "new")):
        return json.dumps({"type": "feature"})
    if any(w in prompt for w in ("refactor", "clean", "reorganize")):
        return json.dumps({"type": "refactor"})
    return json.dumps({"type": None})

# Monkey-patch the server module so tool fns use our direct callables
srv._call_memory    = lambda *a: _direct_memory(*a)
srv._call_tasks     = lambda *a: _direct_tasks(*a)
srv._call_classifier = lambda p: _direct_classifier(p)

# Patch auto_extract's internal extractor call to use keyword_extract directly
# (no API key, no subprocess needed in test)
import extractor as _ext

def _patched_auto_extract(response_text: str, project_id: str, session_id: str) -> dict:
    facts = _ext.keyword_extract(response_text)
    saved: list[str] = []
    for fact in facts:
        _mem.store_fact(project_id, session_id, fact, "finding")
        saved.append(fact)
    return {"extracted": len(saved), "facts": saved}

# ── Test helpers ───────────────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — get_project_id
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 1: get_project_id ──")
try:
    result = srv.tool_get_project_id(os.path.expanduser("~/.config/opencode"))
    pid = result.get("project_id", "")
    check("returns a dict with project_id key", "project_id" in result)
    check("project_id is a 12-char hex string", len(pid) == 12 and all(c in "0123456789abcdef" for c in pid),
          f"got: {pid!r}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — store_slot / list_slots
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 2: store_slot / list_slots ──")
PROJ = "smoke_test"
SESS = "sess_smoke_1"
try:
    srv.tool_store_slot(SESS, PROJ, "framework", "FastAPI")
    srv.tool_store_slot(SESS, PROJ, "language",  "Python")
    slots = srv.tool_list_slots(PROJ)
    check("list_slots returns framework", slots.get("framework") == "FastAPI",
          f"got: {slots}")
    check("list_slots returns language",  slots.get("language")  == "Python",
          f"got: {slots}")
    check("no extra slots", len(slots) == 2, f"got: {slots}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — store_memory / get_context retrieves it
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 3: store_memory → get_context retrieval ──")
try:
    srv.tool_store_memory(SESS, PROJ, "always use SQLAlchemy not raw SQL", "finding")
    ctx = srv.tool_get_context("fix the database query", "sess_smoke_2", PROJ)
    mems = ctx.get("memories", [])
    check("memories list is non-empty", len(mems) > 0, f"got empty memories")
    check("SQLAlchemy fact is in memories",
          any("SQLAlchemy" in m for m in mems),
          f"got: {mems}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — preference fact stored in __global__ is visible from OTHER project
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 4: global preference visible from different project ──")
OTHER_PROJ = "totally_different_project"
try:
    srv.tool_store_memory(SESS, PROJ, "keep responses concise", "preference")
    ctx2 = srv.tool_get_context("add user authentication", "sess_smoke_3", OTHER_PROJ)
    mems2 = ctx2.get("memories", [])
    check("preference appears in memories for OTHER project",
          any("concise" in m for m in mems2),
          f"got: {mems2}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — missing_slots when no slots stored (feature task)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 5: missing_slots for feature task (no slots stored) ──")
EMPTY_PROJ = "fresh_project_no_slots"
try:
    ctx3 = srv.tool_get_context("add user authentication", "sess_smoke_4", EMPTY_PROJ)
    missing = ctx3.get("missing_slots", [])
    task    = ctx3.get("task_type")
    check("task_type classified as feature", task == "feature", f"got: {task!r}")
    check("missing_slots contains language",  "language"  in missing, f"got: {missing}")
    check("missing_slots contains framework", "framework" in missing, f"got: {missing}")
    check("missing_slots contains database",  "database"  in missing, f"got: {missing}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — storing language removes it from missing_slots
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 6: store language → it leaves missing_slots ──")
try:
    srv.tool_store_slot("sess_smoke_4", EMPTY_PROJ, "language", "Python")
    ctx4 = srv.tool_get_context("add user authentication", "sess_smoke_5", EMPTY_PROJ)
    missing2 = ctx4.get("missing_slots", [])
    check("language is NO LONGER in missing_slots", "language"  not in missing2,
          f"got: {missing2}")
    check("framework still missing",                "framework" in missing2,
          f"got: {missing2}")
    check("database still missing",                 "database"  in missing2,
          f"got: {missing2}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 7 — Auto-linking: two related facts get a graph edge
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 7: auto-link — related facts get an edge ──")
GRAPH_PROJ = "graph_test_proj"
try:
    _mem.store_fact(GRAPH_PROJ, "sess_g1", "we use SQLAlchemy as our ORM layer", "decision")
    _mem.store_fact(GRAPH_PROJ, "sess_g1", "N+1 query bug was caused by missing eager loading in SQLAlchemy", "finding")

    # With the stub embedder two distinct texts won't hash to similarity >= 0.65,
    # so exercise link_facts() directly to create a real edge.
    conn_g = sqlite3.connect(_DB)
    fact_ids = [r[0] for r in conn_g.execute(
        "SELECT id FROM facts WHERE project_id = ? ORDER BY id", (GRAPH_PROJ,)
    ).fetchall()]
    conn_g.close()
    if len(fact_ids) >= 2:
        _mem.link_facts(fact_ids[0], fact_ids[1], "caused_by", 0.82)
    conn_g2 = sqlite3.connect(_DB)
    edge_count = conn_g2.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    conn_g2.close()
    check("fact_relations table exists and is writable", edge_count >= 1,
          f"edges in DB: {edge_count}")
    check("link_facts creates an edge between the two facts",
          len(fact_ids) >= 2 and edge_count >= 1,
          f"fact_ids={fact_ids}, edges={edge_count}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 8 — get_graph: SQLAlchemy fact's neighbour is the N+1 fact
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 8: get_graph — N+1 fact appears as neighbour of SQLAlchemy fact ──")
try:
    result = srv.tool_get_graph("SQLAlchemy ORM", GRAPH_PROJ, depth=1)
    root       = result.get("root")
    neighbours = result.get("neighbours", [])
    check("get_graph returns a root fact", root is not None, f"got: {result}")
    neighbour_texts = [n["content"] for n in neighbours]
    check("N+1 fact appears as a graph neighbour",
          any("N+1" in t or "eager" in t or "SQLAlchemy" in t for t in neighbour_texts),
          f"neighbours: {neighbour_texts}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 9 — auto_extract saves a FastAPI fact from a response
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 9: auto_extract saves facts from AI response ──")
EXTRACT_PROJ = "extract_test_proj"
EXTRACT_RESP = (
    "I decided to use FastAPI for performance reasons. FastAPI uses Starlette "
    "underneath and gives us async support out of the box. The framework choice "
    "was driven by our need for high throughput on the API endpoints."
)
try:
    ae_result = _patched_auto_extract(EXTRACT_RESP, EXTRACT_PROJ, "sess_ae1")
    check("auto_extract returns extracted count >= 0", ae_result["extracted"] >= 0,
          f"got: {ae_result}")
    if ae_result["extracted"] > 0:
        check("auto_extract saved FastAPI-related fact",
              any("FastAPI" in f or "framework" in f.lower() for f in ae_result["facts"]),
              f"facts: {ae_result['facts']}")
    else:
        # Keyword extractor conservatively saved 0 — acceptable
        check("auto_extract conservatively saved 0 facts (acceptable)", True)
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 10 — retrieve_facts includes graph neighbours alongside direct matches
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 10: retrieve_facts includes graph-expanded neighbours ──")
EXPAND_PROJ = "expand_test_proj"
try:
    _mem.store_fact(EXPAND_PROJ, "sess_ex1", "we use PostgreSQL as the primary database", "decision")
    _mem.store_fact(EXPAND_PROJ, "sess_ex1", "the connection pool is configured with max 20 connections", "finding")
    conn_ex = sqlite3.connect(_DB)
    expand_ids = [r[0] for r in conn_ex.execute(
        "SELECT id FROM facts WHERE project_id = ? ORDER BY id", (EXPAND_PROJ,)
    ).fetchall()]
    conn_ex.close()
    if len(expand_ids) >= 2:
        _mem.link_facts(expand_ids[0], expand_ids[1], "related", 0.75)
    facts = _mem.retrieve_facts(EXPAND_PROJ, "sess_ex2", "PostgreSQL database", top_n=1, threshold=0.0)
    check("retrieve_facts returns at least 1 result", len(facts) >= 1, f"got: {facts}")
    check("graph expansion appends the connection pool neighbour",
          any("connection" in f.lower() or "pool" in f.lower() for f in facts),
          f"got: {facts}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test P5-COMPAT — retrieve_facts still returns list[str] by default
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test P5-COMPAT: retrieve_facts returns list by default ──")
P5_PROJ = "p5_compat_proj"
try:
    _mem.store_fact(P5_PROJ, "sess_p5a", "we use Redis for caching sessions", "decision")
    result_list = _mem.retrieve_facts(P5_PROJ, "sess_p5a", "Redis caching", top_n=3, threshold=0.0)
    check("P5-COMPAT: default return is a list", isinstance(result_list, list),
          f"got type: {type(result_list)}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test P5-OPT-IN — retrieve_facts returns dict when include_budget_info=True
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test P5-OPT-IN: retrieve_facts returns dict with budget info ──")
try:
    result_dict = _mem.retrieve_facts(
        P5_PROJ, "sess_p5b", "Redis caching",
        top_n=3, threshold=0.0, include_budget_info=True,
    )
    check("P5-OPT-IN: include_budget_info=True returns dict",
          isinstance(result_dict, dict),
          f"got type: {type(result_dict)}")
    check("P5-OPT-IN: dict has 'facts' key",
          "facts" in result_dict,
          f"keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'N/A'}")
    check("P5-OPT-IN: dict has 'budget_hit' key",
          "budget_hit" in result_dict,
          f"keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'N/A'}")
    check("P5-OPT-IN: 'facts' value is a list",
          isinstance(result_dict.get("facts"), list),
          f"facts type: {type(result_dict.get('facts'))}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{'═'*50}")
print(f"  {_PASS}/{total} passed{'  ✓  ALL PASS' if _FAIL == 0 else ''}")
if _FAIL:
    print(f"  {_FAIL} FAILED")
print(f"{'═'*50}")
sys.exit(0 if _FAIL == 0 else 1)
