#!/usr/bin/env python3
"""Preflight MCP Server — local stdio MCP server backed by memory.py / tasks.py.

Exposes five tools to any MCP-compatible AI client (Claude Desktop, Cursor,
Windsurf, Zed, etc.):

    get_project_id  — compute a stable project ID from a working directory
    get_context     — retrieve memories, slots, task history, and missing slots
    store_memory    — persist a fact to long-term memory
    store_slot      — persist a project config value (framework, language, …)
    list_slots      — return all known project config as a flat dict

No external API calls are made. The user's own connected LLM is the only
intelligence. Embeddings are computed locally via fastembed.

Run:
    python ~/.config/preflight/mcp_server.py

MCP client config example (claude_desktop_config.json):
    {
      "mcpServers": {
        "preflight": {
          "command": "python",
          "args": ["~/.config/preflight/mcp_server.py"]
        }
      }
    }
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────

# Backend scripts live in ~/.config/opencode/ (current layout).
# When they are eventually migrated to ~/.config/preflight/, update this path.
_SCRIPTS_DIR = Path.home() / ".config" / "opencode"

MEMORY_PY     = str(_SCRIPTS_DIR / "memory.py")
TASKS_PY      = str(_SCRIPTS_DIR / "tasks.py")
CLASSIFIER_PY = str(_SCRIPTS_DIR / "classifier.py")
EXTRACTOR_PY  = str(_SCRIPTS_DIR / "extractor.py")

# Fail fast at startup rather than producing cryptic subprocess errors later.
for _required in (MEMORY_PY, TASKS_PY, CLASSIFIER_PY):
    if not Path(_required).exists():
        print(
            f"ERROR: required script not found: {_required}\n"
            "Expected backend scripts in ~/.config/opencode/",
            file=sys.stderr,
        )
        sys.exit(1)

# ─── Venv Python (has fastembed + all deps) ───────────────────────────────────

# Prefer the venv that ships with the opencode scripts (has fastembed installed).
# Fall back to the current interpreter only if the venv doesn't exist (CI, etc.).
_VENV_CANDIDATES = [
    _SCRIPTS_DIR / ".venv" / "Scripts" / "python.exe",  # Windows venv layout
    _SCRIPTS_DIR / ".venv" / "bin" / "python",           # Unix venv layout
]
_VENV_PYTHON: str = sys.executable  # default fallback
for _candidate in _VENV_CANDIDATES:
    try:
        if _candidate.exists():
            _VENV_PYTHON = str(_candidate)
            break
    except OSError:
        # Windows may raise OSError on symlinks it can't stat (e.g. Unix-style
        # bin/python links created by virtualenv on WSL-mounted paths).
        # Fall through and try the next candidate.
        pass

# ─── Required slots per task type ─────────────────────────────────────────────

REQUIRED_SLOTS: dict[str, list[str]] = {
    "bug":         ["language", "framework"],
    "feature":     ["language", "framework", "database"],
    "refactor":    ["language", "framework"],
    "test":        ["language", "testing_framework"],
    "docs":        [],
    "performance": ["language", "framework"],
}

# ─── Subprocess helpers ───────────────────────────────────────────────────────

def _run(args: list[str], timeout: int = 30) -> str:
    """Call a Python script with the venv interpreter and return stdout."""
    result = subprocess.run(
        [_VENV_PYTHON, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script error: {result.stderr.strip()}")
    return result.stdout.strip()


def _call_memory(*args: str) -> str:
    return _run([MEMORY_PY, *args])


def _call_tasks(*args: str) -> str:
    return _run([TASKS_PY, *args])


def _call_classifier(payload: str) -> str:
    return _run([CLASSIFIER_PY, payload])


# ─── Project ID helper ────────────────────────────────────────────────────────

def compute_project_id(cwd: str) -> str:
    """Derive a 12-char stable ID for the project rooted at `cwd`.

    Uses the git repository root when available so that all sub-directories
    of the same repo map to the same project ID.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        root = result.stdout.strip() if result.returncode == 0 else cwd
    except Exception:
        root = cwd
    return hashlib.sha256(root.encode()).hexdigest()[:12]


# ─── Tool implementations ─────────────────────────────────────────────────────

def tool_get_project_id(cwd: str) -> dict:
    return {"project_id": compute_project_id(cwd)}


def tool_get_context(
    prompt: str,
    session_id: str,
    project_id: str,
    top_n: int = 3,
) -> dict:
    """Retrieve all context needed to enrich the first prompt of a session."""

    # 1. Classify the task type (keyword-based, no LLM call)
    try:
        clf_result = json.loads(_call_classifier(json.dumps({
            "prompt": prompt,
            "useLLM": False,
            "apiKey": None,
        })))
        task_type: str | None = clf_result.get("type")
    except Exception:
        task_type = None

    # 2. Global memories (user preferences, cross-project conventions)
    try:
        global_result = json.loads(
            _call_memory("retrieve_facts", "__global__", session_id, prompt, "2", "0.25", "true")
        )
        global_facts: list[str] = global_result.get("facts", []) if isinstance(global_result, dict) else global_result
        global_budget_hit: bool = global_result.get("budget_hit", False) if isinstance(global_result, dict) else False
    except Exception:
        global_facts = []
        global_budget_hit = False

    # 3. Project-specific memories
    try:
        project_result = json.loads(
            _call_memory("retrieve_facts", project_id, session_id, prompt, str(top_n), "0.25", "true")
        )
        project_facts: list[str] = project_result.get("facts", []) if isinstance(project_result, dict) else project_result
        project_budget_hit: bool = project_result.get("budget_hit", False) if isinstance(project_result, dict) else False
        retrieved_count: int = project_result.get("retrieved_count", len(project_facts)) if isinstance(project_result, dict) else len(project_facts)
        total_candidates: int = project_result.get("total_candidates", retrieved_count) if isinstance(project_result, dict) else retrieved_count
    except Exception:
        project_facts = []
        project_budget_hit = False
        retrieved_count = 0
        total_candidates = 0

    memories = global_facts + project_facts
    budget_hit = global_budget_hit or project_budget_hit

    # 4. Similar past tasks
    try:
        similar_tasks = json.loads(
            _call_tasks("retrieve_similar", project_id, session_id, prompt, str(top_n), "0.25")
        )
    except Exception:
        similar_tasks = []

    # 5. Project slot fills
    try:
        slot_list: list[dict] = json.loads(_call_memory("retrieve_slot_fills", project_id))
        slots: dict[str, str] = {s["slot_name"]: s["value"] for s in slot_list}
    except Exception:
        slots = {}

    # 6. Missing slots for the detected task type
    required = REQUIRED_SLOTS.get(task_type or "", [])
    missing_slots = [s for s in required if s not in slots]

    # 7. Mark session as enriched so re-enrichment is skipped on follow-ups
    try:
        _call_memory("session_mark", session_id, project_id)
    except Exception:
        pass

    return {
        "memories":        memories,
        "similar_tasks":   similar_tasks,
        "slots":           slots,
        "task_type":       task_type,
        "missing_slots":   missing_slots,
        "budget_hit":      budget_hit,
        "retrieved_count": retrieved_count,
        "total_candidates": total_candidates,
    }


def tool_store_memory(
    session_id: str,
    project_id: str,
    fact: str,
    fact_type: str = "finding",
) -> dict:
    """Persist a fact to long-term memory.

    If fact_type is "preference", stores to the global project (cross-project
    user preferences) regardless of the project_id passed.
    """
    target_project = "__global__" if fact_type == "preference" else project_id
    _call_memory("store_fact", target_project, session_id, fact, fact_type)
    return {"stored": True}


def tool_store_slot(
    session_id: str,
    project_id: str,
    slot_name: str,
    value: str,
) -> dict:
    """Persist a project configuration slot (upserts — safe to call repeatedly)."""
    _call_memory("store_slot_fill", project_id, session_id, slot_name, value)
    return {"stored": True, "slot": slot_name, "value": value}


def tool_list_slots(project_id: str) -> dict:
    """Return all known project config slots as a flat {slot_name: value} dict."""
    try:
        slot_list: list[dict] = json.loads(_call_memory("retrieve_slot_fills", project_id))
        return {s["slot_name"]: s["value"] for s in slot_list}
    except Exception:
        return {}


def tool_auto_extract(
    response_text: str,
    project_id: str,
    session_id: str,
) -> dict:
    """Extract facts from an AI response in a background thread.

    Returns {"status": "queued"} immediately — never blocks the conversation.
    # I/O-bound: threading is safe here; switch to ProcessPoolExecutor if CPU-bound work is added.
    """
    import threading  # noqa: PLC0415

    def _extract_and_store() -> None:
        try:
            raw = _run(
                [EXTRACTOR_PY, json.dumps({"text": response_text, "apiKey": None})],
                timeout=15,
            )
            facts: list[str] = json.loads(raw).get("facts", [])
        except Exception:
            facts = []
        for fact in facts:
            try:
                _call_memory("store_fact", project_id, session_id, fact, "finding")
            except Exception:
                pass

    threading.Thread(target=_extract_and_store, daemon=True).start()
    return {"status": "queued"}


def tool_consolidate_memories(
    project_id: str,
    session_id: str,
) -> dict:
    """Return the last 50 live facts for LLM-assisted memory consolidation.

    Call after ~10 exchanges. Review the returned facts for contradictions
    and redundancies, then call store_memory to update stale entries.
    Contradiction syntax:
      [CONTRADICTION DETECTED: Fact ID {id} \u2014 "{old_snippet}" superseded by "{new_snippet}"]
    """
    try:
        raw = _call_memory("consolidate_memories", project_id, session_id)
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc), "facts": [], "count": 0}


def tool_get_graph(
    fact_content: str,
    project_id: str,
    depth: int = 1,
) -> dict:
    """Find the closest fact to fact_content and return its graph neighbourhood.

    Useful for understanding why a decision was made, tracing bug causes,
    or finding related conventions.
    """
    try:
        raw = _call_memory("get_graph", project_id, fact_content, str(min(depth, 2)))
        return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc), "root": None, "neighbours": []}


# ─── MCP server (stdio transport) ─────────────────────────────────────────────

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def _build_mcp_server() -> "Server":
    server = Server("preflight")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="get_project_id",
                description=(
                    "Compute a stable project ID from a working directory path. "
                    "Call this at the start of every session before get_context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cwd": {
                            "type": "string",
                            "description": "Absolute path to the current working directory.",
                        },
                    },
                    "required": ["cwd"],
                },
            ),
            mcp_types.Tool(
                name="get_context",
                description=(
                    "Retrieve memories, similar past tasks, project slot config, and "
                    "a list of missing slots for the current task. Call once per session "
                    "with the user's first prompt."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt":     {"type": "string"},
                        "session_id": {"type": "string"},
                        "project_id": {"type": "string"},
                    },
                    "required": ["prompt", "session_id", "project_id"],
                },
            ),
            mcp_types.Tool(
                name="store_memory",
                description=(
                    "Persist a fact or insight to long-term memory. Call when the user "
                    "states something important about the codebase, architecture, or "
                    "workflow. Use fact_type='preference' for cross-project user preferences."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "project_id": {"type": "string"},
                        "fact":       {"type": "string"},
                        "fact_type":  {
                            "type": "string",
                            "enum": ["finding", "decision", "preference", "snippet", "summary", "note"],
                            "default": "finding",
                        },
                    },
                    "required": ["session_id", "project_id", "fact"],
                },
            ),
            mcp_types.Tool(
                name="store_slot",
                description=(
                    "Persist a project configuration value (e.g. language, framework, "
                    "database, testing_framework). Safe to call repeatedly — upserts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "project_id": {"type": "string"},
                        "slot_name":  {"type": "string"},
                        "value":      {"type": "string"},
                    },
                    "required": ["session_id", "project_id", "slot_name", "value"],
                },
            ),
            mcp_types.Tool(
                name="list_slots",
                description="Return all stored project config slots as a flat key/value object.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                    },
                    "required": ["project_id"],
                },
            ),
            mcp_types.Tool(
                name="auto_extract",
                description=(
                    "Call after every AI response with the full response text. "
                    "Automatically extracts and saves important facts without the LLM "
                    "having to identify them manually. Non-blocking — always safe to call."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "response_text": {"type": "string"},
                        "project_id":    {"type": "string"},
                        "session_id":    {"type": "string"},
                    },
                    "required": ["response_text", "project_id", "session_id"],
                },
            ),
            mcp_types.Tool(
                name="get_graph",
                description=(
                    "Given a fact or topic string, finds the most similar stored fact "
                    "and returns its connected graph neighbours. Use when the user asks "
                    "WHY a decision was made, HOW a bug was caused, or what is RELATED "
                    "to a topic."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "fact_content": {"type": "string"},
                        "project_id":   {"type": "string"},
                        "depth": {
                            "type": "integer",
                            "default": 1,
                            "description": "Graph traversal depth. Max 2.",
                        },
                    },
                    "required": ["fact_content", "project_id"],
                },
            ),
            mcp_types.Tool(
                name="consolidate_memories",
                description=(
                    "Return the last 50 live facts for LLM-assisted consolidation. "
                    "Call after ~10 exchanges to review stored memories for contradictions "
                    "and redundancies. Then call store_memory to update or remove stale facts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["project_id", "session_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[mcp_types.TextContent]:
        try:
            if name == "get_project_id":
                result = tool_get_project_id(arguments["cwd"])
            elif name == "get_context":
                result = tool_get_context(
                    prompt=arguments["prompt"],
                    session_id=arguments["session_id"],
                    project_id=arguments["project_id"],
                )
            elif name == "store_memory":
                result = tool_store_memory(
                    session_id=arguments["session_id"],
                    project_id=arguments["project_id"],
                    fact=arguments["fact"],
                    fact_type=arguments.get("fact_type", "finding"),
                )
            elif name == "store_slot":
                result = tool_store_slot(
                    session_id=arguments["session_id"],
                    project_id=arguments["project_id"],
                    slot_name=arguments["slot_name"],
                    value=arguments["value"],
                )
            elif name == "list_slots":
                result = tool_list_slots(project_id=arguments["project_id"])
            elif name == "auto_extract":
                result = tool_auto_extract(
                    response_text=arguments["response_text"],
                    project_id=arguments["project_id"],
                    session_id=arguments["session_id"],
                )
            elif name == "get_graph":
                result = tool_get_graph(
                    fact_content=arguments["fact_content"],
                    project_id=arguments["project_id"],
                    depth=arguments.get("depth", 1),
                )
            elif name == "consolidate_memories":
                result = tool_consolidate_memories(
                    project_id=arguments["project_id"],
                    session_id=arguments["session_id"],
                )
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            result = {"error": str(exc)}

        return [mcp_types.TextContent(type="text", text=json.dumps(result))]

    return server


async def _run_server() -> None:
    server = _build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _warmup_embeddings() -> None:
    """Pre-load the fastembed model at startup so the first get_context isn't slow."""
    try:
        # Add the scripts dir to sys.path so utils.py is importable.
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from utils import embed_text  # noqa: PLC0415
        embed_text("warmup")
        print("[preflight] embedding model ready", file=sys.stderr)
    except Exception as exc:
        print(f"[preflight] embedding warmup failed: {exc}", file=sys.stderr)


def _warmup_nlp() -> None:
    """Pre-load the spaCy model at startup to avoid first-call latency (Phase 3)."""
    try:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from extractor import _warmup_nlp as _wnlp  # noqa: PLC0415
        _wnlp()
        print("[preflight] spaCy model ready", file=sys.stderr)
    except Exception as exc:
        print(f"[preflight] spaCy warmup skipped: {exc}", file=sys.stderr)


def _warmup_cross_encoder() -> None:
    """Pre-load the MS-MARCO cross-encoder at startup (Phase 4, ~22MB)."""
    try:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from utils import get_cross_encoder  # noqa: PLC0415
        enc = get_cross_encoder()
        if enc is not None:
            print("[preflight] cross-encoder ready", file=sys.stderr)
        else:
            print("[preflight] cross-encoder not installed (optional)", file=sys.stderr)
    except Exception as exc:
        print(f"[preflight] cross-encoder warmup skipped: {exc}", file=sys.stderr)


def main() -> None:
    if not _MCP_AVAILABLE:
        print(
            "ERROR: 'mcp' package not installed. Run: pip install mcp fastembed",
            file=sys.stderr,
        )
        sys.exit(1)

    _warmup_embeddings()
    _warmup_nlp()
    _warmup_cross_encoder()

    import asyncio
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
