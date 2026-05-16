# Preflight — Local MCP Memory Server

Gives any MCP-compatible AI client (Claude Desktop, Cursor, Windsurf, Zed)
persistent memory that survives across sessions and across clients. Facts,
preferences, and project config are stored locally in SQLite — no cloud, no
API key required.

---

## Install

```bash
pip install mcp fastembed
```

`fastembed` downloads a ~90 MB embedding model on first use. The server
pre-loads it at startup so the delay is visible (not silent during a request).

---

## Claude Desktop config

Add the following to `claude_desktop_config.json`
(usually at `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "preflight": {
      "command": "python",
      "args": ["C:/Users/<USERNAME>/.config/preflight/mcp_server.py"]
    }
  }
}
```

Replace `<USERNAME>` with your Windows username (e.g. `Sheldon Antony`).
Use forward slashes in the path.

---

## Tools

The server exposes five tools. The LLM should call them as follows:

| Tool | When to call |
|---|---|
| `get_project_id` | At the start of every session, before `get_context`. Pass the current working directory. |
| `get_context` | Once per session with the user's first prompt. Returns memories, similar past tasks, stored project config, and a list of slots the user hasn't filled yet. |
| `store_memory` | When the user states something important about the codebase, architecture, or workflow. Use `fact_type="preference"` for cross-project user preferences (e.g. "always use type hints"). |
| `store_slot` | After the user answers a question about project setup (e.g. "what framework are you using?"). Stores key/value config that persists across all future sessions for this project. |
| `list_slots` | To check what project config is already known (language, framework, database, etc.) before asking the user redundant setup questions. |

**What the LLM should do with the output of `get_context`:**
- Prepend `memories` and `similar_tasks` to its system context so past decisions are visible.
- Use `slots` to avoid asking setup questions the user has already answered.
- Ask the user for any values in `missing_slots` (e.g. "What testing framework do you use?") and then call `store_slot` with the answer.

---

## Smoke test

```bash
python ~/.config/preflight/test_mcp.py
```

Runs 15 assertions covering all six tool behaviors without needing the MCP
protocol or a live AI client.

---

## File layout

```
~/.config/opencode/    memory.py, tasks.py, classifier.py, utils.py, memory.db
~/.config/preflight/   mcp_server.py, test_mcp.py, README.md
```
