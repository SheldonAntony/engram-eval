#!/usr/bin/env python3
"""Semantic memory and slot fills for the Preflight plugin.

CLI usage:
    memory.py store_fact      <project_id> <session_id> <text>
    memory.py retrieve_facts  <project_id> <session_id> <prompt> [top_n] [threshold]
    memory.py check_dedup     <key>
    memory.py mark_stored     <key>
    memory.py store_slot_fill <project_id> <session_id> <slot_name> <value>
    memory.py retrieve_slot_fills <project_id> <session_id>
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Feature 1 Bug 3: shared utilities extracted from this module
from utils import cosine_similarity, embed_text

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")


# ─── Database init ────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)

    # Feature 2: project_id column on all tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            project_id      TEXT,
            content         TEXT,
            embedding       TEXT,
            retrieval_count INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dedup (
            key TEXT PRIMARY KEY
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS slot_fills (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            project_id TEXT,
            slot_name  TEXT,
            value      TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature 2: migrations — add project_id if the table already exists
    _ensure_column(conn, "facts",      "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "facts",      "retrieval_count", "INTEGER",   "0")
    _ensure_column(conn, "facts",      "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")
    _ensure_column(conn, "slot_fills", "project_id",      "TEXT",      "'unknown'")
    _ensure_column(conn, "slot_fills", "created_at",      "TIMESTAMP", "CURRENT_TIMESTAMP")

    conn.commit()
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, col: str,
                   col_type: str, default: str) -> None:
    try:
        conn.execute(f"SELECT {col} FROM {table} LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}"
        )


# ─── Deduplication ───────────────────────────────────────────────────────────

def check_dedup(key: str) -> str:
    conn = init_db()
    row = conn.execute("SELECT key FROM dedup WHERE key = ?", (key,)).fetchone()
    conn.close()
    return "EXISTS" if row else "NEW"


def mark_stored(key: str) -> None:
    conn = init_db()
    conn.execute("INSERT OR IGNORE INTO dedup (key) VALUES (?)", (key,))
    conn.commit()
    conn.close()


# ─── Facts (semantic memory) ──────────────────────────────────────────────────

def store_fact(project_id: str, session_id: str, text: str) -> None:
    emb = embed_text(text)
    conn = init_db()
    conn.execute(
        "INSERT INTO facts (project_id, session_id, content, embedding) VALUES (?, ?, ?, ?)",
        (project_id, session_id, text, json.dumps(emb)),
    )
    conn.commit()
    conn.close()


def retrieve_facts(
    project_id: str,
    session_id: str,
    prompt: str,
    top_n: int = 3,
    threshold: float = 0.25,
) -> list[str]:
    """Feature 3: weighted retrieval — semantic + recency + frequency."""
    conn = init_db()
    cursor = conn.execute(
        """SELECT content, embedding, retrieval_count, created_at
           FROM facts
           WHERE project_id = ?
           ORDER BY id DESC LIMIT 200""",
        (project_id,),
    )
    rows = [
        (content, json.loads(emb), rc, ca)
        for content, emb, rc, ca in cursor.fetchall()
        if emb
    ]
    max_rc_row = conn.execute(
        "SELECT COALESCE(MAX(retrieval_count), 1) FROM facts WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()

    if not rows:
        return []

    prompt_emb = embed_text(prompt)
    now = datetime.now(timezone.utc)
    max_rc = max_rc_row[0] if max_rc_row and max_rc_row[0] else 1

    scored: list[tuple[float, str, int]] = []
    for content, emb, rc, created_at in rows:
        # 1. Semantic similarity (weight 0.5)
        sem = cosine_similarity(prompt_emb, emb)

        # 2. Recency score (weight 0.3)
        try:
            ca_dt = _parse_dt(created_at)
        except Exception:
            ca_dt = now
        days = max(0, (now - ca_dt).days)
        recency = 1 / (1 + days)

        # 3. Frequency score (weight 0.2)
        freq = (rc / max_rc) if max_rc > 0 else 0

        combined = 0.5 * sem + 0.3 * recency + 0.2 * freq
        scored.append((combined, content, rc))

    scored.sort(reverse=True, key=lambda x: x[0])

    # Feature 4: configurable threshold; also increment retrieval_count
    results: list[str] = []
    conn = init_db()
    for score, content, _rc in scored[:top_n]:
        if score >= threshold:
            results.append(content)
            conn.execute(
                """UPDATE facts
                   SET retrieval_count = retrieval_count + 1
                   WHERE content = ? AND project_id = ?""",
                (content, project_id),
            )
    conn.commit()
    conn.close()
    return results


# ─── Slot fills ───────────────────────────────────────────────────────────────

def store_slot_fill(
    project_id: str, session_id: str, slot_name: str, value: str
) -> None:
    conn = init_db()
    conn.execute(
        "INSERT INTO slot_fills (project_id, session_id, slot_name, value) VALUES (?, ?, ?, ?)",
        (project_id, session_id, slot_name, value),
    )
    conn.commit()
    conn.close()


def retrieve_slot_fills(
    project_id: str, session_id: str
) -> list[dict]:
    """Feature 1 Bug 1: filter by both session_id and project_id."""
    conn = init_db()
    # Feature 1 Bug 1 fix: was missing session_id filter (fetched across ALL sessions)
    cursor = conn.execute(
        """SELECT session_id, slot_name, value
           FROM slot_fills
           WHERE session_id = ? AND project_id = ?
           ORDER BY created_at DESC LIMIT 50""",
        (session_id, project_id),
    )
    rows = [
        {"session_id": r[0], "slot_name": r[1], "value": r[2]}
        for r in cursor.fetchall()
    ]
    conn.close()
    return rows


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_dt(raw: str | None) -> datetime:
    if not raw:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd == "store_fact":
        project_id, session_id, text = sys.argv[2], sys.argv[3], sys.argv[4]
        store_fact(project_id, session_id, text)

    elif cmd == "retrieve_facts":
        project_id, session_id, prompt = sys.argv[2], sys.argv[3], sys.argv[4]
        top_n     = int(sys.argv[5])   if len(sys.argv) > 5 else 3
        threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
        print(json.dumps(retrieve_facts(project_id, session_id, prompt, top_n, threshold)))

    elif cmd == "check_dedup":
        print(check_dedup(sys.argv[2]))

    elif cmd == "mark_stored":
        mark_stored(sys.argv[2])

    elif cmd == "store_slot_fill":
        project_id, session_id, slot_name, value = (
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        )
        store_slot_fill(project_id, session_id, slot_name, value)

    elif cmd == "retrieve_slot_fills":
        project_id, session_id = sys.argv[2], sys.argv[3]
        print(json.dumps(retrieve_slot_fills(project_id, session_id)))

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}"}), file=sys.stderr)
        sys.exit(1)
