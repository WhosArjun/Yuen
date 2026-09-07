"""
agent/memory.py

Local memory for the agent, backed by SQLite (no paid vector DB, no
external embedding API). Stores:

    - conversation history (per session) so recent context can be
      replayed to the model
    - a small key/value "facts" table for things worth remembering
      about the current project (explicitly saved, not everything)
    - the current task plan (see planner.py, which uses this module)

This is intentionally simple: full conversation history is available on
disk, but only a recent window plus explicitly-saved facts are sent to
the model on each turn, to keep prompts small and relevant.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Memory:
    def __init__(self, db_path: str, session_id: str = "default"):
        self.db_path = db_path
        self.session_id = session_id
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------

    def add_message(self, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, time.time()),
        )
        self.conn.commit()

    def get_recent_messages(self, limit: int = 20) -> List[Dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (self.session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ------------------------------------------------------------
    # Facts (explicit, curated memory -- not a dump of everything)
    # ------------------------------------------------------------

    def remember(self, key: str, value: Any, category: str = "general"):
        serialized = value if isinstance(value, str) else json.dumps(value)
        self.conn.execute(
            "INSERT INTO facts (key, value, category, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, category=excluded.category, "
            "updated_at=excluded.updated_at",
            (key, serialized, category, time.time()),
        )
        self.conn.commit()

    def recall(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self, category: Optional[str] = None) -> Dict[str, str]:
        if category:
            rows = self.conn.execute(
                "SELECT key, value FROM facts WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def forget(self, key: str):
        self.conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        self.conn.commit()

    def search_facts(self, query: str) -> Dict[str, str]:
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT key, value FROM facts WHERE key LIKE ? OR value LIKE ?", (like, like)
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------
    # Plan storage (used by planner.py)
    # ------------------------------------------------------------

    def save_plan(self, steps: List[str]):
        self.conn.execute("DELETE FROM plan_steps WHERE session_id = ?", (self.session_id,))
        now = time.time()
        for i, desc in enumerate(steps):
            self.conn.execute(
                "INSERT INTO plan_steps (session_id, step_index, description, status, "
                "created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
                (self.session_id, i, desc, now, now),
            )
        self.conn.commit()

    def update_step_status(self, step_index: int, status: str):
        self.conn.execute(
            "UPDATE plan_steps SET status = ?, updated_at = ? WHERE session_id = ? AND step_index = ?",
            (status, time.time(), self.session_id, step_index),
        )
        self.conn.commit()

    def get_plan(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT step_index, description, status FROM plan_steps "
            "WHERE session_id = ? ORDER BY step_index",
            (self.session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_plan(self):
        self.conn.execute("DELETE FROM plan_steps WHERE session_id = ?", (self.session_id,))
        self.conn.commit()
