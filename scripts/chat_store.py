"""
chat_store.py

SQLite persistence for the Streamlit app's chat sessions. Local-first,
no server - single file at chat_history.db in the project root
(gitignored, same as vector_store/).

Schema:
  chats(id, title, created_at, updated_at)
  messages(id, chat_id, role, content, sources_json, created_at)

sources_json stores the list of formatted source strings as a JSON
array (TEXT column) - it's display-only data, not queried, so no need
for a separate sources table.

Design notes:
- updated_at on chats is bumped every time a message is added, so the
  sidebar chat list can sort by "most recently active" instead of
  "most recently created".
- Deleting a chat cascades to its messages (ON DELETE CASCADE).
- edit_message_and_truncate() deletes every message AFTER the edited
  one in the same chat, not just the edited one - this matches how
  "edit and resubmit" should behave: the conversation branches from
  the edit point, old replies to a question that no longer exists in
  its original form should not stay in the history.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn(db_path=DB_PATH):
    """
    Yields a sqlite3 connection with foreign keys and row factory set
    up correctly. Commits on success, rolls back on exception, always
    closes. Use this for every DB operation - don't open raw
    connections elsewhere in the app.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Creates the chats/messages tables if they don't exist yet. Safe to call every app startup."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def create_chat(title="New chat", db_path=DB_PATH):
    """Creates a new empty chat and returns its id."""
    now = _now()
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return cur.lastrowid


def list_chats(db_path=DB_PATH):
    """Returns all chats ordered by most-recently-active first, as a list of dicts."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat(chat_id, db_path=DB_PATH):
    """Returns one chat's metadata as a dict, or None if it doesn't exist."""
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        return dict(row) if row else None


def rename_chat(chat_id, new_title, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))


def delete_chat(chat_id, db_path=DB_PATH):
    """Deletes a chat and all its messages (cascade)."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def touch_chat(chat_id, db_path=DB_PATH):
    """Bumps a chat's updated_at to now - call this whenever a message is added to it."""
    with get_conn(db_path) as conn:
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))


def add_message(chat_id, role, content, sources=None, db_path=DB_PATH):
    """
    Adds one message to a chat and bumps the chat's updated_at.
    sources: optional list of strings (formatted source lines), stored as JSON.
    Returns the new message's id.
    """
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    now = _now()
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, sources_json, now),
        )
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        return cur.lastrowid


def get_messages(chat_id, db_path=DB_PATH):
    """
    Returns every message in a chat, oldest first, as a list of dicts
    with keys: id, role, content, sources (already parsed from JSON
    back into a list, or None), created_at.
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, role, content, sources_json, created_at FROM messages "
            "WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        messages = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d.pop("sources_json")) if d["sources_json"] else None
            messages.append(d)
        return messages


def edit_message_and_truncate(message_id, new_content, db_path=DB_PATH):
    """
    Edits a user message's content AND deletes every message that came
    AFTER it in the same chat (including the assistant reply that
    followed it). This is what makes "edit and resubmit" correct: the
    conversation branches from the edit point, so replies to the old
    version of the question must not survive.

    Returns the chat_id the edited message belongs to, so the caller
    can re-run retrieval/generation for the new content and append a
    fresh assistant reply.
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT chat_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No message with id {message_id}")
        chat_id = row["chat_id"]

        conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", (new_content, message_id)
        )
        conn.execute(
            "DELETE FROM messages WHERE chat_id = ? AND id > ?", (chat_id, message_id)
        )
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))
        return chat_id


def search_chats(query, db_path=DB_PATH):
    """
    Case-insensitive substring search over message content AND chat
    titles. Returns a list of dicts: {chat_id, chat_title, message_id,
    role, snippet}, one row per matching message (a chat can appear
    more than once if multiple messages match). Ordered by chat
    updated_at descending, so the most recently active matching chats
    surface first.

    Plain SQL LIKE, not FTS5 - the dataset size here (a personal chat
    history) doesn't need a full-text index, and LIKE keeps this file
    dependency-free.
    """
    like_query = f"%{query}%"
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.id AS message_id, m.chat_id, m.role, m.content,
                   c.title AS chat_title, c.updated_at
            FROM messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE m.content LIKE ? OR c.title LIKE ?
            ORDER BY c.updated_at DESC
            """,
            (like_query, like_query),
        ).fetchall()

        results = []
        for r in rows:
            content = r["content"]
            idx = content.lower().find(query.lower())
            if idx == -1:
                snippet = content[:120]
            else:
                start = max(0, idx - 40)
                end = min(len(content), idx + len(query) + 80)
                snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
            results.append({
                "chat_id": r["chat_id"],
                "chat_title": r["chat_title"],
                "message_id": r["message_id"],
                "role": r["role"],
                "snippet": snippet,
            })
        return results


def generate_title_from_first_message(content, max_words=8):
    """
    Simple deterministic title generator: first N words of the first
    user message, truncated with "...". No API call - keeps chat
    creation instant and free. Called once, right after the first
    message of a new chat is saved.
    """
    words = content.strip().split()
    if not words:
        return "New chat"
    title = " ".join(words[:max_words])
    if len(words) > max_words:
        title += "..."
    return title
