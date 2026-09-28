"""SQLite — chat tarixi (sessiyalar + xabarlar). Vector DB emas, oddiy tarix."""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from . import config


def _connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'Yangi chat',
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


def create_session(title: str = "Yangi chat") -> dict:
    sid = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
        (sid, title, now),
    )
    conn.commit()
    conn.close()
    return {"id": sid, "title": title, "created_at": now}


def list_sessions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(sid: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_messages(sid: str, limit: int = 40) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (sid, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(sid: str, role: str, content: str):
    now = datetime.now().isoformat(timespec="seconds")
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (sid, role, content, now),
    )
    # Birinchi xabardan sarlavha yasash
    count = conn.execute(
        "SELECT COUNT(*) c FROM messages WHERE session_id=?", (sid,)
    ).fetchone()["c"]
    if count <= 2:
        title = content[:40] + ("..." if len(content) > 40 else "")
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))
    conn.commit()
    conn.close()
