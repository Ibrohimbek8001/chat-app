"""SQLite — chat tarixi (sessiyalar + xabarlar). Vector DB emas, oddiy tarix."""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Concurrent yozishda "database is locked" ni kamaytirish
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn


def init_db():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
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
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        # Tez-tez ishlatiladigan so'rovlar uchun indekslar
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id, id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_created
            ON sessions(created_at DESC)
        """)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_session(title: str = "Yangi chat") -> dict:
    now = _now()
    # 8-belgili id da kolliziya ehtimoli kichik, lekin retry bilan himoya
    for _ in range(3):
        sid = str(uuid.uuid4())[:8]
        try:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                    (sid, title, now),
                )
                conn.commit()
            finally:
                conn.close()
            return {"id": sid, "title": title, "created_at": now}
        except sqlite3.IntegrityError:
            continue
    # oxirgi urinish to'liq uuid bilan
    sid = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": sid, "title": title, "created_at": now}


def list_sessions(limit: int = 100) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_session(sid: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_messages(sid: str, limit: int = 40) -> list[dict]:
    """Oxirgi `limit` ta xabarni xronologik tartibda qaytaradi.

    Avvalgi bug: ORDER BY id ASC LIMIT 40 birinchi 40 tani berardi.
    To'g'risi: DESC olib, keyin reverse qilish.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (sid, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in reversed(rows)]


def add_message(sid: str, role: str, content: str):
    now = _now()
    conn = _connect()
    try:
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
    finally:
        conn.close()
