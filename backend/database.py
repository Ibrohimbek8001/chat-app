"""SQLite — chat tarixi (sessiyalar + xabarlar). Vector DB emas, oddiy tarix."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config

DEFAULT_TITLE = "Yangi chat"


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


def _columns(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'Yangi chat',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT,
                sources TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        # Eski bazalar uchun migratsiya (ustunlar keyin qo'shilgan)
        if "updated_at" not in _columns(conn, "sessions"):
            cur.execute("ALTER TABLE sessions ADD COLUMN updated_at TEXT")
            cur.execute("UPDATE sessions SET updated_at = created_at")
        if "sources" not in _columns(conn, "messages"):
            cur.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
        # Tez-tez ishlatiladigan so'rovlar uchun indekslar
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id, id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions(updated_at DESC)
        """)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_session(title: str = DEFAULT_TITLE) -> dict:
    now = _now()
    # 8-belgili id da kolliziya ehtimoli kichik, lekin retry bilan himoya;
    # oxirgi urinish to'liq uuid bilan
    candidates = [str(uuid.uuid4())[:8] for _ in range(3)] + [str(uuid.uuid4())]
    conn = _connect()
    try:
        for sid in candidates:
            try:
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (sid, title, now, now),
                )
                conn.commit()
                return {"id": sid, "title": title, "created_at": now, "updated_at": now}
            except sqlite3.IntegrityError:
                continue
    finally:
        conn.close()
    raise RuntimeError("Sessiya yaratib bo'lmadi")


def list_sessions(limit: int = 100) -> list[dict]:
    """Oxirgi faollik bo'yicha tartiblangan, xabari bor sessiyalar."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at FROM sessions s
            WHERE EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id)
            ORDER BY s.updated_at DESC LIMIT ?
            """,
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


def get_messages(sid: str, limit: int = 40, with_sources: bool = False) -> list[dict]:
    """Oxirgi `limit` ta xabarni xronologik tartibda qaytaradi.

    Avvalgi bug: ORDER BY id ASC LIMIT 40 birinchi 40 tani berardi.
    To'g'risi: DESC olib, keyin reverse qilish.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, content, sources FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (sid, limit),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in reversed(rows):
        m = {"role": r["role"], "content": r["content"]}
        if with_sources:
            try:
                m["sources"] = json.loads(r["sources"]) if r["sources"] else None
            except ValueError:
                m["sources"] = None
        out.append(m)
    return out


def add_exchange(sid: str, question: str, answer: str, sources: list[dict] | None = None):
    """Savol + javobni bitta tranzaksiyada yozadi.

    Sarlavha faqat sessiyaning birinchi savolidan yasaladi (avval AI javobi
    sarlavhani ustidan yozib yuborardi).
    """
    now = _now()
    conn = _connect()
    try:
        is_first = conn.execute(
            "SELECT NOT EXISTS (SELECT 1 FROM messages WHERE session_id=?) AS f", (sid,)
        ).fetchone()["f"]
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
            (sid, question, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at, sources) VALUES (?, 'assistant', ?, ?, ?)",
            (sid, answer, now, json.dumps(sources or [], ensure_ascii=False)),
        )
        if is_first:
            title = question[:40] + ("..." if len(question) > 40 else "")
            conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, now, sid)
            )
        else:
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
        conn.commit()
    finally:
        conn.close()


def rename_session(sid: str, title: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_session(sid: str) -> bool:
    conn = _connect()
    try:
        # FK cascade ga tayanmaymiz — eski bazalarda foreign_keys o'chiq bo'lgan
        conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        cur = conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
