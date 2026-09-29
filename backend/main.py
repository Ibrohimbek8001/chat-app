import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, database
from .rag_chain import RAGService

rag: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: DB + RAG ni shu yerda init qilamiz (import vaqtda emas),
    # server tez o'rnidan turadi
    global rag
    await run_in_threadpool(database.init_db)
    if config.check_api_key():
        try:
            rag = await run_in_threadpool(RAGService)
        except Exception as e:
            print(f"[RAG] init xatolik: {e}")
            rag = None
    else:
        print("[RAG] GEMINI_API_KEY topilmadi. .env ga yozing.")
    yield
    # Shutdown: hozircha maxsus tozalash shart emas


app = FastAPI(title="RAG Chat App (LangChain + Chroma + Gemini)", lifespan=lifespan)

# Frontend shu serverdan beriladi — CORS faqat aniq ruxsat berilgan originlar uchun
if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def require_rag() -> RAGService:
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY topilmadi yoki RAG init bo'lmagan. .env ni tekshiring va serverni qayta ishga tushiring.",
        )
    return rag


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, max_length=64)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=100)


ALLOWED_SUFFIXES = (".pdf", ".txt", ".md")


def _safe_filename(name: str) -> str:
    # Path traversal dan himoya: faqat fayl nomi olinadi
    clean = Path(name or "file").name.strip()
    return clean[:100] or "file"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _resolve_sid(session_id: str | None) -> str:
    if session_id and await run_in_threadpool(database.get_session, session_id):
        return session_id
    return (await run_in_threadpool(database.create_session))["id"]


# ---------- 1. Fayl yuklash -> Chroma ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    svc = require_rag()
    filename = _safe_filename(file.filename or "file")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Faqat .pdf / .txt / .md")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    tmp = Path("uploads") / f"{uuid.uuid4().hex[:8]}_{filename}"
    tmp.parent.mkdir(exist_ok=True)
    try:
        # Hajm limit bilan oqish (memory DoS dan himoya)
        size = 0
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        400, f"Fayl juda katta (>{config.MAX_UPLOAD_MB}MB)"
                    )
                f.write(chunk)
        result = await run_in_threadpool(svc.ingest, str(tmp), filename)
    finally:
        tmp.unlink(missing_ok=True)

    if result.get("chunks", 0) == 0 and not result.get("skipped"):
        raise HTTPException(400, result.get("error", "Fayl o'qilmadi"))
    total = await run_in_threadpool(svc.doc_count)
    return {"ok": True, **result, "total_vectors": total}


# ---------- 2. Oddiy chat (RAG) ----------
@app.post("/api/chat")
async def chat(body: ChatIn):
    svc = require_rag()
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Bo'sh xabar")
    sid = await _resolve_sid(body.session_id)
    history = await run_in_threadpool(database.get_messages, sid)
    out = await svc.aask(message, history)
    if out.get("error"):
        raise HTTPException(502, out["answer"])
    await run_in_threadpool(database.add_exchange, sid, message, out["answer"], out["sources"])
    return {
        "session_id": sid,
        "answer": out["answer"],
        "sources": out["sources"],
        "warning": out.get("warning"),
    }


STOPPED_NOTE = "\n\n_(javob to'xtatildi)_"
ERROR_NOTE = "\n\n_(javob xatolik sababli to'liq emas)_"


async def _stream_gen(svc: RAGService, message: str, sid: str):
    # session_id birinchi event: stop/xatolikda ham frontend sessiyani biladi
    yield _sse({"type": "session", "session_id": sid})
    history = await run_in_threadpool(database.get_messages, sid)
    full: list[str] = []
    sources: list[dict] = []
    note = STOPPED_NOTE  # generator oxirigacha yetmasa — client uzilgan
    saved = False
    try:
        async for part in svc.astream(message, history):
            if part["type"] == "sources":
                sources = part["sources"]
                yield _sse(part)
            elif part["type"] == "error":
                note = ERROR_NOTE
                yield _sse({"type": "error", "text": part["text"]})
                return
            else:
                full.append(part["text"])
                yield _sse({"type": "token", "text": part["text"]})
        answer = "".join(full)
        if answer:
            await run_in_threadpool(database.add_exchange, sid, message, answer, sources)
        saved = True
        yield _sse({"type": "done", "session_id": sid})
    except Exception as e:
        note = ERROR_NOTE
        yield _sse({"type": "error", "text": f"Server xatolik: {e}"})
    finally:
        # Stop / uzilish / xatolik: qisman javobni ham xotiraga yozamiz.
        # Sinxron chaqiruv — bekor qilingan (cancelled) generator ichida await
        # qayta bekor bo'ladi; SQLite yozuvi millisekundlar oladi.
        if not saved and full:
            try:
                database.add_exchange(sid, message, "".join(full) + note, sources)
            except Exception as e:
                print(f"[DB] qisman javobni saqlab bo'lmadi: {e}")


# ---------- 3. Streaming chat (SSE, POST — xabar URL/log ga tushmaydi) ----------
@app.post("/api/chat/stream")
async def chat_stream(body: ChatIn):
    svc = require_rag()
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Bo'sh xabar")
    sid = await _resolve_sid(body.session_id)
    return StreamingResponse(
        _stream_gen(svc, message, sid), media_type="text/event-stream"
    )


# ---------- 4-5. Tarix ----------
@app.get("/api/sessions")
async def sessions():
    return await run_in_threadpool(database.list_sessions)


@app.get("/api/sessions/{sid}")
async def session_detail(sid: str):
    s = await run_in_threadpool(database.get_session, sid)
    if not s:
        raise HTTPException(404, "Sessiya topilmadi")
    messages = await run_in_threadpool(database.get_messages, sid, 200, True)
    return {"session": s, "messages": messages}


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, body: RenameIn):
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "Bo'sh nom")
    if not await run_in_threadpool(database.rename_session, sid, title):
        raise HTTPException(404, "Sessiya topilmadi")
    return {"ok": True, "title": title}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    if not await run_in_threadpool(database.delete_session, sid):
        raise HTTPException(404, "Sessiya topilmadi")
    return {"ok": True}


# ---------- 6. Hujjatlar + statistika ----------
@app.get("/api/documents")
async def list_docs():
    svc = require_rag()
    return await run_in_threadpool(svc.list_documents)


@app.delete("/api/documents/{source}")
async def delete_doc(source: str):
    svc = require_rag()
    removed = await run_in_threadpool(svc.delete_document, source)
    if not removed:
        raise HTTPException(404, "Hujjat topilmadi")
    return {"ok": True, "removed": removed, "total_vectors": await run_in_threadpool(svc.doc_count)}


@app.delete("/api/documents")
async def clear_docs():
    svc = require_rag()
    await run_in_threadpool(svc.clear_documents)
    return {"ok": True, "total_vectors": 0}


@app.get("/api/stats")
async def stats():
    total_vectors = await run_in_threadpool(rag.doc_count) if rag else 0
    sessions_list = await run_in_threadpool(database.list_sessions)
    return {
        "vectors": total_vectors,
        "sessions": len(sessions_list),
        "rag_ready": rag is not None,
        "limits": {
            "max_message_chars": config.MAX_MESSAGE_CHARS,
            "max_upload_mb": config.MAX_UPLOAD_MB,
        },
    }


@app.get("/api/health")
async def health():
    return {"ok": True, "rag_ready": rag is not None}


# ---------- Frontend (hamma API dan KEYIN mount qilinadi!) ----------
FRONT = Path(__file__).resolve().parent.parent / "frontend"
if FRONT.exists():
    app.mount("/", StaticFiles(directory=str(FRONT), html=True), name="frontend")
