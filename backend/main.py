import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, database
from .rag_chain import RAGService

app = FastAPI(title="RAG Chat App (LangChain + Chroma + Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()
rag: RAGService | None = None

if config.check_api_key():
    try:
        rag = RAGService()
    except Exception as e:
        print(f"[RAG] init xatolik: {e}")
        rag = None
else:
    print("[RAG] GEMINI_API_KEY topilmadi. .env ga yozing.")


def require_rag() -> RAGService:
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY topilmadi yoki RAG init bo'lmagan. .env ni tekshiring va serverni qayta ishga tushiring.",
        )
    return rag


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


# ---------- 1. Fayl yuklash -> Chroma ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    svc = require_rag()
    suffix = Path(file.filename or "file").suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "Faqat .pdf / .txt / .md")
    tmp = Path("uploads") / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    tmp.parent.mkdir(exist_ok=True)
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = svc.ingest(str(tmp), file.filename or "file")
    tmp.unlink(missing_ok=True)
    if result.get("chunks", 0) == 0:
        raise HTTPException(400, result.get("error", "Fayl o'qilmadi"))
    return {"ok": True, **result, "total_vectors": svc.doc_count()}


# ---------- 2. Oddiy chat (RAG) ----------
@app.post("/api/chat")
async def chat(body: ChatIn):
    svc = require_rag()
    sid = body.session_id or database.create_session()["id"]
    if not database.get_session(sid):
        sid = database.create_session()["id"]
    history = database.get_messages(sid)
    out = svc.ask(body.message, history)
    database.add_message(sid, "user", body.message)
    database.add_message(sid, "assistant", out["answer"])
    return {"session_id": sid, "answer": out["answer"], "sources": out["sources"]}


# ---------- 3. Streaming chat (SSE) ----------
@app.get("/api/chat/stream")
async def chat_stream(message: str, session_id: str | None = None):
    svc = require_rag()
    sid = session_id or database.create_session()["id"]
    if not database.get_session(sid):
        sid = database.create_session()["id"]
    history = database.get_messages(sid)

    async def gen():
        full = []
        async for part in svc.astream(message, history):
            if part["type"] == "sources":
                yield f"data: {json.dumps({'type': 'sources', 'sources': part['sources']}, ensure_ascii=False)}\n\n"
            else:
                full.append(part["text"])
                yield f"data: {json.dumps({'type': 'token', 'text': part['text']}, ensure_ascii=False)}\n\n"
        # tugagach DB ga yozamiz
        answer = "".join(full)
        database.add_message(sid, "user", message)
        database.add_message(sid, "assistant", answer)
        yield f"data: {json.dumps({'type': 'done', 'session_id': sid}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- 4-5. Tarix ----------
@app.get("/api/sessions")
def sessions():
    return database.list_sessions()


@app.get("/api/sessions/{sid}")
def session_detail(sid: str):
    s = database.get_session(sid)
    if not s:
        raise HTTPException(404, "Sessiya topilmadi")
    return {"session": s, "messages": database.get_messages(sid, limit=200)}


# ---------- 6. Hujjatlarni tozalash + statistika ----------
@app.delete("/api/documents")
def clear_docs():
    svc = require_rag()
    svc.clear_documents()
    return {"ok": True, "total_vectors": 0}


@app.get("/api/stats")
def stats():
    total_vectors = rag.doc_count() if rag else 0
    return {
        "vectors": total_vectors,
        "sessions": len(database.list_sessions()),
        "rag_ready": rag is not None,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "rag_ready": rag is not None}


# ---------- Frontend (hamma API dan KEYIN mount qilinadi!) ----------
FRONT = Path(__file__).resolve().parent.parent / "frontend"
if FRONT.exists():
    app.mount("/", StaticFiles(directory=str(FRONT), html=True), name="frontend")
