"""RAG yuragi — 100% LangChain orqali.

Zanjir:
  Fayl -> pypdf (matn) -> RecursiveCharacterTextSplitter (bo'laklash)
       -> GoogleGenerativeAIEmbeddings -> Chroma (VectorDB, diskda saqlanadi)
  Savol (+tarix) -> mustaqil qidiruv so'rovi -> retriever (top-K bo'lak)
       -> ChatGoogleGenerativeAI -> javob
"""
import hashlib
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    # yangi paket tuzilmasi
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    # yangi: langchain-chroma
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from . import config

SYSTEM_PROMPT = """Sen o'zbek tilida javob beradigan yordamchisan.
Quyidagi KONTEKST (hujjatlardan topilgan parchalar) ga asoslanib javob ber.
Qoidalar:
1. Javobingni asosan KONTEKST dan ol.
2. Kontekstdan topa olmasang, o'z bilimingdan javob ber, lekin "Hujjatlarda topilmadi, umumiy bilimdan javob beraman:" deb boshlab ber.
3. Hech qachon ma'lumot to'qima.
4. Qisqa va aniq yoz."""

CONDENSE_PROMPT = """Senga suhbat tarixi va foydalanuvchining yangi savoli beriladi.
Yangi savolni tarixsiz ham tushunarli bo'ladigan MUSTAQIL qidiruv so'roviga aylantir:
olmoshlarni ("u", "bu", "ular", "it", "they") tarixdagi aniq nomlar bilan almashtir.
Savol tilini saqla. Faqat qayta yozilgan savolni qaytar — izohsiz, qo'shtirnoqsiz.
Savol allaqachon mustaqil bo'lsa, uni o'zgartirmasdan qaytar."""

MAX_CHUNKS_PER_FILE = 500
MAX_TEXT_CHARS = 500_000


def _text(content) -> str:
    """Gemini javobi str yoki qismlar ro'yxati ({"type": "text", ...}) bo'lishi mumkin."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content or "")


class RAGService:
    # Tarix + kontekst limitlari (token overflow dan himoya)
    MAX_HISTORY_MESSAGES = 10
    MAX_HISTORY_CHARS = 6000
    MAX_MESSAGE_CHARS_IN_HISTORY = 2000
    MAX_PDF_PAGES = 200

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=config.GEMINI_API_KEY,
            model=config.EMBED_MODEL,
        )
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.CHAT_MODEL,
            temperature=0.3,
        )
        # Qidiruv so'rovini qayta yozish uchun — deterministik
        self.rewriter = ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.CHAT_MODEL,
            temperature=0,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self._open_collection()

    def _open_collection(self):
        self.vectorstore = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="chat_docs",
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )

    # ---------- INGEST ----------

    def _load_file_text(self, file_path: str) -> str:
        p = Path(file_path)
        if p.suffix.lower() == ".pdf":
            reader = PdfReader(str(p))
            # Katta PDF larni chegaralash (memory + embedding quota himoyasi)
            pages = reader.pages[: self.MAX_PDF_PAGES]
            return "\n".join([(page.extract_text() or "") for page in pages])
        # .txt / .md — katta fayllarni ingest_text kesadi
        return p.read_text(encoding="utf-8", errors="ignore")

    def ingest(self, file_path: str, filename: str) -> dict:
        text = self._load_file_text(file_path).strip()
        if not text:
            return {"chunks": 0, "error": "Fayldan matn o'qilmadi"}
        return self.ingest_text(text, source=filename)

    def ingest_text(self, text: str, source: str) -> dict:
        text = (text or "").strip()
        truncated = len(text) > MAX_TEXT_CHARS
        text = text[:MAX_TEXT_CHARS]
        if not text:
            return {"chunks": 0, "error": "Bo'sh matn"}
        docs = self.splitter.create_documents([text], metadatas=[{"source": source}])
        if not docs:
            return {"chunks": 0, "error": "Bo'laklashda xatolik"}
        # Katta hujjatda chunk portlashdan himoya
        if len(docs) > MAX_CHUNKS_PER_FILE:
            truncated = True
            docs = docs[:MAX_CHUNKS_PER_FILE]

        # Kontent hash -> barqaror ID: bir faylni qayta yuklash dublikat
        # vektorlar yaratmaydi (va embedding kvotasi behuda sarflanmaydi)
        by_id: dict[str, Document] = {}
        for d in docs:
            h = hashlib.sha256(f"{source}\0{d.page_content}".encode("utf-8")).hexdigest()
            by_id.setdefault(h[:32], d)
        existing = set(self.vectorstore.get(ids=list(by_id), include=[]).get("ids", []))
        new_ids = [i for i in by_id if i not in existing]
        if new_ids:
            # Chroma add_documents sync — main.py da threadpool da chaqiriladi
            self.vectorstore.add_documents([by_id[i] for i in new_ids], ids=new_ids)
        return {
            "chunks": len(new_ids),
            "skipped": len(docs) - len(new_ids),
            "truncated": truncated,
            "source": source,
        }

    def clear_documents(self):
        # Kolleksiyani tozalash: qayta yaratish eng ishonchli usul
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass
        self._open_collection()

    def list_documents(self) -> list[dict]:
        data = self.vectorstore.get(include=["metadatas"])
        counts = Counter((m or {}).get("source", "?") for m in data.get("metadatas", []))
        return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]

    def delete_document(self, source: str) -> int:
        ids = self.vectorstore.get(where={"source": source}, include=[]).get("ids", [])
        if ids:
            self.vectorstore.delete(ids=ids)
        return len(ids)

    def doc_count(self) -> int:
        # Chroma versiyalari bo'yicha fallback bilan
        try:
            col = getattr(self.vectorstore, "_collection", None)
            if col is not None:
                return col.count()
            data = self.vectorstore.get(include=[])
            return len(data.get("ids", []))
        except Exception:
            return 0

    # ---------- RETRIEVE + GENERATE (LangChain) ----------

    def _clip_history(self, history: list[dict]) -> list[dict]:
        """Oxirgi N xabar + belgi limiti bo'yicha kesish.

        Har xabar avval kesiladi, keyin sanaladi — shunda eng oxirgi xabar
        har doim sig'adi (avval bitta uzun xabar butun tarixni o'chirardi).
        """
        recent = history[-self.MAX_HISTORY_MESSAGES :]
        total = 0
        out: list[dict] = []
        for m in reversed(recent):
            content = str(m.get("content", ""))[: self.MAX_MESSAGE_CHARS_IN_HISTORY]
            total += len(content)
            if total > self.MAX_HISTORY_CHARS:
                break
            out.append({"role": m.get("role"), "content": content})
        return list(reversed(out))

    async def _condense_query(self, query: str, history: list[dict]) -> str:
        """Tarix + yangi savol -> mustaqil qidiruv so'rovi. Xatoda asl savol."""
        if not config.QUERY_REWRITE or not history:
            return query
        lines = []
        for m in history[-6:]:
            who = "Foydalanuvchi" if m.get("role") == "user" else "Yordamchi"
            lines.append(f"{who}: {str(m.get('content', ''))[:500]}")
        try:
            resp = await self.rewriter.ainvoke([
                ("system", CONDENSE_PROMPT),
                ("human", "TARIX:\n" + "\n".join(lines) + f"\n\nYANGI SAVOL: {query}"),
            ])
            rewritten = _text(resp.content).strip().strip('"').strip()
        except Exception as e:
            print(f"[RAG] query rewrite xatolik: {e}")
            return query
        if not rewritten or len(rewritten) > 1000:
            return query
        return rewritten

    async def aretrieve(self, query: str, history: list[dict]) -> tuple[list[Document], str | None]:
        """(hujjatlar, xatolik matni). Xatolik jimgina yutilmaydi — foydalanuvchiga
        "hujjat topilmadi" va "embedding ishlamayapti" farqini ko'rsatish uchun."""
        search_query = await self._condense_query(query, history)
        try:
            return await self.retriever.ainvoke(search_query), None
        except Exception as e:
            print(f"[RAG] retriever xatolik: {e}")
            return [], f"Hujjat qidiruvida xatolik: {e}"

    def _build_messages(self, query: str, docs: list[Document], history: list[dict]) -> list:
        """LangChain chat modeliga beriladigan xabarlar ro'yxati."""
        context = "\n\n---\n\n".join(
            [f"[Manba: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs]
        )
        if not context:
            context = "(Hujjat yuklanmagan yoki mos parcha topilmadi)"

        messages: list = [("system", f"{SYSTEM_PROMPT}\n\nKONTEKST:\n{context}")]
        # oxirgi N xabarni tarix sifatida qo'shish (memory o'rniga)
        for m in self._clip_history(history):
            role = "human" if m["role"] == "user" else "ai"
            messages.append((role, m["content"]))
        messages.append(("human", query))
        return messages

    def _sources(self, docs: list[Document]) -> list[dict]:
        return [
            {"source": d.metadata.get("source", "?"), "preview": d.page_content[:200]}
            for d in docs
        ]

    async def aask(self, query: str, history: list[dict]) -> dict:
        docs, retrieval_error = await self.aretrieve(query, history)
        messages = self._build_messages(query, docs, history)
        try:
            resp = await self.llm.ainvoke(messages)
        except Exception as e:
            return {
                "answer": f"LLM xatolik: {e}. API kalit/kvota ni tekshiring.",
                "sources": self._sources(docs),
                "error": True,
            }
        return {
            "answer": _text(resp.content),
            "sources": self._sources(docs),
            "warning": retrieval_error,
        }

    async def astream(self, query: str, history: list[dict]):
        """Streaming uchun async generator — main.py da SSE ga o'raladi."""
        docs, retrieval_error = await self.aretrieve(query, history)
        messages = self._build_messages(query, docs, history)
        yield {"type": "sources", "sources": self._sources(docs), "warning": retrieval_error}
        try:
            async for chunk in self.llm.astream(messages):
                text = _text(getattr(chunk, "content", ""))
                if text:
                    yield {"type": "token", "text": text}
        except Exception as e:
            yield {"type": "error", "text": f"LLM xatolik: {e}"}
