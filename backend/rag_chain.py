"""RAG yuragi — 100% LangChain orqali.

Zanjir:
  Fayl -> pypdf (matn) -> RecursiveCharacterTextSplitter (bo'laklash)
       -> GoogleGenerativeAIEmbeddings -> Chroma (VectorDB, diskda saqlanadi)
  Savol -> retriever (top-4 bo'lak) -> ChatGoogleGenerativeAI -> javob
"""
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


class RAGService:
    # Tarix + kontekst limitlari (token overflow dan himoya)
    MAX_HISTORY_MESSAGES = 10
    MAX_HISTORY_CHARS = 6000
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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
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
        # .txt / .md — katta fayllarni kesish (masalan 500KB)
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text[:500_000]

    def ingest(self, file_path: str, filename: str) -> dict:
        text = self._load_file_text(file_path).strip()
        if not text:
            return {"chunks": 0, "error": "Fayldan matn o'qilmadi"}
        return self.ingest_text(text, source=filename)

    def ingest_text(self, text: str, source: str) -> dict:
        # Bo'sh yoki juda katta matnni erta kesish
        text = (text or "").strip()[:500_000]
        if not text:
            return {"chunks": 0, "error": "Bo'sh matn"}
        docs = self.splitter.create_documents(
            [text],
            metadatas=[{"source": source}],
        )
        if not docs:
            return {"chunks": 0, "error": "Bo'laklashda xatolik"}
        # Chroma add_documents sync — main.py da threadpool da chaqiriladi
        # Katta hujjatda chunk portlashdan himoya
        docs = docs[:500]
        self.vectorstore.add_documents(docs)
        return {"chunks": len(docs), "source": source}

    def clear_documents(self):
        # Kolleksiyani tozalash: qayta yaratish eng ishonchli usul
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass
        self.vectorstore = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="chat_docs",
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )

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

    def get_docs(self, query: str) -> list[Document]:
        # doc_count() alohida so'rov edi — to'g'ridan-to'g'ri retriever ga uramiz,
        # bo'sh kolleksiyada Chroma bo'sh ro'yxat qaytaradi
        try:
            return self.retriever.invoke(query)
        except Exception:
            return []

    async def aget_docs(self, query: str) -> list[Document]:
        """Async yo'l — event loop ni bloklamaydi."""
        try:
            return await self.retriever.ainvoke(query)
        except Exception:
            return []

    def _truncate_history(self, history: list[dict]) -> list[dict]:
        """Oxirgi N xabar + belgi limiti bo'yicha kesish."""
        recent = history[-self.MAX_HISTORY_MESSAGES :]
        total = 0
        out: list[dict] = []
        # Oxiridan boshlab yig'amiz, keyin tartiblaymiz
        for m in reversed(recent):
            c = str(m.get("content", ""))
            total += len(c)
            if total > self.MAX_HISTORY_CHARS:
                break
            out.append(m)
        return list(reversed(out))

    def _build_messages(self, query: str, docs: list[Document], history: list[dict]) -> list:
        """LangChain chat modeliga beriladigan xabarlar ro'yxati."""
        context = "\n\n---\n\n".join(
            [f"[Manba: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs]
        )
        if not context:
            context = "(Hujjat yuklanmagan yoki mos parcha topilmadi)"

        messages: list = [("system", f"{SYSTEM_PROMPT}\n\nKONTEKST:\n{context}")]
        # oxirgi N xabarni tarix sifatida qo'shish (memory o'rniga),
        # belgi limiti bilan kesilgan
        for m in self._truncate_history(history):
            role = "human" if m.get("role") == "user" else "ai"
            content = str(m.get("content", ""))[:2000]
            messages.append((role, content))
        messages.append(("human", query))
        return messages

    def _sources(self, docs: list[Document]) -> list[dict]:
        out = []
        for d in docs:
            out.append({
                "source": d.metadata.get("source", "?"),
                "preview": d.page_content[:200],
            })
        return out

    def ask(self, query: str, history: list[dict]) -> dict:
        docs = self.get_docs(query)
        messages = self._build_messages(query, docs, history)
        try:
            resp = self.llm.invoke(messages)
        except Exception as e:
            return {
                "answer": f"LLM xatolik: {e}. API kalit/kvota ni tekshiring.",
                "sources": self._sources(docs),
                "error": True,
            }
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"answer": content, "sources": self._sources(docs)}

    async def aask(self, query: str, history: list[dict]) -> dict:
        docs = await self.aget_docs(query)
        messages = self._build_messages(query, docs, history)
        try:
            resp = await self.llm.ainvoke(messages)
        except Exception as e:
            return {
                "answer": f"LLM xatolik: {e}. API kalit/kvota ni tekshiring.",
                "sources": self._sources(docs),
                "error": True,
            }
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"answer": content, "sources": self._sources(docs)}

    async def astream(self, query: str, history: list[dict]):
        """Streaming uchun async generator — main.py da SSE ga o'raladi."""
        docs = await self.aget_docs(query)
        messages = self._build_messages(query, docs, history)
        # Avval manbalarni yuborish uchun saqlab qo'yamiz
        yield {"type": "sources", "sources": self._sources(docs)}
        try:
            async for chunk in self.llm.astream(messages):
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    text = "".join([str(p) for p in text])
                if text:
                    yield {"type": "token", "text": str(text)}
        except Exception as e:
            yield {"type": "error", "text": f"LLM xatolik: {e}"}
