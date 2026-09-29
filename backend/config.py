import os
from pathlib import Path

from dotenv import load_dotenv

# backend/ dan loyiha ildizidagi .env ni topish (uvicorn qayerdan
# ishga tushirilishidan qat'iy nazar ishlaydi)
_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _resolve_path(value: str, default: str) -> str:
    """Nisbiy path larni loyiha ildiziga nisbatan absolut qiladi."""
    v = (value or default).strip()
    if not v:
        v = default
    p = Path(v)
    if not p.is_absolute():
        p = _ROOT / p
    return str(p)


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int((value or "").strip() or default)
    except (ValueError, AttributeError):
        return default


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001").strip()
CHROMA_DIR = _resolve_path(os.getenv("CHROMA_DIR", "./chroma_db"), "./chroma_db")
DB_PATH = _resolve_path(os.getenv("DB_PATH", "./chat.db"), "./chat.db")
RETRIEVER_K = _safe_int(os.getenv("RETRIEVER_K", "4"), 4)
# Xavfsizlik chegaralari
RETRIEVER_K = max(1, min(RETRIEVER_K, 10))
MAX_UPLOAD_MB = _safe_int(os.getenv("MAX_UPLOAD_MB", "10"), 10)
MAX_UPLOAD_MB = max(1, min(MAX_UPLOAD_MB, 50))
MAX_MESSAGE_CHARS = _safe_int(os.getenv("MAX_MESSAGE_CHARS", "4000"), 4000)


def check_api_key() -> bool:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_KEY_HERE":
        return False
    # Google API kalit formati davr o'tishi bilan o'zgardi:
    # eski: "AIza..." (39 belgi), yangi: "AQ.Ab8..." (50+ belgi).
    # Shuning uchun prefiksga qarab keskin rad etmaymiz —
    # kalit yaroqliligini faqat Google API ga so'rov yuborib bilish mumkin.
    # Juda qisqa/aniq noto'g'ri qiymatlarni esa erta ushlaymiz.
    if len(GEMINI_API_KEY) < 20:
        print(
            f"[RAG] OGOHLANTIRISH: GEMINI_API_KEY juda qisqa "
            f"(uzunlik: {len(GEMINI_API_KEY)}). "
            f"https://aistudio.google.com/apikey dan oling."
        )
        return False
    return True
