import os
from pathlib import Path

from dotenv import load_dotenv

# backend/ dan loyiha ildizidagi .env ni topish (uvicorn qayerdan
# ishga tushirilishidan qat'iy nazar ishlaydi)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001").strip()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db").strip()
DB_PATH = os.getenv("DB_PATH", "./chat.db").strip()
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4").strip() or "4")


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
