"""Runtime configuration. Everything is overridable through environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("RECALL_HOME", Path.home() / ".recall"))
DB_PATH = DATA_DIR / "recall.db"

EMBED_MODEL = os.getenv("RECALL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = Path(os.getenv("RECALL_MODEL_CACHE", Path.home() / ".cache" / "recall" / "models"))

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXT = {".pdf"}
TEXT_EXT = {".txt", ".md", ".csv"}
SUPPORTED_EXT = IMAGE_EXT | PDF_EXT | TEXT_EXT

DEVICE = os.getenv("RECALL_DEVICE", "auto").lower()
WORKERS = int(os.getenv("RECALL_WORKERS", "0"))


def use_cuda() -> bool:
    if DEVICE == "cpu":
        return False
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        ok = False
    if DEVICE == "cuda" and not ok:
        raise RuntimeError("RECALL_DEVICE=cuda but onnxruntime has no CUDAExecutionProvider (install requirements-gpu.txt)")
    return ok


def default_workers() -> int:
    if WORKERS:
        return WORKERS
    return 2 if use_cuda() else max(1, min(8, (os.cpu_count() or 2) // 2))


CHUNK_CHARS = 700
CHUNK_OVERLAP = 120


def kind_of(ext: str) -> str | None:
    ext = ext.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in PDF_EXT:
        return "pdf"
    if ext in TEXT_EXT:
        return "text"
    return None
