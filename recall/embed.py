"""Local text embeddings (fastembed / ONNX Runtime). No network access after the one-time model download."""
import os

import numpy as np

from .config import EMBED_CACHE, EMBED_MODEL, use_cuda

_model = None


def _get_model():
    global _model
    if _model is None:
        if EMBED_CACHE.exists() and any(EMBED_CACHE.iterdir()):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from fastembed import TextEmbedding

        EMBED_CACHE.mkdir(parents=True, exist_ok=True)
        kw = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]} if use_cuda() else {}
        _model = TextEmbedding(EMBED_MODEL, cache_dir=str(EMBED_CACHE), **kw)
    return _model


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _norm(np.array(list(_get_model().passage_embed(texts, batch_size=32)), dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    return _norm(np.array(list(_get_model().query_embed([text])), dtype=np.float32))[0]
