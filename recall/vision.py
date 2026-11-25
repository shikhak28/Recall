"""Local visual search with CLIP (fastembed / ONNX Runtime).

Photos with no readable text get no OCR chunks, so text search can never find them. CLIP puts images and short
text descriptions in the same vector space: "photo of a church with golden domes" lands close to that photo.
Everything runs on the device; models are downloaded once into the Recall model cache.
"""
import numpy as np
from PIL import Image, ImageOps

from .config import CLIP_IMAGE_MODEL, CLIP_TEXT_MODEL, EMBED_CACHE, model_cached, use_cuda

DIM = 512
_img_model = None
_txt_model = None


def _kw(name: str):
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    kw = {"cache_dir": str(EMBED_CACHE)}
    if model_cached(name):
        kw["local_files_only"] = True
    if use_cuda():
        kw["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return kw


def _image_model():
    global _img_model
    if _img_model is None:
        from fastembed import ImageEmbedding

        _img_model = ImageEmbedding(CLIP_IMAGE_MODEL, **_kw(CLIP_IMAGE_MODEL))
    return _img_model


def _text_model():
    global _txt_model
    if _txt_model is None:
        from fastembed import TextEmbedding

        _txt_model = TextEmbedding(CLIP_TEXT_MODEL, **_kw(CLIP_TEXT_MODEL))
    return _txt_model


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


def _load(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((512, 512))
        return im.copy()


def embed_images(paths: list[str]) -> np.ndarray:
    """One unit vector per image. An unreadable image raises; callers handle it per file."""
    if not paths:
        return np.zeros((0, DIM), dtype=np.float32)
    imgs = [_load(p) for p in paths]
    return _norm(np.array(list(_image_model().embed(imgs, batch_size=16)), dtype=np.float32))


def embed_text(text: str) -> np.ndarray:
    return _norm(np.array(list(_text_model().embed([text])), dtype=np.float32))[0]
