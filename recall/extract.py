"""Text extraction: OCR for images and scanned PDF pages, native text for digital PDFs and text files.

All processing is local (RapidOCR ONNX models ship inside the pip package).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import CHUNK_CHARS, CHUNK_OVERLAP, kind_of, use_cuda

_ocr = None


MIN_CONFIDENCE = 0.6


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        gpu = use_cuda()
        _ocr = RapidOCR(params={
            "Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Global.use_cls": False, "Global.log_level": "error",
            "EngineConfig.onnxruntime.use_cuda": gpu,
        })
    return _ocr


@dataclass
class Segment:
    page: int
    text: str
    method: str


def ocr_image(img: Image.Image) -> str:
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 2400:
        img.thumbnail((2400, 2400))
    out = _get_ocr()(np.asarray(img))
    if out.txts is None:
        return ""
    result = [(b, t, s) for b, t, s in zip(out.boxes, out.txts, out.scores) if s >= MIN_CONFIDENCE and t.strip()]
    if sum(ch.isalnum() for _, t, _ in result for ch in t) < 4:
        return ""
    boxes = sorted(result, key=lambda r: (round(min(p[1] for p in r[0]) / 12), min(p[0] for p in r[0])))
    lines, last_y = [], None
    for box, text, _score in boxes:
        y = round(min(p[1] for p in box) / 12)
        if lines and y == last_y:
            lines[-1] += " " + text
        else:
            lines.append(text)
        last_y = y
    return "\n".join(lines)


def extract(path: Path) -> list[Segment]:
    kind = kind_of(path.suffix)
    if kind == "image":
        with Image.open(path) as im:
            return [Segment(1, ocr_image(im), "ocr")]
    if kind == "pdf":
        return _extract_pdf(path)
    if kind == "text":
        return [Segment(1, path.read_text(errors="ignore"), "plain")]
    raise ValueError(f"unsupported file type: {path.suffix}")


def _extract_pdf(path: Path, max_pages: int = 50) -> list[Segment]:
    import pymupdf as fitz

    segs = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text().strip()
            if len(text) >= 20:
                segs.append(Segment(i + 1, text, "pdf-text"))
                continue
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            segs.append(Segment(i + 1, ocr_image(img), "ocr"))
    return segs


def chunk(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            cut = max(text.rfind("\n", start + CHUNK_CHARS // 2, end), text.rfind(" ", start + CHUNK_CHARS // 2, end))
            if cut > start:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in out if c]
