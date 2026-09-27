"""OCR service using RapidOCR (local, CPU-friendly, good Chinese support)."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from rapidocr_onnxruntime import RapidOCR

log = logging.getLogger("ocr")

_ocr: RapidOCR | None = None


def _get_ocr() -> RapidOCR:
    global _ocr
    if _ocr is None:
        log.info("Initializing RapidOCR (loads models, ~10s on first call)...")
        _ocr = RapidOCR()
        log.info("RapidOCR ready")
    return _ocr


def ocr_image(image_path: str | Path) -> dict[str, Any]:
    """Run OCR on an image. Returns {text, lines, raw}.

    lines: list of {text, score, box, y_center} sorted top-to-bottom.
    raw: original RapidOCR output for debugging.
    """
    ocr = _get_ocr()
    result, _ = ocr(str(image_path))

    if not result:
        return {"text": "", "lines": [], "raw": []}

    # result: [[box, text, score], ...]  box is 4 points
    lines = []
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append({
            "text": text.strip(),
            "score": float(score),
            "x_left": min(xs),
            "x_right": max(xs),
            "y_center": (min(ys) + max(ys)) / 2,
            "y_top": min(ys),
            "y_bottom": max(ys),
        })

    # Sort by vertical position to restore reading order
    lines.sort(key=lambda l: l["y_center"])

    # Filter low-confidence garbage (< 0.3) but keep originals for debug
    lines = [l for l in lines if l["score"] >= 0.3 and l["text"]]

    text = "\n".join(l["text"] for l in lines)
    return {"text": text, "lines": lines, "raw": result}
