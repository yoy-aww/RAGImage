"""Text chunker: split OCR text into passages suitable for question generation."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    chunk_id: int
    book_id: str
    page_no: int          # 1-based page index within this book
    image_path: str       # source image file
    text: str
    line_count: int = 0
    char_count: int = 0
    metadata: dict = field(default_factory=dict)


def split_by_pages(images_with_ocr: list[dict]) -> list[Chunk]:
    """Each uploaded image becomes one chunk (1 image = 1 page).

    images_with_ocr: list of {book_id, image_path, page_no, text, lines}
    """
    chunks = []
    for i, item in enumerate(images_with_ocr):
        text = (item.get("text") or "").strip()
        if not text:
            continue  # skip blank pages
        chunks.append(Chunk(
            chunk_id=i,
            book_id=item["book_id"],
            page_no=item["page_no"],
            image_path=item["image_path"],
            text=text,
            line_count=len(item.get("lines", [])),
            char_count=len(text),
            metadata={"image_file": Path(item["image_path"]).name},
        ))
    return chunks


def get_chunk_text(chunk: Chunk) -> str:
    return chunk.text


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
