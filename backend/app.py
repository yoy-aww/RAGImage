"""FastAPI application: book upload → OCR → KB → question gen."""
from __future__ import annotations
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import UPLOAD_DIR, VLM_MODEL
from ocr_service import ocr_image
from chunker import split_by_pages
from knowledge_base import KnowledgeBase
from question_generator import generate_questions, get_all_context_for_book

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("api")

app = FastAPI(title="RAGImage - Book to Questions", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global KB instance
kb = KnowledgeBase()


@app.get("/api/health")
def health():
    return {"ok": True, "model": VLM_MODEL, "stats": kb.stats()}


@app.get("/api/books")
def list_books():
    return {"books": sorted(set(c.book_id for c in kb.chunks))}


@app.get("/api/chunks")
def list_chunks(book_id: str = ""):
    chunks = kb.chunks
    if book_id:
        chunks = [c for c in chunks if c.book_id == book_id]
    return {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "book_id": c.book_id,
                "page_no": c.page_no,
                "char_count": c.char_count,
                "preview": c.text[:120],
            }
            for c in chunks
        ]
    }


@app.post("/api/books")
async def create_book(
    files: list[UploadFile] = File(...),
    book_id: str = Form(default=""),
):
    """Upload one or more book page images; run OCR; add to KB."""
    book_id = book_id or f"book_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    log.info(f"Creating book {book_id} with {len(files)} files")

    ocr_results = []
    errors = []
    for idx, f in enumerate(files, start=1):
        # Save upload
        suffix = Path(f.filename or f"img.jpg").suffix or ".jpg"
        out_path = UPLOAD_DIR / f"{book_id}_p{idx:03d}{suffix}"
        try:
            content = await f.read()
            out_path.write_bytes(content)
        except Exception as e:
            errors.append({"file": f.filename, "error": f"read: {e}"})
            continue

        # OCR
        try:
            ocr = ocr_image(out_path)
        except Exception as e:
            log.exception(f"OCR failed for {out_path}")
            errors.append({"file": f.filename, "error": f"ocr: {e}"})
            continue

        ocr_results.append({
            "book_id": book_id,
            "page_no": idx,
            "image_path": str(out_path),
            "text": ocr["text"],
            "lines": ocr["lines"],
        })

    if not ocr_results:
        raise HTTPException(500, f"No pages OCR'd successfully. Errors: {errors}")

    # If book_id already exists, clear old chunks first (upload = replace)
    existing = [c for c in kb.chunks if c.book_id == book_id]
    if existing:
        log.info(f"Replacing {len(existing)} existing chunks for {book_id}")
        kb.chunks = [c for c in kb.chunks if c.book_id != book_id]
        kb._refit_tfidf()

    chunks = split_by_pages(ocr_results)
    kb.add_chunks(chunks)

    return {
        "book_id": book_id,
        "pages_ocrd": len(ocr_results),
        "chunks_added": len(chunks),
        "total_chars": sum(c.char_count for c in chunks),
        "replaced_existing": len(existing),
        "errors": errors,
        "stats": kb.stats(),
    }


@app.get("/api/books/{book_id}/text")
def get_book_text(book_id: str):
    pages = sorted([c for c in kb.chunks if c.book_id == book_id], key=lambda x: x.page_no)
    if not pages:
        raise HTTPException(404, f"Book {book_id} not found")
    return {
        "book_id": book_id,
        "pages": [
            {"page_no": c.page_no, "text": c.text, "image_path": c.image_path}
            for c in pages
        ],
    }


@app.post("/api/books/{book_id}/questions")
def gen_questions_for_book(
    book_id: str,
    num_questions: int = Form(3),
    difficulty: str = Form("mixed"),
):
    """Generate questions from the whole book."""
    pages = [c for c in kb.chunks if c.book_id == book_id]
    if not pages:
        raise HTTPException(404, f"Book {book_id} not found")
    context = get_all_context_for_book(kb, book_id)
    if len(context) < 50:
        raise HTTPException(400, "Not enough text to generate questions")

    try:
        result = generate_questions(
            context_text=context,
            num_questions=num_questions,
            difficulty=difficulty,
            book_id=book_id,
            page_range=f"1-{max(p.page_no for p in pages)}",
        )
    except Exception as e:
        log.exception("Question generation failed")
        raise HTTPException(500, str(e))

    # Persist questions to disk
    q_path = Path(kb.data_dir) / f"questions_{book_id}.json"
    q_path.write_text(json.dumps(result["questions"], ensure_ascii=False, indent=2), encoding="utf-8")

    return {"book_id": book_id, "questions": result["questions"]}


@app.get("/api/books/{book_id}/questions")
def list_questions(book_id: str):
    q_path = Path(kb.data_dir) / f"questions_{book_id}.json"
    if not q_path.exists():
        return {"book_id": book_id, "questions": []}
    return {"book_id": book_id, "questions": json.loads(q_path.read_text(encoding="utf-8"))}


@app.post("/api/books/{book_id}/clear")
def clear_book(book_id: str):
    before = len(kb.chunks)
    kb.chunks = [c for c in kb.chunks if c.book_id != book_id]
    kb._refit_tfidf()
    kb._save()
    # Also delete uploads
    for f in UPLOAD_DIR.glob(f"{book_id}_*"):
        f.unlink(missing_ok=True)
    q_path = Path(kb.data_dir) / f"questions_{book_id}.json"
    q_path.unlink(missing_ok=True)
    return {"removed_chunks": before - len(kb.chunks), "stats": kb.stats()}


@app.post("/api/clear-all")
def clear_all():
    kb.clear()
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "stats": kb.stats()}


@app.get("/api/images/{filename}")
def get_image(filename: str):
    p = UPLOAD_DIR / filename
    if not p.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(p)


# Serve frontend
@app.get("/")
def index():
    html = Path(__file__).parent.parent / "frontend" / "index.html"
    if html.exists():
        return FileResponse(html)
    return JSONResponse({"error": "frontend not built"})
