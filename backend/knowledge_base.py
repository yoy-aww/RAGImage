"""Knowledge base: stores chunks and retrieves relevant ones for question gen.

MVP: in-memory + JSON file persistence. TF-IDF (sklearn) for Chinese keyword retrieval.
"""
from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA_DIR
from chunker import Chunk, chunks_to_dicts

log = logging.getLogger("kb")


class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.chunks_path = self.data_dir / "chunks.json"
        self.tfidf_path = self.data_dir / "tfidf.pkl"
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if self.chunks_path.exists():
            try:
                data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
                self.chunks = [Chunk(**d) for d in data]
                log.info(f"Loaded {len(self.chunks)} chunks from disk")
            except Exception as e:
                log.warning(f"Failed to load chunks: {e}")
        if self.tfidf_path.exists():
            try:
                with open(self.tfidf_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                log.info("TF-IDF vectorizer loaded")
            except Exception as e:
                log.warning(f"Failed to load TF-IDF: {e}")

    def _save(self):
        self.chunks_path.write_text(
            json.dumps(chunks_to_dicts(self.chunks), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        if self.vectorizer is not None:
            with open(self.tfidf_path, "wb") as f:
                pickle.dump(self.vectorizer, f)

    # ---------- chunk management ----------
    def add_chunks(self, new_chunks: list[Chunk]):
        self.chunks.extend(new_chunks)
        self._refit_tfidf()
        self._save()

    def clear(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._save()

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "books": sorted(set(c.book_id for c in self.chunks)),
            "total_chars": sum(c.char_count for c in self.chunks),
        }

    # ---------- retrieval ----------
    def _refit_tfidf(self):
        """Refit TF-IDF on the full corpus. Chinese-friendly char n-gram."""
        if not self.chunks:
            return
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
            max_features=5000,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k chunks most relevant to query, with scores."""
        if not self.chunks or self.vectorizer is None:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            results.append({
                "chunk_id": c.chunk_id,
                "book_id": c.book_id,
                "page_no": c.page_no,
                "image_path": c.image_path,
                "text": c.text,
                "score": float(scores[i]),
            })
        return results

    # ---------- context for question gen ----------
    def get_context_text(self, chunk_ids: list[int]) -> str:
        """Join chunks by page order into a single context block."""
        ids = set(chunk_ids)
        parts = []
        for c in sorted([c for c in self.chunks if c.chunk_id in ids], key=lambda x: (x.book_id, x.page_no)):
            parts.append(f"【第{c.page_no}页】\n{c.text}")
        return "\n\n".join(parts)
