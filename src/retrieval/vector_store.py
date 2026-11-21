"""FAISS vector store.

Changes vs. your version:
  * Indexes `search_text` (header + body) instead of bare `text`.
  * STALE-INDEX GUARD: a sidecar file records which chunks/model built the index.
    If chunks.json changed (e.g. you re-chunked) the index is rebuilt automatically.
    Before, a stale vector.index silently returned the WRONG chunks - a very common
    cause of "retrieval is returning irrelevant evidence".
  * Prints the embedding model's max_seq_length. all-MiniLM-L6-v2 only reads the
    first 256 tokens (~190 words) of each chunk, so 500-word chunks are mostly
    invisible to it. Use CHUNK_SIZE <= ~180 words with MiniLM, or switch
    EMBEDDING_MODEL to BAAI/bge-small-en-v1.5 (512 tokens) and set
    EMBEDDING_QUERY_PREFIX in config.py.
"""

import hashlib
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config
from src.config import EMBEDDING_MODEL, NVIDIA_PROCESSED_DIR

CHUNKS_PATH = NVIDIA_PROCESSED_DIR / "chunks.json"
INDEX_PATH = NVIDIA_PROCESSED_DIR / "vector.index"
META_PATH = NVIDIA_PROCESSED_DIR / "vector.index.meta.json"

QUERY_PREFIX = getattr(config, "EMBEDDING_QUERY_PREFIX", "")


def _fingerprint(chunks):
    h = hashlib.sha256()
    h.update(EMBEDDING_MODEL.encode())
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(str(len(c.get("search_text") or c["text"])).encode())
    return h.hexdigest()


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[vector] {EMBEDDING_MODEL} max_seq_length={self.model.max_seq_length} tokens")
        self.chunks = []
        self.index = None

    def _load_chunks(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def build(self):
        self._load_chunks()
        texts = [c.get("search_text") or c["text"] for c in self.chunks]

        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(
            json.dumps({"fingerprint": _fingerprint(self.chunks), "n": len(self.chunks)}),
            encoding="utf-8",
        )
        print(f"Indexed {len(texts)} chunks.")

    def _ensure_index(self):
        if self.index is not None:
            return
        self._load_chunks()
        fresh = False
        if INDEX_PATH.exists() and META_PATH.exists():
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            fresh = meta.get("fingerprint") == _fingerprint(self.chunks)
        if not fresh:
            print("Vector index missing or stale (chunks/model changed). Rebuilding...")
            self.build()
        else:
            self.index = faiss.read_index(str(INDEX_PATH))

    def search(self, query, top_k=5):
        self._ensure_index()
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = self.chunks[idx].copy()
            result["score"] = float(score)
            results.append(result)
        return results
