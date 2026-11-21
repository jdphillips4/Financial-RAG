import json

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import NVIDIA_PROCESSED_DIR
from src.retrieval.tokenize import tokenize

CHUNKS_PATH = NVIDIA_PROCESSED_DIR / "chunks.json"


class BM25Retriever:
    """BM25 over `search_text` (header + body), falling back to `text`."""

    def __init__(self, chunks=None):
        if chunks is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        self.chunks = chunks
        self.bm25 = BM25Okapi(
            [tokenize(c.get("search_text") or c["text"]) for c in self.chunks]
        )

    def search(self, query, top_k=5):
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        ranked = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked:
            if scores[index] <= 0:
                continue
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)
        return results
