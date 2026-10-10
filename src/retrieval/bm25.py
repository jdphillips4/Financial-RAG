import json

from rank_bm25 import BM25Okapi

from src.config import NVIDIA_PROCESSED_DIR


CHUNKS_PATH = NVIDIA_PROCESSED_DIR / "chunks.json"


class BM25Retriever:

    def __init__(self):
        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            self.chunks = json.load(f)

        tokenized_documents = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(self, query, top_k=5):
        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])

            results.append(result)

        return results