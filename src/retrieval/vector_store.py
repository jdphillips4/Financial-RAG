import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    NVIDIA_PROCESSED_DIR,
    EMBEDDING_MODEL,
)


CHUNKS_PATH = NVIDIA_PROCESSED_DIR / "chunks.json"
INDEX_PATH = NVIDIA_PROCESSED_DIR / "vector.index"


class VectorStore:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.chunks = []

        self.index = None

    def build(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        faiss.write_index(
            self.index,
            str(INDEX_PATH),
        )

        print(
            f"Indexed {len(texts)} chunks."
        )

    def search(self, query, top_k=5):
        if self.index is None:
            self.index = faiss.read_index(
                str(INDEX_PATH)
            )

            with open(
                CHUNKS_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                self.chunks = json.load(f)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            result = self.chunks[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results