
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()

    def build(self):
        self.vector_store.build()

    def search(
        self,
        query,
        top_k=5,
        vector_weight=0.5,
        keyword_weight=0.5,
        rrf_k=60,
    ):
        vector_results = self.vector_store.search(
            query,
            top_k=top_k * 2,
        )

        keyword_results = self.bm25.search(
            query,
            top_k=top_k * 2,
        )

        combined = {}

        # Add vector search rankings.
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk": result,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": None,
                    "keyword_score": None,
                    "rrf_score": 0.0,
                },
            )

            combined[chunk_id]["vector_rank"] = rank
            combined[chunk_id]["vector_score"] = result["score"]
            combined[chunk_id]["rrf_score"] += (
                vector_weight / (rrf_k + rank)
            )

        # Add keyword search rankings.
        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk": result,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": None,
                    "keyword_score": None,
                    "rrf_score": 0.0,
                },
            )

            combined[chunk_id]["keyword_rank"] = rank
            combined[chunk_id]["keyword_score"] = result["score"]
            combined[chunk_id]["rrf_score"] += (
                keyword_weight / (rrf_k + rank)
            )

        results = []

        for item in combined.values():
            chunk = item["chunk"].copy()

            chunk["vector_rank"] = item["vector_rank"]
            chunk["keyword_rank"] = item["keyword_rank"]
            chunk["vector_score"] = item["vector_score"]
            chunk["keyword_score"] = item["keyword_score"]
            chunk["hybrid_score"] = item["rrf_score"]

            results.append(chunk)

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]