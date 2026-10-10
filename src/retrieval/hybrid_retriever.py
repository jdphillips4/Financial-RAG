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

        for result in vector_results:
            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk": result,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[chunk_id][
                "vector_score"
            ] = result["score"]

        for result in keyword_results:
            chunk_id = result["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk": result,
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                },
            )

            combined[chunk_id][
                "keyword_score"
            ] = result["score"]

        # Normalize scores separately.
        vector_scores = [
            x["vector_score"]
            for x in combined.values()
        ]

        keyword_scores = [
            x["keyword_score"]
            for x in combined.values()
        ]

        max_vector = max(vector_scores, default=1.0)
        max_keyword = max(keyword_scores, default=1.0)

        results = []

        for item in combined.values():
            vector_score = (
                item["vector_score"] / max_vector
                if max_vector != 0
                else 0
            )

            keyword_score = (
                item["keyword_score"] / max_keyword
                if max_keyword != 0
                else 0
            )

            final_score = (
                vector_weight * vector_score
                + keyword_weight * keyword_score
            )

            chunk = item["chunk"].copy()

            chunk["vector_score"] = vector_score
            chunk["keyword_score"] = keyword_score
            chunk["hybrid_score"] = final_score

            results.append(chunk)

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return results[:top_k]