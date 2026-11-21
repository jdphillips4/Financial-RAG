from src.retrieval.bm25 import BM25Retriever
from src.retrieval.filters import parse_query_hints


class HybridRetriever:
    """Reciprocal-rank fusion of vector + BM25, with optional fiscal-year boost.

    Same output keys as before (vector_rank, keyword_rank, hybrid_score, ...), so
    existing scripts keep working. New: `boost_fiscal_year=True` multiplies the
    fused score of chunks whose fiscal_year / filing_type match hints in the query.
    """

    def __init__(self, vector_store=None, bm25=None):
        if vector_store is None:
            from src.retrieval.vector_store import VectorStore  # heavy import, keep lazy
            vector_store = VectorStore()
        self.vector_store = vector_store
        self.bm25 = bm25 or BM25Retriever()

    def build(self):
        self.vector_store.build()

    def search(
        self,
        query,
        top_k=5,
        vector_weight=0.5,
        keyword_weight=0.5,
        rrf_k=60,
        boost_fiscal_year=False,
        fy_boost=1.6,
        form_boost=1.25,
        pool=None,
    ):
        # Over-fetch so the boost has real candidates to promote.
        pool = pool or max(top_k * 6, 30)

        vector_results = self.vector_store.search(query, top_k=pool)
        keyword_results = self.bm25.search(query, top_k=pool)

        combined = {}

        def entry(result):
            return combined.setdefault(
                result["chunk_id"],
                {"chunk": result, "vector_rank": None, "keyword_rank": None,
                 "vector_score": None, "keyword_score": None, "rrf_score": 0.0},
            )

        for rank, result in enumerate(vector_results, start=1):
            e = entry(result)
            e["vector_rank"], e["vector_score"] = rank, result["score"]
            e["rrf_score"] += vector_weight / (rrf_k + rank)

        for rank, result in enumerate(keyword_results, start=1):
            e = entry(result)
            e["keyword_rank"], e["keyword_score"] = rank, result["score"]
            e["rrf_score"] += keyword_weight / (rrf_k + rank)

        hints = parse_query_hints(query) if boost_fiscal_year else None

        results = []
        for item in combined.values():
            chunk = item["chunk"].copy()
            score = item["rrf_score"]

            if hints:
                if hints["fiscal_years"] and chunk.get("fiscal_year") in hints["fiscal_years"]:
                    score *= fy_boost
                if hints["filing_type"] and chunk.get("filing_type") == hints["filing_type"]:
                    score *= form_boost

            chunk["vector_rank"] = item["vector_rank"]
            chunk["keyword_rank"] = item["keyword_rank"]
            chunk["vector_score"] = item["vector_score"]
            chunk["keyword_score"] = item["keyword_score"]
            chunk["hybrid_score"] = score
            results.append(chunk)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]
