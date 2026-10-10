from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)


class FinancialSearchTool:

    def __init__(self):
        self.retriever = HybridRetriever()

    def search(self, query, top_k=5):
        return self.retriever.search(
            query,
            top_k=top_k,
        )


if __name__ == "__main__":
    tool = FinancialSearchTool()

    results = tool.search(
        "Why did NVIDIA gross margin change?"
    )

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(
            f"Source: {result['source_file']}"
        )
        print(
            f"Score: {result['hybrid_score']:.3f}"
        )
        print(result["text"][:1000])