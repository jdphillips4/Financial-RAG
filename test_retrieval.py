from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

questions = [
    "What was NVIDIA's total revenue in fiscal year 2025?",
    "What was NVIDIA's data center revenue in fiscal year 2025?",
    "What risks does NVIDIA identify in its 2025 annual report?",
    "What factors contributed to NVIDIA's revenue growth?",
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = retriever.search(question, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\nRESULT {i}")
        print("Source:", result.get("source_file"))
        print("Hybrid score:", result.get("hybrid_score")) #use hybrid score
        print("Vector rank:", result.get("vector_rank"))
        print("Keyword rank:", result.get("keyword_rank"))
        print("Text:", result.get("text", "")[:700])