from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.models.llm import GeminiLLM


def build_prompt(question, evidence):

    evidence_text = ""

    for i, item in enumerate(evidence, 1):
        evidence_text += (
            f"\n[EVIDENCE {i}]\n"
            f"Source: {item['source_file']}\n"
            f"{item['text']}\n"
        )

    return f"""
You are a financial research assistant.

Answer the user's question using only the
provided NVIDIA SEC filing evidence.

Rules:

1. Do not invent facts.
2. Cite the evidence used for important claims.
3. Distinguish facts from interpretations.
4. If the evidence is insufficient, explicitly
   say that the evidence is insufficient.
5. Do not perform complicated arithmetic mentally.
6. Give a concise but useful explanation.

Question:

{question}

Evidence:

{evidence_text}

Provide:

Conclusion:
...

Supporting Evidence:
...

Analysis:
...
"""


def main():

    retriever = HybridRetriever()

    llm = GeminiLLM()

    question = input(
        "Ask a question about NVIDIA: "
    )

    results = retriever.search(
        question,
        top_k=5,
    )

    prompt = build_prompt(
        question,
        results,
    )

    answer = llm.generate(prompt)

    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    main()