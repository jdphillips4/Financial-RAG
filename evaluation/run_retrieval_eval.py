"""Step 4a: evaluate retrieval ALONE (no LLM involved).

    python -m evaluation.run_retrieval_eval --modes vector,bm25,hybrid,hybrid_fy

Writes evaluation/results/retrieval_eval.jsonl (one line per question x retriever) and
prints a summary. Re-run it after every chunking/embedding change and compare.
"""

import argparse
import time

import pandas as pd

from evaluation.common import (
    RESULTS_DIR, build_retrievers, config_snapshot, load_chunks, load_questions,
)
from evaluation.metrics import KS, retrieval_metrics
from src.utils.run_logger import RunLogger


def search(retrievers, mode, question, top_k):
    if mode == "hybrid_fy":
        return retrievers["hybrid"].search(question, top_k=top_k, boost_fiscal_year=True)
    return retrievers[mode].search(question, top_k=top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="vector,bm25,hybrid,hybrid_fy")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "retrieval_eval.jsonl"))
    ap.add_argument("--fresh", action="store_true", help="delete the old output file first")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = load_questions(args.questions)
    chunks = load_chunks()
    retrievers = build_retrievers(modes)
    cfg = config_snapshot() | {"n_chunks": len(chunks)}

    logger = RunLogger(args.out)
    if args.fresh and logger.path.exists():
        logger.path.unlink()

    scored_qs = [q for q in questions if not q.get("should_abstain")]
    skipped = 0
    max_k = max(KS)

    for mode in modes:
        for q in scored_qs:
            start = time.perf_counter()
            results = search(retrievers, mode, q["question"], max_k)
            ms = round((time.perf_counter() - start) * 1000, 1)

            metrics = retrieval_metrics(results, q)
            if metrics is None:
                skipped += 1
                continue
            logger.log({
                "qid": q["id"], "category": q.get("category"), "mode": mode,
                "retrieval_ms": ms, "retrieved_ids": [r["chunk_id"] for r in results],
                "config": cfg, **metrics,
            })

    df = pd.DataFrame(logger.read())
    if df.empty:
        print("No annotated questions to score. Run: python -m evaluation.annotate_evidence map")
        return
    # keep only this run's config if the file has several
    df = df[df["config"].apply(lambda c: c == cfg)]

    cols = ["mrr"] + [f"hit@{k}" for k in KS] + [f"full@{k}" for k in (3, 5)] + ["retrieval_ms"]
    print(f"\nConfig: {cfg}")
    print(f"Questions scored: {df['qid'].nunique()}  | skipped (no gold evidence): {skipped // max(len(modes), 1)}")
    print("\n=== Retrieval by method ===")
    print(df.groupby("mode")[cols].mean().round(3).to_string())
    print("\n=== hit@5 by category ===")
    print(df.pivot_table(index="category", columns="mode", values="hit@5", aggfunc="mean").round(2).to_string())

    worst = df[(df["mode"] == modes[-1]) & (df["hit@5"] == 0)]
    if len(worst):
        print(f"\n=== Misses for '{modes[-1]}' at k=5 (look at these by hand) ===")
        qtext = {q["id"]: q["question"] for q in questions}
        for _, row in worst.iterrows():
            print(f"  {row['qid']}: {qtext.get(row['qid'])}")


if __name__ == "__main__":
    main()
