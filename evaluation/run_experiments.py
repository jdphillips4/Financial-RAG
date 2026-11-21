"""Steps 4b/5: run models x modes over the fixed question set, logging every run.

    # compare LLMs on identical gold evidence (isolates generation quality)
    python -m evaluation.run_experiments --modes oracle --models gemini:gemini-2.5-flash,openai:gpt-4.1-mini,ollama:qwen3:8b

    # architecture ablation with one model
    python -m evaluation.run_experiments --modes direct,vector,hybrid,hybrid_fy --models gemini:gemini-2.5-flash

    # try the plumbing with no API calls
    python -m evaluation.run_experiments --models fake --modes hybrid --limit 3

Resumable: re-running skips (question, mode, model) combos that already finished.
"""

import argparse
import time

from evaluation.common import (
    RESULTS_DIR, build_retrievers, config_snapshot, gold_chunks_for, load_chunks, load_questions,
)
from src.models.llm import get_llm
from src.pipeline import RAGPipeline
from src.utils.run_logger import RunLogger

MAX_ORACLE_CHUNKS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini:gemini-2.5-flash")
    ap.add_argument("--modes", default="direct,vector,hybrid")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--categories", default=None, help="comma list to restrict categories")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (rate limits)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "runs.jsonl"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    questions = load_questions(args.questions)
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        questions = [q for q in questions if q.get("category") in wanted]
    if args.limit:
        questions = questions[: args.limit]

    chunks = load_chunks()
    retrievers = build_retrievers(modes) if set(modes) - {"direct", "oracle"} else {}
    cfg = config_snapshot() | {"n_chunks": len(chunks)}

    logger = RunLogger(args.out)
    done = set() if args.no_resume else logger.completed(("qid", "mode", "model"))

    total = ok = errors = skipped = 0
    try:
        for model_spec in models:
            llm = get_llm(model_spec)
            pipeline = RAGPipeline(llm, retrievers, top_k=args.top_k, logger=logger)

            for mode in modes:
                for q in questions:
                    key = (q["id"], mode, llm.name)
                    if key in done:
                        skipped += 1
                        continue

                    oracle = None
                    if mode == "oracle":
                        oracle = gold_chunks_for(q, chunks)[:MAX_ORACLE_CHUNKS]
                        if not oracle and not q.get("should_abstain"):
                            skipped += 1          # no gold evidence annotated yet
                            continue

                    rec = pipeline.run(
                        q["question"], mode=mode, qid=q["id"], oracle_chunks=oracle,
                        extra={"category": q.get("category"), "config": cfg},
                    )
                    total += 1
                    errors += bool(rec["error"])
                    ok += not rec["error"]
                    if total % 5 == 0:
                        print(f"[{total}] {llm.name} | {mode} | {q['id']} "
                              f"retr={rec['retrieval_ms']}ms llm={rec['llm_ms']}ms "
                              f"{'ERROR ' + rec['error'][:80] if rec['error'] else 'ok'}")
                    if args.sleep:
                        time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted - finished runs are already saved; re-run to resume.")

    print(f"\nRuns this session: {total} ({ok} ok, {errors} errors) | skipped/resumed: {skipped}")
    print(f"Log: {logger.path}")
    if errors:
        print("Errored runs are NOT counted as done, so re-running retries them.")


if __name__ == "__main__":
    main()
