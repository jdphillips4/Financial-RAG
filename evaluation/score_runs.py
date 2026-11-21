"""Score logged runs against the gold set.

    python -m evaluation.score_runs                        # numeric + abstention + citations
    python -m evaluation.score_runs --judge gemini:gemini-2.5-flash   # + LLM judge for text answers

The judge is itself an LLM, so hand-grade ~20 judged answers and report agreement
(Cohen's kappa) before you trust it - the same validation the proposal asks for.
"""

import argparse

from evaluation.common import RESULTS_DIR, load_chunks, load_questions
from evaluation.metrics import score_run
from src.models.llm import get_llm
from src.utils.run_logger import RunLogger, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RESULTS_DIR / "runs.jsonl"))
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "scored.jsonl"))
    ap.add_argument("--judge", default=None, help="provider:model for grading free-text answers")
    args = ap.parse_args()

    questions = {q["id"]: q for q in load_questions(args.questions)}
    chunk_by_id = {c["chunk_id"]: c for c in load_chunks()}
    judge = get_llm(args.judge) if args.judge else None

    out = RunLogger(args.out)
    if out.path.exists():
        out.path.unlink()

    n = unscored = 0
    for rec in read_jsonl(args.runs):
        q = questions.get(rec.get("qid"))
        if q is None:
            continue
        scored = score_run(rec, q, chunk_by_id, judge_llm=judge)
        out.log({k: v for k, v in scored.items() if k != "run_id"} | {"run_id": rec.get("run_id")})
        n += 1
        unscored += scored["correct"] is None

    print(f"Scored {n} runs -> {out.path}")
    print(f"{unscored} runs have no correctness label (errored, or text question with no judge/gold answer).")


if __name__ == "__main__":
    main()
