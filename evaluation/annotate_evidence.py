"""Helpers for building the gold set's evidence labels.

    # 1) find the passage that answers a question (keyword search over your chunks)
    python -m evaluation.annotate_evidence search "gross margin fiscal 2025" --k 6
    python -m evaluation.annotate_evidence search "export controls" --fy 2025 --form 10-K

    # 2) paste a SHORT verbatim phrase from that passage into the question's
    #    "gold_evidence_snippets" in evaluation/questions.json, then:
    python -m evaluation.annotate_evidence map

`map` checks every snippet against chunks.json and fills gold_chunk_ids. Snippets that
match nothing (typo / wrong text) or match too many chunks (too generic) are reported.
Because grading uses the snippet text, labels survive re-chunking.
"""

import argparse
import json

from evaluation.common import QUESTIONS_PATH, load_chunks, norm, source_ok
from src.retrieval.bm25 import BM25Retriever

TOO_GENERIC = 12


def cmd_search(args):
    retriever = BM25Retriever()
    results = retriever.search(args.query, top_k=args.k * 6)
    shown = 0
    for r in results:
        if args.fy and r.get("fiscal_year") != args.fy:
            continue
        if args.form and r.get("filing_type") != args.form:
            continue
        shown += 1
        print("=" * 78)
        print(f"{r['chunk_id']} | {r.get('filing_type')} FY{r.get('fiscal_year')} {r.get('fiscal_period')} "
              f"| {r.get('section')} | {r.get('chunk_type')} | bm25={r['score']:.1f}")
        print(r["text"][: args.chars])
        if shown >= args.k:
            break
    if not shown:
        print("No matches with those filters.")


def cmd_map(args):
    path = args.questions or QUESTIONS_PATH
    data = json.loads(open(path, encoding="utf-8").read())
    questions = data["questions"] if isinstance(data, dict) else data
    chunks = load_chunks()
    bodies = [(c, norm(c["text"])) for c in chunks]

    needs, problems, ok = [], [], 0
    for q in questions:
        if q.get("should_abstain"):
            q["gold_chunk_ids"] = []
            continue
        snippets = [s for s in q.get("gold_evidence_snippets") or [] if s.strip()]
        if not snippets:
            needs.append(q["id"])
            continue

        ids = []
        for s in snippets:
            hits = [c["chunk_id"] for c, body in bodies
                    if norm(s) in body and source_ok(c, q.get("acceptable_sources"))]
            if not hits:
                problems.append(f"{q['id']}: snippet matches NO chunk -> {s[:70]!r}")
            elif len(hits) > TOO_GENERIC:
                problems.append(f"{q['id']}: snippet matches {len(hits)} chunks (too generic) -> {s[:70]!r}")
            ids.extend(hits)
        q["gold_chunk_ids"] = sorted(set(ids))
        ok += bool(ids)

    if not args.dry_run:
        open(path, "w", encoding="utf-8").write(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"Questions with usable evidence: {ok}")
    print(f"Questions still needing gold_evidence_snippets ({len(needs)}): {', '.join(needs) or '-'}")
    if problems:
        print("\nProblems to fix:")
        for p in problems:
            print("  -", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--fy", type=int, default=None)
    s.add_argument("--form", default=None)
    s.add_argument("--chars", type=int, default=900)
    s.set_defaults(func=cmd_search)

    m = sub.add_parser("map")
    m.add_argument("--questions", default=None)
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(func=cmd_map)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
