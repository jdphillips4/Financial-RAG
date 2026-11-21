"""Retrieval + answer metrics.

Retrieval is judged against gold evidence *snippets* (verbatim text from the filing),
not chunk ids, so scores stay valid when you change chunk size or chunking logic.
"""

import json
import re

from evaluation.common import question_targets, targets_matched_by
from src.pipeline import extract_final_answer

KS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def retrieval_metrics(retrieved_chunks, question, ks=KS):
    """Returns None when the question has no gold evidence (unannotated or abstention)."""
    if question.get("should_abstain"):
        return None
    kind, targets = question_targets(question)
    if kind == "none":
        return None

    matched = [targets_matched_by(c, question, kind, targets) for c in retrieved_chunks]

    out = {"n_targets": len(targets)}
    first = next((i for i, m in enumerate(matched, 1) if m), None)
    out["mrr"] = 1.0 / first if first else 0.0
    out["first_relevant_rank"] = first

    for k in ks:
        top = matched[:k]
        covered = set().union(*top) if top else set()
        out[f"hit@{k}"] = float(bool(covered))
        out[f"coverage@{k}"] = len(covered) / len(targets)
        out[f"full@{k}"] = float(len(covered) == len(targets))
        out[f"precision@{k}"] = (sum(1 for m in top if m) / len(top)) if top else 0.0
    return out


# --------------------------------------------------------------------------- #
# numeric answers
# --------------------------------------------------------------------------- #
_SCALE_TO_MILLIONS = {
    "trillion": 1e6, "billion": 1e3, "million": 1.0, "thousand": 1e-3,
    "tn": 1e6, "bn": 1e3, "b": 1e3, "mm": 1.0, "m": 1.0, "k": 1e-3,
}
_NUM_RE = re.compile(
    r"(?P<neg>\()?\$?\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<suffix>%|percent(?:age)?(?:\s+points?)?|trillion|billion|million|thousand|tn|bn|mm|b|m|k)(?![A-Za-z]))?",
    re.IGNORECASE,
)


def extract_numbers(text):
    """[(value, suffix_lowercase_or_None)] for every number in the text."""
    text = text or ""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num").replace(",", "").rstrip(".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if m.group("neg") and text[m.end("num"):m.end("num") + 1] == ")":
            value = -value
        suffix = (m.group("suffix") or "").lower() or None
        out.append((value, suffix))
    return out


def _within(candidate, gold, rel, abs_tol):
    if abs_tol is not None and abs(candidate - gold) <= abs_tol:
        return True
    if rel is not None and gold != 0 and abs(candidate - gold) / abs(gold) <= rel:
        return True
    return False


def numeric_match(answer_text, gold_numeric):
    """Does any number in the answer equal the gold value (unit-aware, tolerance-aware)?

    gold_numeric = {"value": 130497, "unit": "usd_millions"|"percent"|"usd"|"count",
                    "tolerance_rel": 0.005, "tolerance_abs": null}
    """
    if not gold_numeric or answer_text is None:
        return None
    gold = float(gold_numeric["value"])
    unit = gold_numeric.get("unit", "usd_millions")
    rel = gold_numeric.get("tolerance_rel", 0.005 if unit != "percent" else None)
    abs_tol = gold_numeric.get("tolerance_abs", 0.15 if unit == "percent" else None)

    for value, suffix in extract_numbers(answer_text):
        candidates = {value}
        if unit == "usd_millions":
            if suffix in _SCALE_TO_MILLIONS:
                candidates.add(value * _SCALE_TO_MILLIONS[suffix])
            if value >= 1e7:                      # written in raw dollars
                candidates.add(value / 1e6)
        elif unit == "percent":
            pass                                  # "75.0%", "75 percent", or bare 75.0
        elif unit == "usd" and suffix in _SCALE_TO_MILLIONS:
            candidates.add(value * _SCALE_TO_MILLIONS[suffix] * 1e6)
        for cand in candidates:
            if _within(cand, gold, rel, abs_tol):
                return True
    return False


# --------------------------------------------------------------------------- #
# citations (attribution proxy - NOT claim-level entailment)
# --------------------------------------------------------------------------- #
def citation_metrics(record, question, chunk_by_id):
    cited = record.get("cited_ids") or []
    retrieved = record.get("retrieved_ids") or []
    out = {"has_citation": bool(cited)}
    if not cited:
        out["citation_valid_rate"] = None
        out["citation_relevant_rate"] = None
        return out

    valid = [i for i in cited if 1 <= i <= len(retrieved)]
    out["citation_valid_rate"] = len(valid) / len(cited)

    kind, targets = question_targets(question)
    if kind == "none" or question.get("should_abstain"):
        out["citation_relevant_rate"] = None
    else:
        relevant = 0
        for i in valid:
            chunk = chunk_by_id.get(retrieved[i - 1])
            if chunk and targets_matched_by(chunk, question, kind, targets):
                relevant += 1
        out["citation_relevant_rate"] = relevant / len(cited)
    return out


# --------------------------------------------------------------------------- #
# LLM judge (optional; validate against a human-labelled sample before trusting it)
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are grading an answer to a question about NVIDIA SEC filings.

Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Grade the candidate against the reference:
- "correct": states the same key facts/figures as the reference, with no contradicting claims.
- "partial": gets some key facts right but omits or misstates others.
- "incorrect": wrong, contradicts the reference, or fails to answer (including saying evidence is insufficient).

Respond with only JSON: {{"grade": "correct|partial|incorrect", "reason": "<one sentence>"}}"""

_GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_answer(judge_llm, question, gold_answer, answer):
    result = judge_llm.generate_with_meta(
        JUDGE_PROMPT.format(question=question, gold=gold_answer, answer=answer)
    )
    if result.error:
        return None, f"judge error: {result.error}"
    match = re.search(r"\{.*\}", result.text, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0))
        grade = str(data.get("grade", "")).lower()
        return _GRADE_SCORE.get(grade), data.get("reason", "")
    except Exception:
        return None, f"unparseable judge output: {result.text[:120]}"


# --------------------------------------------------------------------------- #
# per-run scoring
# --------------------------------------------------------------------------- #
def score_run(record, question, chunk_by_id, judge_llm=None):
    """Adds correctness fields to a run record (returns a new dict)."""
    scored = dict(record)
    abstained = bool(record.get("abstained"))
    should_abstain = bool(question.get("should_abstain"))
    final = record.get("final_answer") or extract_final_answer(record.get("answer") or "")

    scored["category"] = question.get("category")
    scored["should_abstain"] = should_abstain
    scored["numeric_correct"] = None
    scored["judge_score"] = None
    scored["judge_reason"] = None

    if record.get("error"):
        scored["correct"] = None
        scored["scoring_note"] = "run errored"
    elif should_abstain:
        scored["correct"] = abstained
        scored["scoring_note"] = "abstention expected"
    elif question.get("gold_numeric"):
        ok = (not abstained) and bool(numeric_match(final, question["gold_numeric"]))
        scored["numeric_correct"] = ok
        scored["correct"] = ok
        scored["scoring_note"] = "numeric"
    elif question.get("gold_answer") and judge_llm is not None:
        if abstained:
            scored["judge_score"], scored["judge_reason"] = 0.0, "abstained"
        else:
            scored["judge_score"], scored["judge_reason"] = judge_answer(
                judge_llm, question["question"], question["gold_answer"], final
            )
        scored["correct"] = None if scored["judge_score"] is None else scored["judge_score"] >= 1.0
        scored["scoring_note"] = "llm judge"
    else:
        scored["correct"] = None
        scored["scoring_note"] = "no gold answer / judge"

    # false abstention on an answerable question
    scored["false_abstain"] = abstained and not should_abstain
    scored["missed_abstain"] = (not abstained) and should_abstain and not record.get("error")

    scored.update(citation_metrics(record, question, chunk_by_id))
    return scored
