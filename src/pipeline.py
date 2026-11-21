"""One pipeline for every experimental condition, with timing + logging built in.

Modes (the ablation axis):
    direct     no retrieval, LLM answers from memory
    vector     FAISS only
    bm25       keyword only
    hybrid     RRF fusion of vector + BM25
    hybrid_fy  hybrid + fiscal-year/form boost
    oracle     gold evidence chunks handed to the LLM (isolates LLM quality from retrieval)

Agentic modes plug in later by registering a callable in AGENTIC_MODES.
"""

import re

from src.utils.run_logger import Timer

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
RETRIEVAL_MODES = {"vector", "bm25", "hybrid", "hybrid_fy"}
AGENTIC_MODES = {}  # e.g. {"agentic": run_agentic_fn} - added in step 5


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def format_evidence(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        period = c.get("fiscal_period")
        label = (
            f"[E{i}] {c.get('filing_type') or '?'} filed {c.get('filing_date') or '?'} "
            f"| FY{c.get('fiscal_year') or '?'}{(' ' + period) if period and period != 'FY' else ''} "
            f"| {c.get('section') or 'section unknown'}"
        )
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(question, chunks):
    return f"""You are a financial research assistant answering questions about NVIDIA using ONLY the SEC filing evidence below.

Rules:
1. Cite every factual claim with evidence ids such as [E1] or [E2][E3].
2. Never use outside knowledge and never guess a number. Copy figures exactly as written.
3. NVIDIA's fiscal year ends in late January: "fiscal 2025" ended January 26, 2025. Check the fiscal year shown on each evidence label.
4. If you calculate anything, write the arithmetic out explicitly (e.g. (130,497 - 60,922) / 60,922).
5. If the evidence does not contain what is needed to answer, the Final Answer must be exactly {ABSTAIN_TOKEN} followed by one sentence saying what is missing.

Question: {question}

Evidence:
{format_evidence(chunks)}

Respond in this format:
Reasoning: <brief, with citations>
Final Answer: <one to three sentences, numbers with units, with citations>"""


def build_direct_prompt(question):
    return f"""You are a financial research assistant. Answer the question about NVIDIA from your own knowledge, with no documents provided.

If you are not confident of the exact answer, the Final Answer must be exactly {ABSTAIN_TOKEN} followed by one sentence explaining why.
NVIDIA's fiscal year ends in late January: "fiscal 2025" ended January 26, 2025.

Question: {question}

Respond in this format:
Reasoning: <brief>
Final Answer: <one to three sentences, numbers with units>"""


# --------------------------------------------------------------------------- #
# answer parsing
# --------------------------------------------------------------------------- #
def extract_final_answer(text):
    """Text after the last 'Final Answer:' marker (whole text if absent)."""
    matches = list(re.finditer(r"final answer\s*[:\-]\s*", text, flags=re.IGNORECASE))
    return text[matches[-1].end():].strip() if matches else text.strip()


def is_abstention(text):
    final = extract_final_answer(text)
    return ABSTAIN_TOKEN.lower() in final.lower().replace(" ", "_") or \
        final.lower().startswith(("insufficient evidence", "the evidence is insufficient"))


def extract_citation_ids(text):
    return sorted({int(n) for n in re.findall(r"\[E(\d+)\]", text)})


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
class RAGPipeline:
    def __init__(self, llm, retrievers=None, top_k=5, logger=None):
        """retrievers: {"vector": obj, "bm25": obj, "hybrid": obj} each with .search(q, top_k=)."""
        self.llm = llm
        self.retrievers = retrievers or {}
        self.top_k = top_k
        self.logger = logger

    def retrieve(self, question, mode, top_k=None):
        top_k = top_k or self.top_k
        if mode == "hybrid_fy":
            return self.retrievers["hybrid"].search(question, top_k=top_k, boost_fiscal_year=True)
        return self.retrievers[mode].search(question, top_k=top_k)

    def run(self, question, mode="hybrid", qid=None, oracle_chunks=None, extra=None):
        """Run one question under one mode. Always returns (and logs) a record, even on failure."""
        timer = Timer()
        record = {
            "qid": qid,
            "question": question,
            "mode": mode,
            "model": getattr(self.llm, "name", type(self.llm).__name__),
            "top_k": self.top_k,
            "retrieved_ids": [],
            "retrieved_meta": [],
            "answer": None,
            "final_answer": None,
            "abstained": None,
            "cited_ids": [],
            "retrieval_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "model_used": None,
            "fallback_used": None,
            "error": None,
            **(extra or {}),
        }

        try:
            chunks = []
            if mode in AGENTIC_MODES:
                return self._finish(record, timer, AGENTIC_MODES[mode](self, question, record))

            if mode == "oracle":
                chunks = list(oracle_chunks or [])
            elif mode in RETRIEVAL_MODES:
                with timer.span("retrieval"):
                    chunks = self.retrieve(question, mode)
            elif mode != "direct":
                raise ValueError(f"Unknown mode '{mode}'")

            record["retrieved_ids"] = [c["chunk_id"] for c in chunks]
            record["retrieved_meta"] = [
                {"chunk_id": c["chunk_id"], "fiscal_year": c.get("fiscal_year"),
                 "filing_type": c.get("filing_type"), "section": c.get("section"),
                 "score": c.get("hybrid_score", c.get("score"))}
                for c in chunks
            ]

            prompt = build_direct_prompt(question) if mode == "direct" else build_prompt(question, chunks)
            record["prompt_chars"] = len(prompt)

            with timer.span("llm"):
                result = self.llm.generate_with_meta(prompt)

            record.update({
                "answer": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "model_used": result.model_used,
                "fallback_used": result.fallback_used,
                "llm_ms": result.latency_ms,
                "error": result.error,
            })
            if not result.error:
                record["final_answer"] = extract_final_answer(result.text)
                record["abstained"] = is_abstention(result.text)
                record["cited_ids"] = extract_citation_ids(result.text)
        except Exception as exc:  # never let one bad question kill a long experiment
            record["error"] = f"{type(exc).__name__}: {exc}"

        return self._finish(record, timer)

    def _finish(self, record, timer, override=None):
        if override:
            record.update(override)
        record["retrieval_ms"] = timer.ms("retrieval")
        if not record["llm_ms"]:
            record["llm_ms"] = timer.ms("llm")
        record["total_ms"] = timer.total_ms()
        if self.logger:
            record = self.logger.log(record)
        return record
