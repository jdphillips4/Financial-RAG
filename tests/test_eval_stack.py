"""Offline tests for the new stack. No API keys, no network, no embedding model needed.

    pip install pytest
    pytest tests/ -q
"""

import json

from evaluation.generate_xbrl_questions import build_questions
from evaluation.metrics import (
    extract_numbers, numeric_match, retrieval_metrics, score_run,
)
from src.ingestion.chunk_documents import (
    chunk_blocks, fiscal_info, parse_heading, split_into_blocks, split_table,
)
from src.ingestion.parse_filings import clean_text
from src.models.llm import FakeLLM
from src.pipeline import (
    ABSTAIN_TOKEN, RAGPipeline, extract_citation_ids, extract_final_answer, is_abstention,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.filters import parse_query_hints
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.tokenize import tokenize
from src.utils.run_logger import RunLogger

# ---------------------------------------------------------------- parsing
HTML = """
<html><body>
<ix:header><ix:hidden>HIDDENTAG 12345 dei:EntityCentralIndexKey</ix:hidden></ix:header>
<div style="display:none">invisible junk</div>
<p>Revenue by Reportable Segments</p>
<table>
 <tr><td>Fiscal Year</td><td>2025</td><td></td><td>2024</td></tr>
 <tr><td>Revenue</td><td>$</td><td>130,497</td><td>$</td><td>60,922</td></tr>
 <tr><td>Other, net</td><td>(1,234</td><td>)</td><td>5.0</td><td>%</td></tr>
</table>
<p>Management discussion text.</p>
</body></html>
"""


def test_parse_removes_hidden_and_merges_cells():
    text = clean_text(HTML)
    assert "HIDDENTAG" not in text and "invisible junk" not in text
    assert "Revenue | $130,497 | $60,922" in text
    assert "(1,234)" in text and "5.0%" in text
    assert "[TABLE]" in text and "[/TABLE]" in text


# ---------------------------------------------------------------- chunking
def test_fiscal_info():
    assert fiscal_info("2025-01-26", "10-K") == (2025, "FY")
    assert fiscal_info("2025-04-27", "10-Q") == (2026, "Q1")
    assert fiscal_info("2025-07-27", "10-Q") == (2026, "Q2")
    assert fiscal_info("2024-10-27", "10-Q") == (2025, "Q3")


def test_heading_detection_ignores_prose_references():
    assert parse_heading("Item 7. Management's Discussion and Analysis")[1].startswith("Item 7.")
    assert parse_heading("PART II Item 1A. Risk Factors")[0] == "II"
    prose = "As discussed in Item 1A. Risk Factors of this report, our results may vary widely from period to period."
    assert parse_heading(prose) is None


def test_chunking_sections_tables_overlap():
    para = " ".join(f"word{i}" for i in range(60))
    rows = "\n".join(f"Row {i} | {i * 100} | {i * 90}" for i in range(80))
    text = (
        "Item 1. Business\n" + para + "\n" + para + "\n" + para + "\n"
        "Item 7. Management's Discussion and Analysis\n" + para + "\n"
        "Revenue by Reportable Segments\n[TABLE]\nYear | 2025 | 2024\n" + rows + "\n[/TABLE]\n"
    )
    chunks = chunk_blocks(split_into_blocks(text), chunk_size=100, overlap=30)

    sections = {c["section"] for c in chunks}
    assert any(s.startswith("Item 1.") for s in sections)
    assert any(s.startswith("Item 7.") for s in sections)
    # a chunk never mixes sections
    business = [c for c in chunks if c["section"].startswith("Item 1.")]
    assert all("Item 7." not in c["text"] for c in business)
    assert len(business) >= 2
    # overlap: consecutive business chunks share text
    assert set(business[0]["text"].split()[-10:]) & set(business[1]["text"].split())
    # big table split with the header repeated, caption attached
    tables = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(tables) > 1
    assert all("Year | 2025 | 2024" in t["text"] for t in tables)
    assert tables[0]["text"].startswith("Revenue by Reportable Segments")
    assert all(t["text"].count("[TABLE]") == 1 for t in tables)


def test_split_table_small_table_untouched():
    t = "[TABLE]\nA | 1\nB | 2\n[/TABLE]"
    assert split_table(t, 100) == [t]


# ---------------------------------------------------------------- retrieval
def make_chunks():
    def c(i, fy, form, text, section="Item 7. MD&A"):
        return {"chunk_id": f"c{i}", "fiscal_year": fy, "filing_type": form, "fiscal_period": "FY",
                "section": section, "chunk_type": "text", "text": text,
                "search_text": f"NVIDIA {form} FY{fy} | {section}\n{text}"}
    return [
        c(0, 2025, "10-K", "Revenue was $130,497 million in fiscal 2025, up from $60,922 million."),
        c(1, 2024, "10-K", "Revenue was $60,922 million in fiscal 2024."),
        c(2, 2025, "10-K", "Gross margin increased to 75.0% due to product mix and lower inventory provisions."),
        c(3, 2025, "10-K", "Export controls may restrict sales of our products to China.", "Item 1A. Risk Factors"),
        c(4, 2023, "10-K", "Our gross margin was 56.9% in fiscal 2023."),
    ]


class OverlapVector:
    """Tiny stand-in for the FAISS store (word overlap), so tests need no model download."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
        q = set(tokenize(query))
        scored = []
        for ch in self.chunks:
            score = len(q & set(tokenize(ch["search_text"]))) / (len(q) or 1)
            scored.append((score, ch))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, ch in scored[:top_k]:
            r = ch.copy()
            r["score"] = s
            out.append(r)
        return out


def test_tokenizer_keeps_numbers():
    toks = tokenize("Revenue of $130,497 million, up 12.5% from the prior year")
    assert "130,497" in toks and "130497" in toks and "12.5%" in toks
    assert "the" not in toks and "of" not in toks


def test_query_hints():
    assert parse_query_hints("What was revenue in fiscal 2025?")["fiscal_years"] == [2025]
    assert parse_query_hints("FY2024 gross margin")["fiscal_years"] == [2024]
    assert parse_query_hints("risks in its 2025 annual report")["filing_type"] == "10-K"
    assert parse_query_hints("what is gross margin")["fiscal_years"] == []


def test_bm25_and_hybrid_fiscal_year_boost():
    chunks = make_chunks()
    bm25 = BM25Retriever(chunks=chunks)
    assert bm25.search("export controls China", top_k=1)[0]["chunk_id"] == "c3"

    hybrid = HybridRetriever(vector_store=OverlapVector(chunks), bm25=bm25)
    top = hybrid.search("What was revenue in fiscal 2024?", top_k=2, boost_fiscal_year=True)
    assert top[0]["fiscal_year"] == 2024
    plain = hybrid.search("What was revenue in fiscal 2024?", top_k=2)
    assert {"vector_rank", "keyword_rank", "hybrid_score"} <= set(plain[0])


# ---------------------------------------------------------------- metrics
def test_retrieval_metrics_snippets_and_sources():
    chunks = make_chunks()
    q = {"id": "t1", "question": "x", "gold_evidence_snippets": ["130,497", "60,922"],
         "acceptable_sources": [{"filing_type": "10-K", "fiscal_year": 2025}]}
    m = retrieval_metrics([chunks[3], chunks[0], chunks[1]], q)
    assert m["mrr"] == 0.5 and m["hit@1"] == 0 and m["hit@3"] == 1
    # chunk 1 contains 60,922 but is FY2024 -> not an acceptable source; chunk 0 holds both numbers
    assert m["coverage@3"] == 1.0
    assert retrieval_metrics(chunks, {"id": "t2", "question": "x", "should_abstain": True}) is None
    assert retrieval_metrics(chunks, {"id": "t3", "question": "x"}) is None


def test_numeric_match_units():
    g = {"value": 130497, "unit": "usd_millions", "tolerance_rel": 0.005}
    assert numeric_match("Revenue was $130.5 billion [E1]", g)
    assert numeric_match("Revenue was $130,497 million", g)
    assert numeric_match("Revenue was 130497", g)
    assert not numeric_match("Revenue was $60.9 billion", g)
    pct = {"value": 114.2, "unit": "percent", "tolerance_abs": 0.3}
    assert numeric_match("an increase of 114.2%", pct)
    assert numeric_match("grew 114 percent", pct)                # 114.0 is 0.2 away -> within 0.3
    assert not numeric_match("grew 110 percent", pct)           # 4.2 away -> wrong
    assert extract_numbers("(1,234)")[0][0] == -1234


def test_abstention_and_answer_parsing():
    text = f"Reasoning: none.\nFinal Answer: {ABSTAIN_TOKEN} The filings do not state it."
    assert is_abstention(text)
    assert not is_abstention("Reasoning: x\nFinal Answer: $130.5 billion [E1]")
    assert extract_final_answer("Reasoning: a\nFinal Answer: 42 [E2]") == "42 [E2]"
    assert extract_citation_ids("a [E1] b [E3][E1]") == [1, 3]


# ---------------------------------------------------------------- pipeline + logging
def test_pipeline_modes_logging_and_scoring(tmp_path):
    chunks = make_chunks()
    bm25 = BM25Retriever(chunks=chunks)
    hybrid = HybridRetriever(vector_store=OverlapVector(chunks), bm25=bm25)
    retrievers = {"bm25": bm25, "vector": OverlapVector(chunks), "hybrid": hybrid}

    def responder(prompt):
        if "130,497" in prompt:
            return "Reasoning: see [E1].\nFinal Answer: Revenue was $130,497 million [E1]."
        return f"Reasoning: none.\nFinal Answer: {ABSTAIN_TOKEN} nothing relevant."

    log = RunLogger(tmp_path / "runs.jsonl")
    pipe = RAGPipeline(FakeLLM(responder, name="fake:test"), retrievers, top_k=3, logger=log)

    q = {"id": "q1", "question": "What was NVIDIA's revenue in fiscal 2025?", "category": "reported_fact",
         "gold_numeric": {"value": 130497, "unit": "usd_millions", "tolerance_rel": 0.005},
         "gold_evidence_snippets": ["130,497"],
         "acceptable_sources": [{"filing_type": "10-K", "fiscal_year": 2025}]}
    by_id = {c["chunk_id"]: c for c in chunks}

    rec = pipe.run(q["question"], mode="hybrid_fy", qid="q1", extra={"category": "reported_fact"})
    assert rec["retrieval_ms"] >= 0 and rec["llm_ms"] >= 0 and rec["total_ms"] > 0
    assert rec["retrieved_ids"] and rec["cited_ids"] == [1] and rec["abstained"] is False
    scored = score_run(rec, q, by_id)
    assert scored["correct"] is True and scored["citation_valid_rate"] == 1.0

    direct = pipe.run(q["question"], mode="direct", qid="q1")
    assert direct["retrieved_ids"] == [] and direct["abstained"] is True
    assert score_run(direct, q, by_id)["false_abstain"] is True

    oracle = pipe.run(q["question"], mode="oracle", qid="q1", oracle_chunks=[chunks[0]])
    assert oracle["retrieved_ids"] == ["c0"] and score_run(oracle, q, by_id)["correct"] is True

    bad = pipe.run(q["question"], mode="nonsense", qid="q1")
    assert bad["error"] and "nonsense" in bad["error"]

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4 and all(json.loads(l)["run_id"] for l in lines)
    assert ("q1", "hybrid_fy", "fake:test") in log.completed()
    assert ("q1", "nonsense", "fake:test") not in log.completed()     # errors are retried on resume


def test_abstention_question_scoring():
    q = {"id": "a1", "question": "AMD revenue?", "should_abstain": True}
    ok = score_run({"qid": "a1", "abstained": True, "final_answer": ABSTAIN_TOKEN}, q, {})
    miss = score_run({"qid": "a1", "abstained": False, "final_answer": "$5B"}, q, {})
    assert ok["correct"] is True and miss["correct"] is False and miss["missed_abstain"] is True


# ---------------------------------------------------------------- xbrl generator
def fake_facts():
    def entry(fy_end, start, val, filed):
        return {"start": start, "end": fy_end, "val": val, "form": "10-K", "filed": filed, "fp": "FY"}
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            entry("2024-01-28", "2023-01-30", 60922e6, "2024-02-21"),
            entry("2025-01-26", "2024-01-29", 130497e6, "2025-02-26"),
        ]}},
        "GrossProfit": {"units": {"USD": [
            entry("2024-01-28", "2023-01-30", 44301e6, "2024-02-21"),
            entry("2025-01-26", "2024-01-29", 97858e6, "2025-02-26"),
        ]}},
    }}}


def test_xbrl_question_generation():
    qs = build_questions(fake_facts(), n_years=2)
    by_text = {q["question"]: q for q in qs}
    rev = by_text["What was NVIDIA's total revenue in fiscal 2025?"]
    assert rev["gold_numeric"]["value"] == 130497 and rev["gold_evidence_snippets"] == ["130,497"]
    growth = by_text["By what percentage did NVIDIA's total revenue change between fiscal 2024 and fiscal 2025?"]
    assert abs(growth["gold_numeric"]["value"] - 114.2) < 0.1
    gm = by_text["What was NVIDIA's gross margin in fiscal 2025, as a percentage of revenue?"]
    assert abs(gm["gold_numeric"]["value"] - 74.99) < 0.1
    assert all(q["verified"] for q in qs) and len({q["id"] for q in qs}) == len(qs)
