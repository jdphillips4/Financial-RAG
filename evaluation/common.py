"""Shared helpers for the evaluation scripts."""

import json
import re
from pathlib import Path

from src.config import NVIDIA_PROCESSED_DIR, PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "evaluation"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"
CHUNKS_PATH = NVIDIA_PROCESSED_DIR / "chunks.json"


def load_questions(path=None):
    data = json.loads(Path(path or QUESTIONS_PATH).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def load_chunks(path=None):
    return json.loads(Path(path or CHUNKS_PATH).read_text(encoding="utf-8"))


def norm(text):
    """Normalise for snippet matching: lowercase, drop table pipes, collapse whitespace."""
    text = (text or "").lower().replace("\xa0", " ")
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def source_ok(chunk, acceptable_sources):
    """True if the chunk comes from one of the acceptable filings (or none are specified)."""
    if not acceptable_sources:
        return True
    for src in acceptable_sources:
        if src.get("filing_type") and chunk.get("filing_type") != src["filing_type"]:
            continue
        if src.get("fiscal_year") and chunk.get("fiscal_year") != src["fiscal_year"]:
            continue
        if src.get("fiscal_period") and chunk.get("fiscal_period") != src["fiscal_period"]:
            continue
        return True
    return False


def question_targets(question):
    """The evidence a retrieval must surface: verbatim snippets (preferred) or chunk ids."""
    snippets = [norm(s) for s in question.get("gold_evidence_snippets") or [] if s.strip()]
    if snippets:
        return "snippet", snippets
    ids = question.get("gold_chunk_ids") or []
    return ("chunk_id", ids) if ids else ("none", [])


def targets_matched_by(chunk, question, kind, targets):
    """Indices of targets this chunk satisfies."""
    if kind == "snippet":
        if not source_ok(chunk, question.get("acceptable_sources")):
            return set()
        body = norm(chunk.get("text"))
        return {i for i, t in enumerate(targets) if t in body}
    if kind == "chunk_id":
        return {i for i, t in enumerate(targets) if t == chunk["chunk_id"]}
    return set()


def gold_chunks_for(question, chunks):
    """All chunks that satisfy any gold target (used by the 'oracle' mode)."""
    kind, targets = question_targets(question)
    if kind == "none":
        return []
    return [c for c in chunks if targets_matched_by(c, question, kind, targets)]


def build_retrievers(modes):
    """Build only what the requested modes need; the embedding model loads once."""
    from src.retrieval.bm25 import BM25Retriever

    modes = set(modes)
    retrievers = {"bm25": BM25Retriever()}
    if modes & {"vector", "hybrid", "hybrid_fy"}:
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.vector_store import VectorStore

        vector = VectorStore()
        retrievers["vector"] = vector
        retrievers["hybrid"] = HybridRetriever(vector_store=vector, bm25=retrievers["bm25"])
    return retrievers


def config_snapshot():
    from src import config

    return {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K,
    }
