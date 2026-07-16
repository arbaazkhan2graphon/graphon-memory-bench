"""Shared dataclasses for the LOCOMO / LongMemEval-S -> Graphon benchmark.

Plain dataclasses (no Pydantic) so everything serializes trivially to
JSON/JSONL and the harness stays easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BENCHMARKS = ("locomo", "longmemeval")
BACKENDS = ("graphon", "bm25", "mem0")
# graphon runs in two answer modes; bm25/mem0 only in shared_reader.
ANSWER_MODES = ("direct", "shared_reader")


@dataclass
class Question:
    """One benchmark question, normalized across datasets."""

    qid: str
    benchmark: str                 # "locomo" | "longmemeval"
    corpus_id: str                 # LOCOMO conversation id / LME question id
    question: str
    gold_answer: str
    category: str                  # LOCOMO category name / LME question_type
    # LOCOMO: gold evidence dialog ids (e.g. "D1:3") -> resolved turn texts.
    evidence_texts: list[str] = field(default_factory=list)
    # LME: gold evidence session ids.
    evidence_session_ids: list[str] = field(default_factory=list)
    question_date: str = ""        # LME only
    is_abstention: bool = False    # LME *_abs questions
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text: str
    score: float | None = None
    source_ref: str = ""           # Graphon SRC key / bm25 chunk id
    session_id: str = ""           # which session file/turn this chunk belongs to


@dataclass
class BackendResult:
    """What a backend returns for one question."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    direct_answer: str = ""        # Graphon's own answer ("" for bm25)
    latency_seconds: float = 0.0
    group_id: str = ""
    error: str = ""


@dataclass
class FactVerdict:
    fact: str
    verdict: str                   # "covered" | "partial" | "missing"
    quote: str                     # verbatim quote from the answer ("" if missing)


@dataclass
class GradeResult:
    coverage: float                # (covered + 0.5*partial) / total
    verdicts: list[FactVerdict] = field(default_factory=list)
    judge_error: str = ""


@dataclass
class RowRecord:
    """One (question x backend x mode) evaluation row, appended to JSONL."""

    qid: str
    benchmark: str
    backend: str                   # "graphon" | "bm25"
    mode: str                      # "direct" | "shared_reader"
    effort: str                    # graphon reasoning effort ("-" for bm25)
    category: str
    question: str
    gold_answer: str
    answer: str
    coverage: float
    recall_at_10: float | None
    verdicts: list[dict[str, str]]
    retrieval_latency: float
    reader_latency: float
    input_tokens: int
    output_tokens: int
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "benchmark": self.benchmark,
            "backend": self.backend,
            "mode": self.mode,
            "effort": self.effort,
            "category": self.category,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "answer": self.answer,
            "coverage": self.coverage,
            "recall_at_10": self.recall_at_10,
            "verdicts": self.verdicts,
            "retrieval_latency": self.retrieval_latency,
            "reader_latency": self.reader_latency,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }
