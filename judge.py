"""Shared LLM roles: reader, key-fact decomposition, and coverage judge.

One model (config ``llm.model``, default gpt-4o) fills every LLM role,
mirroring the graphify harness's "one shared model" fairness rule.

Grading is key-fact coverage:
  1. A cached pre-pass decomposes each gold answer into atomic key facts.
  2. The judge marks each fact covered / partial / missing, citing a verbatim
     quote from the candidate answer for every non-missing verdict.
  3. coverage = (covered + 0.5 * partial) / total.

All token usage is recorded in a spend ledger with a hard --max-spend stop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from schemas import FactVerdict, GradeResult, Question, RetrievedChunk

logger = logging.getLogger("membench_judge")

PROJECT_DIR = Path(__file__).resolve().parent


class SpendLimitExceeded(RuntimeError):
    pass


class Ledger:
    """Per-role token/cost accounting; enforces the max-spend guard."""

    def __init__(self, input_cost_per_m: float, output_cost_per_m: float, max_spend: float):
        self.input_cost_per_m = input_cost_per_m
        self.output_cost_per_m = output_cost_per_m
        self.max_spend = max_spend
        self.roles: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def record(self, role: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            entry = self.roles.setdefault(role, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["calls"] += 1
        if self.total_cost() > self.max_spend:
            raise SpendLimitExceeded(
                f"OpenAI spend ${self.total_cost():.2f} exceeded --max-spend ${self.max_spend:.2f}"
            )

    def total_cost(self) -> float:
        i = sum(r["input_tokens"] for r in self.roles.values())
        o = sum(r["output_tokens"] for r in self.roles.values())
        return i / 1e6 * self.input_cost_per_m + o / 1e6 * self.output_cost_per_m

    def summary(self) -> dict:
        return {
            "roles": self.roles,
            "total_cost_usd": round(self.total_cost(), 4),
            "pricing_per_m": {
                "input": self.input_cost_per_m,
                "output": self.output_cost_per_m,
            },
        }


class LLMClient:
    """OpenAI-compatible chat client shared by reader/decomposer/judge."""

    def __init__(self, cfg: dict, ledger: Ledger) -> None:
        from openai import OpenAI

        lcfg = cfg.get("llm", {})
        self.model = os.environ.get("OPENAI_MODEL") or lcfg.get("model", "gpt-4o")
        self.temperature = float(lcfg.get("temperature", 0.0))
        self.max_tokens = int(lcfg.get("max_tokens", 1024))
        self.ledger = ledger
        kwargs = {}
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            kwargs["base_url"] = base
        self.client = OpenAI(**kwargs)

    def chat(self, role: str, system: str, user: str, json_mode: bool = False,
             attempts: int = 3) -> tuple[str, int, int, float]:
        """Returns (text, input_tokens, output_tokens, latency_seconds)."""
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"} if json_mode else {"type": "text"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except SpendLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last = exc
                if attempt < attempts:
                    time.sleep(3 * attempt)
                    continue
                raise
            latency = time.time() - t0
            usage = resp.usage
            it = getattr(usage, "prompt_tokens", 0) or 0
            ot = getattr(usage, "completion_tokens", 0) or 0
            self.ledger.record(role, it, ot)
            return (resp.choices[0].message.content or "", it, ot, latency)
        raise RuntimeError(f"chat failed: {last}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared reader (answer from retrieved context)
# ---------------------------------------------------------------------------
READER_SYSTEM = """\
You are a helpful assistant answering a question using ONLY the retrieved
memory excerpts below. Be concise and specific. If the excerpts do not contain
the information needed, say the information is not available in the
conversation history. When the question involves dates or times, reason
carefully from the timestamps in the excerpts."""

READER_TEMPLATE = """\
Retrieved memory excerpts (top {k}):
{context}

{date_line}Question: {question}

Answer concisely:"""


def read_answer(llm: LLMClient, question: Question, chunks: list[RetrievedChunk]
                ) -> tuple[str, int, int, float]:
    context = "\n\n".join(
        f"[{i + 1}] {c.text}" for i, c in enumerate(chunks)
    ) or "(no excerpts retrieved)"
    date_line = f"Today's date: {question.question_date}\n" if question.question_date else ""
    user = READER_TEMPLATE.format(
        k=len(chunks), context=context, date_line=date_line, question=question.question
    )
    return llm.chat("reader", READER_SYSTEM, user)


# ---------------------------------------------------------------------------
# Key-fact decomposition (cached)
# ---------------------------------------------------------------------------
DECOMPOSE_SYSTEM = """\
You decompose gold answers into atomic key facts for grading. Each key fact is
one minimal, independently checkable statement a correct answer must convey.
Keep facts short. Most short answers are a single fact. Do not invent facts
that are not in the gold answer. Respond in JSON:
{"facts": ["...", "..."]}"""

DECOMPOSE_TEMPLATE = """\
Question: {question}
Gold answer: {gold}

Decompose the gold answer into atomic key facts (usually 1-3)."""

ABSTENTION_FACT = (
    "The answer states that the requested information is not available / was "
    "never mentioned in the conversation history."
)


class KeyFactStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        if path.exists():
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt key-fact cache; starting fresh.")

    def get_or_build(self, llm: LLMClient, question: Question) -> list[str]:
        with self._lock:
            if question.qid in self.cache:
                return self.cache[question.qid]
        if question.is_abstention:
            facts = [ABSTENTION_FACT]
        else:
            user = DECOMPOSE_TEMPLATE.format(
                question=question.question, gold=question.gold_answer
            )
            text, *_ = llm.chat("decomposer", DECOMPOSE_SYSTEM, user, json_mode=True)
            try:
                facts = [str(f).strip() for f in json.loads(text).get("facts", []) if str(f).strip()]
            except json.JSONDecodeError:
                facts = []
            if not facts:
                facts = [question.gold_answer]
        with self._lock:
            self.cache[question.qid] = facts
            self._save()
        return facts

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Coverage judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are a strict but fair grading judge. You are given a question, a list of
gold key facts, and a candidate answer. For EACH key fact, decide:
  - "covered": the answer clearly conveys the fact (exact wording not required;
    semantically equivalent phrasing, paraphrase, or equivalent date/number
    formats count).
  - "partial": the answer conveys part of the fact or is ambiguous about it.
  - "missing": the answer does not convey the fact or contradicts it.
For "covered" and "partial" you MUST cite a short verbatim quote from the
candidate answer that supports your verdict. Respond in JSON:
{"verdicts": [{"fact": "...", "verdict": "covered|partial|missing", "quote": "..."}]}"""

JUDGE_TEMPLATE = """\
Question: {question}

Gold key facts:
{facts}

Candidate answer:
{answer}

Grade each key fact."""

_VALID_VERDICTS = {"covered", "partial", "missing"}


def grade_answer(llm: LLMClient, question: Question, facts: list[str], answer: str
                 ) -> GradeResult:
    if not answer.strip():
        return GradeResult(
            coverage=0.0,
            verdicts=[FactVerdict(fact=f, verdict="missing", quote="") for f in facts],
        )
    facts_block = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    user = JUDGE_TEMPLATE.format(question=question.question, facts=facts_block, answer=answer)
    try:
        text, *_ = llm.chat("judge", JUDGE_SYSTEM, user, json_mode=True)
        raw = json.loads(text).get("verdicts", [])
    except SpendLimitExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        return GradeResult(coverage=0.0, judge_error=str(exc))

    verdicts: list[FactVerdict] = []
    for i, fact in enumerate(facts):
        v = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        verdict = str(v.get("verdict", "missing")).lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "missing"
        quote = str(v.get("quote", "") or "")
        verdicts.append(FactVerdict(fact=fact, verdict=verdict, quote=quote))
    covered = sum(1 for v in verdicts if v.verdict == "covered")
    partial = sum(1 for v in verdicts if v.verdict == "partial")
    coverage = (covered + 0.5 * partial) / len(facts) if facts else 0.0
    return GradeResult(coverage=round(coverage, 4), verdicts=verdicts)


# ---------------------------------------------------------------------------
# Retrieval recall@10
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9']+")


def _norm_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def locomo_recall_at_k(question: Question, chunks: list[RetrievedChunk],
                       ngram: int = 5, threshold: float = 0.6) -> float | None:
    """Fraction of gold evidence turns present in the concatenated top-k chunks.

    A turn counts as retrieved when >= ``threshold`` of its word n-grams appear
    in the concatenation of the retrieved chunk texts (robust to chunking that
    splits a turn across chunks).
    """
    if not question.evidence_texts:
        return None
    blob_tokens = _norm_tokens(" \n ".join(c.text for c in chunks))
    blob_grams = _ngrams(blob_tokens, ngram)
    hits = 0
    for turn_text in question.evidence_texts:
        grams = _ngrams(_norm_tokens(turn_text), ngram)
        if not grams:
            continue
        frac = len(grams & blob_grams) / len(grams)
        if frac >= threshold:
            hits += 1
    return hits / len(question.evidence_texts)


def lme_recall_at_k(question: Question, chunks: list[RetrievedChunk]) -> float | None:
    """Session-level recall: fraction of gold sessions among retrieved chunks."""
    gold = set(question.evidence_session_ids)
    if not gold:
        return None
    retrieved = {c.session_id for c in chunks if c.session_id}
    return len(gold & retrieved) / len(gold)
