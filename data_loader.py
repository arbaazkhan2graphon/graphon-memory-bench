"""Load and normalize LOCOMO + LongMemEval-S; render corpora to markdown.

Graphon indexes documents rather than raw JSON, so:
  * each LOCOMO conversation is rendered to ONE markdown document
    (session headers with dates, ``Speaker:`` turns, image captions inlined);
  * each LongMemEval haystack session is rendered to its OWN markdown file
    (named after its session id, so retrieved sources map back to sessions).

No scoring metadata (dialog ids, has_answer flags, gold sessions) is ever
written into the rendered text.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

from schemas import Question

logger = logging.getLogger("membench_loader")

PROJECT_DIR = Path(__file__).resolve().parent

# LOCOMO category id -> name (paper ordering; 5=adversarial is excluded from
# scoring, matching how mem0 / supermemory / graphify report).
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or PROJECT_DIR / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def data_dir(cfg: dict) -> Path:
    return PROJECT_DIR / cfg["data"]["dir"]


# ---------------------------------------------------------------------------
# LOCOMO
# ---------------------------------------------------------------------------
def load_locomo(cfg: dict) -> list[dict]:
    path = data_dir(cfg) / "locomo10.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _locomo_sessions(conv: dict) -> list[tuple[int, str, list[dict]]]:
    """Ordered (session_number, date_time, turns) triples."""
    out = []
    for key, val in conv.items():
        if key.startswith("session_") and isinstance(val, list):
            try:
                num = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            out.append((num, conv.get(f"session_{num}_date_time", ""), val))
    out.sort(key=lambda t: t[0])
    return out


def _render_turn_text(turn: dict) -> str:
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        return f"[shares a photo: {caption}] {text}".strip()
    return text


def render_locomo_conversation(sample: dict) -> str:
    """One markdown document per conversation. No dia_ids leak into the text."""
    conv = sample["conversation"]
    a, b = conv.get("speaker_a", "Speaker A"), conv.get("speaker_b", "Speaker B")
    lines = [f"# Conversation between {a} and {b}", ""]
    for num, date_time, turns in _locomo_sessions(conv):
        lines.append(f"## Session {num} — {date_time}")
        lines.append("")
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            lines.append(f"{speaker}: {_render_turn_text(turn)}")
        lines.append("")
    return "\n".join(lines)


def locomo_turn_index(sample: dict) -> dict[str, str]:
    """dia_id -> rendered turn text (used to resolve gold evidence)."""
    index: dict[str, str] = {}
    for _num, _dt, turns in _locomo_sessions(sample["conversation"]):
        for turn in turns:
            did = turn.get("dia_id")
            if did:
                index[did] = _render_turn_text(turn)
    return index


def locomo_questions(samples: list[dict]) -> list[Question]:
    """All scored LOCOMO questions (categories 1-4; adversarial excluded)."""
    questions: list[Question] = []
    for sample in samples:
        sid = sample["sample_id"]
        turn_idx = locomo_turn_index(sample)
        for i, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category")
            if cat not in LOCOMO_CATEGORIES:
                continue
            answer = qa.get("answer")
            if answer is None:
                continue
            evidence_ids = qa.get("evidence") or []
            evidence_texts = [turn_idx[e] for e in evidence_ids if e in turn_idx]
            questions.append(
                Question(
                    qid=f"{sid}_q{i}",
                    benchmark="locomo",
                    corpus_id=sid,
                    question=str(qa["question"]).strip(),
                    gold_answer=str(answer).strip(),
                    category=LOCOMO_CATEGORIES[cat],
                    evidence_texts=evidence_texts,
                    raw={"evidence_ids": evidence_ids, "category_id": cat},
                )
            )
    return questions


# ---------------------------------------------------------------------------
# LongMemEval-S
# ---------------------------------------------------------------------------
def load_longmemeval(cfg: dict) -> list[dict]:
    path = data_dir(cfg) / "longmemeval_s.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_lme_session(turns: list[dict], date: str) -> str:
    lines = [f"# Chat session on {date}", ""]
    for turn in turns:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines)


def lme_session_files(item: dict, out_dir: Path) -> list[Path]:
    """Write one markdown file per haystack session; return the paths.

    Files are named ``<session_id>.md`` so Graphon source file names map
    straight back to session ids for recall@10 scoring.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    sessions = item["haystack_sessions"]
    for sess_id, date, turns in zip(ids, dates, sessions):
        safe = str(sess_id).replace("/", "_")
        path = out_dir / f"{safe}.md"
        path.write_text(render_lme_session(turns, date), encoding="utf-8")
        paths.append(path)
    return paths


def lme_questions(items: list[dict]) -> list[Question]:
    questions: list[Question] = []
    for item in items:
        qid = item["question_id"]
        questions.append(
            Question(
                qid=qid,
                benchmark="longmemeval",
                corpus_id=qid,  # each question has its own haystack/group
                question=str(item["question"]).strip(),
                gold_answer=str(item["answer"]).strip(),
                category=item["question_type"],
                evidence_session_ids=[str(s) for s in item.get("answer_session_ids", [])],
                question_date=str(item.get("question_date", "")),
                is_abstention=str(qid).endswith("_abs"),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def stratified_sample(questions: list[Question], n: int, seed: int) -> list[Question]:
    """Deterministic proportional sample stratified by category.

    Largest-remainder rounding so per-category counts sum exactly to ``n``.
    """
    if n >= len(questions):
        return list(questions)
    by_cat: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)
    total = len(questions)
    cats = sorted(by_cat)
    quotas = {c: n * len(by_cat[c]) / total for c in cats}
    counts = {c: int(quotas[c]) for c in cats}
    remainder = n - sum(counts.values())
    for c in sorted(cats, key=lambda c: quotas[c] - counts[c], reverse=True)[:remainder]:
        counts[c] += 1

    rng = random.Random(seed)
    picked: list[Question] = []
    for c in cats:
        pool = sorted(by_cat[c], key=lambda q: q.qid)
        rng2 = random.Random(f"{seed}:{c}")
        rng2.shuffle(pool)
        picked.extend(pool[: counts[c]])
    rng.shuffle(picked)
    logger.info(
        "Sampled %d/%d questions: %s",
        len(picked), total, {c: counts[c] for c in cats},
    )
    return picked
