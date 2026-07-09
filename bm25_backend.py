"""In-harness BM25 anchor baseline (dependency-free, Okapi BM25).

Indexes the exact same rendered corpus text Graphon receives:
  * LOCOMO      -- one chunk per dialog turn (with session header context);
  * LongMemEval -- one chunk per dialog turn, labeled with its session id.

Answers always go through the shared reader (bm25 has no answer engine of
its own), so the graphify-style comparison "retriever + shared reader +
shared judge" holds. Their BM25 anchor: LOCOMO 31.3% / 0.362, LME 70% / 0.710.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict

from schemas import BackendResult, RetrievedChunk

logger = logging.getLogger("membench_bm25")

_TOKEN_RE = re.compile(r"[a-z0-9']+")

K1 = 1.5
B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _Index:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf: list[Counter] = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(chunks)
        self.idf = {
            term: math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0) for term, dfi in df.items()
        }

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        q_tokens = _tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        for term in q_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += idf * f * (K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for i, score in ranked:
            c = self.chunks[i]
            out.append(
                RetrievedChunk(
                    text=c.text, score=round(score, 4),
                    source_ref=c.source_ref, session_id=c.session_id,
                )
            )
        return out


class BM25Backend:
    name = "bm25"

    def __init__(self, cfg: dict) -> None:
        self.retrieval_k = int(cfg.get("graphon", {}).get("retrieval_k", 10))
        self._indexes: dict[str, _Index] = {}

    def setup_corpus(self, corpus_key: str, chunks: list[RetrievedChunk]) -> None:
        self._indexes[corpus_key] = _Index(chunks)
        logger.info("BM25 index for '%s': %d chunks", corpus_key, len(chunks))

    def has_corpus(self, corpus_key: str) -> bool:
        return corpus_key in self._indexes

    def query(self, corpus_key: str, question: str) -> BackendResult:
        index = self._indexes.get(corpus_key)
        if index is None:
            return BackendResult(error=f"No BM25 index for corpus '{corpus_key}'.")
        t0 = time.time()
        chunks = index.search(question, self.retrieval_k)
        return BackendResult(chunks=chunks, latency_seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# Chunk builders (shared rendering with data_loader)
# ---------------------------------------------------------------------------
def locomo_chunks(sample: dict) -> list[RetrievedChunk]:
    from data_loader import _locomo_sessions, _render_turn_text

    chunks: list[RetrievedChunk] = []
    for num, date_time, turns in _locomo_sessions(sample["conversation"]):
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = f"[Session {num} — {date_time}] {speaker}: {_render_turn_text(turn)}"
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source_ref=str(turn.get("dia_id", "")),
                    session_id=f"session_{num}",
                )
            )
    return chunks


def lme_chunks(item: dict) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for sess_id, date, turns in zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    ):
        for j, turn in enumerate(turns):
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            chunks.append(
                RetrievedChunk(
                    text=f"[{date}] {role}: {content}",
                    source_ref=f"{sess_id}:{j}",
                    session_id=str(sess_id),
                )
            )
    return chunks
