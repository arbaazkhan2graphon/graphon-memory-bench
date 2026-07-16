"""mem0 backend: their hosted platform, run inside this harness.

Apples-to-apples with the other systems:
  * ingest the identical conversations (one mem0 ``user_id`` per corpus);
  * search top-10 memories per question;
  * answers via the same shared gpt-4o reader, graded by the same judge.

Ingestion follows mem0's own LOCOMO evaluation shape: one ``add`` call per
session with ``Speaker: text`` messages and the session date passed both as a
timestamp and inline, so their extraction pipeline sees the same temporal
signal Graphon and BM25 get from the rendered markdown.

Ingested corpora are recorded in .mem0_ingest_cache.json so reruns skip
re-ingesting (mem0 charges per extraction).
"""

from __future__ import annotations

import calendar
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from schemas import BackendResult, RetrievedChunk

logger = logging.getLogger("membench_mem0")

# e.g. "1:56 pm on 8 May, 2023" (LOCOMO) -> datetime
_LOCOMO_DT_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I
)
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def _parse_locomo_dt(s: str) -> int | None:
    m = _LOCOMO_DT_RE.search(s or "")
    if not m:
        return None
    hour, minute, ampm, day, month_name, year = m.groups()
    month = _MONTHS.get(month_name.lower())
    if not month:
        return None
    h = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    try:
        dt = datetime(int(year), month, int(day), h, int(minute))
    except ValueError:
        return None
    return int(dt.timestamp())


def _parse_lme_date(s: str) -> int | None:
    # LME dates look like "2023/05/20 (Sat) 02:21"
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})", s or "")
    if not m:
        return None
    try:
        return int(datetime(int(m[1]), int(m[2]), int(m[3]), 12, 0).timestamp())
    except ValueError:
        return None


class Mem0Backend:
    name = "mem0"

    def __init__(self, cfg: dict) -> None:
        import os

        from mem0 import MemoryClient

        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY is not set (env or .env).")
        self.client = MemoryClient(api_key=api_key)
        self.retrieval_k = int(cfg.get("graphon", {}).get("retrieval_k", 10))
        mcfg = cfg.get("mem0", {})
        self.user_prefix = mcfg.get("user_prefix", "membench")
        self.add_retries = int(mcfg.get("add_retries", 4))

        project_dir = Path(__file__).resolve().parent
        self.cache_file = project_dir / ".mem0_ingest_cache.json"
        self._ingested: dict[str, dict] = self._load_cache()
        self._lock = threading.Lock()

    # -- ingest state ---------------------------------------------------------
    def _load_cache(self) -> dict[str, dict]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt mem0 ingest cache; ignoring.")
        return {}

    def _save_cache(self) -> None:
        with self._lock:
            self.cache_file.write_text(
                json.dumps(self._ingested, indent=2), encoding="utf-8"
            )

    def user_id(self, corpus_key: str) -> str:
        return f"{self.user_prefix}-{corpus_key}".replace(":", "-")

    def memory_count(self, corpus_key: str) -> int:
        # AGENT_RULE: read the "count" field, not len(results) — the API caps
        # each page at 100, and a capped count makes _wait_for_settle declare
        # a still-growing store "settled".
        resp = self.client.get_all(
            filters={"user_id": self.user_id(corpus_key)}, page_size=100
        )
        if isinstance(resp, dict) and "count" in resp:
            return int(resp["count"])
        results = resp.get("results", resp) if isinstance(resp, dict) else resp
        return len(results or [])

    def has_corpus(self, corpus_key: str) -> bool:
        return self._ingested.get(corpus_key, {}).get("done", False)

    # -- ingestion -------------------------------------------------------------
    def _add_with_retry(self, messages: list[dict], user_id: str,
                        timestamp: int | None) -> None:
        kwargs: dict = {"user_id": user_id}
        if timestamp:
            kwargs["timestamp"] = timestamp
        delay = 2.0
        for attempt in range(self.add_retries + 1):
            try:
                self.client.add(messages, **kwargs)
                return
            except Exception as exc:  # noqa: BLE001 - surface after retries
                if attempt == self.add_retries:
                    raise
                logger.warning("mem0 add failed (%s); retry in %.0fs", exc, delay)
                time.sleep(delay)
                delay *= 2

    def ingest_sessions(self, corpus_key: str,
                        sessions: list[tuple[str, int | None, list[dict]]]) -> float | None:
        """Ingest [(session_label, timestamp, messages)] once; returns seconds.

        ``messages`` are mem0-format dicts ({"role", "content"}). Returns None
        when the corpus was already ingested (cache hit).
        """
        state = self._ingested.setdefault(corpus_key, {"done": False, "sessions": []})
        if state["done"]:
            return None
        uid = self.user_id(corpus_key)
        t0 = time.time()
        done_sessions = set(state["sessions"])
        for label, ts, messages in sessions:
            if label in done_sessions:
                continue
            self._add_with_retry(messages, uid, ts)
            state["sessions"].append(label)
            self._save_cache()
        # Adds are processed asynchronously server-side; wait until the
        # extracted memory count stops growing before declaring the corpus ready.
        self._wait_for_settle(corpus_key)
        state["done"] = True
        self._save_cache()
        secs = time.time() - t0
        logger.info("mem0 ingested '%s' (%d sessions) as user %s in %.1fs",
                    corpus_key, len(sessions), uid, secs)
        return secs

    # AGENT_RULE: mem0 extraction lags add() acks by minutes — querying before
    # the memory count is stable for >= 2 minutes silently scores mem0 against
    # a half-built store (observed: "settled" at 17, real count 54).
    def _wait_for_settle(self, corpus_key: str, checks: int = 5,
                         interval: float = 30.0, timeout: float = 1800.0) -> None:
        deadline = time.time() + timeout
        last, stable = -1, 0
        while time.time() < deadline:
            try:
                count = self.memory_count(corpus_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mem0 settle check failed for '%s': %s", corpus_key, exc)
                time.sleep(interval)
                continue
            stable = stable + 1 if count == last else 0
            last = count
            if stable >= checks - 1 and count > 0:
                logger.info("mem0 '%s' settled at %d memories", corpus_key, count)
                return
            time.sleep(interval)
        logger.warning("mem0 '%s' did not settle within %.0fs (last count %d)",
                       corpus_key, timeout, last)

    # -- query ------------------------------------------------------------------
    def query(self, corpus_key: str, question: str) -> BackendResult:
        uid = self.user_id(corpus_key)
        t0 = time.time()
        try:
            resp = self.client.search(
                question, filters={"user_id": uid}, top_k=self.retrieval_k
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            return BackendResult(latency_seconds=time.time() - t0, error=str(exc))
        latency = time.time() - t0

        results = resp.get("results", resp) if isinstance(resp, dict) else resp
        chunks: list[RetrievedChunk] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            text = item.get("memory") or item.get("text") or ""
            if not text:
                continue
            md = item.get("metadata") or {}
            chunks.append(
                RetrievedChunk(
                    text=text,
                    score=item.get("score"),
                    source_ref=str(item.get("id", "")),
                    session_id=str(md.get("session_id", "")),
                )
            )
        return BackendResult(chunks=chunks[: self.retrieval_k],
                             latency_seconds=latency)


# ---------------------------------------------------------------------------
# Session builders (same source data as the other backends)
# ---------------------------------------------------------------------------
def locomo_sessions_for_mem0(sample: dict) -> list[tuple[str, int | None, list[dict]]]:
    """One entry per LOCOMO session: speaker-labeled messages + session date."""
    from data_loader import _locomo_sessions, _render_turn_text

    out: list[tuple[str, int | None, list[dict]]] = []
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Speaker A")
    for num, date_time, turns in _locomo_sessions(conv):
        messages = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({
                "role": role,
                "content": f"({date_time}) {speaker}: {_render_turn_text(turn)}",
            })
        if messages:
            out.append((f"session_{num}", _parse_locomo_dt(date_time), messages))
    return out


def lme_sessions_for_mem0(item: dict) -> list[tuple[str, int | None, list[dict]]]:
    """One entry per LME haystack session, tagged with the session date."""
    out: list[tuple[str, int | None, list[dict]]] = []
    for sess_id, date, turns in zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    ):
        messages = []
        for turn in turns:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = (turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": f"({date}) {content}"})
        if messages:
            out.append((str(sess_id), _parse_lme_date(date), messages))
    return out
