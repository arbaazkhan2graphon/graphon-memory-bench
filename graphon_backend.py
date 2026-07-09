"""Graphon backend: one group per corpus, direct answers + ranked sources.

Corpora:
  * LOCOMO      -- one group per conversation (single markdown document).
  * LongMemEval -- one group per question (one markdown file per haystack
    session, named ``<session_id>.md`` so sources map back to sessions).

Only rendered corpus text is ever uploaded; questions, gold answers, and
evidence labels never reach Graphon.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path

from schemas import BackendResult, RetrievedChunk

logger = logging.getLogger("membench_graphon")

_CITATION_RE = re.compile(r"\[\[SRC:\d+\]\]")


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "").replace("  ", " ").strip()


class GraphonBackend:
    name = "graphon"

    def __init__(self, cfg: dict) -> None:
        import os

        gcfg = cfg.get("graphon", {})
        self.api_key = os.environ.get("GRAPHON_API_KEY")
        if not self.api_key:
            raise ValueError("GRAPHON_API_KEY is not set (env or .env).")
        self.base_url = os.environ.get("GRAPHON_BASE_URL") or gcfg.get("base_url")
        self.group_name_prefix = gcfg.get("group_name_prefix", "membench")
        self.build_timeout = int(gcfg.get("build_timeout_seconds", 3600))
        self.reasoning_effort = gcfg.get("reasoning_effort", "standard")
        self.retrieval_k = int(gcfg.get("retrieval_k", 10))

        project_dir = Path(__file__).resolve().parent
        self.cache_file = project_dir / gcfg.get(
            "groups_cache_file", ".graphon_groups_cache.json"
        )
        self.docs_dir = project_dir / ".graphon_docs"
        self._groups: dict[str, str] = self._load_cache()
        self._verified: set[str] = set()
        self._cache_lock = threading.Lock()

        # The async SDK runs on a dedicated event-loop thread so multiple
        # harness workers can issue queries concurrently.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client = None
        self._client_lock = threading.Lock()

    # -- group cache ---------------------------------------------------------
    def _load_cache(self) -> dict[str, str]:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt group cache at %s; ignoring.", self.cache_file)
        return {}

    def _save_cache(self) -> None:
        with self._cache_lock:
            self.cache_file.write_text(
                json.dumps(self._groups, indent=2), encoding="utf-8"
            )

    # -- async plumbing ------------------------------------------------------
    def _ensure_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            from graphon_client import GraphonClient

            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, daemon=True)
            thread.start()
            self._loop = loop
            self._loop_thread = thread

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            try:
                client = GraphonClient(**kwargs)
            except TypeError:
                client = GraphonClient(api_key=self.api_key)
            asyncio.run_coroutine_threadsafe(client.__aenter__(), loop).result(120)
            self._client = client
            return client

    def _run(self, coro, timeout: float | None = None):
        self._ensure_client()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout or self.build_timeout)

    # -- setup ---------------------------------------------------------------
    def setup_corpus(
        self, corpus_key: str, file_paths: list[Path], force_reindex: bool = False
    ) -> tuple[str, float | None]:
        """Build (or reuse) the group for one corpus.

        Returns (group_id, build_seconds); build_seconds is None when a cached
        group was reused (no ingest happened).
        """
        cached = self._groups.get(corpus_key)
        if cached and not force_reindex:
            if corpus_key in self._verified or self._group_is_ready(cached):
                self._verified.add(corpus_key)
                logger.info("Reusing group for '%s': %s", corpus_key, cached)
                return cached, None
            logger.warning("Cached group %s for '%s' not ready; rebuilding.", cached, corpus_key)

        if cached and force_reindex:
            self._delete_group_safe(cached)
            self._groups.pop(corpus_key, None)
            self._save_cache()

        client = self._ensure_client()
        group_name = f"{self.group_name_prefix}-{corpus_key}".replace(":", "-")
        t0 = time.time()
        group_id = self._run(
            client.upload_process_and_create_group(
                group_name=group_name,
                file_paths=[str(p) for p in file_paths],
                wait_for_ready=True,
                timeout=self.build_timeout,
            )
        )
        build_seconds = time.time() - t0
        self._groups[corpus_key] = group_id
        self._verified.add(corpus_key)
        self._save_cache()
        logger.info(
            "Corpus '%s' (%d files) ready as group %s in %.1fs",
            corpus_key, len(file_paths), group_id, build_seconds,
        )
        return group_id, build_seconds

    def _group_is_ready(self, group_id: str) -> bool:
        try:
            client = self._ensure_client()
            detail = self._run(client.get_group_status(group_id))
            return getattr(detail, "status", None) == "SUCCESS"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not verify group %s: %s", group_id, exc)
            return False

    def _delete_group_safe(self, group_id: str) -> None:
        try:
            client = self._ensure_client()
            self._run(client.delete_group(group_id))
            logger.info("Deleted group %s", group_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete group %s: %s", group_id, exc)

    def delete_corpus(self, corpus_key: str) -> None:
        gid = self._groups.pop(corpus_key, None)
        if gid:
            self._delete_group_safe(gid)
            self._save_cache()

    # -- query ---------------------------------------------------------------
    def query(self, corpus_key: str, question: str) -> BackendResult:
        group_id = self._groups.get(corpus_key)
        if not group_id:
            return BackendResult(error=f"No group for corpus '{corpus_key}'.")
        t0 = time.time()
        try:
            client = self._ensure_client()
            resp = self._run(
                client.query_group(
                    group_id,
                    question,
                    return_source_data=True,
                    reasoning_effort=self.reasoning_effort,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            return BackendResult(
                latency_seconds=time.time() - t0, group_id=group_id, error=str(exc)
            )
        latency = time.time() - t0

        raw_answer = getattr(resp, "answer", "") or ""
        sources = getattr(resp, "sources", {}) or {}
        chunks = self._chunks_from_sources(sources, self.retrieval_k)
        return BackendResult(
            chunks=chunks,
            direct_answer=strip_citations(raw_answer),
            latency_seconds=latency,
            group_id=group_id,
        )

    @staticmethod
    def _chunks_from_sources(sources: dict, k: int) -> list[RetrievedChunk]:
        items: list[RetrievedChunk] = []
        for key, entry in sources.items():
            if not isinstance(entry, dict):
                continue
            score = entry.get("score")
            src = entry.get("source", entry)
            text, fname = "", ""
            if isinstance(src, dict):
                text = src.get("text") or src.get("transcript") or ""
                fname = src.get("file_name") or ""
            if not text:
                continue
            # LME session files are named "<session_id>.md".
            session_id = Path(fname).stem if fname else ""
            items.append(
                RetrievedChunk(
                    text=text,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    source_ref=str(key),
                    session_id=session_id,
                )
            )
        items.sort(key=lambda c: (c.score is not None, c.score or 0.0), reverse=True)
        return items[:k]

    # -- teardown -------------------------------------------------------------
    def teardown(self) -> None:
        if self._client is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.__aexit__(None, None, None), self._loop
                ).result(60)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Graphon client: %s", exc)
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread is not None:
                    self._loop_thread.join(timeout=10)
                self._client = None
                self._loop = None
                self._loop_thread = None
