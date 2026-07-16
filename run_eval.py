"""CLI entrypoint: run LOCOMO / LongMemEval-S against Graphon (+ BM25 anchor).

Pipeline per question (graphify-style): ingest -> index -> search -> answer -> grade.

Systems evaluated:
  graphon/direct         Graphon's own end-to-end answer (headline; how the
                         product is used).
  graphon/shared_reader  gpt-4o answers from Graphon's top-10 sources
                         (structurally identical to how the graphify harness
                         ran every system: retriever + shared reader).
  bm25/shared_reader     in-harness BM25 anchor over the identical corpus.

Examples:
    python run_eval.py --benchmark locomo                  # n from config (300)
    python run_eval.py --benchmark locomo --limit 5        # smoke test
    python run_eval.py --benchmark longmemeval             # n from config (50)
    python run_eval.py --benchmark longmemeval --limit 2 --backends graphon
    python run_eval.py --benchmark locomo --force-reindex
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import metrics
from bm25_backend import BM25Backend, lme_chunks, locomo_chunks
from mem0_backend import Mem0Backend, lme_sessions_for_mem0, locomo_sessions_for_mem0
from data_loader import (
    PROJECT_DIR,
    lme_questions,
    lme_session_files,
    load_config,
    load_locomo,
    load_longmemeval,
    locomo_questions,
    render_locomo_conversation,
    stratified_sample,
)
from dotenv import load_dotenv
from graphon_backend import GraphonBackend
from judge import (
    KeyFactStore,
    Ledger,
    LLMClient,
    SpendLimitExceeded,
    grade_answer,
    lme_recall_at_k,
    locomo_recall_at_k,
    read_answer,
)
from schemas import Question, RowRecord

logger = logging.getLogger("membench")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["locomo", "longmemeval"], required=True)
    p.add_argument("--backends", default="graphon,bm25",
                   help="comma list: graphon,bm25,mem0")
    p.add_argument("--graphon-modes", default="direct,shared_reader",
                   help="comma list: direct,shared_reader")
    p.add_argument("--n", type=int, default=None, help="sample size (default from config)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap questions after sampling (smoke tests)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="re-evaluate rows already present in the rows file")
    p.add_argument("--max-spend", type=float, default=None, help="OpenAI USD cap")
    p.add_argument("--reasoning-effort", choices=["standard", "ultra"], default=None)
    p.add_argument("--cleanup-groups", action="store_true",
                   help="delete Graphon groups after a question is fully scored (LME)")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent question workers (default 4)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------
def prepare_locomo(cfg: dict, questions: list[Question], backends: dict,
                   force_reindex: bool, run_meta: dict) -> None:
    samples = {s["sample_id"]: s for s in load_locomo(cfg)}
    needed = sorted({q.corpus_id for q in questions})
    docs_dir = PROJECT_DIR / ".graphon_docs" / "locomo"
    docs_dir.mkdir(parents=True, exist_ok=True)
    builds = run_meta.setdefault("graphon_builds", {})
    for sid in needed:
        sample = samples[sid]
        if "graphon" in backends:
            doc_path = docs_dir / f"{sid}.md"
            doc_path.write_text(render_locomo_conversation(sample), encoding="utf-8")
            _gid, build_s = backends["graphon"].setup_corpus(
                f"locomo:{sid}", [doc_path], force_reindex=force_reindex
            )
            if build_s is not None:
                builds[f"locomo:{sid}"] = round(build_s, 1)
        if "bm25" in backends:
            backends["bm25"].setup_corpus(f"locomo:{sid}", locomo_chunks(sample))
        if "mem0" in backends:
            secs = backends["mem0"].ingest_sessions(
                f"locomo:{sid}", locomo_sessions_for_mem0(sample)
            )
            if secs is not None:
                run_meta.setdefault("mem0_ingests", {})[f"locomo:{sid}"] = round(secs, 1)


def prepare_lme_corpus(cfg: dict, item: dict, backends: dict,
                       force_reindex: bool, run_meta: dict) -> None:
    qid = item["question_id"]
    if "graphon" in backends:
        sess_dir = PROJECT_DIR / ".graphon_docs" / "lme" / qid
        paths = lme_session_files(item, sess_dir)
        _gid, build_s = backends["graphon"].setup_corpus(
            f"lme:{qid}", paths, force_reindex=force_reindex
        )
        if build_s is not None:
            run_meta.setdefault("graphon_builds", {})[f"lme:{qid}"] = round(build_s, 1)
    if "bm25" in backends:
        backends["bm25"].setup_corpus(f"lme:{qid}", lme_chunks(item))
    if "mem0" in backends:
        secs = backends["mem0"].ingest_sessions(f"lme:{qid}", lme_sessions_for_mem0(item))
        if secs is not None:
            run_meta.setdefault("mem0_ingests", {})[f"lme:{qid}"] = round(secs, 1)


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------
def evaluate_question(
    q: Question,
    corpus_key: str,
    backends: dict,
    graphon_modes: list[str],
    llm: LLMClient,
    facts_store: KeyFactStore,
    cfg: dict,
    done: set[tuple[str, str, str]],
    results_dir: Path,
) -> int:
    """Runs all missing (backend, mode) combos for one question; returns #rows."""
    scfg = cfg.get("scoring", {})
    ngram = int(scfg.get("evidence_ngram", 5))
    threshold = float(scfg.get("evidence_threshold", 0.6))
    graphon_effort = cfg.get("graphon", {}).get("reasoning_effort", "standard")
    written = 0

    plan: list[tuple[str, str]] = []
    if "graphon" in backends:
        plan.extend(("graphon", m) for m in graphon_modes)
    for other in ("bm25", "mem0"):
        if other in backends:
            plan.append((other, "shared_reader"))
    plan = [(b, m) for b, m in plan if (q.qid, b, m) not in done]
    if not plan:
        return 0

    facts = facts_store.get_or_build(llm, q)

    # One retrieval per backend, shared across its modes.
    results: dict[str, object] = {}
    for backend_name in {b for b, _ in plan}:
        results[backend_name] = backends[backend_name].query(corpus_key, q.question)

    for backend_name, mode in plan:
        res = results[backend_name]
        effort = graphon_effort if backend_name == "graphon" else "-"
        if res.error:
            row = RowRecord(
                qid=q.qid, benchmark=q.benchmark, backend=backend_name, mode=mode,
                effort=effort,
                category=q.category, question=q.question, gold_answer=q.gold_answer,
                answer="", coverage=0.0, recall_at_10=None, verdicts=[],
                retrieval_latency=res.latency_seconds, reader_latency=0.0,
                input_tokens=0, output_tokens=0, error=res.error,
            )
            metrics.append_row(results_dir, q.benchmark, row.to_json())
            written += 1
            continue

        # Ultra is an agentic answer harness and may return no source list;
        # recall@10 is only defined when a retrieval surface exists.
        if not res.chunks:
            recall = None
        elif q.benchmark == "locomo":
            recall = locomo_recall_at_k(q, res.chunks, ngram=ngram, threshold=threshold)
        else:
            recall = lme_recall_at_k(q, res.chunks)

        reader_latency, it, ot = 0.0, 0, 0
        if mode == "direct":
            answer = res.direct_answer
        else:
            answer, it, ot, reader_latency = read_answer(llm, q, res.chunks)

        grade = grade_answer(llm, q, facts, answer)
        row = RowRecord(
            qid=q.qid, benchmark=q.benchmark, backend=backend_name, mode=mode,
            effort=effort,
            category=q.category, question=q.question, gold_answer=q.gold_answer,
            answer=answer, coverage=grade.coverage, recall_at_10=recall,
            verdicts=[{"fact": v.fact, "verdict": v.verdict, "quote": v.quote}
                      for v in grade.verdicts],
            retrieval_latency=round(res.latency_seconds, 3),
            reader_latency=round(reader_latency, 3),
            input_tokens=it, output_tokens=ot,
            error=grade.judge_error,
        )
        metrics.append_row(results_dir, q.benchmark, row.to_json())
        done.add((q.qid, backend_name, mode))
        written += 1
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv(PROJECT_DIR / ".env")

    cfg = load_config()
    if args.reasoning_effort:
        cfg.setdefault("graphon", {})["reasoning_effort"] = args.reasoning_effort
    seed = args.seed if args.seed is not None else int(cfg["sampling"]["seed"])
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    lcfg = cfg.get("llm", {})
    max_spend = args.max_spend if args.max_spend is not None else float(
        cfg.get("run", {}).get("max_spend_usd", 25.0)
    )
    ledger = Ledger(float(lcfg.get("input_cost_per_m", 2.5)),
                    float(lcfg.get("output_cost_per_m", 10.0)), max_spend)
    llm = LLMClient(cfg, ledger)
    facts_store = KeyFactStore(PROJECT_DIR / cfg["scoring"]["keyfacts_cache"])

    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]
    graphon_modes = [m.strip() for m in args.graphon_modes.split(",") if m.strip()]
    backends: dict = {}
    if "graphon" in backend_names:
        backends["graphon"] = GraphonBackend(cfg)
    if "bm25" in backend_names:
        backends["bm25"] = BM25Backend(cfg)
    if "mem0" in backend_names:
        backends["mem0"] = Mem0Backend(cfg)

    # ----- questions ------------------------------------------------------
    if args.benchmark == "locomo":
        all_q = locomo_questions(load_locomo(cfg))
        n = args.n or int(cfg["sampling"]["locomo_n"])
    else:
        lme_items = load_longmemeval(cfg)
        all_q = lme_questions(lme_items)
        n = args.n or int(cfg["sampling"]["longmemeval_n"])
    questions = stratified_sample(all_q, n, seed)
    if args.limit:
        questions = questions[: args.limit]
    logger.info("Evaluating %d %s questions", len(questions), args.benchmark)

    rows = metrics.load_rows(results_dir, args.benchmark)
    done = set() if args.no_resume else metrics.completed_keys(rows)

    run_meta = {
        "benchmark": args.benchmark,
        "n_questions": len(questions),
        "seed": seed,
        "backends": backend_names,
        "graphon_modes": graphon_modes,
        "reasoning_effort": cfg.get("graphon", {}).get("reasoning_effort", "standard"),
        "llm_model": llm.model,
        "retrieval_k": int(cfg.get("graphon", {}).get("retrieval_k", 10)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    t_start = time.time()
    total_rows = 0
    stop = threading.Event()
    progress_lock = threading.Lock()

    def _eval_locomo(q: Question) -> int:
        if stop.is_set():
            return 0
        return evaluate_question(
            q, f"locomo:{q.corpus_id}", backends, graphon_modes,
            llm, facts_store, cfg, done, results_dir,
        )

    def _eval_lme(q: Question, item: dict) -> int:
        if stop.is_set():
            return 0
        needed = [
            (b, m) for b, m in
            ([("graphon", m) for m in graphon_modes] if "graphon" in backends else [])
            + [(o, "shared_reader") for o in ("bm25", "mem0") if o in backends]
            if (q.qid, b, m) not in done
        ]
        if not needed:
            return 0
        prepare_lme_corpus(cfg, item, backends, args.force_reindex, run_meta)
        n_rows = evaluate_question(
            q, f"lme:{q.qid}", backends, graphon_modes,
            llm, facts_store, cfg, done, results_dir,
        )
        if args.cleanup_groups and "graphon" in backends:
            backends["graphon"].delete_corpus(f"lme:{q.qid}")
        return n_rows

    try:
        if args.benchmark == "locomo":
            prepare_locomo(cfg, questions, backends, args.force_reindex, run_meta)
            tasks = [(q, None) for q in questions]
            worker = lambda q, _item: _eval_locomo(q)  # noqa: E731
        else:
            items_by_id = {it["question_id"]: it for it in lme_items}
            tasks = [(q, items_by_id[q.qid]) for q in questions]
            worker = lambda q, item: _eval_lme(q, item)  # noqa: E731

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(worker, q, item): q for q, item in tasks}
            finished = 0
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    n_rows = fut.result()
                except SpendLimitExceeded as exc:
                    logger.error("STOPPING: %s", exc)
                    stop.set()
                    continue
                except Exception as exc:
                    logger.error("Question %s failed: %s", q.qid, exc)
                    n_rows = 0
                with progress_lock:
                    total_rows += n_rows
                    finished += 1
                    if finished % 10 == 0 or finished == len(tasks):
                        logger.info("[%d/%d] rows+=%d spend=$%.2f",
                                    finished, len(tasks), total_rows, ledger.total_cost())
    except KeyboardInterrupt:
        logger.warning("Interrupted; partial rows are saved and resumable.")
    finally:
        if "graphon" in backends:
            backends["graphon"].teardown()

    run_meta["wall_seconds"] = round(time.time() - t_start, 1)
    run_meta["new_rows"] = total_rows

    # Summarize over the sampled question set only (rows file may contain more).
    sampled_ids = {q.qid for q in questions}
    all_rows = [r for r in metrics.load_rows(results_dir, args.benchmark)
                if r["qid"] in sampled_ids]
    # Deduplicate: keep the last row per (qid, backend, mode).
    dedup: dict[tuple[str, str, str], dict] = {}
    for r in all_rows:
        dedup[(r["qid"], r["backend"], r["mode"])] = r
    summary = metrics.summarize(list(dedup.values()), run_meta, ledger.summary())
    metrics.write_summary(results_dir, args.benchmark, summary)
    metrics.print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
