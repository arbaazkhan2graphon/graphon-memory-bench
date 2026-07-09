"""Aggregation and result writers.

Artifacts per run (under results/):
  rows_<benchmark>.jsonl        one row per (question x backend x mode), append-only
  summary_<benchmark>_<ts>.json aggregates by backend/mode/category + ledger
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("membench_metrics")

_APPEND_LOCK = threading.Lock()


def rows_path(results_dir: Path, benchmark: str) -> Path:
    return results_dir / f"rows_{benchmark}.jsonl"


def append_row(results_dir: Path, benchmark: str, row: dict) -> None:
    path = rows_path(results_dir, benchmark)
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(results_dir: Path, benchmark: str) -> list[dict]:
    path = rows_path(results_dir, benchmark)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def completed_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    """(qid, backend, mode) combos already evaluated without error."""
    return {(r["qid"], r["backend"], r["mode"]) for r in rows if not r.get("error")}


def _agg(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    covs = [r["coverage"] for r in rows]
    recalls = [r["recall_at_10"] for r in rows if r.get("recall_at_10") is not None]
    lat = sorted(r["retrieval_latency"] + r.get("reader_latency", 0.0) for r in rows)
    return {
        "n": len(rows),
        "qa_accuracy": round(sum(covs) / len(covs), 4),
        "recall_at_10": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_n": len(recalls),
        "strict_full_coverage": round(sum(1 for c in covs if c >= 0.999) / len(covs), 4),
        "avg_latency": round(sum(lat) / len(lat), 3),
        "p95_latency": round(lat[int(0.95 * (len(lat) - 1))], 3),
        "errors": sum(1 for r in rows if r.get("error")),
    }


def summarize(rows: list[dict], meta: dict, ledger_summary: dict) -> dict:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        by_system[f"{r['backend']}/{r['mode']}"].append(r)

    summary: dict[str, Any] = {"meta": meta, "ledger": ledger_summary, "systems": {}}
    for system, srows in sorted(by_system.items()):
        entry = _agg(srows)
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            by_cat[r["category"]].append(r)
        entry["by_category"] = {c: _agg(cr) for c, cr in sorted(by_cat.items())}
        summary["systems"][system] = entry
    return summary


def write_summary(results_dir: Path, benchmark: str, summary: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"summary_{benchmark}_{ts}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary written to %s", path)
    return path


def print_summary(summary: dict) -> None:
    print(f"\n===== {summary['meta'].get('benchmark')} summary =====")
    for system, entry in summary["systems"].items():
        rec = entry.get("recall_at_10")
        rec_s = f"{rec:.3f}" if rec is not None else "  -  "
        print(
            f"  {system:24s} n={entry['n']:4d}  QA(coverage)={entry['qa_accuracy']:.3f}"
            f"  recall@10={rec_s}  strict={entry['strict_full_coverage']:.3f}"
            f"  avg_lat={entry['avg_latency']:.2f}s  errors={entry['errors']}"
        )
    ledger = summary.get("ledger", {})
    print(f"  OpenAI spend: ${ledger.get('total_cost_usd', 0):.2f}")
