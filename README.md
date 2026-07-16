# Graphon Memory Benchmark: LOCOMO + LongMemEval-S

Evaluates **Graphon** as conversational long-term memory on the two academic
benchmarks the memory-systems field reports on — **LOCOMO** (n=300) and
**LongMemEval-S** (n=50) — using a graphify-style harness:

```
ingest  ->  index  ->  search  ->  answer  ->  grade
(render)   (Graphon)  (top-10)   (direct /   (key-fact coverage,
                                  gpt-4o)     auditable quotes)
```

An in-harness **BM25 anchor baseline** runs over the identical corpus with the
same reader and judge, so absolute numbers can be cross-referenced against
published tables that include a BM25 row. **mem0's hosted platform** can also
be run live inside the harness as a competitor baseline.

## Headline results

| Dataset | Metric | Graphon | mem0 hosted (same harness) |
|---|---|---|---|
| LOCOMO (n=300) | QA accuracy (ultra mode) | **85.2%** | 54.6% |
| LOCOMO (n=300) | recall@10 (standard retrieval) | **0.751** | — |
| LongMemEval-S (n=50) | QA accuracy (ultra mode) | **79.5%** | — |
| Graph build | external LLM credits | **$0** | $0 (indexes on their platform) |

Full report: [`results/Graphon_Memory_Benchmark_Report.pdf`](results/Graphon_Memory_Benchmark_Report.pdf)
(run summaries with per-category breakdowns are in `results/summary_*.json`).

**Why our mem0 number (54.6%) differs from their self-reported 92.5%** — judge
leniency, retrieval depth (top-200 vs top-10), and per-benchmark prompt tuning —
is explained number by number in
[`results/Graphon_vs_mem0_Memory_Benchmark.pdf`](results/Graphon_vs_mem0_Memory_Benchmark.pdf)
(regenerate with `make_blog_report.py`).

## Systems evaluated

| System | What it is |
|---|---|
| `graphon/direct` | Graphon's own end-to-end answer (`query_group`) — the headline; how the product is used. |
| `graphon/shared_reader` | gpt-4o answers from Graphon's top-10 retrieved sources — structurally identical to "retriever + shared reader" harnesses. |
| `mem0/shared_reader` | mem0's **hosted platform** run live in this harness: identical conversations ingested via their API (one mem0 user per corpus), top-10 memory search + the same gpt-4o reader. Requires `MEM0_API_KEY`. |
| `bm25/shared_reader` | Okapi BM25 (in-harness, dependency-free) over the same rendered text + gpt-4o reader. |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # fill in GRAPHON_API_KEY + OPENAI_API_KEY

.venv/bin/python download_data.py            # LOCOMO + LongMemEval-S -> data/
.venv/bin/python run_eval.py --benchmark locomo        # n=300 (config)
.venv/bin/python run_eval.py --benchmark longmemeval   # n=50  (config)
.venv/bin/python make_customer_report.py               # PDF report
```

Useful flags: `--limit 5` (smoke), `--workers 6`, `--backends graphon`,
`--reasoning-effort ultra`, `--force-reindex`, `--max-spend 10`,
`--cleanup-groups` (delete LME groups after scoring).

Runs are **resumable**: rows append to `results/rows_<benchmark>.jsonl` and
completed (question x backend x mode) combos are skipped on rerun.

## Datasets (not redistributed; fetched by `download_data.py`)

- **LOCOMO** — `locomo10.json` from [snap-research/locomo](https://github.com/snap-research/locomo),
  pinned to the commit SHA recorded in `data/manifest.json`. 10 multi-session
  conversations, 1,986 QA pairs. Scoring uses the 4 standard categories
  (single-hop, multi-hop, temporal, open-domain); adversarial (category 5) is
  excluded, matching how the field reports. n=300 stratified sample, seed 42.
- **LongMemEval-S** — official cleaned release
  ([xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)).
  500 questions, each with its own ~115k-token haystack (~40-50 sessions).
  n=50 stratified by question type, seed 42, abstention questions included.

## Ingestion (what Graphon sees)

Graphon indexes documents, not raw JSON:

- LOCOMO: each conversation becomes **one markdown document** — session
  headers with dates, `Speaker: text` turns, image captions inlined as
  `[shares a photo: ...]`. One Graphon group per conversation (10 groups).
- LongMemEval: each haystack session becomes **its own markdown file** named
  `<session_id>.md`, so retrieved sources map back to sessions. One group per
  question (50 groups).

No scoring metadata (dialog ids, `has_answer` flags, gold session ids) is ever
written into the rendered text. Questions, gold answers, and evidence labels
never reach Graphon.

## Grading: key-fact coverage (auditable)

1. A cached pre-pass (`results/keyfacts_cache.json`) decomposes each gold
   answer into atomic key facts (usually 1-3; abstention questions get a
   fixed "states the information is not available" fact).
2. The judge marks each fact **covered / partial / missing** and must cite a
   **verbatim quote** from the candidate answer for every non-missing verdict.
   Verdicts + quotes are stored per row in the JSONL, so grades are auditable.
3. `coverage = (covered + 0.5 * partial) / total`. **QA accuracy = mean
   coverage** across questions (a strict all-facts-covered rate is also
   reported as `strict_full_coverage`).

One model (`gpt-4o`, config `llm.model`) fills every LLM role — reader,
decomposer, judge — mirroring the "one shared model" fairness rule.

## Retrieval recall@10

- **LOCOMO**: per question, the fraction of gold evidence turns present in the
  concatenation of the top-10 retrieved chunks. A turn counts as present when
  >= 60% of its word 5-grams appear (robust to chunk boundaries). Mean over
  questions. Vendors define recall@k slightly differently; this definition is
  fixed here and applied identically to every system.
- **LongMemEval-S**: session-level — the fraction of gold
  `answer_session_ids` among the sessions of the top-10 retrieved chunks.

## Cost accounting

- Every run writes a **spend ledger** (tokens + USD per LLM role) into the
  summary JSON and enforces `--max-spend` as a hard stop.
- **Graph build LLM credits from this harness: $0** — Graphon indexes
  server-side; no external LLM tokens are spent on ingest. Graphon group build
  times are recorded per corpus in the run metadata.

## Layout

```
memory_bench/
  README.md
  requirements.txt / .env.example / config.yaml
  download_data.py        # fetch datasets + manifest (pinned SHAs)
  data_loader.py          # parsing, markdown rendering, stratified sampling
  schemas.py              # dataclasses
  graphon_backend.py      # graphon-client SDK wrapper, group cache
  bm25_backend.py         # in-harness Okapi BM25 anchor
  judge.py                # shared reader, key-fact decomposition, coverage judge,
                          # recall@10 scorers, spend ledger
  metrics.py              # row writers, aggregation, summaries
  run_eval.py             # CLI orchestrator (parallel, resumable)
  make_customer_report.py # customer-facing PDF
  results/                # rows_*.jsonl, summary_*.json, report PDF
```
