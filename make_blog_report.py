"""Generate the shareable blog-style PDF: Graphon vs mem0, measured honestly.

Reads the rows files directly (results/rows_*.jsonl) so standard- and
ultra-mode Graphon runs can be reported side by side, together with the
in-harness mem0 result, and renders the "why our mem0 number differs from
their self-reported one" explainer with the public repro repository link.

Usage:
    python make_blog_report.py
    python make_blog_report.py --out results/Graphon_vs_mem0_Memory_Benchmark.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from make_customer_report import (
    ANCHOR_TEAL,
    FIELD_GRAY,
    GRAPHON_BLUE,
    GRAPHON_DARK,
    MEM0_ORANGE,
    _bar_chart,
    _styled_table,
)

logger = logging.getLogger("membench_blog")

PROJECT_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/arbaazkhan2graphon/graphon-memory-bench"

GRAPHON_LIGHT = colors.HexColor("#60a5fa")


def qa_by_system(benchmark: str) -> dict[tuple[str, str, str], float]:
    """(backend, mode, effort) -> mean coverage, from the audited rows file."""
    rows_file = PROJECT_DIR / "results" / f"rows_{benchmark}.jsonl"
    if not rows_file.exists():
        return {}
    # Deduplicate like metrics.summarize: last row wins per question/system,
    # so these means match the summary JSONs digit for digit.
    latest: dict[tuple[str, str, str, str], float] = {}
    for line in rows_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        eff = r.get("effort", "standard")
        latest[(r["qid"], r["backend"], r["mode"], eff)] = r["coverage"]
    stats: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (_qid, backend, mode, eff), cov in latest.items():
        stats[(backend, mode, eff)].append(cov)
    return {k: round(sum(v) / len(v), 4) for k, v in stats.items() if v}


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "\u2014"


def build_report(out_path: Path) -> Path:
    loco = qa_by_system("locomo")
    lme = qa_by_system("longmemeval")

    g_std_l = loco.get(("graphon", "direct", "standard"))
    g_ult_l = loco.get(("graphon", "direct", "ultra"))
    m0_l = loco.get(("mem0", "shared_reader", "-"))
    bm_l = loco.get(("bm25", "shared_reader", "standard"))
    g_std_m = lme.get(("graphon", "direct", "standard"))
    g_ult_m = lme.get(("graphon", "direct", "ultra"))
    m0_m = lme.get(("mem0", "shared_reader", "-"))
    bm_m = lme.get(("bm25", "shared_reader", "standard"))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=21,
                                 textColor=GRAPHON_DARK, spaceAfter=4, leading=25)
    subtitle = ParagraphStyle("S", parent=styles["Normal"], fontSize=12.5,
                              textColor=GRAPHON_BLUE, spaceAfter=2)
    meta_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=9,
                                textColor=FIELD_GRAY)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                        textColor=GRAPHON_DARK, spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10.5, leading=15)
    footnote = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
                              textColor=FIELD_GRAY, leading=11)
    mono = ParagraphStyle("Mono", parent=styles["Normal"], fontSize=9,
                          fontName="Courier", leading=13,
                          backColor=colors.HexColor("#f8fafc"),
                          borderPadding=6)

    flow: list = []

    # ---- header ------------------------------------------------------------
    flow.append(Paragraph("One Benchmark, Three Numbers:", title_style))
    flow.append(Paragraph(
        "What Conversational Memory Scores Actually Measure", title_style))
    flow.append(Paragraph(
        "Graphon vs mem0 on LOCOMO and LongMemEval-S, in one open harness",
        subtitle))
    flow.append(Paragraph(
        f"Graphon AI &nbsp;|&nbsp; {date.today().isoformat()} &nbsp;|&nbsp; "
        f"fully reproducible: <link href='{REPO_URL}'>{REPO_URL}</link>",
        meta_style))
    flow.append(Spacer(1, 12))

    # ---- the claim ----------------------------------------------------------
    flow.append(Paragraph(
        "mem0 reports <b>92.5%</b> on the LOCOMO benchmark. The graphify "
        "benchmark page reports mem0 at <b>27.3%</b> on the same dataset. When "
        "we ran mem0's hosted platform ourselves, we measured <b>54.6%</b>. "
        "All three numbers are real. They differ because a benchmark score is "
        "a property of the <i>evaluation harness</i> — the judge, the retrieval "
        "depth, the prompts — at least as much as of the memory system. This "
        "report publishes our full harness, applies the same strict rules to "
        "every system including our own, and links the code so you can "
        "reproduce every number below.",
        body))

    # ---- results ------------------------------------------------------------
    flow.append(Paragraph("Results", h2))
    flow.append(Paragraph(
        "QA accuracy is key-fact coverage: gold answers are decomposed into "
        "atomic facts; a gpt-4o judge must cite a verbatim quote from the "
        "candidate answer for every fact it credits (half credit for partials). "
        "One shared model answers and judges for every system. LOCOMO n=300, "
        "LongMemEval-S n=50, both stratified with a fixed seed.",
        body))
    flow.append(Spacer(1, 8))

    res_rows = [["System", "LOCOMO", "LongMemEval-S"]]
    res_rows.append(["Graphon — ultra mode (agentic answer harness)",
                     _pct(g_ult_l), _pct(g_ult_m)])
    res_rows.append(["Graphon — standard mode",
                     _pct(g_std_l), _pct(g_std_m)])
    res_rows.append(["mem0 hosted platform (run by us, same harness)*",
                     _pct(m0_l), _pct(m0_m)])
    res_rows.append(["BM25 anchor (in-harness sanity baseline)",
                     _pct(bm_l), _pct(bm_m)])
    flow.append(_styled_table(
        res_rows, [3.4 * inch, 1.5 * inch, 1.5 * inch], highlight_row=1))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "* mem0 on LongMemEval-S is pending: it requires ~2,250 session ingests "
        "through their platform API. The harness supports it (--backends mem0); "
        "anyone can run it from the repository.",
        footnote))
    flow.append(Spacer(1, 8))

    pairs = [
        ("Graphon\n(ultra)", (g_ult_l or 0) * 100),
        ("Graphon\n(standard)", (g_std_l or 0) * 100),
        ("mem0 hosted\n(our harness)", (m0_l or 0) * 100),
        ("BM25 anchor\n(our harness)", (bm_l or 0) * 100),
        ("mem0\n(self-reported)", 92.5),
        ("mem0\n(graphify harness)", 27.3),
    ]
    bar_colors = [GRAPHON_BLUE, GRAPHON_LIGHT, MEM0_ORANGE, ANCHOR_TEAL,
                  FIELD_GRAY, FIELD_GRAY]
    flow.append(_bar_chart(pairs, bar_colors))
    flow.append(Paragraph(
        "LOCOMO QA accuracy. Colored bars were measured in our open harness "
        "under identical grading; gray bars are external numbers for the same "
        "mem0 system under two other harnesses \u2014 the spread is the point.",
        ParagraphStyle("C", parent=styles["Normal"], fontSize=9,
                       textColor=FIELD_GRAY, spaceBefore=2)))

    # ---- why the numbers differ ---------------------------------------------
    flow.append(PageBreak())
    flow.append(Paragraph("Why doesn't mem0 show 92.5% here?", h2))
    flow.append(Paragraph(
        "The same mem0 product carries three public LOCOMO numbers. Nothing "
        "about the memory system changes between them \u2014 only the way answers "
        "are graded and how much retrieved context the answering model gets:",
        body))
    flow.append(Spacer(1, 8))

    diff_rows = [
        ["", "mem0's harness", "This harness", "graphify's harness"],
        ["Reported score", "92.5%", "54.6%", "27.3%"],
        ["Who ran it", "mem0", "Graphon (code public)", "graphify"],
        ["Judge", "Binary CORRECT/WRONG,\ninstructed to be lenient\n('same topic' can pass)",
         "Per-fact verdicts; verbatim\nquote required for credit;\npartial = half credit",
         "Key-fact coverage\n(Kimi K2.6 judge)"],
        ["Retrieval depth", "Up to top-200 memories", "Top-10", "Top-10"],
        ["Answer prompts", "Tuned per benchmark", "One neutral prompt,\nidentical for all systems",
         "One shared reader"],
    ]
    flow.append(_styled_table(
        diff_rows, [1.15 * inch, 1.85 * inch, 1.85 * inch, 1.75 * inch]))
    flow.append(Spacer(1, 10))

    explain = [
        "<b>Judge leniency is the biggest lever.</b> The 'LLM-as-a-judge' score "
        "mem0 headlines is not part of the original LOCOMO benchmark (which used "
        "lexical F1/ROUGE); it was introduced by mem0's own paper. Their judge "
        "prompt instructs the model to be generous and accept answers on the "
        "same topic as the gold answer. An independent audit (Penfield Labs, "
        "April 2026) fed that judge deliberately wrong but topically adjacent "
        "answers for all 1,540 LOCOMO questions: it accepted ~62% of them as "
        "correct. Our judge cannot do that \u2014 it must quote the exact words in "
        "the answer that establish each gold fact, and every verdict is stored "
        "in the run artifacts for inspection.",
        "<b>Retrieval depth changes the task.</b> mem0's headline number gives "
        "the answering model up to 200 retrieved memories per question; at "
        "top-10 \u2014 the depth the field's comparative tables use \u2014 the task is "
        "much harder. We ran every system, including our own, at top-10.",
        "<b>Prompt engineering inflates self-reported numbers.</b> A third-party "
        "reproduction (maximem, 2026) ran mem0's hosted product on LongMemEval "
        "through a standardized harness and measured 73.8% against mem0's "
        "published 93.4% on the same data \u2014 attributing the gap to "
        "benchmark-specific answer and judge prompts, not the memory layer. Our "
        "harness uses one neutral reader prompt for every system.",
        "<b>The same strict rules cut against us too.</b> Under lenient "
        "same-topic grading our numbers would also be far higher. We publish "
        "the strict numbers because they are the ones that predict real product "
        "behavior: a memory system that returns the right topic but the wrong "
        "fact is wrong.",
        "<b>The BM25 anchor keeps everyone honest.</b> A plain BM25 baseline "
        "runs inside the harness over the identical corpus. If you re-run this "
        "benchmark with a different judge model and get different absolute "
        "numbers, the anchor moves with them \u2014 compare system-minus-anchor "
        "deltas across harnesses.",
    ]
    for p in explain:
        flow.append(Paragraph(p, body))
        flow.append(Spacer(1, 5))

    # ---- how mem0 was run ----------------------------------------------------
    flow.append(Paragraph("How we ran mem0 (fairness checklist)", h2))
    fairness = [
        "Their production hosted platform via their official SDK \u2014 not a "
        "re-implementation. One mem0 user per conversation.",
        "Identical source conversations, session dates preserved (passed as "
        "timestamps and inline), mirroring mem0's own LOCOMO evaluation format.",
        "Ingestion allowed to fully settle before any question was asked: we "
        "polled their memory counts until stable (their extraction runs "
        "asynchronously and lags add-calls by minutes) \u2014 2,439 extracted "
        "memories across the 10 conversations.",
        "Top-10 memory search per question; the same gpt-4o reader answered "
        "from mem0's memories and from Graphon's sources with the same prompt; "
        "the same judge graded both.",
        "Retrieval-recall metrics are not reported for mem0: their memories are "
        "extracted facts rather than verbatim transcript, so our n-gram turn "
        "recall would under-credit them. QA accuracy is the comparable column.",
    ]
    for i, item in enumerate(fairness, 1):
        flow.append(Paragraph(f"{i}. {item}", body))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        "Graphon numbers: 'standard' is the default query mode; 'ultra' is "
        "Graphon's agentic answer harness, which reasons over the memory graph "
        "before answering. Both are graded identically to mem0.",
        body))

    # ---- reproduce ------------------------------------------------------------
    flow.append(Paragraph("Reproduce every number", h2))
    flow.append(Paragraph(
        f"The complete harness \u2014 dataset downloaders (pinned SHAs), ingestion, "
        f"backends, judge prompts, and per-question artifacts \u2014 is public:",
        body))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(f"<link href='{REPO_URL}'><b>{REPO_URL}</b></link>", body))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        "git clone " + REPO_URL + ".git<br/>"
        "python3 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt<br/>"
        "cp .env.example .env   # add GRAPHON_API_KEY, OPENAI_API_KEY (+ MEM0_API_KEY)<br/>"
        ".venv/bin/python download_data.py<br/>"
        ".venv/bin/python run_eval.py --benchmark locomo --backends graphon,bm25,mem0<br/>"
        ".venv/bin/python run_eval.py --benchmark locomo --backends graphon "
        "--graphon-modes direct --reasoning-effort ultra --no-resume<br/>"
        ".venv/bin/python run_eval.py --benchmark longmemeval --backends graphon,bm25",
        mono))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        "Every run writes one JSONL row per question with the answer, the "
        "per-fact verdicts, and the verbatim quotes the judge relied on \u2014 the "
        "grading is auditable line by line, and the spend ledger caps OpenAI "
        "cost per run. Sampling is seeded, so you will evaluate the exact "
        "question set behind this report.",
        body))
    flow.append(Spacer(1, 10))
    flow.append(Paragraph(
        "Sources: mem0 92.5% \u2014 mem0.ai research page and mem0ai/memory-benchmarks "
        "(top_200, April 2026). mem0 27.3% \u2014 graphify benchmark page (2026-07-05). "
        "Judge-leniency audit \u2014 Penfield Labs (dev.to, April 2026). Claimed-vs-"
        "observed reproduction \u2014 maximem.ai, 'The state of AI memory in 2026'. "
        "LOCOMO \u2014 Maharana et al., ACL 2024 (snap-research/locomo). "
        "LongMemEval-S \u2014 xiaowu0162/longmemeval-cleaned. Graphon LOCOMO standard "
        "n=299 of 300 (one question lost to a transient API error in the archived "
        "standard run; the ultra run covers all 300).",
        footnote))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        title="One Benchmark, Three Numbers \u2014 Graphon vs mem0",
        author="Graphon AI",
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    doc.build(flow)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the blog-style PDF.")
    parser.add_argument(
        "--out",
        default=str(PROJECT_DIR / "results" / "Graphon_vs_mem0_Memory_Benchmark.pdf"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_report(out_path)
    logger.info("Wrote report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
