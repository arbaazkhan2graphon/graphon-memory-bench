"""Generate the customer-facing PDF report from LOCOMO + LongMemEval-S runs.

Reads the latest ``summary_locomo_*.json`` and ``summary_longmemeval_*.json``
under results/ and renders a branded PDF comparing Graphon against published
vendor numbers (graphify, supermemory, mem0) and the in-harness BM25 anchor.

Usage:
    python make_customer_report.py
    python make_customer_report.py --out results/Graphon_Memory_Benchmark_Report.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import date
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("membench_report")

PROJECT_DIR = Path(__file__).resolve().parent

GRAPHON_BLUE = colors.HexColor("#2563eb")
GRAPHON_DARK = colors.HexColor("#1e293b")
FIELD_GRAY = colors.HexColor("#94a3b8")
ANCHOR_TEAL = colors.HexColor("#0d9488")
MEM0_ORANGE = colors.HexColor("#f59e0b")
LIGHT_ROW = colors.HexColor("#f1f5f9")
GRID = colors.HexColor("#e2e8f0")

# Published vendor numbers from the graphify benchmark page (2026-07-05):
# same academic datasets, their own harness (Kimi K2.6 reader/judge, key-fact
# coverage grading). Cross-referenceable, not identical conditions.
PUBLISHED_LOCOMO = [
    # (system, qa_pct, recall_at_10, ingest_cost_str)
    ("graphify (graph-expand)", 45.3, 0.497, "~$1.40"),
    ("supermemory", 49.7, 0.149, "$15.67"),
    ("BM25 (their harness)", 31.3, 0.362, "$0"),
    ("mem0", 27.3, 0.048, "$3.48"),
]
PUBLISHED_LME = [
    ("graphify (graph-expand)", 76.0, 0.844),
    ("dense RAG (their harness)", 76.0, 0.848),
    ("BM25 (their harness)", 70.0, 0.710),
    ("mem0", 70.0, 0.344),
]

SYSTEM_LABELS = {
    "graphon/direct": "Graphon (direct answer)",
    "graphon/shared_reader": "Graphon + shared reader",
    "bm25/shared_reader": "BM25 anchor + shared reader",
    "mem0/shared_reader": "mem0 (hosted) + shared reader",
}

LOCOMO_CAT_ORDER = ["single-hop", "multi-hop", "temporal", "open-domain"]


def _latest_summary(benchmark: str) -> dict | None:
    paths = sorted(glob.glob(str(PROJECT_DIR / "results" / f"summary_{benchmark}_*.json")))
    if not paths:
        return None
    return json.loads(Path(paths[-1]).read_text(encoding="utf-8"))


def _total_ledger_cost(benchmark: str) -> float:
    """Cumulative OpenAI spend across every run process for a benchmark.

    Each process writes its own per-process ledger into its summary, so the
    sum over all summaries is the true total spent (resumed runs only pay for
    new rows; forced re-runs genuinely spent again).
    """
    total = 0.0
    for p in glob.glob(str(PROJECT_DIR / "results" / f"summary_{benchmark}_*.json")):
        led = json.loads(Path(p).read_text(encoding="utf-8")).get("ledger", {})
        total += float(led.get("total_cost_usd", 0.0))
    return total


def _lme_cost_estimate(loco: dict) -> float | None:
    """Estimate total LME OpenAI spend from per-row reader tokens.

    The LME run may span several resumed processes, so its final ledger only
    covers the last one. Reader tokens are exact (stored per row); judge and
    decomposer costs are estimated from the LOCOMO run's per-call averages.
    """
    rows_file = PROJECT_DIR / "results" / "rows_longmemeval.jsonl"
    if not rows_file.exists():
        return None
    rows = [json.loads(l) for l in rows_file.read_text(encoding="utf-8").splitlines() if l]
    reader_in = sum(r["input_tokens"] for r in rows)
    reader_out = sum(r["output_tokens"] for r in rows)
    led = loco.get("ledger", {})
    roles = led.get("roles", {})
    judge = roles.get("judge", {})
    per_call_in = judge.get("input_tokens", 0) / max(judge.get("calls", 1), 1)
    per_call_out = judge.get("output_tokens", 0) / max(judge.get("calls", 1), 1)
    pricing = led.get("pricing_per_m", {"input": 2.5, "output": 10.0})
    est_in = reader_in + per_call_in * len(rows)
    est_out = reader_out + per_call_out * len(rows)
    return est_in / 1e6 * pricing["input"] + est_out / 1e6 * pricing["output"]


def _pct(x: float) -> float:
    return round(x * 100, 1)


def _bar_chart(
    data_pairs: list[tuple[str, float]],
    bar_colors: list,
    width: float = 460,
    height: float = 220,
    value_max: float = 100,
    fmt: str = "{:.1f}%",
) -> Drawing:
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x, chart.y = 30, 50
    chart.width, chart.height = width - 60, height - 80
    chart.data = [[v for _, v in data_pairs]]
    chart.barWidth = 10
    chart.groupSpacing = 14
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = value_max
    chart.valueAxis.valueStep = value_max / 5
    chart.categoryAxis.labels.boxAnchor = "n"
    chart.categoryAxis.labels.dy = -6
    chart.categoryAxis.labels.fontSize = 7.5
    chart.categoryAxis.categoryNames = [name for name, _ in data_pairs]
    for i, col in enumerate(bar_colors):
        chart.bars[(0, i)].fillColor = col
    d.add(chart)
    n = len(data_pairs)
    step = chart.width / n
    for i, (_, v) in enumerate(data_pairs):
        x = chart.x + step * (i + 0.5)
        y = chart.y + (v / value_max) * chart.height + 4
        lbl = String(x, y, fmt.format(v), fontSize=8, textAnchor="middle")
        lbl.fillColor = GRAPHON_DARK
        d.add(lbl)
    return d


def _styled_table(rows: list[list[str]], col_widths: list[float],
                  header_bg=GRAPHON_DARK, highlight_row: int | None = None) -> Table:
    t = Table(rows, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row),
                      colors.HexColor("#dbeafe")))
        style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row),
                      "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def build_report(out_path: Path) -> Path:
    loco = _latest_summary("locomo")
    lme = _latest_summary("longmemeval")
    if not loco:
        raise FileNotFoundError("No LOCOMO summary in results/; run run_eval.py first.")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=23,
                                 textColor=GRAPHON_DARK, spaceAfter=4)
    subtitle = ParagraphStyle("S", parent=styles["Normal"], fontSize=13,
                              textColor=GRAPHON_BLUE, spaceAfter=2)
    meta_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=9,
                                textColor=FIELD_GRAY)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14,
                        textColor=GRAPHON_DARK, spaceBefore=16, spaceAfter=6)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10.5, leading=15)
    footnote = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
                              textColor=FIELD_GRAY, leading=11)
    caption = ParagraphStyle("C", parent=styles["Normal"], fontSize=9,
                             textColor=FIELD_GRAY, alignment=TA_CENTER, spaceBefore=2)

    l_sys = loco["systems"]
    l_meta = loco["meta"]
    g_direct = l_sys.get("graphon/direct", {})
    g_reader = l_sys.get("graphon/shared_reader", {})
    bm25 = l_sys.get("bm25/shared_reader", {})
    l_mem0 = l_sys.get("mem0/shared_reader", {})

    flow: list = []

    # ---- header ----------------------------------------------------------
    flow.append(Paragraph("Graphon Conversational Memory Benchmark", title_style))
    flow.append(Paragraph("LOCOMO and LongMemEval-S Evaluation Report", subtitle))
    n_lme = lme["meta"]["n_questions"] if lme else 0
    flow.append(Paragraph(
        f"Prepared by Graphon AI &nbsp;|&nbsp; {date.today().isoformat()} &nbsp;|&nbsp; "
        f"LOCOMO n={l_meta['n_questions']}, LongMemEval-S n={n_lme}",
        meta_style,
    ))
    flow.append(Spacer(1, 12))

    # ---- executive summary -----------------------------------------------
    flow.append(Paragraph("Executive Summary", h2))
    lme_direct = lme["systems"].get("graphon/direct", {}) if lme else {}
    # Effort comes from the rows behind each system entry, not run meta: the
    # newest summary may have been written by a process that only ran other
    # backends (e.g. a mem0-only run) with a different --reasoning-effort.
    loco_effort = g_direct.get("effort") or l_meta.get("reasoning_effort", "standard")
    loco_ultra = loco_effort == "ultra"
    # Ultra is an agentic answer harness without a fixed top-10 retrieval, so
    # recall@10 always comes from the standard-mode retrieval surface (the
    # shared_reader rows share the same retrieval as standard direct).
    loco_recall = (g_reader if loco_ultra else g_direct).get("recall_at_10", 0) or 0
    loco_note = (" using Graphon's <b>ultra</b> reasoning mode (an agentic answer "
                 "harness)" if loco_ultra else "")
    exec_bits = [
        f"Graphon was evaluated on the two academic benchmarks the memory-systems "
        f"field reports on: <b>LOCOMO</b> (n={l_meta['n_questions']}, multi-session "
        f"conversational QA) and <b>LongMemEval-S</b> (n={n_lme}, ~115k-token chat "
        f"histories per question). Answers were graded by key-fact coverage with "
        f"auditable verbatim-quote verdicts.",
        f"On LOCOMO, Graphon answered directly at <b>{_pct(g_direct.get('qa_accuracy', 0))}% "
        f"QA accuracy</b>{loco_note}, with <b>retrieval recall@10 of "
        f"{loco_recall:.3f}</b>"
        + (" (measured on Graphon's standard retrieval)." if loco_ultra else "."),
    ]
    lme_effort = (lme_direct.get("effort")
                  or (lme["meta"].get("reasoning_effort", "standard") if lme else "standard"))
    if lme_direct:
        effort_note = (" using Graphon's <b>ultra</b> reasoning mode (an agentic "
                       "answer harness)" if lme_effort == "ultra" else "")
        exec_bits.append(
            f"On LongMemEval-S, Graphon answered directly at "
            f"<b>{_pct(lme_direct.get('qa_accuracy', 0))}% QA accuracy</b>{effort_note}."
        )
    if l_mem0:
        exec_bits.append(
            f"As a live competitor baseline, <b>mem0's hosted platform</b> was run "
            f"inside this same harness — identical conversations ingested through "
            f"their API, top-10 retrieved memories, same shared reader and judge — "
            f"scoring <b>{_pct(l_mem0.get('qa_accuracy', 0))}% on LOCOMO</b>."
        )
    exec_bits.append(
        "Graphon builds its memory index server-side: this harness spent "
        "<b>zero external LLM credits on ingestion</b>."
    )
    for b in exec_bits:
        flow.append(Paragraph(b, body))
        flow.append(Spacer(1, 3))

    # ---- results at a glance ----------------------------------------------
    flow.append(Paragraph("Results at a Glance", h2))
    glance = [["Suite", "Dataset (n)", "Metric", "Graphon", "Published field*"]]
    glance.append([
        "Memory", f"LOCOMO ({l_meta['n_questions']})",
        "QA accuracy (ultra)" if loco_ultra else "QA accuracy",
        f"{_pct(g_direct.get('qa_accuracy', 0))}%",
        "graphify 45.3%, supermemory 49.7%,\nBM25 31.3%, mem0 27.3%",
    ])
    glance.append([
        "Memory", f"LOCOMO ({l_meta['n_questions']})", "recall@10",
        f"{loco_recall:.3f}",
        "graphify 0.497, BM25 0.362,\nsupermemory 0.149, mem0 0.048",
    ])
    if lme_direct:
        qa_label = "QA accuracy (ultra)" if lme_effort == "ultra" else "QA accuracy"
        glance.append([
            "Memory", f"LongMemEval-S ({n_lme})", qa_label,
            f"{_pct(lme_direct.get('qa_accuracy', 0))}%",
            "graphify 76%, dense RAG 76%,\nBM25 70%, mem0 70%",
        ])
    glance.append([
        "Cost", "graph build", "external LLM credits", "$0", "graphify $0",
    ])
    flow.append(_styled_table(
        glance, [0.7 * inch, 1.35 * inch, 1.25 * inch, 0.95 * inch, 2.55 * inch],
        header_bg=GRAPHON_BLUE,
    ))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "* Published numbers are from the graphify benchmark page (2026-07-05), which "
        "ran mem0 and supermemory as adapters in its own harness with a Kimi K2.6 "
        "reader/judge. This report uses the same datasets, sample-size targets, "
        "pipeline shape, and key-fact coverage grading, with gpt-4o as the shared "
        "reader/judge; numbers are cross-referenceable, not identical-conditions. "
        "Our in-harness BM25 anchor (next section) calibrates the two harnesses.",
        footnote,
    ))

    # ---- LOCOMO ------------------------------------------------------------
    flow.append(PageBreak())
    flow.append(Paragraph(f"LOCOMO (n={l_meta['n_questions']})", h2))
    flow.append(Paragraph(
        "Ten multi-session conversations (locomo10.json, snap-research/locomo); "
        "questions sampled with a fixed seed, stratified over the four scored "
        "categories (adversarial excluded, as the field reports). One Graphon "
        "group per conversation; the BM25 anchor indexes the identical rendered "
        "text and shares the same reader and judge."
        + (" Graphon (direct) ran in <b>ultra</b> reasoning mode — an agentic "
           "answer harness inside Graphon rather than a fixed top-10 retrieval, "
           "so recall@10 for that row is measured on Graphon's standard "
           "retrieval (identical to the shared-reader row)."
           if loco_ultra else ""),
        body,
    ))
    flow.append(Spacer(1, 8))

    rows = [["System", "QA accuracy", "recall@10", "Avg latency"]]
    ours = [("graphon/direct", g_direct), ("graphon/shared_reader", g_reader),
            ("mem0/shared_reader", l_mem0), ("bm25/shared_reader", bm25)]
    for key, entry in ours:
        if not entry:
            continue
        label = SYSTEM_LABELS[key]
        rec = entry.get("recall_at_10")
        if key == "graphon/direct" and loco_ultra:
            label = "Graphon (direct answer, ultra)"
            rec = g_reader.get("recall_at_10") if g_reader else None
        if key == "mem0/shared_reader":
            rec = None  # extracted-fact memories; n-gram turn recall not comparable
        rows.append([
            label,
            f"{_pct(entry['qa_accuracy'])}%",
            f"{rec:.3f}" if rec is not None else "-",
            f"{entry['avg_latency']:.2f}s",
        ])
    flow.append(_styled_table(rows, [2.5 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch],
                              highlight_row=1))
    flow.append(Spacer(1, 8))
    if bm25:
        delta_ours = _pct(g_direct.get("qa_accuracy", 0)) - _pct(bm25["qa_accuracy"])
        flow.append(Paragraph(
            f"<b>Anchor calibration.</b> Judges and readers differ across harnesses, "
            f"so absolute numbers shift together with the BM25 anchor. In this "
            f"harness Graphon (direct) leads the shared BM25 anchor by "
            f"<b>{delta_ours:+.1f} points</b>; in the published graphify table, "
            f"graphify leads its BM25 anchor by +14.0 points (45.3% vs 31.3%). "
            f"Reading system-minus-anchor deltas keeps the comparison "
            f"harness-independent.",
            body,
        ))
    flow.append(Spacer(1, 8))

    loco_bar_label = "Graphon\n(direct, ultra)" if loco_ultra else "Graphon\n(direct)"
    comp_pairs = [(loco_bar_label, _pct(g_direct.get("qa_accuracy", 0)))]
    comp_colors = [GRAPHON_BLUE]
    if l_mem0:
        comp_pairs.append(("mem0 hosted\n(our harness)", _pct(l_mem0["qa_accuracy"])))
        comp_colors.append(MEM0_ORANGE)
    if bm25:
        comp_pairs.append(("BM25 anchor\n(our harness)", _pct(bm25["qa_accuracy"])))
        comp_colors.append(ANCHOR_TEAL)
    for name, qa, _rec, _cost in PUBLISHED_LOCOMO:
        comp_pairs.append((name.replace(" (", "\n("), qa))
        comp_colors.append(FIELD_GRAY)
    flow.append(_bar_chart(comp_pairs, comp_colors))
    flow.append(Paragraph(
        "Figure 1. LOCOMO QA accuracy: Graphon, mem0's hosted platform run in "
        "this harness (orange), and our BM25 anchor vs published vendor numbers "
        "(gray).", caption))
    flow.append(Spacer(1, 8))

    rec_label = "Graphon\n(standard retrieval)" if loco_ultra else "Graphon\n(direct)"
    rec_pairs = [(rec_label, loco_recall)]
    rec_colors = [GRAPHON_BLUE]
    if bm25 and bm25.get("recall_at_10") is not None:
        rec_pairs.append(("BM25 anchor\n(our harness)", bm25["recall_at_10"]))
        rec_colors.append(ANCHOR_TEAL)
    for name, _qa, rec, _cost in PUBLISHED_LOCOMO:
        rec_pairs.append((name.replace(" (", "\n("), rec))
        rec_colors.append(FIELD_GRAY)
    flow.append(_bar_chart(rec_pairs, rec_colors, value_max=1.0, fmt="{:.3f}"))
    flow.append(Paragraph(
        "Figure 2. LOCOMO retrieval recall@10 (fraction of gold evidence turns "
        "present in the top-10 retrieved chunks).", caption))

    # per-category table
    flow.append(Paragraph("LOCOMO accuracy by question category", h2))
    g_direct_col = "Graphon direct (ultra)" if loco_ultra else "Graphon direct"
    cat_rows = [["Category", "n", g_direct_col, "Graphon + reader",
                 "mem0 hosted", "BM25 anchor"]]
    for cat in LOCOMO_CAT_ORDER:
        d = g_direct.get("by_category", {}).get(cat)
        if not d:
            continue
        r = g_reader.get("by_category", {}).get(cat, {})
        m0c = l_mem0.get("by_category", {}).get(cat, {}) if l_mem0 else {}
        bmc = bm25.get("by_category", {}).get(cat, {}) if bm25 else {}
        cat_rows.append([
            cat, str(d["n"]),
            f"{_pct(d['qa_accuracy'])}%",
            f"{_pct(r['qa_accuracy'])}%" if r else "-",
            f"{_pct(m0c['qa_accuracy'])}%" if m0c else "-",
            f"{_pct(bmc['qa_accuracy'])}%" if bmc else "-",
        ])
    flow.append(_styled_table(
        cat_rows, [1.15 * inch, 0.5 * inch, 1.45 * inch, 1.35 * inch,
                   1.15 * inch, 1.1 * inch]))
    if l_mem0:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            "mem0 (hosted) was run live inside this harness: the identical "
            "conversations were ingested through their platform API (one mem0 "
            "user per conversation, session dates preserved), each question "
            "searched their top-10 memories, and the same gpt-4o reader and "
            "key-fact judge scored the answers. mem0's own published 92.5% on "
            "LOCOMO uses a binary judge instructed to be lenient and up to 200 "
            "retrieved memories per question; under this harness's stricter "
            "quote-verified grading at top-10, their hosted platform scores as "
            "shown above. recall@10 is n-gram-based against gold dialog turns, "
            "which penalizes mem0's extracted-fact memories; QA accuracy is the "
            "comparable column.",
            footnote,
        ))

    # ---- LongMemEval -------------------------------------------------------
    if lme:
        m_sys = lme["systems"]
        mg_direct = m_sys.get("graphon/direct", {})
        mg_reader = m_sys.get("graphon/shared_reader", {})
        m_bm25 = m_sys.get("bm25/shared_reader", {})

        flow.append(PageBreak())
        flow.append(Paragraph(f"LongMemEval-S (n={n_lme})", h2))
        lme_ultra = lme_effort == "ultra"
        flow.append(Paragraph(
            "Each question carries its own ~115k-token multi-session chat history "
            "(~40-50 sessions). Every haystack session was ingested as its own "
            "timestamped document into a dedicated Graphon group (one group per "
            "question). Sample stratified by question type, abstention questions "
            "included. recall@10 is session-level: the fraction of gold evidence "
            "sessions among the sessions of the top-10 retrieved chunks."
            + (" Graphon (direct) ran in <b>ultra</b> reasoning mode — an agentic "
               "answer harness inside Graphon rather than a fixed top-10 retrieval, "
               "so recall@10 is not reported for that row."
               if lme_ultra else ""),
            body,
        ))
        flow.append(Spacer(1, 8))

        rows = [["System", "QA accuracy", "recall@10", "Avg latency"]]
        for key, entry in [("graphon/direct", mg_direct),
                           ("graphon/shared_reader", mg_reader),
                           ("bm25/shared_reader", m_bm25)]:
            if not entry:
                continue
            label = SYSTEM_LABELS[key]
            rec = entry.get("recall_at_10")
            if key == "graphon/direct" and lme_ultra:
                label = "Graphon (direct answer, ultra)"
                rec = None
            rows.append([
                label,
                f"{_pct(entry['qa_accuracy'])}%",
                f"{rec:.3f}" if rec is not None else "-",
                f"{entry['avg_latency']:.2f}s",
            ])
        flow.append(_styled_table(rows, [2.5 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch],
                                  highlight_row=1))
        flow.append(Spacer(1, 10))

        g_bar_label = "Graphon\n(direct, ultra)" if lme_ultra else "Graphon\n(direct)"
        comp_pairs = [(g_bar_label, _pct(mg_direct.get("qa_accuracy", 0)))]
        comp_colors = [GRAPHON_BLUE]
        if m_bm25:
            comp_pairs.append(("BM25 anchor\n(our harness)", _pct(m_bm25["qa_accuracy"])))
            comp_colors.append(ANCHOR_TEAL)
        for name, qa, _rec in PUBLISHED_LME:
            comp_pairs.append((name.replace(" (", "\n("), qa))
            comp_colors.append(FIELD_GRAY)
        flow.append(_bar_chart(comp_pairs, comp_colors))
        flow.append(Paragraph(
            "Figure 3. LongMemEval-S QA accuracy vs published numbers (gray).",
            caption))

        cat_rows = [["Question type", "n", "Graphon direct", "BM25 anchor"]]
        for cat, d in sorted(mg_direct.get("by_category", {}).items()):
            bmc = m_bm25.get("by_category", {}).get(cat, {}) if m_bm25 else {}
            cat_rows.append([
                cat, str(d["n"]),
                f"{_pct(d['qa_accuracy'])}%",
                f"{_pct(bmc['qa_accuracy'])}%" if bmc else "-",
            ])
        flow.append(Spacer(1, 8))
        flow.append(_styled_table(
            cat_rows, [2.1 * inch, 0.6 * inch, 1.6 * inch, 1.6 * inch]))

    # ---- methodology / cost -------------------------------------------------
    flow.append(PageBreak())
    flow.append(Paragraph("Methodology", h2))
    method_items = [
        "Pipeline per question: ingest (render conversations to markdown) -> index "
        "(Graphon group / BM25) -> search (top-10) -> answer -> grade.",
        "Graphon 'direct' uses Graphon's own end-to-end answer — how the product is "
        "used. 'Shared reader' has gpt-4o answer from Graphon's top-10 retrieved "
        "sources, structurally identical to how retriever-style harnesses run every "
        "system; both are reported.",
        "One model (gpt-4o) fills every external LLM role: shared reader, key-fact "
        "decomposition, and judge. Temperature 0.",
        "Grading is key-fact coverage: gold answers are decomposed into atomic facts "
        "(cached); the judge marks each fact covered / partial / missing and must "
        "cite a verbatim quote from the answer for every non-missing verdict. "
        "coverage = (covered + 0.5 x partial) / total. QA accuracy is mean coverage.",
        "No scoring metadata (dialog ids, gold session ids, has_answer flags) is "
        "ever present in the ingested text. Questions and gold answers never reach "
        "Graphon.",
        "Runs are seeded, resumable, and write a per-role token/cost spend ledger; "
        "every row (answer, verdicts, quotes, latencies) is stored in JSONL for audit.",
    ]
    for i, item in enumerate(method_items, 1):
        flow.append(Paragraph(f"{i}. {item}", body))

    flow.append(Paragraph("Cost", h2))
    ledger = loco.get("ledger", {})
    lme_ledger = lme.get("ledger", {}) if lme else {}
    cost_rows = [["Item", "Value"]]
    cost_rows.append(["Graph/index build, external LLM credits", "$0.00 (Graphon indexes server-side)"])
    cost_rows.append(["LOCOMO run OpenAI spend (reader + judge)",
                      f"${_total_ledger_cost('locomo'):.2f}"])
    if lme_ledger:
        lme_cost = _lme_cost_estimate(loco)
        cost_rows.append(["LongMemEval-S run OpenAI spend (reader + judge)",
                          f"~${lme_cost:.2f}" if lme_cost else
                          f"${lme_ledger.get('total_cost_usd', 0):.2f}"])
    effort_str = loco_effort
    if lme:
        effort_str = f"LOCOMO: {loco_effort}; LongMemEval-S: {lme_effort}"
    cost_rows.append(["Reasoning effort", effort_str])
    cost_rows.append(["Shared reader / judge model", l_meta.get("llm_model", "gpt-4o")])
    flow.append(_styled_table(cost_rows, [3.4 * inch, 2.8 * inch]))
    flow.append(Spacer(1, 10))
    flow.append(Paragraph(
        "Reproducing: see experimental/memory_bench/README.md — download_data.py "
        "fetches the pinned datasets; run_eval.py --benchmark locomo|longmemeval "
        "reruns the evaluation; make_customer_report.py rebuilds this report.",
        footnote,
    ))

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        title="Graphon Conversational Memory Benchmark - LOCOMO & LongMemEval-S",
        author="Graphon AI",
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    doc.build(flow)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the customer PDF report.")
    parser.add_argument(
        "--out",
        default=str(PROJECT_DIR / "results" / "Graphon_Memory_Benchmark_Report.pdf"),
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
