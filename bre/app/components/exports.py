"""Exports of the dashboard: the triage CSV and the "Behavioral Risk Profile" PDF (fpdf2).

In demo mode the first line of every export is the SYNTHETIC DATA banner; every export names
the model version, the market state and the wording rule, and carries the "predicted effect,
not causally validated" label next to every intervention effect.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fpdf import FPDF

from components import chrome

_LATIN_FIXES = {"—": "-", "–": "-", "›": ">", "→": "->", "≥": ">=", "≤": "<=", "Δ": "delta ", "θ": "theta", "γ": "gamma", "²": "^2", "·": "|", "⚠️": "", "…": "..."}


def latin(text: Any) -> str:
    """Text safe for the PDF core fonts (latin-1)."""
    s = "" if text is None else str(text)
    for k, v in _LATIN_FIXES.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def triage_csv(table: pd.DataFrame, *, demo: bool, model_version: str, market_state: dict[str, Any], wording: str = chrome.WORDING) -> str:
    """The triage table as CSV text. Line 1 is the SYNTHETIC DATA banner in demo mode, line 2
    the provenance line, then the header and rows."""
    lines: list[str] = []
    if demo:
        lines.append("# " + chrome.SYNTHETIC_BANNER)
    lines.append(f"# model={model_version}; wording={wording}; intervals=80% over the model's parameter draws; market_state={json.dumps(market_state, sort_keys=True)}; generated={stamp()}")
    lines.append(table.to_csv(index=False).rstrip("\n"))
    return "\n".join(lines) + "\n"


class _ProfilePDF(FPDF):
    def __init__(self, banner: str | None) -> None:
        super().__init__()
        self.banner = banner
        self.set_auto_page_break(auto=True, margin=15)

    def header(self) -> None:  # noqa: D401 - fpdf hook
        if self.banner:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(160, 40, 40)
            self.multi_cell(0, 5, latin(self.banner), align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(1)

    def footer(self) -> None:  # noqa: D401 - fpdf hook
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, latin(f"Behavioral Risk Profile - page {self.page_no()} - every probability is a {chrome.WORDING} under the served model"), align="C")


def _h(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.ln(2)
    pdf.cell(0, 7, latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)


def _p(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.6, latin(text), new_x="LMARGIN", new_y="NEXT")


def _table(pdf: FPDF, header: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> None:
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(width=190, col_widths=widths, text_align="LEFT", line_height=4.8, borders_layout="HORIZONTAL_LINES") as table:
        hdr = table.row()
        for h in header:
            hdr.cell(latin(h))
        for r in rows:
            row = table.row()
            for v in r:
                row.cell(latin(v))
    pdf.set_font("Helvetica", "", 9)


def profile_pdf(
    *,
    demo: bool,
    client: dict[str, Any],
    health: dict[str, Any],
    model_info: dict[str, Any],
    market_state: dict[str, Any],
    scenario: dict[str, Any] | None,
    profile: dict[str, Any],
    heatmap_png: bytes | None,
    heatmap_note: str,
    capacity: dict[str, Any] | None,
    ranking: dict[str, Any] | None,
    responses_summary: dict[str, Any] | None,
    session_log: list[dict[str, Any]],
) -> bytes:
    pdf = _ProfilePDF(chrome.SYNTHETIC_BANNER if demo else None)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, latin("Behavioral Risk Profile"), new_x="LMARGIN", new_y="NEXT")
    _p(pdf, f"Client: {client.get('display_label') or client.get('client_id')} (id {client.get('client_id')}) - generated {stamp()}")
    _p(pdf, f"Served model: {health.get('model_version')} ({health.get('model_type')}, family {model_info.get('family')}); trained on synthetic data: {model_info.get('is_synthetic_training')}; "
            f"evaluation status: {model_info.get('evaluation_status')}")
    _p(pdf, f"Intervals: {model_info.get('interval_source')}")
    _p(pdf, f"Calibration rule: {model_info.get('calibration_rule')}")

    _h(pdf, "Market state (advisor inputs)")
    _p(pdf, json.dumps(market_state, sort_keys=True))
    if scenario:
        _p(pdf, f"Scenario scored by the API: loss {chrome.fmt_pct(abs(scenario.get('loss_pct') or 0))}, contexts {chrome.condition_label(scenario.get('context_tags') or [])}, "
                f"cause frame -> {scenario.get('cause_context')} (similarity {float(scenario.get('cause_similarity') or 0):.2f}{', WEAK MATCH' if scenario.get('weak_match') else ''}); notes: {'; '.join(scenario.get('notes') or []) or 'none'}")

    _h(pdf, "Baseline risk state")
    b = profile.get("baseline") or {}
    rs = profile.get("risk_score") or {}
    cons = profile.get("consistency") or {}
    cov = client.get("covariates") or {}
    q = cov.get("self_reported_risk_tolerance")
    _p(pdf, f"State angle theta = {chrome.fmt_p(b.get('bloch_theta'), 3)} rad; {chrome.WORDING} with no loss and no context = {chrome.prob_text(b.get('p_sell_base_model'), b.get('ci80'), b.get('ci95'))}. {b.get('note', '')}")
    _p(pdf, f"Classical-equivalent risk score (logit of the {chrome.WORDING} in the typical crisis) = {chrome.fmt_p(rs.get('value'))} (80% {chrome.fmt_interval(rs.get('ci80'))}); "
            f"definition: {rs.get('definition')}; R^2 vs state angle across the book: {chrome.fmt_p(rs.get('r2_vs_angle'))} ({rs.get('r2_note')})")
    _p(pdf, f"Client's own questionnaire score (self-reported risk tolerance, 1-7): {q if q is not None else 'not recorded'}")
    _p(pdf, f"Consistency (dephasing rate gamma): {chrome.fmt_p(cons.get('gamma'), 3)} (80% {chrome.fmt_interval(cons.get('ci80'), 3)}); source: {cons.get('source') or cons.get('note')}")

    _h(pdf, "Context sensitivities (population parameters, 80% intervals)")
    sens = profile.get("sensitivities") or {}
    rows = []
    for tag in chrome.ordered_contexts(list(sens)):
        e = sens[tag]
        nc = e.get("n_calibration") or {}
        rows.append([chrome.context_label(tag), chrome.fmt_signed(e.get("delta_p_at_risk_loss")), chrome.fmt_interval(e.get("delta_p_ci80")), chrome.fmt_p(e.get("theta_norm"), 3), chrome.fmt_interval(e.get("theta_norm_ci80"), 3), f"{nc.get('n_responses')} ({nc.get('status')})"])
    _table(pdf, ["context", "delta P(sell)", "80% interval", "|theta_c|", "80% interval", "calibration N"], rows, [34, 24, 34, 22, 34, 42])

    if heatmap_png:
        _h(pdf, "Predicted probability of selling over loss x context")
        _p(pdf, heatmap_note)
        pdf.image(io.BytesIO(heatmap_png), w=190)

    if capacity:
        _h(pdf, "Behavioral drawdown capacity (allocation guardrail)")
        _p(pdf, f"Target {chrome.WORDING}: {chrome.fmt_p(capacity.get('target'))}; capacity: {chrome.fmt_pct(capacity.get('drawdown_capacity'))} ({capacity.get('capacity_status')}). "
                f"Definition: {capacity.get('capacity_definition')}. {chrome.GUARDRAIL_LABEL}.")

    if ranking:
        _h(pdf, f"Intervention ranking - {ranking.get('label') or chrome.INTERVENTION_LABEL_FALLBACK}")
        _p(pdf, f"Before any intervention: {chrome.prob_text(ranking.get('p_before'), ranking.get('ci80_before'))}; calibration: {chrome.calibration_text(ranking.get('n_calibration'))}")
        rows = [[r.get("rank"), r.get("name"), chrome.fmt_p(r.get("p_after")), chrome.fmt_signed(r.get("delta_p")), chrome.fmt_interval(r.get("delta_ci80")), chrome.condition_label(r.get("contexts_after") or []), r.get("label")] for r in ranking.get("ranked", [])]
        _table(pdf, ["rank", "intervention", "P(sell) after", "predicted change", "80% interval", "contexts after", "label"], rows, [12, 40, 22, 24, 30, 32, 30])

    if responses_summary:
        _h(pdf, "Response history (stated choices, not predictions)")
        _p(pdf, f"{responses_summary.get('n_items')} stored items ({responses_summary.get('n_sell_items')} sell/hold items, {responses_summary.get('n_sell')} sold); source: {responses_summary.get('source')}; synthetic: {responses_summary.get('synthetic')}")
        if responses_summary.get("stated_vs_revealed"):
            _p(pdf, f"Stated vs revealed: {responses_summary['stated_vs_revealed']}")

    _h(pdf, "Interventions logged in this session")
    if session_log:
        _table(pdf, ["logged at", "intervention", "predicted change", "note", "persisted"], [[x["logged_at"], x["intervention"], chrome.fmt_signed(x.get("predicted_delta_p")), x.get("note", ""), "no (session only)"] for x in session_log], [36, 40, 28, 60, 26])
    else:
        _p(pdf, "none")
    out = pdf.output()
    return bytes(out)


__all__ = ["latin", "profile_pdf", "stamp", "triage_csv"]
