"""Phase 8 report assembler: ``python -m bre.report`` writes ``reports/REPORT.md`` from generated files only.

CLAUDE.md rule 6 (no fabricated numbers) is implemented structurally: this module never types a
number. Every table and every figure in the report is copied from a file that another phase
generated (the data map, the processed-table README, the Phase 2 recovery summary, the Q1 report,
the A->B transfer report, a Phase 4 output folder, the synthetic profile JSON written by
:func:`generate_profiles`), and ``reports/REPORT_appendix_files.md`` lists every file read, with its
modification time, size and sha256. The only hand-written text is the fixed prose of the
limitations, the "what the dashboard can honestly claim" section and the pointers of the next-step
plan; none of it carries a number that is not already cited from a source file.

Missing inputs never raise: the report carries ``(not available: <path>)`` where the content would
be, and the appendix lists the path under "inputs not available".

Phase 4 (real data) is included only from an explicit ``--phase4 <dir>`` (or
``render(phase4_dir=...)``) holding ``verdict.json``, ``table.md`` and ``structural.md``;
otherwise the section is the single sentence :data:`PHASE4_ABSENT` and nothing else.

The per-investor profile examples are the five synthetic archetypes of the demo book
(``data/synthetic/demo_book.clients.json``) scored by the demo database's active model through
:mod:`bre.predict` (``profile`` and ``predict_sell`` at the five design loss levels; no fitting).
:func:`generate_profiles` writes ``reports/profiles_synthetic.json`` (``is_synthetic: true``) and
the report renders that file, labelling every number SYNTHETIC.

Usage::

    python -m bre.report [--out reports/REPORT.md] [--phase4 reports/phase4/<name>] [--no-profiles]

``render(phase4_dir=None) -> str`` returns the report text without writing anything (tests).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""``bre/``: every relative path in this module is resolved against it."""

DEFAULT_OUT = "reports/REPORT.md"
APPENDIX_NAME = "REPORT_appendix_files.md"
PROFILES_JSON = "reports/profiles_synthetic.json"
DEMO_DB = "data/synthetic/demo.db"

PHASE4_ABSENT = "Phase 4 has not been run on real data; no real-data metrics are reported."
"""The whole Phase 4 section when no complete Phase 4 output folder is given."""

PHASE4_FILES: tuple[str, ...] = ("verdict.json", "table.md", "structural.md")
"""A Phase 4 folder counts as present only when all three exist (``bre.phase4.write_outputs``)."""

SYNTHETIC = "SYNTHETIC"
NA = "(not available: {path})"

INPUTS: dict[str, str] = {
    "data_map": "data/DATA_MAP.md",
    "processed_readme": "data/processed/README.md",
    "recovery_summary": "reports/recovery/summary.md",
    "status": "STATUS.md",
    "plan": "PLAN.md",
    "q1": "reports/q1_qq_equality.md",
    "transfer": "reports/transfer/A_to_B.md",
    "data_gaps": "reports/DATA_GAPS.md",
    "predict_src": "src/bre/predict.py",
    "design_src": "src/bre/design.py",
    "clients_json": "data/synthetic/demo_book.clients.json",
    "registration": "data/REGISTRATION.md",
    "prolific": "instrument/PROLIFIC.md",
    "profiles_json": PROFILES_JSON,
}
"""Every generated (or in-repo) file the report reads, relative to ``bre/``."""

SECTION_TITLES: tuple[str, ...] = (
    "1. Data map",
    "2. Phase 2: model recovery on synthetic data",
    "3. Q1 order-effect model and the QQ equality on dataset D",
    "4. A -> B transfer on public data (real data)",
    "5. Held-out metrics: every fitted model in one table",
    "6. Phase 4: held-out metrics, structural tests and decision-rule verdict (real data)",
    "7. Per-investor profile examples (SYNTHETIC archetypes)",
    "8. Limitations",
    "9. What the dashboard can honestly claim today vs after real data",
    "10. Costed next-step plan",
    "11. Appendix: source files",
)
"""The ``##`` headings of the report, in order (tests check them)."""


# ---------------------------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------------------------


@dataclass
class Sources:
    """Files read while assembling the report (for the appendix) and inputs that were missing."""

    root: Path
    inputs: dict[str, str] = field(default_factory=lambda: dict(INPUTS))
    used: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def path(self, rel: str | Path) -> Path:
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.root / p

    def label(self, rel: str | Path) -> str:
        p = self.path(rel)
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    def na(self, rel: str | Path) -> str:
        return NA.format(path=self.label(rel))

    def read(self, rel: str | Path, *, record_missing: bool = True) -> str | None:
        """The file's text, or None (recorded under ``missing``) when it cannot be read."""
        p = self.path(rel)
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            if record_missing and self.label(rel) not in self.missing:
                self.missing.append(self.label(rel))
            return None
        self.record(p)
        return text

    def record(self, p: Path) -> None:
        label = self.label(p)
        if label in self.used:
            return
        try:
            st = p.stat()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return
        self.used[label] = {
            "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            "bytes": st.st_size,
            "sha256": digest,
        }


# ---------------------------------------------------------------------------------------------
# Markdown helpers (no numbers are produced here; text is moved, never invented)
# ---------------------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_SEP_CELL = re.compile(r"^:?-{2,}:?$")


def _split(text: str) -> list[tuple[int, str, list[str]]]:
    """``(level, title, own_body_lines)`` for the text before the first heading (level 0) and for
    every ATX heading outside code fences; the body stops at the next heading of any level."""
    lines = text.splitlines()
    fence = False
    heads: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = _HEADING.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))
    first = heads[0][0] if heads else len(lines)
    out: list[tuple[int, str, list[str]]] = [(0, "", lines[:first])]
    for k, (i, lvl, title) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.append((lvl, title, lines[i + 1 : j]))
    return out


def _body(text: str, level: int, title_prefix: str) -> list[str] | None:
    """Own body of the first heading at ``level`` whose title starts with ``title_prefix``."""
    for lvl, title, body in _split(text):
        if lvl == level and title.startswith(title_prefix):
            return body
    return None


def _strip_blank(lines: list[str]) -> list[str]:
    a, b = 0, len(lines)
    while a < b and not lines[a].strip():
        a += 1
    while b > a and not lines[b - 1].strip():
        b -= 1
    return lines[a:b]


def _demote(text: str, by: int) -> str:
    """Add ``by`` levels to every heading outside code fences (content otherwise untouched)."""
    out = []
    fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        elif not fence:
            m = _HEADING.match(line)
            if m:
                line = "#" * min(6, len(m.group(1)) + by) + " " + m.group(2)
        out.append(line)
    return "\n".join(out)


def _first_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    rows: list[list[str]] = []
    started = False
    for line in lines:
        s = line.strip()
        if s.startswith("|"):
            started = True
            rows.append([c.strip() for c in s.strip("|").split("|")])
        elif started:
            break
    if len(rows) < 2:
        return None
    header = rows[0]
    body = [r for r in rows[1:] if not all(_SEP_CELL.match(c) for c in r)]
    return header, body


def _table(header: list[str], rows: list[list[str]]) -> str:
    esc = lambda c: str(c).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def _quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())


def _paragraph_starting_with(text: str, prefix: str) -> str | None:
    """The paragraph (consecutive non-blank lines) whose first line starts with ``prefix``;
    a bullet's continuation lines (indented) are part of it."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            j = i + 1
            while j < len(lines) and lines[j].strip() and not (line.startswith(("* ", "- ")) and lines[j].startswith(("* ", "- "))):
                j += 1
            return "\n".join(lines[i:j])
    return None


def _fmt(x: Any, nd: int = 3) -> str:
    """Display a stored value; never computes anything beyond rounding for display."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    if isinstance(x, (list, tuple)):
        return ", ".join(_fmt(v, nd) for v in x)
    return str(x)


def _ast_constants(text: str, names: set[str]) -> dict[str, Any]:
    """Module-level ``NAME = <literal>`` assignments of a Python source (no import needed)."""
    out: dict[str, Any] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                try:
                    out[name] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def _module_docstring(text: str) -> str:
    try:
        return ast.get_docstring(ast.parse(text)) or ""
    except SyntaxError:
        return ""


_RST_ROLE = re.compile(r":(?:data|func|mod|class|attr):`([^`]*)`")


def _git_line(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "git: (not available)"
    if not sha:
        return "git: (not available)"
    return f"git commit `{sha}`" + (f" on branch `{branch}`" if branch else "")


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def _sec_data_map(src: Sources) -> str:
    parts: list[str] = []
    text = src.read(src.inputs["data_map"])
    parts.append("### Data map (`data/DATA_MAP.md`, verbatim)\n")
    parts.append(_demote(text, 2) if text is not None else src.na(src.inputs["data_map"]))
    parts.append("\n### Processed tables (`data/processed/README.md`, generated by `python -m bre.data_cli`)\n")
    readme = src.read(src.inputs["processed_readme"])
    if readme is None:
        parts.append(src.na(src.inputs["processed_readme"]))
        return "\n".join(parts)
    sections = _split(readme)
    prov = [ln for lvl, _, body in sections if lvl <= 1 for ln in body if ln.startswith("- ")]
    if prov:
        parts.append("\n".join(prov) + "\n")
    datasets = _body(readme, 2, "Datasets")
    parts.append("\n".join(_strip_blank(datasets)) if datasets else f"(not available: `## Datasets` table in {src.label(src.inputs['processed_readme'])})")
    not_loaded = _body(readme, 2, "Not loaded")
    if not_loaded:
        parts.append("\n**Not loaded** (from the same README):\n\n" + "\n".join(_strip_blank(not_loaded)))
    return "\n".join(parts)


def _sec_recovery(src: Sources) -> str:
    parts: list[str] = []
    text = src.read(src.inputs["recovery_summary"])
    if text is None:
        parts.append(src.na(src.inputs["recovery_summary"]))
    else:
        sections = _split(text)
        preamble = [ln for lvl, _, body in sections if lvl <= 1 for ln in body]
        parts.append("\n".join(_strip_blank(preamble)))
        by_n = _body(text, 2, "Correct family-level selection by N")
        parts.append("\n### Correct family-level selection by N, and N_target with its rule (`reports/recovery/summary.md`)\n")
        parts.append("\n".join(_strip_blank(by_n)) if by_n else f"(not available: `## Correct family-level selection by N` in {src.label(src.inputs['recovery_summary'])})")
        parts.append("\n### Confusion matrices per N (true generator x selected model / family)\n")
        found = False
        for lvl, title, body in sections:
            if lvl == 2 and title.startswith("N = "):
                found = True
                keep: list[str] = []
                for ln in body:
                    if ln.startswith("Selection-fold NLL") or ln.startswith("G_Q parameter recovery") or ln.startswith("G_C parameter recovery"):
                        break
                    keep.append(ln)
                parts.append(f"#### {title}\n\n" + "\n".join(_strip_blank(keep)) + "\n")
        if not found:
            parts.append(f"(not available: `## N = ...` sections in {src.label(src.inputs['recovery_summary'])})")
        failed = _body(text, 2, "Failed fits")
        if failed is not None:
            n_failed = sum(1 for ln in failed if ln.startswith("- "))
            parts.append(f"Failed fits listed in the summary: {n_failed} (see `## Failed fits` in `{src.label(src.inputs['recovery_summary'])}`).\n")
        parts.append(
            f"Selection-fold NLL per cell, parameter-recovery tables and the failure list: `{src.label(src.inputs['recovery_summary'])}` "
            "(scatter plots `reports/recovery/recovery_scatter_<gen>_N<N>.png`, raw results `reports/recovery/results.json`, "
            "`reports/recovery/N_target.json`)."
        )
    parts.append("\n### Caveat recorded in STATUS.md's progress log\n")
    status = src.read(src.inputs["status"])
    if status is None:
        parts.append(src.na(src.inputs["status"]))
    else:
        log = _body(status, 2, "Progress log")
        pool = log if log is not None else status.splitlines()
        caveats = [ln.lstrip("* ").rstrip() for ln in pool if "CAVEAT" in ln]
        if caveats:
            parts.append("Quoted verbatim from `STATUS.md` (progress log):\n")
            parts.extend(_quote(c) + "\n" for c in caveats)
        else:
            parts.append("No line containing the word CAVEAT is present in the progress log of `STATUS.md`.")
    return "\n".join(parts)


def _sec_q1(src: Sources) -> str:
    text = src.read(src.inputs["q1"])
    if text is None:
        return src.na(src.inputs["q1"])
    parts: list[str] = []
    sections = _split(text)
    preamble = [ln for lvl, _, body in sections if lvl <= 1 for ln in body]
    parts.append("\n".join(_strip_blank(preamble)) + "\n")
    for title in ("Model and statistic", "Provenance"):
        body = _body(text, 2, title)
        if body is not None:
            parts.append(f"### {title}\n\n" + "\n".join(_strip_blank(body)) + "\n")
    # children of "## Results", grouped by the label after the em dash in their heading
    in_results = False
    verified: list[tuple[str, list[str]]] = []
    unverified: dict[str, list[tuple[str, list[str]]]] = {}
    for lvl, title, body in sections:
        if lvl == 2:
            in_results = title.startswith("Results")
            continue
        if in_results and lvl == 3:
            label = title.split(" — ", 1)[1].strip() if " — " in title else title
            if "VERIFIED" in label.upper() and not label.upper().startswith("UNVERIFIED"):
                verified.append((title, body))
            else:
                unverified.setdefault(label, []).append((title, body))
    if not verified and not unverified:
        parts.append("### Results\n\n" + (f"(not available: `## Results` sub-sections in {src.label(src.inputs['q1'])})"))
    else:
        parts.append("### Verified table results (label in the source file: VERIFIED)\n")
        if verified:
            for title, body in verified:
                parts.append(f"#### {title}\n\n" + "\n".join(_strip_blank(body)) + "\n")
        else:
            parts.append(f"(not available: no `### ... — VERIFIED` table in {src.label(src.inputs['q1'])})\n")
        for label, items in unverified.items():
            parts.append(f"### {label}\n")
            parts.append(
                f"The tables below carry the label \"{label}\" in `{src.label(src.inputs['q1'])}` and are reproduced under that label only; "
                "the source file states they are implementation checks, not reported results.\n"
            )
            for title, body in items:
                parts.append(f"#### {title}\n\n" + "\n".join(_strip_blank(body)) + "\n")
    for title in ("Uncertainty", "Interpretation"):
        body = _body(text, 2, title)
        if body is not None:
            parts.append(f"### {title}\n\n" + "\n".join(_strip_blank(body)) + "\n")
    return "\n".join(parts)


def _sec_transfer(src: Sources) -> str:
    text = src.read(src.inputs["transfer"])
    if text is None:
        return src.na(src.inputs["transfer"])
    return f"Reproduced verbatim from `{src.label(src.inputs['transfer'])}` (headings demoted); raw results in `reports/transfer/A_to_B.json`.\n\n" + _demote(text, 2)


def _sec_one_table(src: Sources, phase4_present: bool) -> str:
    parts: list[str] = [
        "Every model with real-data held-out metrics in a generated file, side by side. Rows are copied from the "
        f"held-out-problem tables of `{src.label(src.inputs['transfer'])}` (one row per model and target; the `target` column is added). "
        "The models of PLAN.md section 4 that do not appear here have no real-data held-out metrics yet; they get them from Phase 4 "
        "(section 6" + (", included below)." if phase4_present else ", not run yet).") + "\n"
    ]
    text = src.read(src.inputs["transfer"], record_missing=False)
    if text is None:
        parts.append(src.na(src.inputs["transfer"]))
        return "\n".join(parts)
    header: list[str] | None = None
    rows: list[list[str]] = []
    for lvl, title, body in _split(text):
        m = re.match(r"^Target (\S+): held-out problems", title) if lvl == 2 else None
        if not m:
            continue
        tab = _first_table(body)
        if tab is None:
            continue
        h, rs = tab
        if header is None:
            header = ["target"] + h
        elif h != header[1:]:
            parts.append(f"(table for target {m.group(1)} has different columns; see section 4)")
            continue
        rows.extend([m.group(1)] + r for r in rs)
    if header is None:
        parts.append(f"(not available: `## Target ...: held-out problems` tables in {src.label(src.inputs['transfer'])})")
    else:
        parts.append(_table(header, rows))
        parts.append("\nReal data (choices13k fit, CPC15 / CPC18 held-out problems); the constant-rate column is the reference each transfer is judged against (section 4).")
    if phase4_present:
        parts.append("\nPhase 4 per-split tables (real data, every registered model) are reproduced verbatim in section 6.")
    return "\n".join(parts)


def _sec_phase4(src: Sources, phase4_dir: str | Path | None) -> tuple[str, bool]:
    """The section text and whether a complete Phase 4 folder was included."""
    if phase4_dir is None:
        return PHASE4_ABSENT, False
    d = src.path(phase4_dir)
    if not all((d / f).is_file() for f in PHASE4_FILES):
        for f in PHASE4_FILES:
            if not (d / f).is_file():
                src.missing.append(src.label(d / f))
        return PHASE4_ABSENT, False
    parts: list[str] = [f"Source folder: `{src.label(d)}` (output of `python -m bre.phase4`; every number below is generated there).\n"]
    verdict_raw = src.read(d / "verdict.json")
    verdict_sentence = None
    try:
        payload = json.loads(verdict_raw or "")
        verdict_sentence = payload.get("verdict") if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        payload = None
    if isinstance(verdict_sentence, str):
        parts.append(f"**Verdict (copied from `verdict.json`):** {verdict_sentence}\n")
    table = src.read(d / "table.md")
    parts.append("### Model comparison per split (`table.md`, verbatim)\n")
    parts.append(_demote(table, 3) if table is not None else src.na(d / "table.md"))
    structural = src.read(d / "structural.md")
    parts.append("\n### Structural tests (`structural.md`, verbatim)\n")
    parts.append(_demote(structural, 3) if structural is not None else src.na(d / "structural.md"))
    parts.append("\n### Decision-rule verdict (`verdict.json`, verbatim)\n")
    parts.append("```json\n" + (verdict_raw.rstrip("\n") if verdict_raw is not None else src.na(d / "verdict.json")) + "\n```")
    verdict_md = src.read(d / "verdict.md", record_missing=False)
    if verdict_md is not None:
        parts.append("\n### Verdict text (`verdict.md`, verbatim)\n")
        parts.append(_demote(verdict_md, 3))
    return "\n".join(parts), True


def _sec_profiles(src: Sources) -> str:
    text = src.read(src.inputs["profiles_json"])
    if text is None:
        return (
            src.na(src.inputs["profiles_json"])
            + "\n\nThe file is written by `python -m bre.report` (function `bre.report.generate_profiles`) from the demo "
            "database's active model (`data/synthetic/demo.db`) and the five synthetic archetypes of "
            "`data/synthetic/demo_book.clients.json`; no fitting is involved."
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return f"(not available: {src.label(src.inputs['profiles_json'])} is not valid JSON: {e})"
    if not isinstance(data, dict) or data.get("is_synthetic") is not True:
        return f"(not available: {src.label(src.inputs['profiles_json'])} does not carry `is_synthetic: true`; refusing to render it as synthetic)"
    art = data.get("artifact", {})
    profs = data.get("profiles", [])
    parts: list[str] = []
    parts.append(
        f"**Every number in this section is {SYNTHETIC}.** The clients are the five hand-written archetypes of the synthetic demo book "
        f"(`{data.get('clients_source', src.inputs['clients_json'])}`; their latents are design choices, not fitted values), and the served model is "
        f"`{art.get('version', 'n/a')}` ({art.get('model_name', 'n/a')}, family {art.get('family', 'n/a')}), whose training data are synthetic "
        f"(`is_synthetic_training = {str(art.get('is_synthetic_training')).lower()}`). Outputs were produced by `bre.predict.profile` and "
        f"`bre.predict.predict_sell` on {data.get('generated_at', 'n/a')} and stored in `{src.label(src.inputs['profiles_json'])}`. "
        "Probabilities are predicted probabilities of selling under that model; nothing here is an observation."
    )
    if art.get("notes"):
        parts.append("\nArtifact notes (verbatim): " + str(art["notes"]))
    lr = data.get("book_loss_response")
    if isinstance(lr, dict) and lr.get("monotone_share") is not None:
        parts.append(
            f"\nBook-level loss-response diagnostic stored in the artifact ({SYNTHETIC}; definition in section 8): monotone share "
            f"{_fmt(lr.get('monotone_share'))} of {lr.get('n_clients', 'n/a')} synthetic clients; mean predicted P(sell) per design loss "
            f"{_fmt(lr.get('loss_levels'), 2)}: {_fmt(lr.get('mean_p_sell_by_loss'))}."
        )
    r2 = data.get("risk_score_r2")
    if isinstance(r2, dict) and r2.get("r2") is not None:
        parts.append(f"\nR^2 between the classical-equivalent risk score and the state angle across the synthetic book ({SYNTHETIC}): {_fmt(r2.get('r2'))}, 95% interval {_fmt(r2.get('ci95'))}. {r2.get('note', '')}")
    # summary table
    header = [
        f"archetype ({SYNTHETIC})", "client_id", f"P(sell; L=0, no context) [{SYNTHETIC}]", "80% interval",
        f"Bloch theta (rad) [{SYNTHETIC}]", f"gamma, consistency [{SYNTHETIC}]", f"risk score = logit P(sell; typical crisis, {_fmt(data.get('risk_score_loss'), 2)}) [{SYNTHETIC}]",
        f"P(sell) at that scenario [{SYNTHETIC}]", f"drawdown capacity at target {_fmt(data.get('drawdown_target'), 2)} [{SYNTHETIC}]",
        "generator truth P(sell; L=0) (synthetic design choice)",
    ]
    rows = []
    for p in profs:
        prof = p.get("profile", {})
        base = prof.get("baseline", {})
        cons = prof.get("consistency", {})
        risk = prof.get("risk_score", {})
        cap = p.get("drawdown_capacity", {})
        truth = p.get("truth", {})
        rows.append([
            p.get("archetype", "n/a"), f"`{p.get('client_id', 'n/a')}`", _fmt(base.get("p_sell_base_model")), _fmt(base.get("ci80")),
            _fmt(base.get("bloch_theta")), _fmt(cons.get("gamma")), _fmt(risk.get("value")), _fmt(risk.get("p_sell")),
            f"{_fmt(cap.get('capacity'), 2)} ({cap.get('status', 'n/a')})", _fmt(truth.get("truth_p_sell_base")),
        ])
    parts.append("\n### Summary of the five archetypes (" + SYNTHETIC + ")\n")
    parts.append(_table(header, rows) if rows else "(not available: no profiles in the JSON)")
    parts.append(f"\nDefinitions (from `bre.predict`): risk score = {profs[0]['profile']['risk_score'].get('definition', 'n/a') if profs and profs[0].get('profile', {}).get('risk_score') else 'n/a'}; drawdown capacity: {data.get('capacity_definition', 'n/a')}")
    # per-archetype detail
    for p in profs:
        prof = p.get("profile", {})
        parts.append(f"\n### {p.get('archetype', 'n/a')} — `{p.get('client_id', 'n/a')}` ({SYNTHETIC})\n")
        if p.get("description"):
            parts.append(f"Archetype design (from `bre.demo.ARCHETYPES`, a synthetic design choice): {p['description']}\n")
        parts.append(f"Intake covariates ({SYNTHETIC}): `{json.dumps(p.get('covariates'), sort_keys=True)}`; known to the model: {_fmt(prof.get('known_subject'))}.\n")
        grid = p.get("loss_grid", [])
        gh = ["loss_pct", f"P(sell) [{SYNTHETIC}]", "80% interval", "95% interval", f"delta_LTP [{SYNTHETIC}]", "top driver (context or loss whose removal changes P most)", "delta_p of that driver"]
        grows = []
        for g in grid:
            td = g.get("top_driver", {}) or {}
            grows.append([_fmt(g.get("loss_pct"), 2), _fmt(g.get("p")), _fmt(g.get("ci80")), _fmt(g.get("ci95")), _fmt((g.get("interference") or {}).get("ltp"), 4), str(td.get("driver")), _fmt(td.get("delta_p"))])
        parts.append(f"P(sell) at the five design losses, no context, scenario-first ({SYNTHETIC}):\n\n" + (_table(gh, grows) if grows else "(not available: loss grid)"))
        sens = prof.get("sensitivities", {}) or {}
        sh = ["context", f"rotation norm theta_c [{SYNTHETIC}]", "95% interval", f"delta P(sell) at the risk-score loss when applied singly [{SYNTHETIC}]", "80% interval", "n_calibration (training responses behind the context)"]
        srows = []
        for tag, s in sens.items():
            ncal = s.get("n_calibration")
            ncal_txt = ncal.get("n_responses", ncal.get("status", "n/a")) if isinstance(ncal, dict) else ncal
            srows.append([tag, _fmt(s.get("theta_norm")), _fmt(s.get("theta_norm_ci95")), _fmt(s.get("delta_p_at_risk_loss")), _fmt(s.get("delta_p_ci80")), str(ncal_txt)])
        parts.append(f"\nContext sensitivities ({SYNTHETIC}; population rotation norms, per-subject state):\n\n" + (_table(sh, srows) if srows else "(not available: sensitivities)"))
        crisis = p.get("crisis_grid", {}) or {}
        if crisis.get("p") is not None:
            ch = ["loss (magnitude)", f"P(sell; typical crisis {crisis.get('contexts')}) [{SYNTHETIC}]"]
            crows = [[_fmt(L, 2), _fmt(v)] for L, v in zip(crisis.get("losses", []), crisis.get("p", []))]
            cap = p.get("drawdown_capacity", {}) or {}
            parts.append(f"\nTypical-crisis loss grid and the behavioral drawdown capacity ({SYNTHETIC}; first-crossing rule, target {_fmt(data.get('drawdown_target'), 2)}): capacity {_fmt(cap.get('capacity'), 2)}, status \"{cap.get('status', 'n/a')}\".\n\n" + _table(ch, crows))
        cons = prof.get("consistency", {}) or {}
        if cons.get("gamma") is not None:
            parts.append(f"\nConsistency (dephasing rate gamma, {SYNTHETIC}): {_fmt(cons.get('gamma'))}, 95% interval {_fmt(cons.get('ci95'))}; source: {cons.get('source', 'n/a')}. Generator truth gamma (synthetic design choice): {_fmt((p.get('truth') or {}).get('truth_gamma'))}.")
    return "\n".join(parts)


def _predict_constants(src: Sources) -> dict[str, Any]:
    text = src.read(src.inputs["predict_src"])
    if text is None:
        return {}
    names = {"LOSS_RESPONSE_DEFINITION", "CAPACITY_DEFINITION", "INTERVENTION_LABEL", "PROBABILITY_WORDING", "DRAWDOWN_TARGET", "TYPICAL_CRISIS_CONTEXTS", "RISK_SCORE_LOSS"}
    consts = _ast_constants(text, names)
    doc = _module_docstring(text)
    for para in re.split(r"\n\s*\n", doc):
        if "DRAWDOWN_TARGET" in para and "monotone" in para:
            consts["_drawdown_paragraph"] = _RST_ROLE.sub(r"`\1`", para).strip()
            break
    design = src.read(src.inputs["design_src"], record_missing=False)
    if design is not None:
        consts.update(_ast_constants(design, {"LOSS_PCTS"}))
    return consts


def _sec_limitations(src: Sources) -> str:
    parts: list[str] = []
    parts.append("### Data gaps\n")
    gaps = src.read(src.inputs["data_gaps"])
    if gaps is not None:
        parts.append(f"From `{src.label(src.inputs['data_gaps'])}` (verbatim, headings demoted):\n\n" + _demote(gaps, 3))
    else:
        parts.append(
            src.na(src.inputs["data_gaps"])
            + "\n\nUntil that file exists, the data gaps are those stated in the data map (section 1: status column and the closing "
            "\"Still missing\" paragraph), in `STATUS.md` (\"Blocked / gated\") and in `data/REGISTRATION.md` (datasets E and H are "
            "registration GATEs, CLAUDE.md rule 4). No dataset in hand contains a manipulated loss, a manipulated context and a later real "
            "action for the same person; the intake battery plus the intervention log is the only path to that, and it has collected "
            "no responses yet (the `intake_battery` row of the processed-table README in section 1)."
        )
    parts.append("\n### Loss response of the served model (caveat from the `bre.predict` module docstring)\n")
    consts = _predict_constants(src)
    if not consts:
        parts.append(src.na(src.inputs["predict_src"]))
    else:
        if consts.get("_drawdown_paragraph"):
            parts.append("Quoted from the module docstring (rST role markup removed):\n\n" + _quote(consts["_drawdown_paragraph"]) + "\n")
        grid = consts.get("LOSS_PCTS")
        if consts.get("LOSS_RESPONSE_DEFINITION"):
            txt = consts["LOSS_RESPONSE_DEFINITION"].replace("{grid}", str(list(grid)) if grid is not None else "{grid}")
            parts.append("`LOSS_RESPONSE_DEFINITION`: " + txt + "\n")
        if consts.get("CAPACITY_DEFINITION"):
            txt = consts["CAPACITY_DEFINITION"]
            txt = txt.replace("{grid}", str(list(grid)) if grid is not None else "{grid}")
            txt = txt.replace("{contexts}", str(list(consts["TYPICAL_CRISIS_CONTEXTS"])) if "TYPICAL_CRISIS_CONTEXTS" in consts else "{contexts}")
            txt = txt.replace("{target}", str(consts["DRAWDOWN_TARGET"]) if "DRAWDOWN_TARGET" in consts else "{target}")
            parts.append("`CAPACITY_DEFINITION`: " + txt + "\n")
        if consts.get("PROBABILITY_WORDING"):
            parts.append(f"Every probability the API returns is a \"{consts['PROBABILITY_WORDING']}\" under the served model, and every intervention effect carries the label \"{consts.get('INTERVENTION_LABEL', 'n/a')}\"; no causal claim is made from any of them (CLAUDE.md rule 6).")
    parts.append("\n### The CPC18 mapping of Phase 4 is an analog (PLAN.md section 6)\n")
    plan = src.read(src.inputs["plan"])
    para = _paragraph_starting_with(plan, "Real-data mapping for split (b)") if plan is not None else None
    if para:
        parts.append("Quoted verbatim from `PLAN.md` section 6:\n\n" + _quote(para))
    else:
        parts.append(src.na(src.inputs["plan"]) if plan is None else f"(not available: the paragraph starting \"Real-data mapping for split (b)\" in {src.label(src.inputs['plan'])})")
    parts.append(
        "\nConsequently a Phase 4 verdict, positive or negative, is about retreat to the safer lottery after an experienced loss in CPC18, "
        "not about panic selling of a portfolio; the dashboard's scenarios (loss size, cause frame, social cue, partial recovery, question "
        "order) are calibrated on real data only once the intake battery has collected them."
    )
    parts.append("\n### Other limitations (fixed text; the numbers behind each point are in the section named)\n")
    parts.append(
        "* The served demo model is trained on labelled synthetic data only (section 7 states the artifact and its notes); its probabilities, "
        "intervals, sensitivities and consistency scores describe that synthetic book and nothing else.\n"
        "* The recovery study (section 2) is on synthetic data from generators whose forms match the fitted models; the caveat quoted there "
        "about missing cells applies to the N_target it reports, and the study says nothing about whether real investors behave like any generator.\n"
        "* The A -> B transfer (section 4) is a negative result on public aggregate data: no transferred model beat the constant rate on the "
        "held-out problems, so no public-data population parameter was carried into demo mode.\n"
        "* The QQ-equality check (section 3) covers one verified published table without respondent counts; no significance statement is possible "
        "and the other two tables are unverified transcriptions.\n"
        "* Phase 4 on real data (section 6) is the only pre-registered test of the Q-models against the classical baselines; until it has run, "
        "there is no evidence for or against the Q-models on real decisions.\n"
        "* All real datasets in hand are observational lottery or survey data; nothing in this report supports a causal claim, and every "
        "intervention effect on the dashboard is a modelled counterfactual under the served model."
    )
    return "\n".join(parts)


def _sec_claims() -> str:
    return (
        "### Today (synthetic demo mode)\n\n"
        "* Every screen carries the synthetic banner: the book, the served model and every number derived from them are synthetic "
        "(section 7); the dashboard can claim to *demonstrate* the product end to end, not to describe any investor.\n"
        "* It can claim that the pipeline is reproducible (`make` targets per phase, fixed seeds, tests for unitarity, normalisation and "
        "parameter recovery) and that every displayed number traces to a model parameter, an artifact version and a data source.\n"
        "* It can claim, on synthetic data only, that the model-selection procedure picks the generating family at the sample size stated "
        "in section 2 (with the caveat quoted there), which is what the transparency page shows as \"responses needed for calibration\".\n"
        "* It can show the Q-model quantities (interference terms, order effects, dephasing rate) as *definitions applied to a synthetic "
        "fit*, not as findings about people.\n"
        "* It must say that no public-data parameter is in demo mode, because the transfer in section 4 did not beat a constant rate, and "
        "that Phase 4 has produced whatever section 6 states (the sentence there is the only real-data verdict).\n"
        "* It must not say that any investor \"will\" sell, that any intervention causes anything, or that the Q-models are better than the "
        "classical baselines on real data.\n\n"
        "### After real data\n\n"
        "* When consented intake responses reach the calibration size of section 2, the retrain path (`make retrain`) refits the active model "
        "on real rows, promotes it only if held-out NLL does not degrade, and the transparency page can show held-out NLL against the "
        "classical baselines on real responses; per-client profiles then rest on real answers with intervals from the fit.\n"
        "* When Phase 4 has run on CPC18 (section 6), the dashboard can print the pre-registered verdict verbatim and serve whichever "
        "model won on held-out NLL, naming it when that is a classical model; even a positive verdict is about the experienced-loss analog "
        "described in section 8, and the dashboard must say so.\n"
        "* Claims about later real behaviour (RQ3) need a dataset that links stated preferences to later actions within a person: the gated "
        "panels (E, H) or the dashboard's own intervention log over time; nothing in hand supports them today.\n"
        "* Claims about interventions remain predicted effects under a model until an experiment with random assignment tests them."
    )


def _sec_next_steps(src: Sources) -> str:
    parts: list[str] = []
    gaps = src.read(src.inputs["data_gaps"], record_missing=False)
    if gaps is not None:
        tables = []
        for lvl, title, body in _split(gaps):
            if _first_table(body) is not None:
                tables.append((title, body))
        if tables:
            parts.append(f"Table(s) from `{src.label(src.inputs['data_gaps'])}` (verbatim):\n")
            for title, body in tables:
                if title:
                    parts.append(f"### {title}\n")
                parts.append("\n".join(_strip_blank(body)) + "\n")
        else:
            parts.append(f"`{src.label(src.inputs['data_gaps'])}` exists but holds no table; it is reproduced in full in section 8.")
        return "\n".join(parts)
    parts.append(src.na(src.inputs["data_gaps"]))
    parts.append(
        "\nUntil that file exists, the costed plan is the one recorded in-repo (no figure is computed here):\n\n"
        f"* Paid-panel design and its cost worksheet: `{src.label(src.inputs['prolific'])}`"
        + ("" if src.path(src.inputs["prolific"]).is_file() else f" {src.na(src.inputs['prolific'])}")
        + " (shelved; no money is spent without a GATE approval, CLAUDE.md rule 4).\n"
        f"* Registration steps for the gated panels E (UAS) and H (LISS/DNB): `{src.label(src.inputs['registration'])}`"
        + ("" if src.path(src.inputs["registration"]).is_file() else f" {src.na(src.inputs['registration'])}")
        + " (a GATE: not registered).\n"
        "* Sample size: the N_target and \"responses needed for calibration\" of section 2, with the caveat quoted there.\n"
        "* Owner downloads for the datasets whose loaders are written but untested on the real file: the `instructions` rows of the data map (section 1) and `data/CATALOG.md`."
    )
    plan = src.read(src.inputs["plan"], record_missing=False)
    formula = _paragraph_starting_with(plan, "* Paid-panel cost formula") if plan is not None else None
    if formula:
        parts.append("\nCost formula carried in `PLAN.md` section 9 (verbatim):\n\n" + _quote(formula))
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


@dataclass
class ReportBuild:
    markdown: str
    sources: Sources
    phase4_included: bool
    generated_at: str

    def appendix_markdown(self, report_label: str = DEFAULT_OUT) -> str:
        lines = [
            f"# Source files of `{report_label}`",
            "",
            f"Generated {self.generated_at} by `python -m bre.report`. Every file read while assembling the report, with its "
            "modification time (UTC), size and sha256; the report contains no number that is not in one of these files.",
            "",
            _table(["file", "modified (UTC)", "bytes", "sha256"], [[f"`{k}`", v["modified_utc"], str(v["bytes"]), v["sha256"]] for k, v in sorted(self.sources.used.items())]),
            "",
            "## Files behind the synthetic profile JSON",
            "",
        ]
        text = None
        try:
            text = self.sources.path(self.sources.inputs["profiles_json"]).read_text(encoding="utf-8")
            data = json.loads(text)
            inputs = data.get("inputs", []) if isinstance(data, dict) else []
        except (OSError, json.JSONDecodeError):
            inputs = []
        if inputs:
            lines.append(_table(["file", "modified (UTC)", "bytes", "sha256"], [[f"`{i.get('file')}`", str(i.get("modified_utc")), str(i.get("bytes")), str(i.get("sha256"))] for i in inputs]))
        else:
            lines.append(NA.format(path=self.sources.inputs["profiles_json"]) if text is None else "(the JSON lists no inputs)")
        lines += ["", "## Inputs not available", ""]
        lines += [f"* {NA.format(path=m)}" for m in self.sources.missing] or ["* none"]
        lines.append("")
        return "\n".join(lines)


def build(phase4_dir: str | Path | None = None, *, root: str | Path | None = None, profiles_json: str | Path | None = None) -> ReportBuild:
    """Assemble the report; returns the markdown and the list of files read / not available."""
    root_path = Path(root).resolve() if root is not None else PROJECT_ROOT
    src = Sources(root_path)
    if profiles_json is not None:  # the only input a caller may redirect
        src.inputs["profiles_json"] = str(profiles_json)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    phase4_text, phase4_included = _sec_phase4(src, phase4_dir)
    sections = {
        SECTION_TITLES[0]: _sec_data_map(src),
        SECTION_TITLES[1]: _sec_recovery(src),
        SECTION_TITLES[2]: _sec_q1(src),
        SECTION_TITLES[3]: _sec_transfer(src),
        SECTION_TITLES[4]: _sec_one_table(src, phase4_included),
        SECTION_TITLES[5]: phase4_text,
        SECTION_TITLES[6]: _sec_profiles(src),
        SECTION_TITLES[7]: _sec_limitations(src),
        SECTION_TITLES[8]: _sec_claims(),
        SECTION_TITLES[9]: _sec_next_steps(src),
        SECTION_TITLES[10]: (
            f"Every file read while assembling this report, with its modification time, size and sha256, and every input that was not "
            f"available, is listed in `reports/{APPENDIX_NAME}` (written next to this file by `python -m bre.report`)."
        ),
    }
    head = [
        "# Behavioral Risk Engine — Phase 8 report",
        "",
        f"Generated {generated_at} by `python -m bre.report` ({_git_line(root_path)}). Assembled from generated files only: every "
        f"number is copied from a file listed in `reports/{APPENDIX_NAME}`; nothing is typed by hand (CLAUDE.md rule 6). Results on "
        f"synthetic data are labelled {SYNTHETIC}; real-data results carry that statement in their source. An input that could not "
        "be read is marked \"(not available: <path>)\" where its content would appear.",
        "",
        "Phase 4 folder: " + (f"`{src.label(phase4_dir)}`" if phase4_included else "none (section 6 states the consequence)") + ".",
        "",
        "## Contents",
        "",
    ]
    head += [f"{i + 1}. {t.split('. ', 1)[1]}" for i, t in enumerate(SECTION_TITLES)]
    body = []
    for title, text in sections.items():
        body += ["", f"## {title}", "", text.rstrip("\n"), ""]
    markdown = "\n".join(head + body).rstrip("\n") + "\n"
    return ReportBuild(markdown=markdown, sources=src, phase4_included=phase4_included, generated_at=generated_at)


def render(phase4_dir: str | Path | None = None, *, root: str | Path | None = None, profiles_json: str | Path | None = None) -> str:
    """The report text (nothing is written). ``phase4_dir`` is a ``reports/phase4/<name>`` folder;
    None (or an incomplete folder) yields the :data:`PHASE4_ABSENT` sentence as the whole section."""
    return build(phase4_dir, root=root, profiles_json=profiles_json).markdown


# ---------------------------------------------------------------------------------------------
# Synthetic profile examples (the only step that touches a model; no fitting)
# ---------------------------------------------------------------------------------------------


def _active_artifact(db_path: Path):
    """The artifact of the active ``model_registry`` row, read from the SQLite file directly
    (read-only; no ORM session, so nothing can be written)."""
    import sqlite3

    from bre import predict as P

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT version, training_data_refs, metrics FROM model_registry WHERE is_active ORDER BY promoted_at DESC").fetchall()
    finally:
        con.close()
    if not rows:
        raise LookupError(f"{db_path}: model_registry has no active model")
    version, refs, metrics = rows[0]
    row = {"version": version, "training_data_refs": json.loads(refs) if isinstance(refs, str) else refs, "metrics": json.loads(metrics) if isinstance(metrics, str) else metrics}
    path = P.artifact_path_of(row)
    if path is None:
        raise LookupError(f"{db_path}: active row {version!r} names no artifact directory")
    art = P.load_artifact(path)
    if art.version != version:
        raise LookupError(f"artifact at {path} is version {art.version!r}, registry row is {version!r}")
    return art, P.resolve_artifact_dir(path)


def _file_entry(root: Path, p: Path) -> dict[str, Any]:
    st = p.stat()
    try:
        label = p.resolve().relative_to(root).as_posix()
    except ValueError:
        label = str(p)
    return {"file": label, "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"), "bytes": st.st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def generate_profiles(db_path: str | Path | None = None, out: str | Path = PROFILES_JSON, *, root: str | Path | None = None, clients_json: str | Path | None = None) -> dict[str, Any]:
    """Score the five synthetic archetypes with the demo database's active model through
    :mod:`bre.predict` (``profile``, ``predict_sell`` at the design losses, the typical-crisis
    grid for the drawdown capacity) and write ``out`` with ``is_synthetic: true``. Refuses to run
    when the artifact was not trained on synthetic data or a client is not flagged synthetic
    (CLAUDE.md rule 3); nothing is fitted."""
    import pandas as pd

    from bre import design as D
    from bre import predict as P

    root_path = Path(root).resolve() if root is not None else PROJECT_ROOT
    db = Path(db_path) if db_path is not None else root_path / DEMO_DB
    if not db.is_absolute():
        db = root_path / db
    if not db.is_file():
        raise FileNotFoundError(f"demo database not found: {db} (seed it with `python -m bre.demo --seed-db` or `make api`)")
    cj = Path(clients_json) if clients_json is not None else root_path / INPUTS["clients_json"]
    if not cj.is_absolute():
        cj = root_path / cj
    art, art_dir = _active_artifact(db)
    if not art.is_synthetic_training:
        raise RuntimeError(f"active model {art.version!r} is not flagged is_synthetic_training; the synthetic profile examples require a synthetic-trained model")

    if cj.is_file():
        book = json.loads(cj.read_text(encoding="utf-8"))
        clients = book["clients"] if isinstance(book, dict) else book
        clients_source = cj.relative_to(root_path).as_posix() if cj.is_relative_to(root_path) else str(cj)
    else:  # fall back to the demo database's clients table
        import sqlite3

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(clients)")]
            clients = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM clients")]
        finally:
            con.close()
        clients_source = f"clients table of {db}"
    if not all(bool(c.get("is_synthetic")) for c in clients):
        raise RuntimeError(f"{clients_source}: a client is not flagged is_synthetic; refusing")
    archetypes = [c for c in clients if c.get("archetype")]
    if not archetypes:
        raise LookupError(f"{clients_source}: no archetype clients (rows with a non-null `archetype`)")
    try:
        from bre.demo import ARCHETYPES

        descriptions = {k: v.get("description", "") for k, v in ARCHETYPES.items()}
    except Exception:  # pragma: no cover - descriptions are optional
        descriptions = {}

    book_df = pd.DataFrame([{"client_id": c["client_id"], "covariates": c["covariates"]} for c in clients])
    grid = [float(x) for x in D.LOSS_PCTS]
    magnitudes = sorted(abs(x) for x in grid)
    sc = P.scorer_for(art)
    profiles = []
    r2: dict[str, Any] | None = None
    for c in archetypes:
        cid = str(c["client_id"])
        cov = c["covariates"]
        prof = P.profile(art, cov, None, subject_id=cid, book=book_df if r2 is None else None)
        if r2 is None and prof.get("risk_score", {}).get("r2_vs_angle") is not None:
            r2 = {"r2": prof["risk_score"]["r2_vs_angle"], "ci95": prof["risk_score"].get("r2_ci95"), "note": prof["risk_score"].get("r2_note")}
        loss_grid = []
        for L in grid:
            r = P.predict_sell(art, cov, L, (), D.SCENARIO_FIRST, None, subject_id=cid)
            loss_grid.append({
                "loss_pct": L, "p": r["p"], "ci80": r["ci80"], "ci95": r["ci95"], "n_draws": r.get("n_draws"), "interference": r["interference"],
                "top_driver": r["top_driver"], "n_calibration": r["n_calibration"], "known_subject": r["scenario"]["known_subject"], "is_synthetic": True,
            })
        cov_text = P.covariates_text(cov)
        scen = [P.Scenario(cid, cov_text, -L, P.TYPICAL_CRISIS_CONTEXTS, D.SCENARIO_FIRST, None, tag=f"cap:{L}") for L in magnitudes]
        p_crisis = sc.predict(scen)
        cap, cap_status = P.drawdown_capacity(p_crisis, P.DRAWDOWN_TARGET)
        truth = {k: c[k] for k in c if str(k).startswith("truth_")}
        cov_rec = json.loads(cov) if isinstance(cov, str) else cov
        profiles.append({
            "client_id": cid, "display_label": c.get("display_label"), "archetype": c["archetype"], "is_synthetic": True,
            "source": c.get("source"), "description": descriptions.get(c["archetype"], ""), "covariates": cov_rec, "truth": truth,
            "profile": prof, "loss_grid": loss_grid,
            "crisis_grid": {"contexts": list(P.TYPICAL_CRISIS_CONTEXTS), "losses": magnitudes, "p": [float(v) for v in p_crisis], "question_order_id": D.SCENARIO_FIRST, "is_synthetic": True},
            "drawdown_capacity": {"capacity": cap, "status": cap_status, "target": P.DRAWDOWN_TARGET, "is_synthetic": True},
        })
    art_path = Path(art_dir)
    inputs = [_file_entry(root_path, p) for p in [db, *(sorted(art_path.glob("*")) if art_path.is_dir() else [])] if p.is_file()]
    if cj.is_file():
        inputs.append(_file_entry(root_path, cj))
    payload = {
        "is_synthetic": True,
        "label": SYNTHETIC,
        "note": "Predictions of a synthetic-trained model for synthetic archetype clients; every number is SYNTHETIC and traces to the artifact below.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "python -m bre.report (bre.report.generate_profiles): bre.predict.profile + predict_sell + Scorer.predict + drawdown_capacity; no fitting",
        "db": str(db),
        "clients_source": clients_source,
        "artifact": {
            "version": art.version, "model_name": art.model_name, "family": art.family, "path": str(art_path), "n_params": art.n_params,
            "n_samples": art.n_samples, "created_at": art.created_at, "is_synthetic_training": bool(art.is_synthetic_training), "notes": art.notes,
            "training_data_refs": list(art.training_data_refs), "design_version": art.design_version,
            "calibrated_contexts": {t: {"status": e.get("status"), "n_responses": e.get("n_responses")} for t, e in art.calibrated_contexts.items()},
        },
        "book_loss_response": art.metrics.get("loss_response") if isinstance(art.metrics, dict) else None,
        "risk_score_r2": r2,
        "loss_levels": grid,
        "risk_score_loss": P.RISK_SCORE_LOSS,
        "typical_crisis_contexts": list(P.TYPICAL_CRISIS_CONTEXTS),
        "drawdown_target": P.DRAWDOWN_TARGET,
        "capacity_definition": P.CAPACITY_DEFINITION.format(grid=magnitudes, contexts=list(P.TYPICAL_CRISIS_CONTEXTS), target=P.DRAWDOWN_TARGET),
        "profiles": profiles,
        "inputs": inputs,
    }
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return payload


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bre.report", description="Assemble reports/REPORT.md from generated files (Phase 8). No number is typed by hand.")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path, relative to bre/ (default {DEFAULT_OUT}); the appendix {APPENDIX_NAME} is written next to it")
    ap.add_argument("--phase4", default=None, metavar="DIR", help="reports/phase4/<name> holding verdict.json, table.md and structural.md; absent = the fixed 'not run' sentence")
    ap.add_argument("--profiles-json", default=PROFILES_JSON, help=f"where the synthetic archetype profiles are written and read (default {PROFILES_JSON})")
    ap.add_argument("--no-profiles", action="store_true", help="do not regenerate the synthetic profile JSON (render whatever exists)")
    ap.add_argument("--db", default=None, help=f"SQLite file of the demo database whose active model scores the archetypes (default {DEMO_DB})")
    args = ap.parse_args(argv)

    if not args.no_profiles:
        try:
            generate_profiles(args.db, args.profiles_json)
            print(f"[report] synthetic archetype profiles written to {args.profiles_json}")
        except Exception as e:  # graceful: the section shows the marker
            print(f"[report] synthetic profiles not regenerated ({type(e).__name__}: {e}); rendering whatever exists at {args.profiles_json}", file=sys.stderr)

    result = build(args.phase4, profiles_json=args.profiles_json)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.markdown, encoding="utf-8")
    appendix = out.parent / APPENDIX_NAME
    try:
        report_label = out.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        report_label = str(out)
    appendix.write_text(result.appendix_markdown(report_label), encoding="utf-8")
    print(f"[report] wrote {out} ({len(SECTION_TITLES)} sections, {len(result.sources.used)} source files) and {appendix}")
    print("[report] Phase 4: " + ("included from " + str(args.phase4) if result.phase4_included else PHASE4_ABSENT))
    for m in result.sources.missing:
        print(f"[report] {NA.format(path=m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
