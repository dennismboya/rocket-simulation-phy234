"""Phase 8: ``bre.report`` assembles reports/REPORT.md from generated files only.

Fast and read-only: ``render`` reads the repository's generated files (no fitting, no database
writes). The Phase 4 fixture under ``tests/fixtures/report/phase4`` is labelled FIXTURE in every
file and is included verbatim only when its folder is passed explicitly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bre import report as R

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "report" / "phase4"
BANNED = ("will sell", "quantum advantage")


def _content_lines(text: str) -> list[str]:
    """Non-empty lines that are not markdown headings (headings are demoted when embedded)."""
    return [ln for ln in text.splitlines() if ln.strip() and not re.match(r"^#{1,6}\s", ln)]


@pytest.fixture(scope="module")
def report_text() -> str:
    return R.render()


def test_section_headers_present(report_text: str):
    for title in R.SECTION_TITLES:
        assert f"## {title}" in report_text, title
    assert report_text.startswith("# Behavioral Risk Engine")
    assert "## Contents" in report_text


def test_phase4_absent_sentence_when_no_dir(report_text: str):
    assert R.PHASE4_ABSENT in report_text
    # the section holds the sentence and nothing else
    sec = report_text.split(f"## {R.SECTION_TITLES[5]}", 1)[1].split("\n## ", 1)[0]
    assert sec.strip() == R.PHASE4_ABSENT


def test_no_banned_phrases(report_text: str):
    low = report_text.lower()
    for phrase in BANNED:
        assert phrase not in low, phrase


def test_generated_inputs_are_embedded(report_text: str):
    """The verbatim sources that exist in the repo appear; the unverified Q1 tables sit under
    the label the source file gives them."""
    root = R.PROJECT_ROOT
    if (root / R.INPUTS["data_map"]).is_file():
        assert "| Dataset | Status |" in report_text
    if (root / R.INPUTS["processed_readme"]).is_file():
        assert "| dataset | catalog | file | rows |" in report_text
    if (root / R.INPUTS["recovery_summary"]).is_file():
        assert "correct_share" in report_text
        assert "N_target" in report_text
        assert "True generator x selected family" in report_text
    if (root / R.INPUTS["status"]).is_file() and "CAVEAT" in (root / R.INPUTS["status"]).read_text(encoding="utf-8"):
        assert "> " in report_text and "CAVEAT" in report_text
    if (root / R.INPUTS["q1"]).is_file():
        assert "### Verified table results" in report_text
        assert "### UNVERIFIED secondary transcription" in report_text
        assert "clinton_gore — VERIFIED" in report_text
    if (root / R.INPUTS["transfer"]).is_file():
        assert "held-out problems" in report_text
        assert "| target | model | family |" in report_text
    if (root / R.INPUTS["plan"]).is_file():
        assert "Real-data mapping for split (b)" in report_text
    if (root / R.INPUTS["predict_src"]).is_file():
        assert "need not be monotone" in report_text
        assert "CAPACITY_DEFINITION" in report_text


def test_profiles_section_is_labelled_synthetic(report_text: str):
    sec = report_text.split(f"## {R.SECTION_TITLES[6]}", 1)[1].split("\n## ", 1)[0]
    path = R.PROJECT_ROOT / R.INPUTS["profiles_json"]
    if not path.is_file():
        assert R.NA.format(path=R.INPUTS["profiles_json"]) in sec
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["is_synthetic"] is True
    assert sec.count("SYNTHETIC") >= 5
    for p in data["profiles"]:
        assert p["client_id"] in sec
        assert p["is_synthetic"] is True


def test_phase4_fixture_included_verbatim():
    out = R.render(phase4_dir=FIXTURE_DIR)
    assert R.PHASE4_ABSENT not in out
    assert "FIXTURE" in out
    for name in R.PHASE4_FILES:
        text = (FIXTURE_DIR / name).read_text(encoding="utf-8")
        if name.endswith(".json"):
            assert text.rstrip("\n") in out
        else:
            for ln in _content_lines(text):
                assert ln in out, (name, ln)
            for m in re.finditer(r"^#{1,6}\s+(.*)$", text, flags=re.M):
                assert m.group(1) in out, (name, m.group(1))
    verdict = json.loads((FIXTURE_DIR / "verdict.json").read_text(encoding="utf-8"))["verdict"]
    assert f"**Verdict (copied from `verdict.json`):** {verdict}" in out
    low = out.lower()
    for phrase in BANNED:
        assert phrase not in low


def test_phase4_incomplete_dir_falls_back_to_sentence(tmp_path: Path):
    d = tmp_path / "phase4-partial"
    d.mkdir()
    (d / "table.md").write_text("# FIXTURE partial\n", encoding="utf-8")
    out = R.render(phase4_dir=d)
    assert R.PHASE4_ABSENT in out
    assert "FIXTURE partial" not in out
    out2 = R.render(phase4_dir=tmp_path / "does-not-exist")
    assert R.PHASE4_ABSENT in out2


def test_missing_inputs_are_marked_not_raised(tmp_path: Path):
    build = R.build(root=tmp_path)
    out = build.markdown
    for title in R.SECTION_TITLES:
        assert f"## {title}" in out
    assert R.PHASE4_ABSENT in out
    for key in ("data_map", "recovery_summary", "q1", "transfer", "profiles_json"):
        assert R.NA.format(path=R.INPUTS[key]) in out, key
    assert R.INPUTS["data_map"] in build.sources.missing
    assert build.sources.used == {}
    appendix = build.appendix_markdown()
    assert "## Inputs not available" in appendix
    assert R.INPUTS["recovery_summary"] in appendix


def test_cli_writes_report_and_appendix(tmp_path: Path):
    out = tmp_path / "REPORT.md"
    assert R.main(["--out", str(out), "--no-profiles", "--phase4", str(FIXTURE_DIR)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "FIXTURE" in text and R.PHASE4_ABSENT not in text
    appendix = tmp_path / R.APPENDIX_NAME
    assert appendix.is_file()
    atext = appendix.read_text(encoding="utf-8")
    assert "| file | modified (UTC) | bytes | sha256 |" in atext
    for name in R.PHASE4_FILES:
        assert name in atext


def test_markdown_helpers():
    text = "# T\n\nintro\n\n## A\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n## B\n\n```\n# not a heading\n```\n"
    secs = R._split(text)
    assert [(lvl, title) for lvl, title, _ in secs] == [(0, ""), (1, "T"), (2, "A"), (2, "B")]
    assert R._first_table(R._body(text, 2, "A")) == (["x", "y"], [["1", "2"]])
    demoted = R._demote(text, 2)
    assert "### T" in demoted and "#### A" in demoted and "\n# not a heading\n" in demoted
    assert R._fmt(None) == "n/a" and R._fmt([0.1, 0.25]) == "[0.100, 0.250]" and R._fmt(3) == "3"
