"""BRE intake-battery exports -> DecisionEvent rows (instrument/README.md, battery.json ``output``).

Input: ``data/raw/intake_pilot/*.json`` written by the static instrument's "Download responses"
button (``intake_<session_id>.json``, a JSON array of row objects with exactly the 21 schema
fields in order). The sibling ``intake_<session_id>.session.json`` logs are not loaded but are
read, when present, for the ``delay_override_active`` flag. An absent or empty folder yields an
empty frame.

Pre-specified pilot handling (instrument/README.md, "Pre-specified handling"), applied here:

* a session is complete when its file has ``n_presentations x 3`` rows for its form (the forms
  and their ``n_presentations`` are read from ``instrument/battery.json`` when it is available,
  otherwise the frozen v1.0.0 values ``full: 14``, ``short: 5``); incomplete sessions are kept in
  ``data/raw/`` but not loaded (counted as dropped);
* sessions whose log has ``delay_override_active = true`` are test runs and are excluded;
* duplicate files (same ``session_id``) are loaded once (the first in file-name order);
* ``consent_training`` is kept on every row (fits exclude ``False``; the validator warns).

Contract checks per file (battery.json ``output.row_rules``; a file that fails is skipped and
counted, never silently loaded): every row has exactly the 21 fields in order; ``dataset ==
"intake_battery"``; ``battery_version`` in :data:`KNOWN_BATTERY_VERSIONS`; ``is_synthetic`` and
``incentivized`` are ``false``; ``outcome_behavior`` is ``null``; the file's ``session_id`` is one
value. Timestamps are kept as the ISO-8601 UTC strings the browser wrote; ``covariates`` objects
are serialized to canonical JSON; ``context_tags`` / ``prior_question_ids`` stay lists.
``source_row_ref`` is the instrument's own reference (``<form>:i<NN>[:rep_of_i<MM>]:<question>``)
prefixed with the file name, so a row still traces to ``file:row``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, summarize
from bre.schema import COLUMNS, PROJECT_ROOT, ValidationReport, conform_frame, empty_frame, write_events

DATASET = "intake_battery"
INTAKE_SUBDIR = "intake_pilot"
KNOWN_BATTERY_VERSIONS: tuple[str, ...] = ("1.0.0",)
BATTERY_JSON = PROJECT_ROOT / "instrument" / "battery.json"
DEFAULT_N_PRESENTATIONS: dict[str, int] = {"full": 14, "short": 5}
"""Frozen v1.0.0 values, used when instrument/battery.json is not readable."""
ROWS_PER_PRESENTATION = 3
"""tolerance, sell_hold and allocation_share rows per item presentation."""


def n_presentations_by_form(battery_path: Path = BATTERY_JSON) -> dict[str, int]:
    """``{form_id: n_presentations}`` from battery.json, else :data:`DEFAULT_N_PRESENTATIONS`."""
    try:
        forms = json.loads(Path(battery_path).read_text(encoding="utf-8"))["forms"]
        return {k: int(v["n_presentations"]) for k, v in forms.items()}
    except (OSError, KeyError, TypeError, ValueError):
        return dict(DEFAULT_N_PRESENTATIONS)


def _contract_problem(rows: object) -> str | None:
    """Return a reason when ``rows`` violates the battery output contract, else None."""
    if not isinstance(rows, list) or not rows:
        return "not a non-empty JSON array"
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or list(r.keys()) != COLUMNS:
            return f"row {i}: keys differ from the 21 schema fields in order"
        if r["dataset"] != DATASET:
            return f"row {i}: dataset is {r['dataset']!r}"
        if r["battery_version"] not in KNOWN_BATTERY_VERSIONS:
            return f"row {i}: unknown battery_version {r['battery_version']!r}"
        if r["is_synthetic"] is not False or r["incentivized"] is not False:
            return f"row {i}: is_synthetic / incentivized must be false"
        if r["outcome_behavior"] is not None:
            return f"row {i}: outcome_behavior is not null"
        if not isinstance(r["consent_training"], bool):
            return f"row {i}: consent_training is not a boolean"
    if len({r["session_id"] for r in rows}) != 1:
        return "rows carry more than one session_id"
    return None


def _form_of(rows: list[dict]) -> str:
    ref = str(rows[0].get("source_row_ref", ""))
    return ref.split(":", 1)[0] if ":" in ref else ""


def load_intake_battery_with_meta(raw_dir: Path = RAW_DIR, battery_path: Path = BATTERY_JSON) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the intake-battery DecisionEvent frame (possibly empty) and a meta dict."""
    folder = Path(raw_dir) / INTAKE_SUBDIR
    files = sorted(p for p in folder.glob("*.json") if not p.name.endswith(".session.json")) if folder.is_dir() else []
    n_pres = n_presentations_by_form(battery_path)
    meta: dict[str, Any] = {
        "dataset": DATASET,
        "raw_file": f"{INTAKE_SUBDIR}/*.json",
        "n_files": len(files),
        "n_sessions_loaded": 0,
        "sessions_by_form": {},
        "dropped": {},
        "skipped_files": [],
        "notes": [],
    }
    dropped: dict[str, int] = {}
    seen_sessions: set[str] = set()
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            meta["skipped_files"].append(f"{path.name}: unreadable ({exc})")
            dropped["file unreadable"] = dropped.get("file unreadable", 0) + 1
            continue
        problem = _contract_problem(rows)
        if problem:
            meta["skipped_files"].append(f"{path.name}: {problem}")
            dropped["file violates the battery output contract"] = dropped.get("file violates the battery output contract", 0) + len(rows) if isinstance(rows, list) else dropped.get("file violates the battery output contract", 0) + 1
            continue
        session_id = str(rows[0]["session_id"])
        if session_id in seen_sessions:
            dropped["duplicate session file (same session_id) loaded once"] = dropped.get("duplicate session file (same session_id) loaded once", 0) + len(rows)
            continue
        form = _form_of(rows)
        expected = n_pres.get(form)
        if expected is None or len(rows) != expected * ROWS_PER_PRESENTATION:
            key = f"incomplete session (rows != n_presentations x {ROWS_PER_PRESENTATION} for form {form or '?'})"
            dropped[key] = dropped.get(key, 0) + len(rows)
            continue
        log_path = path.with_name(path.stem + ".session.json")
        if log_path.exists():
            try:
                log = json.loads(log_path.read_text(encoding="utf-8"))
                if isinstance(log, dict) and log.get("delay_override_active") is True:
                    dropped["test run (session log has delay_override_active=true)"] = dropped.get("test run (session log has delay_override_active=true)", 0) + len(rows)
                    continue
            except (OSError, ValueError):
                meta["notes"].append(f"{log_path.name}: session log unreadable; session kept")
        seen_sessions.add(session_id)
        meta["sessions_by_form"][form] = meta["sessions_by_form"].get(form, 0) + 1
        df = pd.DataFrame(rows, columns=COLUMNS)
        df["source_row_ref"] = [f"{path.name}:{r}" for r in df["source_row_ref"].tolist()]
        frames.append(conform_frame(df))
    meta["n_sessions_loaded"] = len(seen_sessions)
    meta["dropped"] = dropped
    frame = conform_frame(pd.concat(frames, ignore_index=True)) if frames else empty_frame()
    if not files:
        meta["notes"].append(f"no intake files under {folder}")
    meta.update(summarize(frame))
    return frame, meta


def load_intake_battery(raw_dir: Path = RAW_DIR, battery_path: Path = BATTERY_JSON) -> pd.DataFrame:
    """Intake-battery exports as a DecisionEvent frame (empty when there are none)."""
    return load_intake_battery_with_meta(raw_dir, battery_path)[0]


def write_intake_battery(out_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR, battery_path: Path = BATTERY_JSON) -> ValidationReport:
    """Write ``intake_battery.parquet`` (+ validation report) and return the report."""
    return write_events(load_intake_battery(raw_dir, battery_path), Path(out_dir) / f"{DATASET}.parquet")
