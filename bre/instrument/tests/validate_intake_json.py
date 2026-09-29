#!/usr/bin/env python
"""Check downloaded intake files (intake_<session_id>.json) against battery.json and the schema code.

Run with:  /home/user/bre-venv/bin/python bre/instrument/tests/validate_intake_json.py FILE_OR_DIR [...]

For every responses file (``*.session.json`` logs are skipped):

1. every row must carry exactly the 21 fields of ``battery.json -> output.schema_fields``, in order
   (the list is also compared with ``db.models.SCHEMA_COLUMNS`` and ``bre.schema.COLUMNS``);
2. the battery's own contract (``battery.json -> output``): ``dataset == "intake_battery"``,
   ``is_synthetic`` and ``incentivized`` false, ``outcome_behavior`` null, ``battery_version`` the
   battery's, ``consent_training`` boolean, ``scenario_id`` in the scenario universe,
   ``question_order_id`` exactly ``tolerance-first`` or ``scenario-first``, every context tag
   ``namespace:value``, ``response_time_ms`` a positive integer, ``timestamp`` ISO 8601 UTC, and per
   question (read from ``source_row_ref``) the elicitation type and coding: tolerance ->
   ``binary_yes_no`` in {0, 1} (1 = Yes), sell_hold -> ``binary_sell`` in {0, 1} (1 = Sell),
   allocation_share -> ``allocation_pct`` in [0, 1];
3. the rows are loaded into a DataFrame, passed through ``bre.schema.conform_frame`` and
   ``bre.schema.validate_frame(expect_synthetic=False)``, and every row is parsed by the
   ``bre.schema.DecisionEvent`` model (``frame_to_events``).

Step 3 requires ``binary_yes_no`` in ``bre.schema.ELICITATION_TYPES`` and ``db.models.ELICITATION_TYPES``
(PLAN.md section 2, amendment of 2026-09-17). Until both carry it, the tolerance rows fail there and
the report names the module that lags. Exit status 0 when every file passes, 1 otherwise. This is a
read-only check; it writes nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE.parent
BRE_ROOT = INSTRUMENT.parent
sys.path.insert(0, str(BRE_ROOT / "src"))
sys.path.insert(0, str(BRE_ROOT))

import pandas as pd  # noqa: E402

from bre import schema as bre_schema  # noqa: E402
from bre.schema import COLUMNS, SchemaError, conform_frame, frame_to_events, validate_frame  # noqa: E402
from db import models as db_models  # noqa: E402
from db.models import SCHEMA_COLUMNS  # noqa: E402

BATTERY: dict = json.loads((INSTRUMENT / "battery.json").read_text(encoding="utf-8"))
SCHEMA_FIELDS: list[str] = list(BATTERY["output"]["schema_fields"])
QUESTIONS: dict = BATTERY["design"]["questions"]
UNIVERSE: dict = {u["scenario_id"]: u for u in BATTERY["design"]["scenario_universe"]}
ORDER_IDS = frozenset({"tolerance-first", "scenario-first"})
CONTEXT_TAG_RE = re.compile(r"^[a-z_]+:[a-z0-9_]+$")
SRC_REF_RE = re.compile(r"^(full|short):i(\d{2})(?::rep_of_i(\d{2}))?:([a-z_]+)$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def files_from_args(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out.extend(sorted(p.glob("intake_*.json")))
        else:
            out.append(p)
    return [p for p in out if not p.name.endswith(".session.json")]


def modules_lacking(value: str) -> list[str]:
    """Names of the schema modules whose ELICITATION_TYPES does not (yet) contain ``value``."""
    return [
        name
        for name, mod in (("db.models", db_models), ("bre.schema", bre_schema))
        if value not in getattr(mod, "ELICITATION_TYPES", ())
    ]


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def response_in_range(etype: str, r: object) -> bool:
    if not _is_number(r):
        return False
    if etype in ("binary_yes_no", "binary_sell", "lottery_choice"):
        return r in (0, 1)
    if etype in ("allocation_pct", "choice_rate"):
        return 0 <= r <= 1  # type: ignore[operator]
    if etype == "likert":
        return float(r).is_integer() and 1 <= r <= 7  # type: ignore[operator]
    return False


def battery_contract_problems(name: str, rows: list[dict]) -> list[str]:
    """Row-level checks against battery.json (independent of the pipeline's schema modules)."""
    problems: list[str] = []

    def P(i: int, msg: str) -> None:
        problems.append(f"{name}: row {i}: {msg}")

    for i, r in enumerate(rows):
        if not isinstance(r, dict) or list(r.keys()) != SCHEMA_FIELDS:
            P(i, "keys differ from battery.json output.schema_fields (21 fields, in order)")
            continue
        if r["dataset"] != "intake_battery":
            P(i, f"dataset is {r['dataset']!r}, not 'intake_battery'")
        if r["is_synthetic"] is not False:
            P(i, f"is_synthetic is {r['is_synthetic']!r}, not false")
        if r["incentivized"] is not False:
            P(i, f"incentivized is {r['incentivized']!r}, not false")
        if r["outcome_behavior"] is not None:
            P(i, "outcome_behavior is not null")
        if not isinstance(r["consent_training"], bool):
            P(i, f"consent_training is {r['consent_training']!r}, not a boolean")
        if r["battery_version"] != BATTERY["battery_version"]:
            P(i, f"battery_version {r['battery_version']!r} != {BATTERY['battery_version']!r}")
        if r["scenario_id"] not in UNIVERSE:
            P(i, f"scenario_id {r['scenario_id']!r} not in battery.json scenario_universe")
        else:
            u = UNIVERSE[r["scenario_id"]]
            if r["loss_pct"] != u["loss_pct"] or r["horizon_days"] != u["horizon_days"]:
                P(i, "loss_pct / horizon_days disagree with the scenario universe entry")
            if r["context_tags"] != u["context_tags"] or r["question_order_id"] != u["question_order_id"]:
                P(i, "context_tags / question_order_id disagree with the scenario universe entry")
        if r["question_order_id"] not in ORDER_IDS:
            P(i, f"question_order_id {r['question_order_id']!r} is not tolerance-first / scenario-first")
        tags = r["context_tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) and CONTEXT_TAG_RE.match(t) for t in tags):
            P(i, f"context_tags {tags!r} are not a list of namespace:value tags")
        if not isinstance(r["prior_question_ids"], list) or not all(q in QUESTIONS for q in r["prior_question_ids"]):
            P(i, f"prior_question_ids {r['prior_question_ids']!r} are not known question ids")
        rt = r["response_time_ms"]
        if not (isinstance(rt, int) and not isinstance(rt, bool) and rt > 0):
            P(i, f"response_time_ms {rt!r} is not a positive integer")
        if not (isinstance(r["timestamp"], str) and ISO_UTC_RE.match(r["timestamp"])):
            P(i, f"timestamp {r['timestamp']!r} is not ISO 8601 UTC")
        if r["position_in_session"] != i:
            P(i, f"position_in_session {r['position_in_session']!r} is not the 0-based row index")
        m = SRC_REF_RE.match(str(r["source_row_ref"]))
        if not m:
            P(i, f"source_row_ref {r['source_row_ref']!r} does not follow output.source_row_ref_format")
            continue
        qid = m.group(4)
        q = QUESTIONS.get(qid)
        if q is None:
            P(i, f"source_row_ref names unknown question {qid!r}")
            continue
        if r["elicitation_type"] != q["elicitation_type"]:
            P(i, f"{qid} row has elicitation_type {r['elicitation_type']!r}; battery.json says {q['elicitation_type']!r}")
        if not response_in_range(r["elicitation_type"], r["response"]):
            P(i, f"response {r['response']!r} is out of range for {r['elicitation_type']!r}")
    used = {r["elicitation_type"] for r in rows if isinstance(r, dict) and "elicitation_type" in r}
    expected_used = set(BATTERY["output"]["elicitation_types_used"].values())
    if rows and used != expected_used:
        problems.append(f"{name}: elicitation types used {sorted(used)} != battery.json {sorted(expected_used)}")
    return problems


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        return [f"{path.name}: not a non-empty JSON array"]
    if SCHEMA_FIELDS != list(SCHEMA_COLUMNS):
        problems.append("battery.json output.schema_fields differs from db.models.SCHEMA_COLUMNS")
    if list(COLUMNS) != list(SCHEMA_COLUMNS):
        problems.append("bre.schema.COLUMNS differs from db.models.SCHEMA_COLUMNS")
    contract = battery_contract_problems(path.name, rows)
    problems.extend(contract)
    n_consent = sum(1 for r in rows if isinstance(r, dict) and r.get("consent_training") is True)
    print(f"{'OK   ' if not contract else 'FAIL '} {path.name}: battery contract, {len(rows)} rows, consent_training on {n_consent}")

    df = pd.DataFrame(rows, columns=list(COLUMNS))
    try:
        df = conform_frame(df)
        report = validate_frame(df, expect_synthetic=False, strict=False)
        problems.extend(f"{path.name}: {e}" for e in report.errors)
        events = frame_to_events(df)
        print(f"OK    {path.name}: schema pipeline, {len(events)} DecisionEvents, warnings={report.warnings}")
    except SchemaError as exc:
        problems.append(f"{path.name}: SchemaError: {exc}")
    except Exception as exc:  # noqa: BLE001 -- any failure is a finding
        problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return problems


def main(argv: list[str]) -> int:
    files = files_from_args(argv)
    if not files:
        print("usage: validate_intake_json.py FILE_OR_DIR [...]  (no intake_*.json found)")
        return 2
    lagging = modules_lacking("binary_yes_no")
    if lagging:
        print(
            "NOTE  binary_yes_no is not yet in ELICITATION_TYPES of: "
            + ", ".join(lagging)
            + " -- the schema-pipeline step will reject the tolerance rows until that is updated "
            "(PLAN.md section 2 amendment); the battery-contract step is unaffected"
        )
    problems: list[str] = []
    for path in files:
        problems.extend(check_file(path))
    for p in problems:
        print("FAIL  " + p)
    print(f"{len(files)} file(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
