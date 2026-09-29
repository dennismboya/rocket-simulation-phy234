"""choices13k aggregate choice rates -> DecisionEvent rows (CATALOG.md entry A).

Source files (``data/raw/choices13k/``): ``c13k_selections.csv`` (14,568 rows: 13,006 problems x
feedback condition / block), ``c13k_problems.json`` (full outcome distributions of both gambles,
keyed by the 0-based row index of the selections file) and ``choices13k_thomas2024_features.csv``
(Thomas et al. 2024 engineered features with the dominance flags ``Dom`` (first-order), ``SOSD``,
``TOSD`` and ``StochDom``; same row order as the selections file, which the loader asserts on
``Ha, pHa, La, Hb, pHb, Lb, Block, Feedback`` for every row).

Mapping (CATALOG.md A):

* ``dataset = "choices13k"``; one aggregate row per CSV row with
  ``subject_id = "agg:<Problem>:<Feedback>:<Block>"`` (``Feedback`` rendered as the CSV's
  ``True``/``False``), ``session_id = "agg"``, ``position_in_session = 0`` (the pseudo-subject is
  unique per row, so the key is unique), ``scenario_id = <Problem>``.
* ``elicitation_type = "choice_rate"``; ``response = bRate`` (share choosing gamble B).
* ``loss_pct = null`` (no portfolio-loss framing). ``covariates = {"weight": n,
  "x_worst_outcome_rel": min outcome / max |outcome| over the outcomes of both gambles with
  probability > 0}`` (the catalog's ``worst_outcome_rel``, carried with the ``x_`` prefix the
  schema requires for dataset-specific keys).
* ``context_tags = ["feedback:on"|"feedback:off", "block:<k>"]``; ``incentivized = True``;
  ``is_synthetic = False``; ``source_row_ref = "c13k_selections.csv:<0-based data row>"`` (the
  same index keys ``c13k_problems.json``).

Dominance filter (``drop_dominated=True``, the default): the catalog says the loader "filters rows
where the Thomas-2024 file flags stochastic dominance and reports the count". The literal reading
uses the ``StochDom`` column, which is non-zero when any of first-, second- or third-order
dominance holds (``StochDom == sign * order``); this removes about 60% of the rows. Because
second- and third-order dominated problems still carry risk-preference information, the flag is
an option: ``dominance_flag="Dom"`` (default: first-order dominance, per CATALOG.md), ``"StochDom"`` (any order,
dominance only), ``"SOSD"`` or ``"TOSD"``. The number of rows dropped, and the counts per flag,
are returned in the meta so the choice can be revisited by the main session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, assemble, dumps, summarize
from bre.schema import ValidationReport, write_events

DATASET = "choices13k"
SELECTIONS_FILE = "c13k_selections.csv"
PROBLEMS_FILE = "c13k_problems.json"
FEATURES_FILE = "choices13k_thomas2024_features.csv"
DOMINANCE_FLAGS = ("StochDom", "Dom", "SOSD", "TOSD")
_JOIN_COLUMNS = ["Ha", "pHa", "La", "Hb", "pHb", "Lb"]


def worst_outcome_rel(problem: dict) -> float:
    """min outcome / max |outcome| over both gambles' outcomes with probability > 0 (0.0 if all zero)."""
    outs = [float(o) for opt in ("A", "B") for p, o in problem[opt] if float(p) > 0]
    if not outs:
        outs = [float(o) for opt in ("A", "B") for _, o in problem[opt]]
    m = max(abs(o) for o in outs)
    return min(outs) / m if m > 0 else 0.0


def read_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Read the selections CSV, the problems JSON and the Thomas-2024 features (row-aligned)."""
    base = Path(raw_dir) / "choices13k"
    sel = pd.read_csv(base / SELECTIONS_FILE)
    with open(base / PROBLEMS_FILE, encoding="utf-8") as fh:
        problems = json.load(fh)
    feats = pd.read_csv(base / FEATURES_FILE, index_col=0)
    if len(feats) != len(sel):
        raise ValueError(f"{FEATURES_FILE} has {len(feats)} rows, {SELECTIONS_FILE} has {len(sel)}")
    for c in _JOIN_COLUMNS:
        if not np.allclose(sel[c].to_numpy(dtype=float), feats[c].to_numpy(dtype=float)):
            raise ValueError(f"{FEATURES_FILE} is not row-aligned with {SELECTIONS_FILE} on {c}")
    if not np.array_equal(sel["Block"].to_numpy(), feats["Block"].to_numpy().astype(int)):
        raise ValueError(f"{FEATURES_FILE} is not row-aligned with {SELECTIONS_FILE} on Block")
    if not np.array_equal(sel["Feedback"].astype(int).to_numpy(), feats["Feedback"].to_numpy().astype(int)):
        raise ValueError(f"{FEATURES_FILE} is not row-aligned with {SELECTIONS_FILE} on Feedback")
    if len(problems) != len(sel):
        raise ValueError(f"{PROBLEMS_FILE} has {len(problems)} entries, {SELECTIONS_FILE} has {len(sel)} rows")
    return sel, problems, feats


def load_choices13k_with_meta(
    raw_dir: Path = RAW_DIR, drop_dominated: bool = True, dominance_flag: str = "Dom"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the choices13k DecisionEvent frame and a meta dict (dropped rows and why)."""
    if dominance_flag not in DOMINANCE_FLAGS:
        raise ValueError(f"dominance_flag must be one of {DOMINANCE_FLAGS}; got {dominance_flag!r}")
    sel, problems, feats = read_raw(raw_dir)
    sel = sel.copy()
    sel["raw_row"] = np.arange(len(sel), dtype="int64")
    flag_counts = {f: int((feats[f].fillna(0).to_numpy() != 0).sum()) for f in DOMINANCE_FLAGS}
    dominated = feats[dominance_flag].fillna(0).to_numpy() != 0
    n_dropped = 0
    if drop_dominated:
        n_dropped = int(dominated.sum())
        sel = sel[~dominated].reset_index(drop=True)
    n = len(sel)
    rows = sel["raw_row"].tolist()
    wor = [worst_outcome_rel(problems[str(r)]) for r in rows]
    # the JSON entry must describe the same gamble A as the CSV row (Ha appears among A's outcomes)
    for r, ha in zip(rows[:2000], sel["Ha"].tolist()[:2000]):
        outs = [float(o) for _, o in problems[str(r)]["A"]]
        if not any(abs(o - float(ha)) < 1e-9 for o in outs):
            raise ValueError(f"{PROBLEMS_FILE} entry {r} does not match {SELECTIONS_FILE} row {r} (Ha={ha})")
    feedback = sel["Feedback"].astype(bool).tolist()
    block = sel["Block"].astype(int).tolist()
    frame = assemble(
        {
            "subject_id": [f"agg:{p}:{fb}:{b}" for p, fb, b in zip(sel["Problem"].tolist(), feedback, block)],
            "dataset": DATASET,
            "session_id": "agg",
            "position_in_session": np.zeros(n, dtype="int64"),
            "scenario_id": sel["Problem"].astype(str).tolist(),
            "context_tags": [["feedback:on" if fb else "feedback:off", f"block:{b}"] for fb, b in zip(feedback, block)],
            "elicitation_type": "choice_rate",
            "response": sel["bRate"].to_numpy(dtype=float),
            "covariates": [dumps({"weight": int(w), "x_worst_outcome_rel": float(x)}) for w, x in zip(sel["n"].tolist(), wor)],
            "incentivized": True,
            "source_row_ref": [f"{SELECTIONS_FILE}:{r}" for r in rows],
        },
        n,
    )
    meta: dict[str, Any] = {
        "dataset": DATASET,
        "raw_file": SELECTIONS_FILE,
        "n_raw_rows": int(len(feats)),
        "drop_dominated": bool(drop_dominated),
        "dominance_flag": dominance_flag,
        "dominance_flag_counts": flag_counts,
        "dropped": ({f"{dominance_flag} != 0 (stochastically dominated problem, Thomas et al. 2024)": n_dropped} if n_dropped else {}),
        "notes": [
            f"features file row-aligned with the selections file on {', '.join(_JOIN_COLUMNS)}, Block, Feedback",
            "dominance flag counts (rows with a non-zero flag): " + ", ".join(f"{k}={v}" for k, v in flag_counts.items()),
        ],
        **summarize(frame),
    }
    return frame, meta


def load_choices13k(raw_dir: Path = RAW_DIR, drop_dominated: bool = True, dominance_flag: str = "Dom") -> pd.DataFrame:
    """choices13k aggregate rates as a DecisionEvent frame (see :func:`load_choices13k_with_meta`)."""
    return load_choices13k_with_meta(raw_dir, drop_dominated=drop_dominated, dominance_flag=dominance_flag)[0]


def write_choices13k(
    out_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR, drop_dominated: bool = True, dominance_flag: str = "Dom"
) -> ValidationReport:
    """Write ``choices13k.parquet`` (+ validation report) and return the report."""
    frame = load_choices13k(raw_dir, drop_dominated=drop_dominated, dominance_flag=dominance_flag)
    return write_events(frame, Path(out_dir) / f"{DATASET}.parquet")
