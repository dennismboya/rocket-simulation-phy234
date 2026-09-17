"""Question-order tables -> DecisionEvent rows (CATALOG.md entry D).

Source file: ``data/raw/order_effects/question_order_tables.json``: for each question pair
(``clinton_gore`` verified; ``black_white`` and ``rose_jackson`` UNVERIFIED secondary
transcriptions, see SOURCE.txt) the eight joint proportions ``P(first answer, second answer)`` in
both orders, keyed ``AyBy, AyBn, AnBy, AnBn`` (A asked first) and ``ByAy, ByAn, BnAy, BnAn``
(B asked first), plus ``verification`` text and an ``intervening_information`` flag.

Mapping (CATALOG.md D):

* ``dataset = "order_effects:<pair>"``; ``subject_id = "agg"``; ``session_id`` = the order
  (``A_first`` / ``B_first``); ``position_in_session`` = 0..3, the cell's position within its
  order in the JSON key order above; ``scenario_id`` = the cell label (e.g. ``AyBn``).
* ``elicitation_type = "choice_rate"``; ``response`` = the joint proportion;
  ``covariates = {"weight": null, "x_verified": bool, "x_intervening_information": bool}`` where
  ``x_verified`` is True iff the ``verification`` text starts with ``VERIFIED`` (a text starting
  with ``UNVERIFIED`` is False).
* ``context_tags = ["order:A_first"|"order:B_first"]``; ``question_order_id`` = ``A_first`` /
  ``B_first``; ``prior_question_ids = []`` (a row is a joint proportion of both answers, not the
  second answer alone); ``loss_pct = null``; ``incentivized = False`` (opinion polls);
  ``is_synthetic = False``; ``source_row_ref = "question_order_tables.json:<pair>:<cell>"``.

Sample sizes are not in the source (``weight`` null). Rows of UNVERIFIED pairs carry
``x_verified = False`` and must not enter a reported result (CATALOG.md D).

Wang et al. (2014) Dataset S1 (missing, owner action in CATALOG.md D): when the owner places the
PNAS SI dataset under ``data/raw/order_effects/wang2014_si/``, :func:`load_wang2014_si` expects
one CSV with one row per survey pair and the columns ``survey`` (id), ``question_a``,
``question_b`` (wording or label), ``n_a_first``, ``n_b_first`` (respondents per order),
``AyBy, AyBn, AnBy, AnBn, ByAy, ByAn, BnAy, BnAn`` (joint proportions, rows of each order
summing to one) and ``intervening_information`` (0/1). It is mapped like the JSON tables with
``dataset = "order_effects:wang2014:<survey>"`` and ``weight`` = the order's respondent count.
Until the file exists the function returns an empty frame and the writer skips it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, assemble, dumps, summarize
from bre.schema import ValidationReport, empty_frame, write_events

DATASET_PREFIX = "order_effects"
RAW_FILE = "question_order_tables.json"
WANG2014_DIR = "wang2014_si"
CELLS: dict[str, tuple[str, ...]] = {
    "A_first": ("AyBy", "AyBn", "AnBy", "AnBn"),
    "B_first": ("ByAy", "ByAn", "BnAy", "BnAn"),
}
_VERIFIED_RE = re.compile(r"^\s*VERIFIED\b")


def read_raw(raw_dir: Path = RAW_DIR) -> dict[str, dict]:
    """Return ``{pair: table}`` from the JSON (keys starting with ``_`` are provenance, skipped)."""
    path = Path(raw_dir) / "order_effects" / RAW_FILE
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    tables = {k: v for k, v in data.items() if not k.startswith("_")}
    for pair, tab in tables.items():
        missing = [c for cells in CELLS.values() for c in cells if c not in tab]
        if missing:
            raise ValueError(f"{RAW_FILE}: pair {pair!r} lacks cells {missing}")
    return tables


def _rows_from_table(dataset: str, tab: dict, verified: bool, weight_by_order: dict[str, object], ref_prefix: str) -> dict[str, list]:
    cols: dict[str, list] = {k: [] for k in ("subject_id", "dataset", "session_id", "position_in_session", "scenario_id", "context_tags", "question_order_id", "elicitation_type", "response", "covariates", "incentivized", "source_row_ref")}
    for order, cells in CELLS.items():
        for pos, cell in enumerate(cells):
            cols["subject_id"].append("agg")
            cols["dataset"].append(dataset)
            cols["session_id"].append(order)
            cols["position_in_session"].append(pos)
            cols["scenario_id"].append(cell)
            cols["context_tags"].append([f"order:{order}"])
            cols["question_order_id"].append(order)
            cols["elicitation_type"].append("choice_rate")
            cols["response"].append(float(tab[cell]))
            cols["covariates"].append(
                dumps({"weight": weight_by_order.get(order), "x_verified": bool(verified), "x_intervening_information": bool(tab.get("intervening_information", False))})
            )
            cols["incentivized"].append(False)
            cols["source_row_ref"].append(f"{ref_prefix}:{cell}")
    return cols


def load_order_effects_with_meta(raw_dir: Path = RAW_DIR, include_unverified: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the order-effects DecisionEvent frame and a meta dict.

    ``include_unverified=False`` drops the pairs whose verification text does not start with
    ``VERIFIED`` (they are otherwise kept with ``x_verified = False``).
    """
    tables = read_raw(raw_dir)
    cols: dict[str, list] = {}
    n_pairs, n_unverified, n_dropped = 0, 0, 0
    for pair, tab in tables.items():
        verified = bool(_VERIFIED_RE.match(str(tab.get("verification", ""))))
        if not verified:
            n_unverified += 1
            if not include_unverified:
                n_dropped += 8
                continue
        n_pairs += 1
        part = _rows_from_table(f"{DATASET_PREFIX}:{pair}", tab, verified, {"A_first": None, "B_first": None}, f"{RAW_FILE}:{pair}")
        for k, v in part.items():
            cols.setdefault(k, []).extend(v)
    n = len(cols.get("subject_id", []))
    frame = assemble(cols, n) if n else empty_frame()
    meta: dict[str, Any] = {
        "dataset": DATASET_PREFIX,
        "raw_file": RAW_FILE,
        "n_pairs": int(len(tables)),
        "n_pairs_loaded": n_pairs,
        "n_pairs_unverified": n_unverified,
        "dropped": ({"unverified pair excluded (include_unverified=False)": n_dropped} if n_dropped else {}),
        "notes": [
            f"{n_unverified} of {len(tables)} pair(s) are UNVERIFIED transcriptions (x_verified=False; never in a reported result)",
            "weight is null: sample sizes are not in the source",
        ],
        **summarize(frame),
    }
    return frame, meta


def load_order_effects(raw_dir: Path = RAW_DIR, include_unverified: bool = True) -> pd.DataFrame:
    """Question-order joint proportions as a DecisionEvent frame."""
    return load_order_effects_with_meta(raw_dir, include_unverified=include_unverified)[0]


def load_wang2014_si(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Wang et al. (2014) Dataset S1 rows, or an empty frame when the SI folder is absent.

    Expected layout: see the module docstring. Validated only once the owner supplies the file.
    """
    folder = Path(raw_dir) / "order_effects" / WANG2014_DIR
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if not files:
        return empty_frame()
    cols: dict[str, list] = {}
    for f in files:
        df = pd.read_csv(f)
        required = ["survey", "n_a_first", "n_b_first", *CELLS["A_first"], *CELLS["B_first"]]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name}: missing columns {missing} (see bre.loaders.order_effects docstring)")
        for r in df.to_dict("records"):
            tab = {c: r[c] for c in [*CELLS["A_first"], *CELLS["B_first"]]}
            tab["intervening_information"] = bool(r.get("intervening_information", 0))
            part = _rows_from_table(
                f"{DATASET_PREFIX}:wang2014:{r['survey']}", tab, True,
                {"A_first": float(r["n_a_first"]), "B_first": float(r["n_b_first"])}, f"{WANG2014_DIR}/{f.name}:{r['survey']}",
            )
            for k, v in part.items():
                cols.setdefault(k, []).extend(v)
    n = len(cols.get("subject_id", []))
    return assemble(cols, n) if n else empty_frame()


def write_order_effects(out_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR, include_unverified: bool = True) -> ValidationReport:
    """Write ``order_effects.parquet`` (+ validation report) and return the report."""
    frame = load_order_effects(raw_dir, include_unverified=include_unverified)
    return write_events(frame, Path(out_dir) / f"{DATASET_PREFIX}.parquet")
