"""CPC15 aggregate block rates -> DecisionEvent rows (CATALOG.md entry B, CPC15 part).

Source file: ``data/raw/cpc15/cpc15_thomas2024_aggregate.csv`` (750 rows = 150 problems x 5
blocks; RothkopfLab/DatasetBias mirror with engineered features). Fields used: ``GameID``,
``Block`` (1..5), ``Feedback`` (0/1; block 1 is without feedback), ``Rate`` (share choosing B).

Mapping (the catalog gives no CPC15-specific line, so the choices13k aggregate mapping is applied):

* ``dataset = "cpc15"``; ``subject_id = "agg:<GameID>:<Block>"`` (feedback is implied by the
  block), ``session_id = "agg"``, ``position_in_session = 0``, ``scenario_id = <GameID>``;
* ``elicitation_type = "choice_rate"``; ``response = Rate``; the number of respondents behind a
  rate is not in the mirror, so ``covariates = {"weight": null}``;
* ``context_tags = ["feedback:on"|"feedback:off", "block:<k>"]``; ``loss_pct = null``;
  ``incentivized = True``; ``is_synthetic = False``;
  ``source_row_ref = "cpc15_thomas2024_aggregate.csv:<0-based data row>"`` (the file's own
  unnamed index column, which the loader checks equals the row position).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, assemble, dumps, summarize
from bre.schema import ValidationReport, write_events

DATASET = "cpc15"
RAW_FILE = "cpc15_thomas2024_aggregate.csv"


def read_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / "cpc15" / RAW_FILE
    df = pd.read_csv(path, index_col=0)
    for c in ("GameID", "Block", "Feedback", "Rate"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing column {c}")
    if not np.array_equal(df.index.to_numpy(), np.arange(len(df))):
        raise ValueError(f"{path.name}: the index column is not the 0-based row position")
    return df


def load_cpc15_with_meta(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the CPC15 DecisionEvent frame and a meta dict."""
    df = read_raw(raw_dir)
    n = len(df)
    game = df["GameID"].astype(int).tolist()
    block = df["Block"].astype(int).tolist()
    feedback = (df["Feedback"].astype(int) == 1).tolist()
    frame = assemble(
        {
            "subject_id": [f"agg:{g}:{b}" for g, b in zip(game, block)],
            "dataset": DATASET,
            "session_id": "agg",
            "position_in_session": np.zeros(n, dtype="int64"),
            "scenario_id": [str(g) for g in game],
            "context_tags": [["feedback:on" if fb else "feedback:off", f"block:{b}"] for fb, b in zip(feedback, block)],
            "elicitation_type": "choice_rate",
            "response": df["Rate"].to_numpy(dtype=float),
            "covariates": [dumps({"weight": None})] * n,
            "incentivized": True,
            "source_row_ref": [f"{RAW_FILE}:{i}" for i in df.index.tolist()],
        },
        n,
    )
    meta: dict[str, Any] = {
        "dataset": DATASET,
        "raw_file": RAW_FILE,
        "n_raw_rows": int(n),
        "n_problems": int(df["GameID"].nunique()),
        "dropped": {},
        "notes": ["weight is null: respondent counts per rate are not in the mirror"],
        **summarize(frame),
    }
    return frame, meta


def load_cpc15(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """CPC15 aggregate rates as a DecisionEvent frame."""
    return load_cpc15_with_meta(raw_dir)[0]


def write_cpc15(out_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR) -> ValidationReport:
    """Write ``cpc15.parquet`` (+ validation report) and return the report."""
    return write_events(load_cpc15(raw_dir), Path(out_dir) / f"{DATASET}.parquet")
