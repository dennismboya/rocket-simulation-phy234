"""CPC18 calibration set (individual choices) -> DecisionEvent rows (CATALOG.md entry B).

Source file: ``data/raw/cpc18/cpc18_calibration_raw.csv.gz`` (510,750 rows, 686 subjects, 210
problems in 7 sets of 30, 25 trials per problem = 5 blocks x 5 trials; block 1 without feedback,
blocks 2-5 with feedback). Every subject played one set; ``Order`` is the position of the problem
in the subject's sequence (1..30) and ``Trial`` the trial within the problem (1..25).

Mapping (CATALOG.md B, schema mapping of the individual data):

* ``dataset = "cpc18"``; ``subject_id = SubjID``; ``session_id = "set<Set>"``;
  ``position_in_session`` = 0-based running index of the row in the subject's ``(Order, Trial)``
  sequence; ``scenario_id = GameID``.
* ``elicitation_type = "lottery_choice"``; ``response = 1`` iff the *safer* option was chosen,
  where safer = the option with the lower outcome variance under its stated distribution (see
  :func:`cpc18_distribution`; ties -> option A is the safer one). The raw choice is kept as
  ``covariates["x_chose_b"]`` (1 = chose B).
* ``loss_pct`` = the previous trial's obtained ``Payoff`` within the same problem divided by the
  problem's outcome range (max outcome - min outcome over both options' distributions, outcomes
  with probability > 0), signed; null on the first trial of a problem and when the range is 0.
  Every payoff lies inside [min, max], but the quotient leaves [-1, 1] when the range is smaller
  than the payoff's magnitude (all-positive or all-negative problems, e.g. outcomes 10..20 and a
  payoff of 15 give 1.5); such values are clipped to +-1 and the number of clipped rows is
  reported in the meta (``loss_norm="range"``, the original catalog rule). The default
  alternative ``loss_norm="max_abs"`` divides by the largest absolute outcome of the problem
  instead, which is always in [-1, 1] and matches the normalization used for choices13k and
  spektor2024. Note that in block 1 (``feedback:off``) and on the first trial of block 2 the
  previous payoff was *not displayed* to the subject; use the ``exp:*`` tags (below) to condition
  on an outcome the subject actually saw.
* ``context_tags``: ``exp:loss`` / ``exp:gain`` for the previous trial's payoff (< 0 / >= 0)
  **only when that payoff was shown**, i.e. the previous trial had ``Feedback == 1`` (this equals
  "block > 1" except on the first trial of block 2, whose predecessor had no feedback). With
  ``pairs=True`` the tags of the two preceding trials are written in chronological order
  (``[exp:<t-2>, exp:<t-1>]``, each only if shown). Then ``feedback:on|off`` (the current
  trial's ``Feedback``) and ``block:<k>``.
* ``prior_question_ids`` = the GameIDs of the problems the subject played before the current one
  (``Order`` smaller), capped at the five most recent, oldest first. The cap keeps the list short;
  the full order is recoverable from ``position_in_session`` and ``scenario_id``.
* ``response_time_ms = RT`` when present (65% missing); ``covariates = {age_band, x_gender,
  x_location, x_chose_b}``; ``incentivized = True``; ``is_synthetic = False``;
  ``source_row_ref = "cpc18_calibration_raw.csv.gz:<0-based data row>"``.

CPC18 lottery definition (``cpc18_distribution``): the organizers' ``CPC18_getDist`` logic of the
CPC18 white paper / BEAST.sd baseline. An option pays ``H`` with probability ``pH`` and ``L``
otherwise when ``LotShape == '-'`` (``LotNum == 1``). Otherwise the ``H`` branch is a lottery with
``LotNum`` outcomes whose expected value is ``H``:

* ``Symm``: outcomes ``H - k/2 + i`` for ``i = 0..k`` (``k = LotNum - 1``) with binomial
  probabilities ``pH * C(k, i) / 2**k``;
* ``R-skew``: outcomes ``H - 1 - LotNum + 2**i`` for ``i = 1..LotNum`` with probabilities
  ``pH / 2**i``, the last one doubled (so they sum to ``pH``);
* ``L-skew``: outcomes ``H + 1 + LotNum - 2**i`` for ``i = 1..LotNum``, same probabilities.

The low outcome ``L`` then receives probability ``1 - pH`` (added to an equal lottery outcome if
one exists, otherwise appended); with ``pH == 1`` there is no ``L`` branch. Ambiguity (``Amb``) and
payoff correlation (``Corr``) do not change an option's marginal distribution: probabilities are
taken as stated.
"""

from __future__ import annotations

import gzip
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre.loaders._common import PROCESSED_DIR, RAW_DIR, age_band, assemble, dumps, summarize
from bre.schema import ValidationReport, write_events

DATASET = "cpc18"
RAW_FILE = "cpc18_calibration_raw.csv.gz"
PRIOR_QUESTIONS_CAP = 5
"""Number of preceding GameIDs kept in ``prior_question_ids`` (most recent, oldest first)."""

RAW_COLUMNS = [
    "SubjID", "Location", "Gender", "Age", "Set", "Condition", "GameID",
    "Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB",
    "Amb", "Corr", "Order", "Trial", "Button", "B", "Payoff", "Forgone", "RT", "Apay", "Bpay",
    "Feedback", "block",
]

LOT_SHAPES = ("-", "Symm", "R-skew", "L-skew")


# ---------------------------------------------------------------------------------------------
# Lottery distributions
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def cpc18_distribution(
    H: float, pH: float, L: float, lot_shape: str, lot_num: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(outcomes, probabilities)`` of one CPC18 option (sorted by outcome).

    Implements the organizers' ``CPC18_getDist`` (see the module docstring). ``lot_shape`` must
    be one of ``'-'``, ``'Symm'``, ``'R-skew'``, ``'L-skew'``; ``lot_num`` is ignored for ``'-'``.
    Probabilities sum to one up to floating-point error.
    """
    if lot_shape not in LOT_SHAPES:
        raise ValueError(f"unknown LotShape {lot_shape!r}; expected one of {LOT_SHAPES}")
    H, pH, L = float(H), float(pH), float(L)
    if lot_shape == "-" or int(lot_num) <= 1:
        if pH >= 1.0:
            outs, probs = [H], [1.0]
        else:
            outs, probs = [L, H], [1.0 - pH, pH]
    else:
        n = int(lot_num)
        high: list[tuple[float, float]] = []
        if lot_shape == "Symm":
            k = n - 1
            for i in range(n):
                high.append((H - k / 2 + i, pH * comb(k, i) / (2**k)))
        else:
            c, sign = (-1 - n, 1.0) if lot_shape == "R-skew" else (1 + n, -1.0)
            for i in range(1, n + 1):
                high.append((H + c + sign * (2**i), pH / (2**i)))
            o, p = high[-1]
            high[-1] = (o, p * 2)
        dist = dict()
        for o, p in high:
            dist[o] = dist.get(o, 0.0) + p
        if pH < 1.0:
            dist[L] = dist.get(L, 0.0) + (1.0 - pH)
        outs = sorted(dist)
        probs = [dist[o] for o in outs]
    return tuple(outs), tuple(probs)


def option_moments(H: float, pH: float, L: float, lot_shape: str, lot_num: int) -> tuple[float, float, float, float]:
    """``(mean, variance, min_outcome, max_outcome)`` of one option (outcomes with p > 0 only)."""
    outs, probs = cpc18_distribution(H, pH, L, lot_shape, lot_num)
    o = np.asarray(outs, dtype=float)
    p = np.asarray(probs, dtype=float)
    mean = float((o * p).sum())
    var = float((o * o * p).sum() - mean * mean)
    keep = o[p > 0]
    return mean, max(var, 0.0), float(keep.min()), float(keep.max())


def safer_option(
    Ha: float, pHa: float, La: float, shape_a: str, num_a: int,
    Hb: float, pHb: float, Lb: float, shape_b: str, num_b: int,
) -> str:
    """``'A'`` or ``'B'``: the option with the lower outcome variance; ties -> ``'A'``."""
    _, var_a, _, _ = option_moments(Ha, pHa, La, shape_a, num_a)
    _, var_b, _, _ = option_moments(Hb, pHb, Lb, shape_b, num_b)
    return "B" if var_b < var_a - 1e-12 else "A"


def problem_table(problems: pd.DataFrame) -> pd.DataFrame:
    """Per-problem variances, safer option and outcome range from the parameter columns."""
    rows = []
    for r in problems.itertuples(index=False):
        ma, va, mina, maxa = option_moments(r.Ha, r.pHa, r.La, r.LotShapeA, r.LotNumA)
        mb, vb, minb, maxb = option_moments(r.Hb, r.pHb, r.Lb, r.LotShapeB, r.LotNumB)
        rows.append(
            {
                "GameID": r.GameID,
                "ev_a": ma, "var_a": va, "ev_b": mb, "var_b": vb,
                "safer": "B" if vb < va - 1e-12 else "A",
                "outcome_min": min(mina, minb),
                "outcome_max": max(maxa, maxb),
            }
        )
    out = pd.DataFrame(rows)
    out["outcome_range"] = out["outcome_max"] - out["outcome_min"]
    out["outcome_max_abs"] = out[["outcome_min", "outcome_max"]].abs().max(axis=1)
    return out


# ---------------------------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------------------------


def read_raw(raw_dir: Path = RAW_DIR, subjects: int | None = None) -> pd.DataFrame:
    """Read the gzipped CSV with its 0-based data-row index in ``raw_row``.

    ``subjects`` keeps only the first ``subjects`` distinct SubjIDs in file order (for tests).
    """
    path = Path(raw_dir) / "cpc18" / RAW_FILE
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, na_values=["NA"], keep_default_na=True)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["raw_row"] = np.arange(len(df), dtype="int64")
    if subjects is not None:
        keep = df["SubjID"].drop_duplicates().iloc[:subjects]
        df = df[df["SubjID"].isin(keep)].reset_index(drop=True)
    return df


LOSS_NORMS = ("range", "max_abs")


def load_cpc18_with_meta(
    raw_dir: Path = RAW_DIR,
    pairs: bool = False,
    subjects: int | None = None,
    loss_norm: str = "max_abs",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the CPC18 DecisionEvent frame and a meta dict (see the module docstring).

    ``pairs``: write the two preceding experienced outcomes as ``exp:*`` tags (chronological
    order) instead of only the last one. ``subjects``: restrict to the first N subjects.
    ``loss_norm``: ``"range"`` (catalog rule, clipped to [-1, 1]) or ``"max_abs"``.
    """
    if loss_norm not in LOSS_NORMS:
        raise ValueError(f"loss_norm must be one of {LOSS_NORMS}; got {loss_norm!r}")
    raw = read_raw(raw_dir, subjects=subjects)
    df = raw.sort_values(["SubjID", "Order", "Trial"], kind="stable").reset_index(drop=True)

    problems = df.drop_duplicates("GameID")[
        ["GameID", "Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB"]
    ]
    # each GameID must have one parameter set
    n_param_sets = df.groupby("GameID")[["Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB"]].nunique().max().max()
    if n_param_sets > 1:
        raise ValueError("cpc18: a GameID maps to more than one parameter set")
    ptab = problem_table(problems).set_index("GameID")
    safer = df["GameID"].map(ptab["safer"])
    chose = np.where(df["B"].to_numpy() == 1, "B", "A")
    response = (chose == safer.to_numpy()).astype(float)

    # previous trials within the same problem (same subject, same Order)
    grp = df.groupby(["SubjID", "Order"], sort=False)
    prev_pay = grp["Payoff"].shift(1)
    prev_fb = grp["Feedback"].shift(1)
    prev2_pay = grp["Payoff"].shift(2)
    prev2_fb = grp["Feedback"].shift(2)
    if loss_norm == "range":
        denom = df["GameID"].map(ptab["outcome_range"]).to_numpy(dtype=float)
    else:
        denom = df["GameID"].map(ptab["outcome_max_abs"]).to_numpy(dtype=float)
    loss = prev_pay.to_numpy(dtype=float) / np.where(denom > 0, denom, np.nan)
    n_clipped = int(np.sum(np.abs(loss[np.isfinite(loss)]) > 1.0))
    loss = np.clip(loss, -1.0, 1.0)

    # position in session
    position = df.groupby("SubjID", sort=False).cumcount().to_numpy()

    # prior GameIDs: per (subject, order) the GameIDs of the preceding orders, capped
    order_games = df.drop_duplicates(["SubjID", "Order"])[["SubjID", "Order", "GameID"]]
    prior_by_key: dict[tuple[int, int], list[str]] = {}
    for subj, g in order_games.groupby("SubjID", sort=False):
        seq = g.sort_values("Order")
        games = [str(int(x)) for x in seq["GameID"].tolist()]
        for i, order in enumerate(seq["Order"].tolist()):
            prior_by_key[(int(subj), int(order))] = games[max(0, i - PRIOR_QUESTIONS_CAP) : i]
    prior_ids = [prior_by_key[(int(s), int(o))] for s, o in zip(df["SubjID"].tolist(), df["Order"].tolist())]

    # context tags
    fb = df["Feedback"].to_numpy()
    block = df["block"].to_numpy()
    pp = prev_pay.to_numpy(dtype=float)
    pf = prev_fb.to_numpy(dtype=float)
    p2p = prev2_pay.to_numpy(dtype=float)
    p2f = prev2_fb.to_numpy(dtype=float)
    tags: list[list[str]] = []
    for i in range(len(df)):
        t: list[str] = []
        if pairs and p2f[i] == 1.0:
            t.append("exp:loss" if p2p[i] < 0 else "exp:gain")
        if pf[i] == 1.0:
            t.append("exp:loss" if pp[i] < 0 else "exp:gain")
        t.append("feedback:on" if fb[i] == 1 else "feedback:off")
        t.append(f"block:{int(block[i])}")
        tags.append(t)

    ages = df["Age"].tolist()
    genders = df["Gender"].tolist()
    locations = df["Location"].tolist()
    chose_b = df["B"].tolist()
    covariates = [
        dumps(
            {
                "age_band": age_band(a),
                "x_gender": str(g),
                "x_location": str(loc),
                "x_chose_b": int(b),
            }
        )
        for a, g, loc, b in zip(ages, genders, locations, chose_b)
    ]
    # drop null age_band keys (validate_covariates drops nulls anyway, but keep the JSON clean)
    covariates = [c.replace('"age_band":null,', "") for c in covariates]

    n = len(df)
    frame = assemble(
        {
            "subject_id": df["SubjID"].astype(str).tolist(),
            "dataset": DATASET,
            "session_id": ("set" + df["Set"].astype(str)).tolist(),
            "position_in_session": position,
            "scenario_id": df["GameID"].astype(str).tolist(),
            "loss_pct": loss,
            "context_tags": tags,
            "prior_question_ids": prior_ids,
            "elicitation_type": "lottery_choice",
            "response": response,
            "response_time_ms": df["RT"].to_numpy(dtype=float),
            "covariates": covariates,
            "incentivized": True,
            "source_row_ref": [f"{RAW_FILE}:{r}" for r in df["raw_row"].tolist()],
        },
        n,
    )
    meta: dict[str, Any] = {
        "dataset": DATASET,
        "raw_file": RAW_FILE,
        "n_raw_rows": int(len(raw)),
        "n_problems": int(len(ptab)),
        "pairs": bool(pairs),
        "subjects_option": subjects,
        "loss_norm": loss_norm,
        "n_loss_pct_clipped": n_clipped,
        "dropped": {},
        "notes": [
            f"loss_pct = previous payoff / {'outcome range' if loss_norm == 'range' else 'max |outcome|'}; clipped to [-1, 1] on {n_clipped} row(s)",
            f"loss_pct null on {int(np.isnan(loss).sum())} row(s) (first trial of a problem or zero range)",
            f"response_time_ms missing on {int(df['RT'].isna().sum())} row(s)",
            f"safer option is A on {int((ptab['safer'] == 'A').sum())} of {len(ptab)} problems",
        ],
        **summarize(frame),
    }
    return frame, meta


def load_cpc18(
    raw_dir: Path = RAW_DIR, pairs: bool = False, subjects: int | None = None, loss_norm: str = "max_abs"
) -> pd.DataFrame:
    """CPC18 individual choices as a DecisionEvent frame (see :func:`load_cpc18_with_meta`)."""
    return load_cpc18_with_meta(raw_dir, pairs=pairs, subjects=subjects, loss_norm=loss_norm)[0]


def write_cpc18(
    out_dir: Path = PROCESSED_DIR,
    raw_dir: Path = RAW_DIR,
    pairs: bool = False,
    subjects: int | None = None,
    loss_norm: str = "max_abs",
) -> ValidationReport:
    """Write ``cpc18.parquet`` (+ validation report) and return the report."""
    frame = load_cpc18(raw_dir, pairs=pairs, subjects=subjects, loss_norm=loss_norm)
    return write_events(frame, Path(out_dir) / f"{DATASET}.parquet")
