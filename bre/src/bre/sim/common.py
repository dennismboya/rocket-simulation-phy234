"""Pieces shared by the synthetic generators G_Q, G_C and G_F (PLAN.md section 5).

Contract (stable; ``bre.sim.gc`` and ``bre.sim.gf`` import it):

* :func:`make_subjects` — subject ids ``S0001``... and one covariate record per subject drawn
  from :func:`bre.design.sample_covariates` (placeholder-uniform marginals, hence synthetic).
* :func:`design_items` — the 170-item design (:func:`bre.design.full_design`) or a subset.
* :func:`assemble_events` — one subject's responses to the items as schema rows: every item is
  stored as two rows in its question order, a ``binary_yes_no`` tolerance row
  (``scenario_id == design.TOLERANCE_QUESTION_ID``) and a ``binary_sell`` row. Tolerance-first:
  tolerance at position ``p``, sell at ``p + 1`` with ``prior_question_ids == ["tolerance"]``.
  Scenario-first: sell at ``p``, tolerance at ``p + 1`` with ``prior_question_ids ==
  [scenario_id]``. Presentation order is a permutation drawn from the caller's ``rng``. The
  layout is the one ``bre.models.data.build_model_data`` reads back (``order_flag`` from
  ``question_order_id``, ``tol_answer`` from the tolerance row at ``position - 1``).
* :func:`write_synthetic` / :func:`read_synthetic` — ``<name>.parquet`` through
  :func:`bre.schema.write_events` (which enforces the ``data/synthetic`` location and
  ``is_synthetic == True``) next to ``<name>.truth.json`` with the latent parameters.

Fixed row values of every generated row: ``is_synthetic = True``, ``incentivized = False``,
``consent_training = True``, ``timestamp = None``, ``response_time_ms = None``,
``outcome_behavior = None``, ``source_row_ref = "sim:<generator>:<seed>:<subject>:<item>"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from bre import design as _design
from bre import schema as S

SUBJECT_ID_PREFIX = "S"
SUBJECT_ID_MIN_WIDTH = 4
"""``S0001``, ``S0002``, ...; the zero padding widens beyond four digits for ``n > 9999`` so that
lexicographic order (the order ``build_model_data`` sorts subjects in) is numeric order."""

TOLERANCE_ROW_SUFFIX = "#tol"
"""Appended to the item id in the ``source_row_ref`` of an item's tolerance row."""


# ---------------------------------------------------------------------------------------------
# Subjects and covariates
# ---------------------------------------------------------------------------------------------


def subject_ids(n: int, prefix: str = SUBJECT_ID_PREFIX) -> list[str]:
    """``["S0001", ..., "S{n:04d}"]`` (wider when ``n`` needs more digits)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    width = max(SUBJECT_ID_MIN_WIDTH, len(str(n)))
    return [f"{prefix}{i + 1:0{width}d}" for i in range(n)]


def covariates_json(covariates: dict[str, Any] | str) -> str:
    """Canonical JSON text of a covariate record (validated, sorted keys, compact separators),
    the form ``bre.schema.DecisionEvent`` stores. A JSON string is re-validated and re-serialized."""
    cleaned = S.validate_covariates(covariates, allow_extras=True)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False)


def make_subjects(rng: np.random.Generator | int, n: int, design: ModuleType | None = None) -> pd.DataFrame:
    """``n`` synthetic subjects: columns ``subject_id`` and ``covariates`` (canonical JSON text).

    Covariates come from ``design.sample_covariates`` (default :mod:`bre.design`), i.e. from the
    placeholder marginals recorded in ``design.COVARIATE_MARGINALS_SOURCE``; the result is
    synthetic by construction. Randomness is drawn from ``rng`` (a Generator, or a seed).
    """
    design = _design if design is None else design
    gen = np.random.default_rng(rng)
    covs = design.sample_covariates(gen, n)
    records = [covs.iloc[i].to_dict() for i in range(n)]
    return pd.DataFrame(
        {
            "subject_id": pd.Series(subject_ids(n), dtype="str"),
            "covariates": pd.Series([covariates_json(r) for r in records], dtype="str"),
        }
    )


def covariate_records(subjects: pd.DataFrame) -> list[dict[str, Any]]:
    """The ``covariates`` column of a :func:`make_subjects` frame as a list of dicts."""
    return [S.json_loads_strict(t) for t in subjects["covariates"].tolist()]


# ---------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------


def design_items(design: ModuleType | pd.DataFrame | None = None) -> pd.DataFrame:
    """The items a generator answers, in :data:`bre.design.DESIGN_COLUMNS` layout.

    ``design`` may be None (the full 170-item design of :mod:`bre.design`), a design module with
    a ``full_design()`` function, a callable returning the frame, or a frame (e.g. a battery
    subset from :func:`bre.design.battery_subset`); a frame is copied and its row order kept.
    """
    if design is None:
        return _design.full_design()
    if isinstance(design, pd.DataFrame):
        items = design.copy().reset_index(drop=True)
    elif hasattr(design, "full_design"):
        items = design.full_design()
    elif callable(design):
        items = design()
    else:
        raise TypeError(f"design must be None, a module, a callable or a DataFrame; got {type(design).__name__}")
    missing = [c for c in _ITEM_COLUMNS if c not in items.columns]
    if missing:
        raise ValueError(f"design items are missing column(s) {missing}")
    return items


_ITEM_COLUMNS: tuple[str, ...] = (
    "item_id",
    "scenario_id",
    "loss_pct",
    "horizon_days",
    "context_tags",
    "question_order_id",
    "prior_question_ids",
)


def item_arrays(items: pd.DataFrame, design: ModuleType | None = None) -> dict[str, np.ndarray]:
    """Numeric view of the items for a vectorized forward pass, in item order.

    Keys: ``loss`` (``|loss_pct|``, 0 when null), ``ctx_idx`` ``(m, 2)`` indices into
    ``design.NONNULL_CONTEXTS`` with ``-1`` padding, ``ctx_mask`` ``(m, 2)``, ``order_flag``
    (1 for tolerance-first) — the same encoding ``bre.models.data.ModelData`` uses, so a
    generator's forward pass and a model's forward pass read identical inputs.
    """
    design = _design if design is None else design
    pos = {t: i for i, t in enumerate(design.NONNULL_CONTEXTS)}
    m = len(items)
    loss = np.abs(pd.to_numeric(items["loss_pct"], errors="raise").astype("float64").fillna(0.0).to_numpy())
    ctx_idx = np.full((m, 2), -1, dtype=np.int64)
    for i, tags in enumerate(items["context_tags"].tolist()):
        tags = list(tags) if not S.is_null(tags) else []
        if len(tags) > 2:
            raise ValueError(f"item {items['item_id'].iloc[i]!r} has {len(tags)} context tags; at most 2")
        for k, t in enumerate(tags):
            if t not in pos:
                raise ValueError(f"unknown context tag {t!r}; allowed {design.NONNULL_CONTEXTS}")
            ctx_idx[i, k] = pos[t]
    order_flag = (items["question_order_id"].astype("object") == design.TOLERANCE_FIRST).to_numpy().astype(np.int64)
    return {"loss": loss, "ctx_idx": ctx_idx, "ctx_mask": ctx_idx >= 0, "order_flag": order_flag}


# ---------------------------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------------------------


def source_row_ref(generator: str, seed: int | None, subject_id: str, item_id: str) -> str:
    """``sim:<generator>:<seed>:<subject>:<item>``; ``seed`` renders as ``none`` when unknown."""
    return f"sim:{generator}:{'none' if seed is None else int(seed)}:{subject_id}:{item_id}"


def assemble_events(
    subject_id: str,
    covariates: dict[str, Any] | str,
    items_df: pd.DataFrame,
    sell_responses: np.ndarray,
    tol_responses: np.ndarray,
    dataset: str,
    battery_version: str,
    session_id: str,
    rng: np.random.Generator | int,
    *,
    generator: str = "unknown",
    seed: int | None = None,
    design: ModuleType | None = None,
) -> pd.DataFrame:
    """One subject's answers to ``items_df`` as a canonical DecisionEvent frame (2 rows per item).

    ``sell_responses`` and ``tol_responses`` are 0/1 arrays aligned with ``items_df`` (the
    tolerance answer of an item is the one given in that item, before or after the sell question
    according to ``question_order_id``). Items are presented in a random order drawn from ``rng``
    (``rng.permutation``); item ``j`` of that order occupies positions ``2j`` and ``2j + 1``:

    * tolerance-first: tolerance row (``binary_yes_no``, ``scenario_id = "tolerance"``,
      ``loss_pct``/``horizon_days`` null, no context tags, ``prior_question_ids = []``) at
      ``2j``; sell row (``binary_sell``, the item's scenario, ``prior_question_ids =
      ["tolerance"]``) at ``2j + 1``;
    * scenario-first: sell row (``prior_question_ids = []``) at ``2j``; tolerance row with
      ``prior_question_ids = [scenario_id]`` at ``2j + 1``.

    Both rows carry the item's ``question_order_id`` and ``source_row_ref =
    "sim:<generator>:<seed>:<subject>:<item_id>"`` (the tolerance row adds
    :data:`TOLERANCE_ROW_SUFFIX`). Fixed values: see the module docstring. The frame is passed
    through :func:`bre.schema.conform_frame`; validation happens in :func:`write_synthetic`.
    """
    design = _design if design is None else design
    gen = np.random.default_rng(rng)
    m = len(items_df)
    sell = np.asarray(sell_responses, dtype=np.float64).reshape(-1)
    tol = np.asarray(tol_responses, dtype=np.float64).reshape(-1)
    if sell.shape != (m,) or tol.shape != (m,):
        raise ValueError(f"responses must have one entry per item ({m}); got {sell.shape} and {tol.shape}")
    if not (np.isin(sell, (0.0, 1.0)).all() and np.isin(tol, (0.0, 1.0)).all()):
        raise ValueError("sell and tolerance responses must be 0/1")
    cov_text = covariates_json(covariates)
    perm = gen.permutation(m)
    cols = {c: [] for c in S.COLUMNS}

    def add(**row: Any) -> None:
        base = dict(
            subject_id=subject_id,
            dataset=dataset,
            session_id=session_id,
            timestamp=None,
            response_time_ms=None,
            covariates=cov_text,
            outcome_behavior=None,
            incentivized=False,
            consent_training=True,
            battery_version=battery_version,
            is_synthetic=True,
        )
        base.update(row)
        for c in S.COLUMNS:
            cols[c].append(base[c])

    records = items_df[list(_ITEM_COLUMNS)].to_dict("records")
    for j, k in enumerate(perm.tolist()):
        it = records[k]
        tol_first = it["question_order_id"] == design.TOLERANCE_FIRST
        p = 2 * j
        tags = list(it["context_tags"]) if not S.is_null(it["context_tags"]) else []
        prior = list(it["prior_question_ids"]) if not S.is_null(it["prior_question_ids"]) else []
        if tol_first and design.TOLERANCE_QUESTION_ID not in prior:
            prior = [design.TOLERANCE_QUESTION_ID] + prior
        ref = source_row_ref(generator, seed, subject_id, str(it["item_id"]))
        loss = it["loss_pct"]
        horizon = it["horizon_days"]
        sell_row = dict(
            position_in_session=p + 1 if tol_first else p,
            scenario_id=str(it["scenario_id"]),
            loss_pct=None if S.is_null(loss) else float(loss),
            horizon_days=None if S.is_null(horizon) else int(horizon),
            context_tags=tags,
            question_order_id=it["question_order_id"],
            prior_question_ids=prior if tol_first else [],
            elicitation_type="binary_sell",
            response=float(sell[k]),
            source_row_ref=ref,
        )
        tol_row = dict(
            position_in_session=p if tol_first else p + 1,
            scenario_id=design.TOLERANCE_QUESTION_ID,
            loss_pct=None,
            horizon_days=None,
            context_tags=[],
            question_order_id=it["question_order_id"],
            prior_question_ids=[] if tol_first else [str(it["scenario_id"])],
            elicitation_type="binary_yes_no",
            response=float(tol[k]),
            source_row_ref=ref + TOLERANCE_ROW_SUFFIX,
        )
        if tol_first:
            add(**tol_row)
            add(**sell_row)
        else:
            add(**sell_row)
            add(**tol_row)
    df = pd.DataFrame({c: pd.Series(cols[c], dtype="object") for c in S.COLUMNS})
    return S.conform_frame(df)


def concat_events(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-subject frames into one canonical frame (empty list -> empty frame)."""
    if not frames:
        return S.empty_frame()
    return S.conform_frame(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------------------------
# Truth files and parquet output
# ---------------------------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/JAX scalars and arrays to plain Python for ``json.dump``."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if hasattr(obj, "tolist") and callable(obj.tolist):  # jax arrays
        return jsonable(np.asarray(obj))
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__} into the truth file")


def truth_path(name: str, out_dir: str | Path | None = None) -> Path:
    out_dir = S.SYNTHETIC_DIR if out_dir is None else Path(out_dir)
    return out_dir / f"{name}.truth.json"


def parquet_path(name: str, out_dir: str | Path | None = None) -> Path:
    out_dir = S.SYNTHETIC_DIR if out_dir is None else Path(out_dir)
    return out_dir / f"{name}.parquet"


def write_synthetic(
    df: pd.DataFrame, truth: dict[str, Any], name: str, out_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Write ``<name>.parquet`` (through :func:`bre.schema.write_events`, which validates the frame,
    requires ``is_synthetic == True`` and refuses any location outside a ``data/synthetic``
    directory) and ``<name>.truth.json`` (the generator's population and per-subject parameters,
    with ``is_synthetic: true`` and the parquet file name added). Returns the two paths.

    ``out_dir`` defaults to :data:`bre.schema.SYNTHETIC_DIR` (``bre/data/synthetic``) so that the
    destination does not depend on the working directory; tests pass a ``.../data/synthetic``
    directory under ``tmp_path``.
    """
    if not name or "/" in name or name.endswith(".parquet"):
        raise ValueError(f"name must be a bare file stem; got {name!r}")
    pq_path = parquet_path(name, out_dir)
    S.write_events(df, pq_path)
    record = dict(truth)
    record["is_synthetic"] = True
    record["file"] = pq_path.name
    record["n_rows"] = int(len(df))
    t_path = truth_path(name, out_dir)
    t_path.write_text(json.dumps(jsonable(record), indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return pq_path, t_path


def read_synthetic(name: str, out_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read back a :func:`write_synthetic` pair: the validated frame and the truth dict."""
    df = S.read_events(parquet_path(name, out_dir))
    truth = S.json_loads_strict(truth_path(name, out_dir).read_text(encoding="utf-8"))
    return df, truth


__all__ = [
    "SUBJECT_ID_MIN_WIDTH",
    "SUBJECT_ID_PREFIX",
    "TOLERANCE_ROW_SUFFIX",
    "assemble_events",
    "concat_events",
    "covariate_records",
    "covariates_json",
    "design_items",
    "item_arrays",
    "jsonable",
    "make_subjects",
    "parquet_path",
    "read_synthetic",
    "source_row_ref",
    "subject_ids",
    "truth_path",
    "write_synthetic",
]
