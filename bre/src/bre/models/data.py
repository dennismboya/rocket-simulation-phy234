"""ModelData: the numpy view of a DecisionEvent frame that every model consumes.

:func:`build_model_data` turns a validated schema frame (``bre.schema``) into flat arrays with one
entry per response row, plus a per-subject standardized covariate matrix. Models never touch the
frame; they read :class:`ModelData` and, through :func:`bre.models.base.standard_features`, the
shared classical feature matrix. Nothing here fits anything, and nothing here reads files.

Conventions:

* Rows keep the order of the input frame; ``index`` records the frame's index labels so any
  per-row output can be joined back.
* Only the rows in :meth:`ModelData.sell_rows` (``row_kind`` ``sell`` or ``rate``) enter a
  model's likelihood. Tolerance rows (``row_kind == "tol"``) are consumed only through
  ``tol_answer`` of the sell row that follows them; ``other`` rows are carried but unused.
* Missing data are explicit: ``loss = 0`` for a null ``loss_pct``, ``tol_answer = NaN`` when no
  prior tolerance answer exists, one missing-indicator column per covariate in ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pandas as pd

from bre import design as _design
from bre import schema as S

ROW_KINDS: tuple[str, ...] = ("sell", "rate", "tol", "other")
"""``row_kind`` values: binary sell/hold rows, aggregate choice rates, tolerance rows, the rest."""

LIKELIHOOD_ROW_KINDS: frozenset[str] = frozenset({"sell", "rate"})
"""Row kinds the sell likelihood applies to (:meth:`ModelData.sell_rows`)."""

DEFAULT_SELL_TYPES: tuple[str, ...] = ("binary_sell",)
"""Elicitation types mapped to ``row_kind == "sell"`` by default."""
DEFAULT_RATE_TYPES: tuple[str, ...] = ("choice_rate",)
"""Elicitation types mapped to ``row_kind == "rate"`` by default."""

MISSING_SUFFIX = "__missing"
"""Suffix of the per-covariate missing-indicator columns of ``ModelData.X``."""

N_CTX_POSITIONS_DEFAULT = 2
"""Context positions per row: the shared design has at most ordered pairs (PLAN.md section 3)."""

SubjectKey = Literal["subject_id", "dataset/subject_id"]


# ---------------------------------------------------------------------------------------------
# Covariate design matrix
# ---------------------------------------------------------------------------------------------


def covariate_columns(spec: dict | None = None) -> tuple[str, ...]:
    """Column names of the covariate design matrix, in :data:`bre.design.COVARIATE_SPEC` order.

    Per categorical field ``f`` with levels ``l``: one column ``f=l`` per level (full one-hot; no
    level is dropped because the missing indicator already breaks the collinearity with an
    intercept only when a value is missing, and the shared encoder carries an L2 penalty) then
    ``f__missing``. Per numeric field: the z-scored column ``f`` then ``f__missing``.
    """
    spec = _design.COVARIATE_SPEC if spec is None else spec
    cols: list[str] = []
    for name, f in spec.items():
        if f.kind == "categorical":
            cols.extend(f"{name}={lvl}" for lvl in f.levels)
        else:
            cols.append(name)
        cols.append(name + MISSING_SUFFIX)
    return tuple(cols)


def covariate_matrix(
    covariates: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]] | None = None,
    spec: dict | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[float, float]]]:
    """Build the standardized covariate matrix ``(m, d_x)`` for ``m`` covariate records.

    Rules: categorical fields become one-hots over their levels (all zero, indicator one when the
    value is missing or not a known level); numeric fields are z-scored as ``(x - mean) / std``
    with ``mean``/``std`` computed over the non-missing records (``std < 1e-12`` is replaced by
    1, so a constant column becomes zero) and set to 0 when missing, with the indicator set to
    one. Pass ``stats`` (``name -> (mean, std)``, as returned by an earlier call) to standardize
    new records with the fitting sample's statistics; the API must do so for new clients.
    Dataset-specific ``x_*`` keys and ``weight`` are ignored here. Returns ``(X, columns, stats)``.
    """
    spec = _design.COVARIATE_SPEC if spec is None else spec
    cols = covariate_columns(spec)
    m = len(covariates)
    X = np.zeros((m, len(cols)), dtype=np.float64)
    out_stats: dict[str, tuple[float, float]] = {}
    j = 0
    for name, f in spec.items():
        if f.kind == "categorical":
            level_pos = {lvl: j + k for k, lvl in enumerate(f.levels)}
            miss = j + len(f.levels)
            for i, rec in enumerate(covariates):
                v = rec.get(name)
                if isinstance(v, str) and v in level_pos:
                    X[i, level_pos[v]] = 1.0
                else:
                    X[i, miss] = 1.0
            j = miss + 1
        else:
            miss = j + 1
            raw = np.full(m, np.nan)
            for i, rec in enumerate(covariates):
                v = rec.get(name)
                if v is not None and not isinstance(v, bool) and isinstance(v, (int, float, np.integer, np.floating)):
                    raw[i] = float(v)
            present = np.isfinite(raw)
            if stats is not None and name in stats:
                mean, std = float(stats[name][0]), float(stats[name][1])
            elif present.any():
                mean = float(raw[present].mean())
                std = float(raw[present].std())
            else:
                mean, std = 0.0, 1.0
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            out_stats[name] = (mean, std)
            X[present, j] = (raw[present] - mean) / std
            X[~present, miss] = 1.0
            j = miss + 1
    return X, cols, out_stats


# ---------------------------------------------------------------------------------------------
# ModelData
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelData:
    """Flat, model-ready arrays built from one DecisionEvent frame (see module docstring).

    Per-subject (``n_subjects`` rows): ``subject_ids`` (unique, sorted), ``X``, ``X_columns``,
    ``covariate_stats``. Per response row (``n`` entries, frame order): everything else. The
    context vocabulary ``ctx_vocab`` starts with :data:`bre.design.NONNULL_CONTEXTS` and is
    extended by any further tag met in the frame (sorted), so real datasets with tags such as
    ``exp:loss`` map to indices beyond the design's four.
    """

    subject_ids: np.ndarray
    """``(n_subjects,)`` str: unique subject keys, sorted."""
    subject_idx: np.ndarray
    """``(n,)`` int64: row -> position in ``subject_ids``."""
    X: np.ndarray
    """``(n_subjects, d_x)`` float64 standardized covariate design matrix (:func:`covariate_matrix`)."""
    X_columns: tuple[str, ...]
    """Names of the ``d_x`` columns of ``X``."""
    covariate_stats: dict[str, tuple[float, float]]
    """Numeric covariate -> ``(mean, std)`` used for z-scoring; reuse for new subjects."""
    loss: np.ndarray
    """``(n,)`` float64: ``|loss_pct|``, 0 when null (the ``L`` of ``loss_unitary``)."""
    loss_signed: np.ndarray
    """``(n,)`` float64: ``loss_pct`` as stored (negative = loss), 0 when null."""
    ctx_idx: np.ndarray
    """``(n, K)`` int64: context tag indices into ``ctx_vocab`` by position, ``-1`` padding."""
    ctx_mask: np.ndarray
    """``(n, K)`` bool: True where ``ctx_idx`` holds a context."""
    ctx_vocab: tuple[str, ...]
    """Context vocabulary: ``NONNULL_CONTEXTS`` followed by the extra tags met in the frame."""
    order_flag: np.ndarray
    """``(n,)`` int64: 1 when the tolerance question preceded the scenario, else 0."""
    tol_answer: np.ndarray
    """``(n,)`` float64: prior tolerance answer of a tolerance-first row (0/1), NaN otherwise."""
    y: np.ndarray
    """``(n,)`` float64: the response (0/1 for binary rows, a share in [0, 1] for rate rows)."""
    w: np.ndarray
    """``(n,)`` float64: likelihood weight (``covariates['weight']`` when present, else 1)."""
    row_kind: np.ndarray
    """``(n,)`` str: one of :data:`ROW_KINDS`."""
    dataset: np.ndarray
    """``(n,)`` str: the frame's ``dataset`` column (for stratified splits)."""
    session_id: np.ndarray
    """``(n,)`` str: the frame's ``session_id`` (sequential models, temporal splits)."""
    position: np.ndarray
    """``(n,)`` int64: ``position_in_session``."""
    is_synthetic: np.ndarray
    """``(n,)`` bool: carried from the frame so no synthetic result can be labeled real."""
    index: np.ndarray
    """``(n,)``: index labels of the source frame rows."""

    # -- sizes ------------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of response rows."""
        return int(self.y.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(self.subject_ids.shape[0])

    @property
    def d_x(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_ctx_positions(self) -> int:
        return int(self.ctx_idx.shape[1])

    @property
    def n_contexts(self) -> int:
        """Size of ``ctx_vocab`` (the ``V`` of a ``(V, 3)`` context parameter table)."""
        return len(self.ctx_vocab)

    # -- row selections ---------------------------------------------------------------------------

    def sell_mask(self) -> np.ndarray:
        """``(n,)`` bool: rows the sell likelihood applies to (``row_kind`` in ``{sell, rate}``)."""
        return np.isin(self.row_kind, list(LIKELIHOOD_ROW_KINDS))

    def sell_rows(self) -> np.ndarray:
        """Indices of the rows the sell likelihood applies to, in frame order."""
        return np.flatnonzero(self.sell_mask())

    def subset(self, rows: np.ndarray) -> "ModelData":
        """Row subset (a boolean mask or an index array) for splits.

        Per-subject fields (``subject_ids``, ``X``, ``X_columns``, ``covariate_stats``) and
        ``ctx_vocab`` are kept whole so parameter tables stay aligned across train/test subsets;
        ``subject_idx`` therefore still indexes the full subject list.
        """
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        per_row = {f.name: getattr(self, f.name)[rows] for f in fields(self) if f.name in _ROW_FIELDS}
        return replace(self, **per_row)

    def jax(self) -> dict[str, Any]:
        """The per-row and per-subject arrays as ``jax.numpy`` arrays (``row_kind``, ``dataset``,
        ``session_id`` and ``index`` are strings and are left out); ``sell_mask`` is added."""
        import jax.numpy as jnp

        out = {
            name: jnp.asarray(getattr(self, name))
            for name in (
                "subject_idx",
                "X",
                "loss",
                "loss_signed",
                "ctx_idx",
                "ctx_mask",
                "order_flag",
                "tol_answer",
                "y",
                "w",
                "position",
                "is_synthetic",
            )
        }
        out["sell_mask"] = jnp.asarray(self.sell_mask())
        return out

    def summary(self) -> dict[str, Any]:
        """Plain-dict overview for logs: sizes, row kinds, context vocabulary, synthetic flag."""
        kinds, counts = np.unique(self.row_kind, return_counts=True)
        return {
            "n_rows": self.n,
            "n_subjects": self.n_subjects,
            "d_x": self.d_x,
            "row_kinds": {str(k): int(c) for k, c in zip(kinds, counts)},
            "n_sell_rows": int(self.sell_mask().sum()),
            "ctx_vocab": list(self.ctx_vocab),
            "n_tolerance_first": int(self.order_flag.sum()),
            "n_tol_answer_observed": int(np.isfinite(self.tol_answer).sum()),
            "datasets": sorted(set(self.dataset.tolist())),
            "is_synthetic": sorted({bool(v) for v in self.is_synthetic.tolist()}),
        }


_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "subject_idx",
        "loss",
        "loss_signed",
        "ctx_idx",
        "ctx_mask",
        "order_flag",
        "tol_answer",
        "y",
        "w",
        "row_kind",
        "dataset",
        "session_id",
        "position",
        "is_synthetic",
        "index",
    }
)


# ---------------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------------


def _parse_covariates(text: object) -> dict[str, Any]:
    if isinstance(text, dict):
        return dict(text)
    if S.is_null(text):
        return {}
    return S.json_loads_strict(text)


def _weights(cov: pd.Series) -> np.ndarray:
    """``covariates['weight']`` per row (1 when absent or null). Only rows whose canonical JSON
    text mentions the key are parsed."""
    w = np.ones(len(cov), dtype=np.float64)
    text = cov.astype("object")
    has = text.map(lambda t: isinstance(t, dict) or (isinstance(t, str) and '"weight"' in t)).to_numpy()
    for i in np.flatnonzero(has):
        val = _parse_covariates(text.iloc[i]).get(S.COVARIATE_WEIGHT_KEY)
        if val is not None and not isinstance(val, bool) and np.isfinite(float(val)):
            w[i] = float(val)
    return w


def _first_per_subject(df: pd.DataFrame, key: pd.Series) -> list[dict[str, Any]]:
    """First covariates record per subject, in sorted-key order."""
    first = df.assign(_k=key.to_numpy()).drop_duplicates("_k").set_index("_k")["covariates"]
    return [_parse_covariates(first.loc[k]) for k in sorted(first.index)]


def build_model_data(
    df: pd.DataFrame,
    design: ModuleType = _design,
    *,
    n_ctx_positions: int = N_CTX_POSITIONS_DEFAULT,
    ctx_vocab: tuple[str, ...] | None = None,
    covariate_stats: dict[str, tuple[float, float]] | None = None,
    sell_types: tuple[str, ...] = DEFAULT_SELL_TYPES,
    rate_types: tuple[str, ...] = DEFAULT_RATE_TYPES,
    subject_key: SubjectKey = "subject_id",
    require_consent: bool = True,
    validate: bool = False,
) -> ModelData:
    """Build a :class:`ModelData` from a DecisionEvent frame (``bre.schema.COLUMNS``).

    Rules, in the order they are applied:

    1. ``validate=True`` runs ``bre.schema.validate_frame(df, strict=True)`` first (loaders and
       ``read_events`` already validate, so it is off by default); missing columns always raise.
    2. ``require_consent`` drops rows with ``consent_training == False`` (the schema's warning
       says to exclude them from fits) before anything else is derived.
    3. Subject key: ``subject_id`` (default); a subject_id met in more than one ``dataset`` then
       raises, and ``subject_key="dataset/subject_id"`` keys subjects by ``"<dataset>/<subject_id>"``
       instead. ``subject_ids`` is the sorted unique key list; ``subject_idx`` maps rows to it.
    4. ``X`` is :func:`covariate_matrix` over the first row of each subject (covariates are a
       subject property; a later row's differing value is ignored), standardized with
       ``covariate_stats`` when given (apply-to-new-data mode) and with the sample's statistics
       otherwise.
    5. ``loss = |loss_pct|`` and ``loss_signed = loss_pct``, both 0 for a null ``loss_pct``.
    6. Contexts: ``ctx_vocab`` = ``design.NONNULL_CONTEXTS`` + sorted extra tags found in the frame
       (or the given ``ctx_vocab``, in which case an unknown tag raises); a row with more than
       ``n_ctx_positions`` tags raises. ``ctx_idx[i, k]`` is the index of the k-th tag of row i.
    7. ``order_flag = 1`` iff ``question_order_id == design.TOLERANCE_FIRST``, or, when
       ``question_order_id`` is null, iff ``design.TOLERANCE_QUESTION_ID`` is listed in
       ``prior_question_ids`` (PLAN.md section 3: the order is recoverable from the schema).
    8. ``tol_answer``: for each row, the ``response`` of the ``binary_yes_no`` row with
       ``scenario_id == design.TOLERANCE_QUESTION_ID`` in the same ``(dataset, subject_id,
       session_id)`` at ``position_in_session == position - 1``; NaN when there is none. The
       value is kept only for tolerance-first rows (``order_flag == 1``) that are not themselves
       tolerance rows: for a scenario-first row the preceding tolerance row, if any, is the
       closing question of the *previous* item and not part of this item's context, and a
       tolerance row has no prior tolerance answer of its own, so both are set to NaN.
    9. ``row_kind``: ``sell`` for ``elicitation_type in sell_types``, ``rate`` for
       ``rate_types``, ``tol`` for ``binary_yes_no`` rows with ``scenario_id ==
       design.TOLERANCE_QUESTION_ID``, ``other`` for everything else. Pass
       ``sell_types=("binary_sell", "lottery_choice")`` for a dataset whose loader coded the
       safe-option analog as ``lottery_choice``.
    10. ``y = response``; ``w = covariates['weight']`` when present and non-null, else 1.
    """
    missing = [c for c in S.COLUMNS if c not in df.columns]
    if missing:
        raise S.SchemaError(f"build_model_data: frame is missing columns {missing}")
    if n_ctx_positions < 1:
        raise ValueError("n_ctx_positions must be >= 1")
    if validate:
        S.validate_frame(df, strict=True)
    if require_consent:
        keep = df["consent_training"].map(bool).to_numpy()
        df = df.loc[keep]
    df = df.copy()

    n = len(df)
    dataset = df["dataset"].astype("object").to_numpy()
    subject = df["subject_id"].astype("object").to_numpy()
    if subject_key == "dataset/subject_id":
        key = pd.Series([f"{d}/{s}" for d, s in zip(dataset, subject)], index=df.index, dtype="object")
    elif subject_key == "subject_id":
        key = pd.Series(subject, index=df.index, dtype="object")
        pairs = pd.DataFrame({"s": subject, "d": dataset}).drop_duplicates()
        dup = pairs["s"][pairs["s"].duplicated()].unique().tolist()
        if dup:
            raise ValueError(
                f"subject_id(s) {dup[:5]} occur in more than one dataset; pass "
                "subject_key='dataset/subject_id' to key subjects by (dataset, subject_id)"
            )
    else:
        raise ValueError(f"unknown subject_key {subject_key!r}")
    codes, uniques = pd.factorize(key, sort=True)
    subject_ids = np.asarray(uniques, dtype=object).astype(str)
    subject_idx = np.asarray(codes, dtype=np.int64)

    cov_records = _first_per_subject(df, key) if n else []
    X, X_columns, stats = covariate_matrix(cov_records, stats=covariate_stats, spec=design.COVARIATE_SPEC)

    loss_signed = pd.to_numeric(df["loss_pct"], errors="raise").astype("float64").fillna(0.0).to_numpy()
    loss = np.abs(loss_signed)

    tags_col = df["context_tags"].map(lambda v: list(v) if not S.is_null(v) else []).tolist()
    base_vocab = tuple(design.NONNULL_CONTEXTS)
    if ctx_vocab is None:
        extra = sorted({t for tags in tags_col for t in tags if t not in base_vocab})
        vocab = base_vocab + tuple(extra)
    else:
        vocab = tuple(ctx_vocab)
        if vocab[: len(base_vocab)] != base_vocab:
            raise ValueError("ctx_vocab must start with design.NONNULL_CONTEXTS")
        unknown = sorted({t for tags in tags_col for t in tags if t not in vocab})
        if unknown:
            raise ValueError(f"context tag(s) {unknown} are not in the given ctx_vocab")
    pos_of = {t: i for i, t in enumerate(vocab)}
    ctx_idx = np.full((n, n_ctx_positions), -1, dtype=np.int64)
    for i, tags in enumerate(tags_col):
        if len(tags) > n_ctx_positions:
            raise ValueError(
                f"row {df.index[i]!r} has {len(tags)} context tags; n_ctx_positions={n_ctx_positions}"
            )
        for k, t in enumerate(tags):
            ctx_idx[i, k] = pos_of[t]
    ctx_mask = ctx_idx >= 0

    qo = df["question_order_id"].astype("object")
    prior = df["prior_question_ids"].map(lambda v: list(v) if not S.is_null(v) else [])
    order_flag = np.asarray(
        [
            1 if (q == design.TOLERANCE_FIRST) or (S.is_null(q) and design.TOLERANCE_QUESTION_ID in p) else 0
            for q, p in zip(qo.tolist(), prior.tolist())
        ],
        dtype=np.int64,
    )

    et = df["elicitation_type"].astype("object").to_numpy()
    sid = df["scenario_id"].astype("object").to_numpy()
    is_tol = (et == "binary_yes_no") & (sid == design.TOLERANCE_QUESTION_ID)
    row_kind = np.full(n, "other", dtype="<U5")
    row_kind[np.isin(et, list(sell_types))] = "sell"
    row_kind[np.isin(et, list(rate_types))] = "rate"
    row_kind[is_tol] = "tol"

    position = pd.to_numeric(df["position_in_session"], errors="raise").astype("int64").to_numpy()
    session = df["session_id"].astype("object").to_numpy()
    y = pd.to_numeric(df["response"], errors="raise").astype("float64").to_numpy()

    tol_answer = np.full(n, np.nan, dtype=np.float64)
    if is_tol.any():
        tol_rows = pd.DataFrame(
            {
                "dataset": dataset[is_tol],
                "subject_id": subject[is_tol],
                "session_id": session[is_tol],
                "position_in_session": position[is_tol] + 1,
                "_tol": y[is_tol],
            }
        )
        rows = pd.DataFrame(
            {
                "dataset": dataset,
                "subject_id": subject,
                "session_id": session,
                "position_in_session": position,
            }
        )
        joined = rows.merge(tol_rows, how="left", on=list(rows.columns))
        tol_answer = joined["_tol"].to_numpy(dtype=np.float64, na_value=np.nan)
    tol_answer = np.where((order_flag == 1) & ~is_tol, tol_answer, np.nan)

    return ModelData(
        subject_ids=subject_ids,
        subject_idx=subject_idx,
        X=X,
        X_columns=tuple(X_columns),
        covariate_stats=dict(stats),
        loss=loss.astype(np.float64),
        loss_signed=loss_signed.astype(np.float64),
        ctx_idx=ctx_idx,
        ctx_mask=ctx_mask,
        ctx_vocab=tuple(vocab),
        order_flag=order_flag,
        tol_answer=tol_answer.astype(np.float64),
        y=y,
        w=_weights(df["covariates"]),
        row_kind=row_kind,
        dataset=dataset.astype(str),
        session_id=session.astype(str),
        position=position,
        is_synthetic=df["is_synthetic"].map(bool).to_numpy(dtype=bool),
        index=np.asarray(df.index),
    )


__all__ = [
    "DEFAULT_RATE_TYPES",
    "DEFAULT_SELL_TYPES",
    "LIKELIHOOD_ROW_KINDS",
    "MISSING_SUFFIX",
    "ModelData",
    "N_CTX_POSITIONS_DEFAULT",
    "ROW_KINDS",
    "build_model_data",
    "covariate_columns",
    "covariate_matrix",
]
