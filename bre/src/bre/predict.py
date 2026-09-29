"""Prediction facade: the only path from a fitted artifact to a number on a screen (Phase 5).

The API (``api/main.py``) calls the functions here and nothing else; the dashboard calls the API.
No model code lives outside ``bre.models``; this module builds scoring rows in the DecisionEvent
schema, hands them to the artifact's model through ``bre.artifact`` and turns the results into
the dictionaries the endpoints return. Every probability is the *predicted probability of
selling* under the served model (never "will sell"), every interval states where it comes
from, every context carries the number of training responses behind its calibration, and every
intervention effect is labelled :data:`INTERVENTION_LABEL`.

Documented constants (design choices, changeable here only):

* :data:`TYPICAL_CRISIS_CONTEXTS` and :data:`RISK_SCORE_LOSS` define the classical-equivalent
  risk score ``logit P(sell | typical crisis, L = -0.20)``: the ordered pair (recession news,
  then a friend who sells) at a 20% loss, scenario-first, is the single scalar an advisor would
  put on a risk questionnaire. It collapses the state to one number; the R² between it and the
  state angle across the book says how much of the state that number retains.
* :data:`DRAWDOWN_TARGET`: the behavioral drawdown capacity is the largest loss on the design
  grid (:data:`bre.design.LOSS_PCTS`) such that ``P(sell | L', typical crisis) <= target`` for
  every grid loss ``L' <= L`` (the loss the investor is predicted to sit through before the
  sell probability first crosses the target; :data:`CAPACITY_DEFINITION`). The loss enters the
  model as a rotation of the state, so ``P(sell | L)`` need not be monotone in ``L``: a plain
  "max L with P(L) <= target" would credit an investor predicted to sell at 15% with sitting
  through 30% whenever the curve comes back under the target, the first-crossing rule does not.
  The capacity is non-increasing under any change that raises ``P(sell | L)`` at every grid loss
  (``tests/test_predict.py``); it is *not* monotone in the size of ``theta_L`` itself, because a
  larger rotation can carry a state away from the sell pole at small losses.
* :data:`STATUS_THRESHOLDS`: triage status from the predicted probability under the market state.
* :data:`LOSS_RESPONSE_DEFINITION`: the loss-response diagnostic of a book
  (:func:`loss_response_diagnostic`, shown on the dashboard's transparency page and stored in
  the demo artifact's metrics): the share of clients whose ``P(sell | L, no context)`` is
  non-decreasing over the design grid, and the mean ``P(sell)`` per loss level.
* :data:`WEAK_MATCH_THRESHOLD`: a free-text cause frame is mapped to the nearest calibrated news
  context by :func:`cause_frame_similarity`; below the threshold the mapping is flagged
  ``weak_match`` and the dashboard must show the flag.
* :data:`ROW_BUCKETS` / :data:`SUBJECT_BUCKETS`: scoring tables are padded to these sizes so
  that JAX compiles the forward pass once per bucket; :func:`warm_up` pre-compiles the buckets a
  300-client book needs.
* :data:`INTERVAL_SOURCE`: the parameter draws behind every interval come from the artifact
  (``bre.artifact`` module docstring: posterior draws for B2, a diagonal Laplace approximation at
  the optimum for MAP fits). The 80% interval is the 10th-90th percentile of ``P(sell)`` over
  the draws, the 95% interval the 2.5th-97.5th; both are conditional on the fitted per-subject
  effects and on the model.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from bre import design as D
from bre import schema as S
from bre.artifact import KEY_SEP, ModelArtifact, flatten_params, jsonable
from bre.models.data import ModelData, covariate_matrix
from bre.models.quantum import core as C
from bre.models.quantum.q2_context_unitary import forward_pure_fast, forward_rho_fast, needs_mixture, none_mask
from bre.registry import PER_SUBJECT_BLOCKS

# ---------------------------------------------------------------------------------------------
# Documented constants
# ---------------------------------------------------------------------------------------------

PROBABILITY_WORDING = "predicted probability of selling"
"""The only wording for ``p``: a predicted probability under the served model, not a forecast
that the investor will sell."""

INTERVENTION_LABEL = "predicted effect, not causally validated"
"""Attached to every intervention effect (the transform is a modeling assumption, ``db.seed``)."""

TYPICAL_CRISIS_CONTEXTS: tuple[str, ...] = ("news:recession", "social:friend_sells")
"""Ordered context pair of the classical-equivalent risk score and the drawdown capacity."""

RISK_SCORE_LOSS = -0.20
"""Loss level of the classical-equivalent risk score (a design grid level)."""

DRAWDOWN_TARGET = 0.25
"""Default target of the behavioral drawdown capacity (module docstring)."""

CAPACITY_DEFINITION = (
    "behavioral drawdown capacity: the largest loss L on the design grid {grid} such that "
    "P(sell | L', {contexts}, scenario-first) <= {target} for every grid loss L' <= L, i.e. the last "
    "grid loss before P(sell) first crosses the target (0 when the smallest grid loss already "
    "crosses it). The loss enters the model as a rotation of the state, so P(sell | L) is not "
    "guaranteed to be monotone in L; the capacity therefore uses this first-crossing (prefix) rule, "
    "not the value at L alone."
)
"""Definition text returned in ``score_book(...).attrs['capacity_definition']`` (and the API's
``meta.capacity_definition``) so the dashboard can show what the number means."""

STATUS_THRESHOLDS: dict[str, float] = {"elevated": 0.25, "high": 0.50}
"""Triage status: ``"high"`` when ``p >= 0.50``, ``"elevated"`` when ``p >= 0.25``, else
``"stable"``; ``"uncalibrated"`` is appended when a scenario context has no training responses."""

WEAK_MATCH_THRESHOLD = 0.5
"""Cause-frame similarity below which the nearest-context mapping is flagged ``weak_match``. The
similarity is ``max(keyword score, character ratio)``: one keyword hit gives 0.5, two give 1.0,
so an unmatched free text is weak unless its characters resemble the context sentence."""

CAUSE_FRAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "news:recession": (
        "recession", "slowdown", "downturn", "economy", "economic", "earnings", "gdp", "growth",
        "unemployment", "inflation", "rates", "demand", "contraction", "profits", "fundamentals",
        "macro", "layoffs", "bankruptcy", "credit", "default", "crisis", "depression", "stagflation",
    ),
    "news:technical": (
        "technical", "outage", "glitch", "fault", "algorithm", "algorithmic", "automated", "flash",
        "liquidity", "plumbing", "system", "exchange", "venue", "halt", "error", "bug", "software",
        "trading", "quant", "margin", "forced", "mechanical", "rebalancing", "settlement",
    ),
}
"""Keyword bags of the two news contexts (matched on lower-cased word tokens)."""

CAUSE_CANDIDATES: tuple[str, ...] = tuple(CAUSE_FRAME_KEYWORDS)

LEVELS: dict[str, float] = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "extreme": 1.0}
"""Ordinal labels of ``media_intensity`` and ``social_cue_prevalence`` as numbers in [0, 1]."""

SOCIAL_CUE_MIN = 0.5
"""``social:friend_sells`` enters the context sequence when the prevalence is >= medium."""

MEDIA_MIN = 0.25
"""A matched cause frame enters the sequence when the media intensity is >= low (``none`` drops it)."""

RECOVERY_MIN = 0.05
"""``market:recovered_5pct`` enters when the recovery from the trough is >= 5%."""

LOSS_RANGE: tuple[float, float] = (min(abs(x) for x in D.LOSS_PCTS), max(abs(x) for x in D.LOSS_PCTS))
"""Design range of the loss magnitude (0.05 to 0.30); market drawdowns are clipped into it."""

ROW_BUCKETS: tuple[int, ...] = (8, 64, 512, 4096, 8192, 32768)
SUBJECT_BUCKETS: tuple[int, ...] = (1, 8, 64, 512, 4096)

K_SAMPLES_MAX = 200
"""At most this many parameter draws are used for intervals (the first K of the artifact)."""

INTERVAL_SOURCE = (
    "percentiles of P(sell) over the artifact's parameter draws (bre.artifact: posterior draws for "
    "B2, diagonal Laplace approximation at the optimum for MAP fits); conditional on the fitted "
    "per-subject effects and on the model"
)

SCORING_DATASET = "api_scoring"
"""``dataset`` of the throw-away schema rows built for scoring (never stored)."""

_PAD_SUBJECT = "~pad"
_PAD_SESSION = "~pad"

# ---------------------------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------------------------

ARTIFACT_REF_PREFIX = "artifact:"
"""A ``training_data_refs`` entry ``artifact:<dir>`` of a model_registry row names the artifact
directory (relative to ``bre/`` or absolute)."""


def active_registry_row(engine: Engine) -> dict[str, Any] | None:
    """The ``model_registry`` row with ``is_active`` (the most recently promoted one when several
    are active), as a plain dict with the JSON columns decoded; None when there is none."""
    from db.models import ModelRegistry

    with engine.connect() as conn:
        rows = conn.execute(
            select(ModelRegistry).where(ModelRegistry.is_active == True).order_by(ModelRegistry.promoted_at.desc())  # noqa: E712
        ).mappings().all()
    if not rows:
        return None
    row = dict(rows[0])
    for col in ("training_data_refs", "metrics", "calibrated_contexts"):
        row[col] = json.loads(row[col]) if isinstance(row[col], str) else row[col]
    if row.get("promoted_at") is not None and hasattr(row["promoted_at"], "isoformat"):
        row["promoted_at"] = row["promoted_at"].isoformat()
    return row


def artifact_path_of(row: dict[str, Any]) -> str | None:
    """The artifact directory named by a registry row: an ``artifact:<dir>`` entry of
    ``training_data_refs`` (list of strings, or dicts with an ``artifact_path`` key), else a
    ``metrics["artifact_path"]`` note."""
    refs = row.get("training_data_refs") or []
    for ref in refs:
        if isinstance(ref, str) and ref.startswith(ARTIFACT_REF_PREFIX):
            return ref[len(ARTIFACT_REF_PREFIX):]
        if isinstance(ref, dict) and ref.get("artifact_path"):
            return str(ref["artifact_path"])
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("artifact_path"):
        return str(metrics["artifact_path"])
    return None


def resolve_artifact_dir(path: str) -> str:
    """Relative artifact paths are relative to ``bre/`` (``bre.schema.PROJECT_ROOT``)."""
    from pathlib import Path

    p = Path(path).expanduser()
    return str(p if p.is_absolute() else S.PROJECT_ROOT / p)


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> ModelArtifact:
    return ModelArtifact.load(path)


def load_artifact(path: str) -> ModelArtifact:
    """Load (and cache by path and ``artifact.json`` mtime) an artifact directory."""
    from pathlib import Path

    full = resolve_artifact_dir(path)
    mtime = Path(full, "artifact.json").stat().st_mtime
    return _load_cached(full, mtime)


def load_active_artifact(engine: Engine) -> ModelArtifact:
    """The artifact of the active ``model_registry`` row (``LookupError`` when there is none or
    the row names no artifact directory)."""
    row = active_registry_row(engine)
    if row is None:
        raise LookupError("model_registry has no active model; run db.demo_seed.seed_demo or bre.fit")
    path = artifact_path_of(row)
    if path is None:
        raise LookupError(f"model_registry row {row['version']!r} names no artifact directory (artifact:<dir> in training_data_refs)")
    art = load_artifact(path)
    if art.version != row["version"]:
        raise LookupError(f"artifact at {path} is version {art.version!r}, registry row is {row['version']!r}")
    return art


# ---------------------------------------------------------------------------------------------
# Scenario rows
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One row to score: a subject (client) facing one loss scenario in one context sequence."""

    subject_id: str
    covariates: str  # canonical JSON text of the six product keys
    loss_pct: float | None
    context_tags: tuple[str, ...] = ()
    question_order_id: str = D.SCENARIO_FIRST
    tol_answer: float | None = None
    tag: str = ""  # free label to find the row again (e.g. "main", "baseline", "drop:news:recession")


def covariates_text(covariates: dict[str, Any] | str | None) -> str:
    """Canonical JSON of a covariate record restricted to the six product keys (unknown keys and
    ``x_*`` extras are dropped; values validated by ``bre.schema.validate_covariates``)."""
    if covariates is None or (isinstance(covariates, float) and math.isnan(covariates)):
        rec: dict[str, Any] = {}
    elif isinstance(covariates, str):
        rec = S.json_loads_strict(covariates) if covariates.strip() else {}
    else:
        rec = dict(covariates)
    rec = {k: v for k, v in rec.items() if k in S.COVARIATE_KEYS}
    cleaned = S.validate_covariates(rec, allow_extras=False)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check_context_tags(tags: Any, ctx_vocab: tuple[str, ...], n_positions: int = 2) -> tuple[str, ...]:
    """Validate an ordered context list against the artifact's vocabulary (``"none"`` and an
    empty list both mean the no-context condition)."""
    if tags is None:
        return ()
    if isinstance(tags, str):
        tags = [tags]
    out = [t for t in tags if t not in (None, "", D.CONTEXT_NONE)]
    unknown = [t for t in out if t not in ctx_vocab]
    if unknown:
        raise ValueError(f"unknown context tag(s) {unknown}; served vocabulary {list(ctx_vocab)}")
    if len(out) > n_positions:
        raise ValueError(f"at most {n_positions} context tags per scenario; got {out}")
    return tuple(out)


def scenario_frame(scenarios: list[Scenario], dataset: str = SCORING_DATASET) -> tuple[pd.DataFrame, np.ndarray]:
    """Schema frame of the scenarios (one ``binary_sell`` row each, preceded by a ``binary_yes_no``
    tolerance row when a tolerance-first scenario carries an observed answer, so that
    ``build_model_data`` recovers ``tol_answer``) and the frame positions of the sell rows."""
    cols: dict[str, list[Any]] = {c: [] for c in S.COLUMNS}
    sell_pos: list[int] = []
    position = 0

    def add(**row: Any) -> None:
        base = dict(
            dataset=dataset, timestamp=None, horizon_days=D.HORIZON_DAYS, response_time_ms=None,
            outcome_behavior=None, incentivized=False, consent_training=True,
            battery_version=D.DESIGN_VERSION, is_synthetic=False, source_row_ref="api:scoring",
        )
        base.update(row)
        for c in S.COLUMNS:
            cols[c].append(base[c])

    for i, sc in enumerate(scenarios):
        session = f"s{i}"
        tags = list(sc.context_tags)
        loss = None if sc.loss_pct is None else float(sc.loss_pct)
        scen_id = f"api|{D.loss_code(loss) if loss is not None else 'L00'}|{D.condition_id(tags) if all(t in D.CONTEXT_CODES for t in tags) else '+'.join(tags) or 'none'}"
        tol_first = sc.question_order_id == D.TOLERANCE_FIRST
        if tol_first and sc.tol_answer is not None:
            add(
                subject_id=sc.subject_id, session_id=session, position_in_session=0, scenario_id=D.TOLERANCE_QUESTION_ID,
                loss_pct=None, horizon_days=None, context_tags=[], question_order_id=sc.question_order_id,
                prior_question_ids=[], elicitation_type="binary_yes_no", response=float(sc.tol_answer),
                covariates=sc.covariates,
            )
            position = 1
        else:
            position = 0
        sell_pos.append(len(cols["subject_id"]))
        add(
            subject_id=sc.subject_id, session_id=session, position_in_session=position, scenario_id=scen_id,
            loss_pct=loss, context_tags=tags, question_order_id=sc.question_order_id,
            prior_question_ids=[D.TOLERANCE_QUESTION_ID] if tol_first else [], elicitation_type="binary_sell",
            response=0.0, covariates=sc.covariates,
        )
    df = pd.DataFrame({c: pd.Series(cols[c], dtype="object") for c in S.COLUMNS})
    return S.conform_frame(df), np.asarray(sell_pos, dtype=np.int64)


def _bucket(n: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if n <= b:
            return b
    return int(2 ** math.ceil(math.log2(max(n, 1))))


def _pad_frame(df: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    """Append dummy scenario-first rows so that the row count and the subject count both hit a
    bucket (:data:`ROW_BUCKETS`, :data:`SUBJECT_BUCKETS`); dummies are subjects ``~pad<i>``
    with empty covariates and no loss, whose predictions are discarded."""
    n_rows = len(df)
    subj_target = _bucket(n_subjects, SUBJECT_BUCKETS)
    n_pad_subjects = max(subj_target - n_subjects, 1)
    row_target = _bucket(n_rows + n_pad_subjects, ROW_BUCKETS)
    n_pad_rows = row_target - n_rows
    subject_ids = [f"{_PAD_SUBJECT}{i:05d}" for i in range(n_pad_subjects)]
    subj_col = [subject_ids[i % n_pad_subjects] for i in range(n_pad_rows)]
    pad = pd.DataFrame(
        {
            "subject_id": subj_col,
            "dataset": SCORING_DATASET,
            "session_id": [f"{_PAD_SESSION}{i}" for i in range(n_pad_rows)],
            "timestamp": None,
            "position_in_session": 0,
            "scenario_id": "pad",
            "loss_pct": np.nan,
            "horizon_days": D.HORIZON_DAYS,
            "context_tags": [[] for _ in range(n_pad_rows)],
            "question_order_id": D.SCENARIO_FIRST,
            "prior_question_ids": [[] for _ in range(n_pad_rows)],
            "elicitation_type": "binary_sell",
            "response": 0.0,
            "response_time_ms": np.nan,
            "covariates": "{}",
            "outcome_behavior": None,
            "incentivized": False,
            "consent_training": True,
            "battery_version": D.DESIGN_VERSION,
            "is_synthetic": False,
            "source_row_ref": "api:pad",
        },
        columns=list(S.COLUMNS),
    )
    return S.conform_frame(pd.concat([df, pad], ignore_index=True))


# ---------------------------------------------------------------------------------------------
# Forward passes (point and draws) on padded tables
# ---------------------------------------------------------------------------------------------

SubjectEffects = dict[str, dict[str, Any]]
"""``subject_id -> {"u": (4,), "log_gamma": float}``: per-subject effects that override the
artifact's per-subject blocks for those subjects (from :func:`refine_subject`)."""


def _apply_subject_effects(model_name: str, params: dict[str, Any], data: ModelData, effects: SubjectEffects | None) -> dict[str, Any]:
    if not effects:
        return params
    ids = {str(s): i for i, s in enumerate(data.subject_ids)}
    for sid, eff in effects.items():
        if sid not in ids:
            continue
        i = ids[sid]
        if "u" in eff and ("enc", "u") in PER_SUBJECT_BLOCKS[model_name]:
            params["enc"]["u"] = np.array(params["enc"]["u"], dtype=np.float64)
            params["enc"]["u"][i] = np.asarray(eff["u"], dtype=np.float64)
        if "log_gamma" in eff and ("log_gamma",) in PER_SUBJECT_BLOCKS[model_name]:
            params["log_gamma"] = np.array(params["log_gamma"], dtype=np.float64)
            params["log_gamma"][i] = float(eff["log_gamma"])
    return params


def _forward_arrays(model_name: str, model: Any, params: dict[str, Any], data: ModelData, rows: dict[str, Any], mixture: bool) -> jnp.ndarray:
    """``P(sell)`` per row from array arguments (so the JIT cache is keyed by shapes, not by the
    data object): the exact fast forward passes the Q-models use in ``predict_proba``."""
    if model_name in ("Q2", "Q4"):
        X = jnp.asarray(rows["X"])
        z = X @ params["enc"]["W"] + params["enc"]["b"] + params["enc"]["u"]
        psi = jax.vmap(C.prepare_state)(z)
        free = jnp.asarray(~none_mask(tuple(data.ctx_vocab)), dtype=jnp.float64)[:, None]
        theta_ctx = jnp.asarray(params["theta_ctx"], dtype=jnp.float64) * free
        if model_name == "Q2":
            return forward_pure_fast(
                psi, params["theta_L"], theta_ctx, params["phi"], rows["loss_levels"], rows["loss_idx"], rows["subject_idx"],
                rows["ctx_idx"], rows["ctx_mask"], rows["order_flag"], rows["tol_answer"], mixture=mixture,
            )
        rho = jax.vmap(C.to_rho)(psi)
        gamma = jnp.exp(jnp.asarray(params["log_gamma"], dtype=jnp.float64))
        return forward_rho_fast(
            rho, gamma, params["theta_L"], theta_ctx, params["phi"], rows["loss_levels"], rows["loss_idx"], rows["subject_idx"],
            rows["ctx_idx"], rows["ctx_mask"], rows["order_flag"], rows["tol_answer"], mixture=mixture,
        )
    if model_name == "B2":
        return jax.nn.sigmoid(model.logits_from_arrays(params, model.arrays(data)))
    return model.predict_proba(params, data)


def _unflatten_tree(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of ``bre.artifact.flatten_params`` that keeps traced JAX leaves as they are."""
    out: dict[str, Any] = {}
    for key, arr in flat.items():
        parts = key.split(KEY_SEP)
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = arr
    return out


def _row_arrays(model: Any, data: ModelData) -> dict[str, Any]:
    if hasattr(model, "_rows"):
        return model._rows(data)
    levels, idx = np.unique(data.loss, return_inverse=True)
    return {
        "loss_levels": levels, "loss_idx": idx.astype(np.int64).reshape(-1), "subject_idx": data.subject_idx, "ctx_idx": data.ctx_idx,
        "ctx_mask": data.ctx_mask, "order_flag": data.order_flag, "tol_answer": data.tol_answer, "loss": data.loss, "X": data.X,
    }


class Scorer:
    """Scores scenario lists with one artifact: point predictions through
    ``model.predict_proba`` on a stacked, bucket-padded ``ModelData``; intervals through the
    same fast forward pass vectorized over the artifact's parameter draws."""

    def __init__(self, artifact: ModelArtifact, k_samples: int = K_SAMPLES_MAX) -> None:
        self.artifact = artifact
        self.model_name = artifact.model_name
        samples = artifact.param_samples or []
        self.samples = samples[: int(k_samples)]
        self._stacked_pop: dict[str, np.ndarray] | None = None
        self._vmapped: dict[tuple[Any, ...], Any] = {}

    # -- data ------------------------------------------------------------------------------------

    def table(self, scenarios: list[Scenario]) -> tuple[ModelData, np.ndarray, int]:
        """Padded ``ModelData`` of the scenarios, the positions of their sell rows, and the number
        of real rows."""
        df, sell_pos = scenario_frame(scenarios)
        n_real = len(df)
        n_subjects = df["subject_id"].nunique()
        padded = _pad_frame(df, n_subjects)
        data = self.artifact.build_data(padded)
        return data, sell_pos, n_real

    def _params(self, data: ModelData, effects: SubjectEffects | None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        p = self.artifact.params_for(data, params)
        return _apply_subject_effects(self.model_name, p, data, effects)

    # -- point predictions -----------------------------------------------------------------------

    def predict(self, scenarios: list[Scenario], effects: SubjectEffects | None = None, params: dict[str, Any] | None = None) -> np.ndarray:
        """``P(sell)`` of every scenario, in order (``model.predict_proba`` on the padded table)."""
        if not scenarios:
            return np.zeros(0)
        data, sell_pos, _ = self.table(scenarios)
        model = self.artifact.model_instance(data)
        p = np.asarray(model.predict_proba(self._params(data, effects, params), data), dtype=np.float64)
        return p[sell_pos]

    def predict_with_data(self, scenarios: list[Scenario], effects: SubjectEffects | None = None) -> tuple[np.ndarray, ModelData, np.ndarray, Any, dict[str, Any]]:
        data, sell_pos, _ = self.table(scenarios)
        model = self.artifact.model_instance(data)
        params = self._params(data, effects)
        p = np.asarray(model.predict_proba(params, data), dtype=np.float64)
        return p[sell_pos], data, sell_pos, model, params

    # -- draws -----------------------------------------------------------------------------------

    def _population_stack(self) -> dict[str, np.ndarray]:
        if self._stacked_pop is None:
            per_subject = {"/".join(path) for path in PER_SUBJECT_BLOCKS[self.model_name]}
            flats = [flatten_params(s) for s in self.samples]
            keys = [k for k in flats[0] if k not in per_subject]
            self._stacked_pop = {k: np.stack([f[k] for f in flats]) for k in keys}
        return self._stacked_pop

    def predict_samples(self, scenarios: list[Scenario], effects: SubjectEffects | None = None) -> np.ndarray | None:
        """``(K, n)`` ``P(sell)`` under the first ``K`` parameter draws; None without draws."""
        if not self.samples or not scenarios:
            return None
        data, sell_pos, _ = self.table(scenarios)
        model = self.artifact.model_instance(data)
        base = self._params(data, effects)
        rows = _row_arrays(model, data)
        mixture = needs_mixture(rows["order_flag"], rows["tol_answer"])
        if self.model_name not in ("Q2", "Q4", "B2"):
            out = np.stack([np.asarray(model.predict_proba(self._params(data, effects, s), data)) for s in self.samples])
            return out[:, sell_pos]
        stacked = self._population_stack()
        base_flat = flatten_params(base)
        fixed = {k: jnp.asarray(v) for k, v in base_flat.items() if k not in stacked}
        key = (self.model_name, data.n, data.n_subjects, len(rows["loss_levels"]), bool(mixture))
        fn = self._vmapped.get(key)
        if fn is None:
            def one(pop_flat, fixed_flat, X, loss_levels, loss_idx, subject_idx, ctx_idx, ctx_mask, order_flag, tol_answer):
                params = _unflatten_tree({**fixed_flat, **pop_flat})
                arr = {"X": X, "loss_levels": loss_levels, "loss_idx": loss_idx, "subject_idx": subject_idx, "ctx_idx": ctx_idx,
                       "ctx_mask": ctx_mask, "order_flag": order_flag, "tol_answer": tol_answer}
                if self.model_name == "B2":
                    return jax.nn.sigmoid(model.logits_from_arrays(params, model.arrays(data)))
                return _forward_arrays(self.model_name, model, params, data, arr, mixture)

            fn = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None, None, None, None, None, None)))
            self._vmapped[key] = fn
        out = fn(
            {k: jnp.asarray(v) for k, v in stacked.items()}, fixed, jnp.asarray(rows["X"]), jnp.asarray(rows["loss_levels"]),
            jnp.asarray(rows["loss_idx"]), jnp.asarray(rows["subject_idx"]), jnp.asarray(rows["ctx_idx"]), jnp.asarray(rows["ctx_mask"]),
            jnp.asarray(rows["order_flag"]), jnp.asarray(rows["tol_answer"]),
        )
        return np.asarray(out, dtype=np.float64)[:, sell_pos]

    def interference(self, scenarios: list[Scenario], effects: SubjectEffects | None = None, mix_weights: dict[str, float] | None = None) -> dict[str, np.ndarray] | None:
        """The Q-model interference terms per scenario (None for a classical model)."""
        if self.model_name not in ("Q2", "Q4") or not scenarios:
            return None
        data, sell_pos, _ = self.table(scenarios)
        model = self.artifact.model_instance(data)
        params = self._params(data, effects)
        terms = model.interference_terms(params, data, mix_weights=mix_weights)
        return {k: np.asarray(v, dtype=np.float64)[sell_pos] for k, v in terms.items()}


_SCORERS: dict[int, Scorer] = {}


def scorer_for(artifact: ModelArtifact) -> Scorer:
    """One :class:`Scorer` per artifact object (its compiled functions are cached on it)."""
    key = id(artifact)
    sc = _SCORERS.get(key)
    if sc is None or sc.artifact is not artifact:
        sc = Scorer(artifact)
        _SCORERS.clear()
        _SCORERS[key] = sc
    return sc


WARM_UP_BOOK_SIZES: tuple[int, ...] = (300, 3)
"""Book sizes :func:`warm_up` compiles by default: the 300-client batch of the acceptance test
and a three-client book (one curl, one screen), which share no row bucket."""


def warm_up(artifact: ModelArtifact, n_clients: int | tuple[int, ...] | list[int] = WARM_UP_BOOK_SIZES, interventions: Any = None) -> dict[str, float]:
    """Compile the forward passes a service needs before its first request: the single-row
    passes (both question orders, with and without a tolerance answer) and, for every book size
    in ``n_clients``, the point pass on the full scenario stack and the draw pass on the main
    rows. ``interventions`` should be the served table: every active intervention adds one
    counterfactual row per client, so a book scored with them lands in a different row bucket
    (:data:`ROW_BUCKETS`) than one scored without. Returns seconds per step."""
    import time

    sc = scorer_for(artifact)
    cov = covariates_text({})
    sizes = [int(n_clients)] if isinstance(n_clients, int) else [int(n) for n in n_clients]
    sizes = [n for n in dict.fromkeys(sizes) if n > 0]
    out: dict[str, float] = {}
    t0 = time.time()
    for order, tol in ((D.SCENARIO_FIRST, None), (D.TOLERANCE_FIRST, None), (D.TOLERANCE_FIRST, 1.0)):
        predict_sell(artifact, {}, -0.1, ("news:recession", "social:friend_sells"), order, tol, subject_id="warm")
    out["single"] = time.time() - t0
    market = {"drawdown_pct": -0.12, "cause_frame": "recession fears", "social_cue_prevalence": "high"}
    for n in sizes:
        t0 = time.time()
        clients = pd.DataFrame({"client_id": [f"warm{i}" for i in range(n)], "display_label": "warm", "covariates": cov})
        score_book(artifact, clients, market, interventions)
        out[f"book_{n}"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------
# Helpers shared by the facade functions
# ---------------------------------------------------------------------------------------------


def _interval(samples: np.ndarray | None, level: float) -> list[float] | None:
    if samples is None or samples.size == 0:
        return None
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def _no_samples_reason(artifact: ModelArtifact) -> str:
    return "no interval: the artifact carries no parameter draws (param_samples is empty)"


def calibration_of(artifact: ModelArtifact, tags: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """``tag -> {n_responses, status}`` for the requested contexts (``"none"`` for the empty
    condition, counted from the artifact metrics when available)."""
    out: dict[str, dict[str, Any]] = {}
    table = artifact.calibrated_contexts or {}
    if not tags:
        n_none = (artifact.metrics or {}).get("n_responses_no_context")
        out[D.CONTEXT_NONE] = {"n_responses": n_none, "status": "calibrated" if (n_none or 0) > 0 else "uncalibrated: prior only"}
    for t in tags:
        entry = table.get(t)
        if entry is None:
            out[t] = {"n_responses": 0, "status": "uncalibrated: prior only"}
        else:
            out[t] = {"n_responses": int(entry.get("n_responses", 0)), "status": str(entry.get("status"))}
    return out


def _status(p: float, calibration: dict[str, dict[str, Any]], thresholds: dict[str, float] | None = None) -> str:
    th = STATUS_THRESHOLDS if thresholds is None else {**STATUS_THRESHOLDS, **thresholds}
    if p >= th["high"]:
        s = "high"
    elif p >= th["elevated"]:
        s = "elevated"
    else:
        s = "stable"
    if any(str(v.get("status", "")).startswith("uncalibrated") for v in calibration.values()):
        s += "; uncalibrated context"
    return s


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _model_meta(artifact: ModelArtifact) -> dict[str, Any]:
    return {
        "model_version": artifact.version,
        "model_type": artifact.model_name,
        "family": artifact.family,
        "synthetic": bool(artifact.is_synthetic_training),
        "wording": PROBABILITY_WORDING,
    }


# ---------------------------------------------------------------------------------------------
# Facade: single prediction
# ---------------------------------------------------------------------------------------------


def _driver_scenarios(subject_id: str, cov: str, loss_pct: float | None, tags: tuple[str, ...], order: str, tol: float | None) -> list[Scenario]:
    """Counterfactuals for the top driver: each context removed in turn, and the loss set to 0."""
    out: list[Scenario] = []
    for t in dict.fromkeys(tags):
        rest = tuple(x for x in tags if x != t)
        out.append(Scenario(subject_id, cov, loss_pct, rest, order, tol, tag=f"drop:{t}"))
    if loss_pct is not None and abs(float(loss_pct)) > 0:
        out.append(Scenario(subject_id, cov, 0.0, tags, order, tol, tag="drop:loss"))
    return out


def _top_driver(p: float, drivers: list[Scenario], p_drivers: np.ndarray) -> dict[str, Any]:
    if not drivers:
        return {"driver": None, "delta_p": 0.0, "p_without": float(p), "note": "no context and no loss in the scenario"}
    deltas = p - p_drivers
    j = int(np.argmax(np.abs(deltas)))
    name = drivers[j].tag.removeprefix("drop:")
    return {"driver": name, "delta_p": float(deltas[j]), "p_without": float(p_drivers[j]),
            "candidates": {d.tag.removeprefix("drop:"): float(p - q) for d, q in zip(drivers, p_drivers)}}


def predict_sell(
    artifact: ModelArtifact,
    covariates: dict[str, Any] | str | None,
    loss_pct: float | None,
    context_tags: Any = (),
    question_order: str = D.SCENARIO_FIRST,
    tol_answer: float | None = None,
    *,
    subject_id: str = "client",
    mix_weights: dict[str, float] | None = None,
    subject_effects: SubjectEffects | None = None,
) -> dict[str, Any]:
    """The predicted probability of selling for one investor in one scenario.

    Returns ``p`` (point prediction), ``ci80`` / ``ci95`` (:data:`INTERVAL_SOURCE`; None with
    ``interval_note`` when the artifact has no draws), ``interference`` (``ltp`` always for a
    Q-model, ``order_effect`` for an ordered pair, ``mix`` when ``mix_weights`` is given; zeros
    with a note for a classical model), ``top_driver`` (the context, or the loss, whose removal
    changes ``p`` the most), ``n_calibration`` (training responses per requested context),
    ``model_version``, ``synthetic``. ``subject_id`` selects a training subject's fitted effects
    when it matches ``artifact.subject_ids``; ``subject_effects`` overrides them.
    """
    if question_order not in D.QUESTION_ORDER_IDS:
        raise ValueError(f"question_order must be one of {D.QUESTION_ORDER_IDS}; got {question_order!r}")
    if loss_pct is not None and not (-1.0 <= float(loss_pct) <= 0.0):
        raise ValueError(f"loss_pct must be a signed loss in [-1, 0]; got {loss_pct}")
    tags = check_context_tags(context_tags, artifact.ctx_vocab, artifact.n_ctx_positions)
    cov = covariates_text(covariates)
    tol = None if (tol_answer is None or question_order != D.TOLERANCE_FIRST) else float(tol_answer)
    sc = scorer_for(artifact)
    main = Scenario(subject_id, cov, loss_pct, tags, question_order, tol, tag="main")
    drivers = _driver_scenarios(subject_id, cov, loss_pct, tags, question_order, tol)
    p_all = sc.predict([main] + drivers, subject_effects)
    p = float(p_all[0])
    samples = sc.predict_samples([main], subject_effects)
    s = None if samples is None else samples[:, 0]
    terms = sc.interference([main], subject_effects, mix_weights=mix_weights)
    if terms is None:
        interference: dict[str, Any] = {"ltp": 0.0, "order_effect": 0.0 if len(tags) == 2 else None, "mix": None,
                                        "note": "classical model: interference terms are zero by construction"}
    else:
        interference = {"ltp": float(terms["ltp"][0])}
        interference["order_effect"] = float(terms["order"][0]) if len(tags) == 2 and np.isfinite(terms["order"][0]) else None
        interference["mix"] = float(terms["mix"][0]) if "mix" in terms else None
        interference["definitions"] = "PLAN.md section 4: ltp = P(sell | scenario-first) - sum_a P(a, sell | tolerance-first); order_effect = P(sell | c1, c2) - P(sell | c2, c1); mix = coherent minus classical mixture of the weighted cause frames"
    L = 0.0 if loss_pct is None else abs(float(loss_pct))
    out = {
        "p": p,
        "ci80": _interval(s, 0.80),
        "ci95": _interval(s, 0.95),
        "interval_note": INTERVAL_SOURCE if s is not None else _no_samples_reason(artifact),
        "n_draws": 0 if s is None else int(s.shape[0]),
        "interference": interference,
        "top_driver": _top_driver(p, drivers, p_all[1:]),
        "n_calibration": calibration_of(artifact, tags),
        "scenario": {
            "loss_pct": None if loss_pct is None else float(loss_pct),
            "context_tags": list(tags),
            "question_order_id": question_order,
            "tol_answer": tol,
            "on_design_grid": any(abs(L - abs(x)) < 1e-9 for x in D.LOSS_PCTS),
            "in_design_range": LOSS_RANGE[0] - 1e-9 <= L <= LOSS_RANGE[1] + 1e-9,
            "known_subject": bool(artifact.subject_ids and subject_id in artifact.subject_ids) or bool(subject_effects and subject_id in subject_effects),
        },
        **_model_meta(artifact),
    }
    return jsonable(out)


# ---------------------------------------------------------------------------------------------
# Facade: per-subject refinement from prior responses (intake answers)
# ---------------------------------------------------------------------------------------------

REFINE_STEPS = 300
REFINE_LR = 0.05
REFINE_ROW_BUCKET = 64
"""Per-subject MAP refinement (:func:`refine_subject`): Adam on the subject's random effects
only (``u`` and, for Q4, ``log_gamma``), population parameters fixed, ``REFINE_STEPS`` steps at
``REFINE_LR``; the prior responses are padded to ``REFINE_ROW_BUCKET`` rows so the step compiles
once."""


def prior_responses_frame(subject_id: str, cov: str, prior_responses: list[dict[str, Any]], ctx_vocab: tuple[str, ...], n_positions: int) -> pd.DataFrame:
    """Schema frame of intake answers: each item ``{loss_pct, context_tags, question_order_id,
    response (0/1 sell), tol_answer (0/1, optional)}`` becomes a sell row (and a tolerance row in
    the item's order when ``tol_answer`` is given)."""
    cols: dict[str, list[Any]] = {c: [] for c in S.COLUMNS}

    def add(**row: Any) -> None:
        base = dict(
            subject_id=subject_id, dataset=SCORING_DATASET, session_id="prior", timestamp=None, horizon_days=D.HORIZON_DAYS,
            response_time_ms=None, covariates=cov, outcome_behavior=None, incentivized=False, consent_training=True,
            battery_version=D.DESIGN_VERSION, is_synthetic=False, source_row_ref="api:prior",
        )
        base.update(row)
        for c in S.COLUMNS:
            cols[c].append(base[c])

    pos = 0
    for k, item in enumerate(prior_responses):
        tags = list(check_context_tags(item.get("context_tags", ()), ctx_vocab, n_positions))
        order = item.get("question_order_id") or D.SCENARIO_FIRST
        if order not in D.QUESTION_ORDER_IDS:
            raise ValueError(f"prior_responses[{k}].question_order_id must be one of {D.QUESTION_ORDER_IDS}")
        loss = item.get("loss_pct")
        loss = None if loss is None else float(loss)
        resp = item.get("response")
        if resp not in (0, 1, 0.0, 1.0, True, False):
            raise ValueError(f"prior_responses[{k}].response must be 0 (hold) or 1 (sell)")
        tol = item.get("tol_answer")
        tol_first = order == D.TOLERANCE_FIRST
        scen = f"prior|{k}"
        tol_row = dict(position_in_session=None, scenario_id=D.TOLERANCE_QUESTION_ID, loss_pct=None, horizon_days=None,
                       context_tags=[], question_order_id=order, prior_question_ids=[], elicitation_type="binary_yes_no",
                       response=None if tol is None else float(tol))
        sell_row = dict(position_in_session=None, scenario_id=scen, loss_pct=loss, context_tags=tags, question_order_id=order,
                        prior_question_ids=[D.TOLERANCE_QUESTION_ID] if tol_first else [], elicitation_type="binary_sell",
                        response=float(resp))
        if tol_first and tol is not None:
            tol_row["position_in_session"] = pos
            add(**tol_row)
            pos += 1
        sell_row["position_in_session"] = pos
        add(**sell_row)
        pos += 1
        if not tol_first and tol is not None:
            tol_row["position_in_session"] = pos
            tol_row["prior_question_ids"] = [scen]
            add(**tol_row)
            pos += 1
    df = pd.DataFrame({c: pd.Series(cols[c], dtype="object") for c in S.COLUMNS})
    return S.conform_frame(df)


def refine_subject(artifact: ModelArtifact, subject_id: str, covariates: dict[str, Any] | str | None, prior_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """MAP of one subject's random effects given its prior answers, population parameters fixed
    (module constants). Returns ``{"u", "log_gamma" (Q4 only), "sd" (diagonal Laplace standard
    deviations of the refined effects), "n_responses", "objective"}``; for a classical model
    without a supported per-subject block the result is empty with a note."""
    import optax

    cov = covariates_text(covariates)
    df = prior_responses_frame(subject_id, cov, prior_responses, artifact.ctx_vocab, artifact.n_ctx_positions)
    n_real = len(df)
    if n_real == 0:
        return {"note": "no prior responses", "n_responses": 0}
    padded = _pad_frame(df, 1)
    # padded rows belong to dummy subjects; they carry no likelihood for this subject anyway
    data = artifact.build_data(padded)
    sub = data.subset(np.flatnonzero(data.sell_mask() & (data.subject_idx == list(data.subject_ids).index(subject_id))))
    model = artifact.model_instance(data)
    base = artifact.params_for(data)
    blocks = PER_SUBJECT_BLOCKS[artifact.model_name]
    if ("enc", "u") not in blocks:
        return {"note": f"{artifact.model_name} has no per-subject random effect to refine", "n_responses": int(sub.n)}
    idx = list(data.subject_ids).index(subject_id)
    free = {"u": jnp.asarray(base["enc"]["u"][idx])}
    if ("log_gamma",) in blocks:
        free["log_gamma"] = jnp.asarray(float(base["log_gamma"][idx]))
    fixed = {k: (jnp.asarray(v) if not isinstance(v, dict) else {kk: jnp.asarray(vv) for kk, vv in v.items()}) for k, v in base.items()}

    def assemble(f):
        p = dict(fixed)
        enc = dict(p["enc"])
        enc["u"] = enc["u"].at[idx].set(f["u"])
        p["enc"] = enc
        if "log_gamma" in f:
            p["log_gamma"] = p["log_gamma"].at[idx].set(f["log_gamma"])
        return p

    def objective(f):
        return model.objective(assemble(f), sub)

    opt = optax.adam(REFINE_LR)

    @jax.jit
    def step(f, s):
        v, g = jax.value_and_grad(objective)(f)
        upd, s = opt.update(g, s, f)
        return optax.apply_updates(f, upd), s, v

    state = opt.init(free)
    val = None
    for _ in range(REFINE_STEPS):
        free, state, val = step(free, state)
    flat, unravel = jax.flatten_util.ravel_pytree(free)
    g = jax.grad(lambda x: objective(unravel(x)))
    hvp = jax.jit(lambda x, v: jax.jvp(g, (x,), (v,))[1])
    eye = np.eye(int(flat.shape[0]))
    diag = np.asarray([float(hvp(flat, jnp.asarray(eye[j]))[j]) for j in range(int(flat.shape[0]))])
    sd = 1.0 / np.sqrt(np.maximum(diag, 1.0))
    sd_tree = unravel(jnp.asarray(sd))
    out = {"u": np.asarray(free["u"], dtype=np.float64).tolist(), "sd": {"u": np.asarray(sd_tree["u"]).tolist()},
           "n_responses": int(sub.n), "objective": float(val), "steps": REFINE_STEPS}
    if "log_gamma" in free:
        out["log_gamma"] = float(free["log_gamma"])
        out["sd"]["log_gamma"] = float(sd_tree["log_gamma"])
    return out


# ---------------------------------------------------------------------------------------------
# Facade: profile
# ---------------------------------------------------------------------------------------------


def _state_of(artifact: ModelArtifact, cov: str, u: np.ndarray | None) -> tuple[float, float, float]:
    """Bloch polar angle, azimuth and ``P(sell | L = 0)`` of a Q-model state for a covariate record."""
    rec = S.json_loads_strict(cov)
    X, _, _ = covariate_matrix([rec], stats=artifact.covariate_stats)
    enc = artifact.params["enc"]
    z = X[0] @ np.asarray(enc["W"]) + np.asarray(enc["b"]) + (np.zeros(4) if u is None else np.asarray(u))
    psi = np.asarray(C.prepare_state(jnp.asarray(z)))
    theta = 2.0 * math.atan2(abs(psi[1]), abs(psi[0]))
    return float(theta), float(np.angle(psi[1])), float(abs(psi[1]) ** 2)


def _theta_norms(params: dict[str, Any], ctx_vocab: tuple[str, ...]) -> dict[str, float]:
    th = np.asarray(params["theta_ctx"], dtype=np.float64)
    return {t: float(np.linalg.norm(th[i])) for i, t in enumerate(ctx_vocab)}


def profile(
    artifact: ModelArtifact,
    covariates: dict[str, Any] | str | None,
    prior_responses: list[dict[str, Any]] | None = None,
    *,
    subject_id: str = "client",
    book: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """An investor's decision-state profile under the served model.

    ``baseline``: the state's Bloch angle and ``P(sell | L = 0, no context)`` (Q-models; the
    encoder output for the covariates plus the subject's random effect when known or refined
    from ``prior_responses``), with 80/95% intervals from the draws. ``sensitivities``: per
    context the rotation norm ``|theta_c|`` (population parameter, CI from the draws) and the
    predicted change in ``P(sell)`` at the risk-score loss when the context is applied singly.
    ``consistency``: the dephasing rate ``gamma`` (Q4): the subject's refined value with its
    Laplace interval when ``prior_responses`` are given, else the population median
    ``exp(gamma_mu)`` with its draw interval. ``risk_score``: the classical-equivalent score
    ``logit P(sell | TYPICAL_CRISIS_CONTEXTS, RISK_SCORE_LOSS)`` with CI, and ``r2_vs_angle``:
    the R² of that score on the state angle across ``book`` (client rows with ``client_id`` and
    ``covariates``; the artifact's training subjects keep their fitted effects) — None without a
    book. ``refined``: the per-subject effects when prior responses were given.
    """
    cov = covariates_text(covariates)
    sc = scorer_for(artifact)
    is_q = artifact.model_name in ("Q2", "Q4")
    effects: SubjectEffects | None = None
    refined: dict[str, Any] | None = None
    if prior_responses:
        refined = refine_subject(artifact, subject_id, cov, prior_responses)
        if "u" in refined:
            effects = {subject_id: {k: refined[k] for k in ("u", "log_gamma") if k in refined}}
    known = bool(artifact.subject_ids and subject_id in artifact.subject_ids)
    u = None
    if effects:
        u = np.asarray(effects[subject_id]["u"])
    elif known and is_q:
        u = np.asarray(artifact.params["enc"]["u"])[list(artifact.subject_ids).index(subject_id)]

    base_sc = Scenario(subject_id, cov, None, (), D.SCENARIO_FIRST, None, tag="baseline")
    risk_sc = Scenario(subject_id, cov, RISK_SCORE_LOSS, TYPICAL_CRISIS_CONTEXTS, D.SCENARIO_FIRST, None, tag="risk")
    singles = [Scenario(subject_id, cov, RISK_SCORE_LOSS, (t,), D.SCENARIO_FIRST, None, tag=f"single:{t}") for t in artifact.ctx_vocab]
    loss_only = Scenario(subject_id, cov, RISK_SCORE_LOSS, (), D.SCENARIO_FIRST, None, tag="loss_only")
    scen = [base_sc, risk_sc, loss_only] + singles
    p = sc.predict(scen, effects)
    smp = sc.predict_samples(scen, effects)

    out: dict[str, Any] = {**_model_meta(artifact), "subject_id": subject_id, "known_subject": known, "refined": refined}
    if is_q:
        theta, phi_az, p0 = _state_of(artifact, cov, u)
        out["baseline"] = {
            "bloch_theta": theta, "bloch_phi": phi_az, "p_sell_base": p0, "p_sell_base_model": float(p[0]),
            "ci80": _interval(None if smp is None else smp[:, 0], 0.80), "ci95": _interval(None if smp is None else smp[:, 0], 0.95),
            "note": "P(sell | L = 0, no context) = sin^2(bloch_theta / 2); the state is the encoder output for the covariates" + (" plus the subject's random effect" if u is not None else " with the random effect at its prior mean 0"),
        }
        norms = _theta_norms(artifact.params, artifact.ctx_vocab)
        norm_samples = None
        if sc.samples:
            norm_samples = np.stack([np.linalg.norm(np.asarray(s["theta_ctx"], dtype=np.float64), axis=1) for s in sc.samples])
        sens = {}
        for i, t in enumerate(artifact.ctx_vocab):
            col = None if norm_samples is None else norm_samples[:, i]
            j = 3 + i
            sens[t] = {
                "theta_norm": norms[t], "theta_norm_ci80": _interval(col, 0.80), "theta_norm_ci95": _interval(col, 0.95),
                "delta_p_at_risk_loss": float(p[j] - p[2]),
                "delta_p_ci80": _interval(None if smp is None else smp[:, j] - smp[:, 2], 0.80),
                "n_calibration": calibration_of(artifact, (t,))[t],
            }
        out["sensitivities"] = sens
        if artifact.model_name == "Q4":
            if refined and "log_gamma" in refined:
                lg, sd = refined["log_gamma"], refined["sd"]["log_gamma"]
                out["consistency"] = {"gamma": math.exp(lg), "ci80": [math.exp(lg - 1.2816 * sd), math.exp(lg + 1.2816 * sd)],
                                      "ci95": [math.exp(lg - 1.96 * sd), math.exp(lg + 1.96 * sd)], "source": "subject MAP from prior responses, diagonal Laplace interval"}
            else:
                g_s = None if not sc.samples else np.asarray([math.exp(float(s["gamma_mu"])) for s in sc.samples])
                g = math.exp(float(artifact.params["log_gamma"][list(artifact.subject_ids).index(subject_id)])) if known else math.exp(float(artifact.params["gamma_mu"]))
                out["consistency"] = {"gamma": g, "ci80": _interval(g_s, 0.80), "ci95": _interval(g_s, 0.95),
                                      "source": "fitted subject value" if known else "population median exp(gamma_mu) with its draw interval"}
        else:
            out["consistency"] = {"gamma": None, "note": f"{artifact.model_name} has no dephasing rate"}
    else:
        out["baseline"] = {"p_sell_base_model": float(p[0]), "ci80": _interval(None if smp is None else smp[:, 0], 0.80),
                           "ci95": _interval(None if smp is None else smp[:, 0], 0.95), "note": "classical model: no state angle"}
        out["sensitivities"] = {t: {"delta_p_at_risk_loss": float(p[3 + i] - p[2]), "n_calibration": calibration_of(artifact, (t,))[t]} for i, t in enumerate(artifact.ctx_vocab)}
        out["consistency"] = {"gamma": None, "note": "classical model"}
    score = _logit(p[1])
    score_s = None if smp is None else np.asarray([_logit(v) for v in smp[:, 1]])
    out["risk_score"] = {
        "value": score, "p_sell": float(p[1]), "ci80": _interval(score_s, 0.80), "ci95": _interval(score_s, 0.95),
        "definition": f"logit P(sell | contexts {list(TYPICAL_CRISIS_CONTEXTS)}, loss {RISK_SCORE_LOSS}, scenario-first)",
        "r2_vs_angle": None, "r2_note": "pass the book to compute the R^2 between the score and the state angle",
    }
    if book is not None and is_q and len(book) >= 3:
        r2 = risk_score_r2(artifact, book)
        out["risk_score"]["r2_vs_angle"] = r2["r2"]
        out["risk_score"]["r2_ci95"] = r2["ci95"]
        out["risk_score"]["r2_note"] = r2["note"]
    return jsonable(out)


def risk_score_r2(artifact: ModelArtifact, book: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict[str, Any]:
    """R² of the classical-equivalent risk score regressed on the Bloch angle across the book's
    clients (fitted effects for training subjects, ``u = 0`` otherwise), with a client-level
    bootstrap 95% interval (``n_boot`` resamples, fixed seed)."""
    sc = scorer_for(artifact)
    covs = [covariates_text(c) for c in book["covariates"].tolist()]
    ids = [str(c) for c in book["client_id"].tolist()]
    scen = [Scenario(i, c, RISK_SCORE_LOSS, TYPICAL_CRISIS_CONTEXTS, D.SCENARIO_FIRST, None, tag="risk") for i, c in zip(ids, covs)]
    p = sc.predict(scen)
    score = np.asarray([_logit(v) for v in p])
    known = artifact.subject_ids or ()
    U = np.asarray(artifact.params["enc"]["u"])
    angles = []
    for i, c in zip(ids, covs):
        u = U[list(known).index(i)] if i in known else None
        angles.append(_state_of(artifact, c, u)[0])
    angle = np.asarray(angles)

    def r2_of(idx):
        a, s = angle[idx], score[idx]
        if np.std(a) < 1e-12 or np.std(s) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, s)[0, 1] ** 2)

    r2 = r2_of(np.arange(len(ids)))
    rng = np.random.default_rng(seed)
    boots = [r2_of(rng.integers(0, len(ids), len(ids))) for _ in range(n_boot)]
    return {"r2": r2, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n_clients": len(ids),
            "note": f"R^2 of the linear regression of the risk score on the state angle over {len(ids)} book clients; bootstrap 95% interval over clients ({n_boot} resamples)"}


# ---------------------------------------------------------------------------------------------
# Facade: market state -> scenario
# ---------------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z][a-z\-']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def cause_frame_similarity(text: str, context: str) -> float:
    """``max(keyword score, character ratio)`` between a free-text cause frame and a news
    context: keyword score = ``min(1, hits / 2)`` over :data:`CAUSE_FRAME_KEYWORDS` (distinct
    token hits; a token counts when it starts with a keyword stem), character ratio =
    ``difflib.SequenceMatcher`` on the lower-cased text against the context sentence of
    :data:`bre.design.CONTEXT_SENTENCES`."""
    toks = set(_tokens(text or ""))
    bag = CAUSE_FRAME_KEYWORDS.get(context, ())
    hits = sum(1 for t in toks if any(t.startswith(k) for k in bag))
    kw = min(1.0, hits / 2.0)
    sentence = D.CONTEXT_SENTENCES.get(context, "").lower()
    ratio = difflib.SequenceMatcher(None, (text or "").lower().strip(), sentence).ratio() if text else 0.0
    return float(max(kw, ratio))


def match_cause_frame(text: str | None, candidates: tuple[str, ...] = CAUSE_CANDIDATES) -> dict[str, Any]:
    """Nearest candidate context to a free-text cause frame with the similarity and the
    ``weak_match`` flag (:data:`WEAK_MATCH_THRESHOLD`); ``context`` is None for empty text."""
    if not text or not str(text).strip() or not candidates:
        return {"context": None, "similarity": 0.0, "weak_match": False, "scores": {}, "text": text}
    scores = {c: cause_frame_similarity(str(text), c) for c in candidates}
    best = max(candidates, key=lambda c: (scores[c], -candidates.index(c)))
    return {"context": best, "similarity": scores[best], "weak_match": bool(scores[best] < WEAK_MATCH_THRESHOLD), "scores": scores, "text": text}


def _level(value: Any, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, str):
        key = value.strip().lower()
        if key in LEVELS:
            return LEVELS[key]
        try:
            value = float(key)
        except ValueError as exc:
            raise ValueError(f"unknown level {value!r}; use one of {list(LEVELS)} or a number in [0, 1]") from exc
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"level must be in [0, 1]; got {x}")
    return x


def market_state_to_contexts(market_state: dict[str, Any], calibrated_contexts: dict[str, dict[str, Any]] | None = None, n_positions: int = 2) -> dict[str, Any]:
    """Map the dashboard's market-state fields to a scenario.

    Fields (all optional; ``bre.market.latest_market_state`` supplies the first four):
    ``drawdown_pct`` (signed fraction, e.g. -0.12; a magnitude above 1 is read as percent),
    ``duration_days``, ``cause_frame`` (free text), ``recovery_pct`` (fraction from the trough),
    ``vix_bucket`` (``bre.market.VIX_BUCKET_LABELS``), ``media_intensity`` and
    ``social_cue_prevalence`` (:data:`LEVELS` labels or numbers in [0, 1]).

    Rules: ``loss_pct`` = the drawdown magnitude clipped to :data:`LOSS_RANGE` (``loss_clipped``
    says when; no drawdown gives the smallest design loss and a note); the cause frame maps to
    the nearest *calibrated* news context (:func:`match_cause_frame`; ``weak_match`` flags a poor
    match, and the context is still applied so the advisor sees what the model would do with
    it) when media intensity is at least :data:`MEDIA_MIN`; ``social:friend_sells`` enters when
    social-cue prevalence is at least :data:`SOCIAL_CUE_MIN`; ``market:recovered_5pct`` enters when
    the recovery is at least :data:`RECOVERY_MIN`. Order of arrival: news, social cue, recovery;
    the design holds at most ``n_positions`` contexts, so later ones are listed in
    ``dropped_contexts``. The VIX bucket only sets the media intensity default (high/extreme ->
    high, low -> low, else medium); the duration has no context in this design (Q3 is not
    served) and is carried in the notes.
    """
    ms = dict(market_state or {})
    notes: list[str] = []
    dd = ms.get("drawdown_pct")
    loss_raw: float | None = None
    if dd is not None and not (isinstance(dd, float) and math.isnan(dd)):
        x = float(dd)
        if abs(x) > 1.0:
            x = x / 100.0
            notes.append("drawdown_pct read as percent and divided by 100")
        loss_raw = -abs(x)
    if loss_raw is None:
        L = LOSS_RANGE[0]
        notes.append(f"no drawdown given: smallest design loss {LOSS_RANGE[0]:.2f} used")
        clipped = True
    else:
        L = min(max(abs(loss_raw), LOSS_RANGE[0]), LOSS_RANGE[1])
        clipped = abs(L - abs(loss_raw)) > 1e-12
        if clipped:
            notes.append(f"drawdown {loss_raw:.3f} clipped to the design range [{-LOSS_RANGE[1]:.2f}, {-LOSS_RANGE[0]:.2f}]")
    vix = ms.get("vix_bucket")
    media_default = {"high": 0.75, "extreme": 0.75, "low": 0.25}.get(str(vix).lower(), 0.5) if vix else 0.5
    media = _level(ms.get("media_intensity"), media_default)
    social = _level(ms.get("social_cue_prevalence"), 0.0)
    rec = ms.get("recovery_pct")
    rec = None if rec is None or (isinstance(rec, float) and math.isnan(rec)) else float(rec)
    if rec is not None and abs(rec) > 1.0:
        rec = rec / 100.0
        notes.append("recovery_pct read as percent and divided by 100")
    table = calibrated_contexts or {}
    candidates = tuple(c for c in CAUSE_CANDIDATES if str(table.get(c, {}).get("status", "calibrated")) == "calibrated") if table else CAUSE_CANDIDATES
    if not candidates:
        candidates = CAUSE_CANDIDATES
        notes.append("no calibrated news context; matching against all news contexts")
    match = match_cause_frame(ms.get("cause_frame"), candidates)
    ordered: list[str] = []
    if match["context"] is not None:
        if media is not None and media >= MEDIA_MIN:
            ordered.append(match["context"])
        else:
            notes.append("cause frame matched but media intensity is none: news context not applied")
    if social is not None and social >= SOCIAL_CUE_MIN:
        ordered.append("social:friend_sells")
    if rec is not None and rec >= RECOVERY_MIN:
        ordered.append("market:recovered_5pct")
    kept, dropped = ordered[:n_positions], ordered[n_positions:]
    if dropped:
        notes.append(f"design holds at most {n_positions} contexts; dropped {dropped}")
    if ms.get("duration_days") is not None:
        notes.append(f"duration_days={ms['duration_days']} carried only; elapsed time is not a context of the served design")
    return {
        "loss_pct": -float(L), "loss_pct_raw": loss_raw, "loss_clipped": bool(clipped),
        "context_tags": kept, "dropped_contexts": dropped, "question_order_id": D.SCENARIO_FIRST,
        "cause_context": match["context"], "cause_similarity": float(match["similarity"]), "weak_match": bool(match["weak_match"]),
        "cause_scores": match["scores"], "media_intensity": media, "social_cue_prevalence": social, "recovery_pct": rec,
        "vix_bucket": vix, "duration_days": ms.get("duration_days"), "notes": notes,
    }


# ---------------------------------------------------------------------------------------------
# Facade: interventions
# ---------------------------------------------------------------------------------------------


def apply_transform(context_tags: tuple[str, ...], question_order: str, transform: dict[str, Any], n_positions: int = 2) -> tuple[tuple[str, ...], str, list[str]]:
    """Apply an intervention's ``mapped_context_transform`` (``db.seed`` vocabulary: ``add``,
    ``remove``, ``question_order``) to a scenario. ``add`` appends tags not already present;
    when the result exceeds ``n_positions`` the earliest tags are dropped (the design holds the
    last two contexts). Returns ``(tags, order, notes)``."""
    tags = list(context_tags)
    notes: list[str] = []
    for t in transform.get("remove", []) or []:
        tags = [x for x in tags if x != t]
    for t in transform.get("add", []) or []:
        if t not in tags:
            tags.append(t)
    if len(tags) > n_positions:
        notes.append(f"transform produced {len(tags)} contexts; kept the last {n_positions}")
        tags = tags[-n_positions:]
    order = transform.get("question_order") or question_order
    return tuple(tags), order, notes


def _interventions_list(interventions: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in interventions or []:
        if isinstance(it, dict):
            d = dict(it)
        else:
            d = {"id": getattr(it, "id", None), "name": it.name, "script": it.script,
                 "mapped_context_transform": it.mapped_context_transform, "active": getattr(it, "active", True)}
        tr = d.get("mapped_context_transform", {})
        d["mapped_context_transform"] = json.loads(tr) if isinstance(tr, str) else dict(tr or {})
        out.append(d)
    return out


def rank_interventions(
    artifact: ModelArtifact,
    covariates: dict[str, Any] | str | None,
    market_state: dict[str, Any],
    interventions: Any,
    *,
    subject_id: str = "client",
    subject_effects: SubjectEffects | None = None,
) -> dict[str, Any]:
    """Interventions ranked by the predicted change in ``P(sell)`` under the market state's
    scenario (most negative change first); each entry carries the change with its 80% draw
    interval (paired per draw) and the label :data:`INTERVENTION_LABEL`. Inactive interventions
    are listed but not ranked."""
    scen = market_state_to_contexts(market_state, artifact.calibrated_contexts, artifact.n_ctx_positions)
    tags = check_context_tags(scen["context_tags"], artifact.ctx_vocab, artifact.n_ctx_positions)
    cov = covariates_text(covariates)
    sc = scorer_for(artifact)
    items = _interventions_list(interventions)
    active = [it for it in items if it.get("active", True)]
    base = Scenario(subject_id, cov, scen["loss_pct"], tags, scen["question_order_id"], None, tag="main")
    alts: list[Scenario] = []
    applied: list[tuple[tuple[str, ...], str, list[str]]] = []
    for it in active:
        t2, o2, notes = apply_transform(tags, scen["question_order_id"], it["mapped_context_transform"], artifact.n_ctx_positions)
        applied.append((t2, o2, notes))
        alts.append(Scenario(subject_id, cov, scen["loss_pct"], t2, o2, None, tag=f"iv:{it['name']}"))
    p = sc.predict([base] + alts, subject_effects)
    smp = sc.predict_samples([base] + alts, subject_effects)
    ranked = []
    for j, it in enumerate(active):
        delta = float(p[j + 1] - p[0])
        d_s = None if smp is None else smp[:, j + 1] - smp[:, 0]
        t2, o2, notes = applied[j]
        ranked.append({
            "name": it["name"], "id": it.get("id"), "script": it.get("script"), "transform": it["mapped_context_transform"],
            "p_before": float(p[0]), "p_after": float(p[j + 1]), "delta_p": delta,
            "delta_ci80": _interval(d_s, 0.80), "delta_ci95": _interval(d_s, 0.95),
            "contexts_after": list(t2), "question_order_after": o2, "notes": notes, "label": INTERVENTION_LABEL,
            "no_change": bool(t2 == tags and o2 == scen["question_order_id"]),
        })
    ranked.sort(key=lambda r: (r["delta_p"], r["name"]))
    for k, r in enumerate(ranked):
        r["rank"] = k + 1
    return jsonable({
        "scenario": scen, "p_before": float(p[0]), "ci80_before": _interval(None if smp is None else smp[:, 0], 0.80),
        "ranked": ranked, "inactive": [it["name"] for it in items if not it.get("active", True)],
        "label": INTERVENTION_LABEL, "n_calibration": calibration_of(artifact, tags), **_model_meta(artifact),
    })


# ---------------------------------------------------------------------------------------------
# Facade: book scoring
# ---------------------------------------------------------------------------------------------


def drawdown_capacity(p_grid: np.ndarray, target: float = DRAWDOWN_TARGET) -> tuple[float, str]:
    """Behavioral drawdown capacity (first-crossing definition, :data:`CAPACITY_DEFINITION`).

    ``p_grid`` holds ``P(sell | L, typical crisis)`` over ``design.LOSS_PCTS`` in ascending
    magnitude. The capacity is the largest grid loss ``L`` such that ``p <= target`` at *every*
    grid loss ``L' <= L`` (a prefix rule), not the largest ``L`` with ``p(L) <= target``: the
    loss unitary is a rotation, so ``P(sell | L)`` may come back under the target at a larger
    loss, and an investor predicted to sell at 15% is not credited with sitting through 30%.
    Returns ``(capacity, status)`` with capacity 0.0 and status ``"below smallest grid loss"``
    when even the smallest loss crosses, ``"at grid maximum"`` when none does.
    """
    losses = sorted(abs(x) for x in D.LOSS_PCTS)
    cap = 0.0
    for L, p in zip(losses, p_grid):
        if p <= target:
            cap = L
        else:
            break
    if cap == 0.0:
        return 0.0, "below smallest grid loss"
    if cap >= losses[-1] - 1e-12:
        return cap, "at grid maximum"
    return cap, "within grid"


LOSS_RESPONSE_DEFINITION = (
    "share of the book's clients whose predicted P(sell | L, no context, scenario-first) is non-decreasing "
    "over the five design loss levels {grid} (each client with its fitted random effects when known to the "
    "model, else at the population mean), and the mean predicted P(sell) per loss level across the book. "
    "A diagnostic of the served model's loss response, not a test of the data: the loss enters as a rotation "
    "of the state, so a non-monotone response is possible under the model."
)


def loss_response_diagnostic(artifact: ModelArtifact, book: pd.DataFrame, *, subject_effects: SubjectEffects | None = None) -> dict[str, Any]:
    """Loss-response diagnostic of a book (:data:`LOSS_RESPONSE_DEFINITION`): ``monotone_share``
    (share of clients with a non-decreasing ``P(sell | L, no context)`` over ``design.LOSS_PCTS``),
    ``mean_p_sell_by_loss`` (per level, across the book), ``p_sell_by_loss_ci80`` (10th-90th
    percentile across clients per level), ``n_clients``, ``loss_levels`` and the definition.
    ``book`` needs ``client_id`` and ``covariates``."""
    ids = [str(c) for c in book["client_id"].tolist()]
    covs = [covariates_text(c) for c in book["covariates"].tolist()]
    losses = [float(x) for x in D.LOSS_PCTS]
    if not ids:
        return {"monotone_share": None, "n_clients": 0, "n_monotone": 0, "loss_levels": losses, "mean_p_sell_by_loss": None, "p_sell_by_loss_ci80": None,
                "context": "none", "question_order_id": D.SCENARIO_FIRST, "definition": LOSS_RESPONSE_DEFINITION.format(grid=losses), **_model_meta(artifact)}
    sc = scorer_for(artifact)
    scen = [Scenario(cid, cov, L, (), D.SCENARIO_FIRST, None, tag=f"loss:{L}") for cid, cov in zip(ids, covs) for L in losses]
    p = sc.predict(scen, subject_effects).reshape(len(ids), len(losses))
    monotone = np.all(np.diff(p, axis=1) >= -1e-12, axis=1)
    lo, hi = np.percentile(p, [10, 90], axis=0)
    return jsonable({
        "monotone_share": float(monotone.mean()), "n_clients": len(ids), "n_monotone": int(monotone.sum()), "loss_levels": losses,
        "mean_p_sell_by_loss": p.mean(axis=0).tolist(), "p_sell_by_loss_ci80": [lo.tolist(), hi.tolist()],
        "context": "none", "question_order_id": D.SCENARIO_FIRST, "definition": LOSS_RESPONSE_DEFINITION.format(grid=losses), **_model_meta(artifact),
    })


def score_book(
    artifact: ModelArtifact,
    clients: pd.DataFrame,
    market_state: dict[str, Any],
    interventions: Any = None,
    *,
    target: float = DRAWDOWN_TARGET,
    crisis_contexts: tuple[str, ...] | list[str] | None = None,
    status_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Triage table of a client book under one market state (one vectorized forward pass over
    every client and counterfactual, one draw pass over the main rows).

    ``crisis_contexts`` (default :data:`TYPICAL_CRISIS_CONTEXTS`) is the ordered context sequence
    of the drawdown capacity and ``status_thresholds`` (default :data:`STATUS_THRESHOLDS`) the
    ``elevated`` / ``high`` boundaries of ``status``; both are the dashboard's editable settings
    (``GET /settings``) and are echoed in ``result.attrs``.

    ``clients`` needs ``client_id`` and ``covariates`` (dict or JSON text; ``display_label``
    optional). Columns: ``p_sell`` (predicted probability of selling under the market state),
    ``p_baseline`` (``L = 0``, no context), ``change_vs_baseline``, ``ci80_low`` / ``ci80_high``,
    ``top_driver`` / ``top_driver_delta``, ``suggested_intervention`` / ``suggested_delta`` (the
    active intervention with the largest predicted reduction; :data:`INTERVENTION_LABEL`),
    ``drawdown_capacity`` / ``capacity_status`` (:func:`drawdown_capacity` at ``target``),
    ``status`` (:data:`STATUS_THRESHOLDS`), ``known_client`` (fitted effects used),
    ``synthetic``. The scenario, notes and calibration are in ``result.attrs``.
    """
    scen = market_state_to_contexts(market_state, artifact.calibrated_contexts, artifact.n_ctx_positions)
    tags = check_context_tags(scen["context_tags"], artifact.ctx_vocab, artifact.n_ctx_positions)
    loss = scen["loss_pct"]
    order = scen["question_order_id"]
    ids = [str(c) for c in clients["client_id"].tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError("client_id values must be unique")
    covs = [covariates_text(c) for c in clients["covariates"].tolist()]
    if "display_label" in clients.columns:
        # pandas' string dtype stores a missing label as NaN; the table carries None
        labels = [None if lbl is None or (isinstance(lbl, float) and math.isnan(lbl)) else str(lbl) for lbl in clients["display_label"].tolist()]
    else:
        labels = list(ids)
    items = [it for it in _interventions_list(interventions) if it.get("active", True)]
    grid = sorted(abs(x) for x in D.LOSS_PCTS)
    crisis = TYPICAL_CRISIS_CONTEXTS if crisis_contexts is None else check_context_tags(list(crisis_contexts), artifact.ctx_vocab, artifact.n_ctx_positions)
    thresholds = STATUS_THRESHOLDS if status_thresholds is None else {**STATUS_THRESHOLDS, **{k: float(v) for k, v in status_thresholds.items()}}
    sc = scorer_for(artifact)

    scen_list: list[Scenario] = []
    layout: list[dict[str, Any]] = []
    for cid, cov in zip(ids, covs):
        start = len(scen_list)
        scen_list.append(Scenario(cid, cov, loss, tags, order, None, tag="main"))
        scen_list.append(Scenario(cid, cov, None, (), D.SCENARIO_FIRST, None, tag="baseline"))
        drivers = _driver_scenarios(cid, cov, loss, tags, order, None)
        scen_list.extend(drivers)
        iv_start = len(scen_list)
        for it in items:
            t2, o2, _ = apply_transform(tags, order, it["mapped_context_transform"], artifact.n_ctx_positions)
            scen_list.append(Scenario(cid, cov, loss, t2, o2, None, tag=f"iv:{it['name']}"))
        cap_start = len(scen_list)
        for L in grid:
            scen_list.append(Scenario(cid, cov, -L, crisis, D.SCENARIO_FIRST, None, tag=f"cap:{L}"))
        layout.append({"main": start, "base": start + 1, "drivers": (start + 2, iv_start), "iv": (iv_start, cap_start), "cap": (cap_start, len(scen_list))})

    p = sc.predict(scen_list)
    mains = [scen_list[l["main"]] for l in layout]
    smp = sc.predict_samples(mains)
    calibration = calibration_of(artifact, tags)
    known = set(artifact.subject_ids or ())
    rows = []
    for i, (cid, label, l) in enumerate(zip(ids, labels, layout)):
        pm = float(p[l["main"]])
        pb = float(p[l["base"]])
        d0, d1 = l["drivers"]
        drivers = scen_list[d0:d1]
        td = _top_driver(pm, drivers, p[d0:d1]) if drivers else {"driver": None, "delta_p": 0.0}
        i0, i1 = l["iv"]
        if items:
            deltas = p[i0:i1] - pm
            j = int(np.argmin(deltas))
            best_name, best_delta = items[j]["name"], float(deltas[j])
        else:
            best_name, best_delta = None, None
        c0, c1 = l["cap"]
        cap, cap_status = drawdown_capacity(p[c0:c1], target)
        ci = _interval(None if smp is None else smp[:, i], 0.80)
        rows.append({
            "client_id": cid, "display_label": label, "p_sell": pm, "p_baseline": pb, "change_vs_baseline": pm - pb,
            "ci80_low": None if ci is None else ci[0], "ci80_high": None if ci is None else ci[1],
            "top_driver": td["driver"], "top_driver_delta": float(td["delta_p"]),
            "suggested_intervention": best_name, "suggested_delta": best_delta,
            "drawdown_capacity": cap, "capacity_status": cap_status,
            "status": _status(pm, calibration, thresholds), "known_client": cid in known, "synthetic": bool(artifact.is_synthetic_training),
        })
    out = pd.DataFrame(rows)
    for col in ("display_label", "top_driver", "suggested_intervention"):
        out[col] = pd.Series([r[col] for r in rows], dtype="object")  # keep None: the string dtype would turn it into NaN
    out.attrs.update({
        "scenario": scen, "n_calibration": calibration, "target": target, "interval": "80% " + INTERVAL_SOURCE if smp is not None else _no_samples_reason(artifact),
        "intervention_label": INTERVENTION_LABEL,
        "capacity_definition": CAPACITY_DEFINITION.format(grid=[-x for x in grid], contexts=list(crisis), target=target),
        "crisis_contexts": list(crisis), "status_thresholds": dict(thresholds),
        "model_version": artifact.version, "model_type": artifact.model_name, "synthetic": bool(artifact.is_synthetic_training), "wording": PROBABILITY_WORDING,
    })
    return out


__all__ = [
    "ARTIFACT_REF_PREFIX",
    "CAPACITY_DEFINITION",
    "CAUSE_FRAME_KEYWORDS",
    "DRAWDOWN_TARGET",
    "INTERVAL_SOURCE",
    "INTERVENTION_LABEL",
    "LEVELS",
    "LOSS_RANGE",
    "LOSS_RESPONSE_DEFINITION",
    "PROBABILITY_WORDING",
    "RISK_SCORE_LOSS",
    "STATUS_THRESHOLDS",
    "Scenario",
    "Scorer",
    "TYPICAL_CRISIS_CONTEXTS",
    "WARM_UP_BOOK_SIZES",
    "WEAK_MATCH_THRESHOLD",
    "active_registry_row",
    "apply_transform",
    "artifact_path_of",
    "calibration_of",
    "cause_frame_similarity",
    "check_context_tags",
    "covariates_text",
    "drawdown_capacity",
    "load_active_artifact",
    "load_artifact",
    "loss_response_diagnostic",
    "market_state_to_contexts",
    "match_cause_frame",
    "predict_sell",
    "profile",
    "prior_responses_frame",
    "rank_interventions",
    "refine_subject",
    "risk_score_r2",
    "scenario_frame",
    "score_book",
    "scorer_for",
    "warm_up",
]
