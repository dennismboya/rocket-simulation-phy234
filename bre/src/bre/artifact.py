"""Model artifacts: what a fit leaves behind and what the API loads (PLAN.md sections 3-5).

A :class:`ModelArtifact` holds one fitted model in a form that can be saved, reloaded and used to
score subjects the model has never seen, with every number traceable to a parameter block and a
training-data reference. The layout on disk is::

    <dir>/artifact.json    every scalar/string field, metrics, calibrated_contexts, ctx_vocab, ...
    <dir>/params.npz       the parameter pytree flattened with "/"-joined keys (numpy float64)
    <dir>/samples.npz      optional: K parameter draws, same keys, leading axis K

Every leaf is stored as float64 (complex128 for complex leaves). B5's leaves are pickled
scikit-learn estimators as ``uint8`` byte arrays: the values 0..255 are exact in float64, so
the artifact round-trips them without a special case and ``B5.estimators`` (``unpickle_blob``)
casts the reloaded float64 array back to bytes. ``n_params`` is the model's own count
(``model.n_params``, through ``registry.param_counts(name, params, model)``): for B5 that is
the number of tree leaves plus MLP weights, not the byte count of the blobs.

Parameter draws (``param_samples``) carry the uncertainty the dashboard shows as intervals:

* B2 (hierarchical Bayesian): posterior draws from the fitted guide / sampler, in the protocol
  layout (``B2.params_from_draw``).
* MAP-fitted models (B1, B4, Q2, Q4): draws from a **diagonal Laplace approximation** at the
  optimum (:func:`bre.fit.laplace_samples`): the population-level parameters ``theta_pop`` are
  given a Gaussian ``N(theta_pop*, diag(1 / H_jj))`` with ``H`` the Hessian of the negative
  log-posterior (``model.objective``) with respect to the population block, computed with
  ``jax.hessian`` at the optimum and with the per-subject blocks (random effects) held at their
  fitted values. Limits, stated once here and in the artifact ``notes``: (1) it is a local
  quadratic approximation around one optimum and knows nothing about other modes or the discrete
  gauge copies of the Q-models; (2) the diagonal ignores every posterior correlation, so
  intervals of correlated parameters (e.g. ``W`` and ``b``) are too narrow or too wide
  individually; (3) unidentified directions have zero curvature (e.g. a ``sigma_z`` component of
  a last unitary, INTERFACE.md section 7) and are floored at ``LAPLACE_MIN_CURVATURE`` / capped
  at ``LAPLACE_MAX_SD`` so that a draw stays finite; (4) per-subject random effects are not
  resampled, so a training subject's interval is conditional on its fitted ``u``; (5) it is a
  posterior under the model's own priors (ridge / random-effect scale), not a bootstrap.
* Externally fitted models (B5): no draws (``param_samples = None``; the ``notes`` say so). The
  dashboard shows a point prediction without an interval for such a model.

Scoring new subjects: :func:`predict_rows` builds a ``ModelData`` from schema rows with the
artifact's context vocabulary and covariate standardization (``build_model_data(...,
ctx_vocab=, covariate_stats=)``), builds the model on it, resizes the per-subject blocks with
:func:`params_for_data` (``u = 0`` and the population values for every subject not in the
training set; a training subject keeps its fitted values when ``subject_ids`` are stored) and
returns ``model.predict_proba``. :func:`predict_rows_samples` does the same for every stored
parameter draw and returns a ``(K, n)`` matrix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre import design as _design
from bre.models.data import ModelData, N_CTX_POSITIONS_DEFAULT, build_model_data
from bre.registry import MODEL_FAMILY, MODEL_REGISTRY, PER_SUBJECT_BLOCKS, make_model

ARTIFACT_VERSION = "1.0.0"
"""Version of the on-disk layout described in the module docstring."""

ARTIFACT_JSON = "artifact.json"
PARAMS_NPZ = "params.npz"
SAMPLES_NPZ = "samples.npz"

KEY_SEP = "/"
"""Separator of the nested pytree keys inside the ``.npz`` files."""

CALIBRATED = "calibrated"
UNCALIBRATED = "uncalibrated: prior only"
"""The two ``status`` values of ``calibrated_contexts``."""


# ---------------------------------------------------------------------------------------------
# Pytree <-> flat numpy dicts
# ---------------------------------------------------------------------------------------------


def flatten_params(params: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Nested dict of arrays -> ``{"enc/W": array, ...}`` (numpy float64/complex copies)."""
    out: dict[str, np.ndarray] = {}
    if isinstance(params, dict):
        for k, v in params.items():
            out.update(flatten_params(v, f"{prefix}{k}{KEY_SEP}"))
        return out
    key = prefix[: -len(KEY_SEP)] if prefix.endswith(KEY_SEP) else prefix
    arr = np.asarray(params)
    if np.iscomplexobj(arr):
        out[key] = np.asarray(arr, dtype=np.complex128)
    else:
        out[key] = np.asarray(arr, dtype=np.float64)
    return out


def unflatten_params(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Inverse of :func:`flatten_params`."""
    out: dict[str, Any] = {}
    for key, arr in flat.items():
        parts = key.split(KEY_SEP)
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = np.asarray(arr)
    return out


def to_numpy_tree(params: Any) -> Any:
    """Every leaf of a (nested dict) pytree as a numpy array."""
    if isinstance(params, dict):
        return {k: to_numpy_tree(v) for k, v in params.items()}
    return np.asarray(params)


def stack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """``K`` pytrees with identical structure -> flat dict with a leading ``K`` axis per key."""
    flats = [flatten_params(s) for s in samples]
    keys = list(flats[0])
    for f in flats[1:]:
        if list(f) != keys:
            raise ValueError("parameter samples do not share one pytree structure")
    return {k: np.stack([f[k] for f in flats]) for k in keys}


def unstack_samples(stacked: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Inverse of :func:`stack_samples`."""
    if not stacked:
        return []
    K = int(next(iter(stacked.values())).shape[0])
    return [unflatten_params({k: v[i] for k, v in stacked.items()}) for i in range(K)]


# ---------------------------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------------------------


@dataclass
class ModelArtifact:
    """One fitted model, its provenance and its uncertainty (module docstring).

    Fields required by the API contract come first; ``subject_ids`` (training subjects, aligned
    with the per-subject parameter blocks) and ``n_ctx_positions`` are optional extras with
    defaults so that the contract fields alone construct a valid artifact.
    """

    version: str
    model_name: str
    family: str
    params: dict[str, Any]
    param_samples: list[dict[str, Any]] | None
    ctx_vocab: tuple[str, ...]
    covariate_stats: dict[str, tuple[float, float]]
    X_columns: tuple[str, ...]
    design_version: str
    training_data_refs: list[str]
    metrics: dict[str, Any]
    calibrated_contexts: dict[str, dict[str, Any]]
    n_params: int
    created_at: str
    is_synthetic_training: bool
    notes: str
    subject_ids: tuple[str, ...] | None = None
    n_ctx_positions: int = N_CTX_POSITIONS_DEFAULT
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model_name {self.model_name!r}; registered {sorted(MODEL_REGISTRY)}")
        if self.family != MODEL_FAMILY[self.model_name]:
            raise ValueError(f"family {self.family!r} does not match model {self.model_name} ({MODEL_FAMILY[self.model_name]})")
        self.params = to_numpy_tree(self.params)
        if self.param_samples is not None:
            self.param_samples = [to_numpy_tree(s) for s in self.param_samples]
        self.ctx_vocab = tuple(self.ctx_vocab)
        self.X_columns = tuple(self.X_columns)
        self.covariate_stats = {k: (float(v[0]), float(v[1])) for k, v in self.covariate_stats.items()}
        self.training_data_refs = list(self.training_data_refs)
        if self.subject_ids is not None:
            self.subject_ids = tuple(str(s) for s in self.subject_ids)
        for tag, entry in self.calibrated_contexts.items():
            if entry.get("status") not in (CALIBRATED, UNCALIBRATED):
                raise ValueError(f"calibrated_contexts[{tag!r}].status must be {CALIBRATED!r} or {UNCALIBRATED!r}")

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def n_samples(self) -> int:
        return 0 if self.param_samples is None else len(self.param_samples)

    @property
    def d_x(self) -> int:
        return len(self.X_columns)

    # -- persistence -----------------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Every field except the parameter arrays, JSON-ready."""
        d = asdict(self)
        d.pop("params")
        d.pop("param_samples")
        d["ctx_vocab"] = list(self.ctx_vocab)
        d["X_columns"] = list(self.X_columns)
        d["covariate_stats"] = {k: [v[0], v[1]] for k, v in self.covariate_stats.items()}
        d["subject_ids"] = None if self.subject_ids is None else list(self.subject_ids)
        d["n_samples"] = self.n_samples
        d["param_keys"] = sorted(flatten_params(self.params))
        d["metrics"] = _jsonable(self.metrics)
        d["calibrated_contexts"] = _jsonable(self.calibrated_contexts)
        d["model_kwargs"] = _jsonable(self.model_kwargs)
        return d

    def save(self, directory: str | Path) -> Path:
        """Write ``artifact.json``, ``params.npz`` and, with draws, ``samples.npz``; returns the dir."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ARTIFACT_JSON).write_text(json.dumps(self.to_json_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        np.savez(directory / PARAMS_NPZ, **flatten_params(self.params))
        samples_path = directory / SAMPLES_NPZ
        if self.param_samples:
            np.savez(samples_path, **stack_samples(self.param_samples))
        elif samples_path.exists():
            samples_path.unlink()
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "ModelArtifact":
        """Read an artifact written by :meth:`save`."""
        directory = Path(directory)
        meta = json.loads((directory / ARTIFACT_JSON).read_text(encoding="utf-8"))
        with np.load(directory / PARAMS_NPZ) as z:
            params = unflatten_params({k: z[k] for k in z.files})
        samples = None
        samples_path = directory / SAMPLES_NPZ
        if samples_path.exists():
            with np.load(samples_path) as z:
                samples = unstack_samples({k: z[k] for k in z.files})
        fields_ = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in meta.items() if k in fields_}
        kwargs["params"] = params
        kwargs["param_samples"] = samples
        kwargs["covariate_stats"] = {k: (v[0], v[1]) for k, v in meta["covariate_stats"].items()}
        if meta.get("subject_ids") is not None:
            kwargs["subject_ids"] = tuple(meta["subject_ids"])
        return cls(**kwargs)

    # -- models and data -------------------------------------------------------------------------

    def template_data(self, n_subjects: int = 1) -> ModelData:
        """An empty ``ModelData`` (no rows, ``n_subjects`` placeholder subjects with zero
        covariates) carrying this artifact's vocabulary and covariate columns, so that a model
        can be instantiated without data (:meth:`model_instance`)."""
        n = 0
        ids = np.asarray([f"_template{i}" for i in range(n_subjects)], dtype=str)
        return ModelData(
            subject_ids=ids,
            subject_idx=np.zeros(n, dtype=np.int64),
            X=np.zeros((n_subjects, self.d_x), dtype=np.float64),
            X_columns=tuple(self.X_columns),
            covariate_stats=dict(self.covariate_stats),
            loss=np.zeros(n),
            loss_signed=np.zeros(n),
            ctx_idx=np.full((n, self.n_ctx_positions), -1, dtype=np.int64),
            ctx_mask=np.zeros((n, self.n_ctx_positions), dtype=bool),
            ctx_vocab=tuple(self.ctx_vocab),
            order_flag=np.zeros(n, dtype=np.int64),
            tol_answer=np.zeros(n),
            y=np.zeros(n),
            w=np.zeros(n),
            row_kind=np.zeros(n, dtype="<U5"),
            dataset=np.zeros(n, dtype=str),
            session_id=np.zeros(n, dtype=str),
            position=np.zeros(n, dtype=np.int64),
            is_synthetic=np.zeros(n, dtype=bool),
            index=np.zeros(n, dtype=np.int64),
        )

    def model_instance(self, data: ModelData | None = None):
        """The model object (``bre.models.base.Model``) of this artifact.

        Every model is sized by the ``ModelData`` it is built on, so pass the data you intend to
        score (``build_data`` builds it with this artifact's vocabulary); without ``data`` the
        model is built on :meth:`template_data` and serves the data-independent methods
        (``B1.coefficients``, ``Q2.context_table``, ``B4.natural_params``, ...).
        """
        if data is not None and tuple(data.ctx_vocab) != self.ctx_vocab:
            raise ValueError("data.ctx_vocab differs from the artifact's ctx_vocab; build it with build_data()")
        if data is not None and tuple(data.X_columns) != self.X_columns:
            raise ValueError("data.X_columns differ from the artifact's X_columns")
        return make_model(self.model_name, self.template_data() if data is None else data, **self.model_kwargs)

    def build_data(self, df: pd.DataFrame, **kwargs: Any) -> ModelData:
        """``build_model_data(df, ctx_vocab=self.ctx_vocab, covariate_stats=self.covariate_stats,
        require_consent=False, ...)``: rows to score are not training rows, so the consent filter
        is off by default (pass ``require_consent=True`` to keep it)."""
        kwargs.setdefault("require_consent", False)
        kwargs.setdefault("n_ctx_positions", self.n_ctx_positions)
        return build_model_data(df, ctx_vocab=self.ctx_vocab, covariate_stats=self.covariate_stats, **kwargs)

    def params_for(self, data: ModelData, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """:func:`params_for_data` of ``params`` (default: this artifact's) for ``data``."""
        return params_for_data(self.model_name, self.params if params is None else params, data, self.subject_ids)

    def population_params(self) -> dict[str, Any]:
        """The parameters without the per-subject blocks (what a new client is scored with)."""
        out = to_numpy_tree(self.params)
        for path in PER_SUBJECT_BLOCKS[self.model_name]:
            node = out
            for p in path[:-1]:
                node = node[p]
            node.pop(path[-1], None)
        return out

    def summary(self) -> dict[str, Any]:
        """Plain dict for logs and the transparency page."""
        return {
            "model_name": self.model_name,
            "family": self.family,
            "n_params": self.n_params,
            "n_samples": self.n_samples,
            "ctx_vocab": list(self.ctx_vocab),
            "training_data_refs": list(self.training_data_refs),
            "is_synthetic_training": self.is_synthetic_training,
            "design_version": self.design_version,
            "created_at": self.created_at,
            "metrics": _jsonable(self.metrics),
            "calibrated_contexts": _jsonable(self.calibrated_contexts),
        }


# ---------------------------------------------------------------------------------------------
# Per-subject blocks for new data
# ---------------------------------------------------------------------------------------------


def _fill_value(rule: str, params: dict[str, Any], block: np.ndarray, n_new: int) -> np.ndarray:
    """The block rows of ``n_new`` unseen subjects under ``rule`` (registry.PER_SUBJECT_BLOCKS)."""
    trailing = block.shape[1:]
    if rule == "zero":
        return np.zeros((n_new,) + trailing, dtype=block.dtype)
    if rule == "mu_ctx":
        mu = np.asarray(params["mu_ctx"], dtype=np.float64)
        return np.broadcast_to(mu, (n_new,) + mu.shape).copy()
    if rule == "gamma_mu":
        return np.full((n_new,) + trailing, float(np.asarray(params["gamma_mu"])), dtype=np.float64)
    raise ValueError(f"unknown fill rule {rule!r}")


def params_for_data(
    model_name: str,
    params: dict[str, Any],
    data: ModelData,
    training_subject_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Resize the per-subject blocks of ``params`` to ``data.subject_ids``.

    Subjects that appear in ``training_subject_ids`` keep their fitted rows; every other subject
    gets the unseen-subject fill of ``registry.PER_SUBJECT_BLOCKS`` (``u = 0``, ``beta_ctx =
    mu_ctx``, ``log_gamma = gamma_mu``). Without ``training_subject_ids`` every subject is
    treated as unseen (``u = 0`` for all: the API rule for new clients). Models without
    per-subject blocks return a numpy copy of ``params``.
    """
    out = to_numpy_tree(params)
    blocks = PER_SUBJECT_BLOCKS[model_name]
    if not blocks:
        return out
    new_ids = [str(s) for s in data.subject_ids]
    known = {} if training_subject_ids is None else {str(s): i for i, s in enumerate(training_subject_ids)}
    src_idx = np.asarray([known.get(s, -1) for s in new_ids], dtype=np.int64)
    for path, rule in blocks.items():
        node = out
        for p in path[:-1]:
            node = node[p]
        block = np.asarray(node[path[-1]])
        filled = _fill_value(rule, out, block, len(new_ids))
        keep = src_idx >= 0
        if keep.any():
            if training_subject_ids is not None and block.shape[0] != len(training_subject_ids):
                raise ValueError(f"block {'/'.join(path)} has {block.shape[0]} rows but {len(training_subject_ids)} training subject ids")
            filled[keep] = block[src_idx[keep]]
        node[path[-1]] = filled
    return out


# ---------------------------------------------------------------------------------------------
# Scoring rows
# ---------------------------------------------------------------------------------------------


def predict_rows(artifact: ModelArtifact, df: pd.DataFrame, *, params: dict[str, Any] | None = None, **data_kwargs: Any) -> np.ndarray:
    """``P(sell)`` for every row of a schema frame ``df`` (``bre.schema.COLUMNS``; ``response``
    may be null for rows to score), built with the artifact's vocabulary and covariate
    standardization and scored with ``u = 0`` for subjects outside the training set. Returns a
    ``(n,)`` float array in frame order (rows of every kind get a value; only sell rows are
    meaningful, as in ``Model.predict_proba``)."""
    data = artifact.build_data(df, **data_kwargs)
    model = artifact.model_instance(data)
    p = model.predict_proba(artifact.params_for(data, params), data)
    return np.asarray(p, dtype=np.float64)


def predict_rows_samples(artifact: ModelArtifact, df: pd.DataFrame, **data_kwargs: Any) -> np.ndarray:
    """``(K, n)`` ``P(sell)`` under each of the artifact's ``param_samples`` (module docstring
    for what the draws mean per model family); ``(0, n)`` when the artifact has none."""
    data = artifact.build_data(df, **data_kwargs)
    model = artifact.model_instance(data)
    if not artifact.param_samples:
        return np.zeros((0, data.n), dtype=np.float64)
    rows = [np.asarray(model.predict_proba(artifact.params_for(data, s), data), dtype=np.float64) for s in artifact.param_samples]
    return np.stack(rows)


# ---------------------------------------------------------------------------------------------
# Calibrated-context bookkeeping (used by bre.fit)
# ---------------------------------------------------------------------------------------------


def context_parameter(model_name: str, params: dict[str, Any], ctx_vocab: tuple[str, ...], tag: str) -> list[float] | None:
    """The parameter block that carries context ``tag`` in ``params``, as a flat list:
    Q2/Q4 the ``theta_ctx`` row (3 su(2) components); B2 ``mu_ctx[c]`` (one value); B4
    ``lambda[c]`` (one value); B3 ``[rho_ctx[c], pr_ctx[c]]`` (reference-point shift and
    recovery-belief term); Q3 ``w_ctx[c]`` (dissonance weight); Q5 ``b_ctx[c]`` (attraction
    term); B1 the per-position main-effect weights ``ctx{k}={tag}`` (K values, read by column
    name from ``B1.columns`` when the model instance is given through ``params["_columns"]``,
    else None); None for B5 and B6 (B5 has no named context parameter; B6's context weights
    are per position and per state, reported through ``B6.natural_params``)."""
    vocab = tuple(ctx_vocab)
    if tag not in vocab:
        return None
    c = vocab.index(tag)
    if model_name in ("Q2", "Q4"):
        return [float(v) for v in np.asarray(params["theta_ctx"])[c]]
    if model_name == "B2":
        return [float(np.asarray(params["mu_ctx"])[c])]
    if model_name == "B4":
        return [float(np.asarray(params["lambda"])[c])]
    if model_name == "B3":
        return [float(np.asarray(params["rho_ctx"])[c]), float(np.asarray(params["pr_ctx"])[c])]
    if model_name == "Q3":
        return [float(np.asarray(params["w_ctx"])[c])]
    if model_name == "Q5":
        return [float(np.asarray(params["b_ctx"])[c])]
    if model_name == "B1":
        cols = params.get("_columns")
        if cols is None:
            return None
        w = np.asarray(params["w"])
        return [float(w[list(cols).index(f"ctx{k}={tag}")]) for k in range(len(cols)) if f"ctx{k}={tag}" in cols]
    return None


def calibrated_contexts_table(
    model_name: str,
    params: dict[str, Any],
    data: ModelData,
    rows: np.ndarray | None = None,
    param_samples: list[dict[str, Any]] | None = None,
    b1_columns: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """``tag -> {"n_responses", "theta", "theta_ci95", "status"}`` for every entry of
    ``data.ctx_vocab``: ``n_responses`` counts the training sell rows (``rows``, default all)
    in which the tag appears; ``theta`` is :func:`context_parameter`; ``theta_ci95`` the
    2.5/97.5 percentiles over ``param_samples`` (None without draws); ``status`` is
    ``"calibrated"`` when ``n_responses > 0`` and ``"uncalibrated: prior only"`` otherwise."""
    sell = data.sell_mask()
    sel = sell if rows is None else (np.isin(np.arange(data.n), np.asarray(rows)) & sell)
    p = dict(params)
    if b1_columns is not None:
        p["_columns"] = tuple(b1_columns)
    out: dict[str, dict[str, Any]] = {}
    for c, tag in enumerate(data.ctx_vocab):
        present = sel & (data.ctx_idx == c).any(axis=1)
        n = int(present.sum())
        theta = context_parameter(model_name, p, data.ctx_vocab, tag)
        ci = None
        if param_samples:
            draws = []
            for s in param_samples:
                sp = dict(s)
                if b1_columns is not None:
                    sp["_columns"] = tuple(b1_columns)
                v = context_parameter(model_name, sp, data.ctx_vocab, tag)
                if v is not None:
                    draws.append(v)
            if draws:
                arr = np.asarray(draws)
                ci = [np.percentile(arr, 2.5, axis=0).tolist(), np.percentile(arr, 97.5, axis=0).tolist()]
        out[tag] = {"n_responses": n, "theta": theta, "theta_ci95": ci, "status": CALIBRATED if n > 0 else UNCALIBRATED}
    return out


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays (and tuples) into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _jsonable(np.asarray(obj))
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, Path):
        return str(obj)
    return obj


jsonable = _jsonable

DESIGN_VERSION = _design.DESIGN_VERSION

__all__ = [
    "ARTIFACT_JSON",
    "ARTIFACT_VERSION",
    "CALIBRATED",
    "DESIGN_VERSION",
    "ModelArtifact",
    "PARAMS_NPZ",
    "SAMPLES_NPZ",
    "UNCALIBRATED",
    "calibrated_contexts_table",
    "context_parameter",
    "flatten_params",
    "jsonable",
    "params_for_data",
    "predict_rows",
    "predict_rows_samples",
    "stack_samples",
    "to_numpy_tree",
    "unflatten_params",
    "unstack_samples",
]
