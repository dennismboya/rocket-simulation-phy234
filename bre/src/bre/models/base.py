"""The model contract every BRE model implements (PLAN.md section 4; INTERFACE.md next to this file).

Pieces:

* :class:`Model` — a ``typing.Protocol`` naming the methods a model exposes to the fitting,
  evaluation and API code (``init_params``, ``log_lik``, ``predict_proba``, ``log_prior``,
  ``n_params``, optionally ``subject_summary``).
* :class:`ModelBase` — an abstract base class with the shared implementations (Bernoulli/binomial
  per-row log-likelihood from ``predict_proba``, zero log-prior, parameter counting, the
  negative-log-posterior objective used by the optimizers).
* :class:`QuantumModelBase` — adds the interference-term logging every Q-model must provide.
* :class:`Encoder` — the linear covariate map with a per-subject random effect shared by the
  Q-models (``d_out = 4`` -> ``prepare_state``) and the classical latent-variable baselines
  (``d_out = 1``), the "equal-footing" architecture of PLAN.md section 4.
* :func:`standard_features` — the shared classical feature matrix.
* :func:`count_params`, :func:`bernoulli_log_lik` — helpers.

Likelihood convention: natural log, one value per response row, zero on rows outside
``ModelData.sell_rows()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre.models.data import ModelData

Params = Any
"""A JAX pytree (nested dicts/tuples of arrays) holding a model's free parameters."""

Family = Literal["classical", "quantum"]

PROB_EPS = 1e-12
"""Probabilities are clipped to ``[PROB_EPS, 1 - PROB_EPS]`` before taking logs."""

INTERFERENCE_KEYS: tuple[str, ...] = ("ltp", "order", "mix")
"""Keys of the dict returned by ``QuantumModelBase.interference_terms``: the LTP interference
``delta_LTP`` per row (always), the order effect ``Delta_order`` per row (NaN unless the row is an
ordered pair) and, when the model defines it, the context-uncertainty term ``delta_mix``."""


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def count_params(params: Params, exclude: tuple[str, ...] = ()) -> int:
    """Number of free real scalars in a pytree: the sum of ``size`` over its array leaves, with a
    complex leaf counting twice per element. ``exclude`` names top-level keys of a dict pytree to
    leave out (e.g. ``("u",)`` to report the count without the per-subject random effects)."""
    if isinstance(params, dict) and exclude:
        params = {k: v for k, v in params.items() if k not in exclude}
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        arr = np.asarray(leaf)
        total += int(arr.size) * (2 if np.iscomplexobj(arr) else 1)
    return total


def bernoulli_log_lik(p, y, w, mask) -> jnp.ndarray:
    """Per-row weighted Bernoulli/binomial log-likelihood::

        ll_i = mask_i * w_i * [ y_i log p_i + (1 - y_i) log(1 - p_i) ]

    ``p`` is clipped to ``[PROB_EPS, 1 - PROB_EPS]``. For a binary row (``y in {0, 1}``,
    ``w = 1``) this is the Bernoulli log-likelihood; for a ``choice_rate`` row with ``y`` the
    observed share and ``w`` the number of respondents it is the binomial log-likelihood up to
    the combinatorial constant. Rows with ``mask_i = False`` contribute exactly 0.
    """
    p = jnp.clip(jnp.asarray(p, dtype=jnp.float64), PROB_EPS, 1.0 - PROB_EPS)
    y = jnp.asarray(y, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    ll = w * (y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p))
    return jnp.where(mask, ll, 0.0)


# ---------------------------------------------------------------------------------------------
# Protocol and base classes
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """What every model exposes. See INTERFACE.md for the full contract."""

    name: str
    family: Family

    def init_params(self, rng_key, data: ModelData, restart: int) -> Params:
        """Initial parameters for restart ``restart`` (0, 1, ...) drawn from ``rng_key``."""

    def log_lik(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` per-row natural-log likelihood; 0 on rows outside ``data.sell_rows()``."""

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell)`` for every row (also computed, though unused, on non-sell rows)."""

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Scalar log-prior (0 for a model without shrinkage)."""

    def n_params(self, params: Params) -> int:
        """Number of free scalars in ``params``."""


class ModelBase(ABC):
    """Abstract base with the shared implementations; concrete models override
    :meth:`init_params` and :meth:`predict_proba` (and :meth:`log_prior` when they shrink).

    ``log_lik`` is derived from ``predict_proba`` through :func:`bernoulli_log_lik` so that every
    model uses exactly the same likelihood convention.
    """

    name: str = "base"
    family: Family = "classical"

    @abstractmethod
    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params: ...

    @abstractmethod
    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray: ...

    def log_lik(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` per-row log-likelihood (see :func:`bernoulli_log_lik`)."""
        p = self.predict_proba(params, data)
        return bernoulli_log_lik(p, data.y, data.w, data.sell_mask())

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Zero unless the model overrides it (hierarchical shrinkage, encoder random effects)."""
        return jnp.zeros((), dtype=jnp.float64)

    def n_params(self, params: Params) -> int:
        return count_params(params)

    def objective(self, params: Params, data: ModelData) -> jnp.ndarray:
        """Negative log-posterior ``-(sum_i log_lik_i + log_prior)``: what the optimizers minimize."""
        return -(jnp.sum(self.log_lik(params, data)) + self.log_prior(params))

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject (state angles, consistency score, ...) or None when the model has
        no per-subject quantities."""
        return None

    @staticmethod
    def restart_key(rng_key, restart: int):
        """The PRNG key of restart ``restart``: ``jax.random.fold_in(rng_key, restart)``."""
        return jax.random.fold_in(rng_key, int(restart))


class QuantumModelBase(ModelBase):
    """Base of the Q-models: same contract plus the interference terms of PLAN.md section 4."""

    family: Family = "quantum"

    @abstractmethod
    def interference_terms(self, params: Params, data: ModelData) -> dict[str, jnp.ndarray]:
        """Dict with at least ``"ltp"`` (``(n,)`` ``delta_LTP`` per row) and ``"order"`` (``(n,)``
        ``Delta_order`` per row, NaN where the row is not an ordered pair); ``"mix"`` optional.
        Keys are :data:`INTERFERENCE_KEYS`."""


# ---------------------------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------------------------

SIGMA_U_INV_GAMMA: tuple[float, float] = (2.0, 0.5)
"""``(alpha, beta)`` of the inverse-gamma hyperprior on each random-effect variance
``sigma_k^2`` in :func:`encoder_log_prior` (prior mean of ``sigma^2`` is ``beta / (alpha - 1)``
= 0.5). Without it the joint MAP over ``(u, sigma)`` is unbounded at ``u = 0, sigma -> 0``."""


def encoder_log_prior(params: dict[str, Any], w_l2: float = 0.0) -> jnp.ndarray:
    """Log-prior of the encoder's random effects with a fitted scale (an L2 penalty on ``u``)::

        log p(u, sigma) = sum_{i,k} log N(u_ik | 0, sigma_k^2)
                          + sum_k log InvGamma(sigma_k^2 | alpha, beta)      (SIGMA_U_INV_GAMMA)
                          - (w_l2 / 2) ||W||^2                               (fixed ridge, default 0)

    with ``sigma_k = exp(params["log_sigma_u"][k])``. The Gaussian term is
    ``-||u_.k||^2 / (2 sigma_k^2) - n_subjects log sigma_k - (n_subjects / 2) log 2 pi``; the
    inverse-gamma term is ``-(alpha + 1) log sigma_k^2 - beta / sigma_k^2`` plus its constant
    ``alpha log beta - log Gamma(alpha)``. Returns a scalar; all constants are included so the
    value is a proper log-density.
    """
    u = jnp.asarray(params["u"], dtype=jnp.float64)
    log_sigma = jnp.asarray(params["log_sigma_u"], dtype=jnp.float64)
    n_subjects = u.shape[0]
    sigma2 = jnp.exp(2.0 * log_sigma)
    gauss = (
        -0.5 * jnp.sum(u**2 / sigma2)
        - n_subjects * jnp.sum(log_sigma)
        - 0.5 * n_subjects * u.shape[1] * jnp.log(2.0 * jnp.pi)
    )
    alpha, beta = SIGMA_U_INV_GAMMA
    inv_gamma = jnp.sum(
        alpha * jnp.log(beta) - jax.scipy.special.gammaln(alpha) - (alpha + 1.0) * jnp.log(sigma2) - beta / sigma2
    )
    ridge = -0.5 * w_l2 * jnp.sum(jnp.asarray(params["W"], dtype=jnp.float64) ** 2)
    return gauss + inv_gamma + ridge


@dataclass(frozen=True)
class Encoder:
    """Linear covariate map with a per-subject random effect (PLAN.md section 4, shared by both
    model families)::

        z_i = W^T x_i + b + u_{s(i)}          z_i in R^{d_out}

    ``x_i`` is the standardized covariate row of the subject ``s(i)`` of response ``i``
    (``ModelData.X[subject_idx]``), ``W`` is ``(d_x, d_out)``, ``b`` is ``(d_out,)`` and ``u`` is
    ``(n_subjects, d_out)`` with ``u_s ~ N(0, diag(sigma^2))`` and fitted ``log_sigma_u``
    (``(d_out,)``; see :func:`encoder_log_prior`). Q-models use ``d_out = 4`` and feed ``z`` to
    ``bre.models.quantum.core.prepare_state``; B2-style latents use ``d_out = 1``.

    Parameters live in a dict with keys ``W``, ``b``, ``u``, ``log_sigma_u``; a model that
    embeds the encoder stores this dict under its own key (conventionally ``"enc"``).
    """

    d_x: int
    d_out: int
    n_subjects: int
    init_scale: float = 0.1
    log_sigma_init: float = float(np.log(0.5))

    def init(self, rng_key, restart: int = 0) -> dict[str, jnp.ndarray]:
        """Parameters of restart ``restart``: ``W ~ N(0, init_scale^2 / d_x)``,
        ``b ~ N(0, init_scale^2)``, ``u = 0``, ``log_sigma_u = log_sigma_init``; the key is
        ``jax.random.fold_in(rng_key, restart)`` so restarts differ and are reproducible."""
        key = jax.random.fold_in(rng_key, int(restart))
        k_w, k_b = jax.random.split(key)
        scale_w = self.init_scale / np.sqrt(max(self.d_x, 1))
        return {
            "W": scale_w * jax.random.normal(k_w, (self.d_x, self.d_out), dtype=jnp.float64),
            "b": self.init_scale * jax.random.normal(k_b, (self.d_out,), dtype=jnp.float64),
            "u": jnp.zeros((self.n_subjects, self.d_out), dtype=jnp.float64),
            "log_sigma_u": jnp.full((self.d_out,), self.log_sigma_init, dtype=jnp.float64),
        }

    def apply_subjects(self, params: dict[str, Any], X) -> jnp.ndarray:
        """``(n_subjects, d_out)``: ``X W + b + u`` for every subject."""
        X = jnp.asarray(X, dtype=jnp.float64)
        return X @ params["W"] + params["b"] + params["u"]

    def apply(self, params: dict[str, Any], X, subject_idx) -> jnp.ndarray:
        """``(n, d_out)``: the subject-level output expanded to response rows."""
        return self.apply_subjects(params, X)[jnp.asarray(subject_idx)]

    def log_prior(self, params: dict[str, Any], w_l2: float = 0.0) -> jnp.ndarray:
        """:func:`encoder_log_prior` of ``params``."""
        return encoder_log_prior(params, w_l2=w_l2)

    def n_params(self, params: dict[str, Any], include_random_effects: bool = True) -> int:
        """Free scalars: ``W``, ``b``, ``log_sigma_u`` and, unless excluded, ``u``."""
        return count_params(params, exclude=() if include_random_effects else ("u",))

    @classmethod
    def for_data(cls, data: ModelData, d_out: int, **kw) -> "Encoder":
        """Encoder sized for ``data`` (``d_x = data.d_x``, ``n_subjects = data.n_subjects``)."""
        return cls(d_x=data.d_x, d_out=d_out, n_subjects=data.n_subjects, **kw)


# ---------------------------------------------------------------------------------------------
# Shared classical feature matrix
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMatrix:
    """``F`` (``(n, d_f)`` float64) with its column names."""

    F: np.ndarray
    columns: tuple[str, ...]

    @property
    def d_f(self) -> int:
        return int(self.F.shape[1])

    def column(self, name: str) -> np.ndarray:
        return self.F[:, self.columns.index(name)]


def standard_features(data: ModelData, loss_column: Literal["loss", "loss_signed"] = "loss") -> FeatureMatrix:
    """The shared classical feature matrix (PLAN.md section 4 ``features(row)``), one row per
    response, columns in this order:

    1. ``data.X_columns``: the subject's standardized covariates, ``X[subject_idx]``;
    2. ``loss`` (or ``loss_signed`` when ``loss_column`` says so);
    3. per-position context one-hots ``ctx{k}={tag}`` for ``k`` in ``0..K-1`` and every tag of
       ``ctx_vocab`` (an ordered pair lights one column at position 0 and one at position 1; the
       ``none`` condition lights none);
    4. ``order_flag``;
    5. ``tol_answered`` (1 when ``tol_answer`` is observed) and ``tol_yes`` (1 when it is 1).

    This is exactly the information the Q-models receive (state from ``X``, ``L``, the context
    sequence, the question order and the prior tolerance answer), so a classical model built on
    it stands on equal footing. Interactions are the model's business (B1 adds pairwise ones).
    """
    n, K, V = data.n, data.n_ctx_positions, data.n_contexts
    blocks: list[np.ndarray] = []
    cols: list[str] = []
    blocks.append(data.X[data.subject_idx])
    cols.extend(data.X_columns)
    blocks.append(getattr(data, loss_column)[:, None])
    cols.append(loss_column)
    ctx = np.zeros((n, K * V), dtype=np.float64)
    for k in range(K):
        valid = data.ctx_mask[:, k]
        ctx[np.flatnonzero(valid), k * V + data.ctx_idx[valid, k]] = 1.0
        cols.extend(f"ctx{k}={tag}" for tag in data.ctx_vocab)
    blocks.append(ctx)
    blocks.append(data.order_flag[:, None].astype(np.float64))
    cols.append("order_flag")
    answered = np.isfinite(data.tol_answer)
    blocks.append(answered[:, None].astype(np.float64))
    cols.append("tol_answered")
    blocks.append((answered & (data.tol_answer == 1.0))[:, None].astype(np.float64))
    cols.append("tol_yes")
    F = np.concatenate(blocks, axis=1).astype(np.float64)
    return FeatureMatrix(F=F, columns=tuple(cols))


__all__ = [
    "Encoder",
    "Family",
    "FeatureMatrix",
    "INTERFERENCE_KEYS",
    "Model",
    "ModelBase",
    "PROB_EPS",
    "Params",
    "QuantumModelBase",
    "SIGMA_U_INV_GAMMA",
    "bernoulli_log_lik",
    "count_params",
    "encoder_log_prior",
    "standard_features",
]
