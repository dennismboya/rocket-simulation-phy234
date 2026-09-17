"""B1 — logistic regression with pairwise interactions (PLAN.md section 4, classical baseline).

Model
-----
For response row ``i`` with the shared main-effects matrix ``F_i`` of
:func:`bre.models.base.standard_features` (covariates, loss, per-position context one-hots,
``order_flag``, ``tol_answered``, ``tol_yes``) and the interaction block ``G_i`` built here::

    logit P(sell_i) = b + w_main . F_i + w_int . G_i

Interaction block (built explicitly, names in :attr:`B1.interaction_columns`), pairwise among
the loss, the per-position context dummies and the order flag:

* ``loss*order_flag``                                       (1 column)
* ``loss*ctx{k}={tag}``      for every position k and tag  (K V columns)
* ``order_flag*ctx{k}={tag}`` for every position k and tag  (K V columns)
* ``ctx{k}={a}*ctx{k'}={b}`` for positions k < k' and tags a != b (K(K-1)/2 V(V-1) columns):
  the ordered-pair terms that let B1 fit any pair effect, including order effects
  (``(a, b)`` and ``(b, a)`` are different columns).

Same-position products (``ctx0=a * ctx0=b``) are identically zero (one tag per position) and
same-tag cross-position products are zero on the shared design (pairs are of distinct contexts),
so neither is built. With the design's ``K = 2``, ``V = 4`` the block has
``1 + 8 + 8 + 12 = 29`` columns.

Parameters: ``{"w": (d_f + d_int,), "b": ()}``; ``n_params = d_f + d_int + 1`` (every weight
plus the intercept). The column order of ``w`` is ``B1.columns`` (main effects first, then
:attr:`B1.interaction_columns`).

Prior: a ridge on the weights (not on the intercept), ``log_prior = -(l2 / 2) ||w||^2`` with
``l2 >= 0`` fixed at construction (0 disables it); this is B1's only shrinkage.

Initialization: ``w, b ~ N(0, init_scale^2)`` with ``init_scale = 0.01`` from
``fold_in(rng_key, restart)``, so restarts differ and are reproducible.

The feature matrix is built with numpy from ``ModelData`` (no frame access) and cached per
``ModelData`` object, so an optimizer loop calling ``objective`` repeatedly does not rebuild it.
The column set is fixed at construction from the data's ``ctx_vocab`` and ``K`` and is
re-derived by name on any other ``ModelData`` with the same vocabulary (train/test subsets share
it by construction of ``ModelData.subset``).
"""

from __future__ import annotations

import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from bre.models.base import FeatureMatrix, ModelBase, Params, standard_features
from bre.models.data import ModelData

INIT_SCALE_DEFAULT = 0.01
"""Standard deviation of the small-normal initialization of ``w`` and ``b``."""

L2_DEFAULT = 1e-2
"""Default ridge strength on the weights (``log_prior = -(l2/2)||w||^2``)."""


def interaction_column_names(
    ctx_vocab: tuple[str, ...], n_ctx_positions: int
) -> tuple[tuple[str, str], ...]:
    """The interaction block as ``(left, right)`` column-name pairs, in the fixed order of the
    module docstring: ``loss*order_flag``, ``loss*ctx``, ``order_flag*ctx``, cross-position
    ordered pairs of distinct tags."""
    K, V = int(n_ctx_positions), tuple(ctx_vocab)
    ctx_cols = [(k, f"ctx{k}={tag}") for k in range(K) for tag in V]
    pairs: list[tuple[str, str]] = [("loss", "order_flag")]
    pairs += [("loss", c) for _, c in ctx_cols]
    pairs += [("order_flag", c) for _, c in ctx_cols]
    for k in range(K):
        for k2 in range(k + 1, K):
            for a in V:
                for b in V:
                    if a != b:
                        pairs.append((f"ctx{k}={a}", f"ctx{k2}={b}"))
    return tuple(pairs)


class B1(ModelBase):
    """Logistic regression on the shared features plus explicit pairwise interactions."""

    name = "B1"
    family = "classical"

    def __init__(
        self,
        data: ModelData,
        l2: float = L2_DEFAULT,
        init_scale: float = INIT_SCALE_DEFAULT,
        loss_column: str = "loss",
    ) -> None:
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.l2 = float(l2)
        self.init_scale = float(init_scale)
        self.loss_column = loss_column
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_ctx_positions = int(data.n_ctx_positions)
        base = standard_features(data, loss_column=loss_column)  # type: ignore[arg-type]
        self.main_columns: tuple[str, ...] = tuple(base.columns)
        pairs = interaction_column_names(self.ctx_vocab, self.n_ctx_positions)
        # the loss column is named after ``loss_column`` in the main block
        pairs = tuple((left.replace("loss", loss_column) if left == "loss" else left, right) for left, right in pairs)
        self._pairs: tuple[tuple[str, str], ...] = pairs
        self.interaction_columns: tuple[str, ...] = tuple(f"{a}*{b}" for a, b in pairs)
        self.columns: tuple[str, ...] = self.main_columns + self.interaction_columns
        self._cache: weakref.WeakKeyDictionary[ModelData, np.ndarray] = weakref.WeakKeyDictionary()

    # -- sizes ------------------------------------------------------------------------------------

    @property
    def d_main(self) -> int:
        return len(self.main_columns)

    @property
    def d_interactions(self) -> int:
        return len(self.interaction_columns)

    @property
    def d(self) -> int:
        """Number of weights (main effects plus interactions); ``n_params = d + 1``."""
        return len(self.columns)

    # -- features ---------------------------------------------------------------------------------

    def features(self, data: ModelData) -> FeatureMatrix:
        """``(n, d)`` design matrix for ``data``: the shared main effects followed by the
        interaction block, columns in :attr:`columns`. Raises if ``data`` has another context
        vocabulary or number of positions than the construction data."""
        if tuple(data.ctx_vocab) != self.ctx_vocab or data.n_ctx_positions != self.n_ctx_positions:
            raise ValueError("B1: data has a different ctx_vocab / n_ctx_positions than the model was built on")
        base = standard_features(data, loss_column=self.loss_column)  # type: ignore[arg-type]
        if tuple(base.columns) != self.main_columns:
            raise ValueError("B1: standard_features columns differ from the construction data")
        col = {name: j for j, name in enumerate(base.columns)}
        G = np.empty((data.n, len(self._pairs)), dtype=np.float64)
        for j, (a, b) in enumerate(self._pairs):
            G[:, j] = base.F[:, col[a]] * base.F[:, col[b]]
        F = np.concatenate([base.F, G], axis=1)
        return FeatureMatrix(F=F, columns=self.columns)

    def _F(self, data: ModelData) -> np.ndarray:
        F = self._cache.get(data)
        if F is None:
            F = self.features(data).F
            self._cache[data] = F
        return F

    # -- model contract ---------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        key = self.restart_key(rng_key, restart)
        k_w, k_b = jax.random.split(key)
        return {
            "w": self.init_scale * jax.random.normal(k_w, (self.d,), dtype=jnp.float64),
            "b": self.init_scale * jax.random.normal(k_b, (), dtype=jnp.float64),
        }

    def logits(self, params: Params, data: ModelData) -> jnp.ndarray:
        F = jnp.asarray(self._F(data))
        return F @ params["w"] + params["b"]

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        return jax.nn.sigmoid(self.logits(params, data))

    def log_prior(self, params: Params) -> jnp.ndarray:
        return -0.5 * self.l2 * jnp.sum(jnp.asarray(params["w"], dtype=jnp.float64) ** 2)

    def n_params(self, params: Params) -> int:
        """``d + 1``: every weight (main effects and interactions) plus the intercept."""
        return int(np.asarray(params["w"]).size) + 1

    def coefficients(self, params: Params) -> pd.Series:
        """Weights by column name (plus ``intercept``)."""
        w = np.asarray(params["w"], dtype=np.float64)
        s = pd.Series(w, index=list(self.columns), dtype="float64")
        s.loc["intercept"] = float(np.asarray(params["b"]))
        return s

    def context_coefficients(self, params: Params) -> pd.Series:
        """Main-effect weights of the per-position context dummies (``ctx{k}={tag}`` columns)."""
        coef = self.coefficients(params)
        return coef[[c for c in self.main_columns if c.startswith("ctx")]]

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """None: B1 has no per-subject quantities (covariates enter as fixed effects)."""
        return None


__all__ = ["B1", "INIT_SCALE_DEFAULT", "L2_DEFAULT", "interaction_column_names"]
