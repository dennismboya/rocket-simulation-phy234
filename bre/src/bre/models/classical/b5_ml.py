"""B5 — gradient boosting and a small MLP (scikit-learn) on the shared features (PLAN.md
section 4, classical baseline).

Model
-----
Two off-the-shelf classifiers on exactly the B1 design matrix (the shared main effects of
:func:`bre.models.base.standard_features` followed by B1's pairwise interaction block,
:meth:`bre.models.classical.b1_logistic.B1.features`), so B5 sees the same information as every
other model and nothing more:

* ``gbt``: ``sklearn.ensemble.HistGradientBoostingClassifier`` (default: 100 iterations,
  learning rate 0.1, max 15 leaves, ``l2_regularization = 1``, no early stopping);
* ``mlp``: ``sklearn.neural_network.MLPClassifier`` with one hidden layer of 16 ReLU units
  (``alpha = 1e-3`` ridge, Adam, 300 iterations);
* ``estimator="ensemble"`` (default): ``P(sell) = (P_gbt + P_mlp) / 2``; ``"gbt"`` or ``"mlp"``
  serve one of the two alone (``model_kwargs`` of the fit config).

Predictions are clipped to ``[PROB_EPS, 1 - PROB_EPS]``. The likelihood is the shared one of
:class:`bre.models.base.ModelBase`; ``log_prior = 0``.

Fitting (no gradients)
----------------------
B5 is not fitted by the optimizer loop: ``jax.grad`` of its objective is meaningless (the
parameters are pickled estimators). The class carries ``requires_external_fit = True`` and
:meth:`fit_external(data, seed)` fits both estimators on the sell rows of ``data`` (binary
``y``; ``w`` as ``sample_weight``) and returns the parameter pytree. :meth:`init_params` calls
the same routine with a seed derived from ``(rng_key, restart)``, so the protocol's
``init_params`` already yields a *fitted* model — a fit driver that then ran Adam would be
wasting its time and must dispatch on ``requires_external_fit`` instead (``params =
model.fit_external(train, seed)``; no restarts beyond the seed, no L-BFGS polish, no Laplace
draws).

Parameters and the artifact
---------------------------
``{"blob": {"gbt": uint8 (n_bytes,), "mlp": uint8 (n_bytes,)}, "meta": {"seed": (), "n_leaves":
(), "n_weights": ()}}``. The two ``blob`` leaves are the pickled estimators as byte arrays:
``bre.artifact.flatten_params`` casts every leaf to float64 and ``np.savez`` stores it, and byte
values 0..255 are exact in float64, so the artifact round-trips without any special case
(:func:`unpickle_blob` casts back to ``uint8`` and unpickles; unpickled estimators are cached
per byte string). No per-subject block exists (``PER_SUBJECT_BLOCKS["B5"] = {}``): a new client
is scored from its covariates through the same features.

``n_params`` = the number of leaves over all trees of the boosting model plus the number of
weights and biases of the MLP (the active estimators only). ``count_params`` on the pytree
would count bytes, so reports must use ``model.n_params``.

Threads: the boosting model's OpenMP loops are run under ``threadpoolctl.threadpool_limits``
with ``n_threads`` (default 1) for fitting and prediction; on a shared 4-core box the
single-threaded fit is an order of magnitude faster than the oversubscribed default, and the
result does not depend on the thread count.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from typing import Any

from threadpoolctl import threadpool_limits

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from bre.models.base import PROB_EPS, ModelBase, Params
from bre.models.classical.b1_logistic import B1
from bre.models.data import ModelData

ESTIMATORS: tuple[str, ...] = ("ensemble", "gbt", "mlp")

GBT_DEFAULTS: dict[str, Any] = {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 1.0, "early_stopping": False}
MLP_DEFAULTS: dict[str, Any] = {"hidden_layer_sizes": (16,), "alpha": 1e-3, "max_iter": 300, "solver": "adam"}

_UNPICKLE_CACHE: dict[str, Any] = {}


def pickle_blob(obj: Any) -> np.ndarray:
    """``pickle.dumps(obj)`` as a ``uint8`` array (an artifact-safe leaf, module docstring)."""
    return np.frombuffer(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8).copy()


def unpickle_blob(arr) -> Any:
    """Inverse of :func:`pickle_blob`; accepts the float64 form the artifact reloads. Cached on
    the byte string's hash so repeated ``predict_proba`` calls do not unpickle again."""
    raw = np.asarray(arr)
    data = raw.tobytes() if raw.dtype == np.uint8 else np.rint(raw).astype(np.uint8).tobytes()
    key = hashlib.blake2b(data, digest_size=16).hexdigest()
    obj = _UNPICKLE_CACHE.get(key)
    if obj is None:
        obj = pickle.loads(data)
        _UNPICKLE_CACHE[key] = obj
    return obj


def gbt_n_leaves(est: HistGradientBoostingClassifier) -> int:
    """Total number of leaf nodes over every tree of a fitted boosting model."""
    return int(sum(p.get_n_leaf_nodes() for stage in est._predictors for p in stage))


def mlp_n_weights(est: MLPClassifier) -> int:
    """Number of weights plus biases of a fitted MLP."""
    return int(sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_))


class B5(ModelBase):
    """Gradient boosting + small MLP on the B1 feature matrix (module docstring)."""

    name = "B5"
    family = "classical"
    requires_external_fit = True
    """The fit driver must call :meth:`fit_external` instead of running gradient steps."""

    def __init__(
        self,
        data: ModelData,
        *,
        estimator: str = "ensemble",
        gbt_kwargs: dict[str, Any] | None = None,
        mlp_kwargs: dict[str, Any] | None = None,
        loss_column: str = "loss",
        n_threads: int | None = 1,
    ) -> None:
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}; got {estimator!r}")
        self.estimator = estimator
        self.n_threads = None if n_threads is None else int(n_threads)
        self.gbt_kwargs = {**GBT_DEFAULTS, **(gbt_kwargs or {})}
        self.mlp_kwargs = {**MLP_DEFAULTS, **(mlp_kwargs or {})}
        self._b1 = B1(data, l2=0.0, loss_column=loss_column)
        self.columns: tuple[str, ...] = self._b1.columns

    # -- features ---------------------------------------------------------------------------------

    def features(self, data: ModelData) -> np.ndarray:
        """``(n, d)`` B1 design matrix (main effects + interaction block), columns :attr:`columns`."""
        return self._b1._F(data)

    @property
    def d(self) -> int:
        return len(self.columns)

    # -- fitting ----------------------------------------------------------------------------------

    def fit_external(self, data: ModelData, seed: int = 0) -> Params:
        """Fit both estimators on ``data.sell_rows()`` (``y`` must be binary; ``w`` is the
        ``sample_weight``) with ``random_state = seed`` and return the parameter pytree."""
        rows = data.sell_rows()
        if rows.size == 0:
            raise ValueError("B5.fit_external: no sell rows to fit on")
        y = np.asarray(data.y, dtype=np.float64)[rows]
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("B5 needs binary responses on the sell rows (choice-rate rows are not supported)")
        X = self.features(data)[rows]
        w = np.asarray(data.w, dtype=np.float64)[rows]
        yi = y.astype(np.int64)
        seed = int(seed)
        gbt = HistGradientBoostingClassifier(random_state=seed, **self.gbt_kwargs)
        mlp = MLPClassifier(random_state=seed, **self.mlp_kwargs)
        with warnings.catch_warnings(), threadpool_limits(limits=self.n_threads):
            warnings.simplefilter("ignore", ConvergenceWarning)
            gbt.fit(X, yi, sample_weight=w)
            mlp.fit(X, yi, sample_weight=w)
        return {
            "blob": {"gbt": pickle_blob(gbt), "mlp": pickle_blob(mlp)},
            "meta": {
                "seed": np.asarray(seed, dtype=np.float64),
                "n_leaves": np.asarray(gbt_n_leaves(gbt), dtype=np.float64),
                "n_weights": np.asarray(mlp_n_weights(mlp), dtype=np.float64),
            },
        }

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        """A *fitted* parameter pytree: :meth:`fit_external` with the seed
        ``randint(fold_in(rng_key, restart))`` (deterministic in ``(rng_key, restart)``)."""
        seed = int(jax.random.randint(self.restart_key(rng_key, restart), (), 0, 2**31 - 1))
        return self.fit_external(data, seed)

    # -- prediction -------------------------------------------------------------------------------

    def estimators(self, params: Params) -> dict[str, Any]:
        """The unpickled estimators that :attr:`estimator` uses (``gbt`` and/or ``mlp``)."""
        names = ("gbt", "mlp") if self.estimator == "ensemble" else (self.estimator,)
        return {n: unpickle_blob(params["blob"][n]) for n in names}

    def predict_proba_parts(self, params: Params, data: ModelData) -> dict[str, np.ndarray]:
        """``P(sell)`` per active estimator (numpy, ``(n,)`` each), before averaging."""
        X = self.features(data)
        with threadpool_limits(limits=self.n_threads):
            return {n: np.asarray(est.predict_proba(X)[:, 1], dtype=np.float64) for n, est in self.estimators(params).items()}

    def predict_proba(self, params: Params, data: ModelData) -> jnp.ndarray:
        """``(n,)`` ``P(sell)``: the estimator's probability, or the ensemble mean; clipped to
        ``[PROB_EPS, 1 - PROB_EPS]``. Not differentiable (module docstring)."""
        parts = self.predict_proba_parts(params, data)
        p = np.mean(np.stack(list(parts.values())), axis=0)
        return jnp.asarray(np.clip(p, PROB_EPS, 1.0 - PROB_EPS), dtype=jnp.float64)

    def n_params(self, params: Params) -> int:
        """Leaves of the boosting trees plus weights and biases of the MLP (active estimators)."""
        ests = self.estimators(params)
        total = 0
        if "gbt" in ests:
            total += gbt_n_leaves(ests["gbt"])
        if "mlp" in ests:
            total += mlp_n_weights(ests["mlp"])
        return int(total)

    # -- reporting --------------------------------------------------------------------------------

    def feature_importance(self, params: Params, data: ModelData, n_repeats: int = 5, seed: int = 0) -> pd.Series | None:
        """Permutation importance of the boosting model on the sell rows of ``data`` (mean
        decrease of the log-loss over ``n_repeats`` shuffles), by column name; None when the
        boosting model is not active."""
        from sklearn.inspection import permutation_importance

        ests = self.estimators(params)
        if "gbt" not in ests:
            return None
        rows = data.sell_rows()
        X = self.features(data)[rows]
        y = np.asarray(data.y)[rows].astype(np.int64)
        with threadpool_limits(limits=self.n_threads):
            r = permutation_importance(ests["gbt"], X, y, scoring="neg_log_loss", n_repeats=int(n_repeats), random_state=int(seed))
        return pd.Series(r.importances_mean, index=list(self.columns), dtype="float64")

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """None: B5 has no per-subject quantities."""
        return None


__all__ = ["B5", "ESTIMATORS", "GBT_DEFAULTS", "MLP_DEFAULTS", "gbt_n_leaves", "mlp_n_weights", "pickle_blob", "unpickle_blob"]
