"""Model layer of the Behavioral Risk Engine (PLAN.md section 4).

Importing this package enables 64-bit floats in JAX **before** any model module builds an array:
the unitarity tests (``U^dagger U = I`` to 1e-10) and the dephasing limits need double precision.
Every model module therefore imports ``bre.models`` (implicitly, as its parent package) rather than
calling ``jax.config.update`` itself.

Layout::

    bre.models.data       ModelData: numpy view of a schema frame that every model consumes
    bre.models.base       Model protocol, ModelBase, Encoder, standard_features, count_params
    bre.models.quantum    Hilbert-space primitives (core) and the Q-models Q1-Q5
    bre.models.classical  the classical baselines B1-B6

The human-readable contract is ``src/bre/models/INTERFACE.md``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from bre.models.base import (  # noqa: E402  (must come after the x64 switch)
    Encoder,
    FeatureMatrix,
    Model,
    ModelBase,
    QuantumModelBase,
    bernoulli_log_lik,
    count_params,
    encoder_log_prior,
    standard_features,
)
from bre.models.data import ModelData, build_model_data  # noqa: E402

__all__ = [
    "Encoder",
    "FeatureMatrix",
    "Model",
    "ModelBase",
    "ModelData",
    "QuantumModelBase",
    "bernoulli_log_lik",
    "build_model_data",
    "count_params",
    "encoder_log_prior",
    "standard_features",
]
