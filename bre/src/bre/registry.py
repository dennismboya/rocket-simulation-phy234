"""Model and generator registry of the fit / eval / recover drivers (PLAN.md sections 4 and 5).

``MODEL_REGISTRY`` maps the PLAN id of every fitted model to its class; :func:`make_model` builds
one on a ``ModelData`` (every model is sized by the data it is built on: context vocabulary,
covariate columns, number of subjects). ``GENERATORS`` lists the synthetic generators of the
recovery study by id (``bre.sim.<id>``) with the family each one belongs to, which is what the
confusion matrix of ``bre.recover`` counts as a correct selection.

``PER_SUBJECT_BLOCKS`` names, per model, the parameter blocks that hold one row per training
subject and how to fill them for a subject the model has never seen (``bre.artifact`` uses it to
score new clients with the population-level parameters only): the encoder random effect ``u``
is set to zero, B2's subject context slopes to their population mean ``mu_ctx``, Q4's
``log_gamma`` to the population location ``gamma_mu``.

Parameter-count notes (INTERFACE.md section 2): reports show two counts for models with a
per-subject block, all free scalars and the count without the per-subject blocks; both come
from :func:`param_counts`, which asks the model (``model.n_params(params)`` and
``model.n_params(params, include_random_effects=False)``) when the instance is given and falls
back to counting the pytree leaves (``count_params``) otherwise. The model's own count is the
documented one (``PARAM_COUNT_NOTES``): it leaves out fixed entries (the ``none`` rows of a
Q-model's context table) and, for B5, counts tree leaves and MLP weights instead of the bytes
of the pickled estimators.
"""

from __future__ import annotations

from typing import Any

from bre.models.base import Model, count_params
from bre.models.classical.b1_logistic import B1
from bre.models.classical.b2_hier import B2
from bre.models.classical.b3_cpt import B3
from bre.models.classical.b4_bayes_updater import B4
from bre.models.classical.b5_ml import B5
from bre.models.classical.b6_hmm import B6
from bre.models.data import ModelData
from bre.models.quantum.q2_context_unitary import Q2
from bre.models.quantum.q3_dynamics import Q3
from bre.models.quantum.q4_open_system import Q4
from bre.models.quantum.q5_qdt import Q5

MODEL_REGISTRY: dict[str, type] = {"B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5, "B6": B6, "Q2": Q2, "Q3": Q3, "Q4": Q4, "Q5": Q5}
"""PLAN id -> model class (B1-B6, Q2-Q5 of PLAN.md section 4; Q1 is aggregate-only). B5 carries
``requires_external_fit = True`` (fitted through ``B5.fit_external``, not by gradient steps)."""

MODEL_FAMILY: dict[str, str] = {name: cls.family for name, cls in MODEL_REGISTRY.items()}
"""PLAN id -> ``"classical"`` or ``"quantum"``."""

GENERATORS: dict[str, str] = {"gq": "quantum", "gc": "classical", "gf": "classical"}
"""Generator id (``bre.sim.<id>``) -> the model family that generated the data (G_Q is a
quantum-probability process; G_C and G_F are Kolmogorov processes)."""

GENERATOR_MATCHED_MODELS: dict[str, tuple[str, ...]] = {
    "gq": ("Q4", "Q2"),
    "gc": ("B2", "B1"),
    "gf": ("B1",),
}
"""Generator id -> the models that can represent it exactly (or up to the per-subject spread it
summarizes by a population table): G_Q is Q4's own process (Q2 when every ``gamma_i = 0``); G_C
is B2's additive random-effects logistic (B1 with position-specific main effects also spans it);
G_F is a logistic with ordered-pair interactions that only B1 spans."""

PER_SUBJECT_BLOCKS: dict[str, dict[tuple[str, ...], str]] = {
    "B1": {},
    "B2": {("enc", "u"): "zero", ("beta_ctx",): "mu_ctx"},
    "B3": {("enc", "u"): "zero"},
    "B4": {("enc", "u"): "zero"},
    "B5": {},
    "B6": {},
    "Q2": {("enc", "u"): "zero"},
    "Q3": {("enc", "u"): "zero"},
    "Q4": {("enc", "u"): "zero", ("log_gamma",): "gamma_mu"},
    "Q5": {("enc", "u"): "zero"},
}
"""Model id -> ``{path_of_block: fill_rule}``; ``path_of_block`` is the nested-key path inside the
parameter pytree, the leading axis of the block is the subject axis. Fill rules for an unseen
subject: ``"zero"`` (the prior mean of the random effect), ``"mu_ctx"`` (the population context
means, a ``(V,)`` vector broadcast to the subject), ``"gamma_mu"`` (the population location of
``log_gamma``, a scalar)."""

PARAM_COUNT_NOTES: dict[str, str] = {
    "B1": "d_f + d_int + 1: every main-effect and interaction weight plus the intercept; no per-subject block.",
    "B2": "d_x + n_subjects (V + 1) + 2 V + 6 (posterior-mean pytree); without u and beta_ctx: d_x + 2 V + 6. The Bayesian effective count is p_waic / p_loo.",
    "B4": "(d_x + 1 + n_subjects + 1) + V + 5; without u: d_x + 2 + V + 5.",
    "Q2": "(4 d_x + 4 + 4 n_subjects + 4) + 3 + 3 V_free + 1; without u: 4 d_x + 8 + 3 + 3 V_free + 1. Two of the four encoder outputs per subject are identifiable (gauge).",
    "Q4": "Q2's count + n_subjects (log_gamma) + 2; without random effects: Q2's population count + 2.",
    "B3": "(3 d_x + 3 + 3 n_subjects + 3) + 2 V + 6 (+1 with separate_beta): encoder to (log lambda, log alpha, log gamma), reference-point and recovery vectors, kappa, tau; without u: 3 d_x + 6 + 2 V + 6.",
    "B5": "leaves of the boosting trees + weights and biases of the MLP (model.n_params; count_params on the pytree would count pickle bytes); no per-subject block.",
    "B6": "d_x + 2 d_row + 5: shared covariate weights, two state-specific row-feature weight vectors, two intercepts (a0, log_da), two transition logits, one initial logit; no per-subject block.",
    "Q3": "(4 d_x + 4 + 4 n_subjects + 4) + 3 + 2 + V_free + 1: encoder, theta_p, the two angles of theta_d, one dissonance weight per free context, phi; without u: 4 d_x + 8 + 6 + V_free.",
    "Q5": "(d_x + 1 + n_subjects + 1) + V + 6: encoder (attraction baseline), b_ctx, b_order, b_tol, pr_logit, log_rho, log_kappa, log_tau; without u: d_x + 2 + V + 6.",
}
"""One line per model on what ``n_params`` counts (INTERFACE.md section 2)."""


def make_model(name: str, data: ModelData, **kwargs: Any) -> Model:
    """Instantiate the model with PLAN id ``name`` on ``data`` (``kwargs`` go to the constructor,
    e.g. ``l2`` for B1 or ``forward`` for Q2)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](data, **kwargs)


def per_subject_paths(name: str) -> tuple[tuple[str, ...], ...]:
    """The nested-key paths of the per-subject blocks of model ``name``."""
    return tuple(PER_SUBJECT_BLOCKS[name])


def _count_without_per_subject_blocks(name: str, params: Any) -> int:
    """``count_params`` of ``params`` with the per-subject blocks of :data:`PER_SUBJECT_BLOCKS`
    removed (the pytree fallback of :func:`param_counts`)."""
    without = dict(params)
    for path in per_subject_paths(name):
        if len(path) == 1:
            without = {k: v for k, v in without.items() if k != path[0]}
        else:
            inner = dict(without[path[0]])
            inner = {k: v for k, v in inner.items() if k != path[1]}
            without[path[0]] = inner
    return int(count_params(without))


def param_counts(name: str, params: Any, model: Model | None = None) -> dict[str, int]:
    """``{"all": ..., "population": ...}``: every free scalar, and the count without the
    per-subject blocks of :data:`PER_SUBJECT_BLOCKS` (equal when the model has none).

    With ``model`` (the instance the parameters belong to) both numbers come from
    ``model.n_params`` — ``all = model.n_params(params)``, ``population =
    model.n_params(params, include_random_effects=False)`` for a model with per-subject blocks
    (every such model takes that keyword; a model without blocks reports ``all`` twice). This is
    the count reports must use: for B5 the pytree leaves are pickled estimators and
    ``count_params`` would count their bytes. Without ``model`` the pytree leaves are counted
    (``count_params``), which agrees with the model's count for every gradient-fitted model on a
    vocabulary without ``none`` entries.
    """
    if model is not None:
        total = int(model.n_params(params))
        if not PER_SUBJECT_BLOCKS.get(name):
            return {"all": total, "population": total}
        try:
            population = int(model.n_params(params, include_random_effects=False))
        except TypeError:  # a model with per-subject blocks but no such keyword: pytree fallback
            population = _count_without_per_subject_blocks(name, params)
        return {"all": total, "population": population}
    return {"all": int(count_params(params)), "population": _count_without_per_subject_blocks(name, params)}


__all__ = [
    "GENERATORS",
    "GENERATOR_MATCHED_MODELS",
    "MODEL_FAMILY",
    "MODEL_REGISTRY",
    "PARAM_COUNT_NOTES",
    "PER_SUBJECT_BLOCKS",
    "make_model",
    "param_counts",
    "per_subject_paths",
]
