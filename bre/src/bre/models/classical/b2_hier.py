"""B2 — hierarchical Bayesian logistic regression in NumPyro (PLAN.md section 4, classical baseline).

Model
-----
For response row ``i`` of subject ``s = s(i)`` with loss ``L_i``, contexts ``c_1..c_m`` (m <= K),
order flag and the prior-tolerance indicators of :func:`bre.models.base.standard_features`::

    logit P(sell_i) = alpha_s + beta_L L_i + sum_k beta_ctx[s, c_k]
                      + beta_order order_flag_i + beta_tol_answered tol_answered_i + beta_tol_yes tol_yes_i

* ``alpha_s = W^T x_s + b + u_s`` is the subject intercept predicted from the standardized
  covariates through the shared :class:`bre.models.base.Encoder` architecture with ``d_out = 1``
  (the equal-footing rule of PLAN.md section 4): ``u_s ~ N(0, sigma_u^2)``, ``sigma_u ~
  HalfNormal(1)``, ``W ~ N(0, 1)``, ``b ~ N(0, 2)``.
* ``beta_ctx[s, c] ~ N(mu_ctx[c], sigma_ctx[c]^2)`` are subject-level context slopes with
  population means ``mu_ctx ~ N(0, 1.5)`` and scales ``sigma_ctx ~ HalfNormal(1)``, one per
  context of ``ModelData.ctx_vocab``. Context effects are additive and position-independent
  (PLAN.md section 6, split (b): "classical models use additive context effects").
* ``beta_L ~ N(0, 10)``, ``beta_order, beta_tol_answered, beta_tol_yes ~ N(0, 1.5)``.
* Likelihood: the shared weighted Bernoulli/binomial :func:`bre.models.base.bernoulli_log_lik`
  over ``data.sell_rows()`` (added with ``numpyro.factor`` so rate rows with weights work).

Prior scales are weakly informative design choices (:class:`B2Prior`); the random effects use the
non-centered parametrization (``u = sigma_u z_u``, ``beta_ctx = mu_ctx + sigma_ctx z_ctx``).

Backends
--------
* :meth:`B2.fit_svi` — stochastic variational inference with an ``AutoNormal`` guide (mean-field
  Gaussian in the unconstrained space) and Adam; used by the recovery study (PLAN.md section 5).
* :meth:`B2.fit_nuts` — the No-U-Turn sampler; for the final real-data fits.

Both return a :class:`B2Posterior`: ``samples`` (``S`` posterior draws of every site, constrained
space) and ``params``, the posterior-mean pytree in the protocol layout so that
``predict_proba``, ``log_lik`` and ``n_params`` work like every other model::

    {"enc": {W (d_x, 1), b (1,), u (n_subjects, 1), log_sigma_u (1,)},
     "beta_L": (), "beta_ctx": (n_subjects, V), "mu_ctx": (V,), "log_sigma_ctx": (V,),
     "beta_order": (), "beta_tol_answered": (), "beta_tol_yes": ()}

``predict_proba(params, data)`` uses these posterior means (a plug-in prediction); with
``sample=True`` it uses one posterior draw chosen with ``key`` from ``samples`` instead.
:meth:`B2.pointwise_log_lik` evaluates the per-row log-likelihood under every draw, ``(S, n)``,
and :meth:`B2.waic` / :meth:`B2.loo` pass the sell rows of that matrix to ArviZ (``az.waic``,
``az.loo`` = PSIS-LOO).

Parameter count (``n_params`` on the posterior-mean pytree, free real scalars)::

    enc:   d_x + 1 + n_subjects + 1
    rest:  1 (beta_L) + n_subjects V (beta_ctx) + V (mu_ctx) + V (log_sigma_ctx) + 3
    total: d_x + n_subjects (V + 1) + 2 V + 6
    without the per-subject blocks u and beta_ctx (include_random_effects=False): d_x + 2 V + 6

With the shared design (``d_x = 21``, ``V = 4``): ``5 n_subjects + 35`` and ``35``. The Bayesian
effective number of parameters is ``p_waic`` / ``p_loo`` from :meth:`waic` / :meth:`loo`, which
is what the WAIC comparison of PLAN.md section 5 uses; both counts are reported side by side.

``log_prior(params)`` is the log-density of the same priors at a point (the encoder block through
:func:`bre.models.base.encoder_log_prior`, i.e. the shared inverse-gamma convention for
``sigma_u`` instead of the HalfNormal above, so a MAP through ``objective`` matches the other
encoder models); the Bayesian backends do not use it.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from bre.models.base import Encoder, ModelBase, Params, bernoulli_log_lik, count_params, standard_features
from bre.models.data import ModelData

SITES: tuple[str, ...] = (
    "W",
    "b",
    "sigma_u",
    "u",
    "beta_L",
    "mu_ctx",
    "sigma_ctx",
    "beta_ctx",
    "beta_order",
    "beta_tol_answered",
    "beta_tol_yes",
)
"""Posterior sites kept in ``B2Posterior.samples`` (constrained space; ``u`` and ``beta_ctx`` are
deterministic transforms of the non-centered latents)."""

POSTERIOR_SUMMARY_SITES: tuple[str, ...] = ("W", "b", "sigma_u", "beta_L", "mu_ctx", "sigma_ctx", "beta_order", "beta_tol_answered", "beta_tol_yes")
"""Population-level sites placed in the ArviZ ``posterior`` group (the per-subject blocks are left
out to keep the InferenceData small)."""


@dataclass(frozen=True)
class B2Prior:
    """Prior scales (standard deviations) of the module docstring."""

    w_scale: float = 1.0
    b_scale: float = 2.0
    sigma_u_scale: float = 1.0
    loss_scale: float = 10.0
    mu_ctx_scale: float = 1.5
    sigma_ctx_scale: float = 1.0
    effect_scale: float = 1.5


@dataclass
class B2Posterior:
    """Result of a B2 fit: posterior draws, the posterior-mean pytree, and fit diagnostics."""

    method: str
    samples: dict[str, np.ndarray]
    params: Params
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return int(np.asarray(self.samples["beta_L"]).shape[0])


def _samples_of(obj: B2Posterior | dict[str, Any]) -> dict[str, jnp.ndarray]:
    smp = obj.samples if isinstance(obj, B2Posterior) else obj
    missing = [s for s in SITES if s not in smp]
    if missing:
        raise ValueError(f"posterior samples lack site(s) {missing}")
    return {k: jnp.asarray(smp[k], dtype=jnp.float64) for k in SITES}


class B2(ModelBase):
    """Hierarchical Bayesian logistic regression (subject intercepts from the shared encoder,
    subject-level context slopes with population means and scales)."""

    name = "B2"
    family = "classical"

    def __init__(self, data: ModelData, prior: B2Prior | None = None) -> None:
        self.prior = B2Prior() if prior is None else prior
        self.enc = Encoder.for_data(data, d_out=1)
        self.ctx_vocab = tuple(data.ctx_vocab)
        self.n_contexts = int(data.n_contexts)
        self.n_subjects = int(data.n_subjects)
        self.d_x = int(data.d_x)
        self._cache: weakref.WeakKeyDictionary[ModelData, dict[str, jnp.ndarray]] = weakref.WeakKeyDictionary()

    # -- data arrays ------------------------------------------------------------------------------

    def arrays(self, data: ModelData) -> dict[str, jnp.ndarray]:
        """The ``jax.numpy`` arrays the model reads (cached per ``ModelData`` object): ``X``,
        ``subject_idx``, ``loss``, ``ctx_idx``, ``ctx_mask``, ``order_flag``, ``tol_answered``,
        ``tol_yes`` (from :func:`standard_features`), ``y``, ``w``, ``mask``."""
        arr = self._cache.get(data)
        if arr is None:
            if tuple(data.ctx_vocab) != self.ctx_vocab or data.n_subjects != self.n_subjects or data.d_x != self.d_x:
                raise ValueError("B2: data has a different ctx_vocab / subject list / covariates than the model was built on")
            fm = standard_features(data)
            arr = {
                "X": jnp.asarray(data.X, dtype=jnp.float64),
                "subject_idx": jnp.asarray(data.subject_idx),
                "loss": jnp.asarray(data.loss, dtype=jnp.float64),
                "ctx_idx": jnp.asarray(data.ctx_idx),
                "ctx_mask": jnp.asarray(data.ctx_mask),
                "order_flag": jnp.asarray(data.order_flag, dtype=jnp.float64),
                "tol_answered": jnp.asarray(fm.column("tol_answered"), dtype=jnp.float64),
                "tol_yes": jnp.asarray(fm.column("tol_yes"), dtype=jnp.float64),
                "y": jnp.asarray(data.y, dtype=jnp.float64),
                "w": jnp.asarray(data.w, dtype=jnp.float64),
                "mask": jnp.asarray(data.sell_mask()),
            }
            self._cache[data] = arr
        return arr

    # -- forward pass -----------------------------------------------------------------------------

    def logits_from_arrays(self, params: Params, arr: dict[str, jnp.ndarray]) -> jnp.ndarray:
        alpha = self.enc.apply(params["enc"], arr["X"], arr["subject_idx"])[:, 0]
        ctx_idx, ctx_mask = arr["ctx_idx"], arr["ctx_mask"]
        beta_ctx = jnp.asarray(params["beta_ctx"], dtype=jnp.float64)[arr["subject_idx"]]  # (n, V)
        safe = jnp.where(ctx_mask, ctx_idx, 0)
        gathered = jnp.take_along_axis(beta_ctx, safe, axis=1)
        ctx = jnp.sum(jnp.where(ctx_mask, gathered, 0.0), axis=1)
        return (
            alpha
            + params["beta_L"] * arr["loss"]
            + ctx
            + params["beta_order"] * arr["order_flag"]
            + params["beta_tol_answered"] * arr["tol_answered"]
            + params["beta_tol_yes"] * arr["tol_yes"]
        )

    def logits(self, params: Params, data: ModelData) -> jnp.ndarray:
        return self.logits_from_arrays(params, self.arrays(data))

    def predict_proba(
        self,
        params: Params,
        data: ModelData,
        *,
        sample: bool = False,
        samples: B2Posterior | dict[str, Any] | None = None,
        key=None,
    ) -> jnp.ndarray:
        """``(n,)`` ``P(sell)``: from ``params`` (the posterior means) by default; with
        ``sample=True`` from one posterior draw of ``samples`` chosen uniformly with ``key``."""
        if sample:
            if samples is None or key is None:
                raise ValueError("predict_proba(sample=True) needs samples= and key=")
            smp = _samples_of(samples)
            s = jax.random.randint(key, (), 0, smp["beta_L"].shape[0])
            params = self.params_from_draw(smp, s)
        return jax.nn.sigmoid(self.logits(params, data))

    # -- protocol pieces --------------------------------------------------------------------------

    def init_params(self, rng_key, data: ModelData, restart: int = 0) -> Params:
        """Small-normal starting point in the protocol layout (used by a MAP fallback through
        ``objective``; the Bayesian backends initialize themselves)."""
        key = self.restart_key(rng_key, restart)
        k_enc, k_l, k_mu, k_o, k_a, k_y = jax.random.split(key, 6)
        f64 = jnp.float64
        V = self.n_contexts
        mu = 0.1 * jax.random.normal(k_mu, (V,), dtype=f64)
        return {
            "enc": self.enc.init(k_enc, 0),
            "beta_L": 0.1 * jax.random.normal(k_l, (), dtype=f64),
            "beta_ctx": jnp.broadcast_to(mu, (self.n_subjects, V)) + 0.0,
            "mu_ctx": mu,
            "log_sigma_ctx": jnp.full((V,), float(np.log(0.5)), dtype=f64),
            "beta_order": 0.1 * jax.random.normal(k_o, (), dtype=f64),
            "beta_tol_answered": 0.1 * jax.random.normal(k_a, (), dtype=f64),
            "beta_tol_yes": 0.1 * jax.random.normal(k_y, (), dtype=f64),
        }

    def log_prior(self, params: Params) -> jnp.ndarray:
        """Log-density of the priors at ``params`` (see the module docstring's last paragraph)."""
        pr = self.prior
        enc = params["enc"]
        lp = self.enc.log_prior(enc)
        lp = lp + jnp.sum(dist.Normal(0.0, pr.w_scale).log_prob(enc["W"]))
        lp = lp + jnp.sum(dist.Normal(0.0, pr.b_scale).log_prob(enc["b"]))
        sigma_ctx = jnp.exp(jnp.asarray(params["log_sigma_ctx"], dtype=jnp.float64))
        lp = lp + jnp.sum(dist.Normal(params["mu_ctx"][None, :], sigma_ctx[None, :]).log_prob(params["beta_ctx"]))
        lp = lp + jnp.sum(dist.HalfNormal(pr.sigma_ctx_scale).log_prob(sigma_ctx) + jnp.log(sigma_ctx))
        lp = lp + jnp.sum(dist.Normal(0.0, pr.mu_ctx_scale).log_prob(params["mu_ctx"]))
        lp = lp + dist.Normal(0.0, pr.loss_scale).log_prob(params["beta_L"])
        for k in ("beta_order", "beta_tol_answered", "beta_tol_yes"):
            lp = lp + dist.Normal(0.0, pr.effect_scale).log_prob(params[k])
        return lp

    def n_params(self, params: Params, include_random_effects: bool = True) -> int:
        """``d_x + n_subjects (V + 1) + 2 V + 6``; without ``u`` and ``beta_ctx``: ``d_x + 2 V + 6``."""
        enc = self.enc.n_params(params["enc"], include_random_effects=include_random_effects)
        skip = ("enc",) if include_random_effects else ("enc", "beta_ctx")
        rest = count_params({k: v for k, v in params.items() if k not in skip})
        return enc + rest

    # -- NumPyro model ----------------------------------------------------------------------------

    def model(self, arr: dict[str, jnp.ndarray]) -> None:
        """The NumPyro model of the module docstring over the arrays of :meth:`arrays`."""
        pr = self.prior
        n_subj, d_x, V = self.n_subjects, self.d_x, self.n_contexts
        W = numpyro.sample("W", dist.Normal(0.0, pr.w_scale).expand([d_x, 1]).to_event(2))
        b = numpyro.sample("b", dist.Normal(0.0, pr.b_scale).expand([1]).to_event(1))
        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(pr.sigma_u_scale).expand([1]).to_event(1))
        mu_ctx = numpyro.sample("mu_ctx", dist.Normal(0.0, pr.mu_ctx_scale).expand([V]).to_event(1))
        sigma_ctx = numpyro.sample("sigma_ctx", dist.HalfNormal(pr.sigma_ctx_scale).expand([V]).to_event(1))
        with numpyro.plate("subjects", n_subj):
            z_u = numpyro.sample("z_u", dist.Normal(0.0, 1.0).expand([1]).to_event(1))
            z_ctx = numpyro.sample("z_ctx", dist.Normal(0.0, 1.0).expand([V]).to_event(1))
        u = numpyro.deterministic("u", sigma_u * z_u)
        beta_ctx = numpyro.deterministic("beta_ctx", mu_ctx + sigma_ctx * z_ctx)
        beta_L = numpyro.sample("beta_L", dist.Normal(0.0, pr.loss_scale))
        beta_order = numpyro.sample("beta_order", dist.Normal(0.0, pr.effect_scale))
        beta_tol_answered = numpyro.sample("beta_tol_answered", dist.Normal(0.0, pr.effect_scale))
        beta_tol_yes = numpyro.sample("beta_tol_yes", dist.Normal(0.0, pr.effect_scale))
        params = {
            "enc": {"W": W, "b": b, "u": u, "log_sigma_u": jnp.log(sigma_u)},
            "beta_L": beta_L,
            "beta_ctx": beta_ctx,
            "mu_ctx": mu_ctx,
            "log_sigma_ctx": jnp.log(sigma_ctx),
            "beta_order": beta_order,
            "beta_tol_answered": beta_tol_answered,
            "beta_tol_yes": beta_tol_yes,
        }
        p = jax.nn.sigmoid(self.logits_from_arrays(params, arr))
        numpyro.factor("sell_lik", jnp.sum(bernoulli_log_lik(p, arr["y"], arr["w"], arr["mask"])))

    # -- posterior draws <-> params ---------------------------------------------------------------

    def params_from_draw(self, samples: B2Posterior | dict[str, Any], s) -> Params:
        """Protocol pytree of draw ``s`` (an int or a traced index)."""
        smp = _samples_of(samples)
        return {
            "enc": {
                "W": smp["W"][s],
                "b": smp["b"][s],
                "u": smp["u"][s],
                "log_sigma_u": jnp.log(smp["sigma_u"][s]),
            },
            "beta_L": smp["beta_L"][s],
            "beta_ctx": smp["beta_ctx"][s],
            "mu_ctx": smp["mu_ctx"][s],
            "log_sigma_ctx": jnp.log(smp["sigma_ctx"][s]),
            "beta_order": smp["beta_order"][s],
            "beta_tol_answered": smp["beta_tol_answered"][s],
            "beta_tol_yes": smp["beta_tol_yes"][s],
        }

    def posterior_mean_params(self, samples: B2Posterior | dict[str, Any]) -> Params:
        """Posterior means of every site in the protocol layout (``log_sigma_* = log`` of the mean
        scale)."""
        smp = _samples_of(samples)
        mean = {k: jnp.mean(v, axis=0) for k, v in smp.items()}
        return {
            "enc": {"W": mean["W"], "b": mean["b"], "u": mean["u"], "log_sigma_u": jnp.log(mean["sigma_u"])},
            "beta_L": mean["beta_L"],
            "beta_ctx": mean["beta_ctx"],
            "mu_ctx": mean["mu_ctx"],
            "log_sigma_ctx": jnp.log(mean["sigma_ctx"]),
            "beta_order": mean["beta_order"],
            "beta_tol_answered": mean["beta_tol_answered"],
            "beta_tol_yes": mean["beta_tol_yes"],
        }

    # -- backends ---------------------------------------------------------------------------------

    def fit_svi(
        self,
        data: ModelData,
        steps: int,
        key,
        *,
        lr: float = 0.01,
        num_samples: int = 200,
        num_particles: int = 1,
    ) -> B2Posterior:
        """SVI with an ``AutoNormal`` guide and Adam(``lr``) for ``steps`` full-batch steps, then
        ``num_samples`` posterior draws. ``diagnostics``: the ELBO loss trace (``losses``), its
        last value and the settings."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        arr = self.arrays(data)
        guide = AutoNormal(self.model)
        svi = SVI(self.model, guide, numpyro.optim.Adam(step_size=float(lr)), Trace_ELBO(num_particles=num_particles))
        k_run, k_draw = jax.random.split(key)
        result = svi.run(k_run, int(steps), arr, progress_bar=False)
        losses = np.asarray(result.losses, dtype=np.float64)
        if not np.isfinite(losses[-1]):
            raise RuntimeError("B2.fit_svi: the ELBO diverged (non-finite final loss)")
        predictive = Predictive(self.model, guide=guide, params=result.params, num_samples=int(num_samples), return_sites=list(SITES))
        draws = predictive(k_draw, arr)
        samples = {k: np.asarray(draws[k], dtype=np.float64) for k in SITES}
        params = self.posterior_mean_params(samples)
        diagnostics = {"losses": losses, "final_loss": float(losses[-1]), "steps": int(steps), "lr": float(lr), "num_samples": int(num_samples), "guide": "AutoNormal"}
        return B2Posterior(method="svi", samples=samples, params=params, diagnostics=diagnostics)

    def fit_nuts(
        self,
        data: ModelData,
        key,
        num_warmup: int = 500,
        num_samples: int = 500,
        *,
        num_chains: int = 1,
        target_accept_prob: float = 0.8,
    ) -> B2Posterior:
        """NUTS (``num_chains`` sequential chains). ``diagnostics``: number of divergent
        transitions, the maximum split-R-hat over the population-level sites, and the settings."""
        arr = self.arrays(data)
        kernel = NUTS(self.model, target_accept_prob=float(target_accept_prob))
        mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=int(num_chains), chain_method="sequential", progress_bar=False)
        mcmc.run(key, arr, extra_fields=("diverging",))
        draws = mcmc.get_samples()
        samples = {k: np.asarray(draws[k], dtype=np.float64) for k in SITES}
        by_chain = mcmc.get_samples(group_by_chain=True)
        rhat = {}
        for site in POSTERIOR_SUMMARY_SITES:
            v = np.asarray(by_chain[site])
            if v.shape[1] >= 4:
                rhat[site] = float(np.nanmax(numpyro.diagnostics.split_gelman_rubin(v)))
        divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        params = self.posterior_mean_params(samples)
        diagnostics = {
            "divergences": divergences,
            "max_rhat": (max(rhat.values()) if rhat else float("nan")),
            "rhat": rhat,
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
        }
        return B2Posterior(method="nuts", samples=samples, params=params, diagnostics=diagnostics)

    # -- posterior predictive evaluation ----------------------------------------------------------

    def pointwise_log_lik(self, samples: B2Posterior | dict[str, Any], data: ModelData, chunk_size: int = 32) -> np.ndarray:
        """``(S, n)`` per-row log-likelihood under every posterior draw (0 off ``sell_rows``),
        evaluated in chunks of ``chunk_size`` draws."""
        smp = _samples_of(samples)
        arr = self.arrays(data)
        S = int(smp["beta_L"].shape[0])

        def one(s):
            p = jax.nn.sigmoid(self.logits_from_arrays(self.params_from_draw(smp, s), arr))
            return bernoulli_log_lik(p, arr["y"], arr["w"], arr["mask"])

        batched = jax.jit(jax.vmap(one))
        out = np.empty((S, data.n), dtype=np.float64)
        for start in range(0, S, max(int(chunk_size), 1)):
            idx = jnp.arange(start, min(start + chunk_size, S))
            out[start : start + len(idx)] = np.asarray(batched(idx))
        return out

    def inference_data(self, samples: B2Posterior | dict[str, Any], data: ModelData, log_lik: np.ndarray | None = None) -> az.InferenceData:
        """ArviZ ``InferenceData`` with the population-level sites as one chain in ``posterior`` and
        the sell-row log-likelihood matrix as ``log_likelihood["sell"]`` (``(1, S, n_sell)``)."""
        smp = _samples_of(samples)
        ll = self.pointwise_log_lik(smp, data) if log_lik is None else np.asarray(log_lik)
        rows = data.sell_rows()
        posterior = {k: np.asarray(smp[k])[None] for k in POSTERIOR_SUMMARY_SITES}
        return az.from_dict(posterior=posterior, log_likelihood={"sell": ll[None, :, rows]})

    def waic(self, samples: B2Posterior | dict[str, Any], data: ModelData, log_lik: np.ndarray | None = None):
        """``arviz.waic`` on the sell rows: an ``ELPDData`` with ``elpd_waic``, ``p_waic``, ``se``,
        ``warning`` and the per-row ``waic_i``. ArviZ is always called with ``pointwise=True``:
        in arviz 0.23 the non-pointwise result is indexed ``waic`` instead of ``elpd_waic`` (and
        its ``__str__`` fails), so the pointwise form is the one with the documented names.
        Pass ``log_lik`` (from :meth:`pointwise_log_lik`) to avoid recomputing it."""
        return az.waic(self.inference_data(samples, data, log_lik), pointwise=True)

    def loo(self, samples: B2Posterior | dict[str, Any], data: ModelData, log_lik: np.ndarray | None = None):
        """``arviz.loo`` (PSIS-LOO) on the sell rows: an ``ELPDData`` with ``elpd_loo``, ``p_loo``,
        ``se``, ``warning``, ``loo_i`` and ``pareto_k`` (same ``pointwise=True`` rule as
        :meth:`waic`)."""
        return az.loo(self.inference_data(samples, data, log_lik), pointwise=True)

    # -- reporting --------------------------------------------------------------------------------

    def population_params(self, params: Params) -> dict[str, Any]:
        """The population-level scalars by name: ``beta_L``, ``beta_order``, ``beta_tol_answered``,
        ``beta_tol_yes``, ``sigma_u``, and ``mu_ctx`` / ``sigma_ctx`` by context tag."""
        return {
            "beta_L": float(params["beta_L"]),
            "beta_order": float(params["beta_order"]),
            "beta_tol_answered": float(params["beta_tol_answered"]),
            "beta_tol_yes": float(params["beta_tol_yes"]),
            "sigma_u": float(jnp.exp(params["enc"]["log_sigma_u"])[0]),
            "mu_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.asarray(params["mu_ctx"]))},
            "sigma_ctx": {tag: float(v) for tag, v in zip(self.ctx_vocab, np.exp(np.asarray(params["log_sigma_ctx"])))},
        }

    def subject_summary(self, params: Params, data: ModelData) -> pd.DataFrame | None:
        """One row per subject: the intercept ``alpha_s = W^T x_s + b + u_s``, ``u_s`` and the
        subject's context slopes (one column per context tag)."""
        alpha = np.asarray(self.enc.apply_subjects(params["enc"], data.X))[:, 0]
        out = pd.DataFrame({"subject_id": data.subject_ids, "intercept": alpha, "u": np.asarray(params["enc"]["u"])[:, 0]})
        beta = np.asarray(params["beta_ctx"])
        for j, tag in enumerate(self.ctx_vocab):
            out[f"beta_ctx[{tag}]"] = beta[:, j]
        return out


__all__ = ["B2", "B2Posterior", "B2Prior", "POSTERIOR_SUMMARY_SITES", "SITES"]
