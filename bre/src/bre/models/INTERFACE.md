# Model interface (binding for every model in `bre/src/bre/models/`)

This file is the human-readable contract behind `base.py`, `data.py` and `quantum/core.py`. The
docstrings carry the equations; this page says what a model must do, where its numbers live, and
how to add one. PLAN.md section 4 (equal-footing rules, interference-term definitions) has
precedence where the two disagree; report the disagreement in STATUS.md rather than silently
diverging.

Import `bre.models` (or anything below it) before creating any JAX array: the package `__init__`
switches JAX to 64-bit. Every model module runs in `float64` / `complex128`.

## 1. Data a model receives: `ModelData` (`data.py`)

Built once per frame by `build_model_data(df, design=bre.design, **options)` from a validated
DecisionEvent frame (`bre.schema.COLUMNS`). Rows keep the frame's order. All arrays are numpy;
`data.jax()` returns the numeric ones as `jax.numpy` arrays plus `sell_mask`.

| field | shape / type | meaning |
|---|---|---|
| `subject_ids` | `(n_subjects,)` str | unique subject keys, sorted (`subject_id`, or `dataset/subject_id` with `subject_key="dataset/subject_id"`) |
| `subject_idx` | `(n,)` int64 | row -> index into `subject_ids` |
| `X` | `(n_subjects, d_x)` float64 | standardized covariates: full one-hot per band, z-scored numerics, one `<name>__missing` indicator per covariate (missing -> 0 in the value column, 1 in the indicator) |
| `X_columns` | tuple[str] | names of the `d_x` columns (`covariate_columns()` order) |
| `covariate_stats` | dict | numeric covariate -> `(mean, std)` used for z-scoring; pass back as `covariate_stats=` to standardize new subjects (the API must) |
| `loss` | `(n,)` float64 | `abs(loss_pct)`, 0 when null; the `L` of `loss_unitary` |
| `loss_signed` | `(n,)` float64 | `loss_pct` as stored (negative = loss), 0 when null; for datasets where the sign carries information (CPC18 analog) |
| `ctx_idx` | `(n, K)` int64 | context tag at position k -> index into `ctx_vocab`; `-1` padding; `K = n_ctx_positions` (default 2) |
| `ctx_mask` | `(n, K)` bool | True where `ctx_idx >= 0`; the `none` condition is an all-False row |
| `ctx_vocab` | tuple[str] | `design.NONNULL_CONTEXTS` followed by extra tags met in the frame (sorted); `V = len(ctx_vocab)` sizes a `(V, 3)` context parameter table |
| `order_flag` | `(n,)` int64 | 1 when the tolerance question came before the scenario (`question_order_id == "tolerance-first"`, or, with a null order id, `"tolerance"` in `prior_question_ids`) |
| `tol_answer` | `(n,)` float64 | the tolerance answer (0/1) observed just before a tolerance-first sell row; NaN otherwise (rule below) |
| `y` | `(n,)` float64 | the response |
| `w` | `(n,)` float64 | likelihood weight: `covariates["weight"]` when present, else 1 |
| `row_kind` | `(n,)` str | `sell` (binary sell/hold), `rate` (aggregate share), `tol` (tolerance row), `other` |
| `dataset`, `session_id`, `position` | `(n,)` | carried for splits and sequential models |
| `is_synthetic` | `(n,)` bool | carried so no synthetic output can be labeled real |
| `index` | `(n,)` | source-frame index labels, for joining outputs back |

Row kinds and the likelihood: only `data.sell_rows()` (kinds `sell` and `rate`; `sell_mask()`
is the boolean form) enter a model's likelihood. `tol` rows are used only through `tol_answer`.
`other` rows (allocation sliders, Likert, non-tolerance yes/no) are carried but unused by the
sell likelihood. `sell_types=("binary_sell", "lottery_choice")` widens the `sell` kind for a
dataset whose loader coded the safe-option analog as `lottery_choice`.

`tol_answer` rule: within the same `(dataset, subject_id, session_id)`, the `binary_yes_no` row
with `scenario_id == "tolerance"` at `position_in_session == position - 1` supplies the answer.
It is kept only for tolerance-first rows that are not themselves tolerance rows; a scenario-first
row that happens to follow the previous item's tolerance row gets NaN (that answer is not part
of its context), as does a row whose tolerance row is missing. The Q-models consult `tol_answer`
only when `order_flag == 1`, and the classical features expose exactly the same information
(`order_flag`, `tol_answered`, `tol_yes`), which is the equal-footing rule in practice.

Other rules (`build_model_data` docstring has the full list): rows with `consent_training ==
False` are dropped (`require_consent=False` keeps them); a subject_id met in two datasets raises
unless `subject_key="dataset/subject_id"`; a row with more than `K` context tags raises; a fixed
`ctx_vocab=` makes unknown tags an error (apply-to-new-data mode).

`data.subset(rows)` selects rows for a split and keeps the per-subject fields and `ctx_vocab`
whole, so parameter tables fitted on the training subset index the same subjects and contexts on
the test subset.

## 2. What a model is: the `Model` protocol (`base.py`)

```python
class Model(Protocol):
    name: str                      # e.g. "Q2", "B1"; unique across models
    family: "classical" | "quantum"
    def init_params(self, rng_key, data: ModelData, restart: int) -> Params
    def log_lik(self, params, data) -> jnp.ndarray        # (n,), natural log, 0 off sell_rows
    def predict_proba(self, params, data) -> jnp.ndarray  # (n,), P(sell) for every row
    def log_prior(self, params) -> jnp.ndarray            # scalar, 0 without shrinkage
    def n_params(self, params) -> int                     # free real scalars
    # optional
    def subject_summary(self, params, data) -> pd.DataFrame | None   # one row per subject
```

`Params` is a JAX pytree; use a dict with named keys so that `count_params(params, exclude=...)`
and the reports can name blocks. Conventions (not enforced): the encoder's dict under `"enc"`;
the context table `"theta_ctx"` of shape `(V, 3)` indexed like `ctx_vocab`; `"theta_L"` `(3,)`;
`"phi"` scalar (tolerance angle); a per-subject decoherence rate as `"log_gamma"`
`(n_subjects,)`.

`ModelBase` (abstract) supplies `log_lik` from `predict_proba` through `bernoulli_log_lik`,
`log_prior = 0`, `n_params = count_params`, `objective(params, data) = -(sum log_lik + log_prior)`
(what the optimizers minimize), `subject_summary = None` and `restart_key`. Subclass it, implement
`init_params` and `predict_proba`, override `log_prior` when the model shrinks. `isinstance(m,
Model)` works at runtime (the protocol is `runtime_checkable`).

`QuantumModelBase` adds the abstract `interference_terms(params, data) -> dict` (section 6).

### Likelihood convention

Natural log, one value per row, `0.0` exactly on rows outside `sell_rows()`:

    ll_i = w_i * [ y_i log p_i + (1 - y_i) log(1 - p_i) ]      with p_i clipped to [1e-12, 1 - 1e-12]

For a binary row this is the Bernoulli log-likelihood; for a `choice_rate` row with `y` the
observed share and `w` the number of respondents it is the binomial log-likelihood up to the
combinatorial constant. Held-out NLL per response (PLAN.md section 6) is `-mean(ll[sell_rows])`
with the weights left in. Models that add other likelihoods (Beta for allocation sliders) keep the
same per-row layout and document the row kinds they cover.

### Parameter counting

`count_params(pytree)` sums `size` over array leaves (complex leaves count twice). Reports show
two numbers for models with an encoder: all free scalars, and `count_params(params,
exclude=("u",))` without the per-subject random effects (`Encoder.n_params(...,
include_random_effects=False)`). Both families share the encoder, so the difference is the same
on both sides.

### Restarts and seeds

`init_params(rng_key, data, restart)` must be deterministic in `(rng_key, restart)`; derive the
restart's key as `jax.random.fold_in(rng_key, restart)` (`ModelBase.restart_key`). PLAN.md
section 4 requires at least 5 restarts per optimizer fit; the fit driver passes `restart = 0..4`
and the same `rng_key` and keeps the best validation log-likelihood.

## 3. The shared encoder (`Encoder`, `encoder_log_prior`)

    z_i = W^T x_i + b + u_{s(i)}        x_i = X[subject_idx[i]],  z_i in R^{d_out}

params: `W (d_x, d_out)`, `b (d_out,)`, `u (n_subjects, d_out)`, `log_sigma_u (d_out,)`.
`Encoder.for_data(data, d_out)` sizes it; `init(rng_key, restart)` draws `W`, `b` and sets
`u = 0`; `apply(params, X, subject_idx)` gives `(n, d_out)`; `apply_subjects(params, X)` gives
`(n_subjects, d_out)`.

`encoder_log_prior(params, w_l2=0.0)` is the log-density of `u ~ N(0, diag(sigma^2))` with the
fitted scale `sigma = exp(log_sigma_u)`, plus an inverse-gamma(2, 0.5) hyperprior on each
`sigma_k^2` (without it the joint MAP over `(u, sigma)` is unbounded at `u = 0, sigma -> 0`),
plus an optional fixed ridge on `W`. A model that embeds the encoder adds this to its `log_prior`.

Q-models use `d_out = 4` and `psi_i = prepare_state(z_i)`; B2-style classical latents use
`d_out = 1` (the subject's log-odds offset). Using the same encoder in both families is the
equal-footing rule of PLAN.md section 4.

## 4. The shared classical feature matrix (`standard_features`)

`standard_features(data, loss_column="loss") -> FeatureMatrix(F, columns)` with `F (n, d_f)`:

1. `X_columns` — the subject's covariates expanded per row;
2. `loss` (or `loss_signed`);
3. `ctx{k}={tag}` for `k = 0..K-1`, every tag of `ctx_vocab` — per-position one-hots (an ordered
   pair lights one column per position; `none` lights none);
4. `order_flag`;
5. `tol_answered` (`tol_answer` observed) and `tol_yes` (`tol_answer == 1`).

Classical models take these columns; interactions (B1: pairwise among loss, contexts, order) are
built inside the model so that the shared matrix stays main-effects only. `FeatureMatrix.column(name)`
returns one column by name.

## 5. Q-model core (`quantum/core.py`)

Basis `e0 = |hold>`, `e1 = |sell>`; states in C^2 (`psi`) or 2x2 density matrices (`rho`).

* `unitary(theta) = expm(sum_k theta_k G_k)`, `G_k = i sigma_k`; `unitaries(thetas)` batched.
* `loss_unitary(theta_L, L) = unitary(L * theta_L)`, `L = |loss_pct|`, so `L = 0` is the identity.
* `P_SELL = diag(0, 1)`, `P_HOLD = diag(1, 0)`, `tolerance_projector(phi) = |yes><yes|`,
  `|yes> = (cos phi, sin phi)`; the "no" projector is `I - P_yes`.
* `born(psi, P) = ||P psi||^2`, `lueders(psi, P) = P psi / ||P psi||` (zero vector when the
  outcome has probability zero); `born_rho`, `lueders_rho` for density matrices; `to_rho`.
* `prepare_state(v)`: `normalize(v[:2] + i v[2:])` with the global phase fixed so the hold
  amplitude is real and non-negative (the gauge, section 7).
* `apply_contexts(psi, thetas, mask)`: `U_1` first, then `U_2`, ...; masked entries skipped.
  `apply_contexts_rho(rho, thetas, mask, gamma)` dephases after every unmasked context.
* `dephase(rho, gamma, t=1)`: off-diagonals times `exp(-2 gamma t)` (Lindblad, `L = sigma_z`).
* `chain_pure(psi, theta_L, L, ctx_thetas, ctx_mask) = U_{c_m} ... U_{c_1} U_L(L) psi`;
  `chain_rho(rho, ..., gamma)` the same with dephasing after `U_L` and after each context.
* `q_forward_pure(psi, theta_L, L, ctx_thetas, ctx_mask, order_flag, phi_tol, tol_answer)`:
  scenario-first `||P_SELL chain psi||^2`; tolerance-first with observed `a`: the chain on the
  Lüders-collapsed `psi_a`; tolerance-first with `tol_answer = NaN`: the mixture
  `sum_a ||P_a psi||^2 p(sell | a)`. `q_forward_rho(rho, ..., gamma)` is the open-system version;
  `gamma = 0` reproduces the pure model, `gamma -> inf` is a classical Markov chain over the
  context steps with `T_ij = |(U_c)_ij|^2` (the initial coherence still enters the first step).
* Batched: `gather_context_thetas(theta_ctx, ctx_idx, ctx_mask) -> (n, K, 3)`,
  `q_forward_pure_batch`, `q_forward_rho_batch` (per-row `gamma`), `ltp_interference_batch`,
  `ltp_interference_rho_batch`. All public functions are `jax.jit`-wrapped and differentiable in
  every continuous argument.

A Q2-shaped `predict_proba` is therefore:

```python
psi = jax.vmap(core.prepare_state)(enc.apply(params["enc"], data.X, data.subject_idx))
ctx = core.gather_context_thetas(params["theta_ctx"], data.ctx_idx, data.ctx_mask)
p = core.q_forward_pure_batch(psi, params["theta_L"], data.loss, ctx, data.ctx_mask,
                              data.order_flag, params["phi"], data.tol_answer)
```

## 6. What a Q-model must log: `interference_terms(params, data)`

Definitions fixed in PLAN.md section 4 and implemented in `core.py`:

* `ltp` (always, `(n,)`): `delta_LTP = P(sell | scenario-first) - sum_a P(a, sell | tolerance-first)`
  evaluated for every row with the row's loss and contexts (`ltp_interference` /
  `ltp_interference_rho`). Classical models predict 0. Note: it vanishes only when the tolerance
  projectors commute with the Heisenberg-picture sell projector of the whole chain (e.g.
  `phi = 0` with no chain or a `sigma_z`-only chain); a generic chain gives a non-zero value even
  at `phi = 0`.
* `order` (always, `(n,)`): `Delta_order(c1, c2) = P(sell | c1, c2) - P(sell | c2, c1)` for rows
  that are ordered pairs (`order_effect` / `order_effect_rho`), NaN otherwise. Zero when the two
  unitaries commute; with dephasing (`gamma > 0`) only when they also commute with `sigma_z`.
* `mix` (when the model defines a mixture cause frame): `delta_mix = ||P_SELL normalize(sum_c
  sqrt(p_c) U_c psi)||^2 - sum_c p_c ||P_SELL U_c psi||^2` (`mixture_interference`).

The evaluation code bootstraps these per subject; the dashboard shows `ltp` with its CI on the
transparency page. Keys are `base.INTERFERENCE_KEYS`.

## 7. Gauge fixing and identifiability (Q-models)

* State: `prepare_state` fixes `||psi|| = 1` and `arg psi_0 = 0`, so an encoder output in R^4
  carries two identifiable numbers (`psi = (cos a, e^{ib} sin a)`).
* `theta_none = 0`: the `none` condition has no unitary (all-False mask row); every context
  parameter is relative to it.
* `U(theta + pi theta/|theta|) = -U(theta)`, and a global sign does not change any probability, so
  `theta` is identified modulo `pi` along its direction: report `|theta| mod pi` and fold the
  sign into the direction. Initialize with `|theta| < pi`.
* A `sigma_z` component of the last unitary before the sell measurement multiplies the sell
  amplitude by a phase and is invisible to `P(sell)` in scenario-first rows; it is identified only
  through what follows (a tolerance-first collapse before it, or a later question). Expect the
  recovery study (Phase 2) to show this; do not read a flat `theta_z` posterior as a bug.
* Q4: `gamma_i` is identified by how far the row-level probabilities sit between the coherent
  (`gamma = 0`) and the Markov (`gamma -> inf`) predictions; with `sigma_z`-only chains they
  coincide and `gamma` is not identified.

## 8. Checklist for adding a model

1. One module per model in `quantum/` or `classical/`, class named after the PLAN id (`Q2`,
   `B1`, ...), `name` and `family` set, subclassing `ModelBase` or `QuantumModelBase`.
2. Inputs come only from `ModelData` (`X`, `loss`/`loss_signed`, `ctx_idx`/`ctx_mask`,
   `order_flag`, `tol_answer`, `w`, `y`) or from `standard_features(data)`. No frame access.
3. `init_params(rng_key, data, restart)` deterministic in `(rng_key, restart)`, restart folded in
   with `fold_in`. Use `Encoder.for_data(data, d_out)` for any covariate-to-latent map.
4. `predict_proba` returns `(n,)` in `[0, 1]` for every row, finite everywhere, and is
   differentiable (`jax.grad` of `objective` finite at the initial parameters).
5. `log_prior` returns the log-density of every shrinkage term (encoder: `Encoder.log_prior`).
6. `n_params` counts free scalars; document what is per subject.
7. Q-models: `interference_terms` with `ltp` and `order`; log them at every fit checkpoint.
8. Optional `subject_summary` with one row per `subject_ids` entry (state angles `a`, `b`;
   Q4 `gamma_i` as "consistency score"; classical latents), joined by `subject_id`.
9. Tests in `tests/test_<model>.py`: shapes; `log_lik` zero off `sell_rows`; probabilities in
   `[0, 1]`; a known-parameter recovery on a small synthetic table from the model's own generator;
   the model's structural identities (e.g. Q2 with all `theta = 0` equals the static Born
   classifier; Q4 with `gamma = 1e3` equals the classical mixture; Q1 satisfies the QQ equality
   on Q1-generated data).
10. Register the model in the fit driver's registry with its parameter-count note, and add one
    line to STATUS.md.
