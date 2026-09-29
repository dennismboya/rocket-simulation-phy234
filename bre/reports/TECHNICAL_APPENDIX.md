# TECHNICAL_APPENDIX.md — the equations actually used, transcribed from the code

Each block names its source as `file:function` (paths relative to `bre/src/bre/` unless stated).
Text in the equations is the docstring's own; nothing is added. Basis `e0 = |hold>`,
`e1 = |sell>`; states in C², `complex128` / `float64` (`models/quantum/core.py` module docstring).

## 1. State preparation and gauge — `models/quantum/core.py:prepare_state`; encoder `models/base.py:Encoder` (INTERFACE.md §3)

Encoder: `z_i = W^T x_i + b + u_{s(i)}`, `x_i` the subject's standardized covariates, `z_i ∈ R^{d_out}`;
Q-models use `d_out = 4`. State: `psi = normalize(v[:2] + i v[2:])` then `psi -> exp(-i arg psi_0) psi`,
so the hold amplitude is real and non-negative; of the four numbers two are identifiable,
`psi = (cos a, e^{ib} sin a)`, `a ∈ [0, π/2]`, `b ∈ (−π, π]`. Random-effect prior
`u ~ N(0, diag(sigma^2))` with an inverse-gamma(2, 0.5) hyperprior on each `sigma_k^2`
(`models/base.py:encoder_log_prior`).

## 2. Unitaries from su(2) generators — `core.py:unitary`, `core.py:loss_unitary`

`G_k = i sigma_k` (skew-Hermitian), `U(theta) = expm(sum_k theta_k G_k) = cos|theta| I + i sin|theta| (theta/|theta|)·sigma`,
computed with `jax.scipy.linalg.expm` so the gradient at `theta = 0` is defined; `theta = 0` gives `I`.
Loss unitary: `U_L(L) = U(L · theta_L)`, `L = |loss_pct| ≥ 0`, so `L = 0` is the identity.
Gauge: `U(theta + π theta/|theta|) = −U(theta)`, so `theta` is identified modulo π along its direction
(`models/quantum/q2_context_unitary.py:fold_theta` maps to `|theta| ≤ π/2`); the discrete gauge
group {identity, conjugate, z_flip, conjugate_z_flip} acts on `(theta, v, phi)` by the sign patterns
in `q2_context_unitary.py:GAUGE_ELEMENTS`; `theta_none = 0` is the reference gauge.

## 3. Context composition — `core.py:apply_contexts`, `core.py:chain_pure`

`psi' = U_{c_m} … U_{c_1} U_L(L) psi` (loss first, then contexts in presentation order; masked
positions are the identity). Fast path with unitaries computed once per parameter table:
`q2_context_unitary.py:forward_pure_fast`.

## 4. Born rule and Lüders update — `core.py:born`, `core.py:lueders`, `core.py:born_rho`, `core.py:lueders_rho`, `core.py:tolerance_projector`

`P_SELL = diag(0, 1)`, `P_HOLD = diag(1, 0)`; tolerance "yes" projector `P_yes = |yes><yes|`,
`|yes> = (cos phi, sin phi)`, "no" projector `I − P_yes`. Born: `P = ||P psi||^2 = <psi|P|psi>`
(pure), `P = Tr(P rho)` (density matrix). Lüders: `psi -> P psi / ||P psi||`,
`rho -> P rho P / Tr(P rho)` (zero state when the outcome has probability zero; idempotent).

## 5. Dephasing — `core.py:dephase`, `core.py:chain_rho`

Master equation `d rho/dt = −i[H, rho] + gamma (L rho L† − rho)` with `H = 0` during the step and
`L = sigma_z`; solution: off-diagonal elements multiplied by `exp(−2 gamma t)`, diagonal constant.
`chain_rho`: `rho -> dephase(U_L rho U_L†)` then, per unmasked context, `rho -> dephase(U_c rho U_c†)`,
all at rate `gamma`. `gamma = 0` is the unitary chain; `gamma -> ∞` projects onto the diagonal.

## 6. Q2 and Q4 forward passes — `core.py:q_forward_pure`, `core.py:q_forward_rho`

Scenario-first: `p = ||P_SELL U_{c_m} … U_{c_1} U_L(L) psi||^2` (Q2) or `p = Tr(P_SELL rho')` (Q4).
Tolerance-first with observed answer `a ∈ {0, 1}` (`P_1 = P_yes(phi)`, `P_0 = I − P_1`):
`psi_a = P_a psi / ||P_a psi||`, then the same chain on `psi_a`. Tolerance-first with the answer
unobserved: the Lüders mixture `p = sum_a ||P_a psi||^2 p(sell | a)`. Q4 limits
(`models/quantum/q4_open_system.py` docstring): `gamma_s = 0` reproduces Q2 exactly;
`gamma_s -> ∞` (`log_gamma = ln 1e3`) makes every step a classical Markov transition with
`T_ij = |(U)_ij|^2`, the initial coherence entering only the first step. Per-investor rate prior
(`q4_open_system.py:lognormal_population_log_prior`): `log gamma_s ~ N(gamma_mu, exp(gamma_log_sigma)^2)`
with an inverse-gamma(2, 1) hyperprior on the variance. Q2 prior (`q2_context_unitary.py:Q2.log_prior`):
encoder prior plus `−(||theta_L||^2 + ||theta_ctx||^2)/(2 s^2)`, `s = 2`.

## 7. Interference terms (definitions fixed in PLAN.md §4) — `core.py:ltp_interference`, `core.py:mixture_interference`, `core.py:order_effect`

`delta_LTP = P(sell | scenario-first) − sum_a P(a, sell | tolerance-first)`, with
`P(a, sell | tolerance-first) = ||P_a psi||^2 P(sell | a)`; a classical model predicts 0.
`delta_mix = ||P_SELL normalize(sum_c sqrt(p_c) U_c psi)||^2 − sum_c p_c ||P_SELL U_c psi||^2`.
`Delta_order(c1, c2) = P(sell | c1, c2) − P(sell | c2, c1)`, `P(sell | c1, c2) = ||P_SELL U_{c2} U_{c1} U_L(L) psi||^2`;
zero when `U_{c1}` and `U_{c2}` commute (open-system versions `ltp_interference_rho`, `order_effect_rho`).

## 8. Q1 cells and the QQ statistic — `models/quantum/q1_order.py:q1_cells`, `qq_statistic`, `q1_cells_with_unitary`

`psi = (cos alpha, sin alpha) ∈ R^2`; `P_Ay = |e0><e0|`, `P_By = |b><b|`, `|b> = (cos phi, sin phi)`;
`P(Ay, By) = ||P_By P_Ay psi||^2` and likewise for the eight cells. Closed form with
`c1 = cos^2 alpha`, `c2 = cos^2(alpha − phi)`, `s = sin^2 phi`:
`AyBy = (1−s) c1, AyBn = s c1, AnBy = s (1−c1), AnBn = (1−s)(1−c1)`;
`ByAy = (1−s) c2, ByAn = s c2, BnAy = s (1−c2), BnAn = (1−s)(1−c2)`.
QQ statistic `q = [P(AyBn) + P(AnBy)] − [P(ByAn) + P(BnAy)] = 0` for any state, dimension and pair
of projectors. Information variant: rotation `U(theta)` between the questions gives
`q = −sin(2 theta) sin(2 phi)`.

## 9. Q3 Hamiltonian — `models/quantum/q3_dynamics.py:hamiltonian_coefficients`, `evolution_unitaries`, `time_from_days`

`H_i = H_payoff(L_i) + H_dissonance(c_i)`, `H_payoff(L) = L (theta_p · sigma)`,
`H_dissonance(c) = (sum_k w[c_k]) (theta_d · sigma)` with unit `theta_d = (sin a cos b, sin a sin b, cos a)`;
`h_i = L_i theta_p + (sum_k w[c_k]) theta_d`; `psi_i(t) = expm(−i H_i t) psi_s(0) = U(−t h_i) psi_s(0)`;
`t_i = log1p(days_since_news_i)`, default `t = 1`; `P(sell) = ||P_sell psi_i(t)||^2`, with the
question-order branches of §6 applied to `psi_s(0)`. `Delta_order ≡ 0` by construction.

## 10. Q5 quantum decision theory — `models/quantum/q5_qdt.py:Q5.utility_factor`, `attraction_factor`

`P(sell) = f_sell + q`, `P(hold) = f_hold − q`, `f_sell + f_hold = 1`;
`EU_sell = u(1 − L)`, `EU_hold = (1 − p_rec) u(1 − kappa L)` (CRRA `u` of §13),
`f_sell = sigmoid(tau (EU_sell − EU_hold))`, `p_rec = sigmoid(pr_logit)`;
`z_i = a_{s(i)} + sum_k b_ctx[c_k] + b_order order_flag + b_tol tol_yes`, `a_s = Encoder(x_s) + u_s`;
`q_i = 0.5 tanh(z_i) · 2 min(f_sell, f_hold) ∈ (−0.5, 0.5)`. Quarter-law check: mean `|q|` over
sell rows with a subject bootstrap next to the reference 0.25 (`Q5.quarter_law_check`).
`delta_LTP` is undefined for Q5 (returned as NaN); `Delta_order = 0` on pair rows.

## 11. B1 and B2 logits — `models/classical/b1_logistic.py:B1.logits`, `models/classical/b2_hier.py:B2.logits_from_arrays`

B1: `logit P(sell_i) = b + w_main · F_i + w_int · G_i`, `F_i` the shared main-effects features
(covariates, loss, per-position context one-hots, `order_flag`, `tol_answered`, `tol_yes`;
`models/base.py:standard_features`), `G_i` the pairwise block `loss*order_flag`, `loss*ctx{k}={tag}`,
`order_flag*ctx{k}={tag}`, `ctx{k}={a}*ctx{k'}={b}` (29 columns on the shared design);
prior `−(l2/2)||w||^2`.
B2: `logit P(sell_i) = alpha_s + beta_L L_i + sum_k beta_ctx[s, c_k] + beta_order order_flag_i + beta_tol_answered tol_answered_i + beta_tol_yes tol_yes_i`,
`alpha_s = W^T x_s + b + u_s` (`d_out = 1`), `u_s ~ N(0, sigma_u^2)`, `sigma_u ~ HalfNormal(1)`,
`beta_ctx[s, c] ~ N(mu_ctx[c], sigma_ctx[c]^2)`, `mu_ctx ~ N(0, 1.5)`, `sigma_ctx ~ HalfNormal(1)`,
`beta_L ~ N(0, 10)`, the three effect betas `~ N(0, 1.5)`; SVI (AutoNormal) or NUTS; WAIC / PSIS-LOO via ArviZ.

## 12. B3 cumulative prospect theory — `models/classical/b3_cpt.py:cpt_value`, `prelec_weight`, `cpt_two_outcome`, `B3.reference_point`, `B3.recovery_probability`, `B3.predict_proba`

`v_i(z) = z^alpha_i` for `z ≥ 0`, `−lambda_i (−z)^beta_i` for `z < 0` (`beta_i = alpha_i` unless `separate_beta`);
`w_i(p) = exp(−(−ln p)^gamma_i)`; `(log lambda_i, log alpha_i, log gamma_i) = W^T x_i + b + u_i`.
`r = r0 + sum_k rho[c_k] + rho_order order_flag + rho_tol tol_yes`; `p_rec = sigmoid(pr0 + sum_k pr_ctx[c_k])`.
`V_sell = v(−L − r)`; hold lottery `{0 with p_rec; −kappa L with 1 − p_rec}`, `z_hi = −r`, `z_lo = −kappa L − r`:
both gains `V = w(p_hi) v(z_hi) + (1 − w(p_hi)) v(z_lo)`; both losses `V = w(p_lo) v(z_lo) + (1 − w(p_lo)) v(z_hi)`;
mixed `V = w(p_hi) v(z_hi) + w(p_lo) v(z_lo)`. `P(sell) = sigmoid(tau (V_sell − V_hold))`.

## 13. B4 Bayesian updater — `models/classical/b4_bayes_updater.py:crra_utility`, `B4.evidence`, `B4.recovery_logit`, `B4.expected_utilities`, `B4.predict_proba`

`prior_i = W^T x_{s(i)} + b + u_{s(i)}`;
`logit p_rec,i = prior_i + sum_{k=1}^{m} delta^{(m−k)} lambda_{c_k} + beta_tol · tol_answer_i`, `delta = sigmoid(delta_logit)`;
`u(w) = (w^{(1−rho)} − 1)/(1 − rho)`, `u -> log w` as `rho -> 1`, wealth floored;
`EU_sell = u(1 − L)`, `EU_hold = (1 − p_rec) u(1 − kappa L)`; `P(sell) = sigmoid(tau (EU_sell − EU_hold))`.

## 14. B6 hidden Markov forward filter — `models/classical/b6_hmm.py:forward_filter`, `transition_matrix`

`P(s_1 = 1) = sigmoid(init_logit)`; `T = [[p_00, 1 − p_00], [1 − p_11, p_11]]`, `p_ss = sigmoid(trans_logit[s])`;
`P(sell_t | s_t = s) = sigmoid(eta_{t,s})`, `eta_{t,s} = a_s + w_row[s] · f_t + w_x · x_{subject(t)}`,
`a = (a0, a0 + exp(log_da))`. Filtered predictive `P(sell_t | y_{<t}) = sum_s alpha_t(s) sigmoid(eta_{t,s})`;
`alpha_1 = pi`; after `y_t`: `alpha_t(s) <- alpha_t(s) exp(w_t [y_t log p + (1 − y_t) log(1 − p)])` normalized,
`alpha_{t+1} = alpha_t T`; non-sell rows do not update the chain.

## 15. Capacity, risk score, loss-response diagnostic — `predict.py:drawdown_capacity`, `CAPACITY_DEFINITION`, `TYPICAL_CRISIS_CONTEXTS`, `RISK_SCORE_LOSS`, `LOSS_RESPONSE_DEFINITION`

Behavioral drawdown capacity: the largest grid loss `L` in {0.05, 0.10, 0.15, 0.20, 0.30} such that
`P(sell | L', (news:recession, social:friend_sells), scenario-first) ≤ 0.25` for **every** grid loss
`L' ≤ L` (first-crossing / prefix rule; 0 when the smallest loss already crosses). Reason: the loss
unitary is a rotation, so `P(sell | L)` may come back under the target at a larger loss. Risk score:
`logit P(sell | typical crisis, L = −0.20)`. Loss-response diagnostic: share of clients whose
`P(sell | L, no context, scenario-first)` is non-decreasing over the grid, and the mean `P(sell)` per level.

## 16. Metrics — `eval.py:nll`, `brier_score`, `ece`, `reliability_table`, `auc`, `paired_nll_difference`, `subject_bootstrap`; likelihood `models/INTERFACE.md` §2

Per-row log-likelihood `ll_i = w_i [y_i log p_i + (1 − y_i) log(1 − p_i)]`, `p_i` clipped to `[1e-12, 1 − 1e-12]`,
0 off sell rows. NLL per response `= −sum w ll / sum w`. Brier `= sum w (p − y)^2 / sum w`.
ECE: 10 equal-width bins, weight-averaged `|frac_pos − mean_p|`. AUC `= P(p_pos > p_neg) + 0.5 P(tie)`.
Paired difference `NLL_a − NLL_b = mean(ll_b − ll_a)` with a subject-level bootstrap 95% interval
(1000 resamples of subjects with replacement, resampled multiplicities as row weights).

## 17. Decision rule — `eval.py:decision_rule`, `DECISION_RULE_TEXT` (PLAN.md §6, verbatim)

> "Q-model supported" only if, on real data, the best Q-model beats the best classical baseline on
> split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower parameter
> count, and the interference-term CI excludes zero. Otherwise the conclusion is "no evidence of a
> quantum-probability advantage in the available data", and the report names the data that would
> resolve it. The dashboard serves whichever model wins on held-out NLL; if that is a classical
> model, the screen says so.

Conditions as coded: `hi < 0` for the paired NLL-difference interval (Q minus classical);
`n_params_q ≤ n_params_c`; the best Q-model's `ltp_mean_ci95` finite and excluding zero (a missing
or NaN interval fails); `is_real_data` true. Split (b) = train on `none` and single-context rows, test
on ordered pairs (`eval.py:split_context_composition`).
