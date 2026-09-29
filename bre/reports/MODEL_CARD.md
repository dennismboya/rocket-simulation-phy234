# MODEL_CARD.md — served model `demo-q4-v2` (Behavioral Risk Engine)

Written 2026-09-29 (Phase 8). Every number below is copied from a file in this repository and
cited by path; nothing is typed from memory. Wording rule of the project (CLAUDE.md): "quantum"
means quantum probability computed on an ordinary CPU; the models are "quantum-probability
models" or "Q-models". The served model was **trained on synthetic data only**
(`runs/models/demo-q4-v2/artifact.json`: `"is_synthetic_training": true`); every screen that
shows its numbers carries the synthetic banner, which cannot be switched off while that is true
(`app/pages/7_Settings.py`).

## 1. Model details

| Item | Value | Source |
|---|---|---|
| Version / registry name | `demo-q4-v2` | `runs/models/demo-q4-v2/artifact.json` (`version`) |
| Model | Q4, open-system context-unitary model (family `quantum`) | same file (`model_name`, `family`); definition in `src/bre/models/quantum/q4_open_system.py` |
| Created | 2026-09-29T05:35:18+00:00 | same file (`created_at`) |
| What it computes | the predicted probability of selling, `P(sell) = Tr(P_sell rho')`, for an investor's decision state `rho` after a loss rotation `U_L(L)`, an ordered chain of context unitaries `U_c`, Lindblad dephasing at the investor's rate `gamma_i` after every step, and an optional Lüders collapse on a prior tolerance answer | `src/bre/models/quantum/core.py` (`q_forward_rho`), `q4_open_system.py` module docstring; equations in `reports/TECHNICAL_APPENDIX.md` |
| Inputs | six product covariates (`age_band`, `wealth_band`, `invest_experience_yrs`, `self_reported_risk_tolerance`, `financial_literacy_score`, `education`) as 21 standardized columns (`X_columns`); loss magnitude; up to 2 ordered context tags from the vocabulary `news:recession`, `news:technical`, `social:friend_sells`, `market:recovered_5pct`; question order; prior tolerance answer | `artifact.json` (`X_columns`, `ctx_vocab`, `n_ctx_positions`); `PLAN.md` §3 |
| Parameters | 435 free scalars in all; 110 population-level (the rest are per-investor random effects: the encoder offset `u` and `log_gamma` for each of the 65 training investors) | `artifact.json` (`n_params`, `metrics.n_params_all`, `metrics.n_params_population`, `n_subjects`); count formula in `q2_context_unitary.py` / `q4_open_system.py` docstrings |
| Fit | MAP by Adam, 1500 steps, 3 restarts, learning rate 0.02, best objective 331.987 (restarts: 331.987, 332.645, 333.995), 34.4 s | `artifact.json` (`metrics.steps`, `restarts`, `lr`, `objective`, `objectives_per_restart`, `seconds`) |
| Uncertainty | diagonal Laplace approximation at the MAP over the 110 population parameters, per-investor blocks fixed, 200 draws; curvature floored at 1.0 (24 of the 110 parameters hit the floor, so their draw s.d. equals the cap 1.0; smallest s.d. 0.00074) | `artifact.json` (`metrics.laplace`); interval rule in `src/bre/predict.py` (`INTERVAL_SOURCE`: 80% = 10th–90th, 95% = 2.5th–97.5th percentile of `P(sell)` over the draws, conditional on the fitted per-subject effects and on the model) |
| Software | Python 3.11, JAX + optax, CPU only | `CLAUDE.md` environment notes; `PLAN.md` §0 |
| Owner / contact | repository owner (`dennismboya/rocket-simulation-phy234`, branch `claude/behavioral-risk-engine-5e2mf6`) | `PLAN.md` §0 |

## 2. Intended use

* **Primary use.** Advisor-facing decision support inside the BRE dashboard (`app/`): per-investor
  predicted probability of selling under an editable loss scenario and context sequence, a book
  triage table, a decision-state profile, a ranked list of interventions, and a transparency page
  (`app/Home.py`, `app/pages/1_Book_triage.py` … `6_Transparency.py`). All numbers reach the
  screen only through the FastAPI service and `src/bre/predict.py` (`predict.py` module docstring).
* **Wording that must be used** (`src/bre/predict.py`, `PROBABILITY_WORDING` and
  `INTERVENTION_LABEL`): every probability is the *"predicted probability of selling"* under the
  served model, never "will sell"; every intervention effect is labelled *"predicted effect, not
  causally validated"* (the intervention-to-context mapping is a modeling assumption recorded in
  `db/seed.py`, not a measured effect).
* **Out of scope.** Financial advice to end clients; any claim about a real investor's future
  trade; any causal claim (CLAUDE.md rule 6). The model is a research artifact whose defensibility
  rests on the Phase 2 recovery study (§7) and, once it completes, the Phase 4 real-data
  evaluation (§5).
* **Today's status.** Demo mode: the model was fitted on a labelled synthetic book and has no
  held-out evaluation (`artifact.json`, `metrics.held_out: null`, `held_out_note`: "demo fit on the
  whole synthetic book; no held-out evaluation (Phase 4 evaluates real data)").

## 3. Factors

* **Investor covariates** (the six product keys above; `PLAN.md` §3). Bands and ranges are design
  choices; the population marginals in `src/bre/design.py` are placeholders until a public-data fit
  replaces them (`PLAN.md` §3). Gender is deliberately not collected (`reports/BENCHMARK_FEATURES.md`).
* **Scenario factors**: loss level on the design grid {−0.05, −0.10, −0.15, −0.20, −0.30}
  (`PLAN.md` §3), context tags singly or as an ordered pair (17 conditions), question order
  (`tolerance-first` / `scenario-first`) and the prior tolerance answer.
* **Per-context calibration** (`artifact.json`, `calibrated_contexts`; a context counts as
  calibrated with at least 30 training sell rows): `news:recession` 431 rows,
  `news:technical` 413, `social:friend_sells` 428, `market:recovered_5pct` 418; the no-context
  condition has 65 rows (`metrics.n_responses_no_context`). All of these rows are synthetic.
* **Subgroup evaluation**: none has been run. No real-data metric exists for any covariate band.

## 4. Metrics (definitions; `src/bre/eval.py`)

Held-out negative log-likelihood per response (primary; `nll`), Brier score (`brier_score`),
expected calibration error with 10 equal-width bins and its reliability table (`ece`,
`reliability_table`), AUC (Mann–Whitney, ties counted half; `auc`), per-subject predictive gain
against a reference model (`predictive_gain_per_subject`), parameter counts (all scalars and
population-only), and subject-level bootstrap 95% intervals with 1000 resamples
(`subject_bootstrap`, `paired_nll_difference`). For B2, WAIC / PSIS-LOO from ArviZ
(`src/bre/models/classical/b2_hier.py`). The likelihood convention is the weighted Bernoulli /
binomial log-likelihood per row (`src/bre/models/INTERFACE.md` §2).

**Training metrics of the served model (synthetic data, no held-out set)** — `artifact.json`,
`metrics`: training NLL per response 0.5591 (`train_nll_per_response` = 0.55908…), training Brier
0.1932 (`train_brier` = 0.19321…), 1040 sell responses from 65 investors (`n_responses`,
`n_subjects`).

**Loss-response diagnostic of the served model** (`artifact.json`, `metrics.loss_response`;
definition `LOSS_RESPONSE_DEFINITION` in `src/bre/predict.py`): the share of the book's clients
whose predicted `P(sell | L, no context, scenario-first)` is non-decreasing over the five design
losses is 0.538 (35 of 65 clients, `monotone_share`, `n_monotone`, `n_clients`); the mean
predicted `P(sell)` per loss level is 0.337, 0.345, 0.361, 0.382, 0.438 at losses −0.05, −0.10,
−0.15, −0.20, −0.30 (`mean_p_sell_by_loss`). This is a diagnostic of the served model, not a test
of any data (see §9, loss-response caveat).

## 5. Real-data evaluation (Phase 4) — NOT COMPLETED

The pre-registered evaluation of `PLAN.md` §6 on real data has **not** completed. State as of
this card:

* `STATUS.md` (phase table): Phase 4 "in progress".
* `runs/phase4/cpc18-pairs/phase4.log` (2026-09-29T13:48:46Z): the full CPC18 run
  (`experiments/p4-cpc18-full.yaml`, `name: cpc18-pairs`, all 686 subjects, 10 models, splits a/b/d;
  split (c) recorded as not applicable because no real dataset in hand manipulates question
  order) is estimated at 2429 min with 4 workers — over the two-hour GATE of CLAUDE.md rule 4 —
  and no fit line follows the estimate in that log. `reports/phase4/cpc18-pairs/` is empty.
* `reports/phase4/README.md` (runtime budget): the whole ten-model run is of the order of 60–80 h
  sequential, 15–20 h with four workers; it must be split into GATE-sized invocations.

**This section is to be filled, verbatim and by code, from** `reports/phase4/cpc18-pairs/verdict.json`
(fields `verdict`, `best_quantum`, `best_classical`, `nll_diff`, `ci95`, `interference_ci`,
`n_params_q`, `n_params_c`) **and** `reports/phase4/cpc18-pairs/table.md` (per-split model tables with
parameter counts), with `structural.md` and `verdict.md` beside them; the secondary dataset writes
the same files under `reports/phase4/psych201-spektor2024/` (splits (a) and (d) only; the
positive verdict is not reachable there, `experiments/p4-psych201-spektor2024.yaml`). Until those
files exist the dashboard's transparency page shows "Decision rule not yet run: no real-data
numbers are shown" and no Phase 4 table (`app/pages/6_Transparency.py`). No real-data metric is
quoted here for the same reason.

**Decision rule** (verbatim, `PLAN.md` §6; the same text is `DECISION_RULE_TEXT` in
`src/bre/eval.py` and is printed on the transparency page):

> "Q-model supported" only if, on real data, the best Q-model beats the best classical baseline on
> split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower parameter
> count, and the interference-term CI excludes zero. Otherwise the conclusion is "no evidence of a
> quantum-probability advantage in the available data", and the report names the data that would
> resolve it. The dashboard serves whichever model wins on held-out NLL; if that is a classical
> model, the screen says so.

(The phrase "quantum-probability advantage" is the owner's verbatim wording and is kept only inside
this quotation, `PLAN.md` §9.) Implementation notes from `src/bre/eval.py::decision_rule`: the
NLL comparison is a paired subject-level bootstrap; "matched or lower" means the Q-model's count is
less than or equal to the classical count; a missing or NaN interference interval does **not**
exclude zero; on synthetic data the positive verdict is never issued.

**What the real data in hand can and cannot test.** The pre-registered split (b) mapping uses
CPC18 (`PLAN.md` §6): response = retreat to the safer option, context = sign of the experienced
outcome on the preceding one or two trials. It is an analog of panic selling, not panic selling,
and it has no tolerance question, so the `delta_LTP` interval on that table is a model-internal
counterfactual (`reports/phase4/README.md`). Real within-subject data with ordered context pairs
and both question orders on the shared design exist only through the intake battery
(`instrument/`), which has collected no sessions yet (`data/processed/README.md`: `intake_battery`
0 rows).

## 6. Evaluation data

* No held-out evaluation of `demo-q4-v2` exists (`artifact.json`, `metrics.held_out: null`).
* Real data available for Phase 4 (`data/processed/README.md`, generated 2026-09-17, all
  `is_synthetic == False`): CPC18 individual choices, 510,750 rows, 686 subjects (catalog B);
  Psych-201 `spektor2024lossaversion`, 73,250 rows, 282 participants; `spektor2019contexteffects`,
  35,480 rows, 130; `olschewski2024skewness`, 55,066 rows, 1,300 (catalog C); choices13k
  aggregate, 12,225 rows after dropping 2,343 first-order-dominated rows (catalog A); CPC15
  aggregate, 750 rows; question-order tables, 24 aggregate cells (catalog D, one verified pair).
  Provenance, licences and known problems: `data/CATALOG.md`; research-question mapping:
  `data/DATA_MAP.md`.
* Real-data result already recorded (negative): the A→B transfer, `reports/transfer/A_to_B.md`.
  Population parts of B1, B3 and Q2 fitted on choices13k and scored without refitting on held-out
  CPC15 and CPC18 problems: on CPC18 held-out problems the transfer NLL per respondent is 0.6935
  (B1), 0.6954 (B3), 0.6947 (Q2) against the constant rate 0.6932; on CPC15 held-out problems
  0.6939, 0.6928, 0.6881 [95% CI 0.6836, 0.6926] against 0.6974. Scored on every row of the
  targets, the correlations of predicted with observed rates are 0.054 / −0.030 / 0.053 (CPC15)
  and 0.014 / −0.066 / 0.005 (CPC18) for B1 / B3 / Q2. `STATUS.md` (2026-09-29 14:06 UTC) records
  the decision that no public-data population parameter enters demo mode, which therefore keeps
  its labelled synthetic defaults.
* Q1 on the one verified question-order table (`reports/q1_qq_equality.md`, Clinton/Gore): observed
  QQ statistic q = −0.0032; no sample sizes in the source, so no interval or test. Implementation
  check only, not a replication.

## 7. Training data

* `artifact.json`, `training_data_refs` and `notes`: the synthetic demo book built by
  `bre.demo.build_demo_book` (seed 0): 60 G_Q investors plus 5 hand-written archetypes
  (`src/bre/demo.py`, `ARCHETYPES`: design choices, not fitted values), 16 items each, 1,040 sell
  responses; stored in `data/synthetic/demo.db`; generated from `bre.sim.gq.POPULATION_DEFAULTS`
  (`src/bre/sim/gq.py`), which are synthetic defaults, not fitted to data (`STATUS.md`
  2026-09-29 05:25 UTC records the change of `theta_L` to (0, −1.5, 0) and `b` to
  (1, 0.45, 0, 0.15) so that the synthetic loss response rises with the loss).
* Every row carries `is_synthetic = True`; the real-versus-synthetic location rule is enforced on
  every parquet write and DB insert (`PLAN.md` §2 amendment; `STATUS.md` Phase 0 summary).
* **Synthetic recovery study (what makes the numbers defensible; `PLAN.md` §5)** — as of
  `reports/recovery/summary.md` and `reports/recovery/N_target.json` (generated
  2026-09-29T13:43:13+00:00): grid N ∈ {50, 100, 200} × generators {G_Q, G_C, G_F} × seeds {0, 1, 2}
  × models {B1, B2, B4, Q2, Q4}; correct family-level selection by held-out NLL in 7 of 9 cells at
  N = 50, 9 of 9 at N = 100 and 9 of 9 at N = 200; **N_target = 100 subjects → 17,000 responses
  needed for calibration** (100 × 170 items; rule and threshold 0.9 in `N_target.json`). Q4
  recovers the G_Q population context parameters after gauge alignment at Pearson r 0.903, 0.976,
  0.970 (seeds 0–2, N = 100) and 0.943, 0.986, 0.887 (N = 200), and the per-subject decoherence
  ranks at Spearman 0.827, 0.832, 0.798 (N = 100); Q2 alone recovers them poorly on the same
  mixed-decoherence data (r 0.594, 0.401, 0.376 at N = 200). B2 recovers the G_C context means at
  r 1.000 and subject intercepts at Spearman 0.987, 0.984, 0.987 (N = 200).
  **Caveat, to be read with the numbers above:** `summary.md` records 135 fits of which 56 failed
  with `BrokenProcessPool` (every B1 and B4 cell at every N, and the G_F N = 50 seed 1 and 2 B2
  cells), and `STATUS.md` (2026-09-29 13:56 UTC) records that the results file held only the 79
  fits of the resumed run, that 14 cells were missing, that the N = 50 row is provisional (the two
  "wrong" N = 50 cells are G_F seeds where B2 is missing), and that the same grid was re-invoked to
  fill the missing cells from cache. The N target above is therefore the value **as of those
  files** and may change when the completion run finishes. An earlier pass with the old generator
  defaults (`reports/recovery_pass1_olddefaults/summary.md`, N = 200 only, 45 fits, 9 of 9
  correct) is superseded and is not a source for any number here.

## 8. Ethical considerations

* **Personal data.** The `clients` table accepts only the six product covariate keys, "so nothing
  identifying can be stored outside `display_label`" (`db/models.py`, `Client` docstring);
  `display_label` is "the only identifying text accepted" and is capped at 80 characters
  (`api/schemas.py`). Dataset loaders must keep PII out of `x_*` covariates (`src/bre/schema.py`).
  Deleting a client cascades to its responses, prediction log and intervention log with an audit
  entry (`db/models.py`).
* **Production note.** The API's admin switch (`BRE_ADMIN=1` or the `X-BRE-Admin: 1` header) is
  "a local convenience, not authentication" (`api/README.md`); the service binds to 127.0.0.1
  (`api/README.md`). A firm deploying this must place it behind its own access controls,
  authentication and audit policy; the repository provides none. Every prediction is written to
  `predictions_log` and `audit_log` (`STATUS.md`, Phase 5 summary).
* **No causal claims, no advice.** Interventions are ranked by predicted change in `P(sell)` and
  carry the label "predicted effect, not causally validated" (`src/bre/predict.py`,
  `rank_interventions`). Nothing in the model is a recommendation to trade.
* **Benchmark definitions are not equated.** The published panic-sale definition of Elkind et al.
  (a 90% decline of a household's equity assets within a month, half or more from trades) is a
  brokerage-account outcome; the dashboard's "sell" is a stated response and the two must not be
  equated (`reports/BENCHMARK_FEATURES.md`).
* **Participant burden and consent** for any future data collection are fixed in `PLAN.md` §9,
  `instrument/IRB_CHECKLIST.md` and `instrument/PROLIFIC.md` §11 (hashed IDs, no free text, opt-in
  `consent_training`); no collection has taken place.

## 9. Caveats and recommendations

1. **Synthetic training.** Every number the served model produces is a synthetic-training result
   (`artifact.json`, `notes`). Do not present any of them as a property of real investors.
2. **Loss-response caveat (rotation model, first-crossing capacity).** The loss enters the model
   as a rotation of the state, `U_L(L) = exp(L θ_L · G)` (`src/bre/models/quantum/core.py`,
   `loss_unitary`), so `P(sell | L)` need not be monotone in `L`. The behavioral drawdown capacity
   therefore uses a first-crossing (prefix) rule: the largest grid loss such that
   `P(sell | L', typical crisis) <= 0.25` for every grid loss `L' <= L`, not the largest `L` with
   `P(L) <= 0.25` (`src/bre/predict.py`, `drawdown_capacity`, `CAPACITY_DEFINITION`,
   `DRAWDOWN_TARGET`). In the served model 35 of 65 demo clients (0.538) have a non-decreasing
   loss response (`artifact.json`); the transparency page shows this share.
3. **Identifiability.** Context parameters are identified only up to a four-element gauge group and
   modulo π along their direction; a σ_z component of the last unitary before the sell measurement
   is invisible in scenario-first rows; Q4's `gamma_i` is not identified when chains are diagonal in
   the sell basis (`src/bre/models/INTERFACE.md` §7; `q2_context_unitary.py`, `GAUGE_ELEMENTS`,
   `fold_theta`). The recovery study compares parameters only after gauge alignment.
4. **Intervals are conditional on the model** and on the fitted per-subject effects
   (`src/bre/predict.py`, `INTERVAL_SOURCE`); they do not include model uncertainty.
5. **New clients** are scored with the random effect `u = 0` and the population median dephasing
   rate `exp(gamma_mu)` until intake answers refine them (`q2_context_unitary.py`,
   `predict_new_subject`; `q4_open_system.py`, `Q4.predict_new_subject`; `src/bre/predict.py`,
   `refine_subject`: 300 Adam steps on the subject's effects only).
6. **Triage thresholds** (`elevated` at `p >= 0.25`, `high` at `p >= 0.50`) and the typical-crisis
   scenario (recession news, then a friend who sells, at a 20% loss) are design constants
   (`src/bre/predict.py`, `STATUS_THRESHOLDS`, `TYPICAL_CRISIS_CONTEXTS`, `RISK_SCORE_LOSS`), not
   fitted or validated cut-offs.
7. **Negative result on record.** No model transferred from choices13k carries information to the
   CPC datasets by the pre-stated rule (§6; `reports/transfer/A_to_B.md`; `STATUS.md` 14:06 UTC).
8. **Recommendation.** Do not promote any model as "supported" until
   `reports/phase4/cpc18-pairs/verdict.json` exists and the verdict is copied here unchanged; if the
   verdict is negative, this card and the transparency page must say so (CLAUDE.md rule 6). The
   data that would resolve the question and their cost are listed in `reports/DATA_GAPS.md`.
