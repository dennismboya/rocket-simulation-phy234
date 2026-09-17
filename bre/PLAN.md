# PLAN.md — Behavioral Risk Engine (BRE)

Written 2026-09-17 by the first session, before any code. Owner approval of this plan is pending
(the session runs unattended); per CLAUDE.md rule 1 work proceeds phase by phase and stops only at
GATEs. Amend this file to redirect the next session.

## 0. Ground truth that shapes the plan

* Repository: the BRE lives in `bre/` of `dennismboya/rocket-simulation-phy234`, branch
  `claude/behavioral-risk-engine-5e2mf6`. The pre-existing PHY234 material at the repo root is left
  untouched. A root `Makefile` delegates to `bre/Makefile` so the definition-of-done command works
  from a clean clone.
* Compute: 4 CPU cores, 15 GB RAM, no GPU. Everything is sized to that.
* Network (session egress policy): PyPI, npm and GitHub are reachable; Hugging Face, OSF, Zenodo,
  PNAS/PMC, arXiv, Yahoo Finance, FRED, CBOE, UAS, Centerdata, FINRA and the CDNs are not. Consequences:
  * Datasets are acquired only when a GitHub-hosted copy exists. Everything else gets a CATALOG entry
    with exact download instructions for the owner and a loader written against the documented format
    but marked "untested on real file".
  * The dashboard's "Load current market" button is implemented with yfinance but cannot be exercised
    here; it fails gracefully to manual entry and the acceptance test uses a fixture.
  * jsPsych is vendored from npm rather than a CDN.
* Stack: Python 3.11; JAX + optax for all optimizer-fitted models (Q1–Q5, B1, B3, B4, B6);
  NumPyro (SVI for recovery study, NUTS for final real-data fits) for B2; scikit-learn for B5;
  FastAPI; Streamlit; SQLAlchemy + SQLite; pytest + Streamlit AppTest. Deviation from the prompt's
  torch suggestion is deliberate: one autodiff framework, CPU wheels on PyPI, `jax.scipy.linalg.expm`
  for unitaries, and NumPyro needs JAX anyway.

## 1. Research questions (pre-specified)

* RQ1 Context dependence: do responses to a loss scenario change with context (cause frame, social
  cue, partial recovery, question order) in ways that violate the law of total probability, order
  invariance, or the QQ equality where it should hold?
* RQ2 Predictive value: do Q-models predict held-out responses better than matched classical
  baselines, for held-out subjects and for context combinations never seen in training?
* RQ3 External validity: do fitted per-investor parameters predict later real financial behavior in
  any dataset with both stated preferences and later actions?

## 2. Unified schema (Phase 0 deliverable, `src/bre/schema.py`)

One row per elicited decision, columns exactly as in the build prompt Section 3. Storage is parquet
(pyarrow). `context_tags` and `prior_question_ids` are list<string>; `covariates` and
`outcome_behavior` are JSON strings (null allowed for the latter). `validate_frame(df)` returns a
`ValidationReport` (nulls per column, range violations, duplicate `(subject_id, dataset, session_id,
position_in_session)`, is_synthetic consistency with the file location) and raises on hard errors.
Every loader and every generator calls it. The DB `responses` table has the same columns.

Amendment 2026-09-17 (after the Phase 0 review), binding for every loader and the DB layer:
* `elicitation_type` gains two values: `binary_yes_no` (response in {0,1}; used for the tolerance
  question and any yes/no survey item) and `choice_rate` (aggregate share in [0,1] choosing the
  target option, e.g. choices13k bRate, CPC block rates, published joint proportions). The four
  original values keep their ranges.
* `covariates` keys: the six product keys, plus `weight` (numeric >= 0 or null; the number of
  respondents behind an aggregate row), plus dataset-specific keys that must carry the prefix `x_`
  (any JSON scalar; loaders are responsible for keeping PII out of them). `clients.covariates` in the
  DB accepts only the six product keys.
* `context_tags` are always `namespace:value` (regex-checked); the psych201 domain tags are
  `domain:gain`, `domain:loss`, `domain:mixed`.
* `question_order_id` values for the shared design are the hyphenated `tolerance-first` and
  `scenario-first`; free-form for external datasets.
* The DB layer reuses the schema validators: `frame_to_responses` calls `validate_frame(strict=True)`
  first, the ORM `@validates` hooks mirror the same rules, and the real-vs-synthetic location rule is
  enforced on every insert path (session `before_flush`), using `os.path.abspath` (no symlink
  resolution) and absolute SQLite file paths fixed at engine creation.

## 3. Shared experimental design (`src/bre/design.py`)

* loss_pct ∈ {-0.05, -0.10, -0.15, -0.20, -0.30}; horizon_days = 365 unless stated.
* Context set C = {none, news:recession, news:technical, social:friend_sells, market:recovered_5pct}.
  Conditions: `none`, each of the four non-null contexts singly, and the 12 ordered pairs of distinct
  non-null contexts → 17 context conditions.
* Question order ∈ {tolerance-first, scenario-first}. The tolerance question is a binary
  "Would you describe yourself as someone who avoids investment losses even at the cost of lower
  returns?" measured before (tolerance-first) or after (scenario-first) the scenario.
* Full design = 5 × 17 × 2 = 170 binary sell/hold items per subject. Used in full by the generators
  and the recovery study; the intake battery samples a randomized, balanced subset (12–16 items for
  the full form, 5 for the short form) and records the assignment.
* Covariates: age_band {18-29,30-44,45-59,60+}, wealth_band {<50k,50-250k,250k-1M,>1M},
  invest_experience_yrs (0–40), self_reported_risk_tolerance (1–7), financial_literacy_score (0–5),
  education {hs, some_college, bachelor, graduate}. Marginals in `design.py` are placeholders until a
  public-data fit (FINRA NFCS Investor Survey when the owner downloads it) replaces them; the file
  records which it is.

## 4. Models (Phase 3) and equal-footing rules

Shared feature map `features(row)`: covariate one-hots + z-scored numerics, loss magnitude, context
dummies (per position for pairs), question-order dummy, prior-answer indicator. Shared encoder
architecture `Encoder(d_out)` = linear map from covariates (+ prior answers) to R^{d_out}; Q-models use
it to produce ψ_i (d_out = 2·d real numbers → complex C^d, normalized), B2 uses it to produce the
subject random-effect mean. Bernoulli likelihood for sell/hold, Beta for allocation sliders.
Parameter counts are computed by `count_params(model)` and reported side by side. Every optimizer fit
uses ≥ 5 random restarts, Adam with early stopping on a validation fold, and L-BFGS polish.

Classical:
* B1 logistic: features + pairwise interactions among (loss, contexts, order).
* B2 hierarchical Bayesian logistic (NumPyro): subject intercepts and context slopes; SVI in the
  recovery study, NUTS for final fits; WAIC/PSIS-LOO via ArviZ.
* B3 cumulative prospect theory with context-shifted reference point, logit link; λ, α, γ per
  subject (hierarchical shrinkage).
* B4 Bayesian belief updater: each context is a likelihood ratio on "market recovers within
  horizon"; posterior drives expected utility with fitted CRRA risk aversion; recency discount δ on
  older evidence. Composes contexts without pair parameters → fair rival on split (b).
* B5 HistGradientBoosting + small MLP (sklearn) on the same features.
* B6 two-state HMM over a discrete risk state for sequential answers.

Quantum-probability:
* Q1 order-effect projector model: questions A, B are rank-1 projectors in R^2 at relative angle φ;
  fit ψ, φ; QQ-equality test statistic and its exact satisfaction on Q1-generated data.
* Q2 context-unitary model (product core): ψ_i = normalize(Encoder(x_i)) ∈ C^2; loss scenario
  applies U_L(L) = exp(L·θ_L·G); each context c applies U_c = exp(Σ_k θ_{c,k} G_k) with
  {G_k} = {iσ_x, iσ_y, iσ_z} (su(2) basis), θ_none ≡ 0 as the reference gauge; the tolerance
  question is a projector P_tol at angle φ, applied with Lüders collapse when asked first.
  P(sell | L, c_1..c_m, order) = ||P_sell U_{c_m}…U_{c_1} U_L(L) ψ||² (scenario-first) or the
  Lüders mixture over tolerance answers (tolerance-first). Composition of unitaries gives pair
  predictions with zero pair parameters.
* Q3 quantum dynamical model: H = H_payoff(L) + H_dissonance(c); ψ(t) = exp(−iHt)ψ(0) with t = time
  since news (days, log-scaled); captures recency.
* Q4 open-system variant of Q2: density matrix ρ, Lindblad dephasing at rate γ_i in the sell basis;
  γ_i → ∞ reduces to the classical mixture. γ_i is the per-investor consistency score.
* Q5 quantum decision theory (Yukalov–Sornette): P = f + q, Σq = 0; quarter-law check.

Interference terms (definitions fixed here, logged by every Q-model):
* LTP interference (primary, used by the decision rule): δ_LTP = P(sell | scenario-first) −
  Σ_a P(a, sell | tolerance-first). Classical models predict δ_LTP = 0 by the law of total probability.
* Context-uncertainty interference (dashboard, when the cause frame is a mixture with weights p_c):
  δ_mix = ||P_sell Σ_c √p_c U_c ψ||² − Σ_c p_c ||P_sell U_c ψ||².
* Order effect for a pair: Δ_order(c1,c2) = P(sell | c1,c2) − P(sell | c2,c1).

Pytest for the math: unitarity of every U (‖U†U − I‖ < 1e-6); probabilities sum to one; Q2 with all
θ = 0 equals a static classifier; Q4 with γ = 10^3 equals the mixture model within 1e-4; Q1 satisfies
the QQ equality exactly on Q1-generated data; Lüders collapse is idempotent.

## 5. Phase 2 recovery study (mandatory, `make recover`)

Generators G_Q, G_C, G_F as specified (each emits the schema with is_synthetic=True under
data/synthetic/). G_Q per subject: ψ_i ∈ C^2 (C^4 for the two-question block), θ_c ~ N(μ_c, σ_c²),
decoherence γ_i ~ LogNormal so subjects span quantum to classical. G_C: latent type, additive
log-odds, conditionally independent answers (Markov variant flag). G_F: logistic with all pairwise
context interactions.

Grid: N ∈ {100, 200, 400, 800} × 3 generators × 5 seeds; models fitted: B1, B2, B4, Q2, Q4 (first
pass, per first-actions item 4), then B3, B6, Q3 in the background. Selection by held-out
log-likelihood (subject split 80/20) and WAIC. Outputs: confusion matrix per N, θ_c and γ_i recovery
scatter, smallest N with ≥ 90% correct selection → "responses needed for calibration" on the
transparency page. Failure at every N triggers identifiability fixes before any real-data fit.
Runtime target: the N=200 pass under 10 minutes on 4 cores; the full grid runs in the background
under runs/ with STATUS.md progress lines.

## 6. Phase 4 evaluation protocol (pre-registered before any real-data fit)

Splits: (a) held-out subjects (20%, stratified by dataset); (b) held-out context compositions: train
on single-context conditions, test on ordered pairs; classical models use additive context effects on
this split, B4 is the natural rival; (c) held-out question order (train tolerance-first, test
scenario-first and vice versa); (d) temporal, where a dataset has waves or sessions.

Metrics: held-out NLL per response (primary), Brier, ECE with reliability plots, AUC, WAIC/PSIS-LOO
for Bayesian fits, parameter count, per-subject predictive-gain distribution, 1000-draw subject-level
bootstrap CIs on all of them.

Structural tests: LTP violation; order effects by scenario; QQ equality on dataset D and any
within-subject order data; interference-term CI; quarter-law check; decoherence-rate distribution.

Decision rule (verbatim, printed on the transparency page):
"Q-model supported" only if, on real data, the best Q-model beats the best classical baseline on
split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower parameter
count, and the interference-term CI excludes zero. Otherwise the conclusion is "no evidence of a
quantum-probability advantage in the available data", and the report names the data that would
resolve it. The dashboard serves whichever model wins on held-out NLL; if that is a classical model,
the screen says so.

Real-data mapping for split (b) (fixed now, before fitting): CPC18 raw individual data, feedback
blocks 2–5. Subject = SubjID; scenario = problem; response = 1 if the safer option (lower-variance)
is chosen ("sell" analog = retreat to safety); "context" = the sign of the experienced outcome on
the preceding one or two trials within the same problem (exp:loss / exp:gain); loss_pct = previous
obtained payoff divided by the problem's outcome range (signed). Singles = one preceding trial;
ordered pairs = the two preceding trials. This is the only public within-subject sequence data with
experienced losses; it is an analog, not panic selling, and the report says so.

## 7. Phases, targets and make targets

| Phase | Deliverable | make target |
|---|---|---|
| 0 | CLAUDE.md, PLAN.md, STATUS.md, scaffold, schema + validators, DB models, tests | `setup`, `test` |
| 1 | data/CATALOG.md, loaders, data/processed/*.parquet, data map, missing list | `data` |
| 2 | generators, recovery study, confusion matrices, N target | `sim`, `recover` |
| 3 | B1–B6, Q1–Q5, fit.py, math tests | `fit` |
| 4 | eval.py, splits, metrics, structural tests, verdict | `eval` |
| 5 | FastAPI service, OpenAPI spec, contract tests | `api` |
| 6 | Streamlit dashboard pages 1–7, demo mode, acceptance tests, retrain | `dashboard`, `retrain` |
| 7 | intake battery, static jsPsych build, PROLIFIC.md | `instrument` |
| 8 | REPORT.md, MODEL_CARD.md, summaries | `report` |

Definition of done: `make setup && make sim && make recover && make fit && make eval && make dashboard`
from a clean clone; all tests green; verdict on the transparency page; STATUS.md lists data gaps
and cost to close each.

## 8. Order of work (Section 12 of the build prompt)

1. CLAUDE.md, PLAN.md, STATUS.md. ✔ (this file)
2. Scaffold: requirements.txt, Makefile, schema.py, db models, schema tests.
3. Q1 + QQ-equality test on dataset D (three published tables; the 70-survey SI is blocked, see CATALOG).
4. Generators + recovery study at N=200 with B1, B2, B4, Q2, Q4; confusion matrix.
5. API + dashboard pages 1–3 on the synthetic demo book, banner on.
6. Catalog/load A (choices13k) and B (CPC15/CPC18); fit B1, B3, Q2; transfer A→B; population
   parameters into demo mode with provenance.
7. Registration instructions for E (UAS) and H (LISS/DNB); GATE noted, no registration performed.
8. Dashboard pages 4–7, retrain path, acceptance tests.
9. Static jsPsych instrument and PROLIFIC.md.
10. Phase 4 on real data in hand; REPORT.md, MODEL_CARD.md; STATUS.md data gaps and costs.

## 9. Open decisions recorded

* Individual-level choices13k data are not in the public repo (aggregate rates per problem only);
  the encoder pretraining on A therefore uses problem-level rates with binomial weights.
* CPC18 individual data come from a third-party GitHub mirror of the Zenodo record; the catalog says
  so and asks the owner to confirm against the original record when Zenodo is reachable.
* Dataset D is three published 2×2×2 tables (Clinton/Gore, Black/White, Rose/Jackson) verified
  against the Ozawa & Khrennikov (2021) PDF; enough for the QQ-equality unit test, not for the
  70-survey replication.
