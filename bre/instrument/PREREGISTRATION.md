# PREREGISTRATION.md — pre-registered hypotheses, analyses and stopping rule

Status: NOT YET REGISTERED at any public registry. The evaluation protocol it relies on (PLAN.md §6)
was committed to this repository in commit `a51f100e733de9a44665b64763bd78338e50cf32` on 2026-09-17,
before any real-data fit. This document is to be registered by the owner (e.g. at the Open Science
Framework, which is unreachable from the build session) before the pilot of `instrument/PROLIFIC.md`
runs; the registry URL and timestamp are then recorded here: registry ____ , URL ____ , date ____ .
Written 2026-09-17.

It covers two analyses: (A) the evaluation of Q-models against classical baselines on the public
data already catalogued (data/CATALOG.md A–D), which follows PLAN.md §6 unchanged, and (B) the
Prolific study of `PROLIFIC.md`, should it ever run. Where (B) differs from (A) the difference is
stated. Tokens of the form `{{TOKEN}}` are quantities not yet known (listed in §12).

---

## 1. Research questions (PLAN.md §1, unchanged)

* RQ1 Context dependence: do responses to a loss scenario change with context (cause frame, social
  cue, partial recovery, question order) in ways that violate the law of total probability, order
  invariance, or the QQ equality where it should hold?
* RQ2 Predictive value: do Q-models predict held-out responses better than matched classical
  baselines, for held-out subjects and for context combinations never seen in training?
* RQ3 External validity: do fitted per-investor parameters predict later real financial behavior in
  any dataset with both stated preferences and later actions?

## 2. Hypotheses

Notation from PLAN.md §4: δ_LTP = P(sell | scenario-first) − Σ_a P(a, sell | tolerance-first);
Δ_order(c1, c2) = P(sell | c1, c2) − P(sell | c2, c1); q = the QQ-equality statistic
p(A_yes, B_no) + p(A_no, B_yes) − p(B_yes, A_no) − p(B_no, A_yes) for the ordered pair of questions
(A = tolerance, B = sell). All tests are two-sided. "CI" means a 95% subject-level bootstrap
confidence interval with 1000 draws (PLAN.md §6) unless stated otherwise.

| Hypothesis | RQ | Statement | Q-model prediction | Classical prediction | Confirmed if |
|---|---|---|---|---|---|
| H1 Law-of-total-probability violation by question order | RQ1 | The probability of selling depends on whether the tolerance question was asked before the scenario. | δ_LTP ≠ 0 | δ_LTP = 0 | CI of the pooled δ_LTP (over loss levels and context conditions) excludes 0 |
| H2a Context order effects | RQ1 | For at least one unordered pair of contexts, the sell probability depends on the order in which the two contexts were presented. | Δ_order ≠ 0 for some pair (non-commuting unitaries) | Δ_order = 0 (additive or Bayesian context effects; B4 predicts order dependence only through the recency discount δ) | at least one of the 6 unordered pairs has a Holm-adjusted CI excluding 0 |
| H2b QQ equality | RQ1 | Where an order effect exists between the tolerance question and the sell question, the QQ equality holds. | q = 0 (Q1) | no constraint; q = 0 only when there is no order effect at all | given H1 confirmed, the CI of q includes 0 and its half-width is smaller than the absolute pooled δ_LTP; if H1 is not confirmed, H2b is not testable and is reported as such |
| H3 Predictive advantage of Q-models | RQ2 | The best Q-model predicts held-out responses on split (b) better than the best classical baseline at matched or lower parameter count. | held-out NLL lower | held-out NLL equal or higher | exactly the decision rule of §7 |
| H4 External validity of per-investor parameters | RQ3 | Per-investor parameters fitted at wave 1 predict (a) that investor's later decisions (wave 2, event wave) and (b) their self-reported real buy/sell actions. | (a) NLL on split (d) lower than the covariates-only model; (b) AUC of the wave-1 sell propensity for "sold" > 0.5 and > the covariates-only AUC | same functional claims for the winning classical model; H4 is about any per-investor model, not Q vs classical | (a) CI of the NLL difference excludes 0; (b) CI of the AUC difference excludes 0. Both (a) and (b) are required for "confirmed"; (b) alone is reported as "self-report only" |

H1, H2a and H2b are structural tests (PLAN.md §6 "Structural tests"); H3 is the single primary
confirmatory test of the project; H4 requires wave-2 or event-wave data and is otherwise "not
testable with the data in hand", which is a permitted conclusion.

For analysis (A) on public data, H1 and H2b are testable only on dataset D (question-order tables;
one verified table), H2a on the CPC18 experienced-outcome analog (PLAN.md §6 mapping), and H4 is not
testable at all unless the gated datasets E or H are obtained (data/REGISTRATION.md).

## 3. Design and variables (analysis B; see `PROLIFIC.md` for the full design)

Independent variables: `loss_pct` (5 levels), context condition (17: none, 4 singles, 12 ordered
pairs; each participant sees 4 none + 4 singles + 4 ordered pairs), `question_order_id` (2, assigned
per item within subject, every loss level with both orders), wave (w1, w2, event), repeat (original
vs repeat; 2 repeats per session). The timed delay page is fixed at 10 s on news items in
`battery.json`; a 0 s / 10 s factor exists only if override O8 of `PROLIFIC.md` §3 is enabled.
Dependent variables: binary sell (`elicitation_type = binary_sell`, primary) and share sold
(`allocation_pct`, secondary; the slider is the share sold, so 1 − response is the share kept).
The tolerance answer (yes/no, asked with every item) is the "A" question of the order tests.
Covariates: PLAN.md §3 bands. Outcome behavior: self-reported buy/sell since wave 1
(`outcome_behavior.action`, `lag_days`).

Unit of analysis: the decision row; subject-level clustering is respected in every resampling step.

## 4. Splits (PLAN.md §6, applied to each dataset)

* (a) held-out subjects: 20% of subjects, stratified by dataset (for analysis B: by wave-1
  completion month), fixed by seed 20260917 recorded in `experiments/splits.json`.
* (b) held-out context compositions: train on single-context conditions and `none`, test on the
  ordered pairs. Classical models use additive context effects on this split; B4 (Bayesian belief
  updater) is the natural rival. For analysis A the pre-registered CPC18 mapping of PLAN.md §6 is
  used verbatim (singles = one preceding trial, ordered pairs = two preceding trials); for analysis B
  the split is by design (`PROLIFIC.md` §4).
* (c) held-out question order: train on tolerance-first items, test on scenario-first items and vice
  versa (within subject in analysis B, since order is assigned per item).
* (d) temporal: train on wave 1, test on wave 2 and on the event wave (analysis B); for analysis A,
  where a dataset has sessions or blocks, train on earlier and test on later.

Every split is materialised once, written to `experiments/splits.json` with the seed, and never
regenerated after the first real-data fit.

## 5. Models compared and equal-footing rules (PLAN.md §4, unchanged)

Classical: B1 logistic with pairwise interactions; B2 hierarchical Bayesian logistic (NumPyro);
B3 cumulative prospect theory with context-shifted reference point; B4 Bayesian belief updater;
B5 HistGradientBoosting and a small MLP; B6 two-state HMM. Quantum-probability: Q1 order-effect
projector model; Q2 context-unitary model; Q3 quantum dynamical model; Q4 open-system (Lindblad)
variant of Q2; Q5 quantum decision theory. Shared feature map and encoder, Bernoulli likelihood for
sell/hold, Beta for the slider; ≥ 5 random restarts, Adam with early stopping on a validation fold,
L-BFGS polish; parameter counts from `count_params`. "Best Q-model" and "best classical baseline"
are chosen by validation-fold NLL within the training data, never by test-split performance.

## 6. Metrics (PLAN.md §6, unchanged)

Held-out NLL per response (primary); Brier; ECE with reliability plots; AUC; WAIC/PSIS-LOO for
Bayesian fits; parameter count; per-subject predictive-gain distribution; 1000-draw subject-level
bootstrap CIs on all of them.

## 7. Decision rule (copied verbatim from PLAN.md §6)

Decision rule (verbatim, printed on the transparency page):
"Q-model supported" only if, on real data, the best Q-model beats the best classical baseline on
split (b) by held-out NLL with a bootstrap 95% CI excluding zero, at matched or lower parameter
count, and the interference-term CI excludes zero. Otherwise the conclusion is "no evidence of a
quantum-probability advantage in the available data", and the report names the data that would
resolve it. The dashboard serves whichever model wins on held-out NLL; if that is a classical model,
the screen says so.

"The interference-term CI" refers to δ_LTP as defined in PLAN.md §4 (the primary interference term);
δ_mix and Δ_order are reported but do not enter the rule. "Matched or lower parameter count" is read
as count(best Q) ≤ count(best classical) by `count_params`.

## 8. Planned analyses, in the order they will be run

1. Data validation: `validate_frame` on every frame; exclusions of `PROLIFIC.md` §10 applied and
   counted; the counts are reported before any model is fitted.
2. Descriptives (no inference): sell rate by loss level, by context condition, by order, by wave;
   slider distribution; response-time distribution; attention and comprehension pass rates.
3. Replicability (analysis B only): agreement of the binary answer between each repeat item and its
   original (rate and Cohen's κ, with CI); mean absolute slider difference; the same between wave 1
   and wave 2 for matched cells. These are descriptive quality metrics, not hypotheses; the per-subject
   repeat inconsistency is kept for the exploratory analysis in §9.
4. Structural tests (H1, H2a, H2b): computed from raw response frequencies, model-free.
   * δ_LTP: for each loss × condition cell, from the per-item order assignment (within subject,
     every loss level seen under both orders by every participant); pooled by inverse-variance
     weighting across cells; CI by subject-level bootstrap.
   * Δ_order: for each of the 6 unordered context pairs, from the two ordered-pair conditions, pooled
     over loss levels; Holm correction over the 6 pairs.
   * q: from the per-item joint (tolerance answer, sell answer): tolerance-first items give the
     A→B order, scenario-first items the B→A order, with no other item intervening because both
     answers belong to the same presentation; pooled over cells; also reported per loss level as
     secondary. Repeat presentations are excluded from the structural tests (originals only).
   * Analysis A: δ_LTP and q on dataset D (verified table only in reported results); Δ_order on the
     CPC18 analog.
5. Model fits on split (b) (H3): all models of §5, then the decision rule of §7. Split (a), (c), (d)
   results and the remaining metrics of §6 are reported alongside as secondary. Interference-term
   CIs from the fitted Q-models are reported next to the model-free estimates of step 4.
6. Structural checks that are model-dependent: Q5 quarter-law check; Q4 decoherence-rate
   distribution γ_i; Q1 QQ-equality exactness on Q1-generated data (unit test, not a result).
7. H4 (analysis B, only if wave 2 or an event wave exists): (a) split (d) NLL of the winning
   per-investor model vs a covariates-only logistic; (b) wave-1 sell propensity, defined as each
   participant's model-implied mean P(sell) over the 85 loss × condition cells under the winning
   model, as a predictor of `outcome_behavior.action ∈ {sold, both}` vs {none, bought}, compared with
   a covariates-only logistic by 10-fold cross-validated AUC; CIs by subject-level bootstrap.
8. Report: every result with its CI, the exclusion counts, the split file hash, the analysis-code
   commit, and the verdict sentence produced by the decision rule.

Inference conventions: two-sided; CI exclusion of the null value is the criterion; no p-values are
reported without the corresponding CI; Holm correction within a hypothesis family where stated;
no correction across families (H1–H4 are reported separately, and only H3 is the primary test).

## 9. Exploratory analyses (labelled as such in the report; never used for the verdict)

* Relation between the per-subject repeat inconsistency (step 3) and the Q4 decoherence rate γ_i.
* Effect of the timed delay (0 vs 10 s) on sell rate and on Q3's fitted time constant, only if
  override O8 of `PROLIFIC.md` §3 is enabled; otherwise the delay is constant and the analysis is
  not run.
* Heterogeneity of δ_LTP and Δ_order by covariate band.
* Slider (allocation) versions of H1 and H2a.
* Event-wave responses vs the same participant's wave-1 responses at the matching loss level, with
  the live drawdown as context.

## 10. Exclusions, missing data, outliers

* Exclusions: exactly `PROLIFIC.md` §10 for analysis B; for analysis A, the per-dataset filters in
  data/CATALOG.md (e.g. the stochastic-dominance filter for choices13k). All counts reported.
* Missing data: a decision row is used only if `response` is present; a missing slider does not
  remove the binary row; missing covariate bands ("prefer not to say" → null, `battery.json`
  `intake.covariate_null_rule`) are one-hot encoded with a missing indicator, not imputed.
* Outliers: none removed beyond the pre-registered response-time floor; no trimming of sliders.
* Pilot rows (`session_id = "pilot"`) are excluded from every confirmatory analysis.

## 11. Stopping rule for data collection (analysis B)

* Wave 1 stops when the number of participants passing all exclusions of `PROLIFIC.md` §10 reaches
  `N_target = {{N_TARGET_FROM_PHASE2}}` (from the Phase 2 recovery study: the smallest N with ≥ 90%
  correct model selection, PLAN.md §5), or when the owner-approved budget
  `{{BUDGET_CAP_USD_OWNER_APPROVAL}}` is exhausted, whichever comes first. Recruitment is opened in
  a single batch of `N_recruit = ceil(N_target / (1 − {{EXCLUSION_RATE_FROM_PILOT}}))` places; if,
  after exclusions, fewer than N_target remain, one top-up batch of the shortfall (again divided by
  1 − e) is opened, at most once. If the target is still not reached, the study is reported as
  under-recruited relative to its pre-registered target and every result is labelled accordingly.
* No interim analysis of any hypothesis is performed while a wave is open. The only look at the data
  before a wave closes is the data-quality check of step 1 (exclusion counts and completion rate),
  which cannot alter the target.
* Wave 2 closes for each participant 28 days after their wave-1 completion, whatever the retention;
  no top-up recruitment for wave 2.
* An event wave closes 72 hours after its launch; at most one per 30 days; whether one runs at all
  depends on the trigger of `PROLIFIC.md` §6.3 and on a fresh owner approval.
* The pilot is 20 participants (design decision) and stops at 20 completions regardless of quality;
  its only outputs are T, e, the response-time floor and bugs.
* Analysis A (public data) has no data collection; its "stopping rule" is that the dataset list of
  data/CATALOG.md at the time of the analysis-code freeze is final for the report.

## 12. Analysis-code freeze, blinding, deviations

* Before any Prolific data are downloaded, the analysis code (`src/bre/eval.py`, the split file, the
  model code) is tagged in git; the tag and commit hash are recorded here:
  `{{ANALYSIS_CODE_COMMIT}}`. Fits on real data use that commit; any later change is a deviation.
* The same code and the same seeds are used on the synthetic recovery study (Phase 2), so the
  pipeline has been exercised end to end before it sees real data.
* Model selection within each family uses validation folds inside the training split only; the
  test split of each analysis is evaluated once, after all models are frozen.
* Deviations from this document are recorded in a "Deviations" section appended below, with date
  and reason, before the deviating analysis is run; results affected by a deviation are labelled.

### Deviations

(none)

## 13. Placeholder tokens used in this file

| Token | Where it comes from |
|---|---|
| `{{N_TARGET_FROM_PHASE2}}` | Phase 2 recovery study (PLAN.md §5): smallest N with ≥ 90% correct model selection |
| `{{EXCLUSION_RATE_FROM_PILOT}}` | pilot: fraction of completions excluded (PROLIFIC.md §10) |
| `{{BUDGET_CAP_USD_OWNER_APPROVAL}}` | the owner's written budget approval (nothing is spent without it) |
| `{{ANALYSIS_CODE_COMMIT}}` | git commit hash of the analysis-code freeze, recorded before real-data download |
