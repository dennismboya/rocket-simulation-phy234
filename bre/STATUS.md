# STATUS.md — resume here

Last updated: 2026-09-17 (session 1). Resume protocol: read this file and PLAN.md, continue from the
first unfinished item in "Next up".

## Resume essentials (read first)

* Virtualenv: `/home/user/bre-venv` (outside the repo). Run make targets as
  `BRE_VENV=/home/user/bre-venv make -C bre <target>`; a clean clone uses `make setup` to build `bre/.venv`.
* Node for the instrument tests: `/opt/node22/bin/node` (`NODE=/opt/node22/bin/node make -C bre instrument`).
* GitHub: `git push` and the GitHub API both return 403 for this session (Claude GitHub App not
  installed on dennismboya/rocket-simulation-phy234). Commits are local on branch
  `claude/behavioral-risk-engine-5e2mf6`; retry `git push -u origin claude/behavioral-risk-engine-5e2mf6`
  at every checkpoint; a git bundle of the branch is sent to the owner as the fallback deliverable.
* Tests: `/home/user/bre-venv/bin/python -m pytest -q` from `bre/` (pyproject sets pythonpath).

## Phase status

| Phase | State | Notes |
|---|---|---|
| 0 Plan + scaffold | done | schema + validators, shared design, SQLAlchemy layer, packaging, 217 tests green; review findings fixed (df29e0a) |
| 1 Data | done (public part) | 8 processed tables under data/processed (README lists counts); owner downloads for E–J still pending |
| 2 Recovery | first pass done | N=200 × 3 seeds × 3 generators × 5 models: 9/9 correct family selection (reports/recovery/); extended grid N∈{50,100,200,400} running in background (runs/recover/grid_n50_400.*) |
| 3 Models | nearly done | core, Q1, B1, B2, B4, Q2, Q4, G_Q/G_C/G_F all present with tests (554+ passing); B3, B5, B6, Q3, Q5 still to write |
| 4 Evaluation | not started | protocol pre-registered in PLAN.md §6 |
| 5 API | in progress | predict facade, demo book and endpoints exist (3f62e10); 8 contract tests being fixed (agent) |
| 6 Dashboard | in progress | pages 1–3 (market bar, book triage, client profile) being built against the API (agent) |
| 7 Instrument | nearly done | battery browser-verified (0460ab2); document consistency fixes from the honesty review landing (agent) |
| 8 Report | not started | |

## Data in hand (raw copies under scratchpad, to be catalogued then copied into data/raw)

* A choices13k: official repo jcpeterson/choices13k (aggregate rates for 13,006 problems; no license
  file in repo). Also the Thomas et al. 2024 mirror with engineered features.
* B CPC15 aggregate (RothkopfLab/DatasetBias data/cpc15.csv, 750 rows); CPC18 calibration-set raw
  individual data (naecker-lab/cpc18 data/raw-data.csv, 510,750 rows, SubjID/Gender/Age/Order/Trial/RT)
  and EstSet210 aggregates.
* D Question-order tables: Clinton/Gore, Black/White, Rose/Jackson (8 joint proportions each) from
  k-kyoko/qinst_ozawa_khrennikov, verified against the bundled Ozawa & Khrennikov 2021 PDF.
* C Psych-201 (Apache-2.0, GitHub): four folders with real trial data — spektor2024lossaversion,
  spektor2019contexteffects, olschewski2024skewness, thoma2025riskychoice. Psych-101 itself (HF) blocked.

## Blocked / gated

* Egress-blocked from this session: Psych-101 (HF), Wang 2014 SI (PNAS), FINRA NFCS, HRS/SCF/PSID,
  GPS (briq), Robintrack, Frey 2017 and Hussain 2024 raw data (OSF), Zenodo originals, market data feeds.
* GATE (registration): UAS (E), LISS/DNB (H). Instructions to be written; not registered.

## Phase 0 summary (five lines)

1. Schema, validators, shared 170-item design and DB layer exist and agree on one contract (PLAN.md §2 amendment).
2. Real-vs-synthetic rule is enforced on parquet writes and on every DB insert path; PII beyond display_label is refused.
3. Datasets A, B (mirror), C (three experiments), D (one verified table) and market series are catalogued and copied; E/H gated; others need owner downloads.
4. The intake battery runs as a static page, verified in Chromium, exporting schema rows; the Prolific design is shelved and costed.
5. Open: GitHub push blocked (App not installed); Wang 2014 SI and Psych-101 not obtainable from this session.

## Phase 1 summary (five lines)

1. Loaders for A (choices13k, 12,225 aggregate rows after first-order-dominance filtering), B (CPC15 aggregate; CPC18 510,750 individual trials, 686 subjects), C (three Psych-201 experiments, 163,796 trials), D (24 aggregate cells) and the intake battery write validated schema tables; `make data` regenerates them and data/processed/README.md.
2. The CPC18 safer-option coding reproduces the organisers' published block rates to 1e-4; the Psych-201 transcript parser reproduces every transcript line.
3. Market module gives VIX buckets (labelled design choice), monthly drawdown episodes, and a yfinance-with-fallback market state.
4. Loader defaults decided: drop first-order dominated choices13k problems only; CPC18 loss normalised by the largest absolute outcome.
5. Still missing: Wang 2014 SI, Psych-101, FINRA, UAS/LISS (gated), GPS, Robintrack — see data/REGISTRATION.md.

## Phase 2 summary (five lines, first pass at N=200)

1. Model selection by held-out NLL picks the true generator's family in 9 of 9 (generator × seed) cells: Q4 on G_Q data, B2 on G_C and on G_F.
2. Q4 recovers the population context parameters at r = 0.92–0.99 after gauge alignment and per-subject decoherence ranks at Spearman 0.78–0.83; Q2 alone recovers them poorly on mixed-decoherence data (r ≈ 0.5), as expected.
3. B2 recovers G_C context means at r = 1.00 and subject intercepts at Spearman 0.98; B1 attenuates (marginal vs conditional coefficients).
4. N_target = 200 is an upper bound so far (only N tested); the extended grid N ∈ {50, 100, 200, 400} is running to find the smallest N; "responses needed for calibration" = N_target × 170 on the shared design.
5. Runtime: 45 fits in 16 min on 4 cores; Q4 is the slowest (2–4 min per fit at N=200).

## Decisions

* Project lives in bre/; JAX stack (see PLAN.md §0).
* Interference-term definitions fixed in PLAN.md §4.

## Next up

1. Generators G_Q/G_C/G_F and models Q2, Q4, B1, B2, B4 (agents running); then fit.py, eval.py, recover.py and the N=200 recovery study.
2. API (Phase 5) and dashboard pages 1–3 on the synthetic demo book.
3. Real-data fits on A/B (B1, B3, Q2), transfer A→B; dashboard pages 4–7; Phase 4; report.

## Estimated remaining effort

Phases 0–8 as planned; no long runs started yet.

## Model recovery checks recorded so far (synthetic data, N=200, single seed; the formal study is Phase 2)

* Q2 on G_Q (gamma = 0): population theta_c correlation 0.995 over 12 components; Bloch polar angle r = 0.87.
* Q4 on G_Q (mixed): log gamma_i Spearman 0.89; theta_c r = 0.986; MAP shrinks the gamma scale (ranks recovered, not values).
* B1 on G_C: context main effects r = 0.98 (attenuated: marginal vs subject-conditional coefficients).
* B2 SVI on G_C (N=100): subject intercept Spearman 0.99; population context means r = 1.00.
* B4 on G_C: lambda ordering Spearman 1.00; magnitudes not separately identified from the temperature.
* Exact gauge group of Q2/Q4 (conjugation, sigma_z conjugation, theta + pi·theta/|theta|) handled by align_gauge/fold_theta before any parameter comparison.

## Progress log (long runs)

* 2026-09-17 20:25 UTC — recovery study (N=200) and API build started in background agents; check runs/recover/*.log.
* 2026-09-17 20:34 UTC — check-in: recovery runner still being written; no run log yet
* 2026-09-29 04:55 UTC — extended recovery grid launched (N 50/100/200/400, seeds 0–2, 3 workers, capped at 110 min); API fixes and dashboard pages 1–3 in agents.
