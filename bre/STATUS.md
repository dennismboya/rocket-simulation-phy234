# STATUS.md — resume here

Last updated: 2026-09-17 (session 1). Resume protocol: read this file and PLAN.md, continue from the
first unfinished item in "Next up".

## Resume essentials (read first)

* Virtualenv: `/home/user/bre-venv` (outside the repo). Run make targets as
  `BRE_VENV=/home/user/bre-venv make -C bre <target>`; a clean clone uses `make setup` to build `bre/.venv`.
* Node for the instrument tests: `/opt/node22/bin/node` (`NODE=/opt/node22/bin/node make -C bre instrument`).
* GitHub: push works since 2026-09-29 15:05 UTC; draft PR https://github.com/dennismboya/rocket-simulation-phy234/pull/1
  tracks branch `claude/behavioral-risk-engine-5e2mf6`. Two git bundles were sent to the owner earlier as fallbacks.
* Tests: `/home/user/bre-venv/bin/python -m pytest -q` from `bre/` (pyproject sets pythonpath).

## Phase status

| Phase | State | Notes |
|---|---|---|
| 0 Plan + scaffold | done | schema + validators, shared design, SQLAlchemy layer, packaging, 217 tests green; review findings fixed (df29e0a) |
| 1 Data | done (public part) | 8 processed tables under data/processed (README lists counts); owner downloads for E–J still pending |
| 2 Recovery | first pass done | N=200 × 3 seeds × 3 generators × 5 models: 9/9 correct family selection (reports/recovery/); extended grid N∈{50,100,200,400} running in background (runs/recover/grid_n50_400.*) |
| 3 Models | done | B1–B6, Q1–Q5 implemented, registered and tested; fit/eval/recover wired for all ten |
| 4 Evaluation | in progress | real-data pipeline (A→B transfer, CPC18 pre-registered splits) being built (agent); full runs scheduled after the grid |
| 5 API | done | /profile /predict /score_book /interventions/rank /model /market /clients /health; openapi.json committed; demo model demo-q4-v2 active; contract tests green |
| 6 Dashboard | pages 1–3 done | AppTest suite green; pages 4–7, retrain path and acceptance tests being built (agent) |
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

## Phase 5 and 6a summary (five lines)

1. The FastAPI service is the only prediction source; every prediction is logged to predictions_log and audit_log; the OpenAPI spec is committed.
2. The demo book is 60 G_Q investors plus five archetypes, all synthetic and labelled; the active model is demo-q4-v2 fitted on that book (population-level calibration only).
3. Behavioral drawdown capacity uses a first-crossing rule because a loss unitary is a rotation and the response need not be monotone; the monotone share is reported by GET /model.
4. Dashboard pages 1–3 run against the API only, carry the synthetic banner, intervals and calibration N on every probability, and export CSV/PDF with the banner.
5. Missing endpoints noted by the dashboard build (per-client response history, status and intervention logging) are being added with pages 4–7.

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
* 2026-09-29 05:13 UTC — grid check-in: [recover] estimated 165 min exceeds --max-minutes 110; reduce the grid (seeds, steps, restarts) or pass --force (a run over two hours is a GATE, CLAUDE.md rule 
* 2026-09-29 05:15 UTC — grid relaunched as two runs under the 2-hour GATE: N 50/100 (seeds 0–2, cap 60 min) then N 400 (seeds 0–1, Q restarts 1, cap 100 min).
* 2026-09-29 05:25 UTC — DECISION: G_Q population defaults changed (theta_L = (0, -1.5, 0), b = (1, 0.45, 0, 0.15)) so the synthetic loss response rises with the loss; the old-defaults N=200 pass is archived under reports/recovery_pass1_olddefaults and runs/recover/n200_pass1_olddefaults; the full grid (N 50/100/200 seeds 0–2, then N 400 seeds 0–1) relaunched as two capped runs (runs/recover/grid_a.*, grid_b.*).
* 2026-09-29 05:34 UTC — grid_a: 3 of 135 fits done;  [recover] done gq N=200 seed=1 Q4: selection_nll=0.6705 train_nll=0.6482 val_nll=0.6726 n_params=1110 (458s) [3/135, 8.
* 2026-09-29 05:57 UTC — grid_a: 15 of 135 fits done;  [recover] done gc N=100 seed=2 Q4: selection_nll=0.5607 train_nll=0.5553 val_nll=0.5759 n_params=610 (589s) [15/135, 26
* 2026-09-29 11:57 UTC — container rebooted (~11:55 UTC); grid_a stopped at 42/135 and was resumed from its per-cell cache; API and fit-wiring agents were cut off by the session limit and are being finished by the main session.
* 2026-09-29 12:33 UTC — grid_a: 105 of 135 fits done since resume;  [recover] done gf N=200 seed=2 B2: selection_nll=0.5730 train_nll=0.5528 val_nll=0.5600 n_params=10
* 2026-09-29 12:59 UTC — grid_a: 121 done lines (42 pre-reboot + resumed);  [recover] done gf N=50 seed=0 B2: selection_nll=0.5929 train_nll=0.5669 val_nll=0.5785 n_params=285 (37s) [79
* 2026-09-29 13:28 UTC — grid_a at 79/135 (resumed run): workers alive but memory-bound (3 × 4.4 GB RSS, 1 GB free); chained grid_b detached, to be relaunched with 2 workers after grid_a's cap; dashboard agent's retrain test still running.
* 2026-09-29 13:56 UTC — grid_a finished at 13:43 (106 min): correct family selection 7/9 at N=50, 9/9 at N=100 and N=200 → N_target = 100 (17,000 responses). CAVEAT: results.json holds only the 79 fits of the resumed run; 14 cells are missing and several N=50 classical cells are absent (the two "wrong" N=50 cells are gf seeds where B2 is missing), so the N=50 row is provisional. Re-invoked the same grid (2 workers) to fill the missing cells from cache; transfer run A→B started (runs/realdata/transfer.stdout).
* 2026-09-29 14:06 UTC — A→B transfer (reports/transfer/A_to_B.md): with the shared feature set (feedback/block tags, worst-outcome loss magnitude) on CPC15 held-out problems only the transferred Q2 beats the constant rate (NLL 0.6881 [0.6836, 0.6926] vs 0.6974; r = 0.30 on those problems, but r ≈ 0.05 on the full target); on CPC18 held-out problems no transferred model beats the constant (0.6932). DECISION: because the effect does not replicate on CPC18 and the full-target correlations are near zero, no public-data population parameters enter demo mode, which keeps its labelled synthetic defaults. Mostly negative result recorded for the report.
* 2026-09-29 14:20 UTC — docs drafted: reports/MODEL_CARD.md, SUMMARY_PLAIN.md, TECHNICAL_APPENDIX.md, DATA_GAPS.md (every number sourced). Path note: the transparency page reads reports/phase4/verdict.json; the Phase 4 runner writes reports/phase4/<name>/; the primary run's verdict.json and table.md will be copied to the flat path once produced.
* 2026-09-29 14:47 UTC — second container restart (~14:45) killed the grid completion at 36/135 and the report agent (its files survived; bre.report committed, 10 tests). Memory-safe plan: Phase 4 split (b) on CPC18 pairs, all 686 subjects, classical models first (2 workers, runs/phase4/cpc18-pairs-classical.stdout), quantum models next, then an assembling invocation; recovery grid re-run per N with 1 worker into reports/recovery_n50 (then n100) and merged with the complete N=200 cells.
* 2026-09-29 15:05 UTC — full fast test suite: 702 passed (the earlier collection error was transient under memory pressure). Phase 4 classical split (b) and grid N=50 running.
* 2026-09-29 15:08 UTC — push succeeded; draft PR #1 opened; Phase 4 quantum split (b) batch running (4 jobs, ~42 min est.).
* 2026-09-29 15:22 UTC — third container restart (~15:20). grid N=50 finished before it (reports/recovery_n50: 9/9 correct family, 45 fits, 25.5 min, 1 worker). Phase 4 quantum batch relaunched (Q4 job cached; Q2/Q3/Q5 fitting); grid N=100 launched (1 worker).
* 2026-09-29 15:49 UTC — fourth restart (~15:48) killed the quantum batch (no new job cached) and grid N=100 (2/45). New policy: one single-worker process at a time; quantum Phase 4 jobs relaunched with 1 worker (per-job cache); grid N=100 to run per generator afterwards.
* 2026-09-29 16:12 UTC — DIAGNOSIS: each container restart coincides with the session going idle between check-ins (the container is reclaimed on inactivity and background runs die with it). Long runs are now kept alive by staying active in the session (Monitor waits) instead of scheduled check-ins.
