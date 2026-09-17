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
| 2 Recovery | not started | |
| 3 Models | in progress | core + interface (18cd33d) and Q1 (e76a1c4) done; Q2/Q4 + G_Q and B1/B2/B4 + G_C/G_F being written (agents) |
| 4 Evaluation | not started | protocol pre-registered in PLAN.md §6 |
| 5 API | not started | |
| 6 Dashboard | not started | |
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

## Decisions

* Project lives in bre/; JAX stack (see PLAN.md §0).
* Interference-term definitions fixed in PLAN.md §4.

## Next up

1. Generators G_Q/G_C/G_F and models Q2, Q4, B1, B2, B4 (agents running); then fit.py, eval.py, recover.py and the N=200 recovery study.
2. API (Phase 5) and dashboard pages 1–3 on the synthetic demo book.
3. Real-data fits on A/B (B1, B3, Q2), transfer A→B; dashboard pages 4–7; Phase 4; report.

## Estimated remaining effort

Phases 0–8 as planned; no long runs started yet.
