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
| 0 Plan + scaffold | in progress | schema, design, db, packaging, 150 tests green; 23 review findings being fixed (contract amendments in PLAN.md §2) |
| 1 Data | in progress | reconnaissance done, see "Data in hand" |
| 2 Recovery | not started | |
| 3 Models | not started | |
| 4 Evaluation | not started | protocol pre-registered in PLAN.md §6 |
| 5 API | not started | |
| 6 Dashboard | not started | |
| 7 Instrument | in progress | battery.json, static jsPsych app (vendored 7.3.4), PROLIFIC.md, IRB_CHECKLIST.md, PREREGISTRATION.md written; browser verification running |
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

## Decisions

* Project lives in bre/; JAX stack (see PLAN.md §0).
* Interference-term definitions fixed in PLAN.md §4.

## Next up

1. Finish Phase 0 review fixes; commit the scaffold and the instrument; retry push.
2. Loaders for A, B, C, D, M → data/processed (catalog and copies done).
3. Q1 + QQ-equality test on D; generators G_Q/G_C/G_F; models B1, B2, B4, Q2, Q4; recovery study at N=200.

## Estimated remaining effort

Phases 0–8 as planned; no long runs started yet.
