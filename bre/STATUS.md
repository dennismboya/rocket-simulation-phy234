# STATUS.md — resume here

Last updated: 2026-09-17 (session 1). Resume protocol: read this file and PLAN.md, continue from the
first unfinished item in "Next up".

## Phase status

| Phase | State | Notes |
|---|---|---|
| 0 Plan + scaffold | in progress | CLAUDE.md, PLAN.md, STATUS.md written; scaffold next |
| 1 Data | in progress | reconnaissance done, see "Data in hand" |
| 2 Recovery | not started | |
| 3 Models | not started | |
| 4 Evaluation | not started | protocol pre-registered in PLAN.md §6 |
| 5 API | not started | |
| 6 Dashboard | not started | |
| 7 Instrument | not started | |
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

1. Scaffold (requirements.txt, Makefile, schema.py, db/, tests).
2. Catalog + copy datasets A, B, D, C into data/raw; write loaders.
3. Q1 + QQ-equality test on D.

## Estimated remaining effort

Phases 0–8 as planned; no long runs started yet.
