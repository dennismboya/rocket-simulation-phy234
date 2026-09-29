# DATA_GAPS.md — every missing dataset, what it would resolve, and what closing it costs

Written 2026-09-29 from `data/CATALOG.md`, `data/REGISTRATION.md`, `data/DATA_MAP.md`,
`data/processed/README.md`, `instrument/PROLIFIC.md`, `instrument/README.md`, `PLAN.md` and
`reports/recovery/N_target.json`. Nothing here has been registered, downloaded or paid for; the
session's egress policy blocks every host below except GitHub (`CLAUDE.md` environment notes).
Money costs are $0 wherever the catalog says the source is free or free-with-registration.
**Owner-time figures are the author's estimates**, not measurements, and exclude ethics-board
waiting time. "Decision-rule input" refers to the three conditions of `PLAN.md` §6 (split (b)
held-out NLL, matched parameter count, interference-term CI) and to the research questions
RQ1–RQ3 of `PLAN.md` §1.

## 1. Table of gaps

| Dataset (catalog entry) | What it would resolve | Access route | Money | Owner time (estimate) | Blocking step |
|---|---|---|---|---|---|
| Wang et al. 2014 PNAS Dataset S1, the 70-survey question-order data (D, missing part) | RQ1 structural test: the QQ-equality replication on real surveys, with sample sizes so an interval can be computed (today only one verified 2×2×2 table, no sample sizes; `reports/q1_qq_equality.md`). Not a decision-rule input (no loss, no context pair). | Download the SI from doi:10.1073/pnas.1407756111 → `data/raw/order_effects/wang2014_si/`; loader `loaders/order_effects.py` is written against the documented columns, untested on the real file (`data/CATALOG.md` D) | $0 | 0.5–1 h | Download (PNAS unreachable from the session); then the loader's first run on the real file |
| Psych-101 (C, Hugging Face `marcelbinz/Psych-101`) | More real within-subject risky-choice experiments for RQ1 (domain/framing/order effects) and RQ2 split (a); no ordered context pairs with both question orders, so not a decision-rule input | `huggingface-cli download marcelbinz/Psych-101 --repo-type dataset --local-dir data/raw/psych101` after checking the dataset card's licence (`data/CATALOG.md` C) | $0 | 1–3 h (licence check, large download, keyword filter run) | Download + licence check (HF blocked from the session) |
| FINRA NFCS Investor Survey 2015 / 2018 / 2021 (F) | Replaces the placeholder covariate marginals in `src/bre/design.py` (`PLAN.md` §3) and informs product feature design; cross-sectional, so no RQ test and no decision-rule input | https://www.finrafoundation.org/nfcs-data-and-downloads → `data/raw/finra_nfcs/<year>/` (`data/REGISTRATION.md`) | $0 | 1–2 h | Download (host unreachable from the session); the marginal-fitting step is not yet written |
| Understanding America Study (E) — GATE | RQ3 external validity: stated risk/expectation items across waves spanning the March 2020 crash with reported financial actions (`data/DATA_MAP.md`); may add real question/response-order randomization variables for RQ1. Not a split (b) input (no manipulated context pairs) | Register at https://uasdata.usc.edu, submit the Data User Agreement, download core waves + the Coronavirus tracking survey + codebooks → `data/raw/uas/<wave>/` (git-ignored) (`data/REGISTRATION.md` E) | $0 | 3–6 h (registration, agreement, codebook reading, downloads) plus the loader, which is not yet written | Registration and acceptance of terms in the owner's name (CLAUDE.md rule 4(b) GATE) |
| LISS panel and DNB Household Survey (H) — GATE | RQ3, "best public candidate" (`data/DATA_MAP.md`): yearly risk-attitude items plus actual asset holdings across 2008–2010 and 2019–2021; non-US sample, labelled as such | Sign the LISS data user statement at https://statements.centerdata.nl/liss-panel-data-statement, request DHS access through Centerdata, download the Assets and risk-attitude modules → `data/raw/liss/`, `data/raw/dhs/` (`data/REGISTRATION.md` H) | $0 (non-commercial research) | 4–8 h (two applications, module selection across ~6 waves, downloads) plus the loader, not yet written | Registration / data-user statement (GATE); approval wait; non-commercial-use restriction to confirm against the intended product use |
| HRS, SCF, PSID (G) | RQ3 sanity checks: risk-tolerance items and holdings changes across 2008 and 2020; covariate priors. Not a decision-rule input | HRS https://hrsdata.isr.umich.edu (free registration); SCF https://www.federalreserve.gov/econres/scfindex.htm (public); PSID https://psidonline.isr.umich.edu (free registration) (`data/CATALOG.md` G) | $0 | 2–4 h per survey | Registration for HRS and PSID (GATE); download for SCF; no loader written |
| Global Preferences Survey (I) | Risk and patience covariate priors only | https://www.briq-institute.org/global-preferences/downloads (site registration) → `data/raw/gps/` | $0 | 0.5–1 h | Site registration (GATE); download |
| Robintrack (J) | Aggregate-only population-composition check around March 2020; explicitly not a panic-sell validation (`data/CATALOG.md` J) | https://robintrack.net/data-download (May 2018 – Aug 2020 export) → `data/raw/robintrack/` | $0 | 0.5–1 h | Download (host unreachable from the session); no loader written |
| Intake-battery pilot (instrument; `data/raw/intake_pilot/` holds 0 files, `data/processed/README.md`) | Instrument check only: median session duration, exclusion/speeding thresholds, bug list (`instrument/README.md` §"Running the unpaid pilot"). Produces real rows in the exact product schema (manipulated loss × context × order, within subject) but at 20–50 volunteers it is below the 100-subject calibration target and is "not for inference" (`instrument/README.md`) | Host the static jsPsych build under `instrument/static/`, recruit 20–50 unpaid volunteers, collect the exported JSON into `data/raw/intake_pilot/`, run `make data` | $0 | 10–20 h (recruiting, briefing, collecting files, checking exports) | Ethics determination for the owner's setting before the first volunteer (`instrument/README.md`); hosting; recruitment |
| Prolific study of the intake battery (shelved; `instrument/PROLIFIC.md`) | **The only source of every decision-rule input at once**: within-subject ordered context pairs (split (b) by design), both question orders (split (c), `delta_LTP` and the QQ test with the tolerance question), a sample at the Phase 2 target, plus wave-2 self-reports and an optional event wave for RQ3 (`instrument/PROLIFIC.md` §1) | Prolific, per `PROLIFIC.md` §13: implement the Prolific-mode overrides, build the path table from an owner-downloaded daily S&P 500 series, ethics review, register `PREREGISTRATION.md`, pilot of 20, wave 1 at `N_recruit = ceil(N_target/(1−e))`, optional wave 2 and event wave | Base pay at N = 100: **$266.60 / $285.60** (10 min) or **$319.92 / $342.72** (12 min), academic / corporate rate; arithmetic in §2. Excluded: bonus stake × E[M], the pilot, wave 2, any event wave, recruitment overhead for exclusions, taxes | 25–40 h (instrument overrides, path-table build, ethics submission, pilot, launch and monitoring, ingest) | Money (GATE, CLAUDE.md rule 4(a)); ethics review (`instrument/IRB_CHECKLIST.md`); pilot measurements that fill the tokens of `PROLIFIC.md` §14 |
| Also: CPC18 Zenodo originals (records 845873, 2571510) (B) | Confirms licence and checksum of the GitHub mirror actually used for the pre-registered split (b) analog | Zenodo (unreachable from the session); compare with `data/raw/cpc18/` (`data/REGISTRATION.md`) | $0 | 0.5 h | Download and checksum |
| Also: daily S&P 500 closing series (M, PROLIFIC §5.3) | Required before any bonus can be paid: the incentive path table must be built from a daily series; the monthly Shiller stand-in "never pays a bonus" | yfinance `^GSPC` or FRED `SP500`, catalogued in `data/CATALOG.md` with SHA-256, then `instrument/build_path_table.py` (not yet written) | $0 | 1–2 h plus the script | Download (market hosts unreachable from the session); script not written |
| Also: full texts of the two benchmark studies (K) | Fills `reports/BENCHMARK_FEATURES.md` beyond the abstracts (covariate checklist); the data themselves are not obtainable | Journal / SSRN / PLOS pages | $0 (open pages) or the journal's fee | 1 h | Reachability from the owner's machine |

## 2. Prolific wave-1 base-pay arithmetic (formula from `PLAN.md` §9 and `instrument/PROLIFIC.md` §9.2)

Cost_academic(N, T) = T/60 × $12/hr × N × 1.333; Cost_corporate(N, T) = T/60 × $12/hr × N × 1.428.
N = N_target = 100 (`reports/recovery/N_target.json`, as of that file; see the caveat in §3).

| T (minutes) | base pay per participant | × N = 100 | × 1.333 academic / non-profit | × 1.428 corporate |
|---|---|---|---|---|
| 10 | 10/60 × 12 = $2.00 | $200.00 | 200.00 × 1.333 = **$266.60** | 200.00 × 1.428 = **$285.60** |
| 12 | 12/60 × 12 = $2.40 | $240.00 | 240.00 × 1.333 = **$319.92** | 240.00 × 1.428 = **$342.72** |

The N = 100 rows of `instrument/PROLIFIC.md` §9.3 give the same four figures. What the formula does
not cover (`PROLIFIC.md` §9.2): the bonus, `Bonus_cost(N) = fee_factor × N × B × E[M]` with the
stake B an owner decision and E[M] from the not-yet-built path table; the pilot of 20 participants
(design decision, `PROLIFIC.md` §13; at the same rates 20 × $2.00 = $40.00 → $53.32 / $57.12 for
10 min and 20 × $2.40 = $48.00 → $63.98 / $68.54 for 12 min, computed here from the same formula and
rounded to the cent); wave 2 (N × retention, unknown until wave 1) and any event wave; recruitment
above N to cover the exclusion rate e, unknown until the pilot (`N_recruit = ceil(N/(1−e))`); taxes.
T itself is a pilot measurement; 10 and 12 minutes are the two durations the brief asks for, and
`target_duration_min` [8, 12] in `battery.json` is an instrument target, not a measurement
(`PROLIFIC.md` §2, §9.2).

## 3. Notes

* **N target caveat.** N_target = 100 is the value in `reports/recovery/N_target.json` generated
  2026-09-29T13:43:13+00:00; `STATUS.md` (13:56 UTC) records that 14 cells of that grid were missing
  and that a completion run was started, so the target — and every cost row derived from it — may
  change. The superseded old-defaults pass (`reports/recovery_pass1_olddefaults/`) gave 200 at the
  only N it tested and is not used here.
* **What no listed dataset gives.** Within-subject manipulated loss × manipulated context × later
  *real* trade. Nothing public provides it (`data/DATA_MAP.md`, last paragraph); the intake battery
  plus `intervention_log` is the only path, and even the Prolific design records later actions as
  self-reports (`instrument/PROLIFIC.md` §1, §6.2).
* **GATEs.** Every registration in the owner's name and every dollar is a GATE (CLAUDE.md rule 4);
  every Phase 4 run over two hours wall-clock is a GATE as well (`reports/phase4/README.md`,
  `runs/phase4/cpc18-pairs/phase4.log`: 2429 min estimated with 4 workers).
