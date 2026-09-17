# data/CATALOG.md — dataset catalog

## Reality check (2026-09-17)

As of 2026 no public dataset contains, within subject, a manipulated loss scenario × a manipulated
context × subsequent real trading. The strategy is triangulation: generic risky-choice data to learn
risk states (A, B, C), question-order survey data for the quantum-specific tests (D), a probability
panel spanning a real market shock for external validity (E, H — gated), and the dashboard's own
intake battery (Phase 7) for the exact product data.

Session constraint: the build session's egress policy only reaches PyPI, npm and GitHub. Datasets
below are marked ACQUIRED (a copy is in data/raw), MIRROR (acquired from a third-party GitHub copy of
the canonical record), INSTRUCTIONS (canonical host unreachable from the session; owner must
download), or GATE (registration required; not performed).

(Entries are appended by the main session as each dataset is inspected and copied.)

## Entries

### A. choices13k — ACQUIRED (aggregate)
* Source: https://github.com/jcpeterson/choices13k (git HEAD 821ae7e), accessed 2026-09-17. Paper: Peterson, Bourgin, Agrawal, Reichman & Griffiths, Science 372:1209–1214 (2021).
* License / terms: no LICENSE file in the repository; the paper's data statement makes the data public. Treat as research use; do not redistribute outside this repo without checking with the authors.
* Local files: `raw/choices13k/c13k_selections.csv` (14,568 rows; 13,006 unique problems × feedback condition/block), `raw/choices13k/c13k_problems.json` (14,568 entries keyed by row index; each has `A` and `B` lists of [probability, payout]), `raw/choices13k/README.md`. Plus `choices13k_thomas2024_features.csv` from https://github.com/RothkopfLab/DatasetBias (Thomas et al., Nat. Hum. Behav. 2024) with engineered features and dominance flags (`Dom`, `SOSD`, `TOSD`, `StochDom`); no LICENSE file there either.
* Fields (selections): Problem, Feedback (bool), n (15–33 participants per row), Block (1–5), Ha, pHa, La, Hb, pHb, Lb, LotShapeB (0–3), LotNumB (1–8), Amb (bool), Corr (−1/0/1), bRate (share choosing B), bRate_std.
* Schema mapping: dataset="choices13k"; subject_id = "agg:<Problem>:<Feedback>:<Block>" (aggregate pseudo-subject; the loader emits one row per CSV row with response = bRate, elicitation_type = choice_rate, and `weight` = n in covariates so fits use binomial weights); scenario_id = problem id; loss_pct = null (no portfolio-loss framing; the minimum outcome / max |outcome| is stored in covariates as `worst_outcome_rel`); context_tags = ["feedback:on"] or ["feedback:off"], plus "block:k"; incentivized = True; is_synthetic = False; source_row_ref = row index.
* Known problems: decision noise on dominated problems (Thomas et al. 2024); the loader filters rows where the Thomas-2024 file flags stochastic dominance and reports the count. Individual-level choices are not public, so subject-level Q-model fits are impossible on A; A serves encoder pretraining and the A→B transfer test only.
* Serves: RQ2 (transfer of the risk-state encoder), covariate-free population priors.

### B. CPC15 (aggregate) and CPC18 (individual) — MIRROR
* CPC15: Erev, Ert, Plonsky, Cohen & Cohen, Psychological Review 124:369–409 (2017). Canonical: https://economics.agri.huji.ac.il/crc2015/raw-data (unreachable from the session). Local copy `raw/cpc15/cpc15_thomas2024_aggregate.csv` (750 rows = 150 problems × 5 blocks) from the RothkopfLab/DatasetBias repository; fields Ha,pHa,La,LotShape*/LotNum A and B, Amb, Corr, engineered features, Block, Feedback, GameID, Rate (B-rate), BEAST_pred, Cogprior_pred. No LICENSE file in the mirror.
* CPC18: Plonsky, Apel, Ert, Tennenholtz & Erev (2018), https://cpc-18.com. Canonical raw data: Zenodo records 845873 (calibration set) and 2571510 (all data), unreachable from the session; license not readable from here. Local copy `raw/cpc18/cpc18_calibration_raw.csv.gz` from https://github.com/naecker-lab/cpc18 (public archive, accessed 2026-09-17): 510,750 rows, 686 subjects (Technion and Rehovot; ages 18–37; 274,625 F / 236,125 M rows), 210 problems in 7 sets of 30, 5 blocks × 5 trials, block 1 without feedback, blocks 2–5 with feedback. Fields: SubjID, Location, Gender, Age, Set, Condition (ByProb/ByFB), GameID, Ha,pHa,La,LotShapeA,LotNumA,Hb,pHb,Lb,LotShapeB,LotNumB,Amb,Corr, Order, Trial, Button, B (1 = chose B), Payoff, Forgone, RT (65% missing), Apay, Bpay, Feedback, block. Also `CPC18_EstSet210.csv` (210 problems, per-block B-rates) and `CPC18_EstSet.csv` (60 competition problems).
* Schema mapping (CPC18 individual): dataset="cpc18"; subject_id = SubjID; session_id = "set<Set>"; position_in_session = running trial index in the subject's order; scenario_id = GameID; response = 1 if the safer option was chosen (lower outcome variance; ties → option A), stored as elicitation_type lottery_choice with the raw B choice kept in covariates as `x_chose_b`; loss_pct = previous trial's obtained payoff divided by the problem's outcome range, signed (null on the first trial of a problem); context_tags = ["exp:loss"|"exp:gain"] for the preceding trial and, in the pair variant, the two preceding trials in order, plus "feedback:on|off", "block:k"; prior_question_ids = preceding GameIDs in the session; covariates = {age_band, x_gender, x_location}; incentivized = True; is_synthetic = False; source_row_ref = file row number.
* Known problems: student sample, lab setting, small stakes; the "sell" analog is retreat to the safer option, not a portfolio decision. RT mostly missing.
* Serves: RQ1 (sequence/order effects of experienced outcomes), RQ2 splits (a), (b) via the pre-registered mapping in PLAN.md §6, and the A→B transfer test.

### C. Psych-201 — ACQUIRED (three experiments); Psych-101 — INSTRUCTIONS
* Psych-201: https://github.com/marcelbinz/Psych-201 (commit c63dea1), Apache-2.0 (LICENSE copied to `raw/psych201/`). Copied folders: `spektor2024lossaversion` (282 participants, 50/50 two-outcome mixed lotteries across sessions/contexts; Spektor, Kellen, Rieskamp & Klauer, JEP:General 2024; original https://osf.io/28qzs/), `spektor2019contexteffects` (130 participants, decisions from experience among 2–3 options with context effects; Spektor, Gluth, Fontanesi & Rieskamp, Psych. Review 2019; https://osf.io/w376r/), `olschewski2024skewness` (1,300 participants, "Broker Game" stock choices from experience; Olschewski, Spektor & Le Mens, PNAS 2024; https://osf.io/aqjdz/). Format: one JSON line per participant with `text` (natural-language transcript with choices marked `<<X>>`), `experiment`, `participant`, optional `RTs`; the folder's generate_prompts.py documents the original columns so the loader can parse the transcript back into trials.
* Schema mapping: dataset = "psych201:<folder>"; subject_id = participant; session_id = block/session label from the transcript; response = 1 for the option with the lower variance (or the safe option where defined), elicitation_type = lottery_choice; context_tags = ["domain:gain"|"domain:loss"|"domain:mixed"] for spektor2024 (derived from the outcome signs), ["options:3"] and experienced-outcome tags (`exp:gain`/`exp:loss`) for the experience tasks; loss_pct = null except spektor2024 where it is min(outcome)/max|outcome| of the chosen lottery. Parsing is regex-based and validated by re-generating the sentence and comparing.
* Known problems: transcripts randomize option labels; RT only for spektor2024; olschewski2024 dividends are "points", not portfolio losses.
* Psych-101 (Hugging Face, marcelbinz/Psych-101): blocked from the session. Owner instructions: check the dataset card's license, then `huggingface-cli download marcelbinz/Psych-101 --repo-type dataset --local-dir data/raw/psych101`; the loader will filter experiments by keyword (risk, framing, order, disjunction).
* Serves: RQ1 (loss/gain domain context within subject), RQ2 split (a); spektor2024 is the best public within-subject loss-context data in hand.

### D. Question-order tables — ACQUIRED (partial, one table verified)
* Local file `raw/order_effects/question_order_tables.json` with `SOURCE.txt`. Clinton/Gore (Moore 2002 Gallup poll) is verified against Ozawa & Khrennikov, J. Math. Psych. 100:102491 (2021), eqs. (149)–(156). Black/White and Rose/Jackson are secondary transcriptions attributed to Wang, Solloway, Shiffrin & Busemeyer, PNAS 111:9431–9436 (2014) and are UNVERIFIED; they are used only in tests explicitly marked unverified and never in a reported result.
* Fields: the eight joint proportions P(first-question answer, second-question answer) for both orders; `intervening_information` flag.
* Schema mapping: dataset = "order_effects:<pair>"; aggregate rows: subject_id = "agg", elicitation_type = choice_rate with response = the joint proportion and `weight` = null in covariates (sample sizes are not in the source); the eight cells of each pair are eight rows, scenario_id = the cell label (e.g. AyBn); context_tags = ["order:A_first"|"order:B_first"]; question_order_id accordingly; is_synthetic = False.
* Missing: the 70-survey Dataset S1 of the PNAS paper (pnas.org unreachable). Owner action: download the SI dataset from doi:10.1073/pnas.1407756111 and place it in `raw/order_effects/wang2014_si/`; the loader `loaders/order_effects.py` documents the expected columns and will be validated when the file exists.
* Serves: RQ1 (QQ-equality structural test), Q1 unit test.

### E. Understanding America Study — GATE (registration)
* https://uasdata.usc.edu — registration form and Data User Agreement (submitted to uas-l@maillist.usc.edu per the site's own summary, 2026-09-17). Free. Terms as summarized on the site: no sharing with non-registered collaborators, download only what the approved research needs, cite UAS, store securely. Not registered; see `REGISTRATION.md`.
* Intended use: Understanding Coronavirus in America tracking survey (biweekly from April 2020) financial-behavior items and core-wave financial literacy / stock-expectation / risk items; March 2020 crash as a natural context change; RQ3 via reported financial actions across waves. Check for question/response order randomization assignment variables.

### F. FINRA NFCS Investor Survey — INSTRUCTIONS (host unreachable from the session)
* https://www.finrafoundation.org/nfcs-data-and-downloads — free downloads (.csv/.dta/.sav) for 2015, 2018, 2021; 2024 wave expected. Owner action: download the Investor Survey zips into `raw/finra_nfcs/<year>/`. Use: covariate marginals for `design.py` (replaces the placeholder), product feature design. Cross-sectional only.

### G. HRS, SCF, PSID — INSTRUCTIONS
* HRS: https://hrsdata.isr.umich.edu (free registration). SCF: https://www.federalreserve.gov/econres/scfindex.htm (public). PSID: https://psidonline.isr.umich.edu (free registration). Not fetched (hosts unreachable / registration). Use: risk-tolerance items and holdings changes across 2008 and 2020 for RQ3 sanity checks; covariate priors.

### H. LISS panel and DNB Household Survey — GATE (registration, non-US)
* LISS: https://www.lissdata.nl (Centerdata, Tilburg). Free for non-commercial research after signing the data user statement at https://statements.centerdata.nl/liss-panel-data-statement (per the site's own summary, 2026-09-17). DNB Household Survey: via Centerdata as well. Not registered; see `REGISTRATION.md`. Use: repeated risk-attitude items plus actual asset holdings across 2008 and 2020 → best public RQ3 candidate.

### I. Global Preferences Survey — INSTRUCTIONS
* https://www.briq-institute.org/global-preferences/downloads (registration on the briq site; free). Not fetched. Use: risk and patience covariate priors only.

### J. Robintrack — INSTRUCTIONS
* https://robintrack.net/data-download (full database CSV export, May 2018 – Aug 2020; the site notes Robinhood closed the API on 2020-08-13). Not fetched. Use: aggregate-only population-composition check around March 2020, not a panic-sell validation.

### K. Not public, benchmark only
* Elkind, Kaminski, Lo, Siah & Wong, "When do investors freak out? Machine learning predictions of panic selling", Journal of Financial Data Science (2022): individual brokerage accounts; not obtainable. Hiroshima University × Rakuten Securities "Survey on Life and Money": not obtainable. Their variable lists are to be extracted from the papers' text into `reports/BENCHMARK_FEATURES.md` once the papers are reachable; from this session neither paper's full text could be fetched, so that file records only what the abstracts state.

### M. Market series (for the market-state bar fallback and the payout-path table) — ACQUIRED
* S&P 500 monthly: https://github.com/datasets/s-and-p-500 (HEAD 78cdceb), files `raw/market/sp500_data.csv` (1,868 monthly rows, 1871-01 to 2026-08: Date, SP500, Dividend, Earnings, CPI, Long Interest Rate, Real Price, Real Dividend, Real Earnings, PE10), `sp500_README.md`, `sp500_datapackage.json`. License ODC-PDDL-1.0 per datapackage.json; sources Robert Shiller (http://www.econ.yale.edu/~shiller/data.htm) extended with FRED. Known problems: `SP500` is Shiller's monthly average of daily closes, not a month-end close; rows after Shiller's last update carry zeros in the non-price columns. Use only `SP500`, at monthly resolution, and label any episode table built from it as monthly-average based; the PROLIFIC.md payout mechanism prefers a daily close series that the owner must download (yfinance or FRED), and the path-table script records which series it used.
* VIX daily: https://github.com/datasets/finance-vix (HEAD f430b9d), files `raw/market/vix_vix-daily.csv` (9,273 rows, 1990-01-02 to 2026-09-15: DATE, OPEN, HIGH, LOW, CLOSE), README and datapackage. License ODC-PDDL-1.0 per the package; original source CBOE.
* Not schema data (no decisions). Used by `src/bre/market.py` for: the "Load current market" fallback when yfinance is unreachable (latest cached values, dated on screen), the VIX-bucket volatility regime, and a monthly-resolution drawdown-and-recovery episode table (stand-in for PROLIFIC.md's payout mechanism until a daily series is downloaded; every episode in that table traces to these rows).
* Serves: dashboard market-state bar; instrument payout design; no RQ.
