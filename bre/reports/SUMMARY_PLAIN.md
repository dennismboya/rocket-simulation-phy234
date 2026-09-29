# SUMMARY_PLAIN.md — the Behavioral Risk Engine in plain English (2026-09-29)

For the owner. Every number here is copied from a file in the repository, named in brackets. Where
a technical word is unavoidable it is explained once, in italics.

## What was built

A working pipeline and a local advisor dashboard. Given an investor's six intake answers, a loss
size and a short sequence of *contexts* (things that happen before the decision: a news story
blaming a recession or a technical glitch, a friend who sells, a partial recovery, or the order in
which two questions are asked), the system returns a **predicted probability of selling** with an
uncertainty band, a book-wide triage table, a client profile, a ranked list of advisor
interventions, and a transparency page that shows where every number comes from
[`app/pages/6_Transparency.py`, `src/bre/predict.py`]. Ten models are implemented: five
*Q-models* (quantum-probability models: the investor's state is a vector that contexts rotate and
questions "measure", so the order of contexts and questions can matter) and five *classical
baselines* (ordinary probability models given the same inputs) [`PLAN.md` §4, `STATUS.md`].
Everything runs on an ordinary CPU; there is no quantum hardware [`CLAUDE.md`].

The model currently served, `demo-q4-v2`, was fitted on a **synthetic** book (computer-generated
investors, labelled as such): 65 investors, 1,040 answers, with no held-out test
[`runs/models/demo-q4-v2/artifact.json`]. Every screen carries a synthetic-data banner that cannot
be switched off while this is true [`app/pages/7_Settings.py`].

## What the synthetic recovery study shows — and does not show

Before touching real data, the pipeline was tested on data it generated itself with known answers
(*recovery study*: generate data from a known model, refit, check that the right model family is
chosen and the known parameters come back). As of `reports/recovery/summary.md` and
`N_target.json`: with 100 synthetic subjects the right family was chosen in 9 of 9 cases, with 200
in 9 of 9, with 50 in 7 of 9. The **calibration target is therefore 100 subjects, i.e. 17,000
answers on the full 170-item design** [`reports/recovery/N_target.json`]. The Q4 model recovered
the known context parameters at correlations of 0.90–0.98 at N = 100 [`reports/recovery/summary.md`].

Caveats: that run had failed and missing cells (`summary.md` lists 56 failed fits out of 135;
`STATUS.md` 13:56 UTC records 14 missing cells and that a completion run was started), so the
N = 50 row is provisional and the target may move when the completion run finishes. What this study
does **not** show: anything about real investors. It shows only that, if people behaved like the
generators, the pipeline would tell the model families apart at that sample size.

## What the public data could and could not test

Real, public, individual-level data in hand [`data/processed/README.md`]: 686 lab subjects making
510,750 lottery choices (CPC18), and three Psych-201 experiments (282, 130 and 1,300 participants).
None of them contains what the product needs — a manipulated portfolio loss, a manipulated context
and a later real trade, within the same person [`data/CATALOG.md`, "Reality check";
`data/DATA_MAP.md`]. CPC18 is used as an *analog* (retreating to the safer option after an
experienced loss), and the report must say so [`PLAN.md` §6].

Two real-data results exist so far, both negative or neutral:

* Transferring models fitted on the choices13k dataset to the CPC datasets carried no usable
  information: held-out *NLL* (negative log-likelihood: how surprised the model is by answers it
  did not see; lower is better) of about 0.69 for every model against a constant-rate baseline of
  0.6932 (CPC18) and 0.6974 (CPC15), with correlations near zero on the full targets
  [`reports/transfer/A_to_B.md`]. Consequently no public-data parameter was allowed into demo mode
  [`STATUS.md` 14:06 UTC].
* The one verified question-order table (Clinton/Gore) gives a QQ statistic of −0.0032, but the
  source has no sample sizes, so this is an implementation check, not a test
  [`reports/q1_qq_equality.md`].

The pre-registered real-data evaluation (Phase 4) **has not completed**: the full CPC18 run is
estimated at 2,429 minutes on 4 cores, above the two-hour limit that requires your approval, and
must be split into pieces [`runs/phase4/cpc18-pairs/phase4.log`, `reports/phase4/README.md`].

## What the dashboard can honestly claim today

* It can show, for any editable scenario, the predicted probability of selling under a model that
  was trained on labelled synthetic data, with intervals, with the number of training answers
  behind each context, and with every number traceable to a parameter and a file.
* It can state the pre-registered decision rule verbatim and that it has not yet been run: the
  transparency page shows "Decision rule not yet run: no real-data numbers are shown"
  [`app/pages/6_Transparency.py`].
* It cannot claim that Q-models predict real investors better than classical models, or that any
  intervention works: interventions are labelled "predicted effect, not causally validated"
  [`src/bre/predict.py`].
* One built-in honesty check: because a loss rotates the state, the predicted probability need not
  rise with the loss; only 35 of the 65 demo clients (0.54) have a non-decreasing response, and
  the "drawdown capacity" number uses a first-crossing rule for that reason
  [`runs/models/demo-q4-v2/artifact.json`, `src/bre/predict.py`].

## What real data would change

The only data that can settle the question are within-person answers to ordered context pairs
under both question orders on the shared design — exactly what the intake battery collects
[`src/bre/eval.py`, `decision_rule`, "data that would resolve it"; `instrument/`]. Routes, in
order of cost: (1) an unpaid volunteer pilot of 20–50 people, an instrument check only
[`instrument/README.md`]; (2) a paid Prolific study at the 100-subject target, whose base pay at
$12/hour would be $266.60 (academic rate) or $285.60 (corporate) for a 10-minute battery and
$319.92 / $342.72 for 12 minutes, before bonuses, a pilot and follow-up waves, and only with your
written approval [`PLAN.md` §9, `instrument/PROLIFIC.md` §9.3]; (3) free-registration panels (UAS,
LISS/DNB) that span the 2020 crash and would test whether stated preferences predict later real
actions [`data/REGISTRATION.md`]. With that data, Phase 4 can return one of two verdicts; either
one goes on the transparency page unchanged. The full table of gaps and costs is
`reports/DATA_GAPS.md`.
