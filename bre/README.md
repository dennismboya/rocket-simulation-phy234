# Behavioral Risk Engine (BRE)

A reproducible research pipeline plus a local advisor dashboard that (1) assembles the best
available behavioral data on context-dependent financial risk decisions, (2) trains
quantum-probability models of an investor's decision state against matched classical baselines,
(3) tests whether the quantum-probability models (Q-models) predict context effects — panic-selling
after losses, question-order effects, preference reversals, overreaction to news — better than the
classical ones, (4) serves per-investor predictions P(sell | loss size, context sequence) through an
API, and (5) delivers an advisor-facing dashboard in which every input is editable, every number
traces to a model parameter and a data source, and the intake questionnaire doubles as the
data-collection instrument. "Quantum" means quantum probability (state vectors or density matrices,
non-commuting projectors, unitaries for context, Born-rule probabilities) computed on an ordinary CPU;
there is no quantum hardware in this project.

Operating rules are in CLAUDE.md, the plan in PLAN.md, the current state in STATUS.md. Everything in
this directory is demonstrable end to end on public data plus clearly labelled synthetic data.

## Quick start

Python 3.11, CPU only. From the repository root or from `bre/`:

```bash
make setup && make sim && make recover && make fit && make eval && make dashboard
```

* `make setup` creates `.venv` (or reuses the one named by `BRE_VENV`), installs the pinned
  `requirements.txt` and this package in editable mode.
* `make test` runs pytest (math tests: unitarity, probabilities sum to one, recovery of known
  parameters; schema and database tests; dashboard acceptance tests).
* `make help` lists every phase target. A phase that is not built yet fails with
  `phase not built yet: see STATUS.md`.
* `make api` serves the FastAPI app on 127.0.0.1:8000; `make dashboard` starts the API in the
  background and opens the Streamlit app.

## Layout

```
bre/
├── CLAUDE.md, PLAN.md, STATUS.md   operating rules, plan, resume point
├── Makefile, pyproject.toml, requirements.txt
├── src/bre/                        the package (src layout)
│   ├── schema.py                   unified DecisionEvent schema, validators, parquet I/O
│   ├── design.py                   loss scenarios, 17 context conditions, question orders, covariates
│   ├── loaders/                    one loader per catalogued dataset  (Phase 1)
│   ├── sim/                        generators G_Q, G_C, G_F            (Phase 2)
│   └── models/classical, models/quantum   B1–B6 and Q1–Q5            (Phase 3)
├── db/                             SQLAlchemy models, engine/session helpers, seed interventions
├── api/                            FastAPI service                     (Phase 5)
├── app/                            Streamlit dashboard, Home.py + pages (Phase 6)
├── instrument/                     static jsPsych intake battery       (Phase 7)
├── experiments/                    experiment configs (see experiments/README.md)
├── data/
│   ├── CATALOG.md, DATA_MAP.md, REGISTRATION.md   provenance, RQ mapping, gated datasets
│   ├── raw/                        catalogued copies, SHA256SUMS.txt
│   ├── processed/                  *.parquet built by `make data` (gitignored)
│   └── synthetic/                  tables with is_synthetic = True (gitignored)
├── runs/                           logs and outputs of fits and studies (gitignored)
├── reports/                        REPORT.md, MODEL_CARD.md, benchmark notes
└── tests/                          pytest suite, conftest.py
```

## Real versus synthetic data

Simulated data lives only under `data/synthetic/`, and every table — real or simulated — carries an
`is_synthetic` column. `bre.schema.validate_frame` refuses a frame whose `is_synthetic` values do not
match its location. No synthetic result may appear in a table, figure or dashboard screen labelled
as real, and there are no fabricated or "illustrative" numbers anywhere: the dashboard's demo mode
runs on the synthetic demo book with its banner on. Negative results are reported as results.

## Batch scoring

The Phase 5 service (`make api`) scores one investor or a batch of investors. The endpoint paths and
schemas are fixed by the OpenAPI document the service publishes at `http://127.0.0.1:8000/docs`
once `api/main.py` exists; this section records the request shape so callers can prepare, and the
curl example is added when the service lands.

Request shape (one item per investor × scenario; a batch is a JSON list of these):

| Field | Meaning |
|---|---|
| `display_label` | optional, free text shown to the advisor; the only identifying field accepted |
| `covariates` | `age_band`, `wealth_band`, `invest_experience_yrs`, `self_reported_risk_tolerance`, `financial_literacy_score`, `education` (PLAN.md section 3) |
| `loss_pct` | signed portfolio loss, one of −0.05, −0.10, −0.15, −0.20, −0.30 (other values are interpolated by the served model and flagged) |
| `horizon_days` | investment horizon, default 365 |
| `context_tags` | ordered list drawn from `none`, `news:recession`, `news:technical`, `social:friend_sells`, `market:recovered_5pct` |
| `question_order_id` | `tolerance-first` or `scenario-first` |
| `prior_answers` | answers already given in this session, if any |
| `model_version` | optional; defaults to the active entry of the model registry |

Response shape per item: `p_sell`, its uncertainty interval, the served `model_version` and model type
(the dashboard says so when the winner is a classical model), and the interference terms defined in
PLAN.md section 4 — `delta_ltp`, `delta_mix` (when the cause frame is a mixture), `delta_order` for a
context pair — each with the parameters and data sources it traces to.

## Production deployment

The service and the dashboard store no personal data beyond an optional display label: no names,
account numbers, contact details or free-text notes that identify a person are accepted or logged,
and a client's rows can be exported and deleted in full. The engine is a decision-support component
only. A firm that deploys it must add its own authentication and access controls, encryption at
rest and in transit, audit and record-keeping to its regulatory standard, and a review of the
served model version before advisors act on it.

## Provenance

Every dataset used anywhere in this project has an entry in `data/CATALOG.md` (source URL, license
or terms, access date, size, fields, mapping to the schema, known problems); nothing is loaded that
is not catalogued. `data/DATA_MAP.md` maps datasets to research questions and
`data/REGISTRATION.md` lists the gated datasets and how the owner can obtain them.
