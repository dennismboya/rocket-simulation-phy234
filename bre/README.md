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
  `requirements.txt` and this package in editable mode. To reuse an existing virtualenv instead
  of creating `.venv`, set `BRE_VENV`: `BRE_VENV=/path/to/venv make test` (in the build
  container the venv is `/home/user/bre-venv`, so `BRE_VENV=/home/user/bre-venv make test`).
  When the interpreter is missing, every target prints a one-line hint naming `BRE_VENV`.
* `make test` runs pytest (math tests: unitarity, probabilities sum to one, recovery of known
  parameters; schema and database tests; dashboard acceptance tests).
* `make instrument` validates `instrument/battery.json`, regenerates `instrument/static/battery.data.js`
  and runs the balance test; `make test-instrument` runs the same checks read-only. Both need Node
  (`NODE ?= node`; in the build container `NODE=/opt/node22/bin/node make instrument`).
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
`is_synthetic` column. `bre.schema.write_events` refuses a frame whose `is_synthetic` values do not
match its location, and the database applies the same rule to every insert (a `before_flush`
listener judges the SQLite file's absolute path, so `frame_to_responses` and a plain
`session.add(Response(...))` are checked alike, and no `responses` table ever mixes the two flags).
No synthetic result may appear in a table, figure or dashboard screen labelled as real, and there
are no fabricated or "illustrative" numbers anywhere: the dashboard's demo mode runs on the
synthetic demo book with its banner on. Negative results are reported as results.

## Schema contract (PLAN.md section 2)

`bre.schema` is the single source of the DecisionEvent rules; `db.models` imports its constants
and validators, so parquet files and the `responses` table accept exactly the same rows.
`elicitation_type` is one of `binary_sell`, `allocation_pct`, `likert`, `lottery_choice`,
`binary_yes_no` (yes/no survey items, e.g. the tolerance question; response in {0, 1}) and
`choice_rate` (aggregate share in [0, 1]). `covariates` in `responses` takes the six product keys,
`weight` (respondents behind an aggregate row) and dataset-specific `x_*` keys; `clients.covariates`
takes the six product keys only. `context_tags` are `namespace:value`. Rows of the shared design
(`battery_version == bre.design.DESIGN_VERSION`, currently `1.0.0`) use the question-order ids
`tolerance-first` / `scenario-first`; external datasets carry free-form order ids.

## Batch scoring

The Phase 5 service (`make api`; `BRE_API_PORT` picks the port) scores one investor or a whole
book. Its contract is `api/openapi.json` (regenerated from the app by `api/export_openapi.py`,
browsable at `http://127.0.0.1:8000/docs`); `api/README.md` documents every endpoint. The
dashboard obtains every number from this service; no model code runs in the UI.

* `POST /predict` — one investor (six covariates), one scenario (`loss_pct` in [-1, 0],
  at most two ordered `context_tags` from `none`, `news:recession`, `news:technical`,
  `social:friend_sells`, `market:recovered_5pct`; `question_order_id`; optional prior tolerance
  answer, mixture weights for a mixed cause frame, and intake answers to refine the investor's
  random effects). Returns the *predicted probability of selling* `p` with its 80% and 95%
  intervals (percentiles over the artifact's parameter draws), the interference terms of
  PLAN.md section 4 (`ltp`; `order_effect` for a pair; `mix` for a mixture), the top driver,
  the number of training responses behind each context, the served `model_version` and type,
  and `synthetic: true` when the model was trained on synthetic data.
* `POST /score_book` (JSON list of clients + one market state) and `POST /score_book/csv`
  (CSV upload with `client_id`, optional `display_label` and the six covariate columns) —
  the triage table: per client `p_sell` under the market state, change versus the no-loss
  baseline, 80% interval, top driver, suggested intervention ("predicted effect, not causally
  validated"), behavioral drawdown capacity and status. The market state (drawdown, duration,
  free-text cause frame, recovery, VIX bucket, media intensity, social-cue prevalence) is mapped
  to a loss on the design range and an ordered context list; a cause frame that matches no
  calibrated context well is flagged `weak_match`.

```bash
curl -s -X POST http://127.0.0.1:8000/score_book -H 'Content-Type: application/json' -d '{
  "clients": [{"client_id": "DEMO-C01", "display_label": "Demo client 01",
               "covariates": {"age_band": "30-44", "wealth_band": "50-250k", "invest_experience_yrs": 5,
                              "self_reported_risk_tolerance": 3, "financial_literacy_score": 2, "education": "bachelor"}}],
  "market_state": {"drawdown_pct": -0.14, "cause_frame": "analysts expect a recession", "social_cue_prevalence": "medium"}
}'
```

Every prediction shown is written to `predictions_log` and `audit_log`; validation errors
return 422 with the field names. In demo mode (the default database `data/synthetic/demo.db`,
seeded on first start with the synthetic demo book of `bre.demo`) `GET /model` reports
`demo_mode: true` and `is_synthetic_training: true`.

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
