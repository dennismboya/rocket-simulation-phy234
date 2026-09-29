# BRE advisor dashboard (Phase 6, pages 1–7)

Streamlit multipage app. Every number on every screen comes from the BRE scoring API
(`api/main.py` and the Phase 6 router `api/phase6.py`); `app/` imports no model code (the
intake page uses `bre.design.battery_subset` for the item draw, which is the shared design, not
a model). In demo mode (the served model was trained on synthetic data, the database lives
under `data/synthetic/`, or the `demo_mode` setting is on) a **SYNTHETIC DATA** banner is shown
on every screen (main area and sidebar) and written as the first line of every export.

```bash
BRE_VENV=/home/user/bre-venv make -C bre dashboard      # streamlit run app/Home.py; starts the API if 127.0.0.1:8000 is not listening
BRE_API_PORT=8010 BRE_VENV=/home/user/bre-venv make -C bre dashboard
BRE_API_URL=http://127.0.0.1:8010 /home/user/bre-venv/bin/python -m streamlit run app/Home.py   # against an API you started yourself
```

Environment: `BRE_API_URL` (default `http://127.0.0.1:$BRE_API_PORT`, port 8000),
`BRE_API_WAIT_S` (how long a page waits for the service to become ready, default 90 s),
`BRE_ADMIN=1` (the dashboard then sends the `X-BRE-Admin: 1` header, which the API accepts as its
admin flag for `POST /model/activate`; a local convenience, not authentication — see
`api/phase6.py`), `BRE_INSTRUMENT_URL` (read by the API: the deployed static instrument shown on
the intake page's "send a link" option; default the repository copy `instrument/static/index.html`).

## Pages

| Page | File | What it shows |
|---|---|---|
| Home | `Home.py` | market-state bar, the book under that state (clients scored, N above threshold, median with interval, calibration N), the served model's transparency card (`GET /model`: version, evaluation status, decision rule, interval source, calibration rule, calibrated contexts with their N, training refs, provenance note) |
| Book triage | `pages/1_Book_triage.py` | one row per client from `POST /score_book`: predicted probability of selling with its 80% interval and calibration N, change vs baseline, top driver, suggested intervention (predicted effect, not causally validated), behavioral drawdown capacity, last contact, editable contact status; sortable, filterable; CSV export; "N clients above threshold" alert with an editable threshold |
| Client profile | `pages/2_Client_profile.py` | baseline risk state (state angle, classical-equivalent score beside the client's own questionnaire score, consistency γ), sensitivity bars with 80% intervals and N, the loss × context heatmap (5 losses × 17 conditions, one `POST /predict` per cell), what-if panel with live recompute, drawdown capacity with an editable target (allocation guardrail), intervention ranking with a session log, response-history timeline (`GET /clients/{id}/responses`), stated-vs-revealed panel (only when `outcome_behavior` exists), "Behavioral Risk Profile" PDF |
| Intake battery (page 4) | `pages/4_Intake_battery.py` | the Phase 7 instrument embedded: **in-session** (consent with the optional training-consent checkbox → `consent_training`; the six intake covariates and the 5-item literacy quiz; the scenario battery drawn with the instrument's randomization scheme — `bre.design.battery_subset` seeded by the 32-hex session id, the full form's two verbatim repeats, randomized question order — one presentation per screen with progress; the assignment and session record are stored with the rows through `POST /clients/{id}/responses`) or **send a link** (the static instrument URL from `GET /intake/config`, the client id to keep with the file, and an upload box for `intake_<session_id>.json` → `POST /intake/upload`, checked with the loader's contract). Full form 12 items + 2 repeats (8–12 min), short form 5 items (about 3 min, annual renewal). In demo mode the rows are stored as SYNTHETIC and the page says so |
| Scenario and context editor (page 5) | `pages/5_Scenario_editor.py` | every context with its calibration status from the active artifact (consented training N, fitted θ with 95% interval, or **uncalibrated: prior only** in red); add a context (e.g. `news:tariff_shock`) or retire one; edit the scenario texts and the intervention scripts / transforms. Every save is a new version with a timestamp (`api/store.py`); past versions and past responses are never rewritten; the intake page records the versions it showed |
| Model and data transparency (page 6) | `pages/6_Transparency.py` | live model version and type with the registry (`GET /model/registry`; an admin can activate another version), the pre-registered decision rule verbatim, the Phase 4 verdict and held-out metrics — shown only once `reports/phase4/verdict.json` exists, until then exactly "Decision rule not yet run: no real-data numbers are shown" and no Phase 4 table — the served model's training metrics and in-sample calibration plot (synthetic under the banner), parameter counts, "responses needed for calibration" from the Phase 2 recovery study, the data provenance table (dataset, N, real or synthetic, licence), the model card when `reports/MODEL_CARD.md` exists, and the **Retrain** button (`POST /retrain` = `make retrain`) |
| Settings (page 7) | `pages/7_Settings.py` | alert threshold, status thresholds, drawdown-capacity target, prior strength (refits), which contexts count as the "typical crisis", retrain budget, demo-mode toggle (can force the banner on, never off while the data is synthetic); export a client's every stored row (JSON) and delete a client (cascades to every table, audit row) |

Components (`components/`): `api_client.py` (the only HTTP layer; cached reads and
predictions), `chrome.py` (page setup, banner, wording and formatting, settings seeding),
`market_bar.py` (the sidebar market state, kept in `st.session_state["market"]`), `store.py`
(session store for contact status and logged interventions), `intake.py` (the in-session
battery: item draw, repeats, scenario text, DecisionEvent rows — no HTTP, no model code),
`charts.py` (matplotlib figures), `exports.py` (CSV and PDF).

Settings (`GET /settings`) seed the session's alert threshold and capacity target on every page;
the API applies the status thresholds and the typical-crisis contexts inside `POST /score_book`,
so pages 1–3 recompute with them.

## Wording and transparency rules implemented

* Every probability is a *predicted probability of selling* under the served model; the app
  never states that an investor "will" do anything (the forbidden two-word phrase is absent from
  `app/`, tested).
* No naked point estimates: every probability is shown with its 80% (and, where the API gives
  it, 95%) interval and the number of training responses behind the calibration of the
  contexts involved (`n_calibration`); population parameters (`|θ_c|`, γ) carry their intervals.
* Every intervention effect is labelled with the API's label "predicted effect, not causally
  validated"; the drawdown capacity is labelled as an allocation guardrail.
* No PII beyond the optional display label; the covariates shown are the six intake fields.
* Exports: the triage CSV's first line is the banner in demo mode, its second line names the
  model version, the wording and the market state; the PDF repeats the banner in the header
  of every page.

## API endpoints used

Phase 5: `GET /health`, `GET /model`, `GET /clients`, `GET /interventions` (through the
ranking), `GET /market`, `GET /openapi.json` (field enums and route probing), `POST /score_book`,
`POST /predict`, `POST /profile`, `POST /interventions/rank`.

Phase 6 (`api/phase6.py`): `GET/PUT /settings`, `GET/POST /contexts`, `GET/POST /scenarios`,
`PUT /interventions`, `GET /interventions/versions`, `GET /intake/config`,
`POST /clients/{client_id}/responses`, `POST /intake/upload`, `GET /clients/{client_id}/responses`,
`GET /clients/{client_id}/export`, `DELETE /clients/{client_id}`, `GET /model/registry`,
`POST /model/activate` (admin), `GET /transparency`, `POST /retrain`.

## What the app still keeps in the browser session

* **Contact status** and **intervention log** — no endpoint writes `clients` status or
  `intervention_log`. Both live in `st.session_state` (`components/store.py`) for the browser
  session and every place that shows them says "kept in this browser session only". The
  "last contact" column is the last status change or logged intervention in the session.
* **Elapsed time / time since news** — the served design has no elapsed-time context; the field is
  sent as `duration_days`, and the API's note ("carried only") is shown next to it.
* The response history now comes from `GET /clients/{client_id}/responses`; the read-only SQLite
  fallback of `components/api_client.py` remains for an older service without the route.

## Intake rows written by the in-session battery

`dataset = "intake_battery"`, `battery_version = "1.0.0"`, `subject_id` = the client id (the API
requires it, so the client's fitted effects are keyed by it), `session_id` = the 32-hex seed,
three rows per presentation (tolerance `binary_yes_no` with `scenario_id = "tolerance"`,
sell/hold `binary_sell`, allocation share `allocation_pct` = slider / 100) in the item's question
order, `position_in_session` over the whole session, `prior_question_ids` = the question ids
already answered in the presentation, `source_row_ref` = `<form>:i<NN>[:rep_of_i<MM>]:<question>`,
`response_time_ms` at item level (the session record says so), `consent_training` from the
consent checkbox, `is_synthetic` = demo mode. Uploaded instrument files keep their rows as
downloaded (the tolerance rows of the static instrument carry the item's scenario id, as its
contract says) with `subject_id` rewritten to the client id and the file name prefixed to
`source_row_ref`; the original subject UUID is kept in the intake-session document. The
database refuses real rows under `data/synthetic/` and synthetic rows elsewhere; the demo
database therefore stores uploads only when "store as synthetic" is on, and says so.

## Tests

```bash
cd bre && /home/user/bre-venv/bin/python -m pytest tests/test_dashboard.py -q -p no:cacheprovider
```

`tests/test_dashboard.py` starts a live API on a free port against a copy of the seeded demo
database and drives the pages with `streamlit.testing.v1.AppTest`: banner on every page, the
drawdown slider changes the triage probabilities, the threshold alert updates, the CSV header
carries the banner, the heatmap is 5 × 17, the PDF carries the banner, a weak cause-frame match
is flagged, the market button fills the bar, a logged intervention shows on the triage page,
and the forbidden "will" + "sell" phrase is absent from `app/`.

`tests/test_dashboard_pages.py` (pages 4–7, API started with `BRE_ADMIN=1`): the banner on
pages 4–7; an in-session short-form battery stores 15 rows with its assignment recorded; the
send-a-link upload; an uncalibrated context is flagged in red on the editor; the exact
"Decision rule not yet run: no real-data numbers are shown" sentence and no Phase 4 table on
the transparency page; the settings page changes the triage threshold; export and delete
(every table empty afterwards); the what-if inputs change every dependent number.
`tests/test_api_admin.py` covers the endpoint contracts and `tests/test_retrain.py` the
retrain path (a worse refit is never promoted).
