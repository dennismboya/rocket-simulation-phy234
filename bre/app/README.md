# BRE advisor dashboard (Phase 6, pages 1–3)

Streamlit multipage app. Every number on every screen comes from the BRE scoring API
(`api/main.py`); `app/` imports no model code. In demo mode (the served model was trained on
synthetic data, or the database lives under `data/synthetic/`) a **SYNTHETIC DATA** banner is
shown on every screen (main area and sidebar) and written as the first line of every export.

```bash
BRE_VENV=/home/user/bre-venv make -C bre dashboard      # streamlit run app/Home.py; starts the API if 127.0.0.1:8000 is not listening
BRE_API_PORT=8010 BRE_VENV=/home/user/bre-venv make -C bre dashboard
BRE_API_URL=http://127.0.0.1:8010 /home/user/bre-venv/bin/python -m streamlit run app/Home.py   # against an API you started yourself
```

Environment: `BRE_API_URL` (default `http://127.0.0.1:$BRE_API_PORT`, port 8000),
`BRE_API_WAIT_S` (how long a page waits for the service to become ready, default 90 s).

## Pages

| Page | File | What it shows |
|---|---|---|
| Home | `Home.py` | market-state bar, the book under that state (clients scored, N above threshold, median with interval, calibration N), the served model's transparency card (`GET /model`: version, evaluation status, decision rule, interval source, calibration rule, calibrated contexts with their N, training refs, provenance note) |
| Book triage | `pages/1_Book_triage.py` | one row per client from `POST /score_book`: predicted probability of selling with its 80% interval and calibration N, change vs baseline, top driver, suggested intervention (predicted effect, not causally validated), behavioral drawdown capacity, last contact, editable contact status; sortable, filterable; CSV export; "N clients above threshold" alert with an editable threshold |
| Client profile | `pages/2_Client_profile.py` | baseline risk state (state angle, classical-equivalent score beside the client's own questionnaire score, consistency γ), sensitivity bars with 80% intervals and N, the loss × context heatmap (5 losses × 17 conditions, one `POST /predict` per cell), what-if panel with live recompute, drawdown capacity with an editable target (allocation guardrail), intervention ranking with a session log, response-history timeline, stated-vs-revealed panel (only when `outcome_behavior` exists), "Behavioral Risk Profile" PDF |

Components (`components/`): `api_client.py` (the only HTTP layer; cached reads and
predictions), `chrome.py` (page setup, banner, wording and formatting), `market_bar.py`
(the sidebar market state, kept in `st.session_state["market"]`), `store.py` (session store
for contact status and logged interventions), `charts.py` (matplotlib figures), `exports.py`
(CSV and PDF).

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

`GET /health`, `GET /model`, `GET /clients`, `GET /interventions` (through the ranking),
`GET /market`, `GET /openapi.json` (field enums and route probing), `POST /score_book`,
`POST /predict`, `POST /profile`, `POST /interventions/rank`.

## Endpoints the Phase 5 API does not have (and what the app does instead)

* **Client response history** — no `GET /clients/{client_id}/responses`. The app probes the
  route through `/openapi.json` and, while it is absent, reads the client's rows **read-only**
  from the SQLite file the service names in `GET /health` (`components/api_client.py`,
  `client_responses`). The panel says so in its source line. Raw stored data only, no model code.
* **Contact status** and **intervention log** — no endpoint writes `clients` status or
  `intervention_log`. Both live in `st.session_state` (`components/store.py`) for the browser
  session and every place that shows them says "kept in this browser session only". The
  "last contact" column is the last status change or logged intervention in the session.
* **Elapsed time / time since news** — the served design has no elapsed-time context; the field is
  sent as `duration_days`, and the API's note ("carried only") is shown next to it.

## Tests

```bash
cd bre && /home/user/bre-venv/bin/python -m pytest tests/test_dashboard.py -q -p no:cacheprovider
```

`tests/test_dashboard.py` starts a live API on a free port against a copy of the seeded demo
database and drives the pages with `streamlit.testing.v1.AppTest`: banner on every page, the
drawdown slider changes the triage probabilities, the threshold alert updates, the CSV header
carries the banner, the heatmap is 5 × 17, the PDF carries the banner, a weak cause-frame match
is flagged, the market button fills the bar, a logged intervention shows on the triage page,
and no "will sell" string exists under `app/`.
