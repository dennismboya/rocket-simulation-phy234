# BRE intake battery -- static app

A jsPsych 7.x single-page app with no build step. It runs from `file://` (double-click
`index.html`) and from any static host (GitHub Pages, any web server). Everything it needs is in
this folder; nothing is fetched from a CDN.

| File | Role |
|---|---|
| `index.html` | Page shell, styles, script tags. |
| `battery.js` | Runner: seeded randomization, timeline, row construction, completion page. |
| `battery.data.js` | Generated copy of `../battery.json` (`window.BRE_BATTERY`). Regenerate with `python ../validate_battery.py --write-static`; never edit by hand. |
| `config.js` | Deployment settings (endpoint, study contact). The only file you edit to deploy. |
| `vendor/` | jsPsych 7.3.4 and three plugins, MIT, versions and hashes in `vendor/VENDOR.md`. |

## Running it

* Locally: open `index.html` in a browser, or serve the folder (`python -m http.server 8000`
  from this directory, then `http://localhost:8000/`).
* URL parameters: `?form=full` (default; 12 scenarios + 2 repeats) or `?form=short` (5 scenarios).
  `&seed=<32 hex characters>` replays a given assignment (the seed is the `session_id`; a replay
  still gets a fresh `subject_id`).

## What a session produces

On the completion page the volunteer can download two files:

* `intake_<session_id>.json` -- the responses: a JSON array with one object per elicited decision
  (tolerance, sell/hold, allocation share), with exactly the 21 DecisionEvent fields in schema
  order. This is the file the pipeline ingests.
* `intake_<session_id>.session.json` -- the session log: seed, form, the full item assignment,
  per-page response times, the measured delay-page duration per item (`delay_display_ms`), the
  delay setting actually used, user agent, timestamps, POST status. Useful for timing and
  data-quality checks; not a schema table.

A copy of both is also written to the browser's `localStorage` under `bre_intake_<session_id>`
(recoverable from the browser's developer tools if a download failed).

## Collecting responses

Default (`ENDPOINT_URL: ""`): download-only. Volunteers download the responses file and send it to
the address in `STUDY_CONTACT`. Nothing leaves the browser on its own.

Optional: set `ENDPOINT_URL` in `config.js` and the app POSTs the same JSON (`Content-Type:
application/json`) to that URL when the completion page loads, shows the outcome, and offers
"Send again". The download buttons remain available either way, so a failed POST loses nothing.

A free "form endpoint" service is the zero-cost option. Several providers offer a free tier whose
whole job is to accept an HTTP POST at a URL they give you and store or email the submission. Any
of them works if it:

1. accepts a JSON body (`application/json`), not only URL-encoded form fields;
2. allows cross-origin requests from your GitHub Pages origin (CORS);
3. accepts a payload of roughly 30-60 KB (a full-form session, pretty-printed);
4. lets you export the stored submissions as JSON or CSV;
5. has a retention and privacy policy you are comfortable holding volunteer data under.

Most such services require the body to be a JSON *object* rather than a bare array, and some
require an access key inside the body. Use `ENDPOINT_WRAP_KEY` (for example `"responses"`) and
`ENDPOINT_EXTRA_FIELDS` (for example `{"access_key": "..."}`) for that. With
`ENDPOINT_INCLUDE_SESSION_LOG: true` the session log is included in the wrapped body under
`session_log`. When exporting from the service, pull out the `responses` array of each submission
and save it as `intake_<session_id>.json` (the `session_id` is inside every row).

Alternatives with the same contract: a small serverless function you own, or a spreadsheet-backed
web app script, both of which accept a JSON POST. No vendor is required or recommended here.

Do not put an access key in `config.js` if the key would let a stranger *read* submissions; a
write-only key is fine because the page is public anyway.

## Deploying to GitHub Pages

This folder is the site root. See `../README.md` for the repository-specific steps (the app lives in
a subfolder of a repository whose root belongs to another project, so it is published with a Pages
workflow that uploads only this folder).

## Testing

* `../validate_battery.py` checks `../battery.json` and that `battery.data.js` is in sync.
* `../tests/test_balance.js` (plain Node) checks the balance rule of `buildAssignment` over
  thousands of random seeds.
* `../tests/smoke_test.js` (Node + jsdom, dev-only) clicks through the whole app under a fake
  browser and checks every produced row, the session log and the POST body.
* `DELAY_SECONDS_OVERRIDE` in `config.js` shortens the timed delay page for manual testing; the value
  actually used is recorded in every session log, so a misconfigured deployment is detectable.
