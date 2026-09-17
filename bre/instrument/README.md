# Intake battery (Phase 7 instrument)

The intake questionnaire of the Behavioral Risk Engine doubles as the data-collection instrument:
each volunteer session yields DecisionEvent rows in the unified schema (PLAN.md section 2) drawn
from the shared experimental design (PLAN.md section 3). This folder holds the battery definition,
the static jsPsych app that administers it, the validator, and the tests.

```
instrument/
  battery.json          single source of truth: design, wording, forms, output contract (v1.0.0)
  validate_battery.py   checks battery.json; --write-static regenerates static/battery.data.js
  static/               the app (index.html, battery.js, config.js, battery.data.js, vendor/)
  static/README.md      running, collecting responses, endpoint option
  static/vendor/VENDOR.md   exact jsPsych versions, hashes, re-vendoring steps
  tests/test_balance.js     balance rule over random seeds (plain Node)
  tests/smoke_test.js       end-to-end click-through under jsdom (dev-only dependency)
```

## What the battery does

* Design (PLAN.md section 3, verbatim): 5 loss levels x 17 context conditions x 2 question orders
  = 170 scenario items, all enumerated in `battery.json` under `design.scenario_universe` with
  stable ids of the form `L-0.15|news:recession>social:friend_sells|tolerance-first`.
* Forms: `full` = 12 items balanced over loss x condition type (none/single/pair) x question order,
  plus 2 verbatim repeats inserted later in the session (test-retest), plus intake; target 8-12
  minutes. `short` = 5 items, no repeats. The balance rules are written out in
  `battery.json -> forms.<form>.balance` and enforced by `tests/test_balance.js`.
* Each item: (tolerance question first, if assigned) -> scenario page with the loss sentence and the
  context sentences in the assigned order -> a timed reading page ("a few days later") whenever a
  news context is present (`design.delay_page.display_seconds`, measured duration logged) ->
  sell/hold buttons -> allocation slider "What share of this position would you sell?" (0-100,
  stored as a fraction) -> (tolerance question last, if assigned).
* Intake before the scenarios: consent (participation checkbox, plus the optional
  "I consent to my anonymised responses being used to train the model" checkbox that sets
  `consent_training`), age band, education, wealth band, years investing, self-reported risk
  tolerance 1-7, and a 5-item financial-literacy quiz (three standard items plus two more; answers
  keyed in `battery.json`; score 0-5).
* Randomization: seeded PRNG (sfc32) whose 128-bit seed comes from `crypto.getRandomValues`; the
  seed is the `session_id`, so `buildAssignment(battery, form, seed)` in `static/battery.js`
  regenerates the exact item subset, presentation order, question orders and repeat positions
  offline. The whole assignment is also written to the session log.
* Output: one row per elicited decision (tolerance, sell/hold, allocation each their own row),
  exactly the 21 schema fields in order; `dataset = "intake_battery"`, `subject_id` = random UUID,
  `session_id` = seed, `incentivized = false`, `is_synthetic = false`, `outcome_behavior = null`,
  `battery_version = "1.0.0"`, `source_row_ref = "<form>:i<NN>[:rep_of_i<MM>]:<question_id>"`.

Validate the definition and regenerate the browser copy after any edit to `battery.json`:

```
/home/user/bre-venv/bin/python bre/instrument/validate_battery.py --write-static
/opt/node22/bin/node bre/instrument/tests/test_balance.js
```

Both exit non-zero on failure; a future `make instrument` target should run exactly these two.

## Deploying to GitHub Pages

The app is plain files; there is no build. The repository root belongs to the PHY234 project, so
do not point Pages at the root or at `/docs`. Publish only `bre/instrument/static/` with a Pages
workflow:

1. In the repository: Settings -> Pages -> Build and deployment -> Source: **GitHub Actions**.
2. Add `.github/workflows/pages-instrument.yml` (owner to add; it lives outside `bre/`):

   ```yaml
   name: Publish intake battery
   on:
     push:
       branches: [main]          # or the branch you release from
       paths: ["bre/instrument/static/**"]
     workflow_dispatch:
   permissions:
     contents: read
     pages: write
     id-token: write
   jobs:
     deploy:
       runs-on: ubuntu-latest
       environment:
         name: github-pages
         url: ${{ steps.deployment.outputs.page_url }}
       steps:
         - uses: actions/checkout@v4
         - uses: actions/configure-pages@v5
         - uses: actions/upload-pages-artifact@v3
           with:
             path: bre/instrument/static
         - id: deployment
           uses: actions/deploy-pages@v4
   ```

3. Before the first push, edit `bre/instrument/static/config.js` (`STUDY_CONTACT`, and
   `ENDPOINT_URL` if you use an endpoint; see `static/README.md`).
4. The site appears at `https://<owner>.github.io/<repo>/`. Check
   `https://<owner>.github.io/<repo>/?form=short` end to end once, download the two files, and
   confirm the responses file has `n_presentations x 3` rows.

Alternatives: copy the folder into its own small repository and publish that from its root with
the same workflow (minus the `path`), or serve the folder from any static host. GitHub Pages
serves over HTTPS, which the endpoint POST and `crypto.randomUUID` both prefer.

If a project-site URL breaks the `vendor/` paths (it should not; all paths are relative), open the
browser console: the page shows any load error in a red box at the top.

## Running the unpaid pilot (20-50 volunteers)

Budget is $0 (CLAUDE.md), so this is a convenience sample of volunteers, and the report must say
so: it validates the instrument (timing, comprehension, replicability, data flow), it is not a
representative sample and it is not incentivized (`incentivized = false` on every row).

1. Freeze the battery: `battery_version` stays `1.0.0` for the whole pilot. Any wording change
   after the first volunteer means a new version and a new pilot; the version is on every row.
2. Recruit 20-50 adults through personal and professional networks. Do not recruit clients of any
   advisor, and do not record who took part beyond a count; the app stores no name or contact.
3. Send each volunteer the link with the form chosen for them: `?form=full` for most (target 8-12
   minutes), `?form=short` for anyone who cannot spare that. Ask them to do it in one sitting, on a
   laptop or phone, without a calculator, and to download the responses file at the end (or note
   the "sent" message if an endpoint is configured) and send the file to `STUDY_CONTACT`.
4. The consent page makes participation and training consent separate. Volunteers who decline
   training consent still complete the battery; their rows carry `consent_training = false` and
   must be excluded from any model fit, but they still serve the instrument checks (timing, order
   of pages, replicability of their own repeats).
5. Collect the files in `data/raw/intake_pilot/` (below). Ask volunteers for the session log too
   when they can; it holds the page timings and the measured delay durations.
6. Pre-specified handling, written here before any data is looked at:
   * A session is complete when its responses file has `n_presentations x 3` rows
     (42 for `full`, 15 for `short`); incomplete sessions are kept in `data/raw/` but not loaded.
   * Sessions whose log has `delay_override_active = true` are test runs and are excluded.
   * Duplicate files (same `session_id`) are loaded once.
   * Test-retest agreement is computed from the repeat rows (`source_row_ref` contains `rep_of`):
     sell/hold agreement and absolute allocation difference against the original item.
   * Speeding is flagged, not excluded, in the pilot: the flag threshold on median per-decision
     `response_time_ms` is `{{RT_FLAG_THRESHOLD_MS_FROM_PILOT}}`, to be set from the pilot's own
     distribution and then frozen for later collection.
7. What the pilot must answer before any paid or larger collection: median and spread of session
   duration for each form (`{{PILOT_MEDIAN_DURATION_MIN}}` -- fill in from the session logs),
   drop-off page, whether the 10-second delay page is tolerated, repeat agreement, and whether the
   endpoint path (if used) delivered every session.
8. Calibration target: the recovery study (Phase 2, PLAN.md section 5) yields the smallest N at
   which model selection is reliable; that figure is `{{N_TARGET_FROM_PHASE2}}` (subjects on the
   full design, the unit the transparency page uses). The pilot's 20-50 volunteers are below it
   by construction;
   the pilot is an instrument check, not a calibration set, and the dashboard's transparency page
   must say so if pilot data are ever used in a fit.

Paid collection (Prolific or similar) is a GATE under CLAUDE.md rule 4(a); `PROLIFIC.md` is a
separate Phase 7 deliverable and nothing here spends money.

## Getting the downloaded JSON into the pipeline

Expected file location: `bre/data/raw/intake_pilot/*.json`.

* `intake_<session_id>.json` -- the responses array (the file the pipeline reads).
* `intake_<session_id>.session.json` -- the session log (kept alongside; skipped by the loader).

A future loader `src/bre/loaders/intake_battery.py` (not written yet) reads every `*.json` in that
folder except `*.session.json`, and for each file:

1. parses the array and checks that every object has exactly the 21 schema fields in order
   (`battery.json -> output.schema_fields`), that `dataset == "intake_battery"`,
   `is_synthetic == false`, `incentivized == false`, `outcome_behavior is null`, and that
   `battery_version` is a version the loader knows (`"1.0.0"`);
2. converts to the parquet schema: `context_tags` and `prior_question_ids` stay list<string>,
   `covariates` becomes a JSON string, `outcome_behavior` stays null, `timestamp` is parsed as
   UTC, `response` stays float (allocation rows are already fractions in [0, 1]);
3. runs `validate_frame` (schema.py, Phase 0) -- range rules, duplicate
   `(subject_id, dataset, session_id, position_in_session)`, `is_synthetic` consistent with the
   file living under `data/raw/`, not `data/synthetic/`;
4. applies the pre-specified pilot handling above (complete sessions only, no
   `delay_override_active` sessions, duplicates once) and keeps `consent_training` on every row so
   that `fit.py` can drop `consent_training == false` rows from training sets;
5. writes `data/processed/intake_battery.parquet` and reports counts per form, per
   `scenario_id`, and the repeat-agreement summary.

Before the loader runs, the dataset needs its entry in `data/CATALOG.md` (CLAUDE.md rule 2): source
= this instrument, terms = volunteer consent as worded on the consent page, access date = pilot
dates, fields = the schema, mapping = identity, known problems = convenience sample, unpaid,
self-selected. Only the main session edits CATALOG.md.

Mapping notes the loader and the models rely on:

* `elicitation_type`: `sell_hold` rows are `binary_sell` (1 = sell); `allocation_share` rows are
  `allocation_pct` (fraction sold); tolerance rows are `lottery_choice` with 1 = "Yes" (prefers to
  avoid losses, the safer option). The schema enum has no dedicated binary-preference type; this
  mapping is an assumption recorded in `battery.json -> design.questions.tolerance.coding_note`
  for the main session to confirm or change (changing it means a new `battery_version`).
* `prior_question_ids` lists the questions already answered *within the same item*, in order, so
  the order effect is readable from each row: under `tolerance-first` the sell/hold row has
  `["tolerance"]`; under `scenario-first` the tolerance row has
  `["sell_hold", "allocation_share"]`.
* `question_order_id` and the ordered `context_tags` are also encoded in `scenario_id`, which is
  therefore unique per (loss, ordered contexts, order) and shared by an original and its repeat;
  `source_row_ref` tells them apart.
* Covariates are identical on every row of a session; "Prefer not to say" is `null`.

## Placeholder tokens in this file

Numbers that must come from data not yet in hand are written as tokens, not invented:

* `{{RT_FLAG_THRESHOLD_MS_FROM_PILOT}}` -- speeding flag threshold, from the pilot's response-time
  distribution.
* `{{PILOT_MEDIAN_DURATION_MIN}}` -- measured session duration per form, from the pilot's session
  logs.
* `{{N_TARGET_FROM_PHASE2}}` -- responses needed for calibration, from the Phase 2 recovery study.
