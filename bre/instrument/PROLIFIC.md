# PROLIFIC.md — shelved Prolific study design for the BRE intake battery

Status: SHELVED. This is a design document only. No study has been created on Prolific, no
participants have been recruited, and nothing has been spent. Per CLAUDE.md rule 4(a),
**no money is to be spent without the owner's explicit approval**; every step below that would
incur a cost is marked GATE and is performed only by the owner.

Written 2026-09-17 (session 1, Phase 7). Binding references: CLAUDE.md; PLAN.md §2 (schema), §3
(shared experimental design), §5 (recovery study, sample-size source), §6 (evaluation protocol);
`db/models.py` (`SCHEMA_COLUMNS`); and the built instrument `instrument/battery.json`
(`battery_version` 1.0.0) with its static jsPsych build under `instrument/static/`. Companion
documents: `instrument/IRB_CHECKLIST.md`, `instrument/PREREGISTRATION.md`.

The Prolific study runs the intake battery exactly as `battery.json` defines it, plus a small set of
Prolific-mode additions listed in §3. Where this document restates the battery, `battery.json` is
the source of truth; where it adds something, the addition is marked "override" and is not yet
implemented in the instrument (that is a main-session decision, PLAN.md §7 Phase 7).

Numbers in this file are of three kinds, each labelled: (i) design decisions fixed by this document
or by `battery.json` (instrument parameters, adjustable before launch, not data); (ii) arithmetic
from the cost formula in §9; (iii) placeholder tokens of the form `{{TOKEN}}` for every quantity
that must come from the Phase 2 recovery study, from a pilot, or from data not yet in hand. Every
token is listed in §14.

---

## 1. Purpose and what the study measures

The intake battery of the dashboard (PLAN.md §7, Phase 7) doubles as the data-collection
instrument. Run on Prolific, it produces the one kind of data no public dataset provides
(data/DATA_MAP.md): within subject, a manipulated loss × a manipulated context × a manipulated
question order, elicited as a sell/hold decision with an incentive-compatible payout, repeated
across time (wave 2) and across a real market shock (event-triggered wave).

Research questions served: RQ1 (context dependence, order effects, LTP violation), RQ2 (predictive
value on the four splits of PLAN.md §6; split (b), held-out context compositions, is produced by
design rather than through the CPC18 analog), RQ3 (external validity, through wave-2 self-reported
actions and the event wave; both are self-report, no brokerage data will exist).

## 2. Design as built in `battery.json` (full form)

| Factor | Levels | Manipulated | Schema field |
|---|---|---|---|
| Loss magnitude | −0.05, −0.10, −0.15, −0.20, −0.30 (`design.loss_pcts`) | within subject | `loss_pct` |
| Horizon | 365 days | fixed | `horizon_days` |
| Cause frame | `news:recession`, `news:technical` | within subject | `context_tags` |
| Social cue | `social:friend_sells` | within subject | `context_tags` |
| Partial recovery | `market:recovered_5pct` | within subject | `context_tags` |
| Context condition | 17 (`design.context_conditions`): `none`, 4 singles, 12 ordered pairs | within subject, balanced subset per participant | `context_tags` (ordered) |
| Question order | `tolerance-first`, `scenario-first` (`design.question_orders`), assigned per item | within subject | `question_order_id`, `prior_question_ids` |
| Timed delay page | shown on every item whose context includes a `news:` tag, `display_seconds` = 10 (`design.delay_page`) | fixed in the battery; optional two-level override in §3 | session log `delay_display_ms` |
| Repeated scenarios | 2 repeats per session (`design.repeated_scenario_rule`) | within subject | `source_row_ref` `rep_of_i..` |
| Elicitation | tolerance (yes/no), sell/hold, allocation slider (`design.questions`) | every item | `elicitation_type`, `response` |

Wording is fixed in `battery.json` (`design.scenario.template`, `design.contexts[*].display`,
`design.questions[*].text`) and is not restated here so that it cannot drift. In brief: the intro
places the participant a year after investing part of their long-term savings in a fund; the loss
sentence states the position is worth L less than what was paid; context sentences follow "Since
then, one more thing has happened:" (single) or "…two more things have happened, in this order:"
(pair) and are shown in the assigned order; the `market:recovered_5pct` sentence says the loss is
measured after a 5% recovery from the low.

Per item (one "presentation"), the page sequence is, for `tolerance-first`: tolerance question →
scenario page (loss + contexts) → delay page if a news tag is present → sell/hold → allocation
slider; for `scenario-first`: scenario page → delay page if news → sell/hold → allocation slider →
tolerance question. So the tolerance question is asked with every item, in one of the two orders,
which is what gives within-subject order data and per-item joint (tolerance, sell) answers.

Questions and coding (from `design.questions`):

* tolerance — "Would you describe yourself as someone who avoids investment losses even at the cost
  of lower returns?" Yes = 1 / No = 0. Stored as its own row with `elicitation_type =
  binary_yes_no` (the value the PLAN.md §2 amendment of 2026-09-17 added to the schema enum for
  this question; see §12).
* sell_hold — "What would you do with this position today?" Sell = 1 / Hold = 0,
  `elicitation_type = binary_sell`.
* allocation_share — "What share of this position would you sell?" Slider 0–100, step 1, start 50,
  movement required; `response = value / 100`, `elicitation_type = allocation_pct`.
  Note the direction: the slider is the share **sold**, so 1 − response is the share kept.

Full form composition (`forms.full`): 12 items = 4 `none` + 4 single + 4 ordered-pair conditions;
each of the four non-null contexts appears exactly once as a single; the four pairs come from a
random cyclic ordering of the four contexts (every context once in first and once in second
position, no unordered pair repeated); each loss level appears at least twice, and every loss level
occurs with both question orders (2 tolerance-first + 2 scenario-first per block, redrawn within a
bounded number of attempts, with `order_balance_satisfied` logged). Two repeats are exact copies of
two distinct items from presentation positions 1–6, each placed at least 6 presentations after its
original; 14 presentations in all. The assignment is a pure function of
(`battery_version`, form, seed) with seed = `session_id`, so it is reconstructible offline
(`static/battery.js buildAssignment`). `target_duration_min` of [8, 12] in `forms.full` is an
instrument parameter chosen by the author, not a measurement; the study's duration T is measured in
the pilot (§9).

Split (b) of PLAN.md §6 is available within every participant: train on the 8 `none` and single
items, test on the 4 ordered pairs. The 5-item short form is not used on Prolific.

## 3. Prolific-mode overrides (not in `battery.json` 1.0.0; each needs an instrument change)

O1 Identity. The battery generates a random UUID as `subject_id`. On Prolific the URL carries
`PROLIFIC_PID`, `STUDY_ID`, `SESSION_ID`; the page keeps `PROLIFIC_PID` in memory only and sends it
with the session payload to the owner-hosted intake endpoint (Phase 5), which replaces it by
`HMAC-SHA256(PROLIFIC_PID, salt)` before anything is stored (§11). `subject_id` is that hash, so
waves link within participant.

O2 Provenance fields. `dataset = "prolific_bre"` (not `intake_battery`), `incentivized = true`,
`session_id = "<wave>:<seed>"` with wave ∈ {`pilot`, `w1`, `w2`, `e<YYYYMMDD>`} so both the wave
and the reconstructible seed are in the uniqueness key; `battery_version` carries a
`+prolific` build suffix.

O3 Transport. The battery offers download buttons; the Prolific build instead POSTs the responses
JSON and the session-log JSON to the intake endpoint on the final page and shows the Prolific
completion code only after a 2xx. Nothing is sent before the final page (as in the battery), so an
abandoned session leaves no rows.

O4 Consent and instructions. The battery's consent says the questionnaire is unpaid. The Prolific
build uses the consent text of `IRB_CHECKLIST.md` §6, then the payout explanation (§5) and a
comprehension check (§8) before the covariates and the first scenario. The battery's instruction
that "some questions repeat across scenarios" is kept.

O5 Attention checks. Three instructed-response items in scenario format inserted at random among
the 14 presentations (§8); not paid, not stored as decision rows.

O6 Payout. The random-lottery selection, the historical-path draw, the bonus computation and the
debrief disclosure of §5, all on the final page, with the drawn episode logged in the session log.

O7 Wave questions. Wave 2 and the event wave prepend the self-report questions of §6 and write
`outcome_behavior`.

O8 Delay factor (optional, off by default). The battery shows a fixed 10 s delay page on news
items. Optionally the Prolific build assigns 0 s to half of a participant's news items (randomly,
balanced) and keeps 10 s on the rest, so that the Q3 dynamical model (PLAN.md §4) gets within-subject
variation in time since news. If enabled, `delay_s` is written per row into `covariates` (the
battery otherwise keeps `covariates` identical on every row of a session) and the analysis in
`PREREGISTRATION.md` §9 becomes applicable. Enabling it is a design decision for the owner; the
default study uses the battery as built.

## 4. Per-participant flow on Prolific (wave 1)

Prolific redirect → consent (O4) → investment-ownership screen (§10) → payout explanation →
comprehension check → covariates and financial-literacy quiz (`intake.covariates`,
`intake.financial_literacy_quiz`) → 14 presentations with 3 embedded attention checks → training-use
consent checkbox (`consent_training`) → payout selection and debrief (O6) → POST (O3) → completion
code.

Every presentation is one incentivized decision under §5, including the repeats, and the
participant does not know which presentations are repeats or which one will be paid.

## 5. Incentive-compatible payout from historical S&P 500 drawdown-and-recovery paths

### 5.1 Principle

Every scenario decision is a real decision about a real stake. At the end of the session one
presentation is selected at random (random-lottery incentive system), the participant's decision on
it is applied to a real historical price path of the S&P 500 drawn from a documented table of
drawdown episodes, and the bonus paid is the stake multiplied by the realized outcome of that
decision on that path. Because the participant does not know which presentation will be paid, and
every one is paid by the same rule, the dominant strategy is to answer each as if it were the one
that counts.

Participants cannot lose money: the bonus is a positive-only payment on top of the base pay. The
"loss" in the scenario is the state the historical path is in when the decision is applied, not a
deduction from the participant's earnings. This is stated in the consent text and the instructions.

### 5.2 Mechanism, precisely

Let B = `{{BONUS_STAKE_USD_OWNER}}` be the bonus stake in USD (an owner decision, not set here).
Let H = 365 calendar days be the horizon (PLAN.md §3, `design.horizon_days`).

1. Selection. After the last presentation, one index k is drawn uniformly from the 14
   presentations (attention-check items excluded). A fair coin then decides whether the `sell_hold`
   or the `allocation_share` answer of presentation k is executed. The tolerance answer is never
   paid.
2. Path draw. Presentation k has loss level L_k and ordered context tags C_k. The path table (§5.3)
   has one row set per entry rule and threshold; the entry rule is:
   * `at_drawdown(L)` if `market:recovered_5pct` ∉ C_k: episodes at which the index's drawdown from
     its prior running peak first reached ≤ L;
   * `after_5pct_rebound(L)` if `market:recovered_5pct` ∈ C_k: episodes at which, after the drawdown
     had reached ≤ L − 0.05, the close first rose ≥ 5% above the running minimum close since that
     date (matching the battery's sentence that the stated loss is measured after the recovery).
   One episode e is drawn uniformly from the applicable row set. The draw does not depend on the
   cause frame or the social cue; the instructions say so ("the news and the friend in a scenario
   are part of the story; the payout uses a real historical episode matched to the size of the loss
   and, where the scenario says the fund has partly recovered, to that recovery").
3. Realized multiplier. With P(t0) the closing level on the episode's entry date t0 and P(t0 + H)
   the closing level on the first trading day on or after t0 + H:
   * M_hold = P(t0 + H) / P(t0);
   * M_sell = 1 (converted to cash at t0; no interest credited);
   * `sell_hold` executed: M = M_hold if the answer was Hold, M = M_sell if Sell;
   * `allocation_share` executed, with s = share sold (the stored `response`):
     M = s · 1 + (1 − s) · M_hold.
4. Payout. bonus = B × M, rounded to the cent, paid through Prolific's bonus facility. No cap and no
   floor: M can be below 1 (a further fall during the year) or above 1. If the owner requires a cap
   for budgeting, it must be stated to participants before the first decision and recorded in
   `battery_version`; a cap weakens incentive compatibility for large recoveries and the report must
   say so.
5. Disclosure. The debrief page shows the selected presentation, the executed answer, the episode's
   entry date, P(t0), P(t0 + H), M and the bonus, and links the complete episode table.

### 5.3 The path table — to be built from a public price series; nothing in it is invented here

`instrument/path_table.csv` does not exist yet and no episode dates or returns appear in this
document. It is to be built by a script (`instrument/build_path_table.py`, Phase 7, not yet written)
from a public daily closing-price series of the S&P 500 once the owner has downloaded it. Yahoo
Finance, FRED and the other market-data hosts are unreachable from the build session (CLAUDE.md
environment notes), so the download is an owner action; the series is catalogued in
`data/CATALOG.md` (source URL, terms, access date, first and last date, number of rows, SHA-256)
before the script is run.

Algorithm (deterministic given the series):

1. Input: trading-day closes P(t), one row per trading day, price index (dividends excluded). If a
   public total-return series is available the owner may substitute it; the choice is recorded and
   the report notes that a price index understates M_hold relative to a total-return investment.
2. Running peak: peak(t) = max_{s ≤ t} P(s). Drawdown: dd(t) = P(t)/peak(t) − 1.
3. A cycle starts at a new all-time high and ends at the next one. Within a cycle, for each
   threshold L ∈ {−0.05, −0.10, −0.15, −0.20, −0.30}, the `at_drawdown(L)` entry is the first day
   with dd(t) ≤ L; at most one entry per threshold per cycle, so entries within a threshold do not
   overlap in their starting state.
4. `after_5pct_rebound(L)`: within a cycle, let t_L be the `at_drawdown(L − 0.05)` entry; the entry
   is the first day t > t_L with P(t) ≥ 1.05 · min_{t_L ≤ s ≤ t} P(s) and dd(t) < 0 (the cycle has
   not ended); at most one per threshold per cycle.
5. An entry is kept only if the series extends at least H calendar days beyond it; the last year of
   the series therefore yields no entries.
6. Each row records: rule, L, cycle start date, t0, P(t0), dd(t0), t0 + H (first trading day on or
   after), P(t0 + H), M_hold, and, for the rebound rule, the trough date and trough close.
7. Output: `instrument/path_table.csv` with a header comment naming the input series, its SHA-256
   and the build date, and `instrument/path_table.md` listing the number of episodes per rule and
   threshold. Those counts are `{{N_EPISODES_PER_RULE_AND_THRESHOLD}}` until built; if any row set is
   empty the study cannot pay that threshold and the design must be revised before launch (the
   script checks this and fails loudly).
8. The expected multiplier per row set, needed for the bonus budget in §9, is
   `{{MEAN_M_HOLD_PER_ROW_SET_FROM_PATH_TABLE}}`.

The script gets a pytest: on a synthetic series with a known drawdown structure it recovers the
known entries; on the real series it checks that every t0 satisfies its rule and that no two entries
of the same rule and threshold fall in the same cycle.

## 6. Waves

### 6.1 Wave 1 (main collection)

Single Prolific study, flow as in §4, `session_id = "w1:<seed>"`. Question order is within
subject and per item (§2), so wave 1 alone yields both orders for every participant, the per-item
joint (tolerance, sell) answers the QQ-equality test needs, and δ_LTP (PLAN.md §4) within subject.

### 6.2 Wave 2 (2–4 weeks after wave 1)

Same participants, re-invited through a Prolific study restricted to a custom allowlist of wave-1
Prolific IDs (this is why the ID-to-hash mapping of §11 must exist until wave 2 closes). Each
participant becomes eligible 14 days after their wave-1 completion and the invitation expires 28
days after it (design decision). Content: the same battery with a fresh seed (a new balanced
assignment; the repeat rule applies again), `session_id = "w2:<seed>"`, and three questions asked
before the scenarios (O7):

* "Since you took part in our earlier study on [date], did you sell any investments?" (yes / no /
  I hold no investments), "buy any?", and "roughly what share of your investable assets did the
  largest such transaction involve?" (bands). Stored as `outcome_behavior` on every wave-2 row:
  `{"action": "sold"|"bought"|"both"|"none"|"no_investments", "lag_days": <days since wave 1>,
  "self_reported": true, "share_band": <band>}`. These are self-reports and the report says so.

Payment: the same formula as wave 1 with wave 2's own measured duration. The retention rate is
unknown until wave 1 runs: `{{WAVE2_RETENTION_FROM_WAVE1}}`.

### 6.3 Event-triggered wave (fired within 48 hours of a > 5% weekly fall in the S&P 500)

Trigger definition. Let F(w−1) be the closing level of the S&P 500 on the last trading day of the
previous calendar week. On every trading day t of the current week compute r(t) = P(t)/F(w−1) − 1.
The trigger fires at the first close with r(t) < −0.05. Checking every close rather than only
Friday's means a mid-week fall fires immediately; either way the launch deadline is 48 hours after
the close that tripped it, so the wave is in the field within 48 hours of any week in which the
index fell more than 5%.

Who runs it. The owner. A script `instrument/event_trigger.py` (Phase 7, not yet written) is run by
the owner on a machine that can reach a quote source (the build session cannot). It takes the day's
close, either fetched by the owner (yfinance `^GSPC`, or any documented public source) or typed in,
appends `date, close, F(w-1), r, fired` to `runs/event_trigger.log`, and prints `TRIGGER` or
`no trigger`. The script never contacts Prolific, never creates a study and never spends money;
launching the event wave is a manual owner action and a GATE under CLAUDE.md rule 4(a). Nothing in
the pipeline is automated past the printed word.

Rules (design decisions):

* Launch within 48 h of the triggering close; collection window 72 h from launch.
* Sample: the wave-1 allowlist (within-subject comparison of hypothetical vs live drawdown), plus a
  fresh sample of the same size only if the owner approves the additional cost.
* Cooldown: at most one event wave per 30 calendar days; no event wave in the 14 days after a
  participant's wave 1 (their wave-2 window takes precedence).
* Content: the wave-1 battery with a fresh seed and, before the scenarios (O7), the participant's own
  current situation ("Do you currently hold stock investments?", "Have you sold any in the past
  7 days?", "Do you intend to sell any in the next 7 days?"), stored as `outcome_behavior` with
  `lag_days` = days since wave 1. Every row carries `context_tags` prefixed with
  `market:live_drawdown` and `covariates.market_state = {"index_close": P(t), "weekly_return":
  r(t), "drawdown_from_peak": dd(t), "as_of": date}` copied from the trigger log, so the live context
  is recorded from the same numbers that fired the trigger.
* `session_id = "e<YYYYMMDD>:<seed>"` with the trigger date.
* The invitation text is neutral, gives no financial advice and does not cite the market fall as the
  reason (`IRB_CHECKLIST.md` §3).

Whether any event wave ever runs depends on the market; the design does not assume one.

## 7. Covariates (`battery.json intake.covariates`, PLAN.md §3; bands only, no free text)

`age_band` {18-29, 30-44, 45-59, 60+}; `wealth_band` {<50k, 50-250k, 250k-1M, >1M}; 
`invest_experience_yrs` (0–40); `self_reported_risk_tolerance` (1–7); `financial_literacy_score`
(0–5, number of correct answers on the battery's quiz, "I don't know" counted as incorrect);
`education` {hs, some_college, bachelor, graduate}. "Prefer not to say" is stored as null
(`intake.covariate_null_rule`). The battery keeps `covariates` identical on every row of a session;
the Prolific build adds the session-level keys `wave`, `attention_pass_count`,
`comprehension_attempts`, `manipulation_check` and, on event-wave rows, `market_state`; with O8
enabled it also adds the per-row key `delay_s`. Per-page response times, `delay_display_ms`,
`order_balance_satisfied` and the drawn payout episode live in the session log, not in the schema.

## 8. Attention checks, comprehension check, manipulation check (O4, O5)

* Comprehension check (gate, before the covariates): three multiple-choice questions on the payout
  rule ("If the paid decision is one where you chose Hold, what determines your bonus?", "Can your
  bonus be negative?", "Does the news or the friend in a scenario change which historical episode
  is used?"). Two attempts; a second failure ends the session with the screen-out payment
  Prolific's rules prescribe, and the participant is not counted toward N.
* Three instructed-response attention checks in scenario format at random positions among the 14
  presentations ("This item checks that you are reading. Choose Sell and set the slider to 100%").
  Not paid, not stored as decision rows; `attention_pass_count` (0–3) is a covariate. Pre-registered
  rule: 0 or 1 passes → excluded from analysis; Prolific rejection only where Prolific's
  attention-check policy permits, and rejection decisions are separate from analysis exclusions.
* Manipulation check (not an exclusion criterion): after the last presentation, "Which of these
  reasons for the fall did you read during the study?" (multi-select over the two cause frames plus
  two distractors). Recorded in `covariates.manipulation_check`.
* Straight-lining flag: identical slider value on every presentation, or the same sell/hold answer
  on every presentation with a median item response time below the floor of §10. Flag stored;
  exclusion per §10.

## 9. Sample size and cost

### 9.1 Sample size

Target number of participants passing all exclusions:

    N_target = {{N_TARGET_FROM_PHASE2}}

This number comes from the Phase 2 recovery study (PLAN.md §5): it is the smallest N in
{100, 200, 400, 800} at which the model-selection confusion matrix, over 3 generators × 5 seeds,
shows ≥ 90% correct selection among the fitted models, i.e. the "responses needed for calibration"
figure that the transparency page will display. Until `make recover` has run there is no target;
the study is shelved in any case. Wave 1 recruits

    N_recruit = ceil( N_target / (1 − e) ),  e = {{EXCLUSION_RATE_FROM_PILOT}}

where e is the fraction of pilot completions excluded under §10.

### 9.2 Cost formula

Let T be the battery's estimated median duration in minutes, measured in the pilot from Prolific's
completion-time report (the battery's `target_duration_min` [8, 12] is an instrument target, not a
measurement):

    T = {{T_BATTERY_MIN_FROM_PILOT}}

Base-pay cost of a wave of N participants, at the $12/hr rate fixed in the owner's brief and with
Prolific's service-fee factor as given in the brief:

    Cost_academic(N, T)  = T/60 × $12/hr × N × 1.333     (academic or non-profit rate)
    Cost_corporate(N, T) = T/60 × $12/hr × N × 1.428     (corporate rate)

Both the hourly rate and the fee factors are to be checked against Prolific's current published
schedule at launch; taxes, if any, are not included. Bonus cost is on top of this:

    Bonus_cost(N) = fee_factor × N × B × E[M],  E[M] = {{MEAN_M_HOLD_PER_ROW_SET_FROM_PATH_TABLE}}
    (averaged over the item mix and the sell_hold/allocation coin; M_sell = 1 bounds the sell branch)

Total for the study = wave 1 + wave 2 (N × {{WAVE2_RETENTION_FROM_WAVE1}}, its own T) + any event
wave, each evaluated with the same formula; plus the pilot (§13).

### 9.3 Evaluation of the base-pay formula (arithmetic only, not data)

These rows evaluate the formula at the two durations the brief asks for. They are not estimates of
T, which is a pilot measurement.

At T = 10 min, base pay per participant = 10/60 × $12 = $2.00:

| N | base pay (N × $2.00) | × 1.333 academic / non-profit | × 1.428 corporate |
|---|---|---|---|
| 100 | $200.00 | $266.60 | $285.60 |
| 200 | $400.00 | $533.20 | $571.20 |
| 400 | $800.00 | $1,066.40 | $1,142.40 |
| 800 | $1,600.00 | $2,132.80 | $2,284.80 |

At T = 12 min, base pay per participant = 12/60 × $12 = $2.40:

| N | base pay (N × $2.40) | × 1.333 academic / non-profit | × 1.428 corporate |
|---|---|---|---|
| 100 | $240.00 | $319.92 | $342.72 |
| 200 | $480.00 | $639.84 | $685.44 |
| 400 | $960.00 | $1,279.68 | $1,370.88 |
| 800 | $1,920.00 | $2,559.36 | $2,741.76 |

(Computed as N × T/60 × 12 × factor, rounded to the cent; e.g. 800 × 2.40 × 1.428 = 2,741.76.)

## 10. Eligibility and exclusion criteria

Prolific prescreeners (design decisions, applied at recruitment):

* age 18 or over; current country of residence United States (the payout series and the covariate
  marginals — FINRA NFCS, data/CATALOG.md entry F — are US); fluent in English;
* Prolific approval rate ≥ 95% and at least 20 previous submissions;
* one submission per person across all waves (Prolific enforces per study; the allowlist enforces
  it across waves);
* investment ownership: Prolific's investment prescreener if one is available at launch; otherwise
  the first screen asks "Do you currently, or did you in the past five years, hold stocks, funds or
  a retirement account invested in the stock market?" and screens out on "No" with the screen-out
  payment Prolific prescribes. Screened-out participants are not counted toward N and leave no rows.

Analysis exclusions (pre-registered; applied before any model fit; counts reported):

1. comprehension check failed twice (session ended, §8);
2. fewer than 2 of 3 attention checks passed;
3. median item response time below `{{RT_FLOOR_MS_FROM_PILOT}}` ms, set as the 2.5th percentile of
   pilot item response times;
4. straight-lining flag (§8) together with criterion 3;
5. incomplete session (no completion code): nothing was sent, so nothing is stored;
6. duplicate hashed `subject_id` within a wave (second and later sessions dropped);
7. any participant who withdraws (§11): all rows deleted.

No exclusion is based on the content of the decisions themselves.

## 11. Data handling — no PII, hashed Prolific IDs

* The instrument is the static jsPsych build under `instrument/static/` (jsPsych vendored from npm,
  no CDN), hosted by the owner; in Prolific mode it POSTs each completed session to the Phase 5
  FastAPI intake endpoint (O3), the only component that ever sees the raw `PROLIFIC_PID`.
* `subject_id = HMAC-SHA256(PROLIFIC_PID, salt)`, hex. The salt is generated once by the owner, kept
  outside the repository and outside the data directory, and never logged. Without it the hash
  cannot be reversed and a participant cannot be re-identified from the dataset.
* Re-invitation for wave 2 and the event wave needs the raw IDs. The endpoint therefore also writes
  an encrypted mapping file (raw ID → hash) on the owner's machine only. It is deleted when the last
  wave closes; from then on no one, including the owner, can link a row to a Prolific account. The
  consent text says that withdrawal after that date is impossible because the data are no longer
  identifiable.
* Not stored, at any layer: name, e-mail, IP address, geolocation, free text. The battery's session
  log records the user agent; the Prolific build drops that field before storage. Demographics are
  collected only as the bands of §7.
* `timestamp` is stored at second precision because the event wave needs it; it is not PII.
* Prolific's own records (payments, messages) stay on Prolific under its terms; the study's export
  from Prolific (which contains raw IDs) is stored encrypted with the mapping file and deleted with it.
* `consent_training` records the separate opt-in for using anonymised responses to train the
  dashboard's models; rows with `consent_training = false` are used for the research report only.
* `incentivized = true` on every scenario row (O2); `is_synthetic = false`; `battery_version` is the
  battery's version plus the `+prolific` suffix; `source_row_ref` follows
  `output.source_row_ref_format` (`full:i03:sell_hold`, `full:i13:rep_of_i02:tolerance`).
* Every ingested frame passes `validate_frame` (PLAN.md §2) before it is written to
  `data/processed/prolific_bre.parquet` and to the `responses` table.

## 12. Schema mapping (PLAN.md §2, `db/models.py::SCHEMA_COLUMNS`; `battery.json output.row_rules` with the overrides of §3)

| Column | Value for this study |
|---|---|
| `subject_id` | HMAC-SHA256 of the Prolific ID (O1) |
| `dataset` | `"prolific_bre"` (O2) |
| `session_id` | `"<wave>:<seed>"`, wave ∈ {pilot, w1, w2, e<YYYYMMDD>}, seed = the battery's 32-hex PRNG seed (O2) |
| `timestamp` | ISO-8601 UTC at the moment the response was given |
| `position_in_session` | 0-based counter over rows in the order answered (tolerance, sell_hold, allocation_share rows of a presentation are consecutive in page order) |
| `scenario_id` | battery format `L{loss_pct:.2f}|{tags joined by '>' or 'none'}|{question_order_id}`, e.g. `L-0.15|news:recession>social:friend_sells|tolerance-first` |
| `loss_pct`, `horizon_days` | the item's loss level; 365 |
| `context_tags` | the item's ordered tags; `[]` for `none`; event wave prepends `market:live_drawdown` |
| `question_order_id` | the item's order |
| `prior_question_ids` | question ids already answered within the same presentation, in order (battery rule; see the open point below) |
| `elicitation_type` / `response` | tolerance: `binary_yes_no`, 1 = Yes; sell_hold: `binary_sell`, 1 = Sell; allocation_share: `allocation_pct`, share sold in [0, 1] |
| `response_time_ms` | question shown to answer given |
| `covariates` | §7 |
| `outcome_behavior` | null in pilot and wave 1; the self-report object of §6.2 / §6.3 otherwise |
| `incentivized` | true (O2) |
| `consent_training` | the participant's opt-in |
| `battery_version` | `1.0.0+prolific` (or later) |
| `is_synthetic` | false |
| `source_row_ref` | battery format |

Open points for the main session (only the main session edits the schema):

* Resolved 2026-09-17: the tolerance answer is stored as a `binary_yes_no` row with 1 = Yes
  (PLAN.md §2 amendment; `battery.json output.elicitation_types_used`). The `lottery_choice`
  stand-in is gone from the instrument; any other table that still uses it for yes/no items
  (e.g. the dataset-D order tables in data/CATALOG.md) is for the main session to migrate.
* `db/models.py` documents `prior_question_ids` as "questions asked before this one in the session";
  `battery.json` restricts it to the same presentation. Either is workable for the Q2 Lüders step
  (which needs to know whether the tolerance question preceded the sell question within the item);
  the loader for this dataset follows whatever the main session decides and the choice is recorded
  in data/CATALOG.md.

## 13. Pilot and launch sequence (every step with a cost is a GATE)

1. Implement the overrides of §3 behind a Prolific-mode flag in the instrument; run the acceptance
   tests offline; three internal dry runs by the owner. No cost.
2. Build the path table from the owner-downloaded series (§5.3); verify no empty row set. No cost.
3. IRB / ethics review per `instrument/IRB_CHECKLIST.md`; register `instrument/PREREGISTRATION.md`
   at a public registry with a timestamp before step 4. No cost (registry) or the board's fee (GATE).
4. GATE — pilot: 20 participants (design decision), wave-1 flow. Outputs: T (median duration),
   e (exclusion rate), the RT floor, bug list. Pilot rows use `session_id = "pilot:<seed>"` and are
   excluded from confirmatory analyses.
5. Fill the tokens of §14 from Phase 2 and the pilot; recompute §9; obtain the owner's written
   approval of the total budget.
6. GATE — wave 1. GATE — wave 2 (scheduled per participant). GATE — event wave (only on a printed
   `TRIGGER`, and only with a fresh approval).
7. Ingest, validate, freeze the analysis code commit, then run Phase 4 per `PREREGISTRATION.md`.

## 14. Placeholder tokens used in this file

Every number that depends on Phase 2, a pilot, or data not in hand is one of these tokens. None has
a value yet.

| Token | Where it comes from |
|---|---|
| `{{N_TARGET_FROM_PHASE2}}` | Phase 2 recovery study (PLAN.md §5): smallest N with ≥ 90% correct model selection |
| `{{EXCLUSION_RATE_FROM_PILOT}}` | pilot: fraction of completions excluded under §10 |
| `{{T_BATTERY_MIN_FROM_PILOT}}` | pilot: median completion time in minutes from Prolific's report |
| `{{RT_FLOOR_MS_FROM_PILOT}}` | pilot: 2.5th percentile of item response times |
| `{{BONUS_STAKE_USD_OWNER}}` | owner decision on the bonus stake B (a spending decision, not made here) |
| `{{N_EPISODES_PER_RULE_AND_THRESHOLD}}` | `build_path_table.py` on the owner-downloaded S&P 500 series |
| `{{MEAN_M_HOLD_PER_ROW_SET_FROM_PATH_TABLE}}` | same script; expected hold multiplier per row set, for the bonus budget |
| `{{WAVE2_RETENTION_FROM_WAVE1}}` | wave 1: share of wave-1 participants who complete wave 2 |

Design decisions fixed by this document or by `battery.json` (adjustable before launch; not data):
full form of 12 items (4 none / 4 single / 4 pair) + 2 repeats from positions 1–6 with gap ≥ 6;
per-item question order, 2 + 2 per block; delay page 10 s on news items (optional 0/10 s factor,
O8); slider start 50 with required movement; 3 attention checks with the ≥ 2-of-3 rule;
comprehension check with 2 attempts; wave-2 window days 14–28; event trigger r(t) < −0.05 vs the
prior week's last close, launch ≤ 48 h, window 72 h, cooldown 30 days; pilot of 20; US residents,
approval ≥ 95%, ≥ 20 prior submissions; horizon 365 days; H = 365 and the 5% rebound rule for the
path table.
