# IRB_CHECKLIST.md — exempt-review checklist for the BRE Prolific study

Status: NO ETHICS APPROVAL EXISTS. Nothing has been submitted to any board. This is a checklist the
owner can take to an institutional review board (IRB) or an independent ethics board when, and if,
the shelved study in `instrument/PROLIFIC.md` is un-shelved. Whether the study qualifies for exempt
review is the board's decision, not this document's. Written 2026-09-17.

How to use: tick each box only when the item is true and documented. Fields marked `____` are for
the owner to fill. Tokens of the form `{{TOKEN}}` are quantities not yet known (listed in §13).
Nothing may be recruited or paid before the board's written determination and the owner's explicit
budget approval (CLAUDE.md rule 4(a)).

---

## 1. Study identification

- [ ] Title: "Context-dependent sell/hold decisions after portfolio losses: an online decision study"
      (working title; edit freely).
- [ ] Principal investigator: ____ ; institution or affiliation: ____ ; contact e-mail: ____ .
- [ ] Board: ____ (institutional IRB / independent ethics board); submission route: ____ .
- [ ] Study type: online behavioral decision study with survey items, single site (Prolific),
      up to three sessions per participant (wave 1, wave 2 at 2–4 weeks, and a possible
      event-triggered wave; `PROLIFIC.md` §6).
- [ ] Funding: none / self-funded / ____ . Budget approval by the owner attached (or "not yet").
- [ ] Commercial interest disclosed: the anonymised responses may be used to train the models of an
      advisor-facing dashboard being built by the PI (`PROLIFIC.md` §11, `consent_training`).
      Participants opt in to that use separately from research use.

## 2. Basis for requesting exempt review (board decides)

Candidate categories under the US Common Rule (45 CFR 46.104(d)), to be confirmed by the board:

- [ ] (d)(2) research involving survey procedures / educational tests, where the information is
      recorded such that the identity of subjects cannot readily be ascertained (hashed Prolific
      IDs, no PII; §5 below), or where disclosure would not reasonably place subjects at risk.
- [ ] (d)(3)(i) benign behavioral interventions with adult subjects who prospectively agree, with
      information collected via survey-type responses. The "intervention" is the presentation of
      hypothetical loss scenarios with framing text; it is brief, harmless, painless, not offensive
      or embarrassing, and there is no deception (§4).
- [ ] If the board is outside the US, or if any participant could be outside the US: the owner
      identifies the equivalent local category and any data-protection requirement (e.g. whether a
      data-protection impact assessment is required). The design restricts recruitment to US
      residents (`PROLIFIC.md` §10), which the board should be told.
- [ ] Limited-IRB-review question: because the hashed IDs and the temporary ID mapping (§5) exist,
      ask the board whether limited review of the privacy safeguards is required.

## 3. Minimal risk

- [ ] Procedures: reading short hypothetical investment scenarios; clicking sell/hold; setting a
      slider (share sold); answering, with each scenario, a yes/no question about one's attitude to
      losses; a 10-second timed pause page on scenarios that mention news; banded demographic and
      financial questions; a five-item financial-literacy quiz; three attention checks; a
      comprehension check. Screen texts are those of `instrument/battery.json`.
- [ ] No physical procedures, no physiological measurement, no recording of audio/video/screen.
- [ ] No deception: every statement shown to participants is true, including the payout rule and
      the fact that the scenario "story" does not affect which historical episode is used
      (`PROLIFIC.md` §5.2). The board is given the full instruction text.
- [ ] Participants cannot lose money: the bonus is a positive-only payment; the "loss" is the state
      of a historical price path, never a deduction from earnings. Stated in consent and instructions.
- [ ] Emotional risk: scenarios describe hypothetical losses; the study does not ask about the
      participant's own losses in detail (wave-2 and event-wave items ask only whether they bought
      or sold, and a band for the size). Participants who find the topic distressing can stop at any
      time (§7). No financial advice is given anywhere in the study; a fixed disclaimer sentence
      appears in the debrief.
- [ ] Event-triggered wave: participants may be invited during a period of real market stress. The
      invitation is neutral (it does not cite the market fall as the reason), the content is the same
      battery, and the timing is disclosed in wave-1 consent ("we may invite you again within a few
      weeks, and possibly at a time of our choosing during the year"). Ask the board whether this
      needs a separate consideration.
- [ ] Sensitive categories: none targeted. Wealth is asked in four bands with "prefer not to say".
- [ ] Duration: `{{T_BATTERY_MIN_FROM_PILOT}}` minutes per session, measured in the pilot.
- [ ] Vulnerable populations: none recruited; no prisoners, children, or people unable to consent.

## 4. Adult participants and recruitment

- [ ] Prolific prescreen: age 18+; US residence; English fluency; approval rate and prior
      submissions thresholds as in `PROLIFIC.md` §10.
- [ ] Consent page re-confirms age ("I am 18 or older") as a required checkbox.
- [ ] Recruitment text (the Prolific study description) is attached and matches the consent text:
      what the task is, the duration, base pay, that a bonus depends on a random real historical
      market outcome, that there may be follow-up invitations.
- [ ] Screening: an investment-ownership screen at the start; screened-out participants receive the
      screen-out payment Prolific prescribes and leave no data other than the screen answer, which is
      not stored.
- [ ] Sample size: `{{N_TARGET_FROM_PHASE2}}` analysable participants (source: Phase 2 recovery
      study, PLAN.md §5), recruited as `N_recruit = ceil(N_target / (1 − {{EXCLUSION_RATE_FROM_PILOT}}))`,
      plus a pilot of 20 (design decision), plus wave 2 and any event wave (same people).

## 5. Anonymisation and data minimisation

- [ ] Collected: decisions, slider values, response times, the tolerance answer, banded
      demographics, banded investable assets, years of experience (0–40), risk tolerance (1–7),
      literacy score (0–5), education band, attention/comprehension counts, wave-2/event-wave
      self-reported buy/sell (yes/no and a size band), timestamps.
- [ ] Not collected: name, e-mail, postal address, IP address, device fingerprint, geolocation,
      free-text responses, any account or brokerage data.
- [ ] Prolific ID handling: the raw ID reaches only the owner-hosted intake endpoint, which stores
      `HMAC-SHA256(PROLIFIC_PID, salt)` as `subject_id`. The salt is generated once, kept outside the
      repository and the data directory, never logged (`PROLIFIC.md` §11).
- [ ] ID mapping for re-invitation: an encrypted file (raw ID → hash) exists on the owner's machine
      only, solely to invite the same people to wave 2 and the event wave. It is deleted on the day
      the last wave closes; the deletion date is recorded in `STATUS.md`. After deletion the dataset
      is not re-identifiable by anyone.
- [ ] Prolific's own export (contains raw IDs and payment records) is stored encrypted alongside the
      mapping and deleted with it; Prolific's platform records are governed by Prolific's terms.
- [ ] Timestamps are kept at second precision (needed for the event wave and response-time models);
      the board is told they are not combined with any identifier after the mapping is deleted.
- [ ] Data are stored in the project's SQLite database and parquet files on owner-controlled
      storage; access limited to the PI (and named collaborators: ____ ).
- [ ] Public release plan for the anonymised dataset (hashed IDs, bands only): ____ (yes/no; if yes,
      the consent text says so and the board is asked whether the banded covariates plus timestamps
      constitute a re-identification risk).

## 6. Informed consent — draft text

Shown as the first page; a required checkbox for each bracketed item; the participant cannot proceed
without all of them. Fill the fields before submission.

> **Consent to take part in a research study**
>
> Who is running this study: ____ (PI), ____ (affiliation), ____ (contact e-mail).
> Ethics review: ____ (board name and reference number, once issued).
>
> What the study is about: how people decide whether to sell or hold an investment after it has
> lost value, and how that decision depends on the surrounding circumstances.
>
> What you will do: read short descriptions of hypothetical situations in which an investment has
> fallen in value, and for each one decide whether you would sell or hold, and what share you would
> sell. With each scenario you will also answer a short question about your attitude to losses;
> some questions repeat across scenarios. There is a short financial-knowledge quiz and a few
> background questions asked in broad ranges (age range, education, range of investable assets,
> years of investing experience). Some items check that you are reading carefully. The session takes
> about {{T_BATTERY_MIN_FROM_PILOT}} minutes.
>
> Payment: you receive a base payment of ${{BASE_PAY_USD_FROM_T}} through Prolific for completing
> the session. In addition, one of your decisions will be selected at random and applied to a real
> historical period of the US stock market (the S&P 500 index). Your bonus equals a stake of
> ${{BONUS_STAKE_USD_OWNER}} multiplied by how that decision would have turned out over the following
> twelve months in that historical period. The bonus can be larger or smaller than the stake but is
> never negative: you cannot lose any of your own money. The stories shown with the scenarios do not
> change which historical period is used; only the size of the loss and, where stated, a partial
> recovery do. The full rule is explained again before your first decision.
>
> Follow-up: we may invite you, through Prolific, to a second session two to four weeks from now,
> and possibly to one further session at a time of our choosing within the next year. Each
> invitation is optional and paid separately. To send invitations we keep a link between your
> Prolific ID and your coded study ID, encrypted, on the researcher's computer only. That link is
> deleted when the last session closes.
>
> Risks: we do not expect any risk beyond those of everyday life. The scenarios are hypothetical.
> Nothing in the study is financial advice.
>
> Privacy: we do not collect your name, e-mail, IP address or any account information. Your Prolific
> ID is replaced by a one-way code before storage. Background questions are recorded only in broad
> ranges. [If applicable:] The anonymised data may be shared publicly for research.
>
> Optional use for model training: separately from the research use, you may allow your anonymised
> answers to be used to train the decision models of a financial-advice software tool being built
> by the researcher. You choose this on the last page; saying no does not affect your payment.
>
> Withdrawal: you may stop at any time by closing the browser; incomplete sessions are deleted and
> not analysed. To withdraw after completing, message us through Prolific; we will delete your data
> if the request arrives before ____ (the date the ID link is deleted). After that date we can no
> longer tell which data are yours, so withdrawal is not possible.
>
> Questions or concerns: contact the researcher at ____ . For questions about your rights as a
> participant, contact ____ (board contact).
>
> [ ] I am 18 or older. [ ] I have read the above and agree to take part. [ ] I understand my
> Prolific ID will be replaced by a code and that withdrawal is possible only until ____ .

- [ ] Reading level and length of the consent text checked by the board.
- [ ] Consent version number and date printed at the bottom of the page and stored in
      `battery_version`.

## 7. Withdrawal

- [ ] During the session: closing the browser ends participation; the intake endpoint discards
      sessions without a completion code (no partial rows are stored beyond the session).
- [ ] After completion, before the mapping deletion date: a Prolific message triggers deletion of
      all rows with that participant's hash, logged in the `audit_log` table with the action
      `delete_subject` and no payload beyond the hash. Payment already made is not reclaimed.
- [ ] After the mapping deletion date: withdrawal impossible (stated in consent).
- [ ] Wave 2 and event-wave invitations are opt-in on Prolific; not responding has no consequence.

## 8. Payment and incentives

- [ ] Base pay = T/60 × $12/hr with T the pilot-measured duration (`PROLIFIC.md` §9); the hourly
      rate is checked against Prolific's current recommended rate at launch; the board is told the
      resulting amount: ${{BASE_PAY_USD_FROM_T}}.
- [ ] Bonus: stake ${{BONUS_STAKE_USD_OWNER}} × realized multiplier from the documented historical
      path table; never negative; paid within 7 days of the wave closing (design decision) through
      Prolific's bonus facility.
- [ ] Rejections: only under Prolific's attention-check policy; analysis exclusions never affect
      payment (`PROLIFIC.md` §8, §10).
- [ ] Screen-outs and timed-out sessions are paid per Prolific's rules.
- [ ] The payout rule involves real historical data and no simulation; the path table is
      attached to the submission once built (`PROLIFIC.md` §5.3) so the board can see the actual
      range of possible bonuses. Until then the range is `{{BONUS_RANGE_FROM_PATH_TABLE}}`.

## 9. Data retention and security

- [ ] Raw Prolific export and ID mapping: encrypted at rest on the owner's machine; deleted on the
      mapping deletion date (last wave close), recorded in `STATUS.md`.
- [ ] Anonymised dataset (hashed IDs, bands only): retained for ____ years after the study ends
      (owner/board policy), then deleted or, if publicly released, governed by the release licence.
- [ ] Backups: ____ (location, encryption, who holds the key).
- [ ] Access control: PI only, plus ____ .
- [ ] Salt for the HMAC: stored ____ (outside the repository; e.g. a password manager); never
      committed; rotated never (rotation would break the linkage between waves).
- [ ] Breach procedure: if the mapping or salt were exposed, participants would be notified through
      Prolific and the board informed within ____ days.

## 10. Additional items the board may ask about

- [ ] No deception; no debrief needed for deception, but a debrief page is shown (payout disclosure
      and a plain statement of the study's purpose).
- [ ] No collection of data from third parties (the "friend" in the social-cue scenario is
      hypothetical).
- [ ] Market data used for payouts are public index closing levels; no participant data are sent to
      any market-data provider.
- [ ] The static instrument is hosted by the owner; no third-party analytics or trackers; the only
      external service is Prolific for recruitment and payment.
- [ ] Software: the static jsPsych build under `instrument/static/` (jsPsych vendored from npm, open
      source) driven by `instrument/battery.json`; the intake endpoint is part of this repository
      (Phase 5); the board can inspect all of it. The Prolific-mode additions (hashed IDs, payout,
      attention checks, wave questions) are listed in `PROLIFIC.md` §3 and are not yet implemented.
- [ ] Pilot (20 participants) is covered by this same protocol and consent; pilot data are excluded
      from confirmatory analyses but retained under the same rules.
- [ ] Amendments: any change to the scenarios, payment, waves or data handling is resubmitted before
      use; the `battery_version` tag changes with every amendment.
- [ ] Unanticipated problems: reported to the board within its required window; recorded in
      `STATUS.md`.
- [ ] Conflict of interest: the PI is building the dashboard that the models will serve; stated in
      §1 and in consent.

## 11. Documents to attach to the submission

- [ ] This checklist, completed.
- [ ] `instrument/PROLIFIC.md` (design), `instrument/PREREGISTRATION.md` (analysis plan).
- [ ] Full instrument text: every screen, in order — `instrument/battery.json` (scenario template,
      context sentences, questions, consent, instructions, quiz) plus the Prolific-mode screens of
      `PROLIFIC.md` §3–§8 once implemented.
- [ ] Consent text (§6) with version and date.
- [ ] Prolific study description and the wave-2 / event-wave invitation texts.
- [ ] `instrument/path_table.csv` and `instrument/path_table.md` once built, with the catalog entry
      for the price series.
- [ ] Data-management description (§5, §9) and the deletion schedule.
- [ ] Budget approval from the owner (nothing is recruited without it).

## 12. What this checklist does not do

It does not constitute approval, does not decide the exemption category, and does not replace the
board's forms. The design assumes US-resident adults recruited on Prolific; any change of population
re-opens every item in §2.

## 13. Placeholder tokens used in this file

| Token | Where it comes from |
|---|---|
| `{{T_BATTERY_MIN_FROM_PILOT}}` | pilot: median completion time in minutes (PROLIFIC.md §9) |
| `{{BASE_PAY_USD_FROM_T}}` | `T/60 × $12`, computed once T is known |
| `{{BONUS_STAKE_USD_OWNER}}` | owner decision on the bonus stake (PROLIFIC.md §5.2) |
| `{{BONUS_RANGE_FROM_PATH_TABLE}}` | min and max of stake × M over the built path table (PROLIFIC.md §5.3) |
| `{{N_TARGET_FROM_PHASE2}}` | Phase 2 recovery study (PLAN.md §5) |
| `{{EXCLUSION_RATE_FROM_PILOT}}` | pilot: fraction of completions excluded (PROLIFIC.md §10) |
