#!/usr/bin/env python
"""Validate bre/instrument/battery.json against PLAN.md section 3 and the schema (PLAN.md section 2).

Run with:  /home/user/bre-venv/bin/python bre/instrument/validate_battery.py [--write-static]

Exit status 0 when every check passes, 1 otherwise. Checks: loss levels, the 17 context conditions
(tags of the form namespace:value), the 2 question orders (exactly tolerance-first / scenario-first),
question texts and elicitation types (tolerance -> binary_yes_no with Yes = 1, sell_hold ->
binary_sell, allocation_share -> allocation_pct; the enum is compared with db.models.ELICITATION_TYPES
when that module imports), the 170-item scenario universe and the uniqueness/format of its scenario
ids, the full-form and short-form item counts and balance tables, the covariate bands, the
financial-literacy quiz keys, the output field list and output.elicitation_types_used, placeholder
tokens, and that static/battery.data.js (the browser copy, needed because fetch() of a local JSON
file fails under file://) is in sync with battery.json. --write-static regenerates that copy first.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATTERY_PATH = HERE / "battery.json"
STATIC_DATA_PATH = HERE / "static" / "battery.data.js"

EXPECTED_VERSION = "1.0.0"
EXPECTED_LOSSES = [-0.05, -0.10, -0.15, -0.20, -0.30]
EXPECTED_HORIZON = 365
NONNULL_CONTEXTS = {"news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct"}
EXPECTED_ORDERS = ["tolerance-first", "scenario-first"]
TOLERANCE_TEXT = (
    "Would you describe yourself as someone who avoids investment losses even at the cost of lower returns?"
)
# Schema enum incl. the PLAN.md section 2 amendment of 2026-09-17 (binary_yes_no, choice_rate).
ELICITATION_TYPES = {"binary_sell", "allocation_pct", "likert", "lottery_choice", "binary_yes_no", "choice_rate"}
QUESTION_TYPES = {"tolerance": "binary_yes_no", "sell_hold": "binary_sell", "allocation_share": "allocation_pct"}
CONTEXT_TAG_RE = re.compile(r"^[a-z_]+:[a-z0-9_]+$")
SCHEMA_COLUMNS = [
    "subject_id", "dataset", "session_id", "timestamp", "position_in_session", "scenario_id",
    "loss_pct", "horizon_days", "context_tags", "question_order_id", "prior_question_ids",
    "elicitation_type", "response", "response_time_ms", "covariates", "outcome_behavior",
    "incentivized", "consent_training", "battery_version", "is_synthetic", "source_row_ref",
]
COVARIATE_BANDS = {
    "age_band": {"18-29", "30-44", "45-59", "60+"},
    "wealth_band": {"<50k", "50-250k", "250k-1M", ">1M"},
    "education": {"hs", "some_college", "bachelor", "graduate"},
}
SCENARIO_ID_RE = re.compile(
    r"^L-0\.(05|10|15|20|30)\|(none|[a-z_]+:[a-z_0-9]+(>[a-z_]+:[a-z_0-9]+)?)\|(tolerance-first|scenario-first)$"
)


def try_import_schema_columns() -> list[str] | None:
    """Prefer the authoritative column tuple from db/models.py when SQLAlchemy is installed."""
    try:
        sys.path.insert(0, str(HERE.parent))
        from db.models import SCHEMA_COLUMNS as cols  # type: ignore

        return list(cols)
    except Exception:
        return None


def try_import_db_elicitation_types() -> list[str] | None:
    """The authoritative elicitation enum from db/models.py, or None when it cannot be imported."""
    try:
        sys.path.insert(0, str(HERE.parent))
        from db.models import ELICITATION_TYPES as types  # type: ignore

        return list(types)
    except Exception:
        return None


def static_data_text(battery: dict) -> str:
    return (
        "// GENERATED from ../battery.json by validate_battery.py --write-static. Do not edit by hand.\n"
        "// Exists because browsers refuse fetch() of a local JSON file under file://.\n"
        "window.BRE_BATTERY = "
        + json.dumps(battery, indent=2, ensure_ascii=False)
        + ";\n"
    )


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: int = 0

    def check(self, cond: bool, label: str, detail: str = "") -> None:
        if cond:
            self.passes += 1
            print(f"PASS  {label}")
        else:
            self.failures.append(label)
            print(f"FAIL  {label}" + (f" -- {detail}" if detail else ""))


def main(argv: list[str]) -> int:
    write_static = "--write-static" in argv
    raw = BATTERY_PATH.read_text(encoding="utf-8")
    battery = json.loads(raw)
    c = Checker()

    # --- identity -------------------------------------------------------------------------------
    c.check(battery.get("battery_version") == EXPECTED_VERSION, "battery_version == 1.0.0", str(battery.get("battery_version")))
    c.check(battery.get("dataset") == "intake_battery", "dataset == intake_battery")

    d = battery["design"]

    # --- losses / horizon -------------------------------------------------------------------------
    c.check(d["loss_pcts"] == EXPECTED_LOSSES, "5 loss levels exactly as PLAN.md section 3", str(d["loss_pcts"]))
    c.check(d["horizon_days"] == EXPECTED_HORIZON, "horizon_days == 365")
    c.check(
        set(d["loss_pct_display"].keys()) == {f"{l:.2f}" for l in EXPECTED_LOSSES},
        "loss_pct_display keyed by every loss level",
    )

    # --- contexts ---------------------------------------------------------------------------------
    ctx = d["contexts"]
    c.check(set(ctx.keys()) == NONNULL_CONTEXTS | {"none"}, "context set C = none + 4 non-null contexts", str(sorted(ctx)))
    c.check(all(ctx[t]["tag"] == t for t in ctx), "context tag fields match their keys")
    c.check(
        all(isinstance(ctx[t].get("display"), str) and ctx[t]["display"].strip() for t in NONNULL_CONTEXTS),
        "every non-null context has display text",
    )
    c.check(ctx["none"]["display"] is None, "none context has no display text")
    c.check(
        {t for t in ctx if ctx[t]["kind"] == "news"} == {"news:recession", "news:technical"},
        "exactly the two news contexts have kind news",
    )
    c.check(
        all(ctx[t]["triggers_delay"] == (ctx[t]["kind"] == "news") for t in ctx),
        "triggers_delay is true for news contexts only",
    )
    c.check(set(d["context_tag_order"]) == NONNULL_CONTEXTS and len(d["context_tag_order"]) == 4, "context_tag_order lists the 4 non-null contexts once")
    c.check(all(CONTEXT_TAG_RE.match(t) for t in NONNULL_CONTEXTS), "every non-null context tag is namespace:value", str(sorted(NONNULL_CONTEXTS)))

    # --- 17 context conditions --------------------------------------------------------------------
    conds = d["context_conditions"]
    c.check(len(conds) == 17, "17 context conditions", f"got {len(conds)}")
    types = [x["condition_type"] for x in conds]
    c.check(types.count("none") == 1 and types.count("single") == 4 and types.count("pair") == 12, "condition types: 1 none, 4 single, 12 pair", str({t: types.count(t) for t in set(types)}))
    ids = [x["condition_id"] for x in conds]
    c.check(len(set(ids)) == 17, "condition ids unique")
    singles = {tuple(x["context_tags"]) for x in conds if x["condition_type"] == "single"}
    c.check(singles == {(t,) for t in NONNULL_CONTEXTS}, "singles cover each non-null context exactly once")
    pairs = {tuple(x["context_tags"]) for x in conds if x["condition_type"] == "pair"}
    c.check(pairs == set(itertools.permutations(sorted(NONNULL_CONTEXTS), 2)), "pairs are the 12 ordered pairs of distinct non-null contexts")
    c.check(all(x["context_tags"] == [] for x in conds if x["condition_type"] == "none"), "none condition has empty context_tags")

    # --- question orders --------------------------------------------------------------------------
    orders = [o["question_order_id"] for o in d["question_orders"]]
    c.check(orders == EXPECTED_ORDERS, "2 question orders: tolerance-first, scenario-first", str(orders))
    seqs = {o["question_order_id"]: o["page_sequence"] for o in d["question_orders"]}
    c.check(
        seqs.get("tolerance-first", [None])[0] == "tolerance" and seqs.get("scenario-first", [None])[-1] == "tolerance",
        "tolerance page is first in tolerance-first and last in scenario-first",
    )
    for oid, seq in seqs.items():
        c.check(
            seq.index("scenario") < seq.index("delay_if_news") < seq.index("sell_hold") < seq.index("allocation_share"),
            f"{oid}: scenario -> delay_if_news -> sell_hold -> allocation_share in that order",
        )

    # --- questions --------------------------------------------------------------------------------
    q = d["questions"]
    c.check(q["tolerance"]["text"] == TOLERANCE_TEXT, "tolerance question text verbatim from PLAN.md")
    c.check(all(q[k]["elicitation_type"] in ELICITATION_TYPES for k in q), "all elicitation types in the schema enum (this file's copy)", str({k: q[k]["elicitation_type"] for k in q}))
    db_types = try_import_db_elicitation_types()
    if db_types is None:
        print("NOTE  db.models could not be imported; the elicitation enum was checked against this file's copy only")
    else:
        c.check(set(db_types) == ELICITATION_TYPES, "elicitation enum agrees with db.models.ELICITATION_TYPES", f"db.models has {sorted(db_types)}")
        c.check(all(q[k]["elicitation_type"] in db_types for k in q), "all elicitation types accepted by db.models.ELICITATION_TYPES", f"db.models has {sorted(db_types)}")
    c.check(q["tolerance"]["elicitation_type"] == "binary_yes_no", "tolerance is binary_yes_no (PLAN.md section 2 amendment)", q["tolerance"]["elicitation_type"])
    c.check(
        next((ch["label"] for ch in q["tolerance"]["choices"] if ch["response"] == 1), None) == "Yes",
        "tolerance codes Yes = 1",
    )
    c.check(q["sell_hold"]["elicitation_type"] == "binary_sell", "sell_hold is binary_sell")
    c.check(next((ch["label"] for ch in q["sell_hold"]["choices"] if ch["response"] == 1), None) == "Sell", "sell_hold codes Sell = 1")
    c.check(q["allocation_share"]["elicitation_type"] == "allocation_pct", "allocation_share is allocation_pct")
    c.check({k: q[k]["elicitation_type"] for k in q} == QUESTION_TYPES, "question -> elicitation_type map is exactly the contracted one", str({k: q[k]["elicitation_type"] for k in q}))
    c.check(q["allocation_share"]["text"] == "What share of this position would you sell?", "allocation question text")
    sl = q["allocation_share"]["slider"]
    c.check(sl["min"] == 0 and sl["max"] == 100, "allocation slider is 0-100")
    c.check(
        {ch["response"] for ch in q["tolerance"]["choices"]} == {0, 1} and {ch["response"] for ch in q["sell_hold"]["choices"]} == {0, 1},
        "binary questions code responses as 0/1",
    )

    # --- scenario template / delay / repeats ------------------------------------------------------
    tpl = d["scenario"]["template"]
    c.check("{loss_pct_display}" in tpl["loss_sentence"] and "{loss_pct_display}" in tpl["reminder"], "scenario template carries the {loss_pct_display} placeholder")
    c.check(not re.search(r"\d", tpl["intro"]), "scenario intro contains no numbers")
    c.check(
        all(not re.search(r"\d", ctx[t]["display"]) for t in ("news:recession", "news:technical", "social:friend_sells")),
        "no numbers in the news/social context sentences",
    )
    c.check(re.findall(r"\d+%", ctx["market:recovered_5pct"]["display"]) == ["5%"], "recovery sentence's only number is 5%")
    dp = d["delay_page"]
    c.check(isinstance(dp["display_seconds"], (int, float)) and dp["display_seconds"] > 0, "delay page has a positive display_seconds")
    rr = d["repeated_scenario_rule"]
    c.check(rr["n_repeats_full_form"] == 2 and rr["n_repeats_short_form"] == 0, "repeat rule: 2 repeats in full, 0 in short")
    c.check(rr["min_gap_items"] >= 1, "repeat min gap >= 1")

    # --- scenario universe ------------------------------------------------------------------------
    uni = d["scenario_universe"]
    c.check(len(uni) == 170, "scenario universe has 170 items (5 x 17 x 2)", f"got {len(uni)}")
    uids = [u["scenario_id"] for u in uni]
    c.check(len(set(uids)) == len(uids), "scenario ids unique")
    c.check(all(SCENARIO_ID_RE.match(s) for s in uids), "scenario ids match the documented format", str([s for s in uids if not SCENARIO_ID_RE.match(s)][:3]))
    regenerated = set()
    for loss in EXPECTED_LOSSES:
        for cond in conds:
            for oid in EXPECTED_ORDERS:
                tags = cond["context_tags"]
                regenerated.add(f"L{loss:.2f}|{'>'.join(tags) if tags else 'none'}|{oid}")
    c.check(set(uids) == regenerated, "scenario universe equals losses x conditions x orders")
    c.check(
        all(
            u["scenario_id"] == f"L{u['loss_pct']:.2f}|{'>'.join(u['context_tags']) if u['context_tags'] else 'none'}|{u['question_order_id']}"
            for u in uni
        ),
        "each universe item's id is consistent with its fields",
    )
    c.check(all(u["has_news_delay"] == any(ctx[t]["kind"] == "news" for t in u["context_tags"]) for u in uni), "has_news_delay consistent with context kinds")
    c.check(all(u["horizon_days"] == EXPECTED_HORIZON for u in uni), "every universe item has horizon_days 365")
    c.check(all(u["question_order_id"] in EXPECTED_ORDERS for u in uni), "every universe item's question_order_id is tolerance-first or scenario-first")
    c.check(all(CONTEXT_TAG_RE.match(t) for u in uni for t in u["context_tags"]), "every universe item's context tags are namespace:value")

    # --- forms ------------------------------------------------------------------------------------
    forms = battery["forms"]
    c.check(set(forms.keys()) == {"full", "short"}, "forms: full and short")
    full, short = forms["full"], forms["short"]
    c.check(full["n_items"] == 12 and full["n_repeats"] == 2 and full["n_presentations"] == 14, "full form: 12 items + 2 repeats = 14 presentations", str((full["n_items"], full["n_repeats"], full["n_presentations"])))
    c.check(short["n_items"] == 5 and short["n_repeats"] == 0 and short["n_presentations"] == 5, "short form: 5 items, no repeats", str((short["n_items"], short["n_repeats"], short["n_presentations"])))
    fb = full["balance"]["condition_type_counts"]
    c.check(fb == {"none": 4, "single": 4, "pair": 4} and sum(fb.values()) == full["n_items"], "full form balanced 4/4/4 over condition types")
    c.check(sum(full["balance"]["orders_per_condition_type"].values()) == 4 and set(full["balance"]["orders_per_condition_type"]) == set(EXPECTED_ORDERS), "full form: 2 + 2 question orders per block")
    sb = short["balance"]["condition_type_counts"]
    c.check(sum(sb.values()) == short["n_items"] and all(v >= 1 for v in sb.values()), "short form condition-type counts sum to 5 and cover every type")
    c.check(full["target_duration_min"] == [8, 12], "full form target duration 8-12 minutes")
    c.check(full["n_repeats"] == rr["n_repeats_full_form"] and short["n_repeats"] == rr["n_repeats_short_form"], "form repeat counts agree with the repeat rule")

    # --- intake -----------------------------------------------------------------------------------
    intake = battery["intake"]
    cov = {x["key"]: x for x in intake["covariates"]}
    c.check(set(cov) | {"financial_literacy_score"} == {"age_band", "wealth_band", "invest_experience_yrs", "self_reported_risk_tolerance", "financial_literacy_score", "education"}, "covariate keys match PLAN.md section 3", str(sorted(cov)))
    for key, bands in COVARIATE_BANDS.items():
        vals = {o["value"] for o in cov[key]["options"] if o["value"] is not None}
        c.check(vals == bands, f"{key} bands match PLAN.md", str(sorted(vals)))
    c.check(cov["invest_experience_yrs"]["min"] == 0 and cov["invest_experience_yrs"]["max"] == 40, "invest_experience_yrs range 0-40")
    c.check(cov["self_reported_risk_tolerance"]["min"] == 1 and cov["self_reported_risk_tolerance"]["max"] == 7, "self_reported_risk_tolerance range 1-7")
    quiz = intake["financial_literacy_quiz"]
    items = quiz["items"]
    c.check(len(items) == 5, "financial literacy quiz has 5 items", f"got {len(items)}")
    c.check(sum(1 for i in items if i["family"] == "big_three") == 3, "quiz includes the three standard items")
    c.check(all(0 <= i["answer_index"] < len(i["options"]) for i in items), "every quiz item's answer_index is a valid option")
    c.check(len({i["item_id"] for i in items}) == 5, "quiz item ids unique")
    c.check(quiz["score_range"] == [0, 5], "financial_literacy_score range 0-5")
    cb = {x["name"]: x for x in intake["consent"]["checkboxes"]}
    c.check("consent_training" in cb and cb["consent_training"]["maps_to"] == "consent_training", "consent page has the consent_training checkbox")
    c.check(cb["consent_training"]["label"] == "I consent to my anonymised responses being used to train the model", "consent_training checkbox label verbatim")

    # --- output -----------------------------------------------------------------------------------
    cols = try_import_schema_columns() or SCHEMA_COLUMNS
    c.check(battery["output"]["schema_fields"] == cols, "output field list equals the schema column list", str(battery["output"]["schema_fields"]))
    c.check(battery["output"]["row_rules"]["outcome_behavior"] is None and battery["output"]["row_rules"]["incentivized"] is False and battery["output"]["row_rules"]["is_synthetic"] is False, "row rules: outcome_behavior null, incentivized false, is_synthetic false")
    c.check(battery["output"].get("elicitation_types_used") == {k: q[k]["elicitation_type"] for k in q}, "output.elicitation_types_used matches design.questions", str(battery["output"].get("elicitation_types_used")))
    c.check("binary_yes_no" in battery["output"]["row_rules"].get("elicitation_type", "") and "lottery_choice" not in json.dumps(q["tolerance"]["elicitation_type"]), "output contract names binary_yes_no for the tolerance rows")

    # --- placeholder tokens -----------------------------------------------------------------------
    tokens_in_file = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", raw)))
    c.check(tokens_in_file == sorted(battery.get("placeholder_tokens", [])), "placeholder_tokens lists every {{TOKEN}} in the file", str(tokens_in_file))

    # --- static copy ------------------------------------------------------------------------------
    expected_static = static_data_text(battery)
    if write_static:
        STATIC_DATA_PATH.write_text(expected_static, encoding="utf-8")
        print(f"wrote {STATIC_DATA_PATH}")
    current = STATIC_DATA_PATH.read_text(encoding="utf-8") if STATIC_DATA_PATH.exists() else None
    c.check(current == expected_static, "static/battery.data.js is in sync with battery.json (run --write-static)")

    print(f"\n{c.passes} checks passed, {len(c.failures)} failed")
    if c.failures:
        for f in c.failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
