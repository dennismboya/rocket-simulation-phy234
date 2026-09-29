"""In-session intake battery (Phase 6, page 4): the item draw, the repeat placement, the
scenario text and the DecisionEvent rows the advisor-led session produces.

No model code and no HTTP here: the page (``pages/4_Intake_battery.py``) keeps the session
state, calls these functions and posts the rows through the API
(``POST /clients/{client_id}/responses``), which validates them with ``bre.schema`` and stores
them with ``db.session.frame_to_responses``.

Randomization scheme (the same as the static instrument, ``instrument/battery.json``):

* the items are drawn with :func:`bre.design.battery_subset` (balanced over loss level,
  condition type and question order; presentation order randomized) from a generator seeded
  with the session's 32-hex-character ``session_id`` (``numpy.random.default_rng(int(seed, 16))``),
  so ``build_presentations(seed, form, ...)`` regenerates the assignment offline;
* the full form inserts ``n_repeats`` verbatim repeats: distinct originals drawn from the first
  six presentation slots, each inserted at a slot drawn uniformly from the slots at least
  ``min_gap_items`` after its original (the second after the first has been placed), exactly
  the instrument's ``repeated_scenario_rule``;
* every presentation yields three rows in the item's question order (tolerance-first:
  tolerance, sell/hold, allocation share; scenario-first: sell/hold, allocation share,
  tolerance), ``position_in_session`` counting rows over the whole session,
  ``prior_question_ids`` the question ids already answered in that presentation.

Row conventions follow the schema and the generator (``bre.sim.common``): the tolerance row has
``scenario_id = "tolerance"`` (``bre.design.TOLERANCE_QUESTION_ID``; that is how
``bre.models.data.build_model_data`` recovers a tolerance-first item's answer), the sell/hold
and allocation rows carry the design's ``scenario_id``; ``source_row_ref`` uses the
instrument's format ``<form>:i<NN>[:rep_of_i<MM>]:<question_id>``; ``response_time_ms`` is the
time from the presentation being shown to its answers being recorded (item level, the same
value on the presentation's rows; the session record says so); ``dataset = "intake_battery"``,
``battery_version = "1.0.0"``, ``incentivized = false``, ``outcome_behavior = null``;
``subject_id`` is the client id (the API requires it) and ``is_synthetic`` is what the page
passes (true in demo mode, where the database only takes synthetic rows).
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any

import numpy as np

from bre import design as D

DATASET = "intake_battery"
BATTERY_VERSION = "1.0.0"
QUESTION_IDS: tuple[str, str, str] = ("tolerance", "sell_hold", "allocation_share")
ELICITATION: dict[str, str] = {"tolerance": "binary_yes_no", "sell_hold": "binary_sell", "allocation_share": "allocation_pct"}
SEQUENCE: dict[str, tuple[str, ...]] = {
    D.TOLERANCE_FIRST: ("tolerance", "sell_hold", "allocation_share"),
    D.SCENARIO_FIRST: ("sell_hold", "allocation_share", "tolerance"),
}
FORMS: dict[str, dict[str, int]] = {"full": {"n_items": 12, "n_repeats": 2}, "short": {"n_items": 5, "n_repeats": 0}}
"""Frozen instrument values (``battery.json`` forms); the page prefers ``GET /intake/config``."""
MIN_GAP_ITEMS = 6
REPEAT_POOL = 6
LOSS_DISPLAY: dict[float, str] = {-0.05: "5%", -0.10: "10%", -0.15: "15%", -0.20: "20%", -0.30: "30%"}
RT_GRANULARITY = "item: milliseconds from the presentation being shown to its answers being recorded; identical on the presentation's rows"


def new_seed() -> str:
    """A fresh 32-hex-character session seed (the instrument's ``session_id`` format)."""
    return secrets.token_hex(16)


def rng_from_seed(seed_hex: str) -> np.random.Generator:
    if len(seed_hex) != 32 or any(c not in "0123456789abcdef" for c in seed_hex.lower()):
        raise ValueError("seed must be 32 lowercase hex characters")
    return np.random.default_rng(int(seed_hex, 16))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_presentations(seed_hex: str, form: str, *, n_items: int | None = None, n_repeats: int | None = None,
                        min_gap: int = MIN_GAP_ITEMS, repeat_pool: int = REPEAT_POOL) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The presentation list and the assignment record of one session (module docstring)."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {list(FORMS)}")
    n_items = int(FORMS[form]["n_items"] if n_items is None else n_items)
    n_repeats = int(FORMS[form]["n_repeats"] if n_repeats is None else n_repeats)
    rng = rng_from_seed(seed_hex)
    subset, record = D.battery_subset(rng, n_items, form)
    items = subset.to_dict(orient="records")
    slots: list[dict[str, Any]] = [{"item": it, "is_repeat": False, "source": None} for it in items]
    repeats: list[dict[str, Any]] = []
    if n_repeats > 0:
        pool = slots[: min(repeat_pool, len(slots))]
        chosen = [pool[i] for i in rng.permutation(len(pool))[:n_repeats]]
        for src in chosen:
            orig = slots.index(src)
            lo = min(orig + min_gap, len(slots))
            hi = len(slots)
            at = lo + int(rng.integers(0, hi - lo + 1))
            slots.insert(at, {"item": src["item"], "is_repeat": True, "source": src})
    presentations: list[dict[str, Any]] = []
    for i, s in enumerate(slots):
        s["index"] = i + 1
    for s in slots:
        rep_of = s["source"]["index"] if s["is_repeat"] else None
        p = {"index": s["index"], "item_id": s["item"]["item_id"], "scenario_id": s["item"]["scenario_id"], "loss_pct": float(s["item"]["loss_pct"]),
             "horizon_days": int(s["item"]["horizon_days"]), "context_tags": list(s["item"]["context_tags"]), "question_order_id": str(s["item"]["question_order_id"]),
             "condition_id": str(s["item"]["context_condition_id"]), "is_repeat": bool(s["is_repeat"]), "rep_of": rep_of,
             "has_news": any(str(t).startswith("news:") for t in s["item"]["context_tags"])}
        presentations.append(p)
        if s["is_repeat"]:
            repeats.append({"index": s["index"], "rep_of": rep_of, "scenario_id": p["scenario_id"]})
    assignment = {
        **record.to_dict(), "seed": seed_hex, "session_id": seed_hex, "battery_version": BATTERY_VERSION,
        "n_repeats": n_repeats, "n_presentations": len(presentations), "min_gap_items": min_gap,
        "presentation": [{"index": p["index"], "scenario_id": p["scenario_id"], "question_order_id": p["question_order_id"], "rep_of": p["rep_of"]} for p in presentations],
        "repeats": repeats,
        "prng": "numpy.random.default_rng(int(seed, 16)); items by bre.design.battery_subset; repeats by the instrument's repeated_scenario_rule",
    }
    return presentations, assignment


def loss_display(loss_pct: float) -> str:
    return LOSS_DISPLAY.get(round(float(loss_pct), 2), f"{abs(float(loss_pct)) * 100:g}%")


def scenario_paragraphs(texts: dict[str, str], presentation: dict[str, Any], context_display: dict[str, str]) -> list[str]:
    """The scenario page as paragraphs: intro, loss sentence, context lead-in and sentences
    (numbered for a pair), reminder. ``texts`` are the newest scenario versions
    (``GET /scenarios``), ``context_display`` the newest context sentences (``GET /contexts``)."""
    ld = loss_display(presentation["loss_pct"])
    out = [str(texts.get("intro", "")), str(texts.get("loss_sentence", "")).replace("{loss_pct_display}", ld)]
    tags = list(presentation["context_tags"])
    if len(tags) == 1:
        out.append(str(texts.get("context_lead_in_single", "")))
        out.append(context_display.get(tags[0], tags[0]))
    elif len(tags) >= 2:
        out.append(str(texts.get("context_lead_in_pair", "")))
        for k, t in enumerate(tags, start=1):
            out.append(f"{k}. {context_display.get(t, t)}")
    if tags:
        out.append(str(texts.get("reminder", "")).replace("{loss_pct_display}", ld))
    return [p for p in out if p]


def source_row_ref(form: str, presentation: dict[str, Any], qid: str) -> str:
    ref = f"{form}:i{int(presentation['index']):02d}"
    if presentation.get("rep_of"):
        ref += f":rep_of_i{int(presentation['rep_of']):02d}"
    return f"{ref}:{qid}"


def rows_for_presentation(
    presentation: dict[str, Any],
    answers: dict[str, Any],
    *,
    form: str,
    subject_id: str,
    session_id: str,
    covariates_json: str,
    consent_training: bool,
    is_synthetic: bool,
    position_start: int,
    timestamp: str,
    response_time_ms: float | None,
) -> list[dict[str, Any]]:
    """The three schema rows of one presentation in the item's question order (module docstring).
    ``answers``: ``tolerance`` (0/1), ``sell_hold`` (0/1), ``allocation_share`` (0..100)."""
    order = presentation["question_order_id"]
    if order not in SEQUENCE:
        raise ValueError(f"unknown question order {order!r}")
    rows: list[dict[str, Any]] = []
    prior: list[str] = []
    pos = int(position_start)
    for qid in SEQUENCE[order]:
        if qid == "tolerance":
            resp = int(answers["tolerance"])
            if resp not in (0, 1):
                raise ValueError("tolerance answer must be 0 or 1")
            specific = {"scenario_id": D.TOLERANCE_QUESTION_ID, "loss_pct": None, "horizon_days": None, "context_tags": [], "response": float(resp)}
        elif qid == "sell_hold":
            resp = int(answers["sell_hold"])
            if resp not in (0, 1):
                raise ValueError("sell/hold answer must be 0 or 1")
            specific = {"scenario_id": presentation["scenario_id"], "loss_pct": float(presentation["loss_pct"]), "horizon_days": int(presentation["horizon_days"]),
                        "context_tags": list(presentation["context_tags"]), "response": float(resp)}
        else:
            share = float(answers["allocation_share"])
            if not 0.0 <= share <= 100.0:
                raise ValueError("allocation share must lie in [0, 100]")
            specific = {"scenario_id": presentation["scenario_id"], "loss_pct": float(presentation["loss_pct"]), "horizon_days": int(presentation["horizon_days"]),
                        "context_tags": list(presentation["context_tags"]), "response": round(share / 100.0, 4)}
        rows.append({
            "subject_id": subject_id, "dataset": DATASET, "session_id": session_id, "timestamp": timestamp, "position_in_session": pos,
            "scenario_id": specific["scenario_id"], "loss_pct": specific["loss_pct"], "horizon_days": specific["horizon_days"],
            "context_tags": specific["context_tags"], "question_order_id": order, "prior_question_ids": list(prior),
            "elicitation_type": ELICITATION[qid], "response": specific["response"],
            "response_time_ms": None if response_time_ms is None else float(int(response_time_ms)),
            "covariates": covariates_json, "outcome_behavior": None, "incentivized": False, "consent_training": bool(consent_training),
            "battery_version": BATTERY_VERSION, "is_synthetic": bool(is_synthetic), "source_row_ref": source_row_ref(form, presentation, qid),
        })
        prior.append(qid)
        pos += 1
    return rows


def covariates_json(values: dict[str, Any]) -> str:
    """Canonical covariate JSON of the six product keys; a null (``Prefer not to say``) is dropped."""
    rec = {k: values[k] for k in D.COVARIATE_KEYS if k in values and values[k] is not None}
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def quiz_score(answers: dict[str, int | None], items: list[dict[str, Any]]) -> int:
    """Number of correct answers of the financial-literacy quiz ("I don't know" counts as wrong)."""
    score = 0
    for it in items:
        a = answers.get(it["item_id"])
        if a is not None and int(a) == int(it["answer_index"]):
            score += 1
    return score


def responses_file_name(session_id: str) -> str:
    return f"intake_{session_id}.json"


__all__ = [
    "BATTERY_VERSION", "DATASET", "ELICITATION", "FORMS", "LOSS_DISPLAY", "MIN_GAP_ITEMS", "QUESTION_IDS", "REPEAT_POOL", "RT_GRANULARITY", "SEQUENCE",
    "build_presentations", "covariates_json", "loss_display", "new_seed", "now_iso", "quiz_score", "responses_file_name", "rng_from_seed",
    "rows_for_presentation", "scenario_paragraphs", "source_row_ref",
]
