"""Seed data: the five hand-written advisor interventions and their context transforms.

Each intervention is an advisor-facing script (plain language, no figures) paired with the
change it is assumed to make to the context sequence fed into the decision model. The transform
vocabulary is deliberately small so the API and dashboard can apply it mechanically:

* ``{"add": [tag, ...]}``     append these context tags (in order) to the client's context sequence;
* ``{"remove": [tag, ...]}``  drop these tags wherever they occur in the sequence;
* ``{"question_order": "tolerance-first" | "scenario-first"}``  set the question order
  (``bre.design.QUESTION_ORDER_IDS``, PLAN.md section 3) under which the scenario is evaluated.

Tags come from the shared context set C = {news:recession, news:technical, social:friend_sells,
market:recovered_5pct} (``bre.design.NONNULL_CONTEXTS``, PLAN.md section 3) so every transform is
evaluable by the fitted models without new parameters; :func:`validate_transform` checks this
vocabulary and ``seed_interventions`` refuses a transform outside it. The mapping is a modeling
assumption recorded here, not a measured effect; the intervention_log table is where its observed
outcomes accumulate.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from bre.design import NONNULL_CONTEXTS, QUESTION_ORDER_IDS
from db.models import Intervention

TRANSFORM_KEYS: tuple[str, ...] = ("add", "remove", "question_order")
"""The transform vocabulary: ``add`` / ``remove`` list context tags, ``question_order`` names an order."""


def validate_transform(transform: dict[str, Any]) -> dict[str, Any]:
    """Rule: keys among :data:`TRANSFORM_KEYS`; ``add``/``remove`` are lists of tags from
    ``bre.design.NONNULL_CONTEXTS``; ``question_order`` is one of ``bre.design.QUESTION_ORDER_IDS``.
    Returns the transform unchanged; raises ValueError otherwise."""
    if not isinstance(transform, dict):
        raise ValueError(f"transform must be a JSON object, got {type(transform).__name__}")
    unknown = sorted(set(transform) - set(TRANSFORM_KEYS))
    if unknown:
        raise ValueError(f"transform has unknown key(s) {unknown}; allowed {TRANSFORM_KEYS}")
    for key in ("add", "remove"):
        if key in transform:
            tags = transform[key]
            if not isinstance(tags, list) or not tags:
                raise ValueError(f"transform[{key!r}] must be a non-empty list of context tags")
            bad = [t for t in tags if t not in NONNULL_CONTEXTS]
            if bad:
                raise ValueError(f"transform[{key!r}] has unknown context tag(s) {bad}; allowed {NONNULL_CONTEXTS}")
    if "question_order" in transform and transform["question_order"] not in QUESTION_ORDER_IDS:
        raise ValueError(
            f"transform['question_order'] must be one of {QUESTION_ORDER_IDS}, got {transform['question_order']!r}"
        )
    return transform

INTERVENTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "reframe_time_horizon",
        "script": (
            "Before we talk about what the market did this week, let us look again at what this "
            "money is for and when you will actually need it. The plan we built together was "
            "sized to your goals and their dates, not to today's headlines. Over the stretch of "
            "time between now and those dates, the reason behind any single downturn matters far "
            "less than staying invested through it. Would you like to walk through the plan's "
            "timeline together so we can judge this move against it?"
        ),
        "mapped_context_transform": {"remove": ["news:recession", "news:technical"]},
    },
    {
        "name": "show_historical_recoveries",
        "script": (
            "It can help to look at how broad markets have behaved after earlier declines of a "
            "similar size. I can show you the record of past drawdowns and the paths that "
            "followed them, including how long recoveries took and how uneven they were along "
            "the way. Nothing in that record guarantees what happens next, but it is the best "
            "evidence we have about what selling at the bottom has cost investors before. Shall "
            "we look at it together?"
        ),
        "mapped_context_transform": {"add": ["market:recovered_5pct"]},
    },
    {
        "name": "precommitment_reminder",
        "script": (
            "When we set up your portfolio, you told me how you wanted to respond to a loss like "
            "this one, and we wrote it down while things were calm. I would like to read that "
            "statement back to you now, in your own words, before we decide anything. If your "
            "view has genuinely changed we can revise the plan, but let us first check whether "
            "the person who wrote it would make the trade you are considering today."
        ),
        "mapped_context_transform": {"question_order": "tolerance-first"},
    },
    {
        "name": "cash_sleeve_check",
        "script": (
            "Let us confirm that your near-term spending is already covered by the cash and "
            "short-term holdings we set aside for exactly this situation. If the money you need "
            "over the coming period is safe, then nothing in the invested portion has to be sold "
            "at today's prices to pay the bills, and the case for selling rests only on a forecast "
            "rather than on a need. Would you like me to go through the cash sleeve with you now?"
        ),
        "mapped_context_transform": {"remove": ["news:recession"]},
    },
    {
        "name": "social_proof_counter",
        "script": (
            "You mentioned that people you know are getting out. It is worth remembering that you "
            "usually hear about the sales and rarely about the quiet decisions to stay put, so "
            "what you see around you is not a fair picture of what most investors are doing. "
            "Their circumstances, goals and time horizons also differ from yours. Can we set "
            "aside what others are doing for a moment and judge this decision on your own plan?"
        ),
        "mapped_context_transform": {"remove": ["social:friend_sells"]},
    },
)


def seed_interventions(session: Session) -> list[str]:
    """Insert the five hand-written interventions that are not yet present; return inserted names.

    Idempotent: an intervention whose ``name`` already exists is left untouched.
    """
    existing = set(session.scalars(select(Intervention.name)).all())
    inserted: list[str] = []
    for spec in INTERVENTIONS:
        if spec["name"] in existing:
            continue
        transform = validate_transform(spec["mapped_context_transform"])
        session.add(
            Intervention(
                name=spec["name"],
                script=spec["script"],
                mapped_context_transform=json.dumps(transform, sort_keys=True, allow_nan=False),
                active=True,
            )
        )
        inserted.append(spec["name"])
    session.flush()
    return inserted


__all__ = ["INTERVENTIONS", "TRANSFORM_KEYS", "seed_interventions", "validate_transform"]
