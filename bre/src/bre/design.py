"""Shared experimental design of the Behavioral Risk Engine (PLAN.md section 3).

This module is the single source of truth for

* the loss scenarios, the context set and the 17 ordered context conditions,
* the two question orders and the exact wording of the two questions,
* the full 5 x 17 x 2 = 170-item sell/hold design used by the generators and the recovery study,
* the covariate specification (bands and numeric ranges) with placeholder marginals, and
* the randomized, balanced intake-battery subset with its assignment record.

Everything here is deterministic given a ``numpy.random.Generator``; nothing here reads data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

DESIGN_VERSION = "1.0.0"
"""Identifier of this design; generators and the intake battery write it into ``battery_version``.

It is the same string as ``battery_version`` in ``instrument/battery.json``: a row whose
``battery_version`` equals it must use one of :data:`QUESTION_ORDER_IDS` as ``question_order_id``
(rows from external datasets carry another ``battery_version`` and a free-form order id).
"""

# ---------------------------------------------------------------------------------------------
# Loss scenarios
# ---------------------------------------------------------------------------------------------

LOSS_PCTS: tuple[float, ...] = (-0.05, -0.10, -0.15, -0.20, -0.30)
"""Signed portfolio losses of the scenario, as fractions of portfolio value (PLAN.md section 3)."""

HORIZON_DAYS: int = 365
"""Investment horizon stated in every scenario unless a dataset says otherwise."""

# ---------------------------------------------------------------------------------------------
# Contexts and context conditions
# ---------------------------------------------------------------------------------------------

CONTEXT_NONE = "none"
"""Label of the baseline condition in which no context sentence is shown (not a schema tag)."""

CONTEXTS: tuple[str, ...] = (
    CONTEXT_NONE,
    "news:recession",
    "news:technical",
    "social:friend_sells",
    "market:recovered_5pct",
)
"""Context set C of PLAN.md section 3, ``none`` first."""

NONNULL_CONTEXTS: tuple[str, ...] = tuple(c for c in CONTEXTS if c != CONTEXT_NONE)
"""The four contexts that are written as ``context_tags`` in the schema."""

CONTEXT_CODES: dict[str, str] = {
    CONTEXT_NONE: "none",
    "news:recession": "rec",
    "news:technical": "tech",
    "social:friend_sells": "friend",
    "market:recovered_5pct": "recov",
}
"""Short codes used inside ``scenario_id`` strings; the mapping is one-to-one."""

_CODE_TO_CONTEXT: dict[str, str] = {v: k for k, v in CONTEXT_CODES.items()}

CONTEXT_SEQUENCE_SEPARATOR = ">"
"""Separates the first from the second context in a condition id, e.g. ``rec>friend``."""


def context_conditions() -> list[tuple[str, ...]]:
    """Return the 17 ordered context conditions of PLAN.md section 3.

    Rule: ``()`` (the ``none`` condition), each of the four non-null contexts singly, and every
    ordered pair ``(c1, c2)`` of distinct non-null contexts, in the order c1 is shown before c2.
    Count: 1 + 4 + 4 * 3 = 17. Each tuple is exactly the ``context_tags`` list of an item.
    """
    singles = [(c,) for c in NONNULL_CONTEXTS]
    pairs = [(a, b) for a in NONNULL_CONTEXTS for b in NONNULL_CONTEXTS if a != b]
    return [()] + singles + pairs


def condition_id(condition: tuple[str, ...] | list[str]) -> str:
    """Encode a context condition as a short id: ``none``, ``rec`` or ``rec>friend``."""
    tags = tuple(condition)
    if not tags:
        return CONTEXT_CODES[CONTEXT_NONE]
    unknown = [t for t in tags if t not in CONTEXT_CODES or t == CONTEXT_NONE]
    if unknown:
        raise ValueError(f"unknown context tag(s) {unknown}; allowed: {NONNULL_CONTEXTS}")
    return CONTEXT_SEQUENCE_SEPARATOR.join(CONTEXT_CODES[t] for t in tags)


def condition_from_id(cid: str) -> tuple[str, ...]:
    """Inverse of :func:`condition_id`: ``"rec>friend"`` -> ``("news:recession", "social:friend_sells")``."""
    if cid == CONTEXT_CODES[CONTEXT_NONE]:
        return ()
    tags = []
    for code in cid.split(CONTEXT_SEQUENCE_SEPARATOR):
        if code not in _CODE_TO_CONTEXT or code == CONTEXT_CODES[CONTEXT_NONE]:
            raise ValueError(f"unknown context code {code!r} in condition id {cid!r}")
        tags.append(_CODE_TO_CONTEXT[code])
    return tuple(tags)


# ---------------------------------------------------------------------------------------------
# Questions and question orders
# ---------------------------------------------------------------------------------------------

TOLERANCE_QUESTION_ID = "tolerance"
"""``scenario_id`` of the tolerance question rows; it has no loss scenario (``loss_pct`` null)."""

SELL_QUESTION_ID = "sell"
"""Name of the sell/hold question inside a question order."""

TOLERANCE_FIRST = "tolerance-first"
SCENARIO_FIRST = "scenario-first"

QUESTION_ORDERS: dict[str, tuple[str, str]] = {
    TOLERANCE_FIRST: (TOLERANCE_QUESTION_ID, SELL_QUESTION_ID),
    SCENARIO_FIRST: (SELL_QUESTION_ID, TOLERANCE_QUESTION_ID),
}
"""``question_order_id`` -> sequence in which the two questions are asked (PLAN.md section 3).

The ids are the hyphenated ``tolerance-first`` / ``scenario-first`` of PLAN.md, the same spelling
``instrument/battery.json`` and ``db.seed`` use; there is no other spelling anywhere.
"""

QUESTION_ORDER_IDS: tuple[str, ...] = tuple(QUESTION_ORDERS)
"""``("tolerance-first", "scenario-first")``."""

TOLERANCE_QUESTION = (
    "Would you describe yourself as someone who avoids investment losses even at the cost of "
    "lower returns?"
)
"""Binary tolerance question (PLAN.md section 3); response 1 = Yes, 0 = No, stored with
``elicitation_type = "binary_yes_no"``."""

TOLERANCE_RESPONSE_OPTIONS: dict[int, str] = {1: "Yes", 0: "No"}

LOSS_SENTENCE = (
    "Imagine that you hold a diversified stock portfolio that you intend to stay invested in for "
    "at least {horizon_text} from today. Over the past month its value has fallen by "
    "{loss_abs_pct}% (a portfolio that was worth ${value_start:,.0f} is now worth "
    "${value_now:,.0f})."
)
"""Loss scenario sentence. Placeholders: horizon_text, loss_abs_pct, value_start, value_now."""

CONTEXT_SENTENCES: dict[str, str] = {
    "news:recession": (
        "Financial news attributes the fall to a broad economic slowdown: analysts now expect a "
        "recession within the coming year, with lower corporate earnings across most sectors."
    ),
    "news:technical": (
        "Financial news attributes the fall to a technical fault: an outage at a major trading "
        "venue triggered automated selling that had nothing to do with the companies' earnings or "
        "prospects."
    ),
    "social:friend_sells": (
        "A close friend whose financial judgment you respect tells you that they have just sold "
        "their entire stock portfolio and moved the money into a savings account."
    ),
    "market:recovered_5pct": (
        "Since that low point the market has partly recovered: your portfolio has regained 5% of "
        "its value from the lowest point it reached."
    ),
}
"""One sentence per non-null context; shown in the order given by the condition tuple."""

SELL_PROMPT = (
    "Would you sell your stock holdings now and move the money to a savings account, or would "
    "you hold the portfolio as planned?"
)
"""Closing question of every sell/hold item (no placeholders)."""

SELL_RESPONSE_OPTIONS: dict[int, str] = {1: "Sell", 0: "Hold"}
"""``binary_sell`` coding: response 1 = sell now, 0 = hold."""

SELL_QUESTION = "{loss_sentence} {context_sentences}{sell_prompt}"
"""Template of a sell/hold item: loss sentence, then the context sentences in order, then the prompt.

``context_sentences`` is empty for the ``none`` condition and otherwise each sentence is followed by
a single space. Use :func:`render_sell_question` to fill it.
"""


def horizon_text(horizon_days: int = HORIZON_DAYS) -> str:
    """Render a horizon in days as advisor-facing text: 365 -> ``"one year"``, 730 -> ``"2 years"``."""
    if horizon_days < 0:
        raise ValueError("horizon_days must be >= 0")
    if horizon_days % 365 == 0 and horizon_days > 0:
        years = horizon_days // 365
        return "one year" if years == 1 else f"{years} years"
    if horizon_days % 30 == 0 and horizon_days > 0:
        months = horizon_days // 30
        return "one month" if months == 1 else f"{months} months"
    return "one day" if horizon_days == 1 else f"{horizon_days} days"


def render_sell_question(
    loss_pct: float,
    context_tags: tuple[str, ...] | list[str] = (),
    horizon_days: int = HORIZON_DAYS,
    portfolio_value: float = 100_000.0,
) -> str:
    """Fill :data:`SELL_QUESTION` for one item.

    Rules: ``loss_pct`` must be in [-1, 0); the value shown is ``portfolio_value * (1 + loss_pct)``;
    the context sentences appear in the order of ``context_tags`` (the context sequence the
    Q-models compose as ``U_{c_m} ... U_{c_1}``); an empty tuple renders the ``none`` condition.
    """
    if not (-1.0 <= loss_pct < 0.0):
        raise ValueError(f"loss_pct must be a signed loss in [-1, 0); got {loss_pct}")
    unknown = [t for t in context_tags if t not in CONTEXT_SENTENCES]
    if unknown:
        raise ValueError(f"no sentence for context tag(s) {unknown}")
    loss_abs_pct = abs(loss_pct) * 100
    loss_abs_text = f"{loss_abs_pct:g}"
    htext = horizon_text(horizon_days)
    loss_sentence = LOSS_SENTENCE.format(
        horizon_text=htext,
        loss_abs_pct=loss_abs_text,
        value_start=portfolio_value,
        value_now=portfolio_value * (1 + loss_pct),
    )
    context_sentences = "".join(CONTEXT_SENTENCES[t] + " " for t in context_tags)
    return SELL_QUESTION.format(
        loss_sentence=loss_sentence, context_sentences=context_sentences, sell_prompt=SELL_PROMPT
    )


# ---------------------------------------------------------------------------------------------
# Full design
# ---------------------------------------------------------------------------------------------

DESIGN_COLUMNS: tuple[str, ...] = (
    "item_id",
    "scenario_id",
    "loss_pct",
    "horizon_days",
    "context_condition_id",
    "context_tags",
    "n_contexts",
    "question_order_id",
    "prior_question_ids",
    "elicitation_type",
    "question_text",
)

N_LOSSES = len(LOSS_PCTS)
N_CONDITIONS = 17
N_ORDERS = len(QUESTION_ORDERS)
N_ITEMS = N_LOSSES * N_CONDITIONS * N_ORDERS
"""5 x 17 x 2 = 170 sell/hold items per subject."""


def loss_code(loss_pct: float) -> str:
    """``-0.10`` -> ``"L10"``: the loss in whole percentage points, zero-padded to two digits."""
    return f"L{int(round(-loss_pct * 100)):02d}"


def scenario_id(loss_pct: float, condition: tuple[str, ...] | list[str]) -> str:
    """Scenario id = loss code and condition id, e.g. ``L10|rec>friend``; it does not encode order."""
    return f"{loss_code(loss_pct)}|{condition_id(condition)}"


def item_id(loss_pct: float, condition: tuple[str, ...] | list[str], question_order_id: str) -> str:
    """Item id = scenario id plus the question order, e.g. ``L10|rec>friend|tolerance-first``."""
    if question_order_id not in QUESTION_ORDERS:
        raise ValueError(f"unknown question order {question_order_id!r}; allowed {QUESTION_ORDER_IDS}")
    return f"{scenario_id(loss_pct, condition)}|{question_order_id}"


def full_design() -> pd.DataFrame:
    """Return the 170-item design, one row per (loss, context condition, question order).

    Rows are ordered loss-major, then condition (in :func:`context_conditions` order), then order
    (``tolerance-first`` before ``scenario-first``). ``prior_question_ids`` lists the tolerance
    question when it is asked before the scenario and is empty otherwise, so the order is
    recoverable from the schema even without ``question_order_id``.
    """
    rows = []
    for loss in LOSS_PCTS:
        for cond in context_conditions():
            for order_id, sequence in QUESTION_ORDERS.items():
                prior = [q for q in sequence[: sequence.index(SELL_QUESTION_ID)]]
                rows.append(
                    {
                        "item_id": item_id(loss, cond, order_id),
                        "scenario_id": scenario_id(loss, cond),
                        "loss_pct": float(loss),
                        "horizon_days": HORIZON_DAYS,
                        "context_condition_id": condition_id(cond),
                        "context_tags": list(cond),
                        "n_contexts": len(cond),
                        "question_order_id": order_id,
                        "prior_question_ids": prior,
                        "elicitation_type": "binary_sell",
                        "question_text": render_sell_question(loss, cond),
                    }
                )
    df = pd.DataFrame(rows, columns=list(DESIGN_COLUMNS))
    df["horizon_days"] = df["horizon_days"].astype("int64")
    df["n_contexts"] = df["n_contexts"].astype("int64")
    return df


# ---------------------------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CovariateField:
    """Specification of one covariate: either categorical (``levels``) or numeric (``lo``..``hi``)."""

    name: str
    kind: Literal["categorical", "numeric"]
    levels: tuple[str, ...] = ()
    lo: float | None = None
    hi: float | None = None
    integer: bool = False

    def is_valid(self, value: object) -> bool:
        """Membership test: a level string for categorical fields; a finite number in [lo, hi]
        (integer-valued when ``integer``) for numeric fields. Booleans are never valid."""
        if isinstance(value, (bool, np.bool_)):
            return False
        if self.kind == "categorical":
            return isinstance(value, str) and value in self.levels
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        x = float(value)
        if not np.isfinite(x) or not (self.lo <= x <= self.hi):
            return False
        return (not self.integer) or float(x).is_integer()


COVARIATE_SPEC: dict[str, CovariateField] = {
    "age_band": CovariateField("age_band", "categorical", levels=("18-29", "30-44", "45-59", "60+")),
    "wealth_band": CovariateField(
        "wealth_band", "categorical", levels=("<50k", "50-250k", "250k-1M", ">1M")
    ),
    "invest_experience_yrs": CovariateField(
        "invest_experience_yrs", "numeric", lo=0, hi=40, integer=True
    ),
    "self_reported_risk_tolerance": CovariateField(
        "self_reported_risk_tolerance", "numeric", lo=1, hi=7, integer=True
    ),
    "financial_literacy_score": CovariateField(
        "financial_literacy_score", "numeric", lo=0, hi=5, integer=True
    ),
    "education": CovariateField(
        "education", "categorical", levels=("hs", "some_college", "bachelor", "graduate")
    ),
}
"""PLAN.md section 3 covariates: bands for the categorical ones, closed integer ranges otherwise."""

COVARIATE_KEYS: tuple[str, ...] = tuple(COVARIATE_SPEC)

COVARIATE_MARGINALS_SOURCE = "placeholder-uniform; replace with FINRA NFCS Investor Survey fit"
"""Provenance of :data:`COVARIATE_MARGINALS`. Until the owner downloads the FINRA NFCS Investor
Survey (egress-blocked, see data/CATALOG.md) every marginal is uniform and covariates are drawn
independently; any table produced from them must be labelled synthetic."""

COVARIATE_MARGINALS: dict[str, dict[str, float]] = {
    name: (
        {level: 1.0 / len(spec.levels) for level in spec.levels}
        if spec.kind == "categorical"
        else {str(v): 1.0 / (int(spec.hi) - int(spec.lo) + 1) for v in range(int(spec.lo), int(spec.hi) + 1)}
    )
    for name, spec in COVARIATE_SPEC.items()
}
"""Marginal probability of every level (categorical) or integer value (numeric); each sums to one.
Source: :data:`COVARIATE_MARGINALS_SOURCE`."""


def sample_covariates(rng: np.random.Generator | int, n: int) -> pd.DataFrame:
    """Draw ``n`` covariate records from :data:`COVARIATE_MARGINALS`, independently per field.

    Rule: each field is sampled from its marginal with ``rng.choice``; numeric fields come back as
    ``int64`` columns, categorical ones as ``str``. Columns are in :data:`COVARIATE_KEYS` order.
    The result is synthetic by construction (see :data:`COVARIATE_MARGINALS_SOURCE`).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    gen = np.random.default_rng(rng)
    out: dict[str, object] = {}
    for name, spec in COVARIATE_SPEC.items():
        marginal = COVARIATE_MARGINALS[name]
        keys = list(marginal)
        probs = np.asarray([marginal[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        draws = gen.choice(len(keys), size=n, p=probs)
        if spec.kind == "categorical":
            out[name] = pd.Series([keys[i] for i in draws], dtype="str")
        else:
            out[name] = pd.Series([int(keys[i]) for i in draws], dtype="int64")
    return pd.DataFrame(out, columns=list(COVARIATE_KEYS))


# ---------------------------------------------------------------------------------------------
# Intake battery subset
# ---------------------------------------------------------------------------------------------

BATTERY_FORMS: dict[str, tuple[int, int]] = {"full": (12, 16), "short": (5, 5)}
"""Allowed ``n_items`` per form (inclusive bounds), PLAN.md section 3."""


@dataclass
class BatteryAssignment:
    """Record of one randomized battery draw, stored next to the responses it produced."""

    design_version: str
    form: str
    n_items: int
    item_ids: list[str]
    condition_ids: list[str]
    loss_counts: dict[str, int]
    order_counts: dict[str, int]
    context_counts: dict[str, int]
    condition_class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-dict form for JSON storage."""
        return {
            "design_version": self.design_version,
            "form": self.form,
            "n_items": self.n_items,
            "item_ids": list(self.item_ids),
            "condition_ids": list(self.condition_ids),
            "loss_counts": dict(self.loss_counts),
            "order_counts": dict(self.order_counts),
            "context_counts": dict(self.context_counts),
            "condition_class_counts": dict(self.condition_class_counts),
        }


def _balanced_multiset(gen: np.random.Generator, levels: list, n: int) -> list:
    """Return ``n`` draws from ``levels`` in which every level appears floor(n/k) or ceil(n/k)
    times (k = len(levels)); which levels get the extra draw and the sequence are random."""
    k = len(levels)
    base = n // k
    extra = n - base * k
    counts = {lvl: base for lvl in levels}
    for idx in gen.choice(k, size=extra, replace=False):
        counts[levels[idx]] += 1
    out = [lvl for lvl in levels for _ in range(counts[lvl])]
    perm = gen.permutation(len(out))
    return [out[i] for i in perm]


def _select_conditions(gen: np.random.Generator, n_items: int) -> list[tuple[str, ...]]:
    """Choose ``n_items`` distinct context conditions: always ``none`` and the four singles, then
    ordered pairs picked round-robin over the first-position context (each context leads
    floor/ceil of the pairs) with the pair within each context chosen at random."""
    singles = [(c,) for c in NONNULL_CONTEXTS]
    chosen: list[tuple[str, ...]] = [()] + singles
    n_pairs = n_items - len(chosen)
    if n_pairs < 0:
        raise ValueError("n_items must be at least 5 (none + four single contexts)")
    if n_pairs > len(NONNULL_CONTEXTS) * (len(NONNULL_CONTEXTS) - 1):
        raise ValueError("n_items exceeds the number of distinct context conditions (17)")
    by_first: dict[str, list[tuple[str, ...]]] = {}
    for a in NONNULL_CONTEXTS:
        pairs = [(a, b) for b in NONNULL_CONTEXTS if b != a]
        perm = gen.permutation(len(pairs))
        by_first[a] = [pairs[i] for i in perm]
    lead_order = [NONNULL_CONTEXTS[i] for i in gen.permutation(len(NONNULL_CONTEXTS))]
    picked: list[tuple[str, ...]] = []
    while len(picked) < n_pairs:
        for lead in lead_order:
            if len(picked) == n_pairs:
                break
            if by_first[lead]:
                picked.append(by_first[lead].pop())
    return chosen + picked


def battery_subset(
    rng: np.random.Generator | int, n_items: int, form: Literal["full", "short"] = "full"
) -> tuple[pd.DataFrame, BatteryAssignment]:
    """Draw a balanced, randomized subset of :func:`full_design` for one intake battery.

    Balance rules (each holds exactly for every draw):

    * conditions: ``none`` and the four single contexts always appear once each; the remaining
      ``n_items - 5`` items are distinct ordered pairs whose first-position contexts are balanced
      to within one item (short form: no pairs);
    * losses: every loss level appears floor(n/5) or ceil(n/5) times (short form: once each);
    * question orders: ``tolerance-first`` and ``scenario-first`` each appear floor(n/2) or
      ceil(n/2) times, the extra one at random when ``n_items`` is odd;
    * presentation: rows are returned in a random presentation order with a ``position`` column
      starting at 0.

    Loss levels and orders are assigned to the chosen conditions by independent random
    permutations, so the same seed always reproduces the same subset (``form`` bounds:
    :data:`BATTERY_FORMS`). Returns the subset (design columns plus ``position``) and the
    :class:`BatteryAssignment` record.
    """
    if form not in BATTERY_FORMS:
        raise ValueError(f"form must be one of {tuple(BATTERY_FORMS)}; got {form!r}")
    lo, hi = BATTERY_FORMS[form]
    if not (lo <= n_items <= hi):
        raise ValueError(f"form {form!r} takes n_items in [{lo}, {hi}]; got {n_items}")
    gen = np.random.default_rng(rng)

    conditions = _select_conditions(gen, n_items)
    losses = _balanced_multiset(gen, list(LOSS_PCTS), n_items)
    orders = _balanced_multiset(gen, list(QUESTION_ORDER_IDS), n_items)
    design = full_design().set_index("item_id")

    ids = [item_id(loss, cond, order) for loss, cond, order in zip(losses, conditions, orders)]
    subset = design.loc[ids].reset_index()
    presentation = gen.permutation(n_items)
    subset = subset.iloc[presentation].reset_index(drop=True)
    subset.insert(0, "position", np.arange(n_items, dtype="int64"))
    subset["context_tags"] = subset["context_tags"].map(list)
    subset["prior_question_ids"] = subset["prior_question_ids"].map(list)

    context_counts = {c: 0 for c in NONNULL_CONTEXTS}
    for tags in subset["context_tags"]:
        for t in tags:
            context_counts[t] += 1
    class_counts = {"none": 0, "single": 0, "pair": 0}
    for k in subset["n_contexts"]:
        class_counts[{0: "none", 1: "single", 2: "pair"}[int(k)]] += 1
    record = BatteryAssignment(
        design_version=DESIGN_VERSION,
        form=form,
        n_items=n_items,
        item_ids=subset["item_id"].tolist(),
        condition_ids=subset["context_condition_id"].tolist(),
        loss_counts={loss_code(l): int((subset["loss_pct"] == l).sum()) for l in LOSS_PCTS},
        order_counts={o: int((subset["question_order_id"] == o).sum()) for o in QUESTION_ORDER_IDS},
        context_counts=context_counts,
        condition_class_counts=class_counts,
    )
    return subset, record
