"""Pydantic (v2) request and response models of the BRE API (Phase 5).

Validation rules mirror the shared design (``bre.design``): covariates are the six product keys
with their bands and integer ranges, loss levels are signed fractions in ``[-1, 0]``, context
tags come from the design's context set (``none`` or an empty list is the no-context
condition, at most two ordered tags per scenario), question orders are the design's two ids.
A violation returns HTTP 422 with the offending field in ``loc``.

Every response derived from a synthetic-training model carries ``synthetic: true`` and every
probability field is a *predicted probability of selling* under the served model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bre import design as D
from bre.market import VIX_BUCKET_LABELS

AgeBand = Literal["18-29", "30-44", "45-59", "60+"]
WealthBand = Literal["<50k", "50-250k", "250k-1M", ">1M"]
Education = Literal["hs", "some_college", "bachelor", "graduate"]
QuestionOrder = Literal["tolerance-first", "scenario-first"]
ContextTag = Literal["none", "news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct"]
VixBucket = Literal["low", "medium", "high", "extreme"]
Level = Literal["none", "low", "medium", "high", "extreme"]

assert tuple(D.QUESTION_ORDER_IDS) == ("tolerance-first", "scenario-first")
assert tuple(D.CONTEXTS) == ("none", "news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct")
assert tuple(VIX_BUCKET_LABELS) == ("low", "medium", "high", "extreme")

CLIENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,63}$"


class Covariates(BaseModel):
    """The six product covariates (PLAN.md section 3); every field is optional (missing values
    are handled by the model's missing indicators)."""

    model_config = ConfigDict(extra="forbid")

    age_band: AgeBand | None = None
    wealth_band: WealthBand | None = None
    invest_experience_yrs: int | None = Field(default=None, ge=0, le=40)
    self_reported_risk_tolerance: int | None = Field(default=None, ge=1, le=7)
    financial_literacy_score: int | None = Field(default=None, ge=0, le=5)
    education: Education | None = None

    def record(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class MarketState(BaseModel):
    """The dashboard's market-state bar (``bre.market.latest_market_state`` fields plus the
    advisor's framing inputs). ``drawdown_pct`` and ``recovery_pct`` are fractions (``-0.12``);
    a magnitude above 1 is read as a percentage."""

    model_config = ConfigDict(extra="forbid")

    drawdown_pct: float | None = Field(default=None, ge=-100.0, le=100.0, description="signed drawdown from the trailing peak, fraction (or percent when |x| > 1)")
    duration_days: int | None = Field(default=None, ge=0)
    cause_frame: str | None = Field(default=None, max_length=500, description="free text: the stated cause of the fall")
    recovery_pct: float | None = Field(default=None, ge=0.0, le=100.0, description="recovery from the trough, fraction (or percent)")
    vix_bucket: VixBucket | None = None
    media_intensity: Level | float | None = Field(default=None, description="none/low/medium/high/extreme or a number in [0, 1]")
    social_cue_prevalence: Level | float | None = Field(default=None, description="none/low/medium/high/extreme or a number in [0, 1]")

    @field_validator("media_intensity", "social_cue_prevalence")
    @classmethod
    def _level_range(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not 0.0 <= float(v) <= 1.0:
            raise ValueError("a numeric level must lie in [0, 1]")
        return v


def _check_tags(tags: list[str]) -> list[str]:
    out = [t for t in tags if t != D.CONTEXT_NONE]
    if len(out) > 2:
        raise ValueError("at most two ordered context tags per scenario (the design's pair limit)")
    if len(set(out)) != len(out):
        raise ValueError("context tags must be distinct")
    return out


class PriorResponse(BaseModel):
    """One intake answer already given: the scenario and the sell/hold response (1 = sell)."""

    model_config = ConfigDict(extra="forbid")

    loss_pct: float | None = Field(default=None, ge=-1.0, le=0.0)
    context_tags: list[ContextTag] = Field(default_factory=list)
    question_order_id: QuestionOrder = "scenario-first"
    response: Literal[0, 1]
    tol_answer: Literal[0, 1] | None = Field(default=None, description="answer to the tolerance question in that item (1 = yes)")

    @field_validator("context_tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return _check_tags(v)


class ClientRef(BaseModel):
    """Optional client identity: predictions are logged under ``client_id`` (created when
    unknown, with ``display_label``), else under the anonymous API client."""

    client_id: str | None = Field(default=None, pattern=CLIENT_ID_PATTERN)
    display_label: str | None = Field(default=None, max_length=80, description="the only identifying text accepted")


class PredictRequest(ClientRef):
    model_config = ConfigDict(extra="forbid")

    covariates: Covariates = Field(default_factory=Covariates)
    loss_pct: float = Field(ge=-1.0, le=0.0, description="signed portfolio loss, e.g. -0.20; design grid -0.05 .. -0.30")
    context_tags: list[ContextTag] = Field(default_factory=list, description="ordered contexts, at most two")
    question_order_id: QuestionOrder = "scenario-first"
    tol_answer: Literal[0, 1] | None = Field(default=None, description="prior tolerance answer for a tolerance-first scenario")
    mix_weights: dict[ContextTag, float] | None = Field(default=None, description="weights of a mixed cause frame for the delta_mix term")
    prior_responses: list[PriorResponse] | None = Field(default=None, description="intake answers to refine the subject's random effects first")

    @field_validator("context_tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return _check_tags(v)

    @field_validator("mix_weights")
    @classmethod
    def _weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return None
        v = {k: float(w) for k, w in v.items() if k != D.CONTEXT_NONE}
        if not v or any(w < 0 for w in v.values()) or sum(v.values()) <= 0:
            raise ValueError("mix_weights must be non-negative with a positive sum over non-null contexts")
        return v


class ProfileRequest(ClientRef):
    model_config = ConfigDict(extra="forbid")

    covariates: Covariates = Field(default_factory=Covariates)
    prior_responses: list[PriorResponse] | None = None
    include_book_r2: bool = Field(default=True, description="compute the risk-score R^2 across the served book")


class ClientIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(pattern=CLIENT_ID_PATTERN)
    display_label: str | None = Field(default=None, max_length=80)
    covariates: Covariates = Field(default_factory=Covariates)


class ScoreBookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clients: list[ClientIn] = Field(min_length=1, max_length=5000)
    market_state: MarketState
    target: float = Field(default=0.25, gt=0.0, lt=1.0, description="drawdown-capacity target P(sell)")
    register_unknown: bool = Field(default=False, description="create client rows for unknown client_ids (else their predictions are logged under the batch client)")

    @field_validator("clients")
    @classmethod
    def _unique(cls, v: list[ClientIn]) -> list[ClientIn]:
        """A field validator (not a model validator) so the 422 names ``clients`` in ``loc``."""
        ids = [c.client_id for c in v]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"client_id values must be unique; duplicated: {dup}")
        return v


class RankRequest(ClientRef):
    model_config = ConfigDict(extra="forbid")

    covariates: Covariates = Field(default_factory=Covariates)
    market_state: MarketState
    prior_responses: list[PriorResponse] | None = None


# ---------------------------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------------------------


class Served(BaseModel):
    """Fields every response carries."""

    model_version: str
    model_type: str
    family: str
    synthetic: bool = Field(description="true when the served model was trained on synthetic data")
    demo_mode: bool
    wording: str = "predicted probability of selling"


class Interference(BaseModel):
    ltp: float | None = None
    order_effect: float | None = None
    mix: float | None = None
    definitions: str | None = None
    note: str | None = None


class TopDriver(BaseModel):
    driver: str | None
    delta_p: float
    p_without: float | None = None
    candidates: dict[str, float] | None = None
    note: str | None = None


class Calibration(BaseModel):
    n_responses: int | None
    status: str


class PredictResponse(Served):
    p: float = Field(description="predicted probability of selling")
    ci80: list[float] | None
    ci95: list[float] | None
    interval_note: str
    n_draws: int
    interference: Interference
    top_driver: TopDriver
    n_calibration: dict[str, Calibration]
    scenario: dict[str, Any]
    client_id: str
    refined: dict[str, Any] | None = None
    prediction_log_id: int


class ProfileResponse(Served):
    subject_id: str
    client_id: str
    known_subject: bool
    baseline: dict[str, Any]
    sensitivities: dict[str, Any]
    consistency: dict[str, Any]
    risk_score: dict[str, Any]
    refined: dict[str, Any] | None = None
    prediction_log_id: int


class ScoreRow(BaseModel):
    client_id: str
    display_label: str | None
    p_sell: float
    p_baseline: float
    change_vs_baseline: float
    ci80_low: float | None
    ci80_high: float | None
    top_driver: str | None
    top_driver_delta: float
    suggested_intervention: str | None
    suggested_delta: float | None
    drawdown_capacity: float
    capacity_status: str
    status: str
    known_client: bool
    synthetic: bool
    prediction_log_id: int | None = None


class ScoreBookResponse(Served):
    rows: list[ScoreRow]
    scenario: dict[str, Any]
    n_calibration: dict[str, Calibration]
    meta: dict[str, Any]


class RankedIntervention(BaseModel):
    rank: int
    name: str
    id: int | None
    script: str | None
    transform: dict[str, Any]
    p_before: float
    p_after: float
    delta_p: float
    delta_ci80: list[float] | None
    delta_ci95: list[float] | None
    contexts_after: list[str]
    question_order_after: str
    notes: list[str]
    label: str
    no_change: bool


class RankResponse(Served):
    client_id: str
    scenario: dict[str, Any]
    p_before: float
    ci80_before: list[float] | None
    ranked: list[RankedIntervention]
    inactive: list[str]
    label: str
    n_calibration: dict[str, Calibration]
    prediction_log_id: int


class ModelInfo(Served):
    version: str
    n_params: int
    n_params_population: int | None
    design_version: str
    created_at: str
    promoted_at: str | None
    metrics: dict[str, Any]
    calibrated_contexts: dict[str, Any]
    calibration_rule: str
    training_data_refs: list[str]
    provenance: dict[str, Any]
    is_synthetic_training: bool
    interval_source: str
    decision_rule: str
    evaluation_status: str
    notes: str
    n_draws: int
    loss_response_monotone_share: float | None = Field(description="share of the served book's clients whose predicted P(sell | L, no context) is non-decreasing over the five design loss levels")
    loss_response: dict[str, Any] = Field(description="the loss-response diagnostic of the served book: monotone_share, n_clients, loss_levels, mean_p_sell_by_loss, p_sell_by_loss_ci80, definition")
    capacity_definition: str = Field(description="the first-crossing definition of the behavioral drawdown capacity at the default target")


class Health(BaseModel):
    status: str
    model_version: str | None
    model_type: str | None
    demo_mode: bool
    synthetic: bool
    db: str
    uptime_s: float


class ClientOut(BaseModel):
    client_id: str
    display_label: str | None
    covariates: dict[str, Any]
    created_at: str
    n_responses: int
    known_to_model: bool
    synthetic: bool


class InterventionOut(BaseModel):
    id: int
    name: str
    script: str
    mapped_context_transform: dict[str, Any]
    active: bool
    label: str


class MarketResponse(BaseModel):
    market_state: dict[str, Any]
    scenario: dict[str, Any]
    demo_mode: bool


__all__ = [
    "CLIENT_ID_PATTERN",
    "Calibration",
    "ClientIn",
    "ClientOut",
    "ClientRef",
    "Covariates",
    "Health",
    "Interference",
    "InterventionOut",
    "MarketResponse",
    "MarketState",
    "ModelInfo",
    "PredictRequest",
    "PredictResponse",
    "PriorResponse",
    "ProfileRequest",
    "ProfileResponse",
    "RankRequest",
    "RankResponse",
    "RankedIntervention",
    "ScoreBookRequest",
    "ScoreBookResponse",
    "ScoreRow",
    "Served",
    "TopDriver",
]
