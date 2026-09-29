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



# ---------------------------------------------------------------------------------------------
# Phase 6 additions: settings, editor documents, intake, registry, transparency, retrain
# ---------------------------------------------------------------------------------------------


class SettingsPatch(BaseModel):
    """``PUT /settings`` body: any subset of the settings keys (``api.store.DEFAULT_SETTINGS``
    documents the defaults and ``SETTINGS_RULES`` the ranges; unknown keys are refused)."""

    model_config = ConfigDict(extra="forbid")

    alert_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    status_thresholds: dict[str, float] | None = Field(default=None, description="{elevated, high}, 0 <= elevated <= high <= 1")
    capacity_target: float | None = Field(default=None, gt=0.0, lt=1.0)
    prior_strength: float | None = Field(default=None, gt=0.0)
    typical_crisis_contexts: list[str] | None = Field(default=None, max_length=2)
    demo_mode: bool | None = None
    retrain: dict[str, Any] | None = Field(default=None, description="{steps, restarts, n_samples, val_frac, seed}")


class SettingsOut(BaseModel):
    settings: dict[str, Any]
    defaults: dict[str, Any]
    rules: dict[str, str]
    version: int
    updated_at: str | None
    demo_mode: bool = Field(description="the effective demo mode: forced by the setting, or by a synthetic model / database")
    demo_mode_forced_by_data: bool


class ContextIn(BaseModel):
    """``POST /contexts``: add a context or write a new version of one (retire with ``active: false``)."""

    model_config = ConfigDict(extra="forbid")

    tag: str = Field(pattern=r"^[A-Za-z0-9_]+:[A-Za-z0-9_.+\-]+$", max_length=80, description="namespace:value, e.g. news:tariff_shock")
    kind: str = Field(default="news", max_length=40)
    display: str = Field(max_length=2000, description="the sentence shown in the intake battery")
    active: bool = True
    triggers_delay: bool | None = None
    note: str | None = Field(default=None, max_length=500)


class ContextOut(BaseModel):
    tag: str
    kind: str
    display: str
    active: bool
    triggers_delay: bool | None = None
    note: str | None = None
    version: int
    updated_at: str
    actor: str
    in_design: bool = Field(description="one of the shared design's contexts (bre.design.NONNULL_CONTEXTS)")
    in_served_vocabulary: bool
    calibration: dict[str, Any] = Field(description="from the active artifact: n_responses, status, theta, theta_ci95, min_responses")


class ScenarioIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(max_length=80)
    text: str = Field(max_length=4000)
    note: str | None = Field(default=None, max_length=500)


class ScenarioOut(BaseModel):
    key: str
    text: str
    note: str | None = None
    version: int
    updated_at: str
    actor: str


class ScenariosOut(BaseModel):
    texts: dict[str, ScenarioOut]
    keys: list[str]
    history: list[dict[str, Any]] | None = None
    design_version: str
    note: str


class InterventionPut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=80)
    script: str | None = Field(default=None, max_length=4000)
    mapped_context_transform: dict[str, Any] | None = None
    active: bool | None = None
    note: str | None = Field(default=None, max_length=500)


class InterventionVersionOut(InterventionOut):
    version: int
    updated_at: str | None
    history: list[dict[str, Any]] | None = None


class IntakeResponsesIn(BaseModel):
    """``POST /clients/{client_id}/responses``: schema rows of one intake session plus the
    randomized assignment and the session record."""

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(min_length=1, max_length=2000, description="DecisionEvent rows (bre.schema.COLUMNS), one per elicited decision")
    assignment: dict[str, Any] = Field(default_factory=dict, description="bre.design.BatteryAssignment.to_dict() plus the presentation order and repeats")
    session: dict[str, Any] = Field(default_factory=dict, description="form, seed, consent, timings, text versions shown")
    display_label: str | None = Field(default=None, max_length=80)


class IntakeResponsesOut(BaseModel):
    client_id: str
    session_id: str
    n_rows: int
    is_synthetic: bool
    consent_training: bool
    dataset: str
    document_version: int
    note: str
    demo_mode: bool


class IntakeConfig(BaseModel):
    instrument_url: str
    instrument_url_source: str
    battery_version: str
    design_version: str
    forms: dict[str, Any]
    url_params: dict[str, Any]
    file_names: dict[str, Any]
    intake: dict[str, Any] = Field(default_factory=dict, description="battery.json intake texts: consent, instructions, covariates, financial_literacy_quiz, questions, delay_page")
    demo_mode: bool
    storage_note: str


class RegistryRow(BaseModel):
    version: str
    model_type: str
    family: str | None
    is_active: bool
    promoted_at: str | None
    created_at: str | None
    is_synthetic_training: bool | None
    n_params: int | None
    train_nll: float | None
    held_out_nll: float | None
    training_data_refs: list[str]
    artifact_dir: str | None
    artifact_exists: bool
    notes: str | None = None


class RegistryOut(BaseModel):
    active_version: str
    rows: list[RegistryRow]
    selectable_model_types: list[str]
    admin: bool
    admin_rule: str


class ActivateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(max_length=120)


class ActivateOut(BaseModel):
    activated: str
    previous: str | None
    model_type: str
    demo_mode: bool
    synthetic: bool


class RetrainIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: str | None = Field(default=None, description="Q2, Q4 or B2 (default: the active model's type)")
    steps: int | None = Field(default=None, ge=1, le=5000)
    restarts: int | None = Field(default=None, ge=1, le=10)
    n_samples: int | None = Field(default=None, ge=0, le=500)
    seed: int | None = None
    n_boot: int | None = Field(default=None, ge=0, le=2000)


class RetrainOut(BaseModel):
    run_id: str
    status: str
    promoted: bool
    reason: str | None
    version: str | None
    model_type: str
    nll_new: float | None
    nll_reference: float | None
    recorded_reference_nll: float | None
    active_version: str
    demo_mode: bool
    result: dict[str, Any]


class TransparencyOut(BaseModel):
    model_version: str
    model_type: str
    family: str
    n_params: int
    n_params_population: int | None
    is_synthetic_training: bool
    demo_mode: bool
    decision_rule: str
    verdict: dict[str, Any] | None = Field(description="reports/phase4/verdict.json when it exists")
    verdict_status: str = Field(description='the verdict sentence, or exactly "Decision rule not yet run: no real-data numbers are shown"')
    phase4_metrics: dict[str, Any] | None = Field(description="held-out metrics of Phase 4; null until the verdict exists")
    real_data_numbers_shown: bool
    training_metrics: dict[str, Any]
    calibration_plot: dict[str, Any] | None
    n_target: dict[str, Any] | None
    provenance: list[dict[str, Any]]
    model_card: str | None
    model_card_path: str | None
    retrain_rule: str
    wording: str = "predicted probability of selling"


class ExportOut(BaseModel):
    client: dict[str, Any]
    responses: list[dict[str, Any]]
    predictions_log: list[dict[str, Any]]
    intervention_log: list[dict[str, Any]]
    intake_sessions: list[dict[str, Any]]
    exported_at: str
    demo_mode: bool
    synthetic: bool


class DeleteOut(BaseModel):
    client_id: str
    deleted: dict[str, int]
    audit_log_id: int


class ResponseRowOut(BaseModel):
    id: int
    dataset: str
    session_id: str
    timestamp: str | None
    position_in_session: int
    scenario_id: str
    loss_pct: float | None
    context_tags: list[str]
    question_order_id: str | None
    elicitation_type: str
    response: float
    response_time_ms: float | None
    outcome_behavior: dict[str, Any] | None
    consent_training: bool
    is_synthetic: bool
    source_row_ref: str


__all__ = [
    "ActivateIn",
    "ActivateOut",
    "CLIENT_ID_PATTERN",
    "ContextIn",
    "ContextOut",
    "DeleteOut",
    "ExportOut",
    "IntakeConfig",
    "IntakeResponsesIn",
    "IntakeResponsesOut",
    "InterventionPut",
    "InterventionVersionOut",
    "RegistryOut",
    "RegistryRow",
    "ResponseRowOut",
    "RetrainIn",
    "RetrainOut",
    "ScenarioIn",
    "ScenarioOut",
    "ScenariosOut",
    "SettingsOut",
    "SettingsPatch",
    "TransparencyOut",
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
