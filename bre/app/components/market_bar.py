"""The market-state bar (Phase 6, page 1): rendered in the sidebar of every page, state kept in
``st.session_state["market"]``. Every field is editable; the payload is sent as the API's
``MarketState`` and the API decides how the state maps to a scenario (loss clipping, cause-frame
matching with its ``weak_match`` flag, which contexts enter and in which order). The bar shows
the mapping the API returned, never a mapping computed here.

Fields: index drawdown % (-50 to +20), drawdown duration (days), cause frame (a calibrated news
context from ``GET /model`` or free text), realized recovery so far %, volatility regime (VIX
bucket), media intensity, social-cue prevalence. "Load current market" calls ``GET /market``
and fills the first four fields (the advisor can override every one of them afterwards).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from components import api_client as api
from components import chrome

NONE_CHOICE = "(no stated cause)"
FREE_TEXT_CHOICE = "free text…"

LEVELS_FALLBACK = ["none", "low", "medium", "high", "extreme"]
VIX_FALLBACK = ["low", "medium", "high", "extreme"]

DRAWDOWN_RANGE = (-50.0, 20.0)
RECOVERY_RANGE = (0.0, 100.0)

DEFAULT_MARKET: dict[str, Any] = {
    "drawdown_pct": -10.0,  # percent (UI units); sent to the API as a fraction
    "duration_days": 30,
    "cause_choice": NONE_CHOICE,
    "cause_frame": "",
    "recovery_pct": 0.0,  # percent
    "vix_bucket": "medium",
    "media_intensity": "medium",
    "social_cue_prevalence": "low",
}

FIELDS = tuple(DEFAULT_MARKET)


def _key(field: str) -> str:
    return f"ms_{field}"


def market() -> dict[str, Any]:
    m = st.session_state.setdefault("market", dict(DEFAULT_MARKET))
    for k, v in DEFAULT_MARKET.items():
        m.setdefault(k, v)
    return m


def _sync(field: str) -> None:
    market()[field] = st.session_state[_key(field)]


def _seed_widget(field: str, options: list[str] | None = None) -> None:
    """Initialise a widget key from the store (once), keeping the value inside the options."""
    m = market()
    if options is not None and m[field] not in options:
        m[field] = options[0]
    if _key(field) not in st.session_state:
        st.session_state[_key(field)] = m[field]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _load_market(base: str, vix_options: list[str]) -> None:
    """``GET /market`` into the store and the widget keys (button callback)."""
    try:
        resp = api.market(base)
    except api.ApiError as exc:
        st.session_state["market_loaded"] = {"error": str(exc)}
        return
    ms = resp.get("market_state") or {}
    m = market()
    if ms.get("drawdown_pct") is not None:
        m["drawdown_pct"] = _clip(round(100.0 * float(ms["drawdown_pct"]), 1), *DRAWDOWN_RANGE)
    if ms.get("duration_days") is not None:
        m["duration_days"] = int(ms["duration_days"])
    if ms.get("recovery_pct") is not None:
        m["recovery_pct"] = _clip(round(100.0 * float(ms["recovery_pct"]), 1), *RECOVERY_RANGE)
    if ms.get("vix_bucket") in vix_options:
        m["vix_bucket"] = ms["vix_bucket"]
    for f in ("drawdown_pct", "duration_days", "recovery_pct", "vix_bucket"):
        st.session_state[_key(f)] = m[f]
    st.session_state["market_loaded"] = {
        "source": ms.get("source"), "resolution": ms.get("resolution"), "as_of": ms.get("as_of"),
        "fallback_reason": ms.get("fallback_reason"), "notes": ms.get("notes") or [],
        "sp500_level": ms.get("sp500_level"), "vix_close": ms.get("vix_close"), "api_scenario": resp.get("scenario"),
    }


def cause_options(model_info: dict[str, Any]) -> list[str]:
    """Calibrated news contexts of the served model (the API matches cause frames to these)."""
    table = model_info.get("calibrated_contexts") or {}
    tags = [t for t, e in table.items() if t.startswith("news:") and str(e.get("status", "")) == "calibrated"]
    return chrome.ordered_contexts(tags)


def payload() -> dict[str, Any]:
    """The ``MarketState`` body sent with every prediction request (fractions, not percents)."""
    m = market()
    dd = float(m["drawdown_pct"])
    cause: str | None
    if m["cause_choice"] == NONE_CHOICE:
        cause = None
    elif m["cause_choice"] == FREE_TEXT_CHOICE:
        cause = str(m["cause_frame"]).strip() or None
    else:
        cause = str(m["cause_choice"])
    return {
        "drawdown_pct": min(dd, 0.0) / 100.0,  # an index above its trailing peak has no drawdown
        "duration_days": int(m["duration_days"]),
        "cause_frame": cause,
        "recovery_pct": float(m["recovery_pct"]) / 100.0,
        "vix_bucket": m["vix_bucket"],
        "media_intensity": m["media_intensity"],
        "social_cue_prevalence": m["social_cue_prevalence"],
    }


def render(model_info: dict[str, Any]) -> dict[str, Any]:
    """Render the bar in the sidebar and return :func:`payload`."""
    base = api.base_url()
    market()
    levels = api.enum_values(base, "MarketState", "media_intensity", LEVELS_FALLBACK)
    vix_options = api.enum_values(base, "MarketState", "vix_bucket", VIX_FALLBACK)
    causes = cause_options(model_info)
    choices = [NONE_CHOICE] + causes + [FREE_TEXT_CHOICE]

    with st.sidebar:
        st.subheader("Market state")
        st.caption("Every field is editable and every page recomputes on change. The API maps this state to the scenario it scores.")
        if api.has_path(base, "/market"):
            st.button("Load current market", on_click=_load_market, args=(base, vix_options), key="ms_load", help="GET /market: the service's cached or live market state (advisor overrides allowed).")
        else:
            st.button("Load current market", disabled=True, key="ms_load", help="disabled: the API has no /market endpoint")
            st.caption("Load disabled: the API has no /market endpoint.")
        info = st.session_state.get("market_loaded")
        if info:
            if info.get("error"):
                st.error(info["error"])
            else:
                as_of = info.get("as_of") or {}
                st.caption(
                    f"Loaded from {info.get('source')} ({info.get('resolution')}), S&P 500 as of {as_of.get('sp500')}, "
                    f"VIX as of {as_of.get('vix')}" + (f"; fallback: {info['fallback_reason']}" if info.get("fallback_reason") else "")
                )

        _seed_widget("drawdown_pct")
        st.slider("Index drawdown from trailing peak (%)", DRAWDOWN_RANGE[0], DRAWDOWN_RANGE[1], step=0.5, format="%.1f%%", key=_key("drawdown_pct"), on_change=_sync, args=("drawdown_pct",))
        _seed_widget("duration_days")
        st.number_input("Drawdown duration (days)", min_value=0, max_value=3650, step=1, key=_key("duration_days"), on_change=_sync, args=("duration_days",), help="Carried to the API; the served design has no elapsed-time context, so it does not change the prediction.")
        _seed_widget("cause_choice", choices)
        st.selectbox("Cause frame", choices, key=_key("cause_choice"), on_change=_sync, args=("cause_choice",), format_func=lambda t: t if t in (NONE_CHOICE, FREE_TEXT_CHOICE) else f"{chrome.context_label(t)} ({t}, calibrated)", help="Calibrated news contexts of the served model (GET /model), or free text the API maps to the nearest calibrated context.")
        if market()["cause_choice"] == FREE_TEXT_CHOICE:
            _seed_widget("cause_frame")
            st.text_input("Stated cause (free text)", key=_key("cause_frame"), on_change=_sync, args=("cause_frame",), max_chars=500, placeholder="e.g. analysts now expect a recession after weak earnings")
        _seed_widget("recovery_pct")
        st.slider("Realized recovery from the trough so far (%)", RECOVERY_RANGE[0], RECOVERY_RANGE[1], step=0.5, format="%.1f%%", key=_key("recovery_pct"), on_change=_sync, args=("recovery_pct",))
        _seed_widget("vix_bucket", vix_options)
        st.selectbox("Volatility regime (VIX bucket)", vix_options, key=_key("vix_bucket"), on_change=_sync, args=("vix_bucket",))
        _seed_widget("media_intensity", levels)
        st.selectbox("Media intensity", levels, key=_key("media_intensity"), on_change=_sync, args=("media_intensity",))
        _seed_widget("social_cue_prevalence", levels)
        st.selectbox("Social-cue prevalence", levels, key=_key("social_cue_prevalence"), on_change=_sync, args=("social_cue_prevalence",))
    return payload()


def render_scenario(scenario: dict[str, Any], n_calibration: dict[str, Any] | None, *, heading: str = "Scenario the model scores under this market state") -> None:
    """The API's mapping of the market state: loss, contexts, cause-frame match, notes."""
    if not scenario:
        return
    st.markdown(f"**{heading}**")
    tags = list(scenario.get("context_tags") or [])
    loss = scenario.get("loss_pct")
    raw = scenario.get("loss_pct_raw")
    parts = [f"loss {chrome.fmt_pct(abs(loss)) if loss is not None else 'n/a'}" + (f" (clipped from {chrome.fmt_pct(abs(raw), 1)})" if scenario.get("loss_clipped") and raw is not None else ""),
             f"contexts: {chrome.condition_label(tags)}",
             f"question order: {scenario.get('question_order_id')}"]
    cause = scenario.get("cause_context")
    if cause:
        parts.append(f"cause frame → {chrome.context_label(cause)} ({cause}, similarity {float(scenario.get('cause_similarity') or 0):.2f})")
    st.write(" · ".join(parts))
    if scenario.get("weak_match"):
        m = market()
        text = m["cause_frame"] if m["cause_choice"] == FREE_TEXT_CHOICE else m["cause_choice"]
        st.warning(
            f"Weak match: the cause frame \"{text}\" resembles none of the calibrated news contexts "
            f"(similarity {float(scenario.get('cause_similarity') or 0):.2f}); the API still applies {chrome.context_label(cause) if cause else 'the nearest context'}, so read these numbers with care.",
            icon="⚠️",
        )
    if scenario.get("dropped_contexts"):
        st.warning(f"The design holds at most two contexts; dropped: {', '.join(chrome.context_label(t) for t in scenario['dropped_contexts'])}.")
    notes = list(scenario.get("notes") or [])
    if n_calibration:
        notes.append("calibration: " + chrome.calibration_text(n_calibration))
    if notes:
        st.caption(" · ".join(notes))


__all__ = ["DEFAULT_MARKET", "FIELDS", "FREE_TEXT_CHOICE", "NONE_CHOICE", "cause_options", "market", "payload", "render", "render_scenario"]
