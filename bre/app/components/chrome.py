"""Page chrome shared by every dashboard page: page config, the API liveness check, the
persistent SYNTHETIC DATA banner (demo mode) and the wording/formatting helpers.

Wording rule (PLAN.md, Phase 6): every probability is a *predicted probability of selling*
under the served model; the dashboard never says that an investor "will" do anything. Every
displayed probability is shown with its interval and the number of training responses (N)
behind the calibration of the contexts involved.
"""

from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

from components import api_client as api

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA — demo mode. Every number on this screen comes from a model fitted on "
    "synthetic responses; no real investor is described here."
)
"""Shown on every screen in demo mode and written as the first line of every export."""

WORDING = "predicted probability of selling"
INTERVENTION_LABEL_FALLBACK = "predicted effect, not causally validated"
GUARDRAIL_LABEL = "allocation guardrail: the largest loss this client is predicted to sit through before the target probability is crossed; not a forecast of behavior"

CONTEXT_LABELS: dict[str, str] = {
    "none": "no context",
    "news:recession": "recession news",
    "news:technical": "technical-fault news",
    "social:friend_sells": "friend sells",
    "market:recovered_5pct": "5% recovery",
}
PAIR_SEPARATOR = " › "

DISPLAY_ORDER = ("news:recession", "news:technical", "social:friend_sells", "market:recovered_5pct")


def context_label(tag: str) -> str:
    return CONTEXT_LABELS.get(tag, tag)


def condition_label(tags: tuple[str, ...] | list[str]) -> str:
    tags = tuple(tags)
    return "no context" if not tags else PAIR_SEPARATOR.join(context_label(t) for t in tags)


def ordered_contexts(tags: list[str]) -> list[str]:
    """Display order of the served contexts: the documented order first, then the rest."""
    known = [t for t in DISPLAY_ORDER if t in tags]
    return known + sorted(t for t in tags if t not in known)


def _wait_for_api(base: str) -> dict[str, Any]:
    """``GET /health`` with retries while the service starts (``make dashboard`` launches it in
    the background; loading and warming the model takes about 15 s). Stops the script with the
    reason after ``BRE_API_WAIT_S`` seconds (default 90)."""
    wait_s = float(os.environ.get("BRE_API_WAIT_S", "90"))
    deadline = time.monotonic() + wait_s
    last = "no answer yet"
    with st.spinner(f"Connecting to the BRE API at {base} …"):
        while True:
            try:
                health = api.health(base)
                if health.get("status") == "ok":
                    return health
                last = f"status {health.get('status')}"
            except api.ApiError as exc:
                last = str(exc)
            if time.monotonic() >= deadline:
                break
            time.sleep(1.0)
    st.error(f"The BRE API at {base} did not become ready within {wait_s:.0f} s ({last}). Start it with `make api` or `make dashboard`, or point BRE_API_URL at a running service.")
    st.stop()
    raise RuntimeError("unreachable")


def page_setup(title: str) -> dict[str, Any]:
    """``st.set_page_config``, the liveness check and the demo banner. Returns ``GET /health``.
    Stops the script with an explanation when the service is unreachable."""
    st.set_page_config(page_title=f"BRE — {title}", layout="wide", initial_sidebar_state="expanded")
    base = api.base_url()
    health = _wait_for_api(base)
    st.session_state["demo_mode"] = bool(health.get("demo_mode", True))
    st.session_state["health"] = health
    if st.session_state["demo_mode"]:
        st.warning(SYNTHETIC_BANNER, icon="⚠️")
        st.sidebar.warning(SYNTHETIC_BANNER, icon="⚠️")
    st.title(title)
    st.caption(
        f"Model {health.get('model_version')} ({health.get('model_type')}) served by {base} · every probability is a "
        f"{WORDING} under that model · intervals and calibration N are shown with every number."
    )
    return health


def demo_mode() -> bool:
    return bool(st.session_state.get("demo_mode", True))


def stop_on_error(exc: Exception) -> None:
    st.error(f"API error: {exc}")
    st.stop()


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------


def fmt_p(p: Any, digits: int = 2) -> str:
    return "n/a" if p is None else f"{float(p):.{digits}f}"


def fmt_interval(ci: Any, digits: int = 2) -> str:
    if not ci or len(ci) != 2 or ci[0] is None:
        return "no interval"
    return f"[{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"


def fmt_signed(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else f"{float(x):+.{digits}f}"


def fmt_pct(x: Any, digits: int = 0) -> str:
    return "n/a" if x is None else f"{100 * float(x):.{digits}f}%"


def prob_text(p: Any, ci80: Any = None, ci95: Any = None) -> str:
    s = fmt_p(p)
    if ci80:
        s += f" (80% {fmt_interval(ci80)}"
        if ci95:
            s += f"; 95% {fmt_interval(ci95)}"
        s += ")"
    return s


def calibration_text(n_cal: dict[str, Any] | None) -> str:
    """``N`` per context of a scenario, e.g. ``recession news N=431 (calibrated)``."""
    if not n_cal:
        return "no calibration counts returned"
    parts = []
    for tag, entry in n_cal.items():
        n = entry.get("n_responses") if isinstance(entry, dict) else None
        status = entry.get("status", "") if isinstance(entry, dict) else ""
        parts.append(f"{context_label(tag)} N={n if n is not None else 'n/a'} ({status})")
    return "; ".join(parts)


def calibration_min_n(n_cal: dict[str, Any] | None) -> int | None:
    ns = [int(e.get("n_responses") or 0) for e in (n_cal or {}).values() if isinstance(e, dict)]
    return min(ns) if ns else None


__all__ = [
    "CONTEXT_LABELS", "GUARDRAIL_LABEL", "INTERVENTION_LABEL_FALLBACK", "SYNTHETIC_BANNER", "WORDING", "calibration_min_n",
    "calibration_text", "condition_label", "context_label", "demo_mode", "fmt_interval", "fmt_p", "fmt_pct", "fmt_signed",
    "ordered_contexts", "page_setup", "prob_text", "stop_on_error",
]
