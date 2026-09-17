"""Market series for the dashboard's market-state bar and the payout-path table (CATALOG.md M).

Two cached series live under ``data/raw/market/``: the Shiller S&P 500 monthly table
(``sp500_data.csv``; ``SP500`` is the *monthly average of daily closes*, not a month-end close)
and the CBOE VIX daily table (``vix_vix-daily.csv``). Neither is decision data; nothing here goes
through the DecisionEvent schema.

Design choices recorded here:

* :data:`VIX_BUCKETS` -- the volatility regime of the market-state bar: ``low`` [0, 15),
  ``medium`` [15, 25), ``high`` [25, 35), ``extreme`` [35, inf). Lower edge inclusive, upper
  edge exclusive. The edges are a DESIGN CHOICE (round numbers near the long-run VIX quartiles
  and the levels commonly quoted as elevated / crisis), not an estimate; change them here only.
* :func:`drawdown_episodes` works on the monthly-average series, so every episode it returns
  carries ``resolution = "monthly-average"`` (peaks and troughs of daily closes are deeper and
  dated differently; the owner must download a daily close series for the PROLIFIC.md payout
  mechanism, see CATALOG.md M).
* :func:`latest_market_state` tries ``yfinance`` (an optional extra, blocked from the build
  session) with a hard 5-second cap and otherwise falls back to the cached files, always saying
  which (``source``) and as of when (``as_of``). It never raises on a network failure.
"""

from __future__ import annotations

import math
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bre.schema import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
MARKET_DIR_NAME = "market"
SP500_FILE = "sp500_data.csv"
VIX_FILE = "vix_vix-daily.csv"

VIX_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 15.0, "low"),
    (15.0, 25.0, "medium"),
    (25.0, 35.0, "high"),
    (35.0, math.inf, "extreme"),
)
"""DESIGN CHOICE: ``(lower_inclusive, upper_exclusive, label)`` VIX close buckets (module docstring)."""

VIX_BUCKET_LABELS: tuple[str, ...] = tuple(b[2] for b in VIX_BUCKETS)

LOOKBACK_MONTHS = 12
"""The market-state bar measures the drawdown from the peak of the trailing 12 months."""

RESOLUTION_MONTHLY = "monthly-average"
RESOLUTION_DAILY = "daily-close"
SOURCE_CACHED = "cached-file"
SOURCE_YFINANCE = "yfinance"
YFINANCE_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------------------------
# Cached series
# ---------------------------------------------------------------------------------------------


def load_sp500_monthly(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Monthly S&P 500 levels: columns ``date`` (first of month, datetime64) and ``sp500`` (float).

    Rows whose ``SP500`` is missing or not positive are dropped (the file carries zeros in the
    non-price columns after Shiller's last update; the price column itself is checked here).
    Sorted by date, unique dates.
    """
    path = Path(raw_dir) / MARKET_DIR_NAME / SP500_FILE
    raw = pd.read_csv(path)
    if "Date" not in raw.columns or "SP500" not in raw.columns:
        raise ValueError(f"{path.name}: expected columns Date and SP500")
    df = pd.DataFrame({"date": pd.to_datetime(raw["Date"]), "sp500": pd.to_numeric(raw["SP500"], errors="coerce")})
    df = df[df["sp500"] > 0].dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def load_vix_daily(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Daily VIX: columns ``date`` (datetime64), ``open``, ``high``, ``low``, ``close`` (float).

    Rows without a close are dropped; sorted by date, unique dates.
    """
    path = Path(raw_dir) / MARKET_DIR_NAME / VIX_FILE
    raw = pd.read_csv(path)
    needed = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = pd.DataFrame({"date": pd.to_datetime(raw["DATE"])})
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(raw[c.upper()], errors="coerce")
    df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def vix_bucket(close: object) -> str | None:
    """Bucket label of a VIX close per :data:`VIX_BUCKETS`; None for null, negative or non-numeric."""
    try:
        x = float(close)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x < 0:
        return None
    for lo, hi, label in VIX_BUCKETS:
        if lo <= x < hi:
            return label
    return None


# ---------------------------------------------------------------------------------------------
# Drawdown episodes
# ---------------------------------------------------------------------------------------------

EPISODE_COLUMNS = (
    "peak_date", "trough_date", "drawdown", "recovery_date", "months_to_trough", "months_to_recovery", "resolution",
)


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def drawdown_episodes(series: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    """Peak-to-trough drawdown episodes of a monthly level series.

    ``series`` is indexed by dates (any parseable index; sorted ascending here) with positive
    levels. Rule: a running peak is the highest level seen so far; the segment from a peak until
    the first later month whose level is at least the peak level (the *recovery*) is an episode
    when its lowest level (the *trough*) is at least ``threshold`` (a positive fraction, e.g.
    0.10 = 10%) below the peak. Returns one row per episode with ``peak_date``, ``trough_date``,
    ``drawdown`` (signed fraction ``trough / peak - 1``, e.g. -0.52), ``recovery_date`` (NaT when
    the peak has not been regained by the end of the series), ``months_to_trough`` (calendar
    months from peak to trough), ``months_to_recovery`` (calendar months from the trough to the
    recovery date; null while unrecovered) and ``resolution = "monthly-average"``. Ties for the
    trough take the earliest month.
    """
    if not (0 < threshold < 1):
        raise ValueError("threshold must be a fraction in (0, 1)")
    s = pd.Series(series).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if (s <= 0).any():
        raise ValueError("levels must be positive")
    dates = list(s.index)
    vals = s.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    if len(vals) == 0:
        return pd.DataFrame(columns=list(EPISODE_COLUMNS))
    peak_i = 0
    trough_i = 0
    for i in range(1, len(vals)):
        if vals[i] >= vals[peak_i]:
            # segment [peak_i, i) closes; i is the recovery month (and the new peak)
            dd = vals[trough_i] / vals[peak_i] - 1.0
            if trough_i > peak_i and dd <= -threshold:
                rows.append(_episode(dates, peak_i, trough_i, i, dd))
            peak_i, trough_i = i, i
        elif vals[i] < vals[trough_i]:
            trough_i = i
    dd = vals[trough_i] / vals[peak_i] - 1.0
    if trough_i > peak_i and dd <= -threshold:
        rows.append(_episode(dates, peak_i, trough_i, None, dd))
    out = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    out["months_to_recovery"] = out["months_to_recovery"].astype("Int64")
    out["months_to_trough"] = out["months_to_trough"].astype("int64") if len(out) else out["months_to_trough"].astype("Int64")
    return out


def _episode(dates: list, peak_i: int, trough_i: int, rec_i: int | None, dd: float) -> dict[str, Any]:
    return {
        "peak_date": dates[peak_i],
        "trough_date": dates[trough_i],
        "drawdown": float(dd),
        "recovery_date": dates[rec_i] if rec_i is not None else pd.NaT,
        "months_to_trough": _months_between(dates[peak_i], dates[trough_i]),
        "months_to_recovery": _months_between(dates[trough_i], dates[rec_i]) if rec_i is not None else pd.NA,
        "resolution": RESOLUTION_MONTHLY,
    }


# ---------------------------------------------------------------------------------------------
# Latest market state
# ---------------------------------------------------------------------------------------------


def _state_from_levels(levels: pd.Series, vix_close: float | None, vix_date: object, source: str, resolution: str) -> dict[str, Any]:
    """Drawdown from the trailing-12-month peak, duration and recovery from the trough."""
    levels = levels.dropna().sort_index()
    as_of = pd.Timestamp(levels.index[-1])
    start = as_of - pd.DateOffset(months=LOOKBACK_MONTHS)
    window = levels[levels.index >= start]
    if window.empty:
        window = levels.iloc[-1:]
    peak_date = pd.Timestamp(window.idxmax())
    peak = float(window.max())
    since_peak = window[window.index >= peak_date]
    trough_date = pd.Timestamp(since_peak.idxmin())
    trough = float(since_peak.min())
    level = float(levels.iloc[-1])
    return {
        "source": source,
        "resolution": resolution,
        "as_of": {"sp500": as_of.date().isoformat(), "vix": _date_str(vix_date)},
        "sp500_level": level,
        "peak_level": peak,
        "peak_date": peak_date.date().isoformat(),
        "trough_level": trough,
        "trough_date": trough_date.date().isoformat(),
        "drawdown_pct": level / peak - 1.0,
        "duration_days": int((as_of - peak_date).days),
        "recovery_pct": (level / trough - 1.0) if trough > 0 else 0.0,
        "vix_close": vix_close,
        "vix_bucket": vix_bucket(vix_close),
        "lookback_months": LOOKBACK_MONTHS,
    }


def _date_str(d: object) -> str | None:
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return None
    if isinstance(d, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(d).date().isoformat()
    return str(d)


def cached_market_state(raw_dir: Path = RAW_DIR) -> dict[str, Any]:
    """Market state from the cached files (``source = "cached-file"``, monthly-average resolution)."""
    sp = load_sp500_monthly(raw_dir)
    vix = load_vix_daily(raw_dir)
    levels = pd.Series(sp["sp500"].to_numpy(), index=pd.DatetimeIndex(sp["date"]))
    vix_close = float(vix["close"].iloc[-1]) if len(vix) else None
    vix_date = vix["date"].iloc[-1] if len(vix) else None
    state = _state_from_levels(levels, vix_close, vix_date, SOURCE_CACHED, RESOLUTION_MONTHLY)
    state["notes"] = [
        "S&P 500 levels are Shiller monthly averages of daily closes (data/raw/market/sp500_data.csv)",
        "VIX close is the last cached CBOE daily row (data/raw/market/vix_vix-daily.csv)",
    ]
    return state


def _fetch_yfinance(timeout_s: float) -> dict[str, Any]:
    """Fetch ~1 year of daily closes for ^GSPC and ^VIX; raises on any problem (caller catches)."""
    import yfinance as yf  # optional extra

    data = yf.download(["^GSPC", "^VIX"], period="1y", interval="1d", progress=False, timeout=timeout_s, group_by="ticker", auto_adjust=False)
    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no rows")
    gspc = data["^GSPC"]["Close"].dropna()
    vix = data["^VIX"]["Close"].dropna()
    if gspc.empty or vix.empty:
        raise RuntimeError("yfinance returned empty close series")
    levels = pd.Series(gspc.to_numpy(dtype=float), index=pd.DatetimeIndex(gspc.index).tz_localize(None))
    state = _state_from_levels(levels, float(vix.iloc[-1]), vix.index[-1], SOURCE_YFINANCE, RESOLUTION_DAILY)
    state["notes"] = ["daily closes from Yahoo Finance via yfinance (^GSPC, ^VIX)"]
    return state


def latest_market_state(raw_dir: Path = RAW_DIR, timeout_s: float = YFINANCE_TIMEOUT_S, allow_network: bool = True) -> dict[str, Any]:
    """Current market state for the dashboard bar; never raises on a network failure.

    Tries ``import yfinance`` and a fetch of ``^GSPC`` and ``^VIX`` in a daemon thread capped at
    ``timeout_s`` seconds; on ImportError, any exception, an empty result or a timeout it falls
    back to :func:`cached_market_state` and records the reason in ``fallback_reason``. Keys:
    ``source`` (``yfinance`` | ``cached-file``), ``resolution``, ``as_of`` (dates per series),
    ``sp500_level``, ``peak_level``/``peak_date`` (trailing 12 months), ``trough_level``/
    ``trough_date`` (since the peak), ``drawdown_pct`` (signed, level / peak - 1),
    ``duration_days`` (peak to as-of), ``recovery_pct`` (level / trough - 1), ``vix_close``,
    ``vix_bucket``, ``notes``.
    """
    reason: str | None = None
    if allow_network:
        result: dict[str, Any] = {}

        def run() -> None:
            try:
                result["state"] = _fetch_yfinance(timeout_s)
            except BaseException as exc:  # noqa: BLE001 - any failure means fallback
                result["error"] = f"{type(exc).__name__}: {exc}"

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_s + 1.0)
        if t.is_alive():
            reason = f"yfinance fetch exceeded {timeout_s:.0f}s"
        elif "state" in result:
            state = result["state"]
            state["fallback_reason"] = None
            return state
        else:
            reason = result.get("error", "yfinance unavailable")
    else:
        reason = "network disabled by caller"
    try:
        state = cached_market_state(raw_dir)
    except Exception as exc:  # noqa: BLE001 - the bar must still render
        state = {
            "source": SOURCE_CACHED, "resolution": RESOLUTION_MONTHLY, "as_of": {"sp500": None, "vix": None},
            "sp500_level": None, "peak_level": None, "peak_date": None, "trough_level": None, "trough_date": None,
            "drawdown_pct": None, "duration_days": None, "recovery_pct": None, "vix_close": None, "vix_bucket": None,
            "lookback_months": LOOKBACK_MONTHS, "notes": [f"cached files unreadable: {type(exc).__name__}: {exc}"],
        }
    state["fallback_reason"] = reason
    return state
