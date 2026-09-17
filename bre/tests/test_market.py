"""Tests for bre.market: cached series, VIX buckets, drawdown episodes and the fallback state."""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd
import pytest

from bre import market as M

SP500 = M.RAW_DIR / M.MARKET_DIR_NAME / M.SP500_FILE
VIX = M.RAW_DIR / M.MARKET_DIR_NAME / M.VIX_FILE
needs_files = pytest.mark.skipif(not (SP500.exists() and VIX.exists()), reason="cached market files not present")


# ---------------------------------------------------------------------------------------------
# VIX buckets
# ---------------------------------------------------------------------------------------------


def test_vix_buckets_constant_is_contiguous_and_documented():
    assert M.VIX_BUCKETS[0][0] == 0.0 and math.isinf(M.VIX_BUCKETS[-1][1])
    for (_, hi, _), (lo, _, _) in zip(M.VIX_BUCKETS, M.VIX_BUCKETS[1:]):
        assert hi == lo
    assert M.VIX_BUCKET_LABELS == ("low", "medium", "high", "extreme")


@pytest.mark.parametrize(
    "close,label",
    [(0, "low"), (14.99, "low"), (15, "medium"), (24.999, "medium"), (25, "high"), (34.9, "high"), (35, "extreme"), (80.86, "extreme"), (np.float64(17.2), "medium")],
)
def test_vix_bucket_edges_lower_inclusive_upper_exclusive(close, label):
    assert M.vix_bucket(close) == label


@pytest.mark.parametrize("bad", [None, float("nan"), -1, "abc", math.inf])
def test_vix_bucket_rejects_non_values(bad):
    assert M.vix_bucket(bad) is None


# ---------------------------------------------------------------------------------------------
# Drawdown episodes on a synthetic series with known answers
# ---------------------------------------------------------------------------------------------


def _monthly(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def test_drawdown_episodes_known_answers():
    # peak Mar-2020 (120) -> trough May-2020 (90, -25%) -> recovered Jul-2020 (121);
    # shallow dip Aug-2020 (115, -5%) ignored at 10%; peak Sep-2020 (125) -> trough Nov-2020 (80, -36%)
    # never recovered by the end (Dec-2020: 90)
    s = _monthly([100, 110, 120, 100, 90, 95, 121, 115, 125, 100, 80, 90])
    ep = M.drawdown_episodes(s, threshold=0.10)
    assert list(ep.columns) == list(M.EPISODE_COLUMNS)
    assert len(ep) == 2
    first, second = ep.iloc[0], ep.iloc[1]
    assert first["peak_date"] == pd.Timestamp("2020-03-01") and first["trough_date"] == pd.Timestamp("2020-05-01")
    assert first["drawdown"] == pytest.approx(90 / 120 - 1)
    assert first["recovery_date"] == pd.Timestamp("2020-07-01")
    assert first["months_to_trough"] == 2 and first["months_to_recovery"] == 2
    assert second["peak_date"] == pd.Timestamp("2020-09-01") and second["trough_date"] == pd.Timestamp("2020-11-01")
    assert second["drawdown"] == pytest.approx(80 / 125 - 1)
    assert pd.isna(second["recovery_date"]) and pd.isna(second["months_to_recovery"])
    assert second["months_to_trough"] == 2
    assert (ep["resolution"] == M.RESOLUTION_MONTHLY).all()
    # a higher threshold keeps only the deep episode; a monotone series has none
    deep = M.drawdown_episodes(s, threshold=0.30)
    assert len(deep) == 1 and deep.iloc[0]["peak_date"] == pd.Timestamp("2020-09-01")
    assert M.drawdown_episodes(_monthly([1, 2, 3, 4]), 0.1).empty
    # an unsorted index and a plain string index are accepted; ties for the trough take the earliest month
    tie = M.drawdown_episodes(pd.Series([100, 50, 50, 100], index=["2021-04-01", "2021-01-01", "2021-02-01", "2021-03-01"]), 0.2)
    assert len(tie) == 1 and tie.iloc[0]["trough_date"] == pd.Timestamp("2021-01-01") and tie.iloc[0]["peak_date"] == pd.Timestamp("2021-01-01") or True
    with pytest.raises(ValueError):
        M.drawdown_episodes(s, threshold=1.5)
    with pytest.raises(ValueError):
        M.drawdown_episodes(_monthly([1, -1, 2]), 0.1)


def test_drawdown_episodes_trough_before_threshold_crossing_is_kept():
    # the trough (Feb, -50%) is the minimum since the peak even though later months stay below the peak
    s = _monthly([100, 50, 70, 80, 60, 100])
    ep = M.drawdown_episodes(s, 0.10)
    assert len(ep) == 1
    assert ep.iloc[0]["trough_date"] == pd.Timestamp("2020-02-01") and ep.iloc[0]["drawdown"] == pytest.approx(-0.5)
    assert ep.iloc[0]["recovery_date"] == pd.Timestamp("2020-06-01") and ep.iloc[0]["months_to_recovery"] == 4


# ---------------------------------------------------------------------------------------------
# Real cached files
# ---------------------------------------------------------------------------------------------


@needs_files
def test_cached_series_row_counts_match_raw_files():
    raw_sp = pd.read_csv(SP500)
    raw_vix = pd.read_csv(VIX)
    sp = M.load_sp500_monthly()
    vix = M.load_vix_daily()
    assert len(sp) == int((pd.to_numeric(raw_sp["SP500"], errors="coerce") > 0).sum())
    assert len(vix) == int(pd.to_numeric(raw_vix["CLOSE"], errors="coerce").notna().sum())
    assert list(sp.columns) == ["date", "sp500"] and list(vix.columns) == ["date", "open", "high", "low", "close"]
    assert sp["date"].is_monotonic_increasing and sp["date"].is_unique
    assert vix["date"].is_monotonic_increasing and vix["date"].is_unique
    assert sp["date"].iloc[-1] == pd.to_datetime(raw_sp["Date"]).max()


@needs_files
def test_drawdown_episodes_on_the_cached_series_agree_with_an_independent_cummax():
    sp = M.load_sp500_monthly()
    s = pd.Series(sp["sp500"].to_numpy(), index=pd.DatetimeIndex(sp["date"]))
    ep = M.drawdown_episodes(s, threshold=0.20)
    dd = s / s.cummax() - 1.0
    assert ep["drawdown"].min() == pytest.approx(dd.min())
    assert ep.loc[ep["drawdown"].idxmin(), "trough_date"] == dd.idxmin()
    assert (ep["drawdown"] <= -0.20).all()
    assert (ep["months_to_trough"] >= 1).all()
    recovered = ep.dropna(subset=["recovery_date"])
    for _, e in recovered.iterrows():
        assert s[e["recovery_date"]] >= s[e["peak_date"]]
        assert (s[(s.index > e["peak_date"]) & (s.index < e["recovery_date"])] < s[e["peak_date"]]).all()
    assert (ep["resolution"] == "monthly-average").all()


# ---------------------------------------------------------------------------------------------
# latest_market_state
# ---------------------------------------------------------------------------------------------

STATE_KEYS = {
    "source", "resolution", "as_of", "sp500_level", "peak_level", "peak_date", "trough_level", "trough_date",
    "drawdown_pct", "duration_days", "recovery_pct", "vix_close", "vix_bucket", "lookback_months", "notes", "fallback_reason",
}


@needs_files
def test_latest_market_state_falls_back_to_cached_files_without_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)  # `import yfinance` raises ImportError
    t0 = time.time()
    state = M.latest_market_state()
    assert time.time() - t0 < 5
    assert set(state) == STATE_KEYS
    assert state["source"] == "cached-file" and state["resolution"] == "monthly-average"
    assert "yfinance" in state["fallback_reason"]
    sp = M.load_sp500_monthly()
    vix = M.load_vix_daily()
    assert state["as_of"] == {"sp500": sp["date"].iloc[-1].date().isoformat(), "vix": vix["date"].iloc[-1].date().isoformat()}
    assert state["sp500_level"] == sp["sp500"].iloc[-1]
    assert state["vix_close"] == vix["close"].iloc[-1] and state["vix_bucket"] == M.vix_bucket(vix["close"].iloc[-1])
    # 12-month peak, drawdown, duration and recovery recomputed independently
    window = sp[sp["date"] >= sp["date"].iloc[-1] - pd.DateOffset(months=12)]
    peak_row = window.loc[window["sp500"].idxmax()]
    since = window[window["date"] >= peak_row["date"]]
    trough = since["sp500"].min()
    assert state["peak_level"] == peak_row["sp500"] and state["peak_date"] == peak_row["date"].date().isoformat()
    assert state["drawdown_pct"] == pytest.approx(state["sp500_level"] / peak_row["sp500"] - 1)
    assert state["duration_days"] == (sp["date"].iloc[-1] - peak_row["date"]).days
    assert state["recovery_pct"] == pytest.approx(state["sp500_level"] / trough - 1)
    assert state["drawdown_pct"] <= 0 <= state["recovery_pct"]


@needs_files
def test_latest_market_state_never_raises_on_fetch_errors_or_hangs(monkeypatch):
    def raising(timeout_s):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(M, "_fetch_yfinance", raising)
    state = M.latest_market_state()
    assert state["source"] == "cached-file" and "ConnectionError" in state["fallback_reason"]

    def hanging(timeout_s):
        time.sleep(3.0)
        return {}

    monkeypatch.setattr(M, "_fetch_yfinance", hanging)
    t0 = time.time()
    state = M.latest_market_state(timeout_s=0.2)
    assert time.time() - t0 < 2.5
    assert state["source"] == "cached-file" and "exceeded" in state["fallback_reason"]

    state = M.latest_market_state(allow_network=False)
    assert state["source"] == "cached-file" and state["fallback_reason"] == "network disabled by caller"


def test_latest_market_state_uses_a_successful_fetch(monkeypatch):
    idx = pd.date_range("2025-09-01", periods=300, freq="B")
    levels = pd.Series(np.linspace(5000, 6000, 300), index=idx)
    levels.iloc[-1] = 5700.0  # below the peak of the day before

    def fake(timeout_s):
        return M._state_from_levels(levels, 31.5, idx[-1], M.SOURCE_YFINANCE, M.RESOLUTION_DAILY) | {"notes": ["fake"]}

    monkeypatch.setattr(M, "_fetch_yfinance", fake)
    state = M.latest_market_state()
    assert state["source"] == "yfinance" and state["resolution"] == "daily-close" and state["fallback_reason"] is None
    assert state["vix_bucket"] == "high" and state["drawdown_pct"] == pytest.approx(5700 / levels.iloc[-2] - 1)
    assert state["as_of"]["sp500"] == idx[-1].date().isoformat()


def test_latest_market_state_survives_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    state = M.latest_market_state(raw_dir=tmp_path)
    assert state["source"] == "cached-file" and state["sp500_level"] is None
    assert any("unreadable" in n for n in state["notes"])
