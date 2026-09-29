"""Matplotlib figures of the dashboard (no other plotting dependency).

One palette for every chart: series blue for single-series marks, orange for the second
category (sell vs hold), a single-hue blue ramp for magnitudes (the heatmap), recessive grey
chrome. Every axis is labelled; probabilities are always titled as the predicted probability
of selling.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES_1 = "#2a78d6"  # blue: single series / hold
SERIES_2 = "#eb6834"  # orange: second category / sell
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
STATUS_CRITICAL = "#d03b3b"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CMAP = LinearSegmentedColormap.from_list("bre_blue", SEQUENTIAL)

P_LABEL = "predicted probability of selling"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    return buf.getvalue()


def heatmap(p: pd.DataFrame, title: str) -> plt.Figure:
    """``p``: rows = loss labels, columns = condition labels, values in [0, 1]. Cells are
    annotated because the figure has no hover; the table beside it carries the intervals."""
    values = p.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * p.shape[1]), 0.55 * p.shape[0] + 1.6))
    im = ax.imshow(values, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(p.shape[1]))
    ax.set_xticklabels([str(c) for c in p.columns], rotation=55, ha="right", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(p.shape[0]))
    ax.set_yticklabels([str(r) for r in p.index], fontsize=8, color=INK_2)
    ax.set_xlabel("context sequence (singles and ordered pairs)", color=INK_2, fontsize=8)
    ax.set_ylabel("portfolio loss", color=INK_2, fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(np.arange(-0.5, p.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, p.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(P_LABEL, fontsize=7.5, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def sensitivity_bars(rows: list[dict[str, Any]], loss_label: str) -> plt.Figure:
    """``rows``: label, delta_p, delta_ci80 (lo, hi), theta_norm, theta_ci80. Two panels:
    the predicted change in P(sell) when the context is applied singly at the risk-score loss,
    and the rotation magnitude |theta_c| of the context (a population parameter)."""
    n = len(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.5 * n + 1.4), sharey=True)
    y = np.arange(n)
    labels = [r["label"] for r in rows]
    for ax, key, ci_key, xlabel in ((ax1, "delta_p", "delta_ci80", f"change in {P_LABEL} at {loss_label} (80% interval)"), (ax2, "theta_norm", "theta_ci80", "context rotation |theta_c| (80% interval)")):
        _style(ax)
        vals = np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])
        ci = [r.get(ci_key) for r in rows]
        lo = np.array([np.nan if c is None else float(c[0]) for c in ci])
        hi = np.array([np.nan if c is None else float(c[1]) for c in ci])
        ax.barh(y, vals, height=0.5, color=SERIES_1, edgecolor="none")
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(vals[ok], y[ok], xerr=np.vstack([np.clip(vals[ok] - lo[ok], 0, None), np.clip(hi[ok] - vals[ok], 0, None)]), fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=3)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
    ax1.set_ylabel("context (applied singly)", fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def capacity_curve(losses_pct: list[float], probs: list[float], target: float, capacity_pct: float | None, condition: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    _style(ax)
    ax.plot(losses_pct, probs, color=SERIES_1, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(target, color=STATUS_CRITICAL, linewidth=1, linestyle="-")
    ax.text(losses_pct[0], target, f" target {target:.2f}", color=STATUS_CRITICAL, fontsize=8, va="bottom")
    if capacity_pct is not None and capacity_pct > 0:
        ax.axvline(capacity_pct, color=MUTED, linewidth=1)
        ax.text(capacity_pct, 0.98, f" capacity {capacity_pct:.0f}%", color=INK_2, fontsize=8, va="top")
    ax.set_ylim(0, 1)
    ax.set_xlabel("portfolio loss (%)", fontsize=8)
    ax.set_ylabel(P_LABEL, fontsize=8)
    ax.set_title(f"under {condition}", fontsize=9, loc="left")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def response_timeline(rows: list[dict[str, Any]]) -> plt.Figure:
    """Sell/hold answers of the intake battery in order (x = item position, y = loss magnitude);
    tolerance answers are drawn at loss 0."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    _style(ax)
    sells = [(r["x"], r["y"]) for r in rows if r["kind"] == "sell"]
    holds = [(r["x"], r["y"]) for r in rows if r["kind"] == "hold"]
    tol = [(r["x"], r["y"]) for r in rows if r["kind"].startswith("tolerance")]
    if holds:
        ax.scatter(*zip(*holds), s=48, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5, label="hold", zorder=3)
    if sells:
        ax.scatter(*zip(*sells), s=48, color=SERIES_2, edgecolor=SURFACE, linewidth=1.5, label="sell", zorder=3)
    if tol:
        ax.scatter(*zip(*tol), s=30, marker="s", color=MUTED, edgecolor=SURFACE, linewidth=1.5, label="tolerance question (loss 0)", zorder=3)
    ax.set_xlabel("item position in the intake session", fontsize=8)
    ax.set_ylabel("portfolio loss (%)", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    ax.set_title("recorded responses (stated choices, not predictions)", fontsize=9, loc="left")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def reliability_plot(rows: list[dict[str, Any]], ece: float | None, title: str) -> plt.Figure:
    """Reliability diagram: mean predicted probability vs observed sell rate per bin (marker
    area by bin count) against the diagonal. ``rows`` are ``bre.eval.reliability_table``
    records (``p_mean``, ``y_rate``, ``n`` or similar keys; missing keys are tolerated)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    _style(ax)
    xs, ys, ns = [], [], []
    for r in rows:
        x = r.get("p_mean", r.get("mean_p", r.get("p")))
        y = r.get("frac_pos", r.get("y_rate", r.get("mean_y", r.get("y"))))
        n = r.get("n", r.get("count", 0))
        if x is None or y is None or not (np.isfinite(float(x)) and np.isfinite(float(y))):
            continue
        xs.append(float(x)); ys.append(float(y)); ns.append(float(n or 0))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    if xs:
        size = 20 + 180 * (np.asarray(ns) / max(max(ns), 1.0))
        ax.scatter(xs, ys, s=size, color=SERIES_1, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="bin (area = count)")
        ax.plot(xs, ys, color=SERIES_1, linewidth=1.2, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"mean {P_LABEL} in bin", fontsize=8)
    ax.set_ylabel("observed sell rate in bin", fontsize=8)
    ax.set_title(title + ("" if ece is None else f" (ECE {ece:.3f})"), fontsize=9, loc="left")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


__all__ = ["CMAP", "P_LABEL", "SERIES_1", "SERIES_2", "capacity_curve", "heatmap", "png_bytes", "reliability_plot", "response_timeline", "sensitivity_bars"]
