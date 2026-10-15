"""Plotly traces colouring the foam mesh and the plate by strain energy density.

Plotly cannot vary the fill colour inside one trace, so values are binned into
``n_bins`` colour levels and each level is one trace of ``None``-separated
triangles (foam) or segments (plate). The trace count is fixed, so the same
helpers build animation frames. A hidden marker trace per field carries its
colorbar.

The foam and plate use different colour scales. The plate scale starts at a
saturated orange, never near white or blue, so the plate stays distinguishable
from the foam at any energy level.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

FOAM_ENERGY_COLORSCALE = [[0.0, "#f7fbff"], [0.35, "#9ecae1"], [0.7, "#3182bd"], [1.0, "#08306b"]]
PLATE_ENERGY_COLORSCALE = [[0.0, "#fdae61"], [0.4, "#f46d43"], [0.75, "#d73027"], [1.0, "#67000d"]]
FOAM_ENERGY_UNITS = "J/m³"
PLATE_ENERGY_UNITS = "J/m² per m width"
DEFAULT_BINS = 24
DISPLAY_PERCENTILE = 99.0


def energy_display_max(values, percentile: float = DISPLAY_PERCENTILE) -> float:
    """Colour-scale top: the ``percentile`` of the finite values (contact edges are spiky)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    top = float(np.percentile(v, percentile))
    return top if top > 0.0 else max(float(v.max()), 1.0e-30)


def _bin_index(values: np.ndarray, vmax: float, n_bins: int) -> np.ndarray:
    frac = np.clip(np.nan_to_num(values, nan=0.0) / float(vmax), 0.0, 1.0)
    return np.minimum((frac * n_bins).astype(int), n_bins - 1)


def _bin_colors(colorscale, n_bins: int) -> list[str]:
    return [str(c) for c in sample_colorscale(colorscale, list((np.arange(n_bins) + 0.5) / n_bins))]


def _colorbar_trace(colorscale, vmax: float, title: str, *, y: float, x_anchor: float, y_anchor: float):
    return go.Scatter(
        x=[x_anchor],
        y=[y_anchor],
        mode="markers",
        marker=dict(
            size=0.1,
            opacity=0.0,
            color=[0.0],
            cmin=0.0,
            cmax=float(vmax),
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=dict(text=title, side="right"), thickness=12, len=0.45, y=y, yanchor="middle"),
        ),
        showlegend=False,
        hoverinfo="skip",
    )


def foam_energy_traces(
    triangles: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    energy: np.ndarray,
    *,
    vmax: float | None = None,
    n_bins: int = DEFAULT_BINS,
    show_colorbar: bool = True,
) -> list[go.Scatter]:
    """``n_bins`` filled-triangle traces (+ a colorbar trace) for the deformed foam mesh."""
    t = np.asarray(triangles, dtype=int)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.asarray(energy, dtype=float)
    top = energy_display_max(W) if vmax is None else float(vmax)
    k = _bin_index(W, top, n_bins)
    colors = _bin_colors(FOAM_ENERGY_COLORSCALE, n_bins)
    # One gap-separated segment per triangle, [v0, v1, v2, nan]; fill="toself" closes
    # each segment. float32 halves the binary payload (animation frames).
    loop = np.vstack([t, np.full((1, t.shape[1]), -1)])
    traces: list[go.Scatter] = []
    for b in range(n_bins):
        sel = loop[:, k == b]
        ids = sel.T.reshape(-1)
        gap = ids < 0
        xs = np.where(gap, np.nan, x[np.maximum(ids, 0)]).astype(np.float32)
        ys = np.where(gap, np.nan, y[np.maximum(ids, 0)]).astype(np.float32)
        traces.append(
            go.Scatter(
                x=xs if xs.size else [None],
                y=ys if ys.size else [None],
                mode="lines",
                fill="toself",
                fillcolor=colors[b],
                # A hairline in the fill colour hides antialiasing seams between triangles.
                line=dict(width=0.5, color=colors[b]),
                name="foam strain energy",
                legendgroup="foam_energy",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if show_colorbar:
        traces.append(
            _colorbar_trace(
                FOAM_ENERGY_COLORSCALE, top, f"foam W ({FOAM_ENERGY_UNITS})",
                y=0.75, x_anchor=float(np.nanmean(x)), y_anchor=float(np.nanmean(y)),
            )
        )
    return traces


def plate_energy_traces(
    x: np.ndarray,
    y: np.ndarray,
    energy: np.ndarray,
    *,
    vmax: float | None = None,
    n_bins: int = DEFAULT_BINS,
    width: float = 5.0,
    show_colorbar: bool = True,
) -> list[go.Scatter]:
    """Dark outline + ``n_bins`` coloured segment traces (+ colorbar) for the plate centreline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.asarray(energy, dtype=float)
    top = energy_display_max(W) if vmax is None else float(vmax)
    # Segment s joins samples s and s+1, coloured by their mean.
    W_seg = 0.5 * (W[:-1] + W[1:])
    k = _bin_index(W_seg, top, n_bins)
    colors = _bin_colors(PLATE_ENERGY_COLORSCALE, n_bins)
    traces: list[go.Scatter] = [
        go.Scatter(
            x=x, y=y, mode="lines", line=dict(width=width + 2.0, color="#1a1a1a"),
            name="deformed plate", legendgroup="plate_energy", hoverinfo="skip",
        )
    ]
    for b in range(n_bins):
        s = np.flatnonzero(k == b)
        xs = np.column_stack([x[s], x[s + 1], np.full(s.size, np.nan)]).reshape(-1)
        ys = np.column_stack([y[s], y[s + 1], np.full(s.size, np.nan)]).reshape(-1)
        traces.append(
            go.Scatter(
                x=xs if xs.size else [None],
                y=ys if ys.size else [None],
                mode="lines",
                line=dict(width=width, color=colors[b]),
                legendgroup="plate_energy",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if show_colorbar:
        traces.append(
            _colorbar_trace(
                PLATE_ENERGY_COLORSCALE, top, f"plate W ({PLATE_ENERGY_UNITS})",
                y=0.25, x_anchor=float(np.nanmean(x)), y_anchor=float(np.nanmean(y)),
            )
        )
    return traces
