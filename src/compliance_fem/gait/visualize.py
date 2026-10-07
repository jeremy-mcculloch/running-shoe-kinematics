"""Reconstruct fixed-frame shoe outlines from gait replay frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.contact_basis import COL_BRX, COL_BRY, COL_CONST
from compliance_fem.contact_lookup import ContactLookupResult
from compliance_fem.contact_topology import ContactType
from compliance_fem.corotation import basis_coefficients, contract_basis
from compliance_fem.gait.replay import GaitReplayResult
from compliance_fem.shape_render import ShapePlotData, build_shape_plot_data


def _contact_type_at(result: GaitReplayResult, i: int) -> ContactType:
    raw = result.contact_type[i]
    if isinstance(raw, ContactType):
        return raw
    return ContactType(str(raw).lower())


def selection_proxy_for_frame(
    lookup: ContactLookupResult,
    result: GaitReplayResult,
    frame: int,
) -> Any:
    """Build a minimal selection object for ``build_shape_plot_data``."""
    i = int(frame)
    row = int(result.record_index[i])
    if row < 0:
        raise ValueError(f"No selected contact record at frame {i}")

    alpha = float(result.alpha[i])
    d_ax = float(result.d_ax[i])
    d_ay = float(result.d_ay[i])
    varphi = float(result.varphi[i])
    gamma = basis_coefficients(alpha, d_ax, d_ay, varphi)

    # Stored bottom bases already hold the exact contact kinematics (rigid modes
    # about the anchor plus the curved-sole closure) on the contact nodes.
    fields = lookup.record_fields([row])
    u_bot = np.asarray(contract_basis(fields["bottom_u"][0], gamma, mode_axis=0), dtype=float)
    v_bot = np.asarray(contract_basis(fields["bottom_v"][0], gamma, mode_axis=0), dtype=float)
    anchor_x = float(lookup.contact_anchor_reference_x[row])
    anchor_y = float(lookup.contact_anchor_reference_y[row])

    return SimpleNamespace(
        selected_row=row,
        contact_type=_contact_type_at(result, i),
        contact_start_index=int(lookup.contact_start_index[row]),
        contact_end_index=int(lookup.contact_end_index[row]),
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        d_ax=d_ax,
        d_ay=d_ay,
        varphi=varphi,
        alpha=alpha,
        phi=float(result.phi[i]),
        theta=float(np.deg2rad(result.theta_deg[i])),
        Fx_star=float(result.elastic_Fx[i]),
        Fy_star=float(result.elastic_Fy[i]),
        full_bottom_u=u_bot,
        full_bottom_v=v_bot,
        x_anchor_rot=float(anchor_x + d_ax),
    )


def shape_for_frame(
    lookup: ContactLookupResult,
    result: GaitReplayResult,
    frame: int,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    n_plot: int = 120,
    show_plate: bool | None = None,
    samples_per_element: int = 16,
) -> ShapePlotData:
    """Fixed-frame deformed geometry for one replay frame.

    When ``show_plate`` is ``None``, the plate is drawn if the lookup stores
    plate Hermite responses (same default as the main GUI).
    """
    selection = selection_proxy_for_frame(lookup, result, frame)
    if show_plate is None:
        show_plate = bool(getattr(lookup, "has_plate_response", False))
    return build_shape_plot_data(
        lookup,
        selection,  # type: ignore[arg-type]
        scale_mode=scale_mode,
        manual_scale=manual_scale,
        n_plot=n_plot,
        show_plate=bool(show_plate),
        show_undeformed_plate=False,
        samples_per_element=int(samples_per_element),
    )


def _foot_xy_payload(shape: ShapePlotData) -> dict[str, Any]:
    """Fixed-order xy arrays for one frame (stable for Plotly animation)."""
    x_bot = np.asarray(shape.x_bottom_def, dtype=float)
    y_bot = np.asarray(shape.y_bottom_def, dtype=float)
    x_top = np.asarray(shape.x_top_def, dtype=float)
    y_top = np.asarray(shape.y_top_def, dtype=float)
    poly_x = np.concatenate([x_bot, x_top[::-1], x_bot[:1]])
    poly_y = np.concatenate([y_bot, y_top[::-1], y_bot[:1]])

    px = py = None
    if getattr(shape, "has_plate", False):
        px = shape.plate_x_def if shape.plate_x_def is not None else shape.x_plate_def
        py = shape.plate_y_def if shape.plate_y_def is not None else shape.y_plate_def
    elif shape.x_plate_def is not None and shape.y_plate_def is not None:
        px, py = shape.x_plate_def, shape.y_plate_def
    if px is None or py is None:
        px = np.array([np.nan, np.nan], dtype=float)
        py = np.array([np.nan, np.nan], dtype=float)
    else:
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)

    x_pad = 0.08 * float(shape.L)
    edges = [p for p in (shape.heel_edge_xy, shape.toe_edge_xy) if p is not None]
    edge_x = np.array([p[0] for p in edges] or [np.nan], dtype=float)
    edge_y = np.array([p[1] for p in edges] or [np.nan], dtype=float)
    return {
        "reference_x": np.array([0.0, shape.L, shape.L, 0.0, 0.0], dtype=float),
        "reference_y": np.array([0.0, 0.0, shape.H, shape.H, 0.0], dtype=float),
        "foam_x": poly_x,
        "foam_y": poly_y,
        "top_x": x_top,
        "top_y": y_top,
        "bottom_x": x_bot,
        "bottom_y": y_bot,
        "plate_x": px,
        "plate_y": py,
        "chord_x": np.asarray(shape.chord_x, dtype=float),
        "chord_y": np.asarray(shape.chord_y, dtype=float),
        "ground_x": np.array([-x_pad, float(shape.L) + x_pad], dtype=float),
        "ground_y": np.array([shape.ground_y, shape.ground_y], dtype=float),
        "contact_x": np.asarray(shape.contact_curve_x, dtype=float),
        "contact_y": np.asarray(shape.contact_curve_y, dtype=float),
        "contact_name": (
            f"{shape.contact_type} contact {{{shape.contact_start_index}..{shape.contact_end_index}}}"
        ),
        "edge_x": edge_x,
        "edge_y": edge_y,
        "anchor_x": np.array([shape.anchor_xy[0]], dtype=float),
        "anchor_y": np.array([shape.anchor_xy[1]], dtype=float),
        "chord_name": f"chord φ={shape.phi_deg:.2f}°",
        "title": (
            f"φ={shape.phi_deg:.2f}°  θ={shape.theta_deg:.2f}°  "
            f"({shape.contact_type}, {shape.scale_mode})"
        ),
        "L": float(shape.L),
        "H": float(shape.H),
    }


def _edge_trace(p: dict[str, Any]):
    import plotly.graph_objects as go

    return go.Scatter(
        x=p["edge_x"], y=p["edge_y"], mode="markers", name="contact edges (heel, toe)",
        marker=dict(symbol="line-ns-open", size=16, color="#2ca02c", line=dict(width=3)),
        hoverinfo="skip",
    )


def _anchor_trace(p: dict[str, Any]):
    import plotly.graph_objects as go

    return go.Scatter(
        x=p["anchor_x"], y=p["anchor_y"], mode="markers", name="neutral anchor (numerical)",
        marker=dict(symbol="x-thin-open", size=10, color="#7f7f7f", line=dict(width=2)),
        hoverinfo="skip",
    )


def plotly_foot_figure(shape: ShapePlotData, *, title: str | None = None):
    """Plotly figure of the fixed-frame deformed shoe for one frame."""
    import plotly.graph_objects as go

    p = _foot_xy_payload(shape)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=p["reference_x"],
            y=p["reference_y"],
            mode="lines",
            name="reference",
            line=dict(color="#888888", width=1, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["foam_x"],
            y=p["foam_y"],
            mode="lines",
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.25)",
            line=dict(width=0),
            name="deformed foam",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["top_x"],
            y=p["top_y"],
            mode="lines",
            name="top",
            line=dict(color="#1f77b4", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["bottom_x"],
            y=p["bottom_y"],
            mode="lines",
            name="bottom",
            line=dict(color="#ff7f0e", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["plate_x"],
            y=p["plate_y"],
            mode="lines",
            name="deformed plate",
            line=dict(color="#1a1a1a", width=3),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["chord_x"],
            y=p["chord_y"],
            mode="lines+markers",
            name=p["chord_name"],
            line=dict(color="#9467bd", width=2, dash="dashdot"),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["ground_x"],
            y=p["ground_y"],
            mode="lines",
            name="ground",
            line=dict(color="#4d4d4d", width=1),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["contact_x"],
            y=p["contact_y"],
            mode="lines",
            name=p["contact_name"],
            line=dict(color="#2ca02c", width=8),
            hoverinfo="skip",
        )
    )
    fig.add_trace(_edge_trace(p))
    fig.add_trace(_anchor_trace(p))

    fx, fy = shape.force_vector
    fmag = float(np.hypot(fx, fy))
    if fmag > 0.0:
        arrow = 0.35 * max(shape.L, shape.H)
        x0, y0 = shape.force_origin
        fig.add_annotation(
            x=x0 + arrow * fx / fmag,
            y=y0 + arrow * fy / fmag,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#d62728",
            text="F_e",
        )

    fig.update_layout(
        title=title or p["title"],
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plotly_foot_animation_figure(
    lookup: ContactLookupResult,
    result: GaitReplayResult,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    fps: float = 10.0,
    loop: bool = True,
    n_plot: int = 80,
    samples_per_element: int = 4,
):
    """Client-side animated shoe outline (no Streamlit redraw between frames).

    Frames only patch ``x``/``y`` on existing traces with ``redraw=False`` so the
    previous outline stays visible until the next frame is applied (no white flash).
    """
    import plotly.graph_objects as go

    n = int(result.times.size)
    payloads: list[dict[str, Any]] = []
    for i in range(n):
        if int(result.record_index[i]) < 0:
            continue
        try:
            shape = shape_for_frame(
                lookup,
                result,
                i,
                scale_mode=scale_mode,
                manual_scale=manual_scale,
                n_plot=n_plot,
                samples_per_element=samples_per_element,
            )
            payloads.append(_foot_xy_payload(shape))
        except (ValueError, AssertionError):
            continue

    if not payloads:
        raise ValueError("No valid frames for foot animation")

    n_traces = 10
    trace_ids = list(range(n_traces))

    def _base_traces(p: dict[str, Any]) -> list:
        return [
            go.Scatter(
                x=p["reference_x"],
                y=p["reference_y"],
                mode="lines",
                name="reference",
                line=dict(color="#888888", width=1, dash="dash"),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["foam_x"],
                y=p["foam_y"],
                mode="lines",
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.25)",
                line=dict(width=0),
                name="deformed foam",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["top_x"],
                y=p["top_y"],
                mode="lines",
                name="top",
                line=dict(color="#1f77b4", width=2),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["bottom_x"],
                y=p["bottom_y"],
                mode="lines",
                name="bottom",
                line=dict(color="#ff7f0e", width=2),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["plate_x"],
                y=p["plate_y"],
                mode="lines",
                name="deformed plate",
                line=dict(color="#1a1a1a", width=3),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["chord_x"],
                y=p["chord_y"],
                mode="lines+markers",
                name=p["chord_name"],
                line=dict(color="#9467bd", width=2, dash="dashdot"),
                marker=dict(size=7),
            ),
            go.Scatter(
                x=p["ground_x"],
                y=p["ground_y"],
                mode="lines",
                name="ground",
                line=dict(color="#4d4d4d", width=1),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=p["contact_x"],
                y=p["contact_y"],
                mode="lines",
                name=p["contact_name"],
                line=dict(color="#2ca02c", width=8),
                hoverinfo="skip",
            ),
            _edge_trace(p),
            _anchor_trace(p),
        ]

    def _xy_patch(p: dict[str, Any]) -> list:
        # Partial updates only — keeps prior pixels until the patch applies.
        return [
            go.Scatter(x=p["reference_x"], y=p["reference_y"]),
            go.Scatter(x=p["foam_x"], y=p["foam_y"]),
            go.Scatter(x=p["top_x"], y=p["top_y"]),
            go.Scatter(x=p["bottom_x"], y=p["bottom_y"]),
            go.Scatter(x=p["plate_x"], y=p["plate_y"]),
            go.Scatter(x=p["chord_x"], y=p["chord_y"], name=p["chord_name"]),
            go.Scatter(x=p["ground_x"], y=p["ground_y"]),
            go.Scatter(x=p["contact_x"], y=p["contact_y"], name=p["contact_name"]),
            go.Scatter(x=p["edge_x"], y=p["edge_y"]),
            go.Scatter(x=p["anchor_x"], y=p["anchor_y"]),
        ]

    xs: list[float] = []
    ys: list[float] = []
    for p in payloads:
        for key in (
            "foam_x",
            "top_x",
            "bottom_x",
            "plate_x",
            "chord_x",
            "ground_x",
            "contact_x",
            "reference_x",
        ):
            arr = np.asarray(p[key], dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                xs.extend(finite.tolist())
        for key in (
            "foam_y",
            "top_y",
            "bottom_y",
            "plate_y",
            "chord_y",
            "ground_y",
            "contact_y",
            "reference_y",
        ):
            arr = np.asarray(p[key], dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                ys.extend(finite.tolist())
    L = float(payloads[0]["L"])
    H = float(payloads[0]["H"])
    if xs and ys:
        x0, x1 = float(min(xs)), float(max(xs))
        y0, y1 = float(min(ys)), float(max(ys))
        pad_x = max(0.05 * L, 0.05 * (x1 - x0 + 1e-12))
        pad_y = max(0.05 * H, 0.05 * (y1 - y0 + 1e-12))
        xrange = [x0 - pad_x, x1 + pad_x]
        yrange = [y0 - pad_y, y1 + pad_y]
    else:
        xrange = [-0.15 * L, 1.15 * L]
        yrange = [-0.35 * H, 1.6 * H]

    duration_ms = float(1000.0 / max(float(fps), 0.5))
    frame_names = [str(k) for k in range(len(payloads))]
    # Plotly.js has no reliable infinite loop flag; replay the sequence many times.
    play_sequence: list[str] | None
    if loop:
        play_sequence = frame_names * 40
    else:
        play_sequence = None  # None → play all frames once from current

    frames = [
        go.Frame(
            data=_xy_patch(p),
            traces=trace_ids,
            name=str(k),
        )
        for k, p in enumerate(payloads)
    ]
    fig = go.Figure(data=_base_traces(payloads[0]), frames=frames)
    fig.update_layout(
        title=payloads[0]["title"],
        uirevision="gait-foot-animation",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.12),
        margin=dict(l=40, r=20, t=80, b=80),
        xaxis=dict(range=xrange, autorange=False, fixedrange=True),
        yaxis=dict(range=yrange, autorange=False, fixedrange=True),
        transition=dict(duration=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            play_sequence,
                            {
                                "frame": {"duration": duration_ms, "redraw": False},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.0,
                len=1.0,
                y=0.0,
                yanchor="top",
                pad=dict(t=40, b=10),
                currentvalue=dict(prefix="frame ", visible=True, xanchor="center"),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(k)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(k),
                    )
                    for k in range(len(payloads))
                ],
            )
        ],
    )
    return fig


@dataclass(frozen=True)
class MomentThetaDiagnostic:
    """Per-frame Mz(θ) curve at the measured φ for wrench-fit debugging."""

    theta_deg: NDArray[np.float64]
    Mz_force_match: NDArray[np.float64]
    Mz_star: float
    phi_target_deg: float
    theta_used_deg: float
    theta_free_deg: float
    status: str
    row: int
    contact_type: str
    theta_min_deg: float
    theta_max_deg: float


def moment_theta_diagnostic(
    lookup: ContactLookupResult,
    result: GaitReplayResult,
    frame: int,
    *,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
    n_theta: int = 121,
) -> MomentThetaDiagnostic:
    """Sweep θ at the frame's measured φ; report force-matched Mz(θ).

    At each θ, prescribe α=tan(θ), match Fx/Fy on the selected record, and
    evaluate Mz. ``theta_prior_fallback`` means the free 3×3 θ at φ* lies
    outside ``[theta_min, theta_max]``, so the replay fixed θ to the force-phi
    prior instead.
    """
    from compliance_fem.corotation import corotation_angle, rotate_force_to_local
    from compliance_fem.gait.wrench_control import (
        _fix_alpha_solve_translations,
        _wrench_modes,
        assemble_Kw,
    )

    i = int(frame)
    row = int(result.record_index[i])
    if row < 0:
        raise ValueError(f"No selected contact record at frame {i}")

    phi_target = float(result.phi[i])
    phi_target_deg = float(np.rad2deg(phi_target))
    theta_used = float(result.theta_deg[i])
    Fx_star = float(result.elastic_Fx[i])
    Fy_star = float(result.elastic_Fy[i])
    Mz_star = float(result.elastic_Mz[i])
    status = str(result.selection_status[i])
    ctype = str(result.contact_type[i])

    scalars = np.asarray(lookup.scalar_lookup[row], dtype=float)
    W = _wrench_modes(scalars)
    Kw = assemble_Kw(scalars)
    phi_ref = float(getattr(lookup, "phi_ref", 0.0))

    varphi = corotation_angle(phi_target, phi_ref)
    r_x = float(np.cos(varphi) - 1.0)
    r_y = float(-np.sin(varphi))
    F_local = rotate_force_to_local(Fx_star, Fy_star, varphi)
    w_star = np.array([F_local[0], F_local[1], Mz_star], dtype=np.float64)
    rhs = w_star - W[:, COL_CONST] - r_x * W[:, COL_BRX] - r_y * W[:, COL_BRY]

    # Unconstrained θ from free 3×3 at φ* (explains prior fallback).
    theta_free = float("nan")
    try:
        svals = np.linalg.svd(Kw, compute_uv=False)
        cond = float(svals[0] / max(float(svals[-1]), 1e-30))
        if np.isfinite(cond) and cond <= 1.0e8 and float(svals[-1]) > 0.0:
            z_free = np.linalg.solve(Kw, rhs)
            if np.all(np.isfinite(z_free)):
                theta_free = float(np.rad2deg(np.arctan(float(z_free[2]))))
    except np.linalg.LinAlgError:
        try:
            z_free, *_ = np.linalg.lstsq(Kw, rhs, rcond=None)
            if np.all(np.isfinite(z_free)):
                theta_free = float(np.rad2deg(np.arctan(float(z_free[2]))))
        except np.linalg.LinAlgError:
            pass

    theta_deg = np.linspace(float(theta_min_deg), float(theta_max_deg), int(n_theta))
    if float(theta_min_deg) <= theta_used <= float(theta_max_deg):
        if not np.any(np.isclose(theta_deg, theta_used, atol=1e-9)):
            theta_deg = np.sort(np.unique(np.concatenate([theta_deg, [theta_used]])))

    Mz_force = np.full(theta_deg.shape, np.nan)
    for k, th in enumerate(theta_deg):
        z_f, _ = _fix_alpha_solve_translations(Kw, rhs, float(th))
        if z_f is not None and np.all(np.isfinite(z_f)):
            gamma_f = np.array(
                [z_f[2], z_f[0], z_f[1], r_x, r_y], dtype=np.float64
            )
            Mz_force[k] = float(contract_basis(W, gamma_f, mode_axis=1)[2])

    return MomentThetaDiagnostic(
        theta_deg=np.asarray(theta_deg, dtype=np.float64),
        Mz_force_match=Mz_force,
        Mz_star=Mz_star,
        phi_target_deg=phi_target_deg,
        theta_used_deg=theta_used,
        theta_free_deg=theta_free,
        status=status,
        row=row,
        contact_type=ctype,
        theta_min_deg=float(theta_min_deg),
        theta_max_deg=float(theta_max_deg),
    )


def plotly_moment_vs_theta_figure(diag: MomentThetaDiagnostic):
    """Plot Mz(θ) at φ* with target θ / Mz markers."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=diag.theta_deg,
            y=diag.Mz_force_match,
            mode="lines",
            name=f"Mz (φ*={diag.phi_target_deg:.2f}°, force match)",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_hline(
        y=diag.Mz_star,
        line=dict(color="#2ca02c", width=1.5, dash="dot"),
        annotation_text=f"Mz*={diag.Mz_star:.4g}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=diag.theta_used_deg,
        line=dict(color="#d62728", width=2),
        annotation_text=f"θ={diag.theta_used_deg:.2f}°",
        annotation_position="top",
    )
    if np.isfinite(diag.theta_free_deg):
        in_range = diag.theta_min_deg <= diag.theta_free_deg <= diag.theta_max_deg
        if in_range:
            mz_at_free = float(
                np.interp(diag.theta_free_deg, diag.theta_deg, diag.Mz_force_match)
            )
            fig.add_trace(
                go.Scatter(
                    x=[diag.theta_free_deg],
                    y=[mz_at_free],
                    mode="markers",
                    name=f"θ_free={diag.theta_free_deg:.2f}°",
                    marker=dict(size=10, color="#ff7f0e", symbol="x"),
                )
            )
        else:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.02,
                showarrow=False,
                align="right",
                text=f"θ_free={diag.theta_free_deg:.2f}° (outside plot range)",
                font=dict(color="#ff7f0e", size=12),
            )

    fig.update_layout(
        title=(
            f"Mz vs θ · row {diag.row} ({diag.contact_type}) · "
            f"status={diag.status}"
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=40, t=80, b=40),
        xaxis_title="θ (deg)",
        yaxis_title="Mz (model units)",
        xaxis=dict(range=[diag.theta_min_deg, diag.theta_max_deg]),
    )
    return fig


# Backwards-compatible names (θ-sweep API; old φ kwargs ignored).
def moment_phi_diagnostic(lookup, result, frame, **kwargs):
    kwargs.pop("phi_min_deg", None)
    kwargs.pop("phi_max_deg", None)
    kwargs.pop("n_phi", None)
    return moment_theta_diagnostic(lookup, result, frame, **kwargs)


plotly_moment_vs_phi_figure = plotly_moment_vs_theta_figure
MomentPhiDiagnostic = MomentThetaDiagnostic
