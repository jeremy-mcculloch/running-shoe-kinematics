"""Fixed-frame shoe shapes, animation and energy series for gait replay frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from compliance_fem.contact.contact_model import ContactModel, selection_proxy
from compliance_fem.gait.replay import GaitReplayResult
from compliance_fem.plotting.shape_render import ShapePlotData, build_shape_plot_data


def selection_proxy_for_frame(model: ContactModel, result: GaitReplayResult, frame: int) -> Any:
    """Renderer state of one solved replay frame (``contact_model.selection_proxy``)."""
    i = int(frame)
    if not result.frames or not bool(result.frame_valid[i]):
        raise ValueError(f"No solved contact state at frame {i}")
    return selection_proxy(model, result.frames[i], phi=float(result.phi[i]))


def shape_for_frame(
    model: ContactModel,
    result: GaitReplayResult,
    frame: int,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    n_plot: int = 120,
    show_plate: bool | None = None,
    samples_per_element: int = 16,
    plate_color_quantity: str = "none",
    foam_energy: bool = False,
) -> ShapePlotData:
    """Fixed-frame deformed geometry for one replay frame (plate drawn when the model has one)."""
    state = selection_proxy_for_frame(model, result, frame)
    if show_plate is None:
        show_plate = bool(model.has_plate_response)
    return build_shape_plot_data(
        model,
        state,
        scale_mode=scale_mode,
        manual_scale=manual_scale,
        n_plot=n_plot,
        show_plate=bool(show_plate),
        show_undeformed_plate=False,
        samples_per_element=int(samples_per_element),
        plate_color_quantity=plate_color_quantity,
        foam_energy=foam_energy,
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
    # Every active patch in one NaN-separated trace (fixed trace count for animation);
    # patch end nodes as edge markers. Gaps between patches are never filled.
    cx: list[float] = []
    cy: list[float] = []
    ex: list[float] = []
    ey: list[float] = []
    for (px_c, py_c) in shape.contact_curves:
        cx += [*np.asarray(px_c, dtype=float), np.nan]
        cy += [*np.asarray(py_c, dtype=float), np.nan]
        ex += [float(px_c[0]), float(px_c[-1])]
        ey += [float(py_c[0]), float(py_c[-1])]
    spans = ", ".join(f"{{{p.start}..{p.end}}}" for p in shape.patches) or "none"
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
        "contact_x": np.asarray(cx or [np.nan], dtype=float),
        "contact_y": np.asarray(cy or [np.nan], dtype=float),
        "contact_name": f"{shape.contact_label} contact {spans}",
        "edge_x": np.asarray(ex or [np.nan], dtype=float),
        "edge_y": np.asarray(ey or [np.nan], dtype=float),
        "anchor_x": np.array([shape.anchor_xy[0]], dtype=float),
        "anchor_y": np.array([shape.anchor_xy[1]], dtype=float),
        "chord_name": f"chord φ={shape.phi_deg:.2f}°",
        "title": (
            f"φ={shape.phi_deg:.2f}°  θ={shape.theta_deg:.2f}°  "
            f"({shape.contact_label}, {shape.scale_mode})"
        ),
        "L": float(shape.L),
        "H": float(shape.H),
    }


def _edge_trace(p: dict[str, Any]):
    import plotly.graph_objects as go

    return go.Scatter(
        x=p["edge_x"], y=p["edge_y"], mode="markers", name="contact patch end nodes",
        marker=dict(symbol="line-ns-open", size=16, color="#2ca02c", line=dict(width=3)),
        hoverinfo="skip",
    )


def _anchor_trace(p: dict[str, Any]):
    import plotly.graph_objects as go

    return go.Scatter(
        x=p["anchor_x"], y=p["anchor_y"], mode="markers", name="display pivot (contact centroid)",
        marker=dict(symbol="x-thin-open", size=10, color="#7f7f7f", line=dict(width=2)),
        hoverinfo="skip",
    )


MAX_ENERGY_FRAMES = 75


def animation_frame_stride(model: ContactModel, result: GaitReplayResult, *, energy: bool = True,
                           max_energy_frames: int = MAX_ENERGY_FRAMES) -> int:
    """Replay frames per animation frame: >1 only when foam energy colouring is on and frames are many."""
    n_valid = int(np.count_nonzero(np.asarray(result.frame_valid)))
    if not (energy and model.render_section) or n_valid <= max_energy_frames:
        return 1
    return int(np.ceil(n_valid / max(int(max_energy_frames), 1)))


def plotly_foot_animation_figure(
    model: ContactModel,
    result: GaitReplayResult,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    fps: float = 10.0,
    loop: bool = True,
    n_plot: int = 80,
    samples_per_element: int = 4,
    energy: bool = True,
    max_energy_frames: int = MAX_ENERGY_FRAMES,
):
    """Client-side animated shoe outline (no Streamlit redraw between frames).

    Frames only patch ``x``/``y`` on existing traces with ``redraw=False`` so the
    previous outline stays visible until the next frame is applied (no white flash).

    With ``energy`` (and a model with a render section / plate response), the
    foam mesh and the plate are coloured by strain energy density on colour
    scales fixed over the whole stance. Each foam frame is ~0.2 MB, so only every
    k-th replay frame is animated, at most ``max_energy_frames`` frames.
    """
    import plotly.graph_objects as go

    from compliance_fem.plotting.energy_plotly import (
        energy_display_max,
        foam_energy_traces,
        plate_energy_traces,
    )

    n = int(result.times.size)
    valid = [i for i in range(n) if bool(result.frame_valid[i])]
    foam_on = bool(energy and model.render_section)
    plate_on = bool(energy and model.has_plate_response)
    stride = animation_frame_stride(model, result, energy=energy, max_energy_frames=max_energy_frames)
    payloads: list[dict[str, Any]] = []
    shapes: list[ShapePlotData] = []
    frame_index: list[int] = []
    for i in valid[::stride]:
        try:
            shape = shape_for_frame(
                model,
                result,
                i,
                scale_mode=scale_mode,
                manual_scale=manual_scale,
                n_plot=n_plot,
                samples_per_element=samples_per_element,
                plate_color_quantity="strain_energy_density" if plate_on else "none",
                foam_energy=foam_on,
            )
            payloads.append(_foot_xy_payload(shape))
            shapes.append(shape)
            frame_index.append(i)
        except (ValueError, AssertionError):
            continue

    if not payloads:
        raise ValueError("No valid frames for foot animation")

    foam_on = foam_on and all(sh.foam_energy_density is not None for sh in shapes)
    plate_on = plate_on and all(
        sh.plate_color_quantity == "strain_energy_density" and sh.plate_color_values is not None for sh in shapes
    )
    foam_vmax = max(energy_display_max(sh.foam_energy_density) for sh in shapes) if foam_on else 1.0
    plate_vmax = max(energy_display_max(sh.plate_color_values) for sh in shapes) if plate_on else 1.0

    def _energy_traces(sh: ShapePlotData, *, colorbar: bool) -> tuple[list, list]:
        foam, plate = [], []
        if foam_on:
            assert sh.foam_triangles is not None and sh.foam_energy_density is not None
            assert sh.foam_x_def is not None and sh.foam_y_def is not None
            foam = foam_energy_traces(
                sh.foam_triangles, sh.foam_x_def, sh.foam_y_def, sh.foam_energy_density,
                vmax=foam_vmax, show_colorbar=colorbar,
            )
        if plate_on:
            px = sh.plate_x_def if sh.plate_x_def is not None else sh.x_plate_def
            py = sh.plate_y_def if sh.plate_y_def is not None else sh.y_plate_def
            assert px is not None and py is not None and sh.plate_color_values is not None
            plate = plate_energy_traces(px, py, sh.plate_color_values, vmax=plate_vmax, show_colorbar=colorbar)
        return foam, plate

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

    def _ordered(base: list, foam: list, plate: list) -> list:
        """Foam bins under the outlines, plate energy over the plain plate line."""
        return [base[0], base[1], *foam, *base[2:5], *plate, *base[5:]]

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

    base0 = _base_traces(payloads[0])
    if foam_on:
        base0[1].update(fillcolor="rgba(0,0,0,0)", showlegend=False)  # the energy mesh replaces the flat fill
    if plate_on:
        base0[4].update(showlegend=False)  # the energy plate carries the legend entry
    foam0, plate0 = _energy_traces(shapes[0], colorbar=True)
    # Colorbar traces (last of each energy list) are static; every other trace is patched.
    foam_bars = foam0[-1:] if foam_on else []
    plate_bars = plate0[-1:] if plate_on else []
    data0 = _ordered(base0, foam0[: len(foam0) - len(foam_bars)], plate0[: len(plate0) - len(plate_bars)])
    trace_ids = list(range(len(data0)))
    data0 = data0 + foam_bars + plate_bars

    def _frame_data(p: dict[str, Any], sh: ShapePlotData) -> list:
        foam, plate = _energy_traces(sh, colorbar=False)
        return _ordered(
            _xy_patch(p),
            [go.Scatter(x=t.x, y=t.y) for t in foam],
            [go.Scatter(x=t.x, y=t.y) for t in plate],
        )

    frames = [
        go.Frame(
            data=_frame_data(p, sh),
            traces=trace_ids,
            name=str(k),
        )
        for k, (p, sh) in enumerate(zip(payloads, shapes))
    ]
    fig = go.Figure(data=data0, frames=frames)
    fig.layout.meta = {"frame_stride": int(stride)}
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
                        label=str(frame_index[k]),
                    )
                    for k in range(len(payloads))
                ],
            )
        ],
    )
    return fig


def stance_strain_energy(model: ContactModel, result: GaitReplayResult, *, samples_per_element: int = 16) -> dict[str, np.ndarray]:
    """Total strain energy (J, whole shoe width) in the foam and in the plate per replay frame.

    Foam: ``sum_e W_e A_e`` over the mesh (``element_strain_energy_density``).
    Plate: ``int 1/2 EI kappa^2 ds`` over its Hermite samples (trapezoid). Both
    are per metre width in the model and are multiplied by ``shoe_width_m``.
    Frames without a solved state, or a model without the render section /
    plate response, give NaN.
    """
    from compliance_fem.contact.contact_model import plate_state_for_solution, render_vertex_displacements
    from compliance_fem.geometry.render import element_areas, element_strain_energy_density

    n = int(result.times.size)
    width = float(result.model_info.get("shoe_width_m", 1.0))
    foam = np.full(n, np.nan)
    plate = np.full(n, np.nan)
    areas = element_areas(model) if model.render_section else None
    for i in range(min(n, len(result.frames))):
        if not bool(result.frame_valid[i]):
            continue
        sol = result.frames[i].contact
        if areas is not None:
            u, v = render_vertex_displacements(model, sol)
            foam[i] = width * float(element_strain_energy_density(model, u, v) @ areas)
        state = plate_state_for_solution(
            model, sol, samples_per_element=samples_per_element, color_quantity="strain_energy_density"
        )
        if state is not None and state.color_values is not None:
            plate[i] = width * float(np.trapezoid(state.color_values, state.sample_s))
    return {"times": np.asarray(result.times, dtype=float), "foam_J": foam, "plate_J": plate}


def strain_energy_stance_figure(series: dict[str, np.ndarray]):
    """Foam (left axis) and plate (right axis) strain energy over the stance."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from compliance_fem.plotting.energy_plotly import FOAM_ENERGY_COLORSCALE, PLATE_ENERGY_COLORSCALE

    foam_color = FOAM_ENERGY_COLORSCALE[-1][1]
    plate_color = PLATE_ENERGY_COLORSCALE[-2][1]
    t = series["times"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=t, y=series["foam_J"], name="foam", line=dict(color=foam_color, width=2.5),
                   hovertemplate="t=%{x:.4f} s<br>foam %{y:.4g} J<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["plate_J"], name="plate", line=dict(color=plate_color, width=2.5),
                   hovertemplate="t=%{x:.4f} s<br>plate %{y:.4g} J<extra></extra>"),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="t (s)")
    fig.update_yaxes(title_text="foam strain energy (J)", color=foam_color, rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="plate strain energy (J)", color=plate_color, rangemode="tozero",
                     showgrid=False, secondary_y=True)
    fig.update_layout(
        title="Strain energy stored over stance (separate axes)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=70, b=50),
    )
    return fig


OUT_OF_PLANE_PANELS = (
    ("lateral_translation_stiffness", "k_z lateral translation (kN/m)", 1e-3),
    ("roll_stiffness", "k_roll roll (N·m/rad)", 1.0),
    ("yaw_stiffness", "k_yaw yaw (N·m/rad)", 1.0),
)


def out_of_plane_stiffness_figure(result: GaitReplayResult, *, x_axis: str = "time"):
    """Three stacked panels of the stored 2.5D stiffness histories (NaN shown as gaps).

    ``x_axis`` is ``"time"`` (s) or ``"stance"`` (stance-normalized percent). Values
    are the ones stored per frame by the replay; nothing is recomputed here.
    Samples with a non-``ok`` status are marked; frames near an active-set change
    are marked separately.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x = np.asarray(result.stance_percent if x_axis == "stance" else result.times, dtype=float)
    xlabel = "stance (%)" if x_axis == "stance" else "t (s)"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for _, t, _ in OUT_OF_PLANE_PANELS])
    status = np.asarray(result.oop_stiffness_status if result.oop_stiffness_status else ["not_available"] * x.size)
    flagged = ~np.isin(status, ["ok", "no_contact"])
    near = np.asarray(result.oop_near_active_set_change, dtype=bool) if result.oop_near_active_set_change.size == x.size \
        else np.zeros(x.size, dtype=bool)
    for row, (name, _, scale) in enumerate(OUT_OF_PLANE_PANELS, start=1):
        y = np.asarray(getattr(result, name), dtype=float)
        if y.size != x.size:
            y = np.full(x.size, np.nan)
        y = y * scale
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2), connectgaps=False,
                                 showlegend=False), row=row, col=1)
        if np.any(flagged & np.isfinite(y)) or np.any(flagged):
            yy = np.where(np.isfinite(y), y, 0.0)
            fig.add_trace(go.Scatter(x=x[flagged], y=yy[flagged], mode="markers", name="invalid / ill-conditioned",
                                     marker=dict(symbol="x", color="#d62728", size=7), showlegend=(row == 1),
                                     text=status[flagged], hovertemplate="%{text}<extra></extra>"), row=row, col=1)
        if np.any(near):
            fig.add_trace(go.Scatter(x=x[near], y=y[near], mode="markers", name="near active-set change",
                                     marker=dict(symbol="circle-open", color="#ff9f1c", size=5), showlegend=(row == 1)),
                          row=row, col=1)
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_layout(height=620, title="2.5D out-of-plane tangent stiffness (local, active set frozen)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
                      margin=dict(l=60, r=20, t=80, b=50))
    return fig


def out_of_plane_csv(result: GaitReplayResult) -> str:
    """CSV of the 2.5D stiffness histories and diagnostics (units in the header)."""
    oop = result.out_of_plane_column_dict()
    oop.pop("gait_cycle_percent", None)
    cols = {"time_s": result.times, "stance_percent": result.stance_percent, **oop}
    keys = list(cols)
    lines = [",".join(keys)]
    for i in range(int(result.times.size)):
        lines.append(",".join(str(cols[k][i]) if np.asarray(cols[k]).dtype == object else f"{float(cols[k][i]):.10g}" for k in keys))
    return "\n".join(lines) + "\n"
