"""Streamlit GUI: force-controlled continuous unilateral contact of the measured sole."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import plotly.io as pio

    # Prefer orjson when available; Plotly falls back to the stdlib encoder.
    pio.json.config.default_engine = "orjson"
except Exception:
    pass

from compliance_fem.contact.basis import kappa_from_min_bend_radius, min_bend_radius_mm_from_kappa
from compliance_fem.contact.contact_model import (
    LOOKUP_SCHEMA_VERSION,
    ContactModel,
    load_contact_model,
    save_contact_model,
    selection_proxy,
)
from compliance_fem.contact.continuous_contact import ContactSolution, ContactTolerances, solve_contact_state
from compliance_fem.contact.continuous_toe import ToeFrameSolution, WarmStart, solve_toe_contact
from compliance_fem.contact.out_of_plane import OutOfPlaneSettings, compute_out_of_plane_stiffness
from compliance_fem.contact.sole_config import DEFAULT_LOWER_FOAM, DEFAULT_UPPER_FOAM, FOAM_FFLEAP, FOAM_MATERIALS
from compliance_fem.contact.toe_spring import ToeSpringConfig
from compliance_fem.gui.params import (
    ANGLES,
    DEFAULTS,
    E_MPA_MAX,
    E_MPA_MIN,
    EI_NM2_MAX,
    EI_NM2_MIN,
    FX_N_MAX,
    FX_N_MIN,
    FY_N_MAX,
    FY_N_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    TOE_LENGTH_MM_MAX,
    TOE_LENGTH_MM_MIN,
    L_MM_MAX,
    L_MM_MIN,
    NU_MAX,
    NU_MIN,
    dual_linear,
    dual_log,
    kn_to_n,
    m_to_mm,
    mm_to_m,
    mpa_to_pa,
    n_to_kn,
    pa_to_mpa,
)
from compliance_fem.plotting.energy_plotly import foam_energy_traces, plate_energy_traces
from compliance_fem.plotting.shape_render import NODE_ACTIVE, NODE_INACTIVE, NODE_TRANSITION, build_shape_plot_data

NOT_AVAILABLE = "N/A"
TOE_PASSIVE = "Passive toe spring (θ solved)"
TOE_PRESCRIBED = "Prescribed θ"


EXPORT_DIR = Path("outputs/gui_contact_lookup")
_SRC_ROOT = Path(__file__).resolve().parents[2]
_APPLIED_MODEL_KEY = "_applied_model_params"


def _canonical_model_params(
    *,
    ffturbo_E_Pa: float,
    ffturbo_nu: float,
    ffleap_E_heel_Pa: float,
    ffleap_E_toe_Pa: float,
    ffleap_nu: float,
    EI_plate_Nm2_per_m: float,
    softplus_toe_length: float,
    softplus_kappa: float,
    reciprocity_tol: float,
    measured: dict,
) -> dict[str, float | int | str]:
    """Round floats so tiny slider noise does not bust the rebuild cache.

    ``measured`` holds the CSV path (hashed by content), shoe length, foam
    assignment and mesh refinement settings.
    """
    csv_path = str(measured["geometry_csv"])
    out = {
        "ffturbo_E_Pa": float(f"{float(ffturbo_E_Pa):.8e}"),
        "ffturbo_nu": round(float(ffturbo_nu), 10),
        "ffleap_E_heel_Pa": float(f"{float(ffleap_E_heel_Pa):.8e}"),
        "ffleap_E_toe_Pa": float(f"{float(ffleap_E_toe_Pa):.8e}"),
        "ffleap_nu": round(float(ffleap_nu), 10),
        "EI_plate_Nm2_per_m": float(f"{float(EI_plate_Nm2_per_m):.8e}"),
        "softplus_toe_length": round(float(softplus_toe_length), 12),
        "softplus_kappa": float(f"{float(softplus_kappa):.8e}"),
        "reciprocity_tol": float(f"{float(reciprocity_tol):.8e}"),
        "geometry_csv": csv_path,
        "geometry_csv_sha256": _file_sha256(csv_path),
        "shoe_length_mm": round(float(measured["shoe_length_mm"]), 9),
        "upper_foam_material": str(measured["upper_foam_material"]),
        "lower_foam_material": str(measured["lower_foam_material"]),
        "mesh_size_mm": round(float(measured["mesh_size_mm"]), 9),
        "toe_refinement": round(float(measured["toe_refinement"]), 9),
        "heel_corner_refinement": round(float(measured["heel_corner_refinement"]), 9),
        "interface_refinement": round(float(measured["interface_refinement"]), 9),
        "plate_end_refinement": round(float(measured["plate_end_refinement"]), 9),
    }
    for key in ("landmark_tolerance", "curvature_max_turn_deg", "min_angle_deg"):
        if key in measured:
            out[key] = float(f"{float(measured[key]):.10e}")
    return out


def _draft_from_setup(setup) -> dict:
    """Applied-model dict taking every measured-sole parameter from a JSON config.

    When ``lookup.output_dir`` holds a lookup built from exactly this setup it is
    loaded directly instead of rebuilding.
    """
    from compliance_fem.gui.params import gui_params_from_setup
    from compliance_fem.contact.model_setup import matching_prebuilt_lookup

    p = gui_params_from_setup(setup)
    out = _canonical_model_params(
        ffturbo_E_Pa=p["ffturbo_E_Pa"],
        ffturbo_nu=p["ffturbo_nu"],
        ffleap_E_heel_Pa=p["ffleap_E_heel_Pa"],
        ffleap_E_toe_Pa=p["ffleap_E_toe_Pa"],
        ffleap_nu=p["ffleap_nu"],
        EI_plate_Nm2_per_m=p["EI_plate_Nm2_per_m"],
        softplus_toe_length=p["softplus_toe_length"],
        softplus_kappa=p["softplus_kappa"],
        reciprocity_tol=p["reciprocity_tol"],
        measured=p,
    )
    out["model_setup"] = str(setup.source_path)
    prebuilt = matching_prebuilt_lookup(setup)
    if prebuilt is not None:
        out["prebuilt_lookup"] = str(prebuilt)
    return out


def _config_export(draft_model: dict, setup) -> None:
    """Download the current measured-sole parameters as a JSON config."""
    from compliance_fem.gui.params import setup_from_gui

    try:
        export = setup if setup is not None else setup_from_gui(draft_model)
        payload = json.dumps(export.to_dict(), indent=2) + "\n"
    except (OSError, ValueError) as exc:
        st.caption(f"Config export unavailable: {exc}")
        return
    st.download_button(
        "Export measured-sole config (JSON)",
        data=payload,
        file_name="measured_sole_config.json",
        mime="application/json",
        key="gui_export_config",
        help="Writes every measured-sole parameter shown in the sidebar (absolute CSV path).",
    )


@st.cache_resource(show_spinner=False)
def _load_setup(path: str, mtime: float):
    from compliance_fem.contact.model_setup import load_model_setup

    del mtime
    return load_model_setup(path)


DEFAULT_GEOMETRY_CSV = "data/geometry/sole_geometry.csv"
DEFAULT_MEASURED_CONFIG = "configs/setup.json"


def _file_sha256(path: str) -> str:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"

PLATE_COLOR_OPTIONS = {
    "None": "none",
    "Normal displacement": "normal_displacement",
    "Tangential displacement": "tangential_displacement",
    "Rotation": "rotation",
    "Curvature": "curvature",
    "Bending moment": "bending_moment",
    "Shear force": "shear_force",
    "Axial constraint force": "axial_constraint_force",
    "Strain energy density": "strain_energy_density",
}

PLATE_SUMMARY_LABELS = {
    "max_abs_normal_displacement": "Max |w|",
    "max_abs_tangential_displacement_variation": "Max |Δu_s|",
    "max_abs_local_rotation": "Max |θ|",
    "max_abs_curvature": "Max |κ|",
    "max_abs_bending_moment": "Max |M|",
    "max_abs_shear_force": "Max |V|",
    "max_abs_axial_constraint_force": "Max |N_ax|",
    "plate_constraint_residual": "‖B_p u‖",
    "max_moment_reference_s": "s at max |M|",
}

HELP_TEXT = """
**phi** is the fixed-frame pitch, counterclockwise positive. With the passive toe
spring it is the rearfoot (heel → MTP) angle and the chord rotation follows from
the solved toe angle, `varphi(alpha) = phi − atan2(dy_mtp + alpha phi1_mtp, L − toe length)`.
With a prescribed θ it is the heel-to-toe chord angle, `varphi = phi − phi_ref`.

**theta** drives the single endpoint-fixed softplus shape mode through
`alpha = tan(theta)` (never `alpha = theta`). Positive theta tilts the toe segment
up relative to the chord; heel and toe heights are set by the frame.

Linearized strain is intentionally retained while boundary positions use the
exact finite rotation. This is fast, but it is not objective for locally large
rotations.
"""

CONTACT_HELP = """
Every bottom node is a potential contact node. The ground is flat (`y = 0`) and
the contact set is the solution of one complementarity problem

`0 ≤ λ ⊥ g ≥ 0`,  `g = q + T_y 1 + H λ`,  `Σλ = −Fy*`,

where `g` is the fixed-frame normal gap (positive open), `λ` the compressive
normal nodal reaction and `T_y` the vertical frame translation. It is solved by
a warm-started primal-dual active-set method from shared top-clamped boundary
operators (no interval table, no FEM solve per evaluation).

Tangential contact is **proportional shear**: each active node carries
`τ = (Fx*/Fy*) λ`, so both prescribed force components are met exactly and
entering / leaving nodes pass continuously through `g = λ = 0`.

Disconnected contact patches are shown separately (never filled in between).
The contact-edge marks are a derived display estimate inside the first free
edge, never an extra constraint.
"""


@st.cache_resource(show_spinner=False)
def _build_model(params_json: str):
    """Rebuild FEM + contact model in a subprocess, then load the NPZ into this process.

    ``params_json`` is the canonical model-parameter dict.
    Isolates Gmsh / SuperLU from Streamlit to avoid macOS native segfaults.
    """
    import hashlib

    params = json.loads(params_json)
    prebuilt = params.get("prebuilt_lookup")
    if prebuilt and Path(prebuilt).is_file():
        return load_contact_model(prebuilt)
    key_payload = {"schema_version": LOOKUP_SCHEMA_VERSION, **params}
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = Path(tempfile.gettempdir()) / "compliance_fem_gui_lookup_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "contact_lookup.npz"
    params_path = cache_dir / "params.json"

    if not npz_path.is_file():
        params_path.write_text(json.dumps(params), encoding="utf-8")
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = (
            str(_SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        ).rstrip(os.pathsep)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "compliance_fem.gui.rebuild_worker",
                "--params-json",
                str(params_path),
                "--output-dir",
                str(cache_dir),
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path.cwd()),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Contact-model rebuild subprocess failed "
                f"(exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if not npz_path.is_file():
            raise RuntimeError(
                "Rebuild worker exited 0 but did not write contact_lookup.npz.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

    return load_contact_model(npz_path)


# Matplotlib tab10-ish palette so the Plotly figure matches the old pyplot look.
_C0 = "#1f77b4"
_C1 = "#ff7f0e"
_C2 = "#2ca02c"
_C3 = "#d62728"
_C5 = "#9467bd"
_C6 = "#e377c2"


MATERIAL_FILL_COLORS = {
    "FFTurbo": "rgba(31, 119, 180, 0.28)",
    "FFLeap": "rgba(255, 127, 14, 0.28)",
}


def _draw_regions(fig, shape, _line, show_reference_geometry: bool) -> None:
    """Stored region loops, heel edge and foam interface of the sole."""
    if show_reference_geometry:
        for k, (name, (xr, yr)) in enumerate(shape.region_ref.items()):
            _line(xr, yr, name="reference geometry" if k == 0 else None, color="black",
                  width=1, dash="dash", legend=(k == 0))
    if shape.foam_energy_density is not None and shape.foam_triangles is not None:
        for trace in foam_energy_traces(
            shape.foam_triangles, shape.foam_x_def, shape.foam_y_def, shape.foam_energy_density
        ):
            fig.add_trace(trace)
    for name, (xd, yd) in shape.region_def.items():
        if shape.foam_energy_density is not None:
            break
        material = shape.region_materials.get(name, "")
        fig.add_trace(
            go.Scatter(
                x=np.asarray(xd, dtype=float),
                y=np.asarray(yd, dtype=float),
                mode="lines",
                fill="toself",
                fillcolor=MATERIAL_FILL_COLORS.get(material, "rgba(127,127,127,0.25)"),
                line=dict(width=0, color="rgba(0,0,0,0)"),
                name=f"{name.replace('_', ' ')} ({material})" if material else name,
                hoverinfo="skip",
            )
        )
    if shape.heel_edge_def is not None:
        _line(*shape.heel_edge_def, name="heel edge", color=_C0, width=1.5)
    if shape.interface_def is not None:
        _line(*shape.interface_def, name="foam interface", color="#7f7f7f", width=1.2, dash="dot")
    if shape.toe_marker_xy is not None:
        fig.add_trace(
            go.Scatter(
                x=[shape.toe_marker_xy[0]], y=[shape.toe_marker_xy[1]], mode="markers",
                name="point toe", marker=dict(symbol="star", size=9, color="#444444"),
                hoverinfo="skip",
            )
        )


def _draw_shape(
    shape,
    *,
    show_chord: bool = True,
    show_toe_joint_line: bool = True,
    show_reference_geometry: bool = True,
) -> go.Figure:
    """Build the fixed-frame deformed-geometry figure as Plotly (fast Streamlit updates)."""
    x_top_d = shape.x_top_def
    x_bot_d = shape.x_bottom_def

    fig = go.Figure()

    def _line(xs, ys, *, name=None, color="black", width=1.5, dash=None, legend=True, **kwargs):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(xs, dtype=float),
                y=np.asarray(ys, dtype=float),
                mode="lines",
                name=name,
                showlegend=bool(legend and name),
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="skip",
                **kwargs,
            )
        )

    if shape.has_render_mesh:
        _draw_regions(fig, shape, _line, show_reference_geometry)

    # Plate: Hermite superposition when available, otherwise schematic interface.
    if shape.has_plate:
        if (
            shape.show_undeformed_plate
            and shape.plate_x_ref is not None
            and shape.plate_y_ref is not None
        ):
            _line(
                shape.plate_x_ref,
                shape.plate_y_ref,
                name="undeformed plate",
                color="#737373",
                width=1.5,
                dash="dash",
            )
        px = shape.plate_x_def if shape.plate_x_def is not None else shape.x_plate_def
        py = shape.plate_y_def if shape.plate_y_def is not None else shape.y_plate_def
        if px is not None and py is not None:
            color_vals = shape.plate_color_values
            if (
                shape.plate_color_quantity == "strain_energy_density"
                and color_vals is not None
                and np.size(color_vals) == np.size(px)
                and np.size(px) > 1
            ):
                for trace in plate_energy_traces(px, py, color_vals):
                    fig.add_trace(trace)
            elif color_vals is not None and np.size(color_vals) == np.size(px) and np.size(px) > 1:
                units = shape.plate_color_units or ""
                cbar_title = f"plate ({units})" if units else "plate"
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(px, dtype=float),
                        y=np.asarray(py, dtype=float),
                        mode="lines+markers",
                        name="deformed plate",
                        line=dict(width=3, color="#1a1a1a"),
                        marker=dict(
                            size=5,
                            color=np.asarray(color_vals, dtype=float),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=cbar_title, thickness=12, len=0.7),
                            line=dict(width=0),
                        ),
                        hovertemplate="%{marker.color:.4g}<extra>plate</extra>",
                    )
                )
            else:
                _line(px, py, name="deformed plate", color="#1a1a1a", width=3)
        if (
            shape.show_plate_nodes
            and shape.plate_node_x is not None
            and shape.plate_node_y is not None
        ):
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(shape.plate_node_x, dtype=float),
                    y=np.asarray(shape.plate_node_y, dtype=float),
                    mode="markers",
                    name="plate nodes",
                    marker=dict(size=7, color="#0d0d0d"),
                    hoverinfo="skip",
                )
            )
        if (
            shape.show_plate_rotations
            and shape.plate_quiver_x is not None
            and shape.plate_quiver_y is not None
            and shape.plate_quiver_u is not None
            and shape.plate_quiver_v is not None
        ):
            qx = np.asarray(shape.plate_quiver_x, dtype=float)
            qy = np.asarray(shape.plate_quiver_y, dtype=float)
            qu = np.asarray(shape.plate_quiver_u, dtype=float)
            qv = np.asarray(shape.plate_quiver_v, dtype=float)
            # None-separated segments keep this a single trace (cheap to marshall).
            xs = np.empty(qx.size * 3, dtype=float)
            ys = np.empty(qy.size * 3, dtype=float)
            xs[0::3] = qx
            ys[0::3] = qy
            xs[1::3] = qx + qu
            ys[1::3] = qy + qv
            xs[2::3] = np.nan
            ys[2::3] = np.nan
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="plate rotation",
                    line=dict(color="#262626", width=1.2),
                    hoverinfo="skip",
                )
            )
    elif shape.x_plate_def is not None and shape.y_plate_def is not None:
        _line(
            shape.x_plate_def,
            shape.y_plate_def,
            name="plate (schematic)",
            color="#262626",
            width=2,
        )

    _line(x_top_d, shape.y_top_def, name="deformed top", color=_C0, width=2)
    _line(x_bot_d, shape.y_bottom_def, name="deformed bottom", color=_C1, width=2)
    if show_chord:
        _line(
            shape.chord_x,
            shape.chord_y,
            name=f"heel-to-toe chord (φ={shape.phi_deg:.3g}°)",
            color=_C5,
            width=1.5,
            dash="dashdot",
        )

    # Ground line, active patches along the deformed bottom, node classes.
    x_pad = 0.05 * shape.L
    _line(
        [-x_pad, shape.L + x_pad],
        [shape.ground_y, shape.ground_y],
        name="ground",
        color="#4d4d4d",
        width=1,
    )
    for k, ((cx, cy), p) in enumerate(zip(shape.contact_curves, shape.patches)):
        _line(
            cx, cy,
            name=f"contact patch {{{p.start}..{p.end}}}, x in [{p.start_x:.3g}, {p.end_x:.3g}]",
            color=_C3, width=6,
        )
    styles = {
        NODE_ACTIVE: ("active nodes", dict(symbol="circle", size=6, color=_C3)),
        NODE_INACTIVE: ("inactive nodes", dict(symbol="circle-open", size=5, color=_C2)),
        NODE_TRANSITION: ("transition nodes (g ≈ λ ≈ 0)", dict(symbol="diamond", size=8, color="#ff9f1c",
                                                              line=dict(width=1, color="black"))),
    }
    for cls, (label, marker) in styles.items():
        sel = shape.node_class == cls
        if np.any(sel):
            fig.add_trace(
                go.Scatter(
                    x=shape.bottom_node_x[sel], y=shape.bottom_node_y[sel], mode="markers", name=label,
                    marker=marker, customdata=np.flatnonzero(sel),
                    hovertemplate="node %{customdata}<extra>" + label + "</extra>",
                )
            )
    if shape.edge_estimate_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in shape.edge_estimate_points], y=[p[1] for p in shape.edge_estimate_points],
                mode="markers", name="derived contact-edge estimate (display only)",
                marker=dict(symbol="line-ns-open", size=14, color=_C3, line=dict(width=2)),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[shape.anchor_xy[0]], y=[shape.anchor_xy[1]], mode="markers",
            name="display pivot (reaction-weighted contact centroid)",
            marker=dict(symbol="x-thin-open", size=11, color="#7f7f7f", line=dict(width=2, color="#7f7f7f")),
            hoverinfo="skip",
        )
    )

    # Applied / resultant force direction in the fixed frame.
    fx, fy = shape.force_vector
    fmag = float(np.hypot(fx, fy))
    if fmag > 0.0:
        arrow = 0.3 * max(shape.L, shape.H)
        x0, y0 = shape.force_origin
        x1 = x0 + arrow * fx / fmag
        y1 = y0 + arrow * fy / fmag
        fig.add_annotation(
            x=x1,
            y=y1,
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
            arrowcolor=_C6,
        )
        # Invisible legend proxy for the annotation arrow.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="fixed-frame force direction",
                line=dict(color=_C6, width=2),
                hoverinfo="skip",
            )
        )

    if show_toe_joint_line:
        fig.add_vline(x=shape.L - shape.toe_length, line=dict(color="#666666", width=1, dash="dot"))
        # Legend entry for the toe-joint marker (add_vline has no legend).
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="L − toe length",
                line=dict(color="#666666", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis_title="x (fixed frame)",
        yaxis_title="y (fixed frame)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        margin=dict(l=50, r=30, t=40, b=40),
        height=480,
        template="plotly_white",
        uirevision="shape-plot",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(constrain="domain")
    return fig


def _fmt_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.2f} ms"


def _profile_figure(shape) -> go.Figure:
    """Gap and normal reaction over every bottom node (material x), active nodes highlighted."""
    from plotly.subplots import make_subplots

    x = shape.bottom_material_x
    act = shape.active
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("fixed-frame normal gap g (mm)", "compressive normal nodal reaction λ (N/m)"))
    fig.add_trace(go.Scatter(x=x, y=shape.bottom_gap * 1e3, mode="lines+markers", name="gap",
                             marker=dict(size=4, color=np.where(act, _C3, _C2)), line=dict(color="#888888")),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=shape.bottom_normal_reaction, name="λ", marker_color=np.where(act, _C3, _C2)),
                  row=2, col=1)
    for p in shape.patches:
        fig.add_vrect(x0=p.start_x, x1=p.end_x, fillcolor=_C3, opacity=0.08, line_width=0, row="all", col="all")
    fig.update_xaxes(title_text="bottom node material x (m)", row=2, col=1)
    fig.update_layout(height=420, showlegend=False, margin=dict(l=50, r=20, t=40, b=40), template="plotly_white")
    return fig


def _solve(model: ContactModel, *, toe_mode: str, Fx: float, Fy: float, phi_deg: float, theta_deg: float,
           toe_cfg: ToeSpringConfig, width: float, tol: ContactTolerances) -> tuple[ContactSolution, ToeFrameSolution | None]:
    """One evaluation, warm-started from the previous one in this session.

    Returns the contact solution and, with the passive toe spring, the coupled toe frame.
    """
    op = model.operator
    warm_key = f"gui_warm_{id(model)}_{toe_mode}"
    warm: WarmStart | None = st.session_state.get(warm_key)
    F = np.array([float(Fx), float(Fy)])
    if toe_mode == TOE_PASSIVE:
        low = float(np.hypot(Fx, Fy)) * width < float(toe_cfg.toe_low_force_threshold_N)
        frame = solve_toe_contact(op, F, float(np.deg2rad(phi_deg)), toe_cfg, width, warm=warm, low_load=low,
                                  tolerances=tol, cold_scan_points=int(toe_cfg.toe_root_scan_points))
        st.session_state[warm_key] = frame.warm_start()
        return frame.contact, frame
    varphi = float(np.deg2rad(phi_deg)) - op.phi_ref
    sol = solve_contact_state(op, varphi, float(np.tan(np.deg2rad(theta_deg))), F,
                              warm_active=None if warm is None else warm.active, tolerances=tol)
    st.session_state[warm_key] = WarmStart(theta_rad=float(np.deg2rad(theta_deg)), active=sol.active.copy())
    return sol, None


def _parse_width_profile(text: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``x_mm, width_mm`` per line (heel -> toe) -> metres; converted to SI exactly once here."""
    xs, bs = [], []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p for p in line.replace(";", ",").replace("\t", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Each line needs 'x_mm, width_mm', got {line!r}.")
        xs.append(float(parts[0]) / 1000.0)
        bs.append(float(parts[1]) / 1000.0)
    return tuple(xs), tuple(bs)


def _out_of_plane_controls(shoe_width_m: float) -> tuple[OutOfPlaneSettings | None, str]:
    """Sidebar inputs of the auxiliary 2.5D stiffness (modelling assumptions are labelled)."""
    enabled = st.checkbox("Enable 2.5D stiffness calculation", value=True, key="gui_oop_enabled")
    source = st.radio("Width profile source", ["Constant (shoe width above)", "Profile (x, width table)"],
                      key="gui_oop_width_source")
    profile_text = ""
    if source.startswith("Profile"):
        profile_text = st.text_area(
            "Width profile (x mm, width mm per line, heel → toe)",
            value="0, 70\n135, 95\n270, 80",
            key="gui_oop_width_profile",
            help="Assumption: normal stiffness uniform across each transverse strip of this width.",
        )
    ratio_mode = st.radio(
        "Lateral-to-normal stiffness ratio η_t (assumption)",
        ["Thin-layer G/M from foam ν", "User value"],
        key="gui_oop_ratio_mode",
        help="Elastic lateral shear / normal stiffness; not a friction coefficient.",
    )
    eta = None
    if ratio_mode == "User value":
        eta = float(st.number_input("η_t", value=0.44, min_value=0.0, step=0.01, format="%.3f", key="gui_oop_eta"))
    yaw_ref = st.radio("Yaw reference", ["center_of_pressure", "shoe_reference"], key="gui_oop_yaw_ref")
    fixed_x_mm = st.number_input(
        "Fixed yaw-reference x (mm, material; also the no-load fallback)", value=135.0, step=1.0,
        key="gui_oop_yaw_x",
    )
    cond = float(st.number_input("Condition-number warning threshold", value=1.0e10, format="%.2e",
                                 key="gui_oop_cond"))
    try:
        x_prof, b_prof = _parse_width_profile(profile_text) if profile_text else (None, None)
        return OutOfPlaneSettings(
            enabled=bool(enabled), shoe_width_m=float(shoe_width_m), shoe_width_x=x_prof, shoe_width_values=b_prof,
            lateral_to_normal_stiffness_ratio=eta, yaw_reference=str(yaw_ref),
            fixed_yaw_reference_x=float(fixed_x_mm) / 1000.0, condition_number_warning=cond,
        ), ""
    except ValueError as exc:
        return None, f"2.5D settings invalid: {exc}"


def _out_of_plane_section(
    model: ContactModel, sol: ContactSolution, settings: OutOfPlaneSettings | None, *,
    poisson_ratio: float | None = None,
) -> None:
    """2.5D tangent stiffness of the displayed (converged) state; never changes the plotted sole.

    ``poisson_ratio`` (contact foam, from the applied GUI materials) is used when the
    model file does not record it.
    """
    st.subheader("2.5D out-of-plane tangent stiffness")
    if settings is None:
        st.caption("Unavailable: fix the 2.5D settings in the sidebar.")
        return
    nu = model.contact_foam_poisson_ratio if model.contact_foam_poisson_ratio is not None else poisson_ratio
    oop = compute_out_of_plane_stiffness(model.operator, sol, settings, poisson_ratio=nu)
    st.session_state["gui_oop_result"] = oop
    if oop.status == "disabled":
        st.caption("2.5D stiffness calculation disabled.")
        return
    cols = st.columns(3)
    cols[0].metric("Lateral k_z (kN/m)", _fmt_value(oop.lateral_translation_stiffness / 1e3))
    cols[1].metric("Roll k_roll (N·m/rad)", _fmt_value(oop.roll_stiffness))
    cols[2].metric("Yaw k_yaw (N·m/rad)", _fmt_value(oop.yaw_stiffness))
    w = oop.shoe_width_summary
    width_txt = (
        f"{1e3 * w['mean_m']:.1f} mm" if w.get("mode") == "constant"
        else f"{1e3 * w.get('min_m', float('nan')):.1f}–{1e3 * w.get('max_m', float('nan')):.1f} mm (profile)"
    )
    cols = st.columns(4)
    cols[0].metric("Active contact nodes", f"{oop.active_contact_node_count}")
    cols[1].metric("Width used", width_txt if oop.active_contact_node_count else f"{1e3 * settings.shoe_width_m:.1f} mm")
    cols[2].metric("η_t", _fmt_value(oop.lateral_to_normal_stiffness_ratio))
    cols[3].metric("Yaw reference x (m)", _fmt_value(oop.yaw_reference_x))
    msg = f"status: {oop.status}; η_t source: {oop.lateral_ratio_source}; yaw reference: {oop.yaw_reference_source}; " \
          f"tangent cond {oop.condition_number:.3e}; time {_fmt_ms(oop.compute_time_s)}."
    if oop.status not in ("ok", "no_contact"):
        st.warning(msg)
    else:
        st.caption(msg)
    for wmsg in oop.warnings:
        st.caption(f"⚠ {wmsg}")
    st.caption(
        "Linearized about the converged sagittal state with the active set frozen (2.5D transverse-strip "
        "approximation, not a 3D FEM). Values may jump when the active set changes. They do not affect the "
        "sagittal solve or the plotted sole."
    )


def _fmt_value(v: float) -> str:
    return NOT_AVAILABLE if not np.isfinite(v) else f"{v:.4g}"


def _state_npz(model: ContactModel, sol: ContactSolution, theta_rad: float) -> bytes:
    """Download payload of one contact state (per metre width, SI)."""
    buf = io.BytesIO()
    d = sol.diagnostics
    np.savez_compressed(
        buf,
        x_bottom=model.x_bottom, y_bottom=model.y_bottom, gap=sol.gap, normal_reaction=sol.lam,
        tangential_reaction=sol.tau, active=sol.active, bottom_displacement_local=sol.d_b,
        top_displacement_local=sol.d_t, top_force_local=sol.F_t, bottom_force_local=sol.f_b,
        frame_translation=sol.T, varphi=sol.varphi, alpha=sol.alpha, F_star=sol.F_star,
        theta_rad=float(theta_rad),
        patches=np.asarray([(p.start, p.end) for p in d["patches"]], dtype=int).reshape(-1, 2),
        diagnostics_json=json.dumps({k: v for k, v in d.items() if k != "patches"}, default=str),
    )
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="Continuous sole contact", layout="wide")
    st.title("Force-controlled continuous unilateral contact (co-rotating top frame)")
    st.caption(
        "Geometry and materials rebuild the measured-sole FEM compliance and the shared contact "
        "operators in a subprocess. Fx, Fy in the sidebar are fixed-frame ground reaction forces "
        "per metre width (leftward/upward positive; the solver's top loads use the opposite signs). "
        "Every bottom node may touch the flat ground; the active set comes from one "
        "complementarity solve (no contact intervals)."
    )
    with st.expander("What do φ and θ mean?", expanded=False):
        st.markdown(HELP_TEXT)
    with st.expander("How is the contact set found?", expanded=False):
        st.markdown(CONTACT_HELP)

    with st.sidebar:
        with st.expander("Geometry", expanded=False):
            measured_setup = None
            source = st.radio(
                "Measured-sole parameters from",
                ["Sidebar controls", "Config file (JSON)"],
                key="gui_setup_source",
                horizontal=True,
                help=(
                    "Config file: shoe length, materials, plate EI, mesh, toe length and "
                    "bend radius all come from the JSON file (the sidebar values for those are ignored)."
                ),
            )
            if source == "Config file (JSON)":
                config_path = st.text_input(
                    "Config file", value=DEFAULT_MEASURED_CONFIG, key="gui_setup_config"
                )
                try:
                    cfg_file = Path(config_path)
                    measured_setup = _load_setup(
                        str(cfg_file.resolve()), cfg_file.stat().st_mtime
                    )
                except (OSError, ValueError) as exc:
                    st.error(f"Cannot use config: {exc}")
                if measured_setup is not None:
                    s = measured_setup.sole
                    st.caption(
                        f"L = {s.shoe_length_mm:g} mm · {s.upper_foam_material} / {s.lower_foam_material} · "
                        f"EI = {s.EI_plate_Nm2_per_m:g} N·m² · mesh {s.mesh_size_m * 1e3:g} mm · "
                        f"toe length = {measured_setup.lookup.toe_length_mm:.1f} mm · "
                        f"R_min(45°) = {measured_setup.lookup.min_bend_radius_mm:g} mm"
                    )
            geometry_csv = st.text_input(
                "Geometry CSV",
                value=DEFAULT_GEOMETRY_CSV,
                key="gui_geometry_csv",
                help="Normalized sole CSV (x/L, y/L; heel-to-toe x, upward y).",
            )
            L_mm = dual_linear(
                "Overall shoe length (mm)",
                key="gui_shoe_length_mm",
                min_value=L_MM_MIN,
                max_value=L_MM_MAX,
                default=DEFAULTS.shoe_length_mm,
                step=1.0,
                fmt="%.1f",
                help="Projected heel-bottom-to-toe-tip length (not the outsole arc length).",
            )
            upper_foam = st.selectbox(
                "Upper foam material", list(FOAM_MATERIALS), index=list(FOAM_MATERIALS).index(DEFAULT_UPPER_FOAM),
                key="gui_upper_foam",
            )
            lower_foam = st.selectbox(
                "Lower foam material", list(FOAM_MATERIALS), index=list(FOAM_MATERIALS).index(DEFAULT_LOWER_FOAM),
                key="gui_lower_foam",
            )
            L = mm_to_m(L_mm)

        with st.expander("Toe Parameters", expanded=False):
            toe_length_mm_max = float(min(TOE_LENGTH_MM_MAX, L_mm - 1.0))
            toe_length_mm = dual_linear(
                "Toe length (mm)",
                key="gui_toe_length_mm",
                min_value=TOE_LENGTH_MM_MIN,
                max_value=toe_length_mm_max,
                default=float(np.clip(m_to_mm(DEFAULTS.softplus_toe_length), TOE_LENGTH_MM_MIN, toe_length_mm_max)),
                step=0.1,
                fmt="%.2f",
                help="Distance from the softplus joint to the toe tip; the joint sits at L − toe length.",
            )
            softplus_toe_length = mm_to_m(toe_length_mm)
            radius_min = min_bend_radius_mm_from_kappa(L, KAPPA_MAX)
            radius_max = min_bend_radius_mm_from_kappa(L, KAPPA_MIN)
            min_bend_radius_mm = dual_log(
                "Minimum bend radius at 45° (mm)",
                key="gui_min_bend_radius_mm",
                min_value=radius_min,
                max_value=radius_max,
                default=min_bend_radius_mm_from_kappa(L, DEFAULTS.softplus_kappa),
                fmt="%.3g",
                help=(
                    "Tightest radius of the toe shape at a 45° bend (α = 1). "
                    "The softplus sharpness is κ = 4L / R."
                ),
            )
            softplus_kappa = kappa_from_min_bend_radius(L, min_bend_radius_mm)

        with st.expander("Material Parameters", expanded=False):
            ffturbo_E_Pa = mpa_to_pa(
                dual_log(
                    "FFTurbo young's modulus (MPa)",
                    key="gui_E1_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.ffturbo_E_Pa),
                    fmt="%.4g",
                )
            )
            ffleap_E_heel_Pa = mpa_to_pa(
                dual_log(
                    "FFLeap young's modulus at heel (MPa)",
                    key="gui_E_heel_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.ffleap_E_heel_Pa),
                    fmt="%.4g",
                )
            )
            ffleap_E_toe_Pa = mpa_to_pa(
                dual_log(
                    "FFLeap young's modulus at toe (MPa)",
                    key="gui_E_toe_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.ffleap_E_toe_Pa),
                    fmt="%.4g",
                )
            )
            ffturbo_nu = dual_linear(
                "FFTurbo Poisson ratio",
                key="gui_nu1",
                min_value=NU_MIN,
                max_value=NU_MAX,
                default=DEFAULTS.ffturbo_nu,
                step=0.001,
                fmt="%.4g",
            )
            ffleap_nu = dual_linear(
                "FFLeap Poisson ratio",
                key="gui_nu2",
                min_value=NU_MIN,
                max_value=NU_MAX,
                default=DEFAULTS.ffleap_nu,
                step=0.001,
                fmt="%.4g",
            )
            EI_plate_Nm2_per_m = dual_log(
                "Plate bending stiffness EI (N·m²)",
                key="gui_EI_nm2",
                min_value=EI_NM2_MIN,
                max_value=EI_NM2_MAX,
                default=DEFAULTS.EI_plate_Nm2_per_m,
                fmt="%.4g",
            )

        with st.expander("Mesh parameters", expanded=False):
            measured_mesh: dict = {
                "mesh_size_mm": dual_linear(
                    "Mesh size (mm)", key="gui_mesh_size_mm", min_value=1.0, max_value=8.0,
                    default=DEFAULTS.mesh_size_mm, step=0.1, fmt="%.2f",
                )
            }
            for key, label, default in (
                ("toe_refinement", "Toe refinement (× mesh size)", 0.4),
                ("heel_corner_refinement", "Heel-corner refinement (× mesh size)", 0.5),
                ("interface_refinement", "Interface refinement (× mesh size)", 0.6),
                ("plate_end_refinement", "Plate-end refinement (× mesh size)", 0.4),
            ):
                measured_mesh[key] = dual_linear(
                    label, key=f"gui_{key}", min_value=0.1, max_value=1.0,
                    default=default, step=0.05, fmt="%.2f",
                )
            reciprocity_tol = st.number_input(
                "Compliance symmetry tolerance",
                value=1.0e-6,
                format="%.2e",
                key="gui_reciprocity_tol",
                help=(
                    "Maximum relative C symmetry/reciprocity error accepted when "
                    "building the lookup."
                ),
            )

        draft_model = _canonical_model_params(
            ffturbo_E_Pa=ffturbo_E_Pa,
            ffturbo_nu=ffturbo_nu,
            ffleap_E_heel_Pa=ffleap_E_heel_Pa,
            ffleap_E_toe_Pa=ffleap_E_toe_Pa,
            ffleap_nu=ffleap_nu,
            EI_plate_Nm2_per_m=EI_plate_Nm2_per_m,
            softplus_toe_length=softplus_toe_length,
            softplus_kappa=softplus_kappa,
            reciprocity_tol=reciprocity_tol,
            measured={
                "geometry_csv": geometry_csv,
                "shoe_length_mm": L_mm,
                "upper_foam_material": upper_foam,
                "lower_foam_material": lower_foam,
                **measured_mesh,
            },
        )
        if measured_setup is not None:
            draft_model = _draft_from_setup(measured_setup)
            if "prebuilt_lookup" in draft_model:
                st.caption(f"Using the matching prebuilt lookup {draft_model['prebuilt_lookup']}.")
        _config_export(draft_model, measured_setup)
        if _APPLIED_MODEL_KEY not in st.session_state:
            st.session_state[_APPLIED_MODEL_KEY] = draft_model

        with st.expander("Toe model", expanded=True):
            toe_mode = st.radio(
                "Toe angle",
                [TOE_PASSIVE, TOE_PRESCRIBED],
                index=0,
                key="gui_toe_mode",
                help="Passive: θ from the exact toe-spring equilibrium; φ is then the rearfoot angle.",
            )
            toe_k = st.number_input("Toe stiffness k (N·m/rad)", value=25.0, min_value=0.0, key="gui_toe_k")
            toe_theta0_deg = st.number_input("Neutral toe angle θ0 (°)", value=0.0, key="gui_toe_theta0")
            shoe_width = st.number_input(
                "Shoe width (m)", value=0.10, min_value=0.01, step=0.01, format="%.3f", key="gui_shoe_width",
                help="Toe spring is per shoe; the plane-strain model is per metre width (Q_total = width × Q).",
            )
            toe_cfg = ToeSpringConfig(
                toe_stiffness_Nm_per_rad=float(toe_k), toe_neutral_angle_rad=float(np.deg2rad(toe_theta0_deg))
            )

        with st.expander("2.5D out-of-plane stiffness", expanded=False):
            oop_settings, oop_error = _out_of_plane_controls(float(shoe_width))
            if oop_error:
                st.error(oop_error)

        with st.expander("Loads & angles", expanded=False):
            phi_deg = st.slider(
                "Foot pitch angle (°, dorsiflexion positive)",
                ANGLES.phi_min_deg,
                ANGLES.phi_max_deg,
                ANGLES.phi_default_deg,
                ANGLES.phi_step_deg,
                key="gui_phi_deg",
            )
            theta_deg = st.slider(
                "Toe bend angle (°)",
                ANGLES.theta_min_deg,
                ANGLES.theta_max_deg,
                ANGLES.theta_default_deg,
                ANGLES.theta_step_deg,
                key="gui_theta_deg",
            )
            # Sidebar uses GRF signs (leftward/upward +). Solver top resultants
            # keep the opposite convention; negate at the GUI boundary only.
            Fx_kn = dual_linear(
                "Horizontal ground reaction force (kN, leftward positive)",
                key="gui_Fx_kn_grf",
                min_value=n_to_kn(-FX_N_MAX),
                max_value=n_to_kn(-FX_N_MIN),
                default=n_to_kn(-DEFAULTS.Fx),
                step=0.01,
                fmt="%.4g",
            )
            Fy_kn = dual_linear(
                "Vertical ground reaction force (kN, upward positive)",
                key="gui_Fy_kn_grf",
                min_value=n_to_kn(-FY_N_MAX),
                max_value=n_to_kn(-FY_N_MIN),
                default=n_to_kn(-DEFAULTS.Fy),
                step=0.01,
                fmt="%.4g",
            )
            Fx = -kn_to_n(Fx_kn)
            Fy = -kn_to_n(Fy_kn)
            st.session_state["gui_query_phi_deg"] = float(phi_deg)
            st.session_state["gui_query_theta_deg"] = float(theta_deg)
            st.session_state["gui_query_Fx"] = float(Fx)
            st.session_state["gui_query_Fy"] = float(Fy)

        with st.expander("Display", expanded=False):
            scale_mode = st.radio(
                "Deformation display scale",
                ["auto", "true", "manual"],
                index=1,
                key="gui_scale_mode_v2",
            )
            manual_scale = st.slider(
                "Manual deformation scale",
                0.1,
                100.0,
                10.0,
                0.1,
                key="gui_manual_scale",
            )
            st.caption("The rigid rotation φ is never scaled; only the elastic part is.")
            show_reference_geometry = st.checkbox(
                "Show reference geometry",
                value=True,
                key="gui_show_reference_geometry",
            )
            show_chord = st.checkbox(
                "Show heel-to-toe chord",
                value=True,
                key="gui_show_chord",
            )
            show_toe_joint_line = st.checkbox(
                "Show toe joint position",
                value=True,
                key="gui_show_toe_joint_line",
            )
            show_profiles = st.checkbox("Show gap / reaction profiles", value=True, key="gui_show_profiles")

        with st.expander("Solver tolerances", expanded=False):
            gap_rel_tol = st.number_input(
                "Gap tolerance (× node spacing)", value=1.0e-9, format="%.2e", key="gui_gap_rel_tol"
            )
            reaction_rel_tol = st.number_input(
                "Reaction tolerance (× |F*|)", value=1.0e-9, format="%.2e", key="gui_reaction_rel_tol"
            )
            max_as_iter = st.number_input(
                "Maximum active-set iterations", value=200, min_value=1, step=10, key="gui_max_as_iter"
            )
            cond_warn = st.number_input(
                "Condition-number warning", value=1.0e10, format="%.2e", key="gui_cond_warn"
            )
            tol = ContactTolerances(
                gap_rel_tol=float(gap_rel_tol), reaction_rel_tol=float(reaction_rel_tol),
                maximum_active_set_iterations=int(max_as_iter), condition_number_warning=float(cond_warn),
            )

        st.header("Apply model")
        pending = st.session_state[_APPLIED_MODEL_KEY] != draft_model
        st.caption(
            "Geometry / materials / softplus / mesh / reciprocity rebuild the FEM + "
            "contact model. Set loads & angles above first; "
            "Apply does not reset them. Fx, Fy, φ, θ also update the plot without Apply."
        )
        if pending:
            st.warning("Draft model parameters differ from the applied model.")
        else:
            st.caption("Draft matches the applied model.")
        if st.button(
            "Apply & rebuild model",
            type="primary",
            use_container_width=True,
            key="gui_apply_rebuild",
            help="Run Gmsh + FEM + contact model once for the draft parameters above.",
        ):
            # No st.rerun(): continue this run so query widgets stay rendered and
            # the rebuilt plot uses the current φ/θ/Fx/Fy.
            st.session_state[_APPLIED_MODEL_KEY] = draft_model

        applied_model = st.session_state[_APPLIED_MODEL_KEY]

    model_key = tuple(sorted(applied_model.items()))
    prev_key = st.session_state.get("_lookup_key")
    cache_hit = prev_key == model_key

    t_rebuild0 = time.perf_counter()
    with st.spinner(
        "Building FEM compliance + contact model in a subprocess…"
        if not cache_hit
        else "Loading cached model…"
    ):
        model = _build_model(json.dumps(applied_model, sort_keys=True))
    t_rebuild = time.perf_counter() - t_rebuild0
    st.session_state["_lookup_key"] = model_key
    if cache_hit:
        st.caption(
            f"Model load {_fmt_ms(t_rebuild)} (cache hit — no FEM recompute)."
        )
    else:
        st.caption(f"Model rebuild {_fmt_ms(t_rebuild)}.")

    softplus_toe_length = float(applied_model["softplus_toe_length"])
    softplus_kappa = float(applied_model["softplus_kappa"])

    with st.sidebar:
        with st.expander("Plate display", expanded=False):
            has_plate_model = bool(getattr(model, "has_plate_response", False))
            if not has_plate_model:
                st.caption(
                    "Plate response unavailable (N/A). The rebuild did not attach "
                    "plate_response — check FEM factorization / plate recovery."
                )
                show_plate = False
                show_undeformed_plate = False
                show_plate_nodes = False
                show_plate_rotations = False
                plate_color_key = "None"
                samples_per_element = 16
            else:
                show_plate = st.checkbox("Show plate", value=True, key="gui_show_plate")
                show_undeformed_plate = st.checkbox(
                    "Show undeformed plate",
                    value=False,
                    key="gui_show_undeformed_plate_v2",
                )
                show_plate_nodes = st.checkbox(
                    "Show plate nodes",
                    value=False,
                    key="gui_show_plate_nodes",
                )
                show_plate_rotations = st.checkbox(
                    "Show plate rotations",
                    value=False,
                    key="gui_show_plate_rotations",
                )
                plate_color_key = st.selectbox(
                    "Plate color by",
                    list(PLATE_COLOR_OPTIONS),
                    index=list(PLATE_COLOR_OPTIONS).index("Strain energy density"),
                    key="gui_plate_color",
                )
                samples_per_element = st.slider(
                    "Samples per plate element",
                    min_value=4,
                    max_value=32,
                    value=16,
                    step=1,
                    key="gui_plate_samples",
                )
        plate_color_quantity = PLATE_COLOR_OPTIONS[plate_color_key]
        foam_energy = st.checkbox(
            "Color foam by strain energy density",
            value=bool(getattr(model, "render_section", None)),
            disabled=not bool(getattr(model, "render_section", None)),
            key="gui_foam_energy",
            help="Per-element plane-strain energy density from the stored render influence (measured sole only).",
        )

        st.subheader("Model summary (SI)")
        op = model.operator
        st.write(
            {
                "L": float(model.L),
                "toe_length": softplus_toe_length,
                "κ": softplus_kappa,
                **{k: applied_model[k] for k in ("ffturbo_E_Pa", "ffturbo_nu", "ffleap_E_heel_Pa", "ffleap_E_toe_Pa", "ffleap_nu", "EI_plate_Nm2_per_m", "mesh_size_mm")},
                "schema_version": int(model.schema_version),
                "phi_ref_deg": float(np.rad2deg(model.phi_ref)),
                "contact_set_model": "continuous node-to-ground complementarity",
                "tangential_contact_law": model.metadata.get("tangential_contact_law"),
                "n_bottom_nodes": int(model.n_bottom_nodes),
                "G symmetry error": float(op.G_symmetry_error),
                "eig(G) min / max": [float(op.G_min_eigenvalue), float(op.G_max_eigenvalue)],
                "build_time_s": float(model.build_time_s),
                "has_plate_response": bool(model.has_plate_response),
            }
        )
        gm = model.geometry_metadata
        if gm:
            st.subheader("Measured geometry")
            st.write(
                {
                    "source_geometry_filename": gm.get("source_geometry_filename"),
                    "shoe_length_mm": gm.get("shoe_length_mm"),
                    "upper_foam_material": gm.get("upper_foam_material"),
                    "lower_foam_material": gm.get("lower_foam_material"),
                    "plate_arc_length_m": gm.get("plate_arc_length_m"),
                    "corner_interior_angles_deg": gm.get("corner_interior_angles_deg"),
                    "region_areas_m2": gm.get("region_areas_m2"),
                    "mesh_quality": gm.get("mesh_quality"),
                    "shared_toe_policy": gm.get("shared_toe_policy"),
                    "n_top_nodes": int(model.n_top_nodes),
                    "n_bottom_nodes": int(model.n_bottom_nodes),
                }
            )

    # Re-read the query snapshot after the (possibly long) model rebuild so the
    # plot always matches the sidebar widgets, not stale defaults.
    phi_deg = float(st.session_state["gui_query_phi_deg"])
    theta_deg = float(st.session_state["gui_query_theta_deg"])
    Fx = float(st.session_state["gui_query_Fx"])
    Fy = float(st.session_state["gui_query_Fy"])

    t0 = time.perf_counter()
    sol, frame = _solve(model, toe_mode=toe_mode, Fx=Fx, Fy=Fy, phi_deg=phi_deg, theta_deg=theta_deg,
                        toe_cfg=toe_cfg, width=float(shoe_width), tol=tol)
    t_calc = time.perf_counter() - t0
    d = sol.diagnostics
    if not sol.converged:
        st.error(f"Contact solve did not converge ({d['lcp_status']}); the state shown is the last iterate.")
    elif frame is not None and not frame.converged:
        st.warning(f"Toe equilibrium: {frame.status} (residual {frame.toe_equilibrium_residual_Nm:.3e} N·m).")
    else:
        st.success(
            f"{d['n_active']} active node(s) in {d['n_patches']} patch(es); "
            f"{d['active_set_iterations']} active-set iteration(s)"
            + (f", {frame.toe_iterations} toe evaluations" if frame is not None else "")
            + "."
        )
    if d["disconnected_contact"]:
        st.warning(
            "Multiple disconnected contact patches: valid geometry, mesh-scale oscillation, or a sign / "
            "conditioning problem. Patches are shown separately."
        )
    if d["active_block_rcond"] > 0 and 1.0 / d["active_block_rcond"] > tol.condition_number_warning:
        st.warning(f"Active-block condition estimate {1.0 / d['active_block_rcond']:.3e} exceeds the warning level.")
    if abs(d["mu_eff"]) > tol.mu_eff_warning:
        st.warning(f"Shear ratio |Fx/Fy| = {abs(d['mu_eff']):.3g} is large for proportional shear.")

    t1 = time.perf_counter()
    state = selection_proxy(model, frame if frame is not None else sol, phi=float(np.deg2rad(phi_deg)))
    shape = build_shape_plot_data(
        model,
        state,
        scale_mode=scale_mode,
        manual_scale=manual_scale,
        show_plate=show_plate,
        show_undeformed_plate=show_undeformed_plate,
        show_plate_nodes=show_plate_nodes,
        show_plate_rotations=show_plate_rotations,
        plate_color_quantity=plate_color_quantity,
        samples_per_element=samples_per_element,
        foam_energy=foam_energy,
    )
    shape_fig = _draw_shape(
        shape,
        show_chord=show_chord,
        show_toe_joint_line=show_toe_joint_line,
        show_reference_geometry=show_reference_geometry,
    )
    t_shape = time.perf_counter() - t1

    na = NOT_AVAILABLE
    theta_out = frame.theta_deg if frame is not None else float(theta_deg)
    key_cols = st.columns(5)
    key_cols[0].metric("Toe angle θ (°)", f"{theta_out:.4g}")
    key_cols[1].metric("Active nodes", f"{d['n_active']}")
    key_cols[2].metric("Contact patches", f"{d['n_patches']}")
    key_cols[3].metric("COP x (fixed, m)", na if not np.isfinite(d["cop_fixed_x"]) else f"{d['cop_fixed_x']:.5g}")
    key_cols[4].metric("Contact solve", _fmt_ms(sol.timings["contact_solve_s"]))

    t2 = time.perf_counter()
    st.plotly_chart(shape_fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": False, "responsive": True})
    st.caption(shape.caption)
    if show_profiles:
        st.plotly_chart(_profile_figure(shape), use_container_width=True)
    t_shape_render = time.perf_counter() - t2

    with st.expander("Contact outputs", expanded=True):
        patches = d["patches"]
        st.write(
            {
                "active nodes": [int(i) for i in np.flatnonzero(sol.active)],
                "patches (first..last node, material x)": [
                    f"{{{p.start}..{p.end}}} x∈[{p.start_x:.5g}, {p.end_x:.5g}]" for p in patches
                ],
                "derived contact-edge estimate x (display only)": [
                    [round(p.edge_estimate_start_x, 6), round(p.edge_estimate_end_x, 6)] for p in patches
                ],
            }
        )
        cols = st.columns(4)
        cols[0].metric("min gap (m)", f"{d['min_gap']:.3e}")
        cols[1].metric("min active λ (N/m)", f"{d['min_reaction']:.4g}")
        cols[2].metric("max |g λ|", f"{d['max_complementarity']:.3e}")
        cols[3].metric("Admissible (tolerances)", str(d["admissible"]))
        cols = st.columns(4)
        cols[0].metric("Σλ normal (N/m)", f"{float(sol.lam.sum()):.5g}")
        cols[1].metric("Στ tangential (N/m)", f"{float(sol.tau.sum()):.5g}")
        cols[2].metric("GRF moment about origin (N·m/m)", f"{d['grf_moment_fixed']:.5g}")
        cols[3].metric("μ_eff = Fx*/Fy*", f"{d['mu_eff']:.4g}")
        cols = st.columns(4)
        cols[0].metric("Force reconstruction error", f"{d['force_residual']:.3e}")
        cols[1].metric("Equilibrium force residual", f"{d['equilibrium_force_residual']:.3e}")
        cols[2].metric("Equilibrium moment residual", f"{d['equilibrium_moment_residual']:.3e}")
        cols[3].metric("Active-block rcond", f"{d['active_block_rcond']:.3e}")
        cols = st.columns(4)
        cols[0].metric("Top Mz (shoe frame, N·m/m)", f"{d['Mz_shoe']:.5g}")
        cols[1].metric("Toe moment about MTP (N·m/m)", f"{d['T_toe']:.5g}")
        cols[2].metric("Active-set iterations", f"{d['active_set_iterations']} ({d['active_set_rule']})")
        cols[3].metric("Frame translation T_y (m)", f"{float(sol.T[1]):.5g}")

    with st.expander("Toe spring", expanded=frame is not None):
        if frame is not None:
            cols = st.columns(4)
            cols[0].metric("θ (°)", f"{frame.theta_deg:.5g}")
            cols[1].metric("Spring moment (N·m)", f"{frame.toe_spring_moment_Nm:.5g}")
            cols[2].metric("Equilibrium residual (N·m)", f"{frame.toe_equilibrium_residual_Nm:.3e}")
            cols[3].metric("Toe evaluations", f"{frame.toe_iterations}")
            st.caption(
                f"status {frame.status}; roots θ = {[round(r, 4) for r in frame.roots_theta_deg]} °; "
                f"stable = {frame.toe_equilibrium_stable}; Q_α = {frame.Q_alpha_shoe_on_foot:.5g} N·m; "
                f"θ = arctan(α) exactly."
            )
        else:
            st.caption("θ is prescribed (no toe-spring solve).")

    lower = str(applied_model.get("lower_foam_material", ""))
    lower_nu = applied_model.get("ffleap_nu" if lower == FOAM_FFLEAP else "ffturbo_nu")
    _out_of_plane_section(model, sol, oop_settings, poisson_ratio=None if lower_nu is None else float(lower_nu))

    with st.expander("Plate results", expanded=False):
        if not has_plate_model:
            st.caption("Plate response unavailable (N/A).")
        elif not show_plate or not shape.has_plate:
            st.caption("Plate display disabled or no plate state for this selection (N/A).")
        else:
            summary = shape.plate_summary or {}
            keys = list(PLATE_SUMMARY_LABELS)
            for i in range(0, len(keys), 4):
                row_keys = keys[i : i + 4]
                mcols = st.columns(4)
                for col, key in zip(mcols, row_keys):
                    val = summary.get(key)
                    label = PLATE_SUMMARY_LABELS[key]
                    col.metric(label, na if val is None else f"{float(val):.6g}")
            st.caption(
                f"Hermite samples: "
                f"{0 if shape.plate_x_def is None else len(shape.plate_x_def)}; "
                f"color = {plate_color_key}; elastic scale s = {shape.scale:.4g}."
            )

    t_total = t_calc + t_shape + t_shape_render
    with st.expander("Timing (this rerun)", expanded=False):
        tcols = st.columns(5)
        tcols[0].metric("Free response", _fmt_ms(sol.timings["free_response_s"]))
        tcols[1].metric("Contact solve", _fmt_ms(sol.timings["contact_solve_s"]))
        tcols[2].metric("Toe outer", _fmt_ms(frame.timings["toe_outer_s"]) if frame is not None else na)
        tcols[3].metric("Reconstruction + plots", _fmt_ms(t_shape + t_shape_render))
        tcols[4].metric("Total evaluation", _fmt_ms(t_total))
        st.caption(
            f"Calculation (all contact solves{' and toe iterations' if frame is not None else ''}): {_fmt_ms(t_calc)}. "
            "The model rebuild runs once per model in a subprocess (cached)."
        )

    st.subheader("Export")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if st.button("Save contact model NPZ"):
        npz_path = save_contact_model(model, EXPORT_DIR)
        st.success(f"Wrote contact model to {npz_path}")
    st.download_button(
        "Download contact state NPZ",
        data=_state_npz(model, sol, np.deg2rad(theta_out)),
        file_name="contact_state.npz",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
