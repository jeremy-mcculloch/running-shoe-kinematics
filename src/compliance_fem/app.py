"""Streamlit GUI for force-controlled contact-edge evaluation."""

from __future__ import annotations

# Force a non-interactive backend before pyplot is imported. The MacOSX
# backend is not safe inside Streamlit's script runner and can segfault.
import matplotlib

matplotlib.use("Agg")

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import plotly.io as pio

    # Prefer orjson when available; Plotly falls back to the stdlib encoder.
    pio.json.config.default_engine = "orjson"
except Exception:
    pass

from compliance_fem.config import RuntimeAngleConfig
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    load_contact_lookup,
    save_contact_lookup,
)
from compliance_fem.contact_topology import ContactMode, ContactType
from compliance_fem.corotation import transform_to_fixed_frame
from compliance_fem.force_control import (
    Tolerances,
    angles_to_coefficients,
    evaluate_from_angles,
    export_evaluation_csv,
    export_selected_result,
)
from compliance_fem.gui_params import (
    DEFAULTS,
    E_MPA_MAX,
    E_MPA_MIN,
    EI_NM2_MAX,
    EI_NM2_MIN,
    FX_N_MAX,
    FX_N_MIN,
    FY_N_MAX,
    FY_N_MIN,
    H_MM_MAX,
    H_MM_MIN,
    KAPPA_MAX,
    KAPPA_MIN,
    LA_MM_MAX,
    LA_MM_MIN,
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
from compliance_fem.shape_render import build_shape_plot_data


ANGLES = RuntimeAngleConfig()
EXPORT_DIR = Path("outputs/gui_contact_lookup")
_SRC_ROOT = Path(__file__).resolve().parents[1]
_APPLIED_MODEL_KEY = "_applied_model_params"


def _canonical_model_params(
    *,
    L: float,
    h1_heel: float,
    h1_toe: float,
    h2_heel: float,
    h2_toe: float,
    E1: float,
    nu1: float,
    E_heel: float,
    E_toe: float,
    nu2: float,
    EI_plate: float,
    nx: int,
    ny1: int,
    ny2: int,
    element_order: int,
    softplus_a: float,
    softplus_kappa: float,
    reciprocity_tol: float,
) -> dict[str, float | int]:
    """Round floats so tiny slider noise does not bust the rebuild cache."""
    return {
        "L": round(float(L), 12),
        "h1_heel": round(float(h1_heel), 12),
        "h1_toe": round(float(h1_toe), 12),
        "h2_heel": round(float(h2_heel), 12),
        "h2_toe": round(float(h2_toe), 12),
        "E1": float(f"{float(E1):.8e}"),
        "nu1": round(float(nu1), 10),
        "E_heel": float(f"{float(E_heel):.8e}"),
        "E_toe": float(f"{float(E_toe):.8e}"),
        "nu2": round(float(nu2), 10),
        "EI_plate": float(f"{float(EI_plate):.8e}"),
        "nx": int(nx),
        "ny1": int(ny1),
        "ny2": int(ny2),
        "element_order": int(element_order),
        "softplus_a": round(float(softplus_a), 12),
        "softplus_kappa": float(f"{float(softplus_kappa):.8e}"),
        "reciprocity_tol": float(f"{float(reciprocity_tol):.8e}"),
    }

PLATE_COLOR_OPTIONS = {
    "None": "none",
    "Normal displacement": "normal_displacement",
    "Tangential displacement": "tangential_displacement",
    "Rotation": "rotation",
    "Curvature": "curvature",
    "Bending moment": "bending_moment",
    "Shear force": "shear_force",
    "Axial constraint force": "axial_constraint_force",
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
**phi** is the absolute fixed-frame angle of the current heel-to-toe chord
(counterclockwise positive). The rigid rotation applied to the reference mesh is
`varphi = phi - phi_ref`, where `phi_ref` is the reference chord angle stored in
the lookup file.

**theta** drives the single endpoint-fixed softplus shape mode through its
amplitude `alpha = tan(theta)`. Positive theta tilts the toe segment up relative
to the chord. The mode vanishes at both x=0 and x=L, so heel and toe heights are
set by the frame, not by theta.

Lookup tables are generated once, independently of phi. Finite rotation enters
only through scalar combinations of five precomputed contact modes with
`gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)]`, where
`(d_ax, d_ay)` is the anchor translation solved from `Fx`, `Fy`.

Linearized strain is intentionally retained while boundary positions use the
exact finite rotation. This is fast, but it is not objective for locally large
rotations.
"""

CONTACT_MODE_HELP = """
Three contiguous ground-contact topologies are stored: **heel** contact on
`[0, l]`, **toe** contact on `[l, L]`, and **full** contact on `[0, L]`. The
transition node `i` always belongs to the contact set.

**Auto** reads the two full-contact corner normal reactions first. Both
compressive selects full contact immediately. A tensile heel corner routes the
search to toe candidates, a tensile toe corner routes to heel candidates, and
two tensile corners search both partial families. If no routed family is
admissible the app reports that the supported topology family is insufficient -
that happens when the true contact patch is an interior interval, which none of
the three topologies can represent.

Full contact has no lift-off edge. Its basis modes are anchored at the numerical
point `x_a = L/2`, which is a decomposition point only and is drawn distinctly
from a real contact edge.
"""


@st.cache_resource(show_spinner=False)
def _build_layered_lookup(
    L: float,
    h1_heel: float,
    h1_toe: float,
    h2_heel: float,
    h2_toe: float,
    E1: float,
    nu1: float,
    E_heel: float,
    E_toe: float,
    nu2: float,
    EI_plate: float,
    nx: int,
    ny1: int,
    ny2: int,
    element_order: int,
    softplus_a: float,
    softplus_kappa: float,
    reciprocity_tol: float,
):
    """Rebuild FEM + lookup in a subprocess, then load the NPZ into this process.

    Isolates Gmsh / SuperLU from Streamlit to avoid macOS native segfaults.
    """
    import hashlib

    params = _canonical_model_params(
        L=L,
        h1_heel=h1_heel,
        h1_toe=h1_toe,
        h2_heel=h2_heel,
        h2_toe=h2_toe,
        E1=E1,
        nu1=nu1,
        E_heel=E_heel,
        E_toe=E_toe,
        nu2=nu2,
        EI_plate=EI_plate,
        nx=nx,
        ny1=ny1,
        ny2=ny2,
        element_order=element_order,
        softplus_a=softplus_a,
        softplus_kappa=softplus_kappa,
        reciprocity_tol=reciprocity_tol,
    )
    key = hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
                "compliance_fem.gui_rebuild_worker",
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
                "Layered lookup rebuild subprocess failed "
                f"(exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if not npz_path.is_file():
            raise RuntimeError(
                "Lookup worker exited 0 but did not write contact_lookup.npz.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

    return load_contact_lookup(npz_path)


# Matplotlib tab10-ish palette so the Plotly figure matches the old pyplot look.
_C0 = "#1f77b4"
_C1 = "#ff7f0e"
_C2 = "#2ca02c"
_C3 = "#d62728"
_C5 = "#9467bd"
_C6 = "#e377c2"


def _draw_shape(
    shape,
    selection=None,
    *,
    show_chord: bool = True,
    show_a_line: bool = True,
    show_reference_geometry: bool = True,
) -> go.Figure:
    """Build the fixed-frame deformed-geometry figure as Plotly (fast Streamlit updates)."""
    del selection  # kept for call-site compatibility
    x = shape.x_dense
    x_top_r = shape.x_top_ref
    x_bot_r = shape.x_bottom_ref
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

    # Reference (undeformed, unrotated) geometry.
    if show_reference_geometry:
        if shape.y_plate_ref is None:
            _line(
                [0, shape.L, shape.L, 0, 0],
                [0, 0, shape.H, shape.H, 0],
                name="reference geometry",
                color="black",
                width=1,
                dash="dash",
            )
        else:
            _line(x_bot_r, shape.y_bottom_ref, color="black", width=1, dash="dash", legend=False)
            if not shape.has_plate:
                _line(x, shape.y_plate_ref, color="black", width=1, dash="dash", legend=False)
            _line(
                x_top_r,
                shape.y_top_ref,
                name="reference geometry",
                color="black",
                width=1,
                dash="dash",
            )
            _line(
                [0, 0],
                [shape.y_bottom_ref[0], shape.y_top_ref[0]],
                color="black",
                width=1,
                dash="dash",
                legend=False,
            )
            _line(
                [shape.L, shape.L],
                [shape.y_bottom_ref[-1], shape.y_top_ref[-1]],
                color="black",
                width=1,
                dash="dash",
                legend=False,
            )

    # Rotated / deformed foam fill.
    poly_x = np.concatenate([x_bot_d, x_top_d[::-1], x_bot_d[:1]])
    poly_y = np.concatenate([shape.y_bottom_def, shape.y_top_def[::-1], shape.y_bottom_def[:1]])
    fig.add_trace(
        go.Scatter(
            x=poly_x,
            y=poly_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.20)",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="rotated/deformed",
            hoverinfo="skip",
        )
    )

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
            if color_vals is not None and np.size(color_vals) == np.size(px) and np.size(px) > 1:
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
    _line(
        [x_bot_d[0], x_top_d[0]],
        [shape.y_bottom_def[0], shape.y_top_def[0]],
        color=_C0,
        width=1.5,
        legend=False,
    )
    _line(
        [x_bot_d[-1], x_top_d[-1]],
        [shape.y_bottom_def[-1], shape.y_top_def[-1]],
        color=_C0,
        width=1.5,
        legend=False,
    )
    if show_chord:
        _line(
            shape.chord_x,
            shape.chord_y,
            name=f"heel-to-toe chord (φ={shape.phi_deg:.3g}°)",
            color=_C5,
            width=1.5,
            dash="dashdot",
        )

    # Ground line, contact interval, free interval(s), and the contact edge or
    # the numerical full-contact anchor.
    x_pad = 0.05 * shape.L
    _line(
        [-x_pad, shape.L + x_pad],
        [shape.ground_y, shape.ground_y],
        name="ground",
        color="#4d4d4d",
        width=1,
    )
    span_lo, span_hi = shape.contact_span
    _line(
        [span_lo, span_hi],
        [shape.ground_y, shape.ground_y],
        name=f"{shape.contact_type} contact [{span_lo:.3g}, {span_hi:.3g}]",
        color=_C3,
        width=6,
    )
    for k, (free_lo, free_hi) in enumerate(shape.free_spans):
        _line(
            [free_lo, free_hi],
            [shape.ground_y, shape.ground_y],
            name="free bottom" if k == 0 else None,
            color=_C2,
            width=6,
            legend=(k == 0),
        )
    if shape.is_anchor_only:
        # Open square: x_a is a numerical basis anchor, not a lift-off edge.
        fig.add_trace(
            go.Scatter(
                x=[shape.anchor_x],
                y=[shape.ground_y],
                mode="markers",
                name=f"numerical anchor x_a=L/2 (x_rot={shape.x_anchor_rot:.4g})",
                marker=dict(
                    symbol="square-open",
                    size=11,
                    color="#404040",
                    line=dict(width=1.5, color="#404040"),
                ),
                hoverinfo="skip",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[shape.l],
                y=[shape.ground_y],
                mode="markers",
                name=f"contact edge l (x_rot={shape.x_contact_rot:.4g})",
                marker=dict(symbol="circle", size=9, color=_C3),
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

    if show_a_line:
        fig.add_vline(x=shape.a, line=dict(color="#666666", width=1, dash="dot"))
        # Legend entry for the a-marker (add_vline has no legend).
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="a",
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


def _diag_plot(x, y, xlabel, ylabel, title, admissible=None, selected=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, "-o", markersize=3)
    has_legend = False
    if admissible is not None and np.any(admissible):
        ax.plot(x[admissible], np.asarray(y)[admissible], "o", color="C2", label="admissible")
        has_legend = True
    if selected is not None and selected >= 0:
        ax.axvline(x[selected], color="C3", ls="--", label="selected")
        has_legend = True
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if has_legend:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fmt_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.1f} ms"


def main() -> None:
    st.set_page_config(page_title="Contact topology lookup", layout="wide")
    st.title("Force-controlled contact evaluation (heel / full / toe, co-rotating top frame)")
    st.caption(
        "Geometry and materials rebuild the layered FEM compliance and contact lookup in-process. "
        "Fx, Fy in the sidebar are fixed-frame ground reaction forces: leftward/upward "
        "positive (solver top resultants use the opposite signs). φ is the absolute "
        "heel-to-toe chord angle and θ drives the single softplus shape mode through "
        "α = tan(θ). Selection uses fixed-frame normal gaps, normal reactions, and the "
        "two full-contact corner reactions only (never R_t, T_toe, or x_cm)."
    )
    with st.expander("What do φ and θ mean?", expanded=False):
        st.markdown(HELP_TEXT)
    with st.expander("How is the contact topology chosen?", expanded=False):
        st.markdown(CONTACT_MODE_HELP)

    with st.sidebar:
        with st.expander("Geometry", expanded=False):
            L_mm = dual_linear(
                "Shoe length (mm)",
                key="gui_L_mm",
                min_value=L_MM_MIN,
                max_value=L_MM_MAX,
                default=m_to_mm(DEFAULTS.L),
                step=1.0,
                fmt="%.1f",
            )
            L = mm_to_m(L_mm)
            h1_heel = mm_to_m(
                dual_linear(
                    "FFTurbo thickness at heel (mm)",
                    key="gui_h1_heel_mm",
                    min_value=H_MM_MIN,
                    max_value=H_MM_MAX,
                    default=m_to_mm(DEFAULTS.h1_heel),
                    step=0.1,
                    fmt="%.2f",
                )
            )
            h1_toe = mm_to_m(
                dual_linear(
                    "FFTurbo thickness at toe (mm)",
                    key="gui_h1_toe_mm",
                    min_value=H_MM_MIN,
                    max_value=H_MM_MAX,
                    default=m_to_mm(DEFAULTS.h1_toe),
                    step=0.1,
                    fmt="%.2f",
                )
            )
            h2_heel = mm_to_m(
                dual_linear(
                    "FFLeap thickness at heel (mm)",
                    key="gui_h2_heel_mm",
                    min_value=H_MM_MIN,
                    max_value=H_MM_MAX,
                    default=m_to_mm(DEFAULTS.h2_heel),
                    step=0.1,
                    fmt="%.2f",
                )
            )
            h2_toe = mm_to_m(
                dual_linear(
                    "FFLeap thickness at toe (mm)",
                    key="gui_h2_toe_mm",
                    min_value=H_MM_MIN,
                    max_value=H_MM_MAX,
                    default=m_to_mm(DEFAULTS.h2_toe),
                    step=0.1,
                    fmt="%.2f",
                )
            )

        with st.expander("Toe Parameters", expanded=False):
            la_mm_max = float(min(LA_MM_MAX, L_mm))
            L_minus_a_mm = dual_linear(
                "Toe length (mm)",
                key="gui_L_minus_a_mm",
                min_value=LA_MM_MIN,
                max_value=la_mm_max,
                default=float(np.clip(m_to_mm(DEFAULTS.L_minus_a), LA_MM_MIN, la_mm_max)),
                step=0.1,
                fmt="%.2f",
                help="Softplus transition distance from the toe; a = L − (toe length).",
            )
            softplus_a = float(L - mm_to_m(L_minus_a_mm))
            softplus_kappa = dual_log(
                "Toe joint curvature (1/m)",
                key="gui_kappa",
                min_value=KAPPA_MIN,
                max_value=KAPPA_MAX,
                default=DEFAULTS.softplus_kappa,
                fmt="%.4g",
            )

        with st.expander("Material Parameters", expanded=False):
            E1 = mpa_to_pa(
                dual_log(
                    "FFTurbo young's modulus (MPa)",
                    key="gui_E1_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.E1),
                    fmt="%.4g",
                )
            )
            E_heel = mpa_to_pa(
                dual_log(
                    "FFLeap young's modulus at heel (MPa)",
                    key="gui_E_heel_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.E_heel),
                    fmt="%.4g",
                )
            )
            E_toe = mpa_to_pa(
                dual_log(
                    "FFLeap young's modulus at toe (MPa)",
                    key="gui_E_toe_mpa",
                    min_value=E_MPA_MIN,
                    max_value=E_MPA_MAX,
                    default=pa_to_mpa(DEFAULTS.E_toe),
                    fmt="%.4g",
                )
            )
            nu1 = dual_linear(
                "FFTurbo Poisson ratio",
                key="gui_nu1",
                min_value=NU_MIN,
                max_value=NU_MAX,
                default=DEFAULTS.nu1,
                step=0.001,
                fmt="%.4g",
            )
            nu2 = dual_linear(
                "FFLeap Poisson ratio",
                key="gui_nu2",
                min_value=NU_MIN,
                max_value=NU_MAX,
                default=DEFAULTS.nu2,
                step=0.001,
                fmt="%.4g",
            )
            EI_plate = dual_log(
                "Plate bending stiffness EI (N·m²)",
                key="gui_EI_nm2",
                min_value=EI_NM2_MIN,
                max_value=EI_NM2_MAX,
                default=DEFAULTS.EI_plate,
                fmt="%.4g",
            )

        with st.expander("Mesh parameters", expanded=False):
            nx = int(
                dual_linear(
                    "Elements along x direction",
                    key="gui_nx",
                    min_value=4.0,
                    max_value=200.0,
                    default=float(DEFAULTS.nx),
                    step=1.0,
                    fmt="%.0f",
                )
            )
            ny1 = int(
                dual_linear(
                    "Elements along y direction in FFTurbo",
                    key="gui_ny1",
                    min_value=1.0,
                    max_value=40.0,
                    default=float(DEFAULTS.ny1),
                    step=1.0,
                    fmt="%.0f",
                )
            )
            ny2 = int(
                dual_linear(
                    "Elements along y direction in FFLeap",
                    key="gui_ny2",
                    min_value=1.0,
                    max_value=40.0,
                    default=float(DEFAULTS.ny2),
                    step=1.0,
                    fmt="%.0f",
                )
            )
            element_order = int(
                dual_linear(
                    "Element polynomial order",
                    key="gui_element_order",
                    min_value=1.0,
                    max_value=2.0,
                    default=float(DEFAULTS.element_order),
                    step=1.0,
                    fmt="%.0f",
                )
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
            L=L,
            h1_heel=h1_heel,
            h1_toe=h1_toe,
            h2_heel=h2_heel,
            h2_toe=h2_toe,
            E1=E1,
            nu1=nu1,
            E_heel=E_heel,
            E_toe=E_toe,
            nu2=nu2,
            EI_plate=EI_plate,
            nx=nx,
            ny1=ny1,
            ny2=ny2,
            element_order=element_order,
            softplus_a=softplus_a,
            softplus_kappa=softplus_kappa,
            reciprocity_tol=reciprocity_tol,
        )
        if _APPLIED_MODEL_KEY not in st.session_state:
            st.session_state[_APPLIED_MODEL_KEY] = draft_model

        # Query controls must be rendered BEFORE Apply. Calling st.rerun() above
        # these widgets deletes their backend state while the frontend still
        # shows the old slider positions — exactly the "plot uses defaults"
        # bug. Persist copies under non-widget keys as well.
        contact_mode = ContactMode.AUTO

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
            show_a_line = st.checkbox(
                "Show toe joint position",
                value=True,
                key="gui_show_a_line",
            )
            show_diagnostics = st.checkbox(
                "Show diagnostic plots",
                value=False,
                key="gui_show_diagnostics",
                help=(
                    "Off by default: each rerun used to open ~19 Matplotlib figures, and "
                    "Streamlit script cancellation left them unclosed until the process "
                    "segfaulted. Enable only when you need the J(l) / gap / reaction charts."
                ),
            )

        with st.expander("Solver tolerances", expanded=False):
            tau_g = st.number_input(
                "Contact gap tolerance",
                value=0.0,
                format="%.2e",
                key="gui_tau_g",
            )
            tau_R = st.number_input(
                "Contact reaction tolerance",
                value=0.0,
                format="%.2e",
                key="gui_tau_R",
            )
            kf_cond_warn = st.number_input(
                "Force-solve conditioning warning threshold",
                value=float(ANGLES.kf_cond_warn),
                format="%.2e",
                key="gui_kf_cond_warn",
            )

        st.header("Apply model")
        pending = st.session_state[_APPLIED_MODEL_KEY] != draft_model
        st.caption(
            "Geometry / materials / softplus / mesh / reciprocity rebuild the FEM + "
            f"lookup (~10 s at nx={DEFAULTS.nx}). Set loads & angles above first; "
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
            help="Run Gmsh + FEM + contact lookup once for the draft parameters above.",
        ):
            # No st.rerun(): continue this run so query widgets stay rendered and
            # the rebuilt plot uses the current φ/θ/Fx/Fy.
            st.session_state[_APPLIED_MODEL_KEY] = draft_model

        applied_model = st.session_state[_APPLIED_MODEL_KEY]

    model_key = tuple(sorted(applied_model.items()))
    prev_key = st.session_state.get("_layered_lookup_key")
    cache_hit = prev_key == model_key

    t_rebuild0 = time.perf_counter()
    with st.spinner(
        "Building layered FEM compliance + contact lookup in a subprocess…"
        if not cache_hit
        else "Loading cached model…"
    ):
        lookup = _build_layered_lookup(
            L=float(applied_model["L"]),
            h1_heel=float(applied_model["h1_heel"]),
            h1_toe=float(applied_model["h1_toe"]),
            h2_heel=float(applied_model["h2_heel"]),
            h2_toe=float(applied_model["h2_toe"]),
            E1=float(applied_model["E1"]),
            nu1=float(applied_model["nu1"]),
            E_heel=float(applied_model["E_heel"]),
            E_toe=float(applied_model["E_toe"]),
            nu2=float(applied_model["nu2"]),
            EI_plate=float(applied_model["EI_plate"]),
            nx=int(applied_model["nx"]),
            ny1=int(applied_model["ny1"]),
            ny2=int(applied_model["ny2"]),
            element_order=int(applied_model["element_order"]),
            softplus_a=float(applied_model["softplus_a"]),
            softplus_kappa=float(applied_model["softplus_kappa"]),
            reciprocity_tol=float(applied_model["reciprocity_tol"]),
        )
    t_rebuild = time.perf_counter() - t_rebuild0
    st.session_state["_layered_lookup_key"] = model_key
    if cache_hit:
        st.caption(
            f"Model load {_fmt_ms(t_rebuild)} (cache hit — no FEM recompute)."
        )
    else:
        st.caption(f"Model rebuild {_fmt_ms(t_rebuild)}.")

    L = float(applied_model["L"])
    h1_heel = float(applied_model["h1_heel"])
    h1_toe = float(applied_model["h1_toe"])
    h2_heel = float(applied_model["h2_heel"])
    h2_toe = float(applied_model["h2_toe"])
    softplus_a = float(applied_model["softplus_a"])
    softplus_kappa = float(applied_model["softplus_kappa"])
    E1 = float(applied_model["E1"])
    nu1 = float(applied_model["nu1"])
    E_heel = float(applied_model["E_heel"])
    E_toe = float(applied_model["E_toe"])
    nu2 = float(applied_model["nu2"])
    EI_plate = float(applied_model["EI_plate"])
    nx = int(applied_model["nx"])
    ny1 = int(applied_model["ny1"])
    ny2 = int(applied_model["ny2"])
    element_order = int(applied_model["element_order"])
    reciprocity_tol = float(applied_model["reciprocity_tol"])

    with st.sidebar:
        with st.expander("Plate display", expanded=False):
            has_plate_lookup = bool(getattr(lookup, "has_plate_response", False))
            if not has_plate_lookup:
                st.caption(
                    "Plate response unavailable (N/A). Layered rebuild did not attach "
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
                    index=0,
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

        varphi, alpha, r_x, r_y = angles_to_coefficients(phi_deg, theta_deg, lookup.phi_ref)
        st.markdown(f"**φ_ref** = `{np.rad2deg(lookup.phi_ref):.6g}°`")
        st.markdown(f"**varphi** = φ − φ_ref = `{np.rad2deg(varphi):.6g}°`")
        st.markdown(f"**α** = tan(θ) = `{alpha:.6g}`")
        st.markdown(f"**cos(varphi) − 1** = `{r_x:.6g}`,  **−sin(varphi)** = `{r_y:.6g}`")
        st.caption(
            "γ = [α, d_ax, d_ay, cos(varphi)−1, −sin(varphi)] in saved basis order "
            "(top_shape_phi1, contact_translation_x/y, contact_rotation_x/y)."
        )

        st.subheader("Model summary (SI)")
        st.write(
            {
                "L": L,
                "h1_heel": h1_heel,
                "h1_toe": h1_toe,
                "h2_heel": h2_heel,
                "h2_toe": h2_toe,
                "a": softplus_a,
                "κ": softplus_kappa,
                "E1": E1,
                "nu1": nu1,
                "E_heel": E_heel,
                "E_toe": E_toe,
                "nu2": nu2,
                "EI_plate": EI_plate,
                "nx": nx,
                "ny1": ny1,
                "ny2": ny2,
                "element_order": element_order,
                "schema_version": lookup.schema_version,
                "geometry_type": lookup.geometry_type,
                "phi_ref_deg": float(np.rad2deg(lookup.phi_ref)),
                "n_records": lookup.n_records,
                "n_heel": int(lookup.rows_for(ContactType.HEEL).size),
                "n_toe": int(lookup.rows_for(ContactType.TOE).size),
                "n_full": int(lookup.rows_for(ContactType.FULL).size),
                "full_contact_anchor_x": lookup.full_contact_anchor_x,
                "has_plate_response": bool(lookup.has_plate_response),
                "LOOKUP_SCHEMA_VERSION": LOOKUP_SCHEMA_VERSION,
            }
        )

    tolerances = Tolerances(tau_g=tau_g, tau_R=tau_R, kf_cond_warn=kf_cond_warn)

    # Proximity to the previous selection is a final tie-breaker only; it exists
    # so the rendered topology does not flicker between numerically
    # indistinguishable candidates as the sliders move.
    previous_index = st.session_state.get("previous_edge_node_id")

    # Re-read query snapshot after the (possibly long) model rebuild so the plot
    # always matches the sidebar widgets, not stale defaults.
    phi_deg = float(st.session_state["gui_query_phi_deg"])
    theta_deg = float(st.session_state["gui_query_theta_deg"])
    Fx = float(st.session_state["gui_query_Fx"])
    Fy = float(st.session_state["gui_query_Fy"])

    t0 = time.perf_counter()
    selection = evaluate_from_angles(
        lookup,
        Fx,
        Fy,
        phi_deg,
        theta_deg,
        tolerances=tolerances,
        mode=contact_mode,
        previous_index=previous_index,
    )
    t_calc = time.perf_counter() - t0

    if selection.selected_row is None:
        st.error(selection.message)
        st.info(
            f"Calculation time: {_fmt_ms(t_calc)}; model rebuild: {_fmt_ms(t_rebuild)}"
        )
        st.stop()
    st.session_state["previous_edge_node_id"] = selection.selected_index

    if selection.approximate:
        st.warning(selection.message)
    else:
        st.success(selection.message)

    if selection.kf_illconditioned:
        st.warning(
            f"Translation-to-force matrix K_F is ill-conditioned "
            f"(cond={selection.kf_cond:.3e}). No regularization was applied."
        )

    na = "N/A"
    is_full = selection.is_full_contact
    span_lo, span_hi = selection.contact_span(lookup.L)

    t1 = time.perf_counter()
    shape = build_shape_plot_data(
        lookup,
        selection,
        scale_mode=scale_mode,
        manual_scale=manual_scale,
        show_plate=show_plate,
        show_undeformed_plate=show_undeformed_plate,
        show_plate_nodes=show_plate_nodes,
        show_plate_rotations=show_plate_rotations,
        plate_color_quantity=plate_color_quantity,
        samples_per_element=samples_per_element,
    )
    shape_fig = _draw_shape(
        shape,
        selection,
        show_chord=show_chord,
        show_a_line=show_a_line,
        show_reference_geometry=show_reference_geometry,
    )
    t_shape = time.perf_counter() - t1

    # Key outputs: top-right corner displacements (fixed frame, true scale),
    # toe moment about (a, H_a), and heel moment about the top-left corner.
    if (
        selection.top_u is not None
        and selection.top_v is not None
        and selection.anchor_x is not None
        and selection.d_ax is not None
        and selection.d_ay is not None
        and lookup.x_top is not None
        and len(lookup.x_top) > 0
    ):
        x_ref = float(lookup.x_top[-1])
        y_ref = (
            float(lookup.y_top[-1])
            if lookup.y_top is not None
            else float(lookup.H)
        )
        x_def, y_def = transform_to_fixed_frame(
            np.asarray([x_ref]),
            np.asarray([y_ref]),
            np.asarray([selection.top_u[-1]]),
            np.asarray([selection.top_v[-1]]),
            float(selection.anchor_x),
            float(selection.d_ax),
            float(selection.d_ay),
            float(selection.varphi),
            elastic_scale=1.0,
        )
        u_tr = float(np.asarray(x_def).reshape(-1)[0] - x_ref)
        v_tr = float(np.asarray(y_def).reshape(-1)[0] - y_ref)
    else:
        u_tr = v_tr = float("nan")
    t_toe = float(selection.T_toe) if selection.T_toe is not None else float("nan")
    row = selection.selected_row
    ev = selection.evaluation
    if (
        row is not None
        and ev.top_force_x is not None
        and ev.top_force_y is not None
        and lookup.x_top is not None
        and len(lookup.x_top) > 0
    ):
        y_top = (
            np.asarray(lookup.y_top, dtype=float)
            if lookup.y_top is not None
            else np.full(len(lookup.x_top), float(lookup.H))
        )
        # Heel moment: top nodal forces about the top-left corner (x=0 heel).
        x_tl = float(lookup.x_top[0])
        y_tl = float(y_top[0])
        m_heel = float(
            np.dot(lookup.x_top - x_tl, ev.top_force_y[row])
            - np.dot(y_top - y_tl, ev.top_force_x[row])
        )
    else:
        m_heel = float("nan")

    key_cols = st.columns(4)
    key_cols[0].metric("x toe displacement (mm)", f"{u_tr * 1.0e3:.6g}")
    key_cols[1].metric("y toe displacement (mm)", f"{v_tr * 1.0e3:.6g}")
    key_cols[2].metric("Toe moment", f"{t_toe:.6g}")
    key_cols[3].metric("Heel moment", f"{m_heel:.6g}")
    st.caption(
        "Toe corner displacements in the fixed frame (true scale, mm); "
        "Toe moment about (a, H_a); Heel moment = (x−x_tl)ᵀf_y − (y−y_tl)ᵀf_x "
        "of top nodal forces about the top-left corner (x_tl, y_tl)."
    )

    t2 = time.perf_counter()
    st.plotly_chart(
        shape_fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": False,
            "responsive": True,
        },
    )
    st.caption(shape.caption)
    t_shape_render = time.perf_counter() - t2

    with st.expander("Detailed outputs", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Contact type", selection.contact_type.value)
        cols[1].metric("Contact interval", f"[{span_lo:.4g}, {span_hi:.4g}]")
        cols[2].metric("Material contact edge l", na if is_full else f"{selection.l:.6g}")
        cols[3].metric(
            "Rotated contact coordinate",
            na if is_full else f"{selection.x_contact_rot:.6g}",
        )
        cols = st.columns(4)
        cols[0].metric(
            "Heel corner R_n^F (x=0)",
            na if selection.Rn_heel_corner is None else f"{selection.Rn_heel_corner:.6g}",
        )
        cols[1].metric(
            "Toe corner R_n^F (x=L)",
            na if selection.Rn_toe_corner is None else f"{selection.Rn_toe_corner:.6g}",
        )
        cols[2].metric("Full contact valid", str(selection.full_contact_valid))
        cols[3].metric(
            "Full valid (all reactions)", str(selection.full_contact_valid_strict)
        )
        cols = st.columns(4)
        cols[0].metric(
            "Edge-free gap g^F",
            na if not selection.has_free_edge else f"{selection.edge_free_gap:.6g}",
        )
        cols[1].metric(
            "Edge-contact R_n^F",
            na if is_full else f"{selection.edge_contact_reaction_normal:.6g}",
        )
        cols[2].metric(
            "Max free-surface penetration",
            na if is_full else f"{max(0.0, -selection.min_free_gap):.6g}",
        )
        cols[3].metric("Min contact R_n^F", f"{selection.min_contact_reaction:.6g}")
        cols = st.columns(4)
        cols[0].metric("Force reconstruction error", f"{selection.force_residual:.3e}")
        cols[1].metric("cond(K_F)", f"{selection.kf_cond:.3e}")
        cols[2].metric("Admissible", str(selection.exactly_admissible))
        cols[3].metric("Violation J", f"{selection.violation_score:.3e}")
        cols = st.columns(4)
        cols[0].metric("Anchor x_a", f"{selection.anchor_x:.6g}")
        cols[1].metric("Rotated anchor x_a,rot", f"{selection.x_anchor_rot:.6g}")
        cols[2].metric("d_ax", f"{selection.d_ax:.6g}")
        cols[3].metric("d_ay", f"{selection.d_ay:.6g}")
        cols = st.columns(4)
        cols[0].metric(
            "Fx* / reconstructed GRF (fixed)",
            f"{-selection.Fx_star:.4g} / {-selection.Fx:.4g}",
        )
        cols[1].metric(
            "Fy* / reconstructed GRF (fixed)",
            f"{-selection.Fy_star:.4g} / {-selection.Fy:.4g}",
        )
        cols[2].metric("x_cm (fixed frame)", f"{selection.x_cm:.6g}")
        cols[3].metric("x_cm − x_a (fixed)", f"{selection.x_cm_rel:.6g}")
        cols = st.columns(4)
        cols[0].metric("Toe moment about (a, H_a)", f"{selection.T_toe:.6g}")
        cols[1].metric(
            "R_t at edge (tangential)",
            na if is_full else f"{selection.edge_contact_reaction_tangential:.6g}",
        )
        cols[2].metric("# admissible in family", str(selection.n_admissible))
        cols[3].metric(
            "Routed families", "/".join(k.value for k in selection.routed_families)
        )
        st.caption(
            f"Routing: {selection.routing_reason} "
            "x_cm uses the visualization gauge r_a^F = (x_a, 0); only x_cm − x_a is gauge "
            "independent. R_t may take either sign because contact remains perfectly sticking. "
            "Full-contact validity is the corner-only criterion; the all-reaction column is a "
            "stricter diagnostic that never replaces it."
        )

    with st.expander("Plate results", expanded=False):
        if not has_plate_lookup:
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

    # Diagnostics are opt-in: building ~19 figures on every slider tick leaked
    # Matplotlib state when Streamlit cancelled mid-rerun, which led to segfaults.
    plt.close("all")
    t_diag_build = 0.0
    t_diag_render = 0.0
    if show_diagnostics:
        t3 = time.perf_counter()
        ev = selection.evaluation
        adm = ev.admissible & ev.well_conditioned
        figs = []
        try:
            # Heel and toe families share the l axis, so they are plotted separately;
            # the single full-contact record has no l and is annotated instead.
            for family in (ContactType.HEEL, ContactType.TOE):
                rows = ev.rows_for(family)
                if rows.size == 0:
                    continue
                sel_local = (
                    int(np.flatnonzero(rows == selection.selected_row)[0])
                    if selection.selected_row in rows
                    else -1
                )
                x = ev.l[rows]
                for values, ylabel, title in (
                    (ev.violation_score[rows], "J", f"{family.value}: violation score J(l)"),
                    (ev.min_free_gap[rows], "min g^F", f"{family.value}: minimum free gap"),
                    (
                        ev.min_contact_reaction[rows],
                        "min R_n^F",
                        f"{family.value}: minimum contact reaction",
                    ),
                    (ev.d_ax[rows], "d_ax", f"{family.value}: anchor translation d_ax(l)"),
                    (ev.d_ay[rows], "d_ay", f"{family.value}: anchor translation d_ay(l)"),
                    (ev.x_cm[rows], "x_cm", f"{family.value}: fixed-frame center of effort"),
                    (ev.T_toe[rows], "T_toe", f"{family.value}: toe moment"),
                    (ev.kf_cond[rows], "cond(K_F)", f"{family.value}: conditioning of K_F(l)"),
                ):
                    figs.append(
                        _diag_plot(x, values, "l", ylabel, title, adm[rows], sel_local)
                    )

            full_row = ev.full_contact_row
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(
                ["heel corner (x=0)", "toe corner (x=L)"],
                [ev.Rn_heel_corner[full_row], ev.Rn_toe_corner[full_row]],
                color=["C0", "C1"],
            )
            ax.axhline(0.0, color="0.3", lw=1)
            ax.set_ylabel("R_n^F")
            ax.set_title(
                f"Full-contact corner normal reactions "
                f"(valid={bool(ev.full_contact_valid[full_row])}, "
                f"J={ev.violation_score[full_row]:.3e})"
            )
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            figs.append(fig)

            marker_x = selection.anchor_x if selection.is_full_contact else selection.l
            marker_style = ":" if selection.is_full_contact else "--"
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(lookup.x_bottom, selection.full_bottom_gap, "-o", markersize=3)
            ax.axvline(marker_x, color="C3", ls=marker_style)
            ax.set_title(
                f"Selected ({selection.contact_type.value}) bottom fixed-frame normal gap"
            )
            ax.set_xlabel("x (material)")
            ax.set_ylabel("g^F")
            ax.grid(True, alpha=0.3)
            figs.append(fig)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(
                lookup.x_bottom,
                selection.full_bottom_reaction_normal,
                "-o",
                markersize=3,
                label="R_n^F",
            )
            ax.plot(
                lookup.x_bottom,
                selection.full_bottom_reaction_tangential,
                "-s",
                markersize=3,
                label="R_t^F",
            )
            ax.axvline(marker_x, color="C3", ls=marker_style)
            ax.set_title(
                f"Selected ({selection.contact_type.value}) bottom nodal reaction force "
                "(not pressure)"
            )
            ax.set_xlabel("x (material)")
            ax.set_ylabel("nodal reaction force")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            figs.append(fig)
            t_diag_build = time.perf_counter() - t3

            with st.expander("Diagnostic plots", expanded=True):
                t4 = time.perf_counter()
                for fig in figs:
                    st.pyplot(fig)
                t_diag_render = time.perf_counter() - t4
                st.caption(
                    f"Diagnostic build {_fmt_ms(t_diag_build)}, "
                    f"render {_fmt_ms(t_diag_render)}"
                )
        finally:
            for fig in figs:
                plt.close(fig)
            plt.close("all")
    else:
        st.caption("Diagnostic plots disabled (enable in the sidebar to build them).")

    t_plot_total = t_shape + t_shape_render + t_diag_build
    t_total = t_rebuild + t_calc + t_plot_total

    with st.expander("Timing (this rerun)", expanded=False):
        tcols = st.columns(5)
        tcols[0].metric("Model rebuild", _fmt_ms(t_rebuild))
        tcols[1].metric("Calculation", _fmt_ms(t_calc))
        tcols[2].metric("Shape build", _fmt_ms(t_shape))
        tcols[3].metric("Shape plotly", _fmt_ms(t_shape_render))
        tcols[4].metric("Diag. build", _fmt_ms(t_diag_build))
        st.caption(
            f"Measured total (rebuild + calc + shape + diag build): {_fmt_ms(t_total)}. "
            "Model rebuild = FEM compliance + contact lookup in a subprocess "
            "(cached via st.cache_resource; isolates Gmsh from Streamlit). "
            "Calculation = evaluate_from_angles (rotation, 2x2 solve, superposition). "
            "Shape/diag build = figure construction (Plotly shape / Matplotlib diagnostics). "
            "Shape plotly = Streamlit Plotly marshalling (orjson when installed). "
            "Diagnostic figures are only built when enabled in the sidebar."
        )

    st.subheader("Export")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if st.button("Save contact lookup NPZ/CSV/JSON"):
        npz_path = save_contact_lookup(lookup, EXPORT_DIR)
        st.success(f"Wrote lookup to {npz_path}")
    tmp_csv = EXPORT_DIR / "query_table.csv"
    export_evaluation_csv(selection, tmp_csv)
    st.download_button(
        "Download candidate table CSV",
        data=tmp_csv.read_text(encoding="utf-8"),
        file_name="candidate_evaluation.csv",
        mime="text/csv",
    )
    tmp_npz = EXPORT_DIR / "selected_result.npz"
    export_selected_result(selection, lookup, tmp_npz)
    st.download_button(
        "Download selected result NPZ",
        data=tmp_npz.read_bytes(),
        file_name="selected_result.npz",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
