"""Streamlit GUI for force-controlled single-interval contact evaluation."""

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

from compliance_fem.config import DEFAULT_LOWER_FOAM, DEFAULT_UPPER_FOAM, FOAM_MATERIALS, RuntimeAngleConfig
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    load_contact_lookup,
    save_contact_lookup,
)
from compliance_fem.contact_topology import (
    CONTACT_MODE_LABELS,
    ContactMode,
    ContactType,
    parse_contact_mode,
    validate_interval,
)
from compliance_fem.corotation import transform_to_fixed_frame
from compliance_fem.force_control import (
    DISCONNECTED_CONTACT_WARNING,
    NOT_AVAILABLE,
    Tolerances,
    angles_to_coefficients,
    format_optional,
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
    measured: dict | None = None,
) -> dict[str, float | int]:
    """Round floats so tiny slider noise does not bust the rebuild cache.

    ``measured`` (measured-sole geometry only) adds the CSV path, a content hash
    of the CSV, shoe length, foam assignment and mesh refinement settings.
    """
    out = {
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
    if measured is not None:
        csv_path = str(measured["geometry_csv"])
        out.update(
            {
                "geometry_model": GEOMETRY_MODEL_MEASURED,
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
        )
        for key in ("landmark_tolerance", "curvature_max_turn_deg", "min_angle_deg"):
            if key in measured:
                out[key] = float(f"{float(measured[key]):.10e}")
    return out


def _measured_draft_from_setup(setup, layered_model: dict) -> dict:
    """Applied-model dict taking every measured-sole parameter from a JSON config.

    Layered-only entries (thicknesses, nx, ny) keep their sidebar values. When
    ``lookup.output_dir`` holds a lookup built from exactly this setup it is
    loaded directly instead of rebuilding.
    """
    from compliance_fem.gui_params import gui_params_from_setup
    from compliance_fem.measured_config_file import matching_prebuilt_lookup

    p = gui_params_from_setup(setup)
    out = _canonical_model_params(
        L=setup.sole.L,
        h1_heel=layered_model["h1_heel"],
        h1_toe=layered_model["h1_toe"],
        h2_heel=layered_model["h2_heel"],
        h2_toe=layered_model["h2_toe"],
        E1=p["E1"],
        nu1=p["nu1"],
        E_heel=p["E_heel"],
        E_toe=p["E_toe"],
        nu2=p["nu2"],
        EI_plate=p["EI_plate"],
        nx=layered_model["nx"],
        ny1=layered_model["ny1"],
        ny2=layered_model["ny2"],
        element_order=layered_model["element_order"],
        softplus_a=p["softplus_a"],
        softplus_kappa=p["softplus_kappa"],
        reciprocity_tol=p["reciprocity_tol"],
        measured=p,
    )
    out["measured_config_file"] = str(setup.source_path)
    prebuilt = matching_prebuilt_lookup(setup)
    if prebuilt is not None:
        out["prebuilt_lookup"] = str(prebuilt)
    return out


def _measured_config_export(draft_model: dict, setup) -> None:
    """Download the current measured-sole parameters as a JSON config."""
    from compliance_fem.gui_params import measured_setup_from_gui

    try:
        export = setup if setup is not None else measured_setup_from_gui(draft_model)
        payload = json.dumps(export.to_dict(), indent=2) + "\n"
    except (OSError, ValueError) as exc:
        st.caption(f"Config export unavailable: {exc}")
        return
    st.download_button(
        "Export measured-sole config (JSON)",
        data=payload,
        file_name="measured_sole_config.json",
        mime="application/json",
        key="gui_export_measured_config",
        help="Writes every measured-sole parameter shown in the sidebar (absolute CSV path).",
    )


@st.cache_resource(show_spinner=False)
def _load_measured_setup(path: str, mtime: float):
    from compliance_fem.measured_config_file import load_measured_sole_config

    del mtime
    return load_measured_sole_config(path)


GEOMETRY_MODEL_LAYERED = "layered_plate"
GEOMETRY_MODEL_MEASURED = "measured_sole"
GEOMETRY_MODEL_LABELS = {
    GEOMETRY_MODEL_LAYERED: "Layered plate (parametric thicknesses)",
    GEOMETRY_MODEL_MEASURED: "Measured carbon-plated sole (CSV)",
}
DEFAULT_GEOMETRY_CSV = "data/geometry/sole_geometry_normalized.csv"
DEFAULT_MEASURED_CONFIG = "configs/measured_sole_270mm.json"


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
Ground contact is one contiguous interval of bottom nodes `I_ij = {i, ..., j}`
(`0 <= i <= j < N_b`). Every interval is precomputed: **heel-attached**
(`i = 0`), **toe-attached** (`j = N_b - 1`), **full** (both), and **interior**
(neither end touches, e.g. a rocker or concave sole). The label is derived from
`(i, j)`; all intervals use the same equations.

**Auto** checks every interval with fixed-frame normal gaps on the free nodes
and normal reactions on all contact nodes. It searches near the previous
interval first, then widens, then searches globally. Continuity only breaks
ties among admissible intervals. **Specific** forces one `(i, j)`.

The basis modes are anchored at the interval midpoint `x_a = (x_i + x_j)/2`
on the reference bottom profile. This anchor is a numerical decomposition point
(open grey cross), not a contact edge; the contact edges are `x_i` and `x_j`.

The implementation supports one contiguous contact interval. If the normal
reactions become tensile inside an otherwise active interval, or if two
separated sole regions simultaneously contact the ground with a free region
between them, a multi-interval contact model is required.
"""


@st.cache_resource(show_spinner=False)
def _build_layered_lookup(params_json: str):
    """Rebuild FEM + lookup in a subprocess, then load the NPZ into this process.

    ``params_json`` is the canonical model-parameter dict (layered or measured).
    Isolates Gmsh / SuperLU from Streamlit to avoid macOS native segfaults.
    """
    import hashlib

    params = json.loads(params_json)
    prebuilt = params.get("prebuilt_lookup")
    if prebuilt and Path(prebuilt).is_file():
        return load_contact_lookup(prebuilt)
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
                "Lookup rebuild subprocess failed "
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


MATERIAL_FILL_COLORS = {
    "FFTurbo": "rgba(31, 119, 180, 0.28)",
    "FFLeap": "rgba(255, 127, 14, 0.28)",
}


def _draw_measured_regions(fig, shape, _line, show_reference_geometry: bool) -> None:
    """Measured sole: stored region loops, heel edge and foam interface (no thickness rebuild)."""
    if show_reference_geometry:
        for k, (name, (xr, yr)) in enumerate(shape.region_ref.items()):
            _line(xr, yr, name="reference geometry" if k == 0 else None, color="black",
                  width=1, dash="dash", legend=(k == 0))
    for name, (xd, yd) in shape.region_def.items():
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

    measured = bool(getattr(shape, "is_measured", False))
    if measured:
        _draw_measured_regions(fig, shape, _line, show_reference_geometry)

    # Reference (undeformed, unrotated) geometry.
    if show_reference_geometry and not measured:
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
    if not measured:
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
    if not measured:
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

    # Ground line, contact interval along the deformed (curved) bottom, free
    # bottom segments, contact edges, and the numerical interval anchor.
    x_pad = 0.05 * shape.L
    _line(
        [-x_pad, shape.L + x_pad],
        [shape.ground_y, shape.ground_y],
        name="ground",
        color="#4d4d4d",
        width=1,
    )
    i_c, j_c = shape.contact_start_index, shape.contact_end_index
    span_lo, span_hi = shape.contact_span
    _line(
        shape.contact_curve_x,
        shape.contact_curve_y,
        name=f"{shape.contact_type} contact {{{i_c}..{j_c}}}, x in [{span_lo:.3g}, {span_hi:.3g}]",
        color=_C3,
        width=6,
    )
    for k, (fx_c, fy_c) in enumerate(shape.free_curves):
        _line(
            fx_c,
            fy_c,
            name="free bottom" if k == 0 else None,
            color=_C2,
            width=3,
            dash="dash",
            legend=(k == 0),
        )
    if shape.contact_node_x is not None and shape.contact_node_x.size:
        fig.add_trace(
            go.Scatter(
                x=shape.contact_node_x, y=shape.contact_node_y, mode="markers",
                name="contact nodes", marker=dict(symbol="circle", size=5, color=_C3),
                hoverinfo="skip",
            )
        )
    if shape.free_node_x is not None and shape.free_node_x.size:
        fig.add_trace(
            go.Scatter(
                x=shape.free_node_x, y=shape.free_node_y, mode="markers",
                name="free nodes", marker=dict(symbol="circle-open", size=5, color=_C2),
                hoverinfo="skip",
            )
        )
    for label, pt, sym in (
        ("heel contact edge x_i", shape.heel_edge_xy, "triangle-right"),
        ("toe contact edge x_j", shape.toe_edge_xy, "triangle-left"),
    ):
        if pt is not None:
            fig.add_trace(
                go.Scatter(
                    x=[pt[0]], y=[pt[1]], mode="markers", name=label,
                    marker=dict(symbol=sym, size=12, color=_C3, line=dict(width=1, color="black")),
                    hoverinfo="skip",
                )
            )
    adj = [p for p in (shape.heel_adjacent_xy, shape.toe_adjacent_xy) if p is not None]
    if adj:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in adj], y=[p[1] for p in adj], mode="markers",
                name="adjacent free nodes", marker=dict(symbol="diamond-open", size=10, color=_C2),
                hoverinfo="skip",
            )
        )
    # Open grey cross: the interval midpoint is a numerical anchor, not a contact edge.
    fig.add_trace(
        go.Scatter(
            x=[shape.anchor_xy[0]],
            y=[shape.anchor_xy[1]],
            mode="markers",
            name=f"neutral anchor x_a=(x_i+x_j)/2 (numerical; x_rot={shape.x_anchor_rot:.4g})",
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


def _fmt_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.1f} ms"


def main() -> None:
    st.set_page_config(page_title="Contact topology lookup", layout="wide")
    st.title("Force-controlled single-interval contact evaluation (co-rotating top frame)")
    st.caption(
        "Geometry and materials rebuild the layered FEM compliance and contact lookup in-process. "
        "Fx, Fy in the sidebar are fixed-frame ground reaction forces: leftward/upward "
        "positive (solver top resultants use the opposite signs). φ is the absolute "
        "heel-to-toe chord angle and θ drives the single softplus shape mode through "
        "α = tan(θ). The contact set is one contiguous interval of bottom nodes; selection "
        "uses fixed-frame normal gaps on every free node and normal reactions on every "
        "contact node (never R_t, T_toe, or x_cm)."
    )
    with st.expander("What do φ and θ mean?", expanded=False):
        st.markdown(HELP_TEXT)
    with st.expander("How is the contact topology chosen?", expanded=False):
        st.markdown(CONTACT_MODE_HELP)

    with st.sidebar:
        with st.expander("Geometry", expanded=False):
            geometry_model = st.radio(
                "Geometry model",
                list(GEOMETRY_MODEL_LABELS),
                format_func=GEOMETRY_MODEL_LABELS.get,
                index=0,
                key="gui_geometry_model",
            )
            is_measured = geometry_model == GEOMETRY_MODEL_MEASURED
            measured_setup = None
            if is_measured:
                source = st.radio(
                    "Measured-sole parameters from",
                    ["Sidebar controls", "Config file (JSON)"],
                    key="gui_measured_source",
                    horizontal=True,
                    help=(
                        "Config file: shoe length, materials, plate EI, mesh, a and κ all come from "
                        "the JSON file (the sidebar values for those are ignored)."
                    ),
                )
                if source == "Config file (JSON)":
                    config_path = st.text_input(
                        "Config file", value=DEFAULT_MEASURED_CONFIG, key="gui_measured_config"
                    )
                    try:
                        cfg_file = Path(config_path)
                        measured_setup = _load_measured_setup(
                            str(cfg_file.resolve()), cfg_file.stat().st_mtime
                        )
                    except (OSError, ValueError) as exc:
                        st.error(f"Cannot use config: {exc}")
                    if measured_setup is not None:
                        s = measured_setup.sole
                        st.caption(
                            f"L = {s.shoe_length_mm:g} mm · {s.upper_foam_material} / {s.lower_foam_material} · "
                            f"EI = {s.EI_plate:g} N·m² · mesh {s.mesh_size * 1e3:g} mm · "
                            f"a = {measured_setup.a * 1e3:.1f} mm · κ = {measured_setup.kappa:g} 1/m"
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
                    default=270.0,
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
            else:
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
            if is_measured:
                st.caption("Layer thicknesses below apply to the layered model only.")
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
            measured_mesh: dict = {}
            if is_measured:
                measured_mesh["mesh_size_mm"] = dual_linear(
                    "Mesh size (mm)", key="gui_mesh_size_mm", min_value=1.0, max_value=8.0,
                    default=3.0, step=0.1, fmt="%.2f",
                )
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
                st.caption("Element counts below apply to the layered model only.")
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
            measured=(
                {
                    "geometry_csv": geometry_csv,
                    "shoe_length_mm": L_mm,
                    "upper_foam_material": upper_foam,
                    "lower_foam_material": lower_foam,
                    **measured_mesh,
                }
                if is_measured
                else None
            ),
        )
        if is_measured and measured_setup is not None:
            draft_model = _measured_draft_from_setup(measured_setup, draft_model)
            if "prebuilt_lookup" in draft_model:
                st.caption(f"Using the matching prebuilt lookup {draft_model['prebuilt_lookup']}.")
        if is_measured:
            _measured_config_export(draft_model, measured_setup)
        if _APPLIED_MODEL_KEY not in st.session_state:
            st.session_state[_APPLIED_MODEL_KEY] = draft_model

        # Query controls must be rendered BEFORE Apply. Calling st.rerun() above
        # these widgets deletes their backend state while the frontend still
        # shows the old slider positions — exactly the "plot uses defaults"
        # bug. Persist copies under non-widget keys as well.
        with st.expander("Contact interval", expanded=False):
            mode_label = st.radio(
                "Contact mode",
                list(CONTACT_MODE_LABELS.values()),
                index=0,
                key="gui_contact_mode",
                help="Auto searches every stored interval; the others restrict or force it.",
            )
            contact_mode = parse_contact_mode(mode_label)
            specific_i = st.number_input("Specific interval start i", min_value=0, value=0, step=1, key="gui_specific_i")
            specific_j = st.number_input("Specific interval end j", min_value=0, value=0, step=1, key="gui_specific_j")
            st.caption("Start/end indices are used only in Specific mode (validated: 0 ≤ i ≤ j < N_b).")

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
        "Building FEM compliance + contact lookup in a subprocess…"
        if not cache_hit
        else "Loading cached model…"
    ):
        lookup = _build_layered_lookup(json.dumps(applied_model, sort_keys=True))
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
            "(top_shape_alpha, contact_translation_x/y, contact_rotation_x/y); the "
            "curved-sole closure column has a fixed coefficient of 1."
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
                "contact_set_model": "single_contiguous_interval",
                "n_bottom_nodes": int(lookup.n_bottom_nodes),
                "n_records": lookup.n_records,
                "n_valid_records": int(np.count_nonzero(lookup.valid_mask)),
                "n_heel": int(lookup.rows_for(ContactType.HEEL).size),
                "n_interior": int(lookup.rows_for(ContactType.INTERIOR).size),
                "n_toe": int(lookup.rows_for(ContactType.TOE).size),
                "n_full": int(lookup.rows_for(ContactType.FULL).size),
                "lookup_build_time_s": float(getattr(lookup, "build_time_s", float("nan")) or float("nan")),
                "has_plate_response": bool(lookup.has_plate_response),
                "LOOKUP_SCHEMA_VERSION": LOOKUP_SCHEMA_VERSION,
            }
        )
        if getattr(lookup, "is_measured", False):
            gm = lookup.geometry_metadata
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
                    "n_top_nodes": int(lookup.n_top_nodes),
                    "n_bottom_nodes": int(lookup.n_bottom_nodes),
                }
            )

    tolerances = Tolerances(tau_g=tau_g, tau_R=tau_R, kf_cond_warn=kf_cond_warn)

    # Proximity to the previous selection is a final tie-breaker only; it exists
    # so the rendered topology does not flicker between numerically
    # indistinguishable candidates as the sliders move.
    previous_interval = st.session_state.get("previous_contact_interval")
    if previous_interval is not None and int(previous_interval[1]) >= int(lookup.n_bottom_nodes):
        previous_interval = None

    # Re-read query snapshot after the (possibly long) model rebuild so the plot
    # always matches the sidebar widgets, not stale defaults.
    phi_deg = float(st.session_state["gui_query_phi_deg"])
    theta_deg = float(st.session_state["gui_query_theta_deg"])
    Fx = float(st.session_state["gui_query_Fx"])
    Fy = float(st.session_state["gui_query_Fy"])

    specific = None
    if contact_mode is ContactMode.SPECIFIC:
        specific = (int(specific_i), int(specific_j))
        try:
            validate_interval(specific[0], specific[1], int(lookup.n_bottom_nodes))
        except ValueError as exc:
            st.error(f"Invalid specific interval: {exc}")
            st.stop()

    t0 = time.perf_counter()
    selection = evaluate_from_angles(
        lookup,
        Fx,
        Fy,
        phi_deg,
        theta_deg,
        tolerances=tolerances,
        mode=contact_mode,
        previous_interval=previous_interval,
        specific_interval=specific,
    )
    t_calc = time.perf_counter() - t0

    if selection.selected_row is None:
        st.error(selection.message)
        st.info(
            f"Calculation time: {_fmt_ms(t_calc)}; model rebuild: {_fmt_ms(t_rebuild)}"
        )
        st.stop()
    st.session_state["previous_contact_interval"] = selection.interval

    if selection.approximate:
        st.warning(selection.message)
    else:
        st.success(selection.message)

    if selection.kf_illconditioned:
        st.warning(
            f"Translation-to-force matrix K_F is ill-conditioned "
            f"(cond={selection.kf_cond:.3e}). No regularization was applied."
        )

    na = NOT_AVAILABLE
    span_lo, span_hi = selection.contact_span()

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
            y_anchor=float(selection.anchor_y),
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

    _num = format_optional

    with st.expander("Detailed outputs", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Topology label", selection.topology_label)
        cols[1].metric("Interval (i, j)", f"({selection.contact_start_index}, {selection.contact_end_index})")
        cols[2].metric("Contact x_i … x_j", f"[{span_lo:.4g}, {span_hi:.4g}]")
        cols[3].metric("Neutral anchor x_a (numerical)", f"{selection.anchor_x:.6g}")
        cols = st.columns(4)
        cols[0].metric("Heel-edge R_n^F", _num(selection.heel_edge_normal_reaction))
        cols[1].metric("Toe-edge R_n^F", _num(selection.toe_edge_normal_reaction))
        cols[2].metric("Heel adjacent free gap", _num(selection.heel_adjacent_free_gap))
        cols[3].metric("Toe adjacent free gap", _num(selection.toe_adjacent_free_gap))
        cols = st.columns(4)
        cols[0].metric(
            "Min free gap (node)",
            na if selection.min_free_gap_node < 0 else f"{selection.min_free_gap:.6g} (#{selection.min_free_gap_node})",
        )
        cols[1].metric("Max free penetration", _num(selection.max_free_penetration))
        cols[2].metric(
            "Min contact R_n^F (node)",
            f"{selection.min_contact_reaction:.6g} (#{selection.min_contact_reaction_node})",
        )
        cols[3].metric("Max contact tension", _num(selection.max_contact_tension))
        cols = st.columns(4)
        cols[0].metric("Force reconstruction error", _num(selection.force_reconstruction_error, ".3e"))
        cols[1].metric("Force-control cond(K_F)", _num(selection.force_control_condition_number, ".3e"))
        cols[2].metric("Admissible", str(selection.exactly_admissible))
        cols[3].metric("Candidate violation score", _num(selection.candidate_violation_score, ".3e"))
        cols = st.columns(4)
        cols[0].metric("Search method", selection.candidate_search_method)
        cols[1].metric("Rotated anchor x_a,rot", f"{selection.x_anchor_rot:.6g}")
        cols[2].metric("Anchor translation d_ax", f"{selection.d_ax:.6g}")
        cols[3].metric("Anchor translation d_ay", f"{selection.d_ay:.6g}")
        if selection.disconnected_contact_warning:
            st.warning(DISCONNECTED_CONTACT_WARNING)
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
        cols[1].metric("Complementarity score", _num(selection.complementarity_score, ".3e"))
        cols[2].metric("# admissible intervals", str(selection.n_admissible))
        cols[3].metric("Contact nodes", str(selection.contact_end_index - selection.contact_start_index + 1))
        st.caption(
            "Absent adjacent nodes (interval touching the heel or toe end) are shown as N/A. "
            "x_cm uses the visualization gauge r_a^F = (x_a, 0); only x_cm − x_a is gauge "
            "independent. Tangential reactions may take either sign because contact remains "
            "perfectly sticking. The anchor x_a is a decomposition point, not a contact edge."
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
        figs = []
        try:
            n_b = int(lookup.n_bottom_nodes)
            starts, ends = ev.contact_start_index, ev.contact_end_index
            for values, title in (
                (np.log10(np.maximum(ev.violation_score, 1e-30)), "log10 violation score V(i, j)"),
                (ev.admissible.astype(float), "admissible intervals (1 = admissible)"),
                (ev.min_free_gap, "minimum free gap g^F(i, j)"),
                (ev.min_contact_reaction, "minimum contact R_n^F(i, j)"),
            ):
                grid = np.full((n_b, n_b), np.nan)
                ok = ev.evaluated & np.isfinite(values)
                grid[ends[ok], starts[ok]] = values[ok]
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
                ax.plot([selection.contact_start_index], [selection.contact_end_index], "rx", ms=10)
                ax.set_xlabel("start index i")
                ax.set_ylabel("end index j")
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                figs.append(fig)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(lookup.x_bottom, selection.full_bottom_gap, "-o", markersize=3)
            for xe in (span_lo, span_hi):
                ax.axvline(xe, color="C3", ls="--")
            ax.axvline(selection.anchor_x, color="0.5", ls=":")
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
            for xe in (span_lo, span_hi):
                ax.axvline(xe, color="C3", ls="--")
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
