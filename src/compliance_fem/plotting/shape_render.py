"""Fixed-frame deformed-geometry plot data (GUI-independent).

Every rendered point is mapped with the complete transformation

    r^F = r_a^F + Q(varphi) [ (X - X_a) + s_d (d^T - d_a^T) ],

so the rigid co-rotation is never exaggerated; only the elastic part is scaled
by the display factor ``s_d``. ``X_a = (x_a, y_a)`` is the interval-midpoint
anchor on the reference bottom profile and the gauge ``r_a^F = (x_a, 0)`` puts
it on the ground. The anchor is a numerical decomposition point only: the
physical contact boundaries are the heel and toe contact edges ``x_i`` and
``x_j`` of the selected interval ``I_ij``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from compliance_fem.contact.lookup import ContactLookupResult
from compliance_fem.contact.topology import ContactType
from compliance_fem.contact.corotation import transform_to_fixed_frame
from compliance_fem.contact.force_control import ContactSelection, top_displacement_vector

if TYPE_CHECKING:
    from compliance_fem.fem.plate_response import PlateRuntimeState

GAUGE_DESCRIPTION = (
    "Visualization gauge: the interval-midpoint anchor X_a = ((x_i + x_j)/2, y_b) is "
    "pinned to the ground point r_a^F = (x_a, 0). The anchor is a numerical "
    "decomposition point, not a contact boundary; the contact edges are x_i and x_j. "
    "Absolute horizontal position is not determined by the model."
)


@dataclass(frozen=True)
class ShapePlotData:
    """Coordinates for reference and fixed-frame deformed rendering."""

    x_dense: np.ndarray
    x_top_ref: np.ndarray
    x_bottom_ref: np.ndarray
    y_top_ref: np.ndarray
    y_bottom_ref: np.ndarray
    x_top_def: np.ndarray
    x_bottom_def: np.ndarray
    y_top_def: np.ndarray
    y_bottom_def: np.ndarray
    chord_x: np.ndarray
    chord_y: np.ndarray
    force_origin: tuple[float, float]
    force_vector: tuple[float, float]
    ground_y: float
    contact_type: str
    contact_span: tuple[float, float]
    free_spans: tuple[tuple[float, float], ...]
    anchor_x: float
    x_anchor_rot: float
    toe_length: float
    L: float
    H: float
    phi_deg: float
    theta_deg: float
    varphi_deg: float
    scale: float
    scale_mode: str
    caption: str
    # Interval I_ij and its fixed-frame markers.
    contact_start_index: int = -1
    contact_end_index: int = -1
    anchor_y: float = 0.0
    contact_curve_x: np.ndarray | None = None
    contact_curve_y: np.ndarray | None = None
    free_curves: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    contact_node_x: np.ndarray | None = None
    contact_node_y: np.ndarray | None = None
    free_node_x: np.ndarray | None = None
    free_node_y: np.ndarray | None = None
    heel_edge_xy: tuple[float, float] | None = None
    toe_edge_xy: tuple[float, float] | None = None
    heel_adjacent_xy: tuple[float, float] | None = None
    toe_adjacent_xy: tuple[float, float] | None = None
    anchor_xy: tuple[float, float] = (0.0, 0.0)
    y_plate_ref: np.ndarray | None = None
    x_plate_def: np.ndarray | None = None
    y_plate_def: np.ndarray | None = None
    plate_is_schematic: bool = False
    has_plate: bool = False
    plate_x_ref: np.ndarray | None = None
    plate_y_ref: np.ndarray | None = None
    plate_x_def: np.ndarray | None = None
    plate_y_def: np.ndarray | None = None
    plate_color_values: np.ndarray | None = None
    plate_color_units: str = ""
    plate_summary: dict = field(default_factory=dict)
    plate_node_x: np.ndarray | None = None
    plate_node_y: np.ndarray | None = None
    plate_quiver_x: np.ndarray | None = None
    plate_quiver_y: np.ndarray | None = None
    plate_quiver_u: np.ndarray | None = None
    plate_quiver_v: np.ndarray | None = None
    show_undeformed_plate: bool = False
    show_plate_nodes: bool = False
    show_plate_rotations: bool = False
    # Exact nodal rendering of the two foam regions from the lookup's render influence.
    has_render_mesh: bool = False
    region_ref: dict = field(default_factory=dict)  # name -> (x, y) closed loop, reference
    region_def: dict = field(default_factory=dict)  # name -> (x, y) closed loop, fixed frame
    region_materials: dict = field(default_factory=dict)  # name -> material name
    heel_edge_ref: tuple | None = None
    heel_edge_def: tuple | None = None
    interface_ref: tuple | None = None
    interface_def: tuple | None = None
    toe_marker_xy: tuple[float, float] | None = None

    @property
    def contact_start_x(self) -> float:
        return float(self.contact_span[0])

    @property
    def contact_end_x(self) -> float:
        return float(self.contact_span[1])


MAX_CONTACT_DRIFT_FRACTION = 0.02


def choose_display_scale(
    H: float,
    w: np.ndarray,
    g: np.ndarray,
    mode: str,
    manual_scale: float,
    max_scale: float = 50.0,
    u: np.ndarray | None = None,
    rotation_drift: float = 0.0,
) -> tuple[float, str]:
    """Return (scale, mode_label) for elastic display exaggeration.

    The rotating-frame displacement of a contact node contains the rigid term
    ``(Q(varphi)^T - I)(X - X_a)``, which the transform scales along with the
    elastic part. ``rotation_drift`` (its largest magnitude) caps the auto scale
    so the contact interval is not visibly lifted off the ground.
    """
    if mode == "true":
        return 1.0, "true scale (s=1)"
    if mode == "manual":
        return float(manual_scale), f"manual elastic scale s={manual_scale:g}"
    amp = max(float(np.max(np.abs(w))), float(np.max(np.abs(g))), 1e-30)
    if u is not None and np.size(u):
        amp = max(amp, float(np.max(np.abs(u))))
    target = 0.25 * H
    s = min(max_scale, target / amp)
    drift = float(rotation_drift)
    if drift > 0.0:
        s = min(s, MAX_CONTACT_DRIFT_FRACTION * H / drift)
    s = max(s, 1.0)
    return float(s), f"auto-exaggerated elastic scale s={s:.3g} (display only; not true scale)"


def _profile(x: np.ndarray, x_nodes: np.ndarray | None, y_nodes: np.ndarray | None, fallback: float) -> np.ndarray:
    if x_nodes is None or y_nodes is None:
        return np.full_like(x, fallback)
    return np.interp(x, x_nodes, y_nodes)


def _as_contact_type(raw) -> ContactType:
    if isinstance(raw, ContactType):
        return raw
    return ContactType(str(raw))


def _plate_rotation_quivers(
    plate: PlateRuntimeState,
    *,
    tick_length: float,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small ticks along the deformed centerline indicating local plate rotation."""
    x = np.asarray(plate.sample_x_fixed, dtype=float)
    y = np.asarray(plate.sample_y_fixed, dtype=float)
    rot = np.asarray(plate.sample_rotation_fixed, dtype=float)
    if x.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    idx = np.arange(0, x.size, max(int(stride), 1), dtype=int)
    nx = -np.sin(rot[idx])
    ny = np.cos(rot[idx])
    half = 0.5 * float(tick_length)
    return x[idx], y[idx], half * nx, half * ny


def _selection_interval(selection, lookup: ContactLookupResult) -> tuple[int, int]:
    i = getattr(selection, "contact_start_index", None)
    j = getattr(selection, "contact_end_index", None)
    if i is None or j is None:
        row = int(selection.selected_row)
        i = int(lookup.contact_start_index[row])
        j = int(lookup.contact_end_index[row])
    return int(i), int(j)


def build_shape_plot_data(
    lookup: ContactLookupResult,
    selection: ContactSelection,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    n_plot: int = 200,
    plate: PlateRuntimeState | None = None,
    show_plate: bool = True,
    show_undeformed_plate: bool = True,
    show_plate_nodes: bool = False,
    show_plate_rotations: bool = False,
    plate_color_quantity: str = "none",
    samples_per_element: int = 16,
) -> ShapePlotData:
    """Construct fixed-frame deformed geometry for the selected interval record."""
    anchor_raw = getattr(selection, "anchor_x", None)
    d_ax_raw = getattr(selection, "d_ax", None)
    d_ay_raw = getattr(selection, "d_ay", None)
    if selection.selected_row is None or anchor_raw is None or d_ax_raw is None or d_ay_raw is None:
        raise ValueError("Cannot build shape plot without a selected record.")

    x_a = float(anchor_raw)
    y_a = float(getattr(selection, "anchor_y", 0.0) or 0.0)
    d_ax = float(d_ax_raw)
    d_ay = float(d_ay_raw)
    varphi = float(selection.varphi)
    L = float(lookup.L)
    i_c, j_c = _selection_interval(selection, lookup)
    ctype_raw = getattr(selection, "contact_type", None)
    ctype = _as_contact_type(ctype_raw) if ctype_raw is not None else ContactType.INTERIOR

    x_b = np.asarray(lookup.x_bottom, dtype=float)
    n_b = x_b.size
    y_b = np.asarray(lookup.y_bottom, dtype=float) if lookup.y_bottom is not None else np.zeros(n_b)
    x_i, x_j = float(x_b[i_c]), float(x_b[j_c])
    span = (x_i, x_j)
    free_spans = tuple(s for s in ((float(x_b[0]), x_i), (x_j, float(x_b[-1]))) if s[1] > s[0])

    x = np.linspace(0.0, L, n_plot)
    u_top, v_top = top_displacement_vector(x, selection.alpha, lookup.L, lookup.softplus_toe_length, lookup.softplus_kappa)
    u_bot_nodes = np.nan_to_num(np.asarray(selection.full_bottom_u, dtype=float))
    v_bot_nodes = np.nan_to_num(np.asarray(selection.full_bottom_v, dtype=float))
    u_bot = np.interp(x, x_b, u_bot_nodes)
    v_bot = np.interp(x, x_b, v_bot_nodes)

    lever = max(abs(x_i - x_a), abs(x_j - x_a))
    scale, label = choose_display_scale(
        lookup.H, v_top, v_bot, scale_mode, manual_scale,
        u=np.concatenate([u_top, u_bot]),
        rotation_drift=float(np.hypot(np.cos(varphi) - 1.0, np.sin(varphi))) * lever,
    )
    render_disp = None
    if getattr(lookup, "render_section", None):
        render_disp = _render_displacements(lookup, selection)
        scale, label = choose_display_scale(
            lookup.H, render_disp[1], np.zeros(1), scale_mode, manual_scale, u=render_disp[0],
            rotation_drift=float(np.hypot(np.cos(varphi) - 1.0, np.sin(varphi))) * lever,
        )

    y_top_ref = _profile(x, lookup.x_top, lookup.y_top, lookup.H)
    y_bottom_ref = _profile(x, x_b, y_b, 0.0)
    y_plate_ref = None
    if lookup.x_plate is not None and lookup.y_plate is not None:
        y_plate_ref = np.interp(x, lookup.x_plate, lookup.y_plate)

    def _to_fixed(xr, yr, u, v):
        return transform_to_fixed_frame(xr, yr, u, v, x_a, d_ax, d_ay, varphi, elastic_scale=scale, y_anchor=y_a)

    x_top_def, y_top_def = _to_fixed(x, y_top_ref, u_top, v_top)
    x_bottom_def, y_bottom_def = _to_fixed(x, y_bottom_ref, u_bot, v_bot)

    # Nodal bottom points (exact nodal displacements, no interpolation).
    xn, yn = _to_fixed(x_b, y_b, u_bot_nodes, v_bot_nodes)
    contact_ids = np.arange(i_c, j_c + 1)
    free_ids = np.setdiff1d(np.arange(n_b), contact_ids)

    def _curve(lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Deformed bottom between nodes lo..hi (nodal polyline; bottom edges are linear in x)."""
        ids = np.arange(lo, hi + 1)
        return xn[ids].copy(), yn[ids].copy()

    contact_curve = _curve(i_c, j_c)
    free_curves = []
    if i_c > 0:
        free_curves.append(_curve(0, i_c))
    if j_c < n_b - 1:
        free_curves.append(_curve(j_c, n_b - 1))

    def _pt(k: int) -> tuple[float, float] | None:
        if 0 <= k < n_b:
            return float(xn[k]), float(yn[k])
        return None

    x_plate_def = y_plate_def = None
    if y_plate_ref is not None:
        x_plate_def, y_plate_def = _to_fixed(x, y_plate_ref, np.zeros_like(x), np.zeros_like(x))

    chord_x = np.array([x_top_def[0], x_top_def[-1]], dtype=float)
    chord_y = np.array([y_top_def[0], y_top_def[-1]], dtype=float)
    force_origin = (float(0.5 * (chord_x[0] + chord_x[1])), float(0.5 * (chord_y[0] + chord_y[1])))
    force_vector = (float(selection.Fx_star), float(selection.Fy_star))
    x_anchor_rot = float(x_a + d_ax)

    plate_state = plate
    if plate_state is None and show_plate and getattr(lookup, "has_plate_response", False):
        from compliance_fem.fem.plate_response import plate_state_for_selection

        plate_state = plate_state_for_selection(
            lookup, selection, elastic_scale=scale, samples_per_element=samples_per_element,
            color_quantity=plate_color_quantity,
        )
    elif (
        plate_state is not None
        and str(plate_color_quantity or "none").lower() != str(plate_state.color_quantity).lower()
    ):
        from compliance_fem.fem.plate_response import plate_state_for_selection

        plate_state = plate_state_for_selection(
            lookup, selection, elastic_scale=scale, samples_per_element=samples_per_element,
            color_quantity=plate_color_quantity,
        )

    has_plate = bool(show_plate and plate_state is not None)
    plate_is_schematic = (not has_plate) and (y_plate_ref is not None)
    plate_x_ref = plate_y_ref = None
    plate_x_hermite = plate_y_hermite = None
    plate_color_values = None
    plate_color_units = ""
    plate_summary: dict = {}
    plate_node_x = plate_node_y = None
    plate_quiver_x = plate_quiver_y = plate_quiver_u = plate_quiver_v = None

    if has_plate and plate_state is not None:
        mesh = plate_state.mesh
        zeros_u = np.zeros(mesh.n_nodes, dtype=float)
        plate_x_ref, plate_y_ref = _to_fixed(mesh.reference_x, mesh.reference_y, zeros_u, zeros_u)
        plate_x_hermite = np.asarray(plate_state.sample_x_fixed, dtype=float)
        plate_y_hermite = np.asarray(plate_state.sample_y_fixed, dtype=float)
        x_plate_def = plate_x_hermite
        y_plate_def = plate_y_hermite
        plate_color_values = plate_state.color_values
        plate_color_units = str(plate_state.color_units or "")
        plate_summary = dict(plate_state.summary)
        if show_plate_nodes:
            plate_node_x, plate_node_y = _to_fixed(
                mesh.reference_x, mesh.reference_y, plate_state.u_local, plate_state.v_local
            )
        if show_plate_rotations:
            tick = 0.04 * max(float(lookup.H), float(L) * 0.05, 1e-6)
            plate_quiver_x, plate_quiver_y, plate_quiver_u, plate_quiver_v = _plate_rotation_quivers(
                plate_state, tick_length=tick
            )

    render: dict = {}
    if render_disp is not None:
        render = _render_fields(lookup, render_disp, _to_fixed)
        x, y_top_ref, y_bottom_ref = render["x_top_ref"], render["y_top_ref"], render["y_bottom_ref"]
        x_top_def, y_top_def = render["x_top_def"], render["y_top_def"]
        x_bottom_def, y_bottom_def = render["x_bottom_def"], render["y_bottom_def"]
        y_plate_ref = None
        if not has_plate:
            x_plate_def = y_plate_def = None
        plate_is_schematic = False
        # phi_ref uses the top selector, whose last node precedes the bottom-owned toe vertex.
        chord_x = np.array([x_top_def[0], x_top_def[-2]], dtype=float)
        chord_y = np.array([y_top_def[0], y_top_def[-2]], dtype=float)
        force_origin = (float(0.5 * (chord_x[0] + chord_x[1])), float(0.5 * (chord_y[0] + chord_y[1])))

    n_contact = j_c - i_c + 1
    caption = (
        f"{ctype.value} interval I = {{{i_c}..{j_c}}} ({n_contact} contact node"
        f"{'s' if n_contact != 1 else ''}) on reference x in [{x_i:.4g}, {x_j:.4g}]. "
        f"phi = {np.rad2deg(selection.phi):.3g} deg, "
        f"varphi = {np.rad2deg(varphi):.3g} deg (rigid, never scaled). "
        f"Elastic display scale: {label}. {GAUGE_DESCRIPTION}"
    )
    if has_plate:
        caption += " Internal plate shown from Hermite superposition of stored plate basis fields."
    elif plate_is_schematic:
        caption += (
            " Internal plate is carried rigidly with the frame at its undeformed "
            "reference offset (schematic; interface displacement is not stored)."
        )

    if render:
        caption += (
            " Both foam regions are drawn from exact nodal displacements "
            "(stored render influence of the free-body FEM operator)."
        )

    return ShapePlotData(
        x_dense=x,
        x_top_ref=x.copy(),
        x_bottom_ref=render["x_bottom_ref"] if render else x.copy(),
        y_top_ref=y_top_ref,
        y_bottom_ref=y_bottom_ref,
        x_top_def=x_top_def,
        x_bottom_def=x_bottom_def,
        y_top_def=y_top_def,
        y_bottom_def=y_bottom_def,
        chord_x=chord_x,
        chord_y=chord_y,
        force_origin=force_origin,
        force_vector=force_vector,
        ground_y=0.0,
        contact_type=ctype.value,
        contact_span=span,
        free_spans=free_spans,
        anchor_x=x_a,
        x_anchor_rot=x_anchor_rot,
        toe_length=float(lookup.softplus_toe_length),
        L=L,
        H=float(lookup.H),
        phi_deg=float(np.rad2deg(selection.phi)),
        theta_deg=float(np.rad2deg(selection.theta)),
        varphi_deg=float(np.rad2deg(varphi)),
        scale=scale,
        scale_mode=label,
        caption=caption,
        contact_start_index=i_c,
        contact_end_index=j_c,
        anchor_y=y_a,
        contact_curve_x=contact_curve[0],
        contact_curve_y=contact_curve[1],
        free_curves=tuple(free_curves),
        contact_node_x=xn[contact_ids],
        contact_node_y=yn[contact_ids],
        free_node_x=xn[free_ids],
        free_node_y=yn[free_ids],
        heel_edge_xy=_pt(i_c),
        toe_edge_xy=_pt(j_c),
        heel_adjacent_xy=_pt(i_c - 1) if i_c > 0 else None,
        toe_adjacent_xy=_pt(j_c + 1) if j_c < n_b - 1 else None,
        anchor_xy=(x_a, 0.0),
        y_plate_ref=y_plate_ref,
        x_plate_def=x_plate_def,
        y_plate_def=y_plate_def,
        plate_is_schematic=plate_is_schematic,
        has_plate=has_plate,
        plate_x_ref=plate_x_ref,
        plate_y_ref=plate_y_ref,
        plate_x_def=plate_x_hermite,
        plate_y_def=plate_y_hermite,
        plate_color_values=plate_color_values,
        plate_color_units=plate_color_units,
        plate_summary=plate_summary,
        plate_node_x=plate_node_x,
        plate_node_y=plate_node_y,
        plate_quiver_x=plate_quiver_x,
        plate_quiver_y=plate_quiver_y,
        plate_quiver_u=plate_quiver_u,
        plate_quiver_v=plate_quiver_v,
        show_undeformed_plate=bool(show_undeformed_plate and has_plate),
        show_plate_nodes=bool(show_plate_nodes and has_plate),
        show_plate_rotations=bool(show_plate_rotations and has_plate),
        has_render_mesh=bool(render),
        region_ref=render.get("region_ref", {}),
        region_def=render.get("region_def", {}),
        region_materials=render.get("region_materials", {}),
        heel_edge_ref=render.get("heel_edge_ref"),
        heel_edge_def=render.get("heel_edge_def"),
        interface_ref=render.get("interface_ref"),
        interface_def=render.get("interface_def"),
        toe_marker_xy=render.get("toe_marker_xy"),
    )


def _render_displacements(lookup, selection) -> tuple[np.ndarray, np.ndarray]:
    """Local (u, v) at the measured render nodes for the selected record and gamma."""
    from compliance_fem.contact.corotation import affine_coefficients
    from compliance_fem.geometry.render import render_node_displacements

    varphi = float(selection.varphi)
    gamma = np.array(
        [float(selection.alpha), float(selection.d_ax), float(selection.d_ay),
         np.cos(varphi) - 1.0, -np.sin(varphi)],
        dtype=float,
    )
    return render_node_displacements(lookup, int(selection.selected_row), affine_coefficients(gamma))


def _render_fields(lookup, disp: tuple[np.ndarray, np.ndarray], to_fixed) -> dict:
    """Fixed-frame curves and region loops of the measured sole (nodal, no interpolation)."""
    from compliance_fem.geometry.render import render_index

    sec = lookup.render_section
    u_r, v_r = disp
    pts = np.asarray(sec["mesh_points"], dtype=float)

    def _fixed(name: str):
        ids = np.asarray(sec[name], dtype=int)
        k = render_index(lookup, ids)
        x0, y0 = pts[0, ids], pts[1, ids]
        return (x0, y0), to_fixed(x0, y0, u_r[k], v_r[k])

    out: dict = {}
    (xt, yt), (xtd, ytd) = _fixed("top_curve_node_ids")
    (xb, yb), (xbd, ybd) = _fixed("bottom_curve_node_ids")
    out.update(
        x_top_ref=xt, y_top_ref=yt, x_top_def=xtd, y_top_def=ytd,
        x_bottom_ref=xb, y_bottom_ref=yb, x_bottom_def=xbd, y_bottom_def=ybd,
    )
    out["heel_edge_ref"], out["heel_edge_def"] = _fixed("heel_edge_node_ids")
    out["interface_ref"], out["interface_def"] = _fixed("interface_node_ids")
    region_ref, region_def = {}, {}
    for name, key in (("upper_foam", "upper_foam_loop"), ("lower_foam", "lower_foam_loop")):
        ref, dfm = _fixed(key)
        region_ref[name] = (np.append(ref[0], ref[0][0]), np.append(ref[1], ref[1][0]))
        region_def[name] = (np.append(dfm[0], dfm[0][0]), np.append(dfm[1], dfm[1][0]))
    out["region_ref"] = region_ref
    out["region_def"] = region_def
    meta = getattr(lookup, "geometry_metadata", {}) or {}
    out["region_materials"] = {
        "upper_foam": str(meta.get("upper_foam_material", "")),
        "lower_foam": str(meta.get("lower_foam_material", "")),
    }
    out["toe_marker_xy"] = (float(xtd[-1]), float(ytd[-1]))
    return out
