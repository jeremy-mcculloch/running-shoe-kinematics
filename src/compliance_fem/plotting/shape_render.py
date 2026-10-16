"""Fixed-frame deformed-geometry plot data (GUI-independent) for one continuous contact state.

Every rendered point is mapped with

    r^F = r_p^F + Q(varphi) [ (X - X_p) + s_d (d - d_p) ],

where ``X_p`` is the reaction-weighted contact centroid (``contact_model.display_pivot``),
``d_p`` its local displacement and ``r_p^F`` its true fixed-frame position. With
``s_d = 1`` this is the exact placement ``r^F = T + Q (X + d)``; the rigid
rotation is never exaggerated, only the elastic part. The pivot is continuous
in the solution, so exaggerated shapes do not jump at active-set changes.

Contact is shown per node: active nodes, inactive nodes and *transition* nodes
(active with a reaction below ``TRANSITION_FRACTION`` of the peak, or inactive
with a gap below ``TRANSITION_FRACTION * h``). Disconnected patches are drawn
separately; the range between them is never shaded. The patch edge estimates
are derived display quantities (``continuous_contact.contact_edge_estimate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from compliance_fem.contact.basis import shape_mode_phi1
from compliance_fem.contact.corotation import transform_to_fixed_frame

if TYPE_CHECKING:
    from compliance_fem.contact.contact_model import ContactModel
    from compliance_fem.fem.plate_response import PlateRuntimeState

TRANSITION_FRACTION = 0.02
NODE_ACTIVE = "active"
NODE_INACTIVE = "inactive"
NODE_TRANSITION = "transition"


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
    contact_label: str
    toe_length: float
    L: float
    H: float
    phi_deg: float
    theta_deg: float
    varphi_deg: float
    scale: float
    scale_mode: str
    caption: str
    # Per bottom node (heel -> toe): material x, deformed fixed-frame position,
    # gap, compressive normal reaction, active mask and display class.
    bottom_material_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bottom_node_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bottom_node_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bottom_gap: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bottom_normal_reaction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    node_class: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    # One deformed polyline per connected active patch, plus patch metadata.
    contact_curves: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    patches: tuple[Any, ...] = ()
    # Derived (display-only) patch edge points in the fixed frame.
    edge_estimate_points: tuple[tuple[float, float], ...] = ()
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
    plate_color_quantity: str = "none"
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
    # Exact nodal rendering of the two foam regions from the render influence.
    has_render_mesh: bool = False
    region_ref: dict = field(default_factory=dict)  # name -> (x, y) closed loop, reference
    region_def: dict = field(default_factory=dict)  # name -> (x, y) closed loop, fixed frame
    region_materials: dict = field(default_factory=dict)  # name -> material name
    heel_edge_ref: tuple | None = None
    heel_edge_def: tuple | None = None
    interface_ref: tuple | None = None
    interface_def: tuple | None = None
    toe_marker_xy: tuple[float, float] | None = None
    # Deformed foam mesh (fixed frame) and per-element strain energy density (J/m^3).
    foam_triangles: np.ndarray | None = None  # (3, n_el) vertex ids
    foam_x_def: np.ndarray | None = None  # (n_vertices,)
    foam_y_def: np.ndarray | None = None
    foam_energy_density: np.ndarray | None = None  # (n_el,)

    @property
    def n_patches(self) -> int:
        return len(self.patches)


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


def node_classes(active: np.ndarray, gap: np.ndarray, lam: np.ndarray, h: float) -> np.ndarray:
    """Display class per bottom node: active / inactive / transition (near g = lambda = 0)."""
    a = np.asarray(active, dtype=bool)
    lam = np.asarray(lam, dtype=float)
    peak = float(np.max(lam)) if lam.size and np.max(lam) > 0 else 0.0
    trans = (a & (lam <= TRANSITION_FRACTION * peak)) | (~a & (np.asarray(gap) <= TRANSITION_FRACTION * float(h)))
    out = np.where(a, NODE_ACTIVE, NODE_INACTIVE).astype(object)
    out[trans] = NODE_TRANSITION
    return out


def build_shape_plot_data(
    model: ContactModel,
    state,
    *,
    scale_mode: str = "true",
    manual_scale: float = 1.0,
    n_plot: int = 200,
    show_plate: bool = True,
    show_undeformed_plate: bool = True,
    show_plate_nodes: bool = False,
    show_plate_rotations: bool = False,
    plate_color_quantity: str = "none",
    samples_per_element: int = 16,
    foam_energy: bool = False,
) -> ShapePlotData:
    """Fixed-frame deformed geometry of one contact state (``contact_model.selection_proxy``).

    With ``foam_energy`` and a measured render section, the deformed foam mesh
    and each element's strain energy density are included (true displacement;
    the display scale only moves the drawn vertices).
    """
    from compliance_fem.contact.contact_model import plate_state_for_solution, render_vertex_displacements

    sol = state.solution
    op = model.operator
    x_a, y_a = float(state.anchor_x), float(state.anchor_y)
    d_ax, d_ay = float(state.d_ax), float(state.d_ay)
    anchor_fixed = tuple(state.anchor_fixed)
    varphi = float(state.varphi)
    L = float(op.L)
    x_b, y_b = op.x_bottom, op.y_bottom
    n_b = x_b.size
    u_bot_nodes = np.asarray(state.full_bottom_u, dtype=float)
    v_bot_nodes = np.asarray(state.full_bottom_v, dtype=float)

    x = np.linspace(0.0, L, n_plot)
    u_top = np.zeros_like(x)
    v_top = float(state.alpha) * shape_mode_phi1(x, L, op.toe_length, op.kappa)
    u_bot = np.interp(x, x_b, u_bot_nodes)
    v_bot = np.interp(x, x_b, v_bot_nodes)
    render_disp = render_vertex_displacements(model, sol) if model.render_section else None
    if render_disp is not None:
        k_out = _outline_index(model)
        scale, label = choose_display_scale(
            op.H, render_disp[1][k_out], np.zeros(1), scale_mode, manual_scale, u=render_disp[0][k_out]
        )
    else:
        scale, label = choose_display_scale(op.H, v_top, v_bot, scale_mode, manual_scale, u=u_bot)

    y_top_ref = np.interp(x, op.x_top, op.y_top)
    y_bottom_ref = np.interp(x, x_b, y_b)
    y_plate_ref = None
    if model.x_plate is not None and model.y_plate is not None:
        y_plate_ref = np.interp(x, model.x_plate, model.y_plate)

    def _to_fixed(xr, yr, u, v):
        return transform_to_fixed_frame(
            xr, yr, u, v, x_a, d_ax, d_ay, varphi, anchor=anchor_fixed, elastic_scale=scale, y_anchor=y_a
        )

    x_top_def, y_top_def = _to_fixed(x, y_top_ref, u_top, v_top)
    x_bottom_def, y_bottom_def = _to_fixed(x, y_bottom_ref, u_bot, v_bot)
    xn, yn = _to_fixed(x_b, y_b, u_bot_nodes, v_bot_nodes)
    patches = tuple(state.patches)
    curves = tuple((xn[p.start : p.end + 1].copy(), yn[p.start : p.end + 1].copy()) for p in patches)
    edges = []
    for p in patches:
        for xe in (p.edge_estimate_start_x, p.edge_estimate_end_x):
            if np.isfinite(xe):
                edges.append((float(np.interp(xe, x_b, xn)), float(np.interp(xe, x_b, yn))))

    x_plate_def = y_plate_def = None
    if y_plate_ref is not None:
        x_plate_def, y_plate_def = _to_fixed(x, y_plate_ref, np.zeros_like(x), np.zeros_like(x))
    chord_x = np.array([x_top_def[0], x_top_def[-1]], dtype=float)
    chord_y = np.array([y_top_def[0], y_top_def[-1]], dtype=float)
    force_origin = (float(0.5 * (chord_x[0] + chord_x[1])), float(0.5 * (chord_y[0] + chord_y[1])))
    force_vector = (float(state.Fx_star), float(state.Fy_star))

    plate_state = (
        plate_state_for_solution(
            model, sol, elastic_scale=scale, samples_per_element=samples_per_element,
            color_quantity=plate_color_quantity,
        )
        if show_plate
        else None
    )
    has_plate = plate_state is not None
    plate_is_schematic = (not has_plate) and (y_plate_ref is not None)
    plate_x_ref = plate_y_ref = None
    plate_x_hermite = plate_y_hermite = None
    plate_color_values = None
    plate_color_units = ""
    plate_summary: dict = {}
    plate_node_x = plate_node_y = None
    plate_quiver_x = plate_quiver_y = plate_quiver_u = plate_quiver_v = None
    if plate_state is not None:
        mesh = plate_state.mesh
        zeros_u = np.zeros(mesh.n_nodes, dtype=float)
        plate_x_ref, plate_y_ref = _to_fixed(mesh.reference_x, mesh.reference_y, zeros_u + d_ax, zeros_u + d_ay)
        plate_x_hermite = np.asarray(plate_state.sample_x_fixed, dtype=float)
        plate_y_hermite = np.asarray(plate_state.sample_y_fixed, dtype=float)
        x_plate_def, y_plate_def = plate_x_hermite, plate_y_hermite
        plate_color_values = plate_state.color_values
        plate_color_units = str(plate_state.color_units or "")
        plate_summary = dict(plate_state.summary)
        if show_plate_nodes:
            plate_node_x, plate_node_y = _to_fixed(
                mesh.reference_x, mesh.reference_y, plate_state.u_local, plate_state.v_local
            )
        if show_plate_rotations:
            tick = 0.04 * max(float(op.H), L * 0.05, 1e-6)
            plate_quiver_x, plate_quiver_y, plate_quiver_u, plate_quiver_v = _plate_rotation_quivers(
                plate_state, tick_length=tick
            )

    render: dict = {}
    if render_disp is not None:
        render = _render_fields(model, render_disp, _to_fixed, foam_energy=foam_energy)
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

    n_act = int(np.count_nonzero(state.active))
    span = ", ".join(f"{{{p.start}..{p.end}}}" for p in patches) or "none"
    caption = (
        f"{state.contact_label} contact: {n_act} active node{'s' if n_act != 1 else ''} in "
        f"{len(patches)} patch{'es' if len(patches) != 1 else ''} {span}. "
        f"phi = {np.rad2deg(state.phi):.3g} deg, varphi = {np.rad2deg(varphi):.3g} deg (rigid, never scaled). "
        f"Elastic display scale: {label}. Display pivot: reaction-weighted contact centroid."
    )
    if has_plate:
        caption += " Plate from the shared plate influence of the coupled FEM operator."
    elif plate_is_schematic:
        caption += " Plate drawn rigidly at its reference offset (schematic)."
    if render:
        caption += " Foam regions from exact nodal displacements (shared render influence)."

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
        contact_label=str(state.contact_label),
        toe_length=float(op.toe_length),
        L=L,
        H=float(op.H),
        phi_deg=float(np.rad2deg(state.phi)),
        theta_deg=float(np.rad2deg(state.theta)),
        varphi_deg=float(np.rad2deg(varphi)),
        scale=scale,
        scale_mode=label,
        caption=caption,
        bottom_material_x=x_b.copy(),
        bottom_node_x=xn,
        bottom_node_y=yn,
        bottom_gap=np.asarray(state.gap, dtype=float),
        bottom_normal_reaction=np.asarray(state.lam, dtype=float),
        active=np.asarray(state.active, dtype=bool),
        node_class=node_classes(state.active, state.gap, state.lam, op.h_mesh),
        contact_curves=curves,
        patches=patches,
        edge_estimate_points=tuple(edges),
        anchor_xy=(float(anchor_fixed[0]), float(anchor_fixed[1])),
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
        plate_color_quantity=str(plate_state.color_quantity) if plate_state is not None else "none",
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
        foam_triangles=render.get("foam_triangles"),
        foam_x_def=render.get("foam_x_def"),
        foam_y_def=render.get("foam_y_def"),
        foam_energy_density=render.get("foam_energy_density"),
    )


def _outline_index(lookup) -> np.ndarray:
    """Positions of the two region-loop nodes inside ``render_node_ids``."""
    from compliance_fem.geometry.render import render_index

    sec = lookup.render_section
    ids = np.unique(np.concatenate([np.asarray(sec["upper_foam_loop"]), np.asarray(sec["lower_foam_loop"])]))
    return render_index(lookup, ids.astype(int))


def _render_fields(lookup, disp: tuple[np.ndarray, np.ndarray], to_fixed, *, foam_energy: bool = False) -> dict:
    """Fixed-frame curves and region loops of the measured sole (nodal, no interpolation)."""
    from compliance_fem.geometry.render import element_strain_energy_density, render_index

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
    if foam_energy:
        out["foam_triangles"] = np.asarray(sec["mesh_triangles"], dtype=int)
        k_all = render_index(lookup, np.arange(pts.shape[1]))
        out["foam_x_def"], out["foam_y_def"] = to_fixed(pts[0], pts[1], u_r[k_all], v_r[k_all])
        out["foam_energy_density"] = element_strain_energy_density(lookup, u_r[k_all], v_r[k_all])
    return out
