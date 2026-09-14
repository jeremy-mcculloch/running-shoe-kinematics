"""Fixed-frame deformed-geometry plot data (GUI-independent).

Every rendered point is mapped with the complete transformation

    r^F = r_a^F + Q(varphi) [ (X - X_a) + s_d (d^T - d_a^T) ],

so the rigid co-rotation is never exaggerated; only the elastic part is scaled
by the display factor ``s_d``. The gauge ``r_a^F = (x_anchor, 0)`` pins the
record anchor to its reference ground coordinate: the selected contact edge for
heel or toe contact, and the numerical anchor ``L/2`` for full contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from compliance_fem.contact_lookup import ContactLookupResult
from compliance_fem.contact_topology import ContactType, contact_span
from compliance_fem.corotation import transform_to_fixed_frame
from compliance_fem.force_control import ContactSelection, top_displacement_vector

if TYPE_CHECKING:
    from compliance_fem.plate_response import PlateRuntimeState

GAUGE_DESCRIPTION = (
    "Visualization gauge: the record anchor is pinned to its reference ground "
    "coordinate r_a^F = (x_anchor, 0) — the physical contact edge for heel/toe, "
    "or the numerical L/2 anchor for full contact (not a lift-off edge). "
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
    l: float
    anchor_x: float
    x_contact_rot: float | None
    x_anchor_rot: float
    is_anchor_only: bool
    a: float
    L: float
    H: float
    phi_deg: float
    theta_deg: float
    varphi_deg: float
    scale: float
    scale_mode: str
    caption: str
    y_plate_ref: np.ndarray | None = None
    x_plate_def: np.ndarray | None = None
    y_plate_def: np.ndarray | None = None
    geometry_type: str = "rectangle"
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

    The rotating-frame displacement of a contact node is dominated by the rigid
    term ``(Q(varphi)^T - I)(X - X_a)``, which the transform scales along with
    everything else. Exaggerating it would visibly lift the contact interval off
    the ground, so ``rotation_drift`` (the largest magnitude of that rigid term)
    caps the auto scale: past the cap the plot falls back to true scale, which is
    the right answer anyway because at that point the rotation is already the
    visible effect.
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


def _contact_lever(ctype: ContactType, l: float, L: float, anchor: float) -> float:
    """Distance from the anchor to the farthest contact material point."""
    if ctype is ContactType.HEEL:
        # Contact is [0, l]; farthest from anchor l is at 0.
        return float(l) if np.isfinite(l) else float(anchor)
    if ctype is ContactType.TOE:
        # Contact is [l, L]; farthest from anchor l is at L.
        return float(L - l) if np.isfinite(l) else float(L - anchor)
    # Full contact on [0, L]; farthest of the ends from the numerical anchor.
    return float(max(anchor, L - anchor))


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
    # Tick along the deformed normal (theta = dw/ds, so normal is rot + pi/2 from tangent).
    nx = -np.sin(rot[idx])
    ny = np.cos(rot[idx])
    half = 0.5 * float(tick_length)
    return x[idx], y[idx], half * nx, half * ny


def build_shape_plot_data(
    lookup: ContactLookupResult,
    selection: ContactSelection,
    *,
    scale_mode: str = "auto",
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
    """Construct fixed-frame deformed geometry for the selected record."""
    anchor_raw = getattr(selection, "anchor_x", None)
    if anchor_raw is None:
        anchor_raw = getattr(selection, "l", None)
    d_ax_raw = getattr(selection, "d_ax", None)
    if d_ax_raw is None:
        d_ax_raw = getattr(selection, "d_lx", None)
    d_ay_raw = getattr(selection, "d_ay", None)
    if d_ay_raw is None:
        d_ay_raw = getattr(selection, "d_ly", None)

    if selection.selected_row is None or anchor_raw is None or d_ax_raw is None or d_ay_raw is None:
        raise ValueError("Cannot build shape plot without a selected record.")

    x_a = float(anchor_raw)
    d_ax = float(d_ax_raw)
    d_ay = float(d_ay_raw)
    varphi = float(selection.varphi)
    L = float(lookup.L)

    ctype_raw = getattr(selection, "contact_type", None)
    if ctype_raw is None:
        ctype = ContactType.TOE
    else:
        ctype = _as_contact_type(ctype_raw)
    is_anchor_only = ctype is ContactType.FULL

    l_raw = getattr(selection, "l", None)
    if l_raw is None or (isinstance(l_raw, float) and not np.isfinite(l_raw)):
        l_edge = float("nan") if is_anchor_only else float(x_a)
    else:
        l_edge = float(l_raw)

    span = contact_span(ctype, None if is_anchor_only else l_edge, L)
    if ctype is ContactType.FULL:
        free_spans: tuple[tuple[float, float], ...] = ()
    elif ctype is ContactType.HEEL:
        free_spans = ((l_edge, L),)
    else:
        free_spans = ((0.0, l_edge),)

    x = np.linspace(0.0, L, n_plot)
    u_top, v_top = top_displacement_vector(
        x,
        selection.alpha,
        lookup.L,
        lookup.softplus_a,
        lookup.softplus_kappa,
    )
    u_bot_nodes = selection.full_bottom_u
    v_bot_nodes = selection.full_bottom_v
    assert u_bot_nodes is not None and v_bot_nodes is not None
    u_bot = np.interp(x, lookup.x_bottom, np.nan_to_num(u_bot_nodes))
    v_bot = np.interp(x, lookup.x_bottom, np.nan_to_num(v_bot_nodes))

    # Lever arm from the anchor to the farthest contact material point; only
    # contact nodes carry the rigid (Q^T - I)(X - X_a) term that must not
    # visibly lift off the ground under elastic exaggeration.
    lever = _contact_lever(ctype, l_edge, L, x_a)
    scale, label = choose_display_scale(
        lookup.H,
        v_top,
        v_bot,
        scale_mode,
        manual_scale,
        u=np.concatenate([u_top, u_bot]),
        rotation_drift=float(np.hypot(np.cos(varphi) - 1.0, np.sin(varphi))) * lever,
    )

    y_top_ref = _profile(x, lookup.x_top, lookup.y_top, lookup.H)
    y_bottom_ref = _profile(x, lookup.x_bottom, lookup.y_bottom, 0.0)
    y_plate_ref = None
    if lookup.x_plate is not None and lookup.y_plate is not None:
        y_plate_ref = np.interp(x, lookup.x_plate, lookup.y_plate)

    def _to_fixed(y_ref, u, v):
        return transform_to_fixed_frame(
            x, y_ref, u, v, x_a, d_ax, d_ay, varphi, elastic_scale=scale
        )

    x_top_def, y_top_def = _to_fixed(y_top_ref, u_top, v_top)
    x_bottom_def, y_bottom_def = _to_fixed(y_bottom_ref, u_bot, v_bot)
    x_plate_def = y_plate_def = None
    if y_plate_ref is not None:
        # Fallback schematic: interface carried rigidly when no Hermite plate.
        x_plate_def, y_plate_def = _to_fixed(
            y_plate_ref, np.zeros_like(x), np.zeros_like(x)
        )

    # Heel-to-toe chord of the deformed top surface in the fixed frame.
    chord_x = np.array([x_top_def[0], x_top_def[-1]], dtype=float)
    chord_y = np.array([y_top_def[0], y_top_def[-1]], dtype=float)

    force_origin = (float(0.5 * (chord_x[0] + chord_x[1])), float(0.5 * (chord_y[0] + chord_y[1])))
    force_vector = (float(selection.Fx_star), float(selection.Fy_star))

    x_contact_rot_raw = getattr(selection, "x_contact_rot", None)
    if x_contact_rot_raw is None and not is_anchor_only:
        x_contact_rot_raw = getattr(selection, "x_l_rot", None)
    if is_anchor_only:
        x_contact_rot: float | None = None
    elif x_contact_rot_raw is None or (
        isinstance(x_contact_rot_raw, float) and not np.isfinite(x_contact_rot_raw)
    ):
        x_contact_rot = None
    else:
        x_contact_rot = float(x_contact_rot_raw)

    x_anchor_rot_raw = getattr(selection, "x_anchor_rot", None)
    if x_anchor_rot_raw is None:
        x_anchor_rot_raw = getattr(selection, "x_l_rot", None)
    if x_anchor_rot_raw is None:
        x_anchor_rot = float(x_a + d_ax)
    else:
        x_anchor_rot = float(x_anchor_rot_raw)

    # Optional Hermite plate superposition (schema-6 layered lookups).
    plate_state = plate
    if plate_state is None and show_plate and getattr(lookup, "has_plate_response", False):
        from compliance_fem.plate_response import plate_state_for_selection

        plate_state = plate_state_for_selection(
            lookup,
            selection,
            elastic_scale=scale,
            samples_per_element=samples_per_element,
            color_quantity=plate_color_quantity,
        )
    elif (
        plate_state is not None
        and str(plate_color_quantity or "none").lower() != str(plate_state.color_quantity).lower()
    ):
        # Rebuild if the caller passed a plate with a different color quantity.
        from compliance_fem.plate_response import plate_state_for_selection

        plate_state = plate_state_for_selection(
            lookup,
            selection,
            elastic_scale=scale,
            samples_per_element=samples_per_element,
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
        # Undeformed plate in the fixed frame: rigid pose only (u = v = 0).
        plate_x_ref, plate_y_ref = transform_to_fixed_frame(
            mesh.reference_x,
            mesh.reference_y,
            zeros_u,
            zeros_u,
            x_a,
            d_ax,
            d_ay,
            varphi,
            elastic_scale=scale,
        )
        # Dense Hermite centerline (already includes elastic_scale).
        plate_x_hermite = np.asarray(plate_state.sample_x_fixed, dtype=float)
        plate_y_hermite = np.asarray(plate_state.sample_y_fixed, dtype=float)
        # Replace schematic plate polylines with Hermite samples.
        x_plate_def = plate_x_hermite
        y_plate_def = plate_y_hermite
        plate_color_values = plate_state.color_values
        plate_color_units = str(plate_state.color_units or "")
        plate_summary = dict(plate_state.summary)
        if show_plate_nodes:
            plate_node_x, plate_node_y = transform_to_fixed_frame(
                mesh.reference_x,
                mesh.reference_y,
                plate_state.u_local,
                plate_state.v_local,
                x_a,
                d_ax,
                d_ay,
                varphi,
                elastic_scale=scale,
            )
        if show_plate_rotations:
            tick = 0.04 * max(float(lookup.H), float(L) * 0.05, 1e-6)
            plate_quiver_x, plate_quiver_y, plate_quiver_u, plate_quiver_v = (
                _plate_rotation_quivers(plate_state, tick_length=tick)
            )

    caption = (
        f"{ctype.value} contact on [{span[0]:.4g}, {span[1]:.4g}]. "
        f"phi = {np.rad2deg(selection.phi):.3g} deg, "
        f"varphi = {np.rad2deg(varphi):.3g} deg (rigid, never scaled). "
        f"Elastic display scale: {label}. {GAUGE_DESCRIPTION}"
    )
    if is_anchor_only:
        caption += (
            " The marked anchor x_a = L/2 is a numerical basis decomposition "
            "point, not a lift-off edge."
        )
    if has_plate:
        caption += " Internal plate shown from Hermite superposition of stored plate basis fields."
    elif plate_is_schematic:
        caption += (
            " Internal plate is carried rigidly with the frame at its undeformed "
            "reference offset (schematic; interface displacement is not stored)."
        )

    return ShapePlotData(
        x_dense=x,
        x_top_ref=x.copy(),
        x_bottom_ref=x.copy(),
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
        l=l_edge,
        anchor_x=x_a,
        x_contact_rot=x_contact_rot,
        x_anchor_rot=x_anchor_rot,
        is_anchor_only=is_anchor_only,
        a=float(lookup.softplus_a),
        L=L,
        H=float(lookup.H),
        phi_deg=float(np.rad2deg(selection.phi)),
        theta_deg=float(np.rad2deg(selection.theta)),
        varphi_deg=float(np.rad2deg(varphi)),
        scale=scale,
        scale_mode=label,
        caption=caption,
        y_plate_ref=y_plate_ref,
        x_plate_def=x_plate_def,
        y_plate_def=y_plate_def,
        geometry_type=str(lookup.geometry_type),
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
    )
