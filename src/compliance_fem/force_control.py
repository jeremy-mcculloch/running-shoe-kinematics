"""Force-controlled contact evaluation in a top-attached co-rotating frame.

Runtime inputs are the fixed-frame top resultants ``Fx``, ``Fy``, the absolute
heel-to-toe chord angle ``phi``, and the toe-bending shape angle ``theta``.

Every stored contact record (heel, toe, or the single full-contact record) is
reconstructed the same way:

1. Rotate the requested fixed-frame force into the local frame,
   ``F^{T,*} = Q(varphi)^T F^{F,*}``.
2. Solve the 2x2 system ``K_F [d_ax, d_ay]^T = F^{T,*} - F_known^T`` for the
   record's anchor translation: the contact-edge translation ``d_l`` for heel or
   toe contact, the numerical anchor translation ``d_a`` for full contact.
3. Superpose the five basis responses with
   ``gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)]``.
4. Transform free-bottom gaps, contact reactions, and the two full-contact
   corner reactions into the fixed frame.
5. Apply topology-aware admissibility to the fixed-frame normal gap and normal
   reaction only.

Topology selection (``ContactMode.AUTO``) follows the corner-reaction routing
rule: full contact is chosen outright when both fixed-frame corner normal
reactions are compressive; a tensile corner routes the search to the opposite
partial family (or both families when both corners are tensile / full is
ill-conditioned). Forced ``HEEL`` / ``TOE`` / ``FULL`` modes restrict the pool.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from compliance_fem.contact_basis import (
    BASIS_MODE_NAMES,
    N_BASIS_MODES,
    shape_mode_phi1,
)
from compliance_fem.contact_lookup import (
    CORNER_HEEL,
    CORNER_TOE,
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    ContactLookupResult,
    contact_type_code,
    contact_type_from_code,
    kf_from_scalars,
)
from compliance_fem.contact_topology import ContactMode, ContactType
from compliance_fem.corotation import (
    basis_coefficients,
    contact_displacement,
    contract_basis,
    corotation_angle,
    fixed_frame_normal_component,
    fixed_frame_tangential_component,
    known_coefficients,
    rotate_force_to_local,
    rotate_vector_to_fixed,
    rotation_coefficients,
    shape_amplitude,
    transform_to_fixed_frame,
)

if TYPE_CHECKING:
    from compliance_fem.plate_response import PlateRuntimeState

# Re-export for callers that import ContactMode from this module.
__all__ = (
    "ContactMode",
    "ContactType",
    "Tolerances",
    "CandidateEvaluation",
    "ContactSelection",
    "INSUFFICIENT_TOPOLOGY_MESSAGE",
    "angles_to_coefficients",
    "evaluate_candidates",
    "select_contact_candidate",
    "evaluate_from_angles",
    "top_displacement_vector",
    "evaluation_to_csv_rows",
    "export_evaluation_csv",
    "export_selected_result",
)

INSUFFICIENT_TOPOLOGY_MESSAGE = (
    "The supported contact topology family is insufficient: neither the routed "
    "partial-contact family nor full contact contains an admissible state."
)


@dataclass(frozen=True)
class Tolerances:
    """Thresholds for force conditioning and unilateral admissibility.

    ``tau_g`` and ``tau_R`` are the admissibility tolerances on the fixed-frame
    normal gap and normal reaction. They also serve as the denominators of the
    dimensionless violation score ``J`` when positive; when left at zero (the
    default, i.e. exact unilateral inequalities) ``J`` falls back to the
    data-driven scales ``g_scale`` and ``R_scale`` so it stays finite and
    comparable across candidates.
    """

    tau_g: float = 0.0
    tau_R: float = 0.0
    F_abs_tol: float = 1e-8
    F_rel_tol: float = 1e-8
    fy_zero_tol: float = 1e-14
    g_floor: float = 1e-30
    R_floor: float = 1e-30
    kf_cond_warn: float = 1e8


@dataclass
class CandidateEvaluation:
    """Force-controlled reconstruction for every stored contact record."""

    contact_type_codes: np.ndarray
    l: np.ndarray
    anchor_x: np.ndarray
    candidate_indices: np.ndarray
    has_free_edge: np.ndarray
    phi: float
    theta: float
    phi_ref: float
    varphi: float
    alpha: float
    r_x: float
    r_y: float
    Fx_star: float
    Fy_star: float
    Fx_star_local: float
    Fy_star_local: float
    d_ax: np.ndarray
    d_ay: np.ndarray
    x_anchor_rot: np.ndarray
    y_anchor_rot: np.ndarray
    x_contact_rot: np.ndarray
    y_contact_rot: np.ndarray
    Fx_local: np.ndarray
    Fy_local: np.ndarray
    Fx: np.ndarray
    Fy: np.ndarray
    M: np.ndarray
    x_cm: np.ndarray
    x_cm_rel: np.ndarray
    T_toe: np.ndarray
    edge_free_gap: np.ndarray
    edge_contact_reaction_normal: np.ndarray
    edge_contact_reaction_tangential: np.ndarray
    corner_rx_heel: np.ndarray
    corner_ry_heel: np.ndarray
    corner_rx_toe: np.ndarray
    corner_ry_toe: np.ndarray
    corner_reaction_local: np.ndarray
    corner_reaction_fixed: np.ndarray
    corner_reaction_normal: np.ndarray
    Rn_heel_corner: np.ndarray
    Rn_toe_corner: np.ndarray
    full_bottom_gap: np.ndarray
    full_bottom_u: np.ndarray
    full_bottom_v: np.ndarray
    full_bottom_reaction_normal: np.ndarray
    full_bottom_reaction_tangential: np.ndarray
    full_bottom_reaction_x: np.ndarray
    full_bottom_reaction_y: np.ndarray
    top_u: np.ndarray
    top_v: np.ndarray
    top_force_x: np.ndarray
    top_force_y: np.ndarray
    min_free_gap: np.ndarray
    min_contact_reaction: np.ndarray
    gap_violation: np.ndarray
    reaction_violation: np.ndarray
    force_residual: np.ndarray
    force_residual_norm: np.ndarray
    violation_score: np.ndarray
    edge_score: np.ndarray
    admissible: np.ndarray
    well_conditioned: np.ndarray
    kf_cond: np.ndarray
    kf_det: np.ndarray
    kf_svals: np.ndarray
    kf_illconditioned: np.ndarray
    full_contact_row: int
    full_contact_valid: np.ndarray
    strict_full_contact_valid: np.ndarray
    routed_families: tuple[ContactType, ...]
    routing_reason: str
    g_scale: float
    R_scale: float
    F_reference: float
    F_abs_tol: float
    F_rel_tol: float
    J_gap_scale: float
    J_reaction_scale: float
    J_force_scale: float
    diagnostics: dict = field(default_factory=dict)

    # --- compatibility aliases -------------------------------------------------

    @property
    def d_lx(self) -> np.ndarray:
        """Alias of ``d_ax`` (anchor / contact-edge translation)."""
        return self.d_ax

    @property
    def d_ly(self) -> np.ndarray:
        """Alias of ``d_ay``."""
        return self.d_ay

    @property
    def x_l_rot(self) -> np.ndarray:
        """Alias of ``x_contact_rot`` (NaN on the full-contact row)."""
        return self.x_contact_rot

    @property
    def y_l_rot(self) -> np.ndarray:
        """Alias of ``y_contact_rot``."""
        return self.y_contact_rot

    @property
    def full_contact_valid_strict(self) -> bool:
        """Scalar diagnostic: strict validity of the full-contact row."""
        return bool(self.strict_full_contact_valid[int(self.full_contact_row)])

    @property
    def full_contact_valid_flag(self) -> bool:
        """Scalar corner-validity flag of the full-contact row."""
        return bool(self.full_contact_valid[int(self.full_contact_row)])

    @property
    def Rn_heel_corner_full(self) -> float:
        """Fixed-frame heel-corner normal reaction of the full-contact row."""
        return float(self.Rn_heel_corner[int(self.full_contact_row)])

    @property
    def Rn_toe_corner_full(self) -> float:
        """Fixed-frame toe-corner normal reaction of the full-contact row."""
        return float(self.Rn_toe_corner[int(self.full_contact_row)])

    def contact_type(self, row: int) -> ContactType:
        """Explicit topology of one record."""
        return contact_type_from_code(int(self.contact_type_codes[int(row)]))

    def rows_for(self, contact_type: ContactType | str) -> np.ndarray:
        """Row indices belonging to one topology family."""
        code = contact_type_code(contact_type)
        return np.flatnonzero(np.asarray(self.contact_type_codes, dtype=int) == code)


@dataclass
class ContactSelection:
    """Selected contact record plus the full record evaluation."""

    evaluation: CandidateEvaluation
    contact_mode: ContactMode
    selected_row: int | None
    selected_index: int | None
    contact_type: ContactType | None
    l: float | None
    anchor_x: float | None
    has_free_edge: bool
    phi: float
    theta: float
    varphi: float
    alpha: float
    d_ax: float | None
    d_ay: float | None
    x_anchor_rot: float | None
    y_anchor_rot: float | None
    x_contact_rot: float | None
    y_contact_rot: float | None
    Fx_star: float
    Fy_star: float
    Fx: float | None
    Fy: float | None
    Fx_local: float | None
    Fy_local: float | None
    M: float | None
    x_cm: float | None
    x_cm_rel: float | None
    T_toe: float | None
    edge_free_gap: float | None
    edge_contact_reaction_normal: float | None
    edge_contact_reaction_tangential: float | None
    min_free_gap: float | None
    min_contact_reaction: float | None
    force_residual: float | None
    Rn_heel_corner: float | None
    Rn_toe_corner: float | None
    full_contact_valid: bool | None
    full_contact_valid_strict: bool | None
    routed_families: tuple[ContactType, ...]
    routing_reason: str
    full_bottom_gap: np.ndarray | None
    full_bottom_u: np.ndarray | None
    full_bottom_v: np.ndarray | None
    full_bottom_reaction_normal: np.ndarray | None
    full_bottom_reaction_tangential: np.ndarray | None
    top_u: np.ndarray | None
    top_v: np.ndarray | None
    exactly_admissible: bool
    approximate: bool
    n_admissible: int
    admissible_rows: np.ndarray
    violation_score: float | None
    edge_score: float | None
    kf_cond: float | None
    kf_illconditioned: bool
    message: str
    plate: PlateRuntimeState | None = None

    @property
    def d_lx(self) -> float | None:
        return self.d_ax

    @property
    def d_ly(self) -> float | None:
        return self.d_ay

    @property
    def x_l_rot(self) -> float | None:
        """Alias of ``x_contact_rot`` for partial-contact compatibility."""
        return self.x_contact_rot

    @property
    def y_l_rot(self) -> float | None:
        return self.y_contact_rot

    @property
    def force_residual_norm(self) -> float | None:
        return self.force_residual

    @property
    def is_full_contact(self) -> bool:
        """True when the selected record is the full-contact record."""
        return self.contact_type is ContactType.FULL

    def contact_span(self, L: float) -> tuple[float, float]:
        """Material contact interval of the selected record."""
        if self.contact_type is ContactType.FULL:
            return 0.0, float(L)
        if self.contact_type is ContactType.HEEL:
            return 0.0, float(self.l)
        return float(self.l), float(L)

    def routed_family_names(self) -> list[str]:
        """Routed topology family names as plain strings."""
        return [kind.value for kind in self.routed_families]


def angles_to_coefficients(
    phi_deg: float,
    theta_deg: float,
    phi_ref: float = 0.0,
) -> tuple[float, float, float, float]:
    """Convert GUI degrees to ``(varphi, alpha, r_x, r_y)``.

    ``phi`` is the absolute fixed-frame heel-to-toe chord angle and ``phi_ref``
    the reference chord angle of the undeformed mesh, both in radians for
    ``phi_ref``. Only ``varphi = phi - phi_ref`` enters the rotation terms.
    """
    phi = float(np.deg2rad(float(phi_deg)))
    theta = float(np.deg2rad(float(theta_deg)))
    varphi = corotation_angle(phi, phi_ref)
    alpha = shape_amplitude(theta)
    r_x, r_y = rotation_coefficients(varphi)
    return varphi, alpha, r_x, r_y


def _require_topology_lookup(lookup: ContactLookupResult) -> None:
    if int(lookup.schema_version) != LOOKUP_SCHEMA_VERSION:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    s = np.asarray(lookup.scalar_lookup)
    if s.ndim != 3 or s.shape[1] != N_BASIS_MODES:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    if (
        lookup.contact_type_codes is None
        or lookup.contact_mask is None
        or lookup.anchor_reference_x is None
        or lookup.corner_reactions_local is None
        or lookup.edge_contact_node_ids is None
        or lookup.edge_free_node_ids is None
    ):
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)


def _kf_table(lookup: ContactLookupResult, scalars: np.ndarray) -> np.ndarray:
    """Return the per-record 2x2 translation-to-force matrices."""
    n = scalars.shape[0]
    kf = lookup.kf_matrix
    if kf is not None and np.shape(kf) == (n, 2, 2):
        return np.asarray(kf, dtype=float)
    rebuilt = np.full((n, 2, 2), np.nan, dtype=float)
    for row in range(n):
        rebuilt[row] = kf_from_scalars(scalars[row])[0]
    return rebuilt


def evaluate_candidates(
    lookup: ContactLookupResult,
    Fx: float,
    Fy: float,
    phi_deg: float,
    theta_deg: float,
    tolerances: Tolerances | None = None,
) -> CandidateEvaluation:
    """Reconstruct every contact record for prescribed fixed-frame Fx, Fy and angles.

    All records are reconstructed, not only the routed families: a 2x2 solve plus
    five basis contractions per record is negligible, and the complete sweep is
    what the diagnostic plots and the routing decision are read from. The
    corner-reaction routing in :func:`select_contact_candidate` then restricts
    which family the selection may come from.
    """
    _require_topology_lookup(lookup)
    tol = tolerances or Tolerances()
    scalars = np.asarray(lookup.scalar_lookup, dtype=float)
    n = scalars.shape[0]
    n_t = len(lookup.x_top)
    n_b = len(lookup.x_bottom)

    phi = float(np.deg2rad(float(phi_deg)))
    theta = float(np.deg2rad(float(theta_deg)))
    varphi, alpha, r_x, r_y = angles_to_coefficients(phi_deg, theta_deg, lookup.phi_ref)

    Fx_star = float(Fx)
    Fy_star = float(Fy)
    F_local_star = rotate_force_to_local(Fx_star, Fy_star, varphi)

    kf = _kf_table(lookup, scalars)
    gamma_known = known_coefficients(alpha, varphi)
    # F_known^T = tan(theta) F_phi1 + (cos(varphi) - 1) F_Brx - sin(varphi) F_Bry
    F_known_x = contract_basis(scalars[:, :, SCALAR_FX], gamma_known, mode_axis=1)
    F_known_y = contract_basis(scalars[:, :, SCALAR_FY], gamma_known, mode_axis=1)

    kf_cond = np.full(n, np.nan)
    kf_det = np.full(n, np.nan)
    kf_svals = np.full((n, 2), np.nan)
    kf_ill = np.zeros(n, dtype=bool)
    well = np.zeros(n, dtype=bool)
    d_ax = np.full(n, np.nan)
    d_ay = np.full(n, np.nan)

    F_scale = max(
        abs(Fx_star),
        abs(Fy_star),
        float(np.nanmax(np.abs(kf))) if np.any(np.isfinite(kf)) else 1.0,
        1.0,
    )
    threshold = tol.F_abs_tol + tol.F_rel_tol * F_scale

    for row in range(n):
        A = np.asarray(kf[row], dtype=float)
        if not np.all(np.isfinite(A)):
            kf_ill[row] = True
            continue
        kf_det[row] = float(np.linalg.det(A))
        try:
            svals = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            kf_ill[row] = True
            kf_cond[row] = float("inf")
            continue
        kf_svals[row] = svals
        smin = float(svals[-1])
        smax = float(svals[0])
        kf_cond[row] = smax / max(smin, 1e-30)
        if smin <= threshold or not np.isfinite(kf_cond[row]):
            # Singular or numerically singular: report, never regularize.
            kf_ill[row] = True
            continue
        rhs = F_local_star - np.array([F_known_x[row], F_known_y[row]], dtype=float)
        try:
            sol = linalg.solve(A, rhs)
        except linalg.LinAlgError:
            kf_ill[row] = True
            continue
        d_ax[row] = float(sol[0])
        d_ay[row] = float(sol[1])
        well[row] = True
        if kf_cond[row] >= tol.kf_cond_warn:
            kf_ill[row] = True

    Fx_local = np.full(n, np.nan)
    Fy_local = np.full(n, np.nan)
    Fx_fixed = np.full(n, np.nan)
    Fy_fixed = np.full(n, np.nan)
    M = np.full(n, np.nan)
    T_toe = np.full(n, np.nan)
    x_cm = np.full(n, np.nan)
    x_cm_rel = np.full(n, np.nan)
    edge_free_gap = np.full(n, np.nan)
    edge_Rn = np.full(n, np.nan)
    edge_Rt = np.full(n, np.nan)
    gap = np.full((n, n_b), np.nan)
    bottom_u = np.full((n, n_b), np.nan)
    bottom_v = np.full((n, n_b), np.nan)
    reaction_n = np.full((n, n_b), np.nan)
    reaction_t = np.full((n, n_b), np.nan)
    reaction_x = np.full((n, n_b), np.nan)
    reaction_y = np.full((n, n_b), np.nan)
    corner_local = np.full((n, 2, 2), np.nan)
    corner_fixed = np.full((n, 2, 2), np.nan)
    corner_normal = np.full((n, 2), np.nan)
    corner_rx_heel = np.full(n, np.nan)
    corner_ry_heel = np.full(n, np.nan)
    corner_rx_toe = np.full(n, np.nan)
    corner_ry_toe = np.full(n, np.nan)
    Rn_heel = np.full(n, np.nan)
    Rn_toe = np.full(n, np.nan)
    top_u = np.full((n, n_t), np.nan)
    top_v = np.full((n, n_t), np.nan)
    top_fx = np.full((n, n_t), np.nan)
    top_fy = np.full((n, n_t), np.nan)
    force_residual = np.full(n, np.nan)

    W = np.asarray(lookup.basis_top_displacements, dtype=float)
    gap_v_basis = lookup.gap_v_basis if lookup.gap_v_basis is not None else lookup.gap_basis
    gap_u_basis = lookup.gap_u_basis
    reac_y_basis = (
        lookup.reaction_y_basis if lookup.reaction_y_basis is not None else lookup.reaction_basis
    )
    reac_x_basis = lookup.reaction_x_basis
    if gap_u_basis is None or reac_x_basis is None:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    top_fx_basis = lookup.top_force_x_basis
    top_fy_basis = lookup.top_force_y_basis
    if top_fx_basis is None or top_fy_basis is None:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    corner_basis = np.asarray(lookup.corner_reactions_local, dtype=float)

    x_bottom = np.asarray(lookup.x_bottom, dtype=float)
    x_top = np.asarray(lookup.x_top, dtype=float)
    y_top = (
        np.asarray(lookup.y_top, dtype=float)
        if lookup.y_top is not None
        else np.full(n_t, float(lookup.H))
    )
    candidate_l = np.asarray(lookup.candidate_l, dtype=float)
    anchor_x = np.asarray(lookup.anchor_reference_x, dtype=float)
    edge_contact_ids = np.asarray(lookup.edge_contact_node_ids, dtype=int)
    edge_free_ids = np.asarray(lookup.edge_free_node_ids, dtype=int)
    type_codes = np.asarray(lookup.contact_type_codes, dtype=int)

    for row in np.flatnonzero(well):
        free, contact = lookup.record_sets(row)
        x_a = float(anchor_x[row])
        gamma = basis_coefficients(alpha, d_ax[row], d_ay[row], varphi)

        s = contract_basis(scalars[row], gamma, mode_axis=0)
        Fx_local[row] = float(s[SCALAR_FX])
        Fy_local[row] = float(s[SCALAR_FY])
        M[row] = float(s[SCALAR_MV])
        # Lever arms rho_a, eta_a are reference quantities about (a, H_a), and
        # the moment about corresponding physical points is rotation invariant,
        # so the toe moment superposes as a scalar.
        T_toe[row] = float(s[SCALAR_TOE])

        fx_fixed, fy_fixed = rotate_vector_to_fixed(Fx_local[row], Fy_local[row], varphi)
        Fx_fixed[row] = float(fx_fixed)
        Fy_fixed[row] = float(fy_fixed)
        force_residual[row] = float(np.hypot(fx_fixed - Fx_star, fy_fixed - Fy_star))

        u_b = contract_basis(gap_u_basis[row], gamma, mode_axis=0)
        v_b = contract_basis(gap_v_basis[row], gamma, mode_axis=0)
        # Contact nodes are prescribed exactly, not stored in the free-surface
        # basis; fill them from the rotating-frame contact displacement about
        # this record's anchor.
        if contact.size:
            u_c, v_c = contact_displacement(x_bottom[contact], x_a, d_ax[row], d_ay[row], varphi)
            u_b[contact] = u_c
            v_b[contact] = v_c
        bottom_u[row] = u_b
        bottom_v[row] = v_b

        # Signed fixed-frame normal gap relative to the record anchor. Contact
        # nodes come out identically zero, which is exactly the statement that
        # they lie on the ground line.
        dx = (x_bottom + u_b) - (x_a + d_ax[row])
        dy = v_b - d_ay[row]
        gap[row] = fixed_frame_normal_component(dx, dy, varphi)

        rx = contract_basis(reac_x_basis[row], gamma, mode_axis=0)
        ry = contract_basis(reac_y_basis[row], gamma, mode_axis=0)
        reaction_x[row] = rx
        reaction_y[row] = ry
        reaction_n[row] = fixed_frame_normal_component(rx, ry, varphi)
        reaction_t[row] = fixed_frame_tangential_component(rx, ry, varphi)

        # Corner reactions come from their own stored basis so the fast
        # full-contact validity test never has to touch the interior vector.
        # Finite only for the full-contact record (elsewhere the basis is NaN).
        c_local = contract_basis(corner_basis[row], gamma, mode_axis=0)
        corner_local[row] = c_local
        corner_rx_heel[row] = float(c_local[CORNER_HEEL, 0])
        corner_ry_heel[row] = float(c_local[CORNER_HEEL, 1])
        corner_rx_toe[row] = float(c_local[CORNER_TOE, 0])
        corner_ry_toe[row] = float(c_local[CORNER_TOE, 1])
        cfx, cfy = rotate_vector_to_fixed(c_local[:, 0], c_local[:, 1], varphi)
        corner_fixed[row, :, 0] = cfx
        corner_fixed[row, :, 1] = cfy
        corner_normal[row] = fixed_frame_normal_component(c_local[:, 0], c_local[:, 1], varphi)
        Rn_heel[row] = float(corner_normal[row, CORNER_HEEL])
        Rn_toe[row] = float(corner_normal[row, CORNER_TOE])

        U_top = contract_basis(W, gamma, mode_axis=1)
        top_u[row] = U_top[:n_t]
        top_v[row] = U_top[n_t:]
        ftx = contract_basis(top_fx_basis[row], gamma, mode_axis=0)
        fty = contract_basis(top_fy_basis[row], gamma, mode_axis=0)
        top_fx[row] = ftx
        top_fy[row] = fty

        if edge_free_ids[row] >= 0:
            edge_free_gap[row] = float(gap[row][edge_free_ids[row]])
        if edge_contact_ids[row] >= 0:
            edge_Rn[row] = float(reaction_n[row][edge_contact_ids[row]])
            edge_Rt[row] = float(reaction_t[row][edge_contact_ids[row]])

        # Fixed-frame center of effort: rotate both the top nodal reactions and
        # the top nodal positions, using the r_a^F = (x_anchor, 0) gauge.
        _ftx_F, fty_F = rotate_vector_to_fixed(ftx, fty, varphi)
        x_top_F, _y_top_F = transform_to_fixed_frame(
            x_top,
            y_top,
            top_u[row],
            top_v[row],
            x_a,
            d_ax[row],
            d_ay[row],
            varphi,
        )
        fy_total = float(np.sum(fty_F))
        # Undefined when the requested fixed-frame vertical force is below the
        # force tolerance, and likewise when the reconstructed total is only
        # solver noise, which would otherwise divide into a meaningless x_cm.
        if abs(Fy_star) > tol.fy_zero_tol and abs(fy_total) > threshold:
            x_cm[row] = float(np.sum(x_top_F * fty_F) / fy_total)
            x_cm_rel[row] = x_cm[row] - x_a

    g_scale = (
        max(float(np.nanmax(np.abs(gap))), tol.g_floor) if np.any(np.isfinite(gap)) else tol.g_floor
    )
    R_scale = (
        max(float(np.nanmax(np.abs(reaction_n))), tol.R_floor)
        if np.any(np.isfinite(reaction_n))
        else tol.R_floor
    )
    # Section-15 denominators: the admissibility tolerances when they are set,
    # otherwise the data-driven scales so J stays finite and dimensionless.
    s_g = float(tol.tau_g) if tol.tau_g > 0.0 else g_scale
    s_R = float(tol.tau_R) if tol.tau_R > 0.0 else R_scale
    s_F = max(threshold, tol.g_floor)

    min_free_gap = np.full(n, np.nan)
    min_contact_reaction = np.full(n, np.nan)
    gap_violation = np.full(n, np.nan)
    reaction_violation = np.full(n, np.nan)
    violation = np.full(n, np.nan)
    edge = np.full(n, np.nan)
    admissible = np.zeros(n, dtype=bool)
    full_contact_valid = np.zeros(n, dtype=bool)
    strict_full_contact_valid = np.zeros(n, dtype=bool)

    full_row = lookup.full_contact_row()
    full_code = contact_type_code(ContactType.FULL)

    for row in range(n):
        if not well[row]:
            continue
        free, contact = lookup.record_sets(row)
        g = gap[row]
        r = reaction_n[row]
        is_full = int(type_codes[row]) == full_code

        if free.size:
            min_free_gap[row] = float(np.min(g[free]))
            gap_violation[row] = float(max(0.0, -min_free_gap[row]))
        else:
            min_free_gap[row] = np.inf
            gap_violation[row] = 0.0
        if contact.size:
            min_contact_reaction[row] = float(np.min(r[contact]))
            reaction_violation[row] = float(max(0.0, -min_contact_reaction[row]))
        else:
            min_contact_reaction[row] = np.inf
            reaction_violation[row] = 0.0

        # Section 15: J = (v_g/s_g)^2 + (v_R/s_R)^2 + (F_err/s_F)^2
        violation[row] = (
            (gap_violation[row] / s_g) ** 2
            + (reaction_violation[row] / s_R) ** 2
            + (force_residual[row] / s_F) ** 2
        )

        force_ok = bool(force_residual[row] <= s_F)

        if is_full:
            # Corner-only validity: interior tension stays diagnostic.
            rn0 = float(Rn_heel[row])
            rnL = float(Rn_toe[row])
            corners_ok = bool(
                np.isfinite(rn0)
                and np.isfinite(rnL)
                and rn0 >= -tol.tau_R
                and rnL >= -tol.tau_R
            )
            full_contact_valid[row] = corners_ok
            if contact.size:
                strict_full_contact_valid[row] = bool(
                    corners_ok and np.all(r[contact] >= -tol.tau_R)
                )
            else:
                strict_full_contact_valid[row] = corners_ok
            # Default admissibility ignores interior tension.
            admissible[row] = corners_ok and force_ok
            # Full contact has no free/contact edge; use J as the ranking score.
            edge[row] = violation[row]
        else:
            gap_ok = True if free.size == 0 else bool(np.all(g[free] >= -tol.tau_g))
            reac_ok = True if contact.size == 0 else bool(np.all(r[contact] >= -tol.tau_R))
            admissible[row] = gap_ok and reac_ok and force_ok
            if np.isfinite(edge_free_gap[row]) and np.isfinite(edge_Rn[row]):
                edge[row] = (edge_free_gap[row] / s_g) ** 2 + (edge_Rn[row] / s_R) ** 2

    routed, reason = _route_families(
        bool(well[full_row]),
        float(Rn_heel[full_row]) if well[full_row] else float("nan"),
        float(Rn_toe[full_row]) if well[full_row] else float("nan"),
        float(tol.tau_R),
        bool(full_contact_valid[full_row]),
    )

    local_err = (
        np.max(
            np.abs(
                np.stack(
                    [Fx_local[well] - F_local_star[0], Fy_local[well] - F_local_star[1]]
                )
            )
        )
        if np.any(well)
        else float("nan")
    )
    fixed_err = (
        np.max(np.abs(np.stack([Fx_fixed[well] - Fx_star, Fy_fixed[well] - Fy_star])))
        if np.any(well)
        else float("nan")
    )

    # x_contact_rot = candidate_l + d_ax when the record has a material edge;
    # NaN for full contact (candidate_l is NaN).
    x_contact_rot = candidate_l + d_ax
    y_contact_rot = np.where(np.isfinite(candidate_l), d_ay, np.nan)

    return CandidateEvaluation(
        contact_type_codes=type_codes.copy(),
        l=candidate_l.copy(),
        anchor_x=anchor_x.copy(),
        candidate_indices=np.asarray(lookup.candidate_indices, dtype=int).copy(),
        has_free_edge=np.asarray(lookup.has_free_edge, dtype=bool).copy(),
        phi=phi,
        theta=theta,
        phi_ref=float(lookup.phi_ref),
        varphi=varphi,
        alpha=alpha,
        r_x=r_x,
        r_y=r_y,
        Fx_star=Fx_star,
        Fy_star=Fy_star,
        Fx_star_local=float(F_local_star[0]),
        Fy_star_local=float(F_local_star[1]),
        d_ax=d_ax,
        d_ay=d_ay,
        x_anchor_rot=anchor_x + d_ax,
        y_anchor_rot=d_ay.copy(),
        x_contact_rot=x_contact_rot,
        y_contact_rot=y_contact_rot,
        Fx_local=Fx_local,
        Fy_local=Fy_local,
        Fx=Fx_fixed,
        Fy=Fy_fixed,
        M=M,
        x_cm=x_cm,
        x_cm_rel=x_cm_rel,
        T_toe=T_toe,
        edge_free_gap=edge_free_gap,
        edge_contact_reaction_normal=edge_Rn,
        edge_contact_reaction_tangential=edge_Rt,
        corner_rx_heel=corner_rx_heel,
        corner_ry_heel=corner_ry_heel,
        corner_rx_toe=corner_rx_toe,
        corner_ry_toe=corner_ry_toe,
        corner_reaction_local=corner_local,
        corner_reaction_fixed=corner_fixed,
        corner_reaction_normal=corner_normal,
        Rn_heel_corner=Rn_heel,
        Rn_toe_corner=Rn_toe,
        full_bottom_gap=gap,
        full_bottom_u=bottom_u,
        full_bottom_v=bottom_v,
        full_bottom_reaction_normal=reaction_n,
        full_bottom_reaction_tangential=reaction_t,
        full_bottom_reaction_x=reaction_x,
        full_bottom_reaction_y=reaction_y,
        top_u=top_u,
        top_v=top_v,
        top_force_x=top_fx,
        top_force_y=top_fy,
        min_free_gap=min_free_gap,
        min_contact_reaction=min_contact_reaction,
        gap_violation=gap_violation,
        reaction_violation=reaction_violation,
        force_residual=force_residual,
        force_residual_norm=force_residual.copy(),
        violation_score=violation,
        edge_score=edge,
        admissible=admissible,
        well_conditioned=well,
        kf_cond=kf_cond,
        kf_det=kf_det,
        kf_svals=kf_svals,
        kf_illconditioned=kf_ill,
        full_contact_row=int(full_row),
        full_contact_valid=full_contact_valid,
        strict_full_contact_valid=strict_full_contact_valid,
        routed_families=routed,
        routing_reason=reason,
        g_scale=g_scale,
        R_scale=R_scale,
        F_reference=F_scale,
        F_abs_tol=tol.F_abs_tol,
        F_rel_tol=tol.F_rel_tol,
        J_gap_scale=s_g,
        J_reaction_scale=s_R,
        J_force_scale=s_F,
        diagnostics={
            "smin_threshold": threshold,
            "n_ill_conditioned": int(np.count_nonzero(~well)),
            "fy_zero": abs(Fy_star) <= tol.fy_zero_tol,
            "max_local_force_residual": float(local_err),
            "max_fixed_force_residual": float(fixed_err),
            "basis_order": list(BASIS_MODE_NAMES),
            "gauge": "r_a^F = (x_anchor, 0)",
            "J_definition": "(v_g/s_g)^2 + (v_R/s_R)^2 + (|F^F - F^F*|/s_F)^2",
        },
    )


def _route_families(
    full_well: bool,
    Rn_heel: float,
    Rn_toe: float,
    tau_R: float,
    full_valid: bool,
) -> tuple[tuple[ContactType, ...], str]:
    """Corner-reaction routing from full contact to the partial families.

    Positive normal reaction is compression. A tensile corner means the body is
    lifting there, so the search moves to the family that keeps the *other* end
    on the ground.
    """
    if full_well and full_valid:
        return (
            (ContactType.FULL,),
            "both corners compressive",
        )
    if not full_well:
        return (
            (ContactType.HEEL, ContactType.TOE),
            "full-contact K_F not well-conditioned; searching heel and toe",
        )
    # Well but not valid: route from the corner signs (treat non-finite as tensile).
    heel_rn = Rn_heel if np.isfinite(Rn_heel) else float("-inf")
    toe_rn = Rn_toe if np.isfinite(Rn_toe) else float("-inf")
    heel_tensile = heel_rn < -tau_R
    toe_tensile = toe_rn < -tau_R
    if heel_tensile and not toe_tensile:
        return (
            (ContactType.TOE,),
            f"heel corner tensile ({heel_rn:.4g}); searching toe family",
        )
    if toe_tensile and not heel_tensile:
        return (
            (ContactType.HEEL,),
            f"toe corner tensile ({toe_rn:.4g}); searching heel family",
        )
    return (
        (ContactType.HEEL, ContactType.TOE),
        f"both corners tensile (heel {heel_rn:.4g}, toe {toe_rn:.4g}); "
        "searching heel and toe",
    )


def _pool_rows(
    evaluation: CandidateEvaluation,
    mode: ContactMode,
) -> tuple[np.ndarray, tuple[ContactType, ...]]:
    """Rows the selection may draw from, and the families they represent."""
    if mode is ContactMode.AUTO:
        families = evaluation.routed_families
    else:
        families = (ContactType(mode.value),)
    rows = np.concatenate([evaluation.rows_for(kind) for kind in families]).astype(int)
    return np.sort(rows), families


def _tie_break(
    rows: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    previous_index: int | None,
) -> int:
    """Pick the smallest score, breaking exact ties toward ``previous_index``.

    Proximity to the previous selection is only a final tie-breaker; it exists
    to stop the GUI flickering between numerically indistinguishable candidates.
    When ``previous_index`` is None, the smallest row index among ties wins.
    """
    finite = np.isfinite(scores)
    if not np.any(finite):
        return int(rows[0])
    best = float(np.min(scores[finite]))
    window = 1e-9 * max(1.0, abs(best))
    tied = np.flatnonzero(finite & (scores <= best + window))
    if tied.size == 1:
        return int(rows[tied[0]])
    if previous_index is None:
        return int(rows[tied[0]])
    distance = np.abs(indices[tied] - int(previous_index))
    return int(rows[tied[int(np.argmin(distance))]])


def select_contact_candidate(
    evaluation: CandidateEvaluation,
    tolerances: Tolerances | None = None,
    mode: ContactMode | str = ContactMode.AUTO,
    previous_index: int | None = None,
) -> ContactSelection:
    """Select a contact record from fixed-frame normal gap / normal reaction only.

    ``mode`` is the GUI contact-mode control. ``AUTO`` applies the
    corner-reaction routing: full contact when both corners are compressive,
    otherwise only the routed partial family or families. Forced modes restrict
    the search to a single topology. Tangential reaction, toe moment, and center
    of effort are never used. Full contact is never returned silently when either
    corner is tensile.
    """
    _ = tolerances  # selection uses flags already computed in evaluation
    mode = ContactMode(mode)
    well = evaluation.well_conditioned
    full_row = int(evaluation.full_contact_row)
    rn_heel = float(evaluation.Rn_heel_corner[full_row])
    rn_toe = float(evaluation.Rn_toe_corner[full_row])
    full_valid = bool(evaluation.full_contact_valid[full_row])
    full_valid_strict = bool(evaluation.strict_full_contact_valid[full_row])

    def _empty(message: str) -> ContactSelection:
        return _build_selection(
            evaluation,
            mode,
            None,
            False,
            True,
            np.array([], dtype=int),
            message,
            rn_heel=rn_heel if np.isfinite(rn_heel) else None,
            rn_toe=rn_toe if np.isfinite(rn_toe) else None,
            full_valid=full_valid,
            full_valid_strict=full_valid_strict,
        )

    if not np.any(well):
        return _empty(
            "No well-conditioned records: K_F is singular or numerically singular "
            "everywhere. Cannot select a contact topology."
        )

    # Forced FULL: always report the full row and its corner validity.
    if mode is ContactMode.FULL:
        row = full_row
        if not well[row]:
            return _empty(
                "Full contact forced, but the full-contact K_F is not well-conditioned."
            )
        if full_valid:
            message = "Full contact forced; both corner normal reactions are compressive."
            exactly, approximate = True, False
        else:
            message = (
                "Full contact forced, but a corner normal reaction is tensile "
                f"(heel {rn_heel:.4g}, toe {rn_toe:.4g}); "
                "auto mode would route to a partial family."
            )
            exactly, approximate = False, True
        return _build_selection(
            evaluation,
            mode,
            row,
            exactly,
            approximate,
            np.flatnonzero(evaluation.admissible & well),
            message,
            rn_heel=rn_heel,
            rn_toe=rn_toe,
            full_valid=full_valid,
            full_valid_strict=full_valid_strict,
        )

    rows, families = _pool_rows(evaluation, mode)
    pool = rows[well[rows]] if rows.size else np.array([], dtype=int)
    if pool.size == 0:
        return _empty(
            f"No well-conditioned record in the {'/'.join(k.value for k in families)} "
            f"family. {evaluation.routing_reason}"
        )

    admissible_rows = pool[evaluation.admissible[pool]]

    # AUTO + both corners compressive: select full immediately. Never fall
    # through to a silent full return when either corner is tensile.
    if mode is ContactMode.AUTO and families == (ContactType.FULL,):
        row = full_row
        message = f"Full contact. {evaluation.routing_reason}"
        return _build_selection(
            evaluation,
            mode,
            row,
            True,
            False,
            admissible_rows,
            message,
            rn_heel=rn_heel,
            rn_toe=rn_toe,
            full_valid=full_valid,
            full_valid_strict=full_valid_strict,
        )

    if admissible_rows.size >= 1:
        row = _tie_break(
            admissible_rows,
            evaluation.edge_score[admissible_rows],
            evaluation.candidate_indices[admissible_rows],
            previous_index,
        )
        exactly, approximate = True, False
        if admissible_rows.size == 1:
            message = "Exactly one admissible well-conditioned candidate."
        else:
            message = (
                f"{admissible_rows.size} admissible candidates; selected the smallest "
                "edge score."
            )
        if mode is ContactMode.AUTO:
            message = f"{message} {evaluation.routing_reason}"
    else:
        # Approximate: least-violating well row in the routed families.
        row = _tie_break(
            pool,
            evaluation.violation_score[pool],
            evaluation.candidate_indices[pool],
            previous_index,
        )
        exactly, approximate = False, True
        message = (
            f"{INSUFFICIENT_TOPOLOGY_MESSAGE} Showing the least-violating "
            f"{evaluation.contact_type(row).value} record (approximate). "
            f"{evaluation.routing_reason}"
        )

    if evaluation.kf_illconditioned[row]:
        message = message + " Warning: K_F is ill-conditioned."
    return _build_selection(
        evaluation,
        mode,
        row,
        exactly,
        approximate,
        admissible_rows,
        message,
        rn_heel=rn_heel if np.isfinite(rn_heel) else None,
        rn_toe=rn_toe if np.isfinite(rn_toe) else None,
        full_valid=full_valid,
        full_valid_strict=full_valid_strict,
    )


def _build_selection(
    evaluation: CandidateEvaluation,
    mode: ContactMode,
    row: int | None,
    exactly: bool,
    approximate: bool,
    admissible_rows: np.ndarray,
    message: str,
    *,
    rn_heel: float | None,
    rn_toe: float | None,
    full_valid: bool | None,
    full_valid_strict: bool | None,
) -> ContactSelection:
    """Assemble a ContactSelection, with all per-record fields None when unselected."""
    common = {
        "evaluation": evaluation,
        "contact_mode": mode,
        "phi": evaluation.phi,
        "theta": evaluation.theta,
        "varphi": evaluation.varphi,
        "alpha": evaluation.alpha,
        "Fx_star": evaluation.Fx_star,
        "Fy_star": evaluation.Fy_star,
        "Rn_heel_corner": rn_heel,
        "Rn_toe_corner": rn_toe,
        "full_contact_valid": full_valid,
        "full_contact_valid_strict": full_valid_strict,
        "routed_families": evaluation.routed_families,
        "routing_reason": evaluation.routing_reason,
        "exactly_admissible": exactly,
        "approximate": approximate,
        "n_admissible": int(np.size(admissible_rows)),
        "admissible_rows": np.asarray(admissible_rows, dtype=int),
        "message": message,
    }
    if row is None:
        return ContactSelection(
            selected_row=None,
            selected_index=None,
            contact_type=None,
            l=None,
            anchor_x=None,
            has_free_edge=False,
            d_ax=None,
            d_ay=None,
            x_anchor_rot=None,
            y_anchor_rot=None,
            x_contact_rot=None,
            y_contact_rot=None,
            Fx=None,
            Fy=None,
            Fx_local=None,
            Fy_local=None,
            M=None,
            x_cm=None,
            x_cm_rel=None,
            T_toe=None,
            edge_free_gap=None,
            edge_contact_reaction_normal=None,
            edge_contact_reaction_tangential=None,
            min_free_gap=None,
            min_contact_reaction=None,
            force_residual=None,
            full_bottom_gap=None,
            full_bottom_u=None,
            full_bottom_v=None,
            full_bottom_reaction_normal=None,
            full_bottom_reaction_tangential=None,
            top_u=None,
            top_v=None,
            violation_score=None,
            edge_score=None,
            kf_cond=None,
            kf_illconditioned=True,
            **common,
        )
    row = int(row)
    kind = evaluation.contact_type(row)
    l_val = float(evaluation.l[row])
    return ContactSelection(
        selected_row=row,
        selected_index=int(evaluation.candidate_indices[row]),
        contact_type=kind,
        l=None if kind is ContactType.FULL or not np.isfinite(l_val) else l_val,
        anchor_x=float(evaluation.anchor_x[row]),
        has_free_edge=bool(evaluation.has_free_edge[row]),
        d_ax=float(evaluation.d_ax[row]),
        d_ay=float(evaluation.d_ay[row]),
        x_anchor_rot=float(evaluation.x_anchor_rot[row]),
        y_anchor_rot=float(evaluation.y_anchor_rot[row]),
        x_contact_rot=(
            float(evaluation.x_contact_rot[row])
            if np.isfinite(evaluation.x_contact_rot[row])
            else None
        ),
        y_contact_rot=(
            float(evaluation.y_contact_rot[row])
            if np.isfinite(evaluation.y_contact_rot[row])
            else None
        ),
        Fx=float(evaluation.Fx[row]),
        Fy=float(evaluation.Fy[row]),
        Fx_local=float(evaluation.Fx_local[row]),
        Fy_local=float(evaluation.Fy_local[row]),
        M=float(evaluation.M[row]),
        x_cm=float(evaluation.x_cm[row]),
        x_cm_rel=float(evaluation.x_cm_rel[row]),
        T_toe=float(evaluation.T_toe[row]),
        edge_free_gap=(
            float(evaluation.edge_free_gap[row])
            if np.isfinite(evaluation.edge_free_gap[row])
            else None
        ),
        edge_contact_reaction_normal=(
            float(evaluation.edge_contact_reaction_normal[row])
            if np.isfinite(evaluation.edge_contact_reaction_normal[row])
            else None
        ),
        edge_contact_reaction_tangential=(
            float(evaluation.edge_contact_reaction_tangential[row])
            if np.isfinite(evaluation.edge_contact_reaction_tangential[row])
            else None
        ),
        min_free_gap=float(evaluation.min_free_gap[row]),
        min_contact_reaction=float(evaluation.min_contact_reaction[row]),
        force_residual=float(evaluation.force_residual[row]),
        full_bottom_gap=evaluation.full_bottom_gap[row].copy(),
        full_bottom_u=evaluation.full_bottom_u[row].copy(),
        full_bottom_v=evaluation.full_bottom_v[row].copy(),
        full_bottom_reaction_normal=evaluation.full_bottom_reaction_normal[row].copy(),
        full_bottom_reaction_tangential=evaluation.full_bottom_reaction_tangential[row].copy(),
        top_u=evaluation.top_u[row].copy(),
        top_v=evaluation.top_v[row].copy(),
        violation_score=float(evaluation.violation_score[row]),
        edge_score=(
            float(evaluation.edge_score[row])
            if np.isfinite(evaluation.edge_score[row])
            else None
        ),
        kf_cond=float(evaluation.kf_cond[row]),
        kf_illconditioned=bool(evaluation.kf_illconditioned[row]),
        **common,
    )


def evaluate_from_angles(
    lookup: ContactLookupResult,
    Fx: float,
    Fy: float,
    phi_deg: float,
    theta_deg: float,
    tolerances: Tolerances | None = None,
    mode: ContactMode | str = ContactMode.AUTO,
    previous_index: int | None = None,
) -> ContactSelection:
    """Fixed-frame (Fx, Fy, phi, theta) -> record evaluation -> selected topology."""
    evaluation = evaluate_candidates(
        lookup, Fx, Fy, phi_deg, theta_deg, tolerances=tolerances
    )
    return select_contact_candidate(
        evaluation, tolerances=tolerances, mode=mode, previous_index=previous_index
    )


def top_displacement_vector(
    x: np.ndarray,
    alpha: float,
    L: float,
    a: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotating-frame top displacement: u = 0, v = alpha * phi1(x)."""
    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x, dtype=float)
    v = float(alpha) * shape_mode_phi1(x, L=L, a=a, kappa=kappa)
    return u, v


def _f(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def evaluation_to_csv_rows(selection: ContactSelection) -> list[dict]:
    """Build CSV row dicts for the full record table."""
    ev = selection.evaluation
    rows = []
    for i in range(len(ev.l)):
        rows.append(
            {
                "contact_type": ev.contact_type(i).value,
                "edge_node_id": int(ev.candidate_indices[i]),
                "l": _f(ev.l[i]),
                "anchor_x": _f(ev.anchor_x[i]),
                "x_contact_rot": _f(ev.x_contact_rot[i]),
                "x_anchor_rot": _f(ev.x_anchor_rot[i]),
                "x_l_rot": _f(ev.x_l_rot[i]),
                "d_ax": _f(ev.d_ax[i]),
                "d_ay": _f(ev.d_ay[i]),
                "d_lx": _f(ev.d_lx[i]),
                "d_ly": _f(ev.d_ly[i]),
                "phi_deg": float(np.rad2deg(ev.phi)),
                "theta_deg": float(np.rad2deg(ev.theta)),
                "varphi_deg": float(np.rad2deg(ev.varphi)),
                "alpha": float(ev.alpha),
                "Fx_star": ev.Fx_star,
                "Fy_star": ev.Fy_star,
                "Fx": _f(ev.Fx[i]),
                "Fy": _f(ev.Fy[i]),
                "Fx_local": _f(ev.Fx_local[i]),
                "Fy_local": _f(ev.Fy_local[i]),
                "M": _f(ev.M[i]),
                "x_cm": _f(ev.x_cm[i]),
                "x_cm_rel": _f(ev.x_cm_rel[i]),
                "T_toe": _f(ev.T_toe[i]),
                "edge_free_gap": _f(ev.edge_free_gap[i]),
                "edge_contact_reaction_normal": _f(ev.edge_contact_reaction_normal[i]),
                "edge_contact_reaction_tangential": _f(ev.edge_contact_reaction_tangential[i]),
                "corner_rx_heel": _f(ev.corner_rx_heel[i]),
                "corner_ry_heel": _f(ev.corner_ry_heel[i]),
                "corner_rx_toe": _f(ev.corner_rx_toe[i]),
                "corner_ry_toe": _f(ev.corner_ry_toe[i]),
                "Rn_heel_corner": _f(ev.Rn_heel_corner[i]),
                "Rn_toe_corner": _f(ev.Rn_toe_corner[i]),
                "full_contact_valid": bool(ev.full_contact_valid[i]),
                "strict_full_contact_valid": bool(ev.strict_full_contact_valid[i]),
                "min_free_gap": _f(ev.min_free_gap[i]),
                "min_contact_reaction": _f(ev.min_contact_reaction[i]),
                "force_residual": _f(ev.force_residual[i]),
                "force_residual_norm": _f(ev.force_residual_norm[i]),
                "violation_score": _f(ev.violation_score[i]),
                "edge_score": _f(ev.edge_score[i]),
                "kf_cond": _f(ev.kf_cond[i]),
                "kf_det": _f(ev.kf_det[i]),
                "admissible": bool(ev.admissible[i]),
                "well_conditioned": bool(ev.well_conditioned[i]),
                "kf_illconditioned": bool(ev.kf_illconditioned[i]),
                "selected": selection.selected_row is not None and i == selection.selected_row,
            }
        )
    return rows


def export_evaluation_csv(selection: ContactSelection, path: Path | str) -> Path:
    """Write the evaluated record table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = evaluation_to_csv_rows(selection)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_selected_result(
    selection: ContactSelection,
    lookup: ContactLookupResult,
    path: Path | str,
) -> Path:
    """Save selected result arrays to a compressed NPZ."""
    if selection.selected_row is None:
        raise ValueError("No selected record to export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        contact_type=selection.contact_type.value,
        contact_mode=selection.contact_mode.value,
        l=selection.l if selection.l is not None else np.nan,
        anchor_x=selection.anchor_x,
        has_free_edge=selection.has_free_edge,
        x_contact_rot=(
            selection.x_contact_rot if selection.x_contact_rot is not None else np.nan
        ),
        y_contact_rot=(
            selection.y_contact_rot if selection.y_contact_rot is not None else np.nan
        ),
        x_l_rot=selection.x_l_rot if selection.x_l_rot is not None else np.nan,
        x_anchor_rot=selection.x_anchor_rot,
        y_anchor_rot=selection.y_anchor_rot,
        d_ax=selection.d_ax,
        d_ay=selection.d_ay,
        d_lx=selection.d_lx,
        d_ly=selection.d_ly,
        phi=selection.phi,
        theta=selection.theta,
        varphi=selection.varphi,
        phi_ref=lookup.phi_ref,
        alpha=selection.alpha,
        Fx_star=selection.Fx_star,
        Fy_star=selection.Fy_star,
        Fx=selection.Fx,
        Fy=selection.Fy,
        Fx_local=selection.Fx_local,
        Fy_local=selection.Fy_local,
        M=selection.M,
        x_cm=selection.x_cm,
        x_cm_rel=selection.x_cm_rel,
        T_toe=selection.T_toe,
        edge_free_gap=(
            selection.edge_free_gap if selection.edge_free_gap is not None else np.nan
        ),
        edge_contact_reaction_normal=(
            selection.edge_contact_reaction_normal
            if selection.edge_contact_reaction_normal is not None
            else np.nan
        ),
        edge_contact_reaction_tangential=(
            selection.edge_contact_reaction_tangential
            if selection.edge_contact_reaction_tangential is not None
            else np.nan
        ),
        Rn_heel_corner=(
            selection.Rn_heel_corner if selection.Rn_heel_corner is not None else np.nan
        ),
        Rn_toe_corner=(
            selection.Rn_toe_corner if selection.Rn_toe_corner is not None else np.nan
        ),
        full_contact_valid=(
            selection.full_contact_valid
            if selection.full_contact_valid is not None
            else False
        ),
        full_contact_valid_strict=(
            selection.full_contact_valid_strict
            if selection.full_contact_valid_strict is not None
            else False
        ),
        routed_families=np.asarray(
            [kind.value for kind in selection.routed_families], dtype=object
        ),
        routing_reason=selection.routing_reason,
        kf_cond=selection.kf_cond,
        x_top=lookup.x_top,
        x_bottom=lookup.x_bottom,
        contact_mask=lookup.contact_mask[selection.selected_row],
        top_u=selection.top_u,
        top_v=selection.top_v,
        bottom_u=selection.full_bottom_u,
        bottom_v=selection.full_bottom_v,
        bottom_gap=selection.full_bottom_gap,
        bottom_reaction_normal=selection.full_bottom_reaction_normal,
        bottom_reaction_tangential=selection.full_bottom_reaction_tangential,
        exactly_admissible=selection.exactly_admissible,
        approximate=selection.approximate,
        softplus_a=lookup.softplus_a,
        softplus_kappa=lookup.softplus_kappa,
        L=lookup.L,
        H=lookup.H,
        gauge="r_a^F = (x_anchor, 0)",
    )
    return path
