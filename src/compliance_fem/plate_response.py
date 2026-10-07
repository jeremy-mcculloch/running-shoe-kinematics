"""Internal plate response: primal recovery, Hermite postprocess, runtime fields.

The plate is the Euler–Bernoulli / Hermite interface of the layered foam model.
All expensive work happens during lookup generation by recovering the full
primal FEM displacement for each contact-state basis response from the same
augmented free-body operator used to build the compliance tables. Runtime
only contracts the five stored plate basis fields with ``gamma`` and evaluates
Hermite interpolants.

Sign conventions (must match ``plate.py`` / ``config.LayeredPlateConfig``)
-------------------------------------------------------------------------
* Plate tangent ``t_p`` points heel → toe.
* Plate normal ``n_p = (-t_y, t_x)``; for a flat plate ``n_p = (0, 1)`` (up).
* Rotation DOF ``θ = dw/ds`` with ``w = n_p · (u, v)``.
* Axial constraint rows are unscaled: ``t_p · (u_{j+1} - u_j) = 0``.
  The Lagrange multiplier therefore has force units and equals the element
  axial constraint force under this convention (positive tension when the
  multiplier resists relative extension in the ``+t_p`` sense of the row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from compliance_fem.boundaries import build_vector_selector
from compliance_fem.contact_basis import N_AFFINE_COLUMNS
from compliance_fem.corotation import (
    basis_coefficients,
    contract_basis,
    transform_to_fixed_frame,
)
from compliance_fem.plate import plate_normal_deflection, plate_translational_dofs

if TYPE_CHECKING:
    from compliance_fem.compliance import ComplianceResult
    from compliance_fem.contact_lookup import ContactLookupResult
    from compliance_fem.force_control import ContactSelection

PLATE_MODEL = "Euler-Bernoulli"
PLATE_AXIAL_MODEL = "exact_inextensibility_constraint"
CONSTRAINT_ROW_NORMALIZATION = "unscaled_t_dot_delta_u"
PLATE_ROTATION_SIGN = "theta = dw/ds with w = n_p · (u, v)"
PLATE_NORMAL_SIGN = "n_p = (-t_y, t_x); flat plate n_p = (0, 1) upward"
COORDINATE_FRAME = "corotated_local"
DEFAULT_SAMPLES_PER_ELEMENT = 16

REGENERATE_PLATE_MESSAGE = (
    "This contact-lookup file has no plate_response section (schema < 6 or "
    "rectangle-only table). Regenerate the layered lookup with the current "
    "contact_lookup CLI while the FEM factorization is available in-process, "
    "or disable plate rendering."
)


@dataclass(frozen=True)
class PlateMeshInfo:
    """Reference plate mesh shared by every contact record."""

    node_ids: np.ndarray
    reference_x: np.ndarray
    reference_y: np.ndarray
    reference_arc_length: np.ndarray
    element_connectivity: np.ndarray
    element_lengths: np.ndarray
    element_tangents: np.ndarray
    element_normals: np.ndarray
    u_dof_ids: np.ndarray
    v_dof_ids: np.ndarray
    rotation_dof_ids: np.ndarray
    constraint_ids: np.ndarray
    plate_tangent: np.ndarray
    plate_normal: np.ndarray
    EI_plate: float
    n_foam: int
    n_primal: int
    n_lambda: int

    @property
    def n_nodes(self) -> int:
        return int(self.reference_x.size)

    @property
    def n_elements(self) -> int:
        return int(self.element_connectivity.shape[0])


@dataclass
class PlateBasisFields:
    """Five-mode plate fields for one contact record (local rotating frame)."""

    u_local: np.ndarray  # (5, n_nodes)
    v_local: np.ndarray
    rotation_local: np.ndarray
    constraint_multiplier: np.ndarray  # (5, n_constraints)
    axial_force: np.ndarray  # (5, n_elements)
    bp_residual: np.ndarray  # (5,)
    top_bc_residual: np.ndarray  # (5,)
    contact_bc_residual: np.ndarray  # (5,)


@dataclass
class PlateRuntimeState:
    """Superposed plate response for the selected contact record."""

    mesh: PlateMeshInfo
    u_local: np.ndarray
    v_local: np.ndarray
    rotation_local: np.ndarray
    constraint_multiplier: np.ndarray
    axial_force: np.ndarray
    u_tangential: np.ndarray
    w_normal: np.ndarray
    maximum_plate_axial_displacement_difference: float
    plate_constraint_residual_norm: float
    sample_s: np.ndarray
    sample_x_local: np.ndarray
    sample_y_local: np.ndarray
    sample_x_fixed: np.ndarray
    sample_y_fixed: np.ndarray
    sample_rotation_local: np.ndarray
    sample_rotation_fixed: np.ndarray
    sample_curvature: np.ndarray
    sample_bending_moment: np.ndarray
    sample_shear_force: np.ndarray
    sample_axial_force: np.ndarray
    sample_w: np.ndarray
    sample_u_s: np.ndarray
    sample_element_id: np.ndarray
    color_quantity: str = "none"
    color_values: np.ndarray | None = None
    color_units: str = ""
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hermite shape functions (local DOF order w_i, θ_i, w_j, θ_j)
# ---------------------------------------------------------------------------


def hermite_N(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return N1..N4 for cubic Hermite interpolation on [0, L]."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    N2 = L * (xi - 2.0 * xi**2 + xi**3)
    N3 = 3.0 * xi**2 - 2.0 * xi**3
    N4 = L * (-(xi**2) + xi**3)
    return N1, N2, N3, N4


def hermite_dN_ds(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dN_i/ds (θ = dw/ds)."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    dxi = 1.0 / L
    dN1 = (-6.0 * xi + 6.0 * xi**2) * dxi
    dN2 = 1.0 - 4.0 * xi + 3.0 * xi**2
    dN3 = (6.0 * xi - 6.0 * xi**2) * dxi
    dN4 = -2.0 * xi + 3.0 * xi**2
    return dN1, dN2, dN3, dN4


def hermite_d2N_ds2(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return d²N_i/ds² (curvature shape functions)."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    dxi2 = 1.0 / L**2
    d2N1 = (-6.0 + 12.0 * xi) * dxi2
    d2N2 = (-4.0 + 6.0 * xi) / L
    d2N3 = (6.0 - 12.0 * xi) * dxi2
    d2N4 = (-2.0 + 6.0 * xi) / L
    return d2N1, d2N2, d2N3, d2N4


def hermite_d3N_ds3(length: float) -> tuple[float, float, float, float]:
    """Return constant d³N_i/ds³ (shear shape functions)."""
    L = float(length)
    return 12.0 / L**3, 6.0 / L**2, -12.0 / L**3, 6.0 / L**2


def hermite_deflection(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    N1, N2, N3, N4 = hermite_N(xi, length)
    return N1 * w_i + N2 * theta_i + N3 * w_j + N4 * theta_j


def hermite_rotation(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    dN1, dN2, dN3, dN4 = hermite_dN_ds(xi, length)
    return dN1 * w_i + dN2 * theta_i + dN3 * w_j + dN4 * theta_j


def hermite_curvature(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    d2N1, d2N2, d2N3, d2N4 = hermite_d2N_ds2(xi, length)
    return d2N1 * w_i + d2N2 * theta_i + d2N3 * w_j + d2N4 * theta_j


def hermite_shear(
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
    EI: float,
) -> float:
    d3N1, d3N2, d3N3, d3N4 = hermite_d3N_ds3(length)
    w_ppp = d3N1 * w_i + d3N2 * theta_i + d3N3 * w_j + d3N4 * theta_j
    return float(EI) * float(w_ppp)


# ---------------------------------------------------------------------------
# Mesh extraction and primal recovery
# ---------------------------------------------------------------------------


def _arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.zeros(x.size, dtype=float)
    s[1:] = np.cumsum(seg)
    return s


def extract_plate_mesh_info(result: ComplianceResult) -> PlateMeshInfo:
    """Build plate mesh / DOF index arrays from a layered ComplianceResult."""
    if result.basis is None or result.B_p is None or result.n_primal is None:
        raise ValueError("ComplianceResult must retain basis, B_p, and n_primal for plate recovery.")
    if result.x_plate is None or result.y_plate is None:
        raise ValueError("ComplianceResult is missing plate reference coordinates.")
    cfg = result.config
    if result.plate_node_ids is not None and not hasattr(cfg, "plate_tangent"):
        node_ids = np.asarray(result.plate_node_ids, dtype=int)
        p = np.asarray(result.basis.mesh.p, dtype=float)
        x = p[0, node_ids].copy()
        y = p[1, node_ids].copy()
        chord = np.array([x[-1] - x[0], y[-1] - y[0]], dtype=float)
        t_ref = chord / float(np.linalg.norm(chord))
        n_ref = np.array([-t_ref[1], t_ref[0]], dtype=float)
    elif hasattr(cfg, "plate_tangent"):
        from compliance_fem.layered_geometry import ordered_interface_nodes

        # Use the same mesh that produced the factorization so DOF indices match.
        node_ids, x, y = ordered_interface_nodes(result.basis.mesh, cfg)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        t_ref = np.asarray(cfg.plate_tangent, dtype=float).reshape(2)
        n_ref = np.asarray(cfg.plate_normal, dtype=float).reshape(2)
    else:
        raise ValueError("Plate mesh extraction requires a plated (layered or measured) model.")
    if not np.allclose(x, result.x_plate, atol=1e-8) or not np.allclose(y, result.y_plate, atol=1e-8):
        raise ValueError(
            "Live plate coordinates disagree with ComplianceResult.x_plate/y_plate; "
            "refusing to recover plate fields from a mismatched mesh."
        )

    u_dofs, v_dofs = plate_translational_dofs(result.basis, node_ids)
    n_plate = int(u_dofs.size)
    n_foam = int(result.basis.N)
    n_theta = int(result.n_plate_rotation_dofs or 0)
    if n_theta not in (0, n_plate):
        raise ValueError(f"Unexpected n_plate_rotation_dofs={n_theta} for n_plate={n_plate}.")
    rotation_dofs = (
        np.arange(n_foam, n_foam + n_plate, dtype=int)
        if n_theta == n_plate
        else np.full(n_plate, -1, dtype=int)
    )
    n_constraints = n_plate - 1
    constraint_ids = np.arange(n_constraints, dtype=int)
    conn = np.column_stack([np.arange(n_constraints), np.arange(1, n_plate)])
    lengths = np.zeros(n_constraints, dtype=float)
    tangents = np.zeros((n_constraints, 2), dtype=float)
    normals = np.zeros((n_constraints, 2), dtype=float)
    t_cfg = t_ref
    n_cfg = n_ref
    for e in range(n_constraints):
        i, j = int(conn[e, 0]), int(conn[e, 1])
        delta = np.array([x[j] - x[i], y[j] - y[i]], dtype=float)
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise ValueError(f"Degenerate plate element {e}.")
        t_e = delta / length
        n_e = np.array([-t_e[1], t_e[0]], dtype=float)
        # Match assembly orientation: flip if opposite the config normal.
        if float(np.dot(n_e, n_cfg)) < 0.0:
            n_e = -n_e
            t_e = -t_e
        lengths[e] = length
        tangents[e] = t_e
        normals[e] = n_e

    return PlateMeshInfo(
        node_ids=np.asarray(node_ids, dtype=int),
        reference_x=x,
        reference_y=y,
        reference_arc_length=_arc_length(x, y),
        element_connectivity=conn,
        element_lengths=lengths,
        element_tangents=tangents,
        element_normals=normals,
        u_dof_ids=u_dofs,
        v_dof_ids=v_dofs,
        rotation_dof_ids=rotation_dofs,
        constraint_ids=constraint_ids,
        plate_tangent=t_cfg,
        plate_normal=n_cfg,
        EI_plate=float(cfg.EI_plate),
        n_foam=n_foam,
        n_primal=int(result.n_primal),
        n_lambda=int(result.n_lambda or n_constraints),
    )


def recover_primal_from_boundary_forces(
    result: ComplianceResult,
    F_t: np.ndarray,
    F_c: np.ndarray,
    contact_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover (q, lambda_p, alpha_rigid) from top and contact nodal forces.

    Uses the factorized augmented free-body operator; never inverts K.
    Contact forces are scattered onto the full bottom selector support.
    """
    if result.factorization is None or result.basis is None or result.n_primal is None:
        raise ValueError("ComplianceResult must retain factorization, basis, and n_primal.")
    n_q = int(result.n_primal)
    n_lambda = int(result.n_lambda or 0)
    n_extra = n_lambda + 3

    u_top, v_top = result.selector_dofs("top")
    u_bot, v_bot = result.selector_dofs("bottom")
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bot, v_bot, n_q)

    F_t = np.asarray(F_t, dtype=float).reshape(-1)
    F_c = np.asarray(F_c, dtype=float).reshape(-1)
    contact = np.asarray(contact_nodes, dtype=int)
    n_b = int(u_bot.size)
    F_bottom = np.zeros(2 * n_b, dtype=float)
    if contact.size:
        n_c = int(contact.size)
        if F_c.size != 2 * n_c:
            raise ValueError(f"F_c has length {F_c.size}; expected {2 * n_c}.")
        F_bottom[contact] = F_c[:n_c]
        F_bottom[contact + n_b] = F_c[n_c:]

    f = np.asarray(S_top.T @ F_t + S_bottom.T @ F_bottom, dtype=float).reshape(-1)
    rhs = np.concatenate([f, np.zeros(n_extra, dtype=float)])
    sol = np.asarray(result.factorization.solve(rhs), dtype=float).reshape(-1)
    q = sol[:n_q]
    lam = sol[n_q : n_q + n_lambda] if n_lambda else np.zeros(0, dtype=float)
    alpha = sol[n_q + n_lambda :]
    return q, lam, alpha


def axial_force_from_multiplier(lambda_p: np.ndarray) -> np.ndarray:
    """Convert raw B_p multipliers to axial forces.

    With unscaled rows ``t · (u_{j+1} - u_j)``, the multiplier already has force
    units. Both arrays are stored; this identity documents the convention.
    """
    return np.asarray(lambda_p, dtype=float).copy()


def extract_plate_nodal_fields(
    q: np.ndarray,
    lambda_p: np.ndarray,
    mesh: PlateMeshInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, theta, lambda_raw, axial_force) at plate nodes/elements."""
    q = np.asarray(q, dtype=float).reshape(-1)
    u = q[mesh.u_dof_ids]
    v = q[mesh.v_dof_ids]
    if np.all(mesh.rotation_dof_ids >= 0):
        theta = q[mesh.rotation_dof_ids]
    else:
        theta = np.zeros(mesh.n_nodes, dtype=float)
    lam = np.asarray(lambda_p, dtype=float).reshape(-1)
    if lam.size != mesh.n_elements:
        # Pad or trim defensively.
        out = np.zeros(mesh.n_elements, dtype=float)
        n = min(lam.size, mesh.n_elements)
        out[:n] = lam[:n]
        lam = out
    return u, v, theta, lam, axial_force_from_multiplier(lam)


def verify_plate_orientation(mesh: PlateMeshInfo, atol: float = 1e-10) -> None:
    """Assert postprocess normals/tangents match assembly signs."""
    for e in range(mesh.n_elements):
        t = mesh.element_tangents[e]
        n = mesh.element_normals[e]
        cross = t[0] * n[1] - t[1] * n[0]
        if abs(cross - 1.0) > 1e-6 and abs(cross + 1.0) > 1e-6:
            # Should be a positively oriented orthonormal pair (det = +1).
            if abs(abs(cross) - 1.0) > 1e-6:
                raise ValueError(f"Element {e} tangent/normal are not orthonormal (det={cross}).")
        if abs(cross - 1.0) > 1e-6:
            raise ValueError(
                f"Element {e} normal does not equal (-t_y, t_x); det(t,n)={cross}."
            )
    if mesh.n_elements:
        mean_n = np.mean(mesh.element_normals, axis=0)
        if float(np.linalg.norm(mean_n - mesh.plate_normal)) > 1e-6:
            # For a straight plate they must agree.
            if float(np.max(np.linalg.norm(mesh.element_normals - mesh.plate_normal, axis=1))) > 1e-6:
                pass  # element-specific directions are allowed for non-straight plates


def rigid_primal_from_alpha(
    result: ComplianceResult,
    mesh: PlateMeshInfo,
    alpha: np.ndarray,
    x_r: float,
    y_r: float,
) -> np.ndarray:
    """Continuum rigid displacement matching ``vector_rigid_mode_matrix`` about ``(x_r, y_r)``.

    Free-body force recovery returns the particular solution ``q = C F`` in the FEM
    gauge. Boundary solves then add rigid amplitudes ``Alpha`` so that
    ``C F + R Alpha = W``. This helper builds the matching primal field so plate
    recovery can reconstruct the full BC-satisfying response.
    """
    from compliance_fem.rigid_modes import _component_dofs

    if result.basis is None or result.n_primal is None:
        raise ValueError("ComplianceResult must retain basis and n_primal.")
    alpha = np.asarray(alpha, dtype=float).reshape(3)
    ax, ay, omega = float(alpha[0]), float(alpha[1]), float(alpha[2])
    q_r = np.zeros(int(result.n_primal), dtype=float)
    x_dofs = _component_dofs(result.basis, 0)
    y_dofs = _component_dofs(result.basis, 1)
    q_r[x_dofs] = ax - omega * (result.basis.doflocs[1, x_dofs] - float(y_r))
    q_r[y_dofs] = ay + omega * (result.basis.doflocs[0, y_dofs] - float(x_r))
    if np.all(mesh.rotation_dof_ids >= 0):
        # For θ = dw/ds with w = n·(u,v), a plane rigid rotation has θ ≡ ω.
        q_r[mesh.rotation_dof_ids] = omega
    return q_r


def recover_plate_basis_for_record(
    result: ComplianceResult,
    mesh: PlateMeshInfo,
    F_t: np.ndarray,
    F_c: np.ndarray,
    contact_nodes: np.ndarray,
    W_t: np.ndarray,
    W_c: np.ndarray,
    x_top: np.ndarray,
    x_contact: np.ndarray,
    Alpha: np.ndarray | None = None,
    x_r: float = 0.0,
    y_r: float = 0.0,
) -> PlateBasisFields:
    """Recover plate fields for one contact record by direct per-column solves.

    ``F_t``, ``F_c``, ``W_t``, ``W_c`` have shape ``(n_dof, n_cols)`` (six affine
    columns for schema v9+). When ``Alpha`` (shape ``(3, n_cols)``) is provided,
    the free-body particular solution is completed by the same rigid motion used
    in the boundary solve so that recovered boundary traces match ``W_t`` / ``W_c``.
    This is the reference path that :func:`plate_basis_from_influence` is checked against.
    """
    del x_top, x_contact  # reserved for callers / future residual diagnostics
    verify_plate_orientation(mesh)
    n_modes = int(np.asarray(F_t).shape[1])
    n_nodes = mesh.n_nodes
    n_el = mesh.n_elements
    u_b = np.zeros((n_modes, n_nodes))
    v_b = np.zeros((n_modes, n_nodes))
    th_b = np.zeros((n_modes, n_nodes))
    lam_b = np.zeros((n_modes, n_el))
    ax_b = np.zeros((n_modes, n_el))
    bp_res = np.zeros(n_modes)
    top_res = np.zeros(n_modes)
    contact_res = np.zeros(n_modes)

    u_top, v_top = result.selector_dofs("top")
    S_top = build_vector_selector(u_top, v_top, mesh.n_primal)
    u_bot, v_bot = result.selector_dofs("bottom")
    contact = np.asarray(contact_nodes, dtype=int)
    if contact.size:
        S_c = build_vector_selector(u_bot[contact], v_bot[contact], mesh.n_primal)
    else:
        S_c = sparse.csc_matrix((0, mesh.n_primal))

    B = result.B_p
    assert B is not None
    Alpha_arr = None if Alpha is None else np.asarray(Alpha, dtype=float)
    if Alpha_arr is not None and Alpha_arr.shape != (3, n_modes):
        raise ValueError(f"Alpha has shape {Alpha_arr.shape}; expected (3, {n_modes}).")

    for k in range(n_modes):
        q, lam, _alpha = recover_primal_from_boundary_forces(
            result, F_t[:, k], F_c[:, k] if F_c.size else np.zeros(0), contact
        )
        if Alpha_arr is not None:
            q = q + rigid_primal_from_alpha(result, mesh, Alpha_arr[:, k], x_r, y_r)
        u, v, th, lam_k, ax = extract_plate_nodal_fields(q, lam, mesh)
        u_b[k] = u
        v_b[k] = v
        th_b[k] = th
        lam_b[k] = lam_k
        ax_b[k] = ax
        bp_res[k] = float(np.linalg.norm(B @ q))
        top_disp = np.asarray(S_top @ q, dtype=float)
        top_res[k] = float(
            np.linalg.norm(top_disp - W_t[:, k]) / max(np.linalg.norm(W_t[:, k]), 1.0)
        )
        if contact.size:
            c_disp = np.asarray(S_c @ q, dtype=float)
            contact_res[k] = float(
                np.linalg.norm(c_disp - W_c[:, k])
                / max(np.linalg.norm(W_c[:, k]), np.linalg.norm(W_t[:, k]), 1.0)
            )

    return PlateBasisFields(
        u_local=u_b,
        v_local=v_b,
        rotation_local=th_b,
        constraint_multiplier=lam_b,
        axial_force=ax_b,
        bp_residual=bp_res,
        top_bc_residual=top_res,
        contact_bc_residual=contact_res,
    )


@dataclass
class PlateInfluence:
    """Plate fields per unit boundary load / unit rigid amplitude.

    Columns of the load blocks follow the stacked boundary load
    ``[F_t (2 n_t); F_bottom (2 n_b)]`` (component-major); the rigid blocks have
    the three columns of ``vector_rigid_mode_matrix`` about ``(x_r, y_r)``. All
    blocks come from one multi-RHS solve of the factorized augmented free-body
    operator, so per-record plate recovery is a pair of small matrix products.
    """

    n_top: int
    n_bottom: int
    u: np.ndarray  # (n_plate, 2 n_t + 2 n_b)
    v: np.ndarray
    theta: np.ndarray
    lam: np.ndarray  # (n_el, 2 n_t + 2 n_b)
    bp: np.ndarray  # (n_constraint_rows, 2 n_t + 2 n_b)
    top: np.ndarray  # (2 n_t, 2 n_t + 2 n_b)
    bottom: np.ndarray  # (2 n_b, 2 n_t + 2 n_b)
    rigid_u: np.ndarray  # (n_plate, 3)
    rigid_v: np.ndarray
    rigid_theta: np.ndarray
    rigid_bp: np.ndarray
    rigid_top: np.ndarray
    rigid_bottom: np.ndarray


def build_plate_influence(
    result: ComplianceResult,
    mesh: PlateMeshInfo,
    *,
    x_r: float,
    y_r: float,
) -> PlateInfluence:
    """Precompute plate influence matrices with a single multi-RHS factorized solve."""
    if result.factorization is None or result.basis is None or result.n_primal is None:
        raise ValueError("ComplianceResult must retain factorization, basis, and n_primal.")
    verify_plate_orientation(mesh)
    n_q = int(result.n_primal)
    n_lambda = int(result.n_lambda or 0)
    n_extra = n_lambda + 3
    u_top, v_top = result.selector_dofs("top")
    u_bot, v_bot = result.selector_dofs("bottom")
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bot, v_bot, n_q)
    loads = sparse.hstack([S_top.T, S_bottom.T]).tocsc()
    rhs = np.zeros((n_q + n_extra, loads.shape[1]), dtype=float)
    rhs[:n_q] = loads.toarray()
    sol = np.asarray(result.factorization.solve(rhs), dtype=float)
    q = sol[:n_q]
    lam_raw = sol[n_q : n_q + n_lambda] if n_lambda else np.zeros((0, loads.shape[1]))
    lam = np.zeros((mesh.n_elements, loads.shape[1]), dtype=float)
    n_copy = min(lam_raw.shape[0], mesh.n_elements)
    lam[:n_copy] = lam_raw[:n_copy]
    has_theta = bool(np.all(mesh.rotation_dof_ids >= 0))
    B = result.B_p
    assert B is not None

    q_rigid = np.column_stack(
        [rigid_primal_from_alpha(result, mesh, e, x_r, y_r) for e in np.eye(3)]
    )
    return PlateInfluence(
        n_top=int(u_top.size),
        n_bottom=int(u_bot.size),
        u=q[mesh.u_dof_ids],
        v=q[mesh.v_dof_ids],
        theta=q[mesh.rotation_dof_ids] if has_theta else np.zeros((mesh.n_nodes, q.shape[1])),
        lam=lam,
        bp=np.asarray(B @ q, dtype=float),
        top=np.asarray(S_top @ q, dtype=float),
        bottom=np.asarray(S_bottom @ q, dtype=float),
        rigid_u=q_rigid[mesh.u_dof_ids],
        rigid_v=q_rigid[mesh.v_dof_ids],
        rigid_theta=q_rigid[mesh.rotation_dof_ids] if has_theta else np.zeros((mesh.n_nodes, 3)),
        rigid_bp=np.asarray(B @ q_rigid, dtype=float),
        rigid_top=np.asarray(S_top @ q_rigid, dtype=float),
        rigid_bottom=np.asarray(S_bottom @ q_rigid, dtype=float),
    )


def plate_basis_from_influence(
    influence: PlateInfluence,
    F_t: np.ndarray,
    F_c: np.ndarray,
    contact_nodes: np.ndarray,
    Alpha: np.ndarray,
    W_t: np.ndarray,
    W_c: np.ndarray,
) -> PlateBasisFields:
    """Plate fields for one record from precomputed influence matrices (no solves)."""
    F_t = np.asarray(F_t, dtype=float)
    n_cols = F_t.shape[1]
    contact = np.asarray(contact_nodes, dtype=int)
    n_b = influence.n_bottom
    F_bottom = np.zeros((2 * n_b, n_cols), dtype=float)
    if contact.size:
        n_c = int(contact.size)
        F_bottom[contact] = F_c[:n_c]
        F_bottom[contact + n_b] = F_c[n_c:]
    loads = np.vstack([F_t, F_bottom])
    Al = np.asarray(Alpha, dtype=float)
    u = influence.u @ loads + influence.rigid_u @ Al
    v = influence.v @ loads + influence.rigid_v @ Al
    th = influence.theta @ loads + influence.rigid_theta @ Al
    lam = influence.lam @ loads
    bp = influence.bp @ loads + influence.rigid_bp @ Al
    top = influence.top @ loads + influence.rigid_top @ Al
    W_t = np.asarray(W_t, dtype=float)
    W_c = np.asarray(W_c, dtype=float)
    top_res = np.linalg.norm(top - W_t, axis=0) / np.maximum(np.linalg.norm(W_t, axis=0), 1.0)
    if contact.size:
        bottom = influence.bottom @ loads + influence.rigid_bottom @ Al
        rows = np.concatenate([contact, contact + n_b])
        denom = np.maximum.reduce(
            [np.linalg.norm(W_c, axis=0), np.linalg.norm(W_t, axis=0), np.ones(n_cols)]
        )
        contact_res = np.linalg.norm(bottom[rows] - W_c, axis=0) / denom
    else:
        contact_res = np.zeros(n_cols)
    return PlateBasisFields(
        u_local=u.T.copy(),
        v_local=v.T.copy(),
        rotation_local=th.T.copy(),
        constraint_multiplier=lam.T.copy(),
        axial_force=axial_force_from_multiplier(lam.T),
        bp_residual=np.linalg.norm(bp, axis=0),
        top_bc_residual=top_res,
        contact_bc_residual=contact_res,
    )


# ---------------------------------------------------------------------------
# Runtime superposition and Hermite rendering
# ---------------------------------------------------------------------------


COLOR_UNITS = {
    "none": "",
    "normal_displacement": "length",
    "tangential_displacement": "length",
    "rotation": "rad",
    "curvature": "1/length",
    "bending_moment": "force·length",
    "shear_force": "force",
    "axial_constraint_force": "force",
}


def nodal_tangential_normal(
    u: np.ndarray,
    v: np.ndarray,
    mesh: PlateMeshInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodal (u_s, w) in node-averaged element frames.

    Each node uses the normalized mean of its adjacent element tangents, which
    reduces to the global plate tangent for a straight plate.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if mesh.n_elements == 0:
        t = np.broadcast_to(mesh.plate_tangent, (u.size, 2))
    else:
        acc = np.zeros((mesh.n_nodes, 2), dtype=float)
        conn = mesh.element_connectivity
        np.add.at(acc, conn[:, 0], mesh.element_tangents)
        np.add.at(acc, conn[:, 1], mesh.element_tangents)
        t = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    u_s = t[:, 0] * u + t[:, 1] * v
    w = -t[:, 1] * u + t[:, 0] * v
    return u_s, w


def sample_plate_centerline(
    mesh: PlateMeshInfo,
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    axial_force: np.ndarray,
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT,
) -> dict[str, np.ndarray]:
    """Hermite-sample the plate centerline without duplicating shared nodes."""
    if samples_per_element < 2:
        raise ValueError("samples_per_element must be at least 2.")
    u_s_nodes, w_nodes = nodal_tangential_normal(u, v, mesh)
    EI = float(mesh.EI_plate)

    s_list: list[float] = []
    x_list: list[float] = []
    y_list: list[float] = []
    rot_list: list[float] = []
    curv_list: list[float] = []
    mom_list: list[float] = []
    shear_list: list[float] = []
    ax_list: list[float] = []
    w_list: list[float] = []
    us_list: list[float] = []
    eid_list: list[int] = []

    for e in range(mesh.n_elements):
        i = int(mesh.element_connectivity[e, 0])
        j = int(mesh.element_connectivity[e, 1])
        L_e = float(mesh.element_lengths[e])
        t_e = mesh.element_tangents[e]
        n_e = mesh.element_normals[e]
        # Endpoint values in the element frame.
        w_i = float(n_e[0] * u[i] + n_e[1] * v[i])
        w_j = float(n_e[0] * u[j] + n_e[1] * v[j])
        th_i = float(theta[i])
        th_j = float(theta[j])
        us_i = float(t_e[0] * u[i] + t_e[1] * v[i])
        us_j = float(t_e[0] * u[j] + t_e[1] * v[j])
        X_i = np.array([mesh.reference_x[i], mesh.reference_y[i]])
        # Interior samples only for e>0 at the left endpoint to avoid duplicates.
        n_samp = samples_per_element if e == 0 else samples_per_element - 1
        if e == 0:
            xi = np.linspace(0.0, 1.0, samples_per_element)
        else:
            xi = np.linspace(0.0, 1.0, samples_per_element)[1:]
        s_e = mesh.reference_arc_length[i] + xi * L_e
        w = hermite_deflection(xi, L_e, w_i, th_i, w_j, th_j)
        th = hermite_rotation(xi, L_e, w_i, th_i, w_j, th_j)
        kappa = hermite_curvature(xi, L_e, w_i, th_i, w_j, th_j)
        V = hermite_shear(L_e, w_i, th_i, w_j, th_j, EI)
        us = (1.0 - xi) * us_i + xi * us_j
        # Reference point along the element chord.
        X = X_i[None, :] + (xi * L_e)[:, None] * t_e[None, :]
        r = X + us[:, None] * t_e[None, :] + w[:, None] * n_e[None, :]
        s_list.extend(s_e.tolist())
        x_list.extend(r[:, 0].tolist())
        y_list.extend(r[:, 1].tolist())
        rot_list.extend(th.tolist())
        curv_list.extend(kappa.tolist())
        mom_list.extend((EI * kappa).tolist())
        shear_list.extend([V] * int(xi.size))
        ax_list.extend([float(axial_force[e])] * int(xi.size))
        w_list.extend(w.tolist())
        us_list.extend(us.tolist())
        eid_list.extend([e] * int(xi.size))

    return {
        "s": np.asarray(s_list, dtype=float),
        "x_local": np.asarray(x_list, dtype=float),
        "y_local": np.asarray(y_list, dtype=float),
        "rotation_local": np.asarray(rot_list, dtype=float),
        "curvature": np.asarray(curv_list, dtype=float),
        "bending_moment": np.asarray(mom_list, dtype=float),
        "shear_force": np.asarray(shear_list, dtype=float),
        "axial_force": np.asarray(ax_list, dtype=float),
        "w": np.asarray(w_list, dtype=float),
        "u_s": np.asarray(us_list, dtype=float),
        "element_id": np.asarray(eid_list, dtype=int),
        "u_s_nodes": u_s_nodes,
        "w_nodes": w_nodes,
    }


def build_plate_runtime_state(
    mesh: PlateMeshInfo,
    u_basis: np.ndarray,
    v_basis: np.ndarray,
    theta_basis: np.ndarray,
    lambda_basis: np.ndarray,
    axial_basis: np.ndarray,
    gamma: np.ndarray,
    *,
    anchor_x: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    elastic_scale: float = 1.0,
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT,
    color_quantity: str = "none",
    anchor_y: float = 0.0,
) -> PlateRuntimeState:
    """Contract plate basis fields and build Hermite fixed-frame samples."""
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    u = contract_basis(u_basis, gamma, mode_axis=0)
    v = contract_basis(v_basis, gamma, mode_axis=0)
    th = contract_basis(theta_basis, gamma, mode_axis=0)
    lam = contract_basis(lambda_basis, gamma, mode_axis=0)
    ax = contract_basis(axial_basis, gamma, mode_axis=0)
    u_s, w = nodal_tangential_normal(u, v, mesh)
    axial_diff = float(np.max(np.abs(np.diff(u_s)))) if u_s.size > 1 else 0.0
    # Constraint residual from nodal tangential differences (kinematic form).
    bp_terms = np.zeros(mesh.n_elements)
    for e in range(mesh.n_elements):
        i, j = mesh.element_connectivity[e]
        t = mesh.element_tangents[e]
        du = np.array([u[j] - u[i], v[j] - v[i]])
        bp_terms[e] = float(t @ du)
    bp_norm = float(np.linalg.norm(bp_terms))

    samples = sample_plate_centerline(mesh, u, v, th, ax, samples_per_element)
    # Reconstruct reference chord points and elastic displacements so the shared
    # foam/plate transform applies s_d only to the elastic part (never to varphi).
    x_ref = np.zeros_like(samples["x_local"])
    y_ref = np.zeros_like(samples["y_local"])
    for idx, e in enumerate(samples["element_id"]):
        i = int(mesh.element_connectivity[e, 0])
        L_e = float(mesh.element_lengths[e])
        t_e = mesh.element_tangents[e]
        s0 = float(mesh.reference_arc_length[i])
        xi = (float(samples["s"][idx]) - s0) / L_e if L_e > 0 else 0.0
        X_i = np.array([mesh.reference_x[i], mesh.reference_y[i]])
        X = X_i + xi * L_e * t_e
        x_ref[idx] = X[0]
        y_ref[idx] = X[1]
    u_samp = samples["x_local"] - x_ref
    v_samp = samples["y_local"] - y_ref
    x_fix, y_fix = transform_to_fixed_frame(
        x_ref,
        y_ref,
        u_samp,
        v_samp,
        float(anchor_x),
        float(d_ax),
        float(d_ay),
        float(varphi),
        elastic_scale=float(elastic_scale),
        y_anchor=float(anchor_y),
    )
    rot_fixed = np.asarray(samples["rotation_local"], dtype=float) + float(varphi)
    # Unwrap for continuous plotting.
    rot_fixed = np.unwrap(rot_fixed)

    color = str(color_quantity or "none").lower()
    color_map = {
        "none": None,
        "normal_displacement": samples["w"],
        "tangential_displacement": samples["u_s"],
        "rotation": samples["rotation_local"],
        "curvature": samples["curvature"],
        "bending_moment": samples["bending_moment"],
        "shear_force": samples["shear_force"],
        "axial_constraint_force": samples["axial_force"],
    }
    if color not in color_map:
        raise ValueError(f"Unknown plate color quantity '{color_quantity}'.")
    color_values = color_map[color]

    # Summary diagnostics.
    abs_M = np.abs(samples["bending_moment"])
    i_max = int(np.argmax(abs_M)) if abs_M.size else 0
    summary = {
        "max_abs_normal_displacement": float(np.max(np.abs(w))) if w.size else 0.0,
        "max_abs_tangential_displacement_variation": axial_diff,
        "max_abs_local_rotation": float(np.max(np.abs(th))) if th.size else 0.0,
        "max_abs_curvature": float(np.max(np.abs(samples["curvature"]))) if samples["curvature"].size else 0.0,
        "max_abs_bending_moment": float(np.max(abs_M)) if abs_M.size else 0.0,
        "max_abs_shear_force": float(np.max(np.abs(samples["shear_force"]))) if samples["shear_force"].size else 0.0,
        "max_abs_axial_constraint_force": float(np.max(np.abs(ax))) if ax.size else 0.0,
        "plate_constraint_residual": bp_norm,
        "max_moment_reference_s": float(samples["s"][i_max]) if samples["s"].size else float("nan"),
        "max_moment_local_x": float(samples["x_local"][i_max]) if samples["x_local"].size else float("nan"),
        "max_moment_local_y": float(samples["y_local"][i_max]) if samples["y_local"].size else float("nan"),
    }

    return PlateRuntimeState(
        mesh=mesh,
        u_local=u,
        v_local=v,
        rotation_local=th,
        constraint_multiplier=lam,
        axial_force=ax,
        u_tangential=u_s,
        w_normal=w,
        maximum_plate_axial_displacement_difference=axial_diff,
        plate_constraint_residual_norm=bp_norm,
        sample_s=samples["s"],
        sample_x_local=samples["x_local"],
        sample_y_local=samples["y_local"],
        sample_x_fixed=x_fix,
        sample_y_fixed=y_fix,
        sample_rotation_local=samples["rotation_local"],
        sample_rotation_fixed=rot_fixed,
        sample_curvature=samples["curvature"],
        sample_bending_moment=samples["bending_moment"],
        sample_shear_force=samples["shear_force"],
        sample_axial_force=samples["axial_force"],
        sample_w=samples["w"],
        sample_u_s=samples["u_s"],
        sample_element_id=samples["element_id"],
        color_quantity=color,
        color_values=None if color_values is None else np.asarray(color_values, dtype=float),
        color_units=COLOR_UNITS[color],
        summary=summary,
    )


def empty_plate_basis(n_nodes: int, n_elements: int) -> PlateBasisFields:
    """NaN plate basis for records when FEM recovery is unavailable."""
    C = N_AFFINE_COLUMNS
    return PlateBasisFields(
        u_local=np.full((C, n_nodes), np.nan),
        v_local=np.full((C, n_nodes), np.nan),
        rotation_local=np.full((C, n_nodes), np.nan),
        constraint_multiplier=np.full((C, n_elements), np.nan),
        axial_force=np.full((C, n_elements), np.nan),
        bp_residual=np.full(C, np.nan),
        top_bc_residual=np.full(C, np.nan),
        contact_bc_residual=np.full(C, np.nan),
    )


def plate_mesh_from_lookup(lookup: ContactLookupResult) -> PlateMeshInfo | None:
    """Reconstruct ``PlateMeshInfo`` from saved lookup arrays.

    Runtime Hermite sampling only needs geometry, ``EI_plate``, and
    tangents/normals. ``n_foam`` / ``n_primal`` are not required at runtime and
    default to zero when absent from ``plate_metadata``.
    """
    if not getattr(lookup, "has_plate_response", False):
        return None
    required = (
        lookup.plate_node_ids,
        lookup.plate_reference_x,
        lookup.plate_reference_y,
        lookup.plate_reference_arc_length,
        lookup.plate_element_connectivity,
        lookup.plate_element_lengths,
        lookup.plate_element_tangents,
        lookup.plate_element_normals,
        lookup.plate_u_dof_ids,
        lookup.plate_v_dof_ids,
        lookup.plate_rotation_dof_ids,
        lookup.plate_constraint_ids,
        lookup.plate_tangent,
        lookup.plate_normal,
    )
    if any(arr is None for arr in required) or not np.isfinite(lookup.EI_plate):
        return None

    meta = dict(getattr(lookup, "plate_metadata", None) or {})
    n_foam = int(meta.get("n_foam", 0) or 0)
    n_primal = int(meta.get("n_primal", 0) or 0)
    n_lambda = int(np.asarray(lookup.plate_constraint_ids).size)

    return PlateMeshInfo(
        node_ids=np.asarray(lookup.plate_node_ids, dtype=int),
        reference_x=np.asarray(lookup.plate_reference_x, dtype=float),
        reference_y=np.asarray(lookup.plate_reference_y, dtype=float),
        reference_arc_length=np.asarray(lookup.plate_reference_arc_length, dtype=float),
        element_connectivity=np.asarray(lookup.plate_element_connectivity, dtype=int),
        element_lengths=np.asarray(lookup.plate_element_lengths, dtype=float),
        element_tangents=np.asarray(lookup.plate_element_tangents, dtype=float),
        element_normals=np.asarray(lookup.plate_element_normals, dtype=float),
        u_dof_ids=np.asarray(lookup.plate_u_dof_ids, dtype=int),
        v_dof_ids=np.asarray(lookup.plate_v_dof_ids, dtype=int),
        rotation_dof_ids=np.asarray(lookup.plate_rotation_dof_ids, dtype=int),
        constraint_ids=np.asarray(lookup.plate_constraint_ids, dtype=int),
        plate_tangent=np.asarray(lookup.plate_tangent, dtype=float).reshape(2),
        plate_normal=np.asarray(lookup.plate_normal, dtype=float).reshape(2),
        EI_plate=float(lookup.EI_plate),
        n_foam=n_foam,
        n_primal=n_primal,
        n_lambda=n_lambda,
    )


def plate_state_for_selection(
    lookup: ContactLookupResult,
    selection: ContactSelection,
    *,
    elastic_scale: float = 1.0,
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT,
    color_quantity: str = "none",
) -> PlateRuntimeState | None:
    """Contract stored plate basis fields for the selected contact record."""
    if not getattr(lookup, "has_plate_response", False):
        return None
    row = getattr(selection, "selected_row", None)
    if row is None:
        return None
    if selection.d_ax is None or selection.d_ay is None or selection.anchor_x is None:
        return None
    mesh = plate_mesh_from_lookup(lookup)
    if mesh is None:
        return None
    row = int(row)
    try:
        f = lookup.record_fields([row], plate=True)
    except ValueError:
        return None

    gamma = basis_coefficients(
        selection.alpha, selection.d_ax, selection.d_ay, selection.varphi
    )
    return build_plate_runtime_state(
        mesh,
        f["plate_u_local"][0],
        f["plate_v_local"][0],
        f["plate_rotation_local"][0],
        f["plate_constraint_multiplier"][0],
        f["plate_axial_force"][0],
        gamma,
        anchor_x=float(selection.anchor_x),
        d_ax=float(selection.d_ax),
        d_ay=float(selection.d_ay),
        varphi=float(selection.varphi),
        elastic_scale=float(elastic_scale),
        samples_per_element=int(samples_per_element),
        color_quantity=str(color_quantity),
        anchor_y=float(getattr(selection, "anchor_y", 0.0) or 0.0),
    )
