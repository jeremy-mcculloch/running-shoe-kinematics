"""Reference implementations that tests check the runtime code against.

Each recomputes a quantity independently of the production path (direct
per-column plate solves, the analytic affine toe reduction, exact contact
kinematics, the interval enumeration order, toe-spring energy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse

from compliance_fem.contact.corotation import rotation_coefficients
from compliance_fem.contact.toe_spring import (
    COL_ALPHA,
    COL_BRX,
    COL_BRY,
    COL_BX,
    COL_BY,
    COL_CONST,
    ToeRoot,
    _affine_tables,
    spring_d2U_dalpha2,
    spring_energy,
)
from compliance_fem.fem.compliance import ComplianceResult
from compliance_fem.fem.plate_response import (
    PlateBasisFields,
    PlateMeshInfo,
    axial_force_from_multiplier,
    build_vector_selector,
    rigid_primal_from_alpha,
    verify_plate_orientation,
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


@dataclass(frozen=True)
class ToeAffineModel:
    """Force-controlled reduction for one record (per unit width).

    Local resultant ``F = K_d d + F_alpha alpha + F_r``; prescribing ``F = F*`` gives
    ``d(alpha) = d0 + d1 alpha`` and ``Q(alpha) = q0 + q1 alpha``.
    """

    K_d: np.ndarray
    F_alpha: np.ndarray
    F_r: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    q0: float
    q1: float
    kd_cond: float
    kd_smin: float
    well_conditioned: bool

    def displacement(self, alpha: float) -> np.ndarray:
        return self.d0 + self.d1 * float(alpha)

    def generalized_force(self, alpha):
        return self.q0 + self.q1 * np.asarray(alpha, dtype=float)


def toe_affine_model(
    scalars_row: np.ndarray,
    q_basis_row: np.ndarray,
    F_local_star: np.ndarray,
    r_x: float,
    r_y: float,
    *,
    fx_index: int = 0,
    fy_index: int = 1,
    cond_limit: float = 1.0e8,
    smin_threshold: float = 0.0,
) -> ToeAffineModel:
    """Build ``d0, d1, q0, q1`` analytically from the stored affine tables.

    The curved-sole closure (affine column 0) enters ``F_r`` and ``q0`` with
    coefficient 1. ``K_d`` is LU-factored once and both right-hand sides
    ``F* - F_r`` and ``-F_alpha`` are solved against it; no inverse is formed.
    """
    S, q = _affine_tables(scalars_row, q_basis_row)
    K_d = np.array(
        [
            [S[COL_BX, fx_index], S[COL_BY, fx_index]],
            [S[COL_BX, fy_index], S[COL_BY, fy_index]],
        ],
        dtype=float,
    )
    F_alpha = np.array([S[COL_ALPHA, fx_index], S[COL_ALPHA, fy_index]], dtype=float)
    F_r = (
        np.array([S[COL_CONST, fx_index], S[COL_CONST, fy_index]])
        + float(r_x) * np.array([S[COL_BRX, fx_index], S[COL_BRX, fy_index]])
        + float(r_y) * np.array([S[COL_BRY, fx_index], S[COL_BRY, fy_index]])
    )
    nan2 = np.full(2, np.nan)
    finite = (
        np.all(np.isfinite(K_d))
        and np.all(np.isfinite(F_alpha))
        and np.all(np.isfinite(F_r))
        and np.all(np.isfinite(q))
    )
    if not finite:
        return ToeAffineModel(K_d, F_alpha, F_r, nan2, nan2, np.nan, np.nan, np.inf, np.nan, False)
    svals = np.linalg.svd(K_d, compute_uv=False)
    smin = float(svals[-1])
    cond = float(svals[0] / smin) if smin > 0.0 else float("inf")
    if smin <= float(smin_threshold) or not np.isfinite(cond) or cond > float(cond_limit):
        return ToeAffineModel(K_d, F_alpha, F_r, nan2, nan2, np.nan, np.nan, cond, smin, False)
    lu_piv = linalg.lu_factor(K_d, check_finite=True)
    rhs = np.column_stack([np.asarray(F_local_star, dtype=float).reshape(2) - F_r, -F_alpha])
    D = linalg.lu_solve(lu_piv, rhs, check_finite=True)
    d0 = D[:, 0].copy()
    d1 = D[:, 1].copy()
    q_d = np.array([q[COL_BX], q[COL_BY]], dtype=float)
    q_r = q[COL_CONST] + float(r_x) * q[COL_BRX] + float(r_y) * q[COL_BRY]
    q0 = float(q_r + q_d @ d0)
    q1 = float(q[COL_ALPHA] + q_d @ d1)
    return ToeAffineModel(K_d, F_alpha, F_r, d0, d1, q0, q1, cond, smin, True)


def total_potential(alpha, q0: float, q1: float, k: float, theta0: float = 0.0):
    """Reduced potential with Pi'(alpha) = -g(alpha): U - (q0 alpha + q1 alpha^2 / 2)."""
    a = np.asarray(alpha, dtype=float)
    return spring_energy(a, k, theta0) - (float(q0) * a + 0.5 * float(q1) * a * a)


def potential_curvature(alpha, q1: float, k: float, theta0: float = 0.0):
    """Pi''(alpha) = d2U/dalpha2 - q1; stable when > 0."""
    return spring_d2U_dalpha2(alpha, k, theta0) - float(q1)


def relaxed_root_index(roots: tuple[ToeRoot, ...] | list[ToeRoot], theta0: float = 0.0) -> int | None:
    """Root on the branch connected to the unloaded neutral state (closest to theta0)."""
    if not roots:
        return None
    return int(np.argmin([abs(r.theta_rad - float(theta0)) for r in roots]))


def fixed_frame_normal_component(dx_local: np.ndarray, dy_local: np.ndarray, varphi: float) -> np.ndarray:
    """Ground-normal (fixed-frame y) component of a local vector.

    Used for both the signed nonpenetration gap and the compression-only
    normal reaction: ``sin(varphi) dx^T + cos(varphi) dy^T``.
    """
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    return s * np.asarray(dx_local, dtype=float) + c * np.asarray(dy_local, dtype=float)


def contact_displacement(
    x_contact: np.ndarray,
    x_anchor: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    y_contact: np.ndarray | None = None,
    y_anchor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact prescribed contact displacement in the rotating frame.

    ``d_c^T(x) = d_a^T + (Q(varphi)^T - I)(x - x_a, 0) + (0, -(y - y_a))``:

        u_c = d_ax + (cos(varphi) - 1)(x - x_anchor)
        v_c = d_ay - sin(varphi)(x - x_anchor) - (y - y_anchor)

    The last term is the curved-sole closure (zero for a flat sole); it is
    independent of ``varphi`` and is not rotated. The expression is identical
    for every interval, with no sign branching on the topology.

    The apparent infinitesimal axial strain in the ``u_c`` term is an accepted
    approximation of this project; it is not replaced by a small-angle form.
    """
    x = np.asarray(x_contact, dtype=float)
    ds = x - float(x_anchor)
    r_x, r_y = rotation_coefficients(varphi)
    u = float(d_ax) + r_x * ds
    v = float(d_ay) + r_y * ds
    if y_contact is not None:
        v = v - (np.asarray(y_contact, dtype=float) - float(y_anchor))
    return u, v


def interval_arrays(n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of every interval in enumeration order."""
    n = int(n_bottom)
    starts = np.concatenate([np.full(n - i, i, dtype=int) for i in range(n)])
    ends = np.concatenate([np.arange(i, n, dtype=int) for i in range(n)])
    return starts, ends
