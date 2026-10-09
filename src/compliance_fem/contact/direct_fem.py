"""Direct sparse FEM validation of interval-contact boundary solutions.

For one contact interval ``I_ij`` the direct solve prescribes the top
displacement and the contact-node displacement of one affine column (or of a
combined runtime state) on the full sparse FEM system, leaves every free bottom
node unconstrained, and recovers nodal reactions from the residual. It shares no
code path with the boundary-compliance solve except the assembled stiffness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skfem import condense

from compliance_fem.fem.boundaries import split_uv
from compliance_fem.fem.compliance import ComplianceResult
from compliance_fem.contact.basis import (
    AFFINE_COLUMN_NAMES,
    N_AFFINE_COLUMNS,
    build_contact_affine_matrix,
    build_top_affine_matrix,
)
from compliance_fem.contact.lookup import (
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    CandidateSolution,
    compute_toe_moment,
    get_compliance_block_matrix,
    solve_candidate,
)
from compliance_fem.contact.topology import ContactInterval, interval_anchor_x, interval_anchor_y
from compliance_fem.contact.corotation import affine_coefficients


@dataclass(frozen=True)
class DirectFEMComparison:
    """Comparison between the compliance boundary solve and direct FEM for one column."""

    contact_type: str
    contact_start_index: int
    contact_end_index: int
    column_index: int
    column_name: str
    top_force_rel_error: float
    contact_reaction_rel_error: float
    free_gap_rel_error: float
    Fy_rel_error: float
    M_rel_error: float
    T_toe_rel_error: float
    f_t_compliance: np.ndarray
    f_t_fem: np.ndarray
    r_c_compliance: np.ndarray
    r_c_fem: np.ndarray
    g_f_compliance: np.ndarray
    g_f_fem: np.ndarray
    free_traction_rel: float

    @property
    def max_rel_error(self) -> float:
        return float(
            max(self.top_force_rel_error, self.contact_reaction_rel_error, self.free_gap_rel_error)
        )


@dataclass(frozen=True)
class DirectFEMSolution:
    """Direct sparse FEM solution of one contact interval.

    Vector quantities are component-major (all u then all v): ``f_t`` top
    reactions, ``r_c`` contact reactions, ``g_f`` free-node displacements and
    ``r_free`` free-node residual tractions.
    """

    u: np.ndarray
    f_t: np.ndarray
    r_c: np.ndarray
    g_f: np.ndarray
    r_free: np.ndarray
    Fy: float
    M: float
    Mz: float
    T_toe: float
    free: np.ndarray
    contact: np.ndarray
    x_anchor: float
    y_anchor: float


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    denom = max(np.linalg.norm(b), np.linalg.norm(a), 1e-30)
    return float(np.linalg.norm(a - b) / denom)


def solve_direct_fem_contact(
    result: ComplianceResult,
    spec: ContactInterval,
    column_index: int,
    toe_length: float,
    kappa: float,
    gamma: np.ndarray | None = None,
) -> DirectFEMSolution:
    """Solve the sparse FEM BVP for one contact interval and one affine column.

    Prescribes the full vector top displacement and the full vector contact
    displacement of affine column ``column_index`` (0 = curved-sole closure,
    1..5 = ``top_shape_alpha`` .. ``contact_rotation_y``), using the interval's
    own contact set and midpoint anchor. With ``gamma`` (length 5) the combined
    state ``z_0 + sum_k gamma_k z_k`` is prescribed instead. The operator is
    ``K0`` plus the plate constraint ``B_p``.
    """
    if result.K is None or result.basis is None:
        raise ValueError("ComplianceResult must retain K and basis for direct FEM validation.")

    basis = result.basis
    K0 = result.K.tocsc()
    n_primal = int(result.n_primal if result.n_primal is not None else basis.N)
    u_top, v_top = result.selector_dofs("top")
    u_bot, v_bot = result.selector_dofs("bottom")
    x_top = np.asarray(result.x_top, dtype=float)
    y_top = np.asarray(result.y_top, dtype=float)
    x_bottom = np.asarray(result.x_bottom, dtype=float)
    y_bottom = np.asarray(result.y_bottom, dtype=float)
    n_t = int(x_top.size)
    n_b = int(x_bottom.size)
    if spec.n_bottom != n_b:
        raise ValueError(f"Interval built for N_b={spec.n_bottom}; mesh has N_b={n_b}.")
    free, contact = spec.sets()
    x_a = interval_anchor_x(x_bottom, spec.start, spec.end)
    y_a = interval_anchor_y(x_bottom, y_bottom, x_a)

    coeff = (
        np.eye(N_AFFINE_COLUMNS)[int(column_index)]
        if gamma is None
        else affine_coefficients(np.asarray(gamma, dtype=float))
    )
    W = build_top_affine_matrix(x_top, result.config.L, toe_length, kappa)
    w_u, w_v = split_uv(W @ coeff, n_t)
    x0 = np.zeros(n_primal, dtype=float)
    x0[u_top] = w_u
    x0[v_top] = w_v
    W_c = build_contact_affine_matrix(x_bottom[contact], x_a, y_bottom[contact], y_a)
    c_u, c_v = split_uv(W_c @ coeff, int(contact.size))
    x0[u_bot[contact]] = c_u
    x0[v_bot[contact]] = c_v

    D = np.unique(np.concatenate([u_top, v_top, u_bot[contact], v_bot[contact]]))

    if result.B_p is not None:
        B = result.B_p.tocsc()
        n_l = B.shape[0]
        A = sparse.bmat([[K0, B.T], [B, sparse.csc_matrix((n_l, n_l))]], format="csc")
        x0_aug = np.zeros(n_primal + n_l, dtype=float)
        x0_aug[:n_primal] = x0
        A_cond, f_cond, x_out, I_out = condense(A, x=x0_aug, D=D)
        x_full = x_out.copy()
        x_full[I_out] = spsolve(A_cond.tocsc(), f_cond)
        u = x_full[:n_primal]
        residual = np.asarray(K0 @ u + B.T @ x_full[n_primal:], dtype=float)
    else:
        K_cond, f_cond, x_out, I_out = condense(K0, x=x0, D=D)
        u = x_out.copy()
        u[I_out] = spsolve(K_cond.tocsc(), f_cond)
        residual = np.asarray(K0 @ u, dtype=float)

    f_t = np.concatenate([residual[u_top], residual[v_top]])
    r_c = np.concatenate([residual[u_bot[contact]], residual[v_bot[contact]]])
    g_f = np.concatenate([u[u_bot[free]], u[v_bot[free]]]) if free.size else np.zeros(0)
    r_free = np.concatenate([residual[u_bot[free]], residual[v_bot[free]]]) if free.size else np.zeros(0)

    f_tx = residual[u_top]
    f_ty = residual[v_top]
    T_toe, _T_vert, _H_mtp = compute_toe_moment(f_tx, f_ty, x_top, y_top, result.config.L - toe_length)
    return DirectFEMSolution(
        u=u,
        f_t=np.asarray(f_t, dtype=float),
        r_c=np.asarray(r_c, dtype=float),
        g_f=np.asarray(g_f, dtype=float),
        r_free=np.asarray(r_free, dtype=float),
        Fy=float(np.sum(f_ty)),
        M=float(np.dot(x_top, f_ty)),
        Mz=float(np.dot(x_top, f_ty) - np.dot(y_top, f_tx)),
        T_toe=float(T_toe),
        free=free,
        contact=contact,
        x_anchor=x_a,
        y_anchor=y_a,
    )


def compare_compliance_to_direct_fem(
    result: ComplianceResult,
    spec: ContactInterval,
    column_index: int,
    toe_length: float,
    kappa: float,
    *,
    blocks=None,
    sol: CandidateSolution | None = None,
) -> DirectFEMComparison:
    """Compare the compliance boundary solution with direct FEM for one interval column."""
    if blocks is None:
        blocks = get_compliance_block_matrix(result)
    if sol is None:
        W = build_top_affine_matrix(blocks.x_top, blocks.L, toe_length, kappa)
        sol = solve_candidate(blocks, spec, W, 0.5 * blocks.L, toe_length=toe_length)
    if not sol.valid:
        raise ValueError(f"Interval {spec.start}..{spec.end} rejected: {sol.rejection_reason}")
    k = int(column_index)
    fem = solve_direct_fem_contact(result, spec, k, toe_length, kappa)

    f_t_c = sol.F_t[:, k]
    r_c_c = sol.F_c[:, k]
    g_f_c = sol.G_f[:, k] if sol.n_free else np.zeros(0)
    f_t_f, r_c_f, g_f_f, r_free = fem.f_t, fem.r_c, fem.g_f, fem.r_free

    Fy_c = float(sol.scalars[k, SCALAR_FY])
    M_c = float(sol.scalars[k, SCALAR_MV])
    T_c = float(sol.scalars[k, SCALAR_TOE])
    Fy_f, M_f, T_f = fem.Fy, fem.M, fem.T_toe
    # Resultants that vanish by symmetry are normalized by the total nodal force.
    scale = max(float(np.sum(np.abs(f_t_f))), 1e-30)

    def scalar_err(c: float, f: float, ref: float) -> float:
        return abs(c - f) / max(abs(f), abs(c), ref, 1e-30)

    return DirectFEMComparison(
        contact_type=spec.topology_label.value,
        contact_start_index=int(spec.start),
        contact_end_index=int(spec.end),
        column_index=k,
        column_name=AFFINE_COLUMN_NAMES[k],
        top_force_rel_error=_relative_error(f_t_c, f_t_f),
        contact_reaction_rel_error=_relative_error(r_c_c, r_c_f),
        free_gap_rel_error=_relative_error(g_f_c, g_f_f),
        Fy_rel_error=scalar_err(Fy_c, Fy_f, scale),
        M_rel_error=scalar_err(M_c, M_f, scale * blocks.L),
        T_toe_rel_error=scalar_err(T_c, T_f, scale * blocks.L),
        f_t_compliance=f_t_c,
        f_t_fem=f_t_f,
        r_c_compliance=r_c_c,
        r_c_fem=r_c_f,
        g_f_compliance=g_f_c,
        g_f_fem=g_f_f,
        free_traction_rel=float(np.linalg.norm(r_free) / max(np.linalg.norm(f_t_f), 1.0)),
    )


def pick_candidate_near(x_bottom: np.ndarray, target: float, interior_only: bool = True) -> int:
    """Return bottom index nearest to target x, optionally excluding endpoints."""
    indices = np.arange(1, len(x_bottom) - 1) if interior_only else np.arange(len(x_bottom))
    return int(indices[np.argmin(np.abs(x_bottom[indices] - target))])


def standard_validation_intervals(x_bottom: np.ndarray, L: float) -> dict[str, ContactInterval]:
    """Representative intervals: heel, toe, full, symmetric interior, asymmetric
    off-centre interior, one free node on each side, and a one-node interval."""
    n = len(x_bottom)
    i_q1 = pick_candidate_near(x_bottom, 0.25 * L)
    i_q3 = pick_candidate_near(x_bottom, 0.75 * L)
    i_c = pick_candidate_near(x_bottom, 0.5 * L)
    i_a0 = pick_candidate_near(x_bottom, 0.55 * L)
    i_a1 = pick_candidate_near(x_bottom, 0.85 * L)
    return {
        "flat_heel": ContactInterval(0, i_q1, n),
        "flat_toe": ContactInterval(i_q3, n - 1, n),
        "full": ContactInterval(0, n - 1, n),
        "symmetric_interior": ContactInterval(i_q1, n - 1 - i_q1, n),
        "asymmetric_interior": ContactInterval(i_a0, max(i_a1, i_a0), n),
        "one_free_node_each_side": ContactInterval(1, n - 2, n),
        "single_node": ContactInterval(i_c, i_c, n),
    }


def run_standard_direct_comparisons(
    result: ComplianceResult,
    toe_length: float,
    kappa: float,
    intervals: dict[str, ContactInterval] | None = None,
) -> dict[str, list[DirectFEMComparison]]:
    """Compare each validation interval against direct FEM for all six affine columns."""
    blocks = get_compliance_block_matrix(result)
    W = build_top_affine_matrix(blocks.x_top, blocks.L, toe_length, kappa)
    specs = intervals or standard_validation_intervals(result.x_bottom, result.config.L)
    out: dict[str, list[DirectFEMComparison]] = {}
    for name, spec in specs.items():
        sol = solve_candidate(blocks, spec, W, 0.5 * blocks.L, toe_length=toe_length)
        out[name] = [
            compare_compliance_to_direct_fem(result, spec, k, toe_length, kappa, blocks=blocks, sol=sol)
            for k in range(N_AFFINE_COLUMNS)
        ]
    return out
