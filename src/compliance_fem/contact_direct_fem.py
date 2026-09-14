"""Direct sparse FEM validation of contact-edge boundary solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skfem import condense

from compliance_fem.boundaries import split_uv, vector_boundary_data
from compliance_fem.compliance import ComplianceResult
from compliance_fem.contact_basis import (
    N_BASIS_MODES,
    build_contact_displacement_matrix,
    build_top_displacement_matrix,
)
from compliance_fem.contact_lookup import (
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    compute_toe_moment,
    from_compliance_result,
    solve_candidate,
)
from compliance_fem.contact_topology import (
    ContactRecordSpec,
    ContactType,
    anchor_reference_x,
)


@dataclass(frozen=True)
class DirectFEMComparison:
    """Comparison between compliance boundary solve and direct FEM."""

    contact_type: str
    candidate_index: int
    basis_index: int
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


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(np.linalg.norm(b), np.linalg.norm(a), 1e-30)
    return float(np.linalg.norm(a - b) / denom)


def solve_direct_fem_contact(
    result: ComplianceResult,
    spec: ContactRecordSpec,
    basis_index: int,
    a: float,
    kappa: float,
) -> dict[str, np.ndarray | float]:
    """Solve the sparse FEM BVP for one contact record and one basis mode.

    Prescribes the full vector top displacement ``(u_t, v_t)`` and the full
    vector contact displacement ``(u_c, v_c)`` for the requested basis mode,
    using the record's own contact set and basis anchor. Does not pin
    free-bottom or side horizontal DOFs. Layered results use ``K0`` plus
    ``B_p`` (no pin-on-K path).
    """
    if result.K is None or result.basis is None:
        raise ValueError("ComplianceResult must retain K and basis for direct FEM validation.")

    basis = result.basis
    K0 = result.K.tocsc()
    n_primal = int(result.n_primal if result.n_primal is not None else basis.N)
    u_top, v_top, x_top, y_top = vector_boundary_data(basis, "top")
    u_bot, v_bot, x_bottom, _y_bot = vector_boundary_data(basis, "bottom")
    n_t = int(x_top.size)
    n_b = int(x_bottom.size)
    free, contact = spec.sets(n_b)
    x_anchor = anchor_reference_x(spec, x_bottom, result.config.L)

    W = build_top_displacement_matrix(x_top, result.config.L, a, kappa)
    w_u, w_v = split_uv(W[:, basis_index], n_t)

    x0 = np.zeros(n_primal, dtype=float)
    x0[u_top] = w_u
    x0[v_top] = w_v
    if contact.size:
        W_c = build_contact_displacement_matrix(x_bottom[contact], x_anchor)
        c_u, c_v = split_uv(W_c[:, basis_index], int(contact.size))
        x0[u_bot[contact]] = c_u
        x0[v_bot[contact]] = c_v

    D = np.unique(
        np.concatenate(
            [
                u_top,
                v_top,
                u_bot[contact] if contact.size else np.array([], dtype=int),
                v_bot[contact] if contact.size else np.array([], dtype=int),
            ]
        )
    )

    if result.B_p is not None:
        B = result.B_p.tocsc()
        n_l = B.shape[0]
        A = sparse.bmat(
            [
                [K0, B.T],
                [B, sparse.csc_matrix((n_l, n_l))],
            ],
            format="csc",
        )
        x0_aug = np.zeros(n_primal + n_l, dtype=float)
        x0_aug[:n_primal] = x0
        A_cond, f_cond, x_out, I_out = condense(A, x=x0_aug, D=D)
        u_I = spsolve(A_cond.tocsc(), f_cond)
        x_full = x_out.copy()
        x_full[I_out] = u_I
        u = x_full[:n_primal]
        lam = x_full[n_primal:]
        residual = np.asarray(K0 @ u + B.T @ lam, dtype=float)
    else:
        K_cond, f_cond, x_out, I_out = condense(K0, x=x0, D=D)
        u_I = spsolve(K_cond.tocsc(), f_cond)
        u = x_out.copy()
        u[I_out] = u_I
        residual = np.asarray(K0 @ u, dtype=float)

    f_t = np.concatenate([residual[u_top], residual[v_top]])
    if contact.size:
        r_c = np.concatenate([residual[u_bot[contact]], residual[v_bot[contact]]])
    else:
        r_c = np.zeros(0, dtype=float)
    if free.size:
        g_f = np.concatenate([u[u_bot[free]], u[v_bot[free]]])
        r_free = np.concatenate([residual[u_bot[free]], residual[v_bot[free]]])
    else:
        g_f = np.zeros(0, dtype=float)
        r_free = np.zeros(0, dtype=float)

    f_tx = residual[u_top]
    f_ty = residual[v_top]
    Fy = float(np.sum(f_ty))
    M = float(np.dot(x_top, f_ty))
    T_toe, _T_vert, _H_a = compute_toe_moment(f_tx, f_ty, x_top, y_top, a)
    return {
        "u": u,
        "f_t": np.asarray(f_t, dtype=float),
        "r_c": np.asarray(r_c, dtype=float),
        "g_f": np.asarray(g_f, dtype=float),
        "r_free": np.asarray(r_free, dtype=float),
        "Fy": Fy,
        "M": M,
        "T_toe": float(T_toe),
        "free": free,
        "contact": contact,
    }


def compare_compliance_to_direct_fem(
    result: ComplianceResult,
    spec: ContactRecordSpec,
    basis_index: int,
    a: float,
    kappa: float,
) -> DirectFEMComparison:
    """Compare compliance boundary solution against direct FEM for one record."""
    blocks = from_compliance_result(result)
    W = build_top_displacement_matrix(blocks.x_top, blocks.L, a, kappa)
    x_r = 0.5 * blocks.L
    sol = solve_candidate(blocks, spec, W, x_r, a=a)
    fem = solve_direct_fem_contact(result, spec, basis_index, a, kappa)

    f_t_c = np.asarray(sol["F_t"])[:, basis_index]
    r_c_c = np.asarray(sol["F_c"])[:, basis_index]
    g_f_c = np.asarray(sol["G_f"])[:, basis_index] if sol["n_free"] else np.zeros(0)

    f_t_f = np.asarray(fem["f_t"])
    r_c_f = np.asarray(fem["r_c"])
    g_f_f = np.asarray(fem["g_f"])
    r_free = np.asarray(fem["r_free"])

    Fy_c = float(sol["scalars"][basis_index, SCALAR_FY])
    M_c = float(sol["scalars"][basis_index, SCALAR_MV])
    T_c = float(sol["scalars"][basis_index, SCALAR_TOE])
    Fy_f = float(fem["Fy"])
    M_f = float(fem["M"])
    T_f = float(fem["T_toe"])
    free_trac = float(np.linalg.norm(r_free) / max(np.linalg.norm(f_t_f), 1.0))

    return DirectFEMComparison(
        contact_type=spec.contact_type.value,
        candidate_index=-1 if spec.edge_node_id is None else int(spec.edge_node_id),
        basis_index=basis_index,
        top_force_rel_error=_relative_error(f_t_c, f_t_f),
        contact_reaction_rel_error=_relative_error(r_c_c, r_c_f),
        free_gap_rel_error=_relative_error(g_f_c, g_f_f),
        Fy_rel_error=abs(Fy_c - Fy_f) / max(abs(Fy_f), abs(Fy_c), 1e-30),
        M_rel_error=abs(M_c - M_f) / max(abs(M_f), abs(M_c), 1e-30),
        T_toe_rel_error=abs(T_c - T_f) / max(abs(T_f), abs(T_c), 1e-30),
        f_t_compliance=f_t_c,
        f_t_fem=f_t_f,
        r_c_compliance=r_c_c,
        r_c_fem=r_c_f,
        g_f_compliance=g_f_c,
        g_f_fem=g_f_f,
        free_traction_rel=free_trac,
    )


def pick_candidate_near(x_bottom: np.ndarray, target: float, interior_only: bool = True) -> int:
    """Return bottom index nearest to target x, optionally excluding endpoints."""
    if interior_only:
        indices = np.arange(1, len(x_bottom) - 1)
    else:
        indices = np.arange(len(x_bottom))
    return int(indices[np.argmin(np.abs(x_bottom[indices] - target))])


def standard_comparison_specs(x_bottom: np.ndarray, L: float) -> tuple[ContactRecordSpec, ...]:
    """One representative record per topology: heel near L/4, toe near 3L/4, full."""
    return (
        ContactRecordSpec(ContactType.HEEL, pick_candidate_near(x_bottom, 0.25 * L)),
        ContactRecordSpec(ContactType.TOE, pick_candidate_near(x_bottom, 0.75 * L)),
        ContactRecordSpec(ContactType.FULL, None),
    )


def run_standard_direct_comparisons(
    result: ComplianceResult,
    a: float,
    kappa: float,
) -> list[DirectFEMComparison]:
    """Compare heel, toe, and full contact against direct FEM for all five modes."""
    specs = standard_comparison_specs(result.x_bottom, result.config.L)
    return [
        compare_compliance_to_direct_fem(result, spec, k, a, kappa)
        for spec in specs
        for k in range(N_BASIS_MODES)
    ]
