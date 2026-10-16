"""Direct sparse FEM reference for continuous contact states (test-only).

Given the top shape amplitude ``alpha`` and the local bottom nodal forces
``f_b`` of a contact state, the full sparse system ``K u = f`` (plus the plate
inextensibility constraints ``B_p``) is solved with the top selector clamped to
``alpha * w_alpha``. It shares no code with the condensed boundary operators
except the assembled stiffness, so it independently checks ``G_bb``,
``d_alpha``, the top forces, the full-field (render) recovery and the plate
recovery. Must not be imported from ``src/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skfem import condense

from compliance_fem.fem.compliance import ComplianceResult

_Condensed = tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class DirectState:
    u: np.ndarray  # full primal vector (foam dofs + plate rotation dofs)
    multipliers: np.ndarray  # B_p multipliers (axial constraint forces)
    top_force: np.ndarray  # (2 n_t,) reaction on the clamped top = foot-on-shoe load
    bottom_displacement: np.ndarray  # (2 n_b,) component-major


def solve_top_clamped(result: ComplianceResult, w_top: np.ndarray, f_bottom: np.ndarray) -> DirectState:
    """Solve ``K u = S_b^T f_b`` with the top selector prescribed to ``w_top`` (component-major)."""
    if result.K is None or result.basis is None:
        raise ValueError("ComplianceResult must retain K and basis.")
    K0 = result.K.tocsc()
    n_q = int(result.n_primal if result.n_primal is not None else result.basis.N)
    u_top, v_top = result.selector_dofs("top")
    u_bot, v_bot = result.selector_dofs("bottom")
    n_t, n_b = u_top.size, u_bot.size
    w = np.asarray(w_top, dtype=float)
    f_b = np.asarray(f_bottom, dtype=float)
    x0 = np.zeros(n_q)
    x0[u_top] = w[:n_t]
    x0[v_top] = w[n_t:]
    f = np.zeros(n_q)
    np.add.at(f, u_bot, f_b[:n_b])
    np.add.at(f, v_bot, f_b[n_b:])
    D = np.unique(np.concatenate([u_top, v_top]))
    if result.B_p is not None:
        B = result.B_p.tocsc()
        n_l = B.shape[0]
        A = sparse.bmat([[K0, B.T], [B, sparse.csc_matrix((n_l, n_l))]], format="csc")
        rhs = np.concatenate([f, np.zeros(n_l)])
        x0a = np.concatenate([x0, np.zeros(n_l)])
        A_c, f_c, x_out, I_out = cast(_Condensed, condense(A, rhs, x=x0a, D=D))
        x = x_out.copy()
        x[I_out] = spsolve(A_c.tocsc(), f_c)
        u, lam = x[:n_q], x[n_q:]
        residual = np.asarray(K0 @ u + B.T @ lam - f, dtype=float)
    else:
        K_c, f_c, x_out, I_out = cast(_Condensed, condense(K0, f, x=x0, D=D))
        u = x_out.copy()
        u[I_out] = spsolve(K_c.tocsc(), f_c)
        lam = np.zeros(0)
        residual = np.asarray(K0 @ u - f, dtype=float)
    return DirectState(
        u=u,
        multipliers=lam,
        top_force=np.concatenate([residual[u_top], residual[v_top]]),
        bottom_displacement=np.concatenate([u[u_bot], u[v_bot]]),
    )


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.linalg.norm(a), 1e-300))
