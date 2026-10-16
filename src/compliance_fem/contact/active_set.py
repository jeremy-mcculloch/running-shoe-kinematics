"""Warm-started primal-dual active-set solver for the force-controlled contact LCP.

Problem (``n`` potential contact nodes)::

    g = q + t 1 + H lambda,      0 <= lambda  perp  g >= 0,      1^T lambda = Lambda,

with ``Lambda > 0`` the prescribed total normal force and ``t`` a free scalar
(the rigid normal translation; it is the Lagrange multiplier of the force
constraint). For symmetric positive definite ``H`` these are exactly the KKT
conditions of ``min 1/2 lambda^T H lambda + q^T lambda  s.t.  lambda >= 0,
1^T lambda = Lambda``; for a nonsymmetric ``H`` (proportional shear) it is a
mixed LCP with the same active-set structure.

For an active set ``A`` the equality block

    [ H_AA  1 ] [lambda_A]   [-q_A  ]
    [ 1^T   0 ] [   t    ] = [Lambda]

is LU-factored (never inverted) and ``g_I = q_I + t + H_IA lambda_A``.

Iteration: primal-dual active-set (PDAS) block updates, drop active nodes with
``lambda < -tau_lambda`` and add inactive nodes with ``g < -tau_g``. If an
active set repeats (cycling) the solver switches to Murty's least-index single
pivot rule, which terminates finitely for P-matrices. Cost per iteration is
``O(a^3 + n a)``; the factorization is a refactorization of the ``a + 1``
block (:class:`ActiveBlockSolver` keeps the API open for rank-one updates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

STATUS_CONVERGED = "converged"
STATUS_MAX_ITER = "max_active_set_iterations"
STATUS_SINGULAR = "singular_active_block"
STATUS_NO_LOAD = "no_normal_load"


class ActiveBlockSolver:
    """Solves the ``(a + 1)`` KKT block of one active set by LU factorization.

    Refactors when the active set changes. ``factorizations`` counts them so a
    future rank-one Cholesky up/downdate can replace :meth:`_factor` without
    changing callers.
    """

    def __init__(self, H: np.ndarray) -> None:
        self.H = np.asarray(H, dtype=float)
        self.factorizations = 0
        self.last_rcond = float("nan")

    def _factor(self, idx: np.ndarray):
        a = idx.size
        K = np.empty((a + 1, a + 1))
        K[:a, :a] = self.H[np.ix_(idx, idx)]
        K[:a, a] = 1.0
        K[a, :a] = 1.0
        K[a, a] = 0.0
        lu = linalg.lu_factor(K, check_finite=False)
        rcond, _ = linalg.lapack.dgecon(lu[0], float(np.linalg.norm(K, 1)), norm="1")
        self.last_rcond = float(rcond)
        self.factorizations += 1
        return lu

    def solve(self, idx: np.ndarray, q: np.ndarray, total: float) -> tuple[np.ndarray, float]:
        lu = self._factor(idx)
        rhs = np.concatenate([-q[idx], [float(total)]])
        x = linalg.lu_solve(lu, rhs, check_finite=False)
        return x[:-1], float(x[-1])


@dataclass
class MixedLCPResult:
    lam: np.ndarray
    t: float
    gap: np.ndarray
    active: np.ndarray  # bool (n,)
    iterations: int
    status: str
    pivot_rule: str
    factorizations: int
    rcond: float
    history: list = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status in (STATUS_CONVERGED, STATUS_NO_LOAD)


def solve_mixed_lcp(
    H: np.ndarray,
    q: np.ndarray,
    total: float,
    *,
    active0: np.ndarray | None = None,
    gap_tol: float,
    reaction_tol: float,
    max_iter: int = 200,
    rcond_min: float = 1.0e-14,
) -> MixedLCPResult:
    """Solve ``0 <= lambda perp q + t 1 + H lambda >= 0`` with ``sum(lambda) = total``.

    ``active0`` (bool mask) warm-starts the active set; empty or ``None`` starts
    from the node with the smallest free gap ``q``. ``total <= 0`` returns the
    unloaded state ``lambda = 0``, ``t = -min(q)`` (the sole just touches).
    """
    H = np.asarray(H, dtype=float)
    q = np.asarray(q, dtype=float)
    n = q.size
    if not total > 0.0:
        t = -float(np.min(q))
        g = q + t
        act = np.zeros(n, dtype=bool)
        return MixedLCPResult(np.zeros(n), t, g, act, 0, STATUS_NO_LOAD, "none", 0, float("nan"))

    A = np.zeros(n, dtype=bool) if active0 is None else np.asarray(active0, dtype=bool).copy()
    if not A.any():
        A[int(np.argmin(q))] = True
    solver = ActiveBlockSolver(H)
    seen: set[bytes] = set()
    rule = "pdas"
    lam = np.zeros(n)
    t = 0.0
    g = q.copy()
    history: list = []
    for it in range(1, int(max_iter) + 1):
        idx = np.flatnonzero(A)
        lam_A, t = solver.solve(idx, q, total)
        if not np.all(np.isfinite(lam_A)) or not solver.last_rcond > rcond_min:
            return MixedLCPResult(lam, t, g, A, it, STATUS_SINGULAR, rule, solver.factorizations, solver.last_rcond, history)
        lam = np.zeros(n)
        lam[idx] = lam_A
        # Active gaps are zero up to the block-solve roundoff, which is kept (not zeroed)
        # so the reported complementarity residual is honest.
        g = q + t + H[:, idx] @ lam_A
        drop = A & (lam < -reaction_tol)
        add = ~A & (g < -gap_tol)
        history.append(int(idx.size))
        if not (drop.any() or add.any()):
            return MixedLCPResult(lam, t, g, A, it, STATUS_CONVERGED, rule, solver.factorizations, solver.last_rcond, history)
        seen.add(A.tobytes())
        if rule == "pdas":
            A_new = (A & ~drop) | add
            if not A_new.any():
                A_new[int(np.argmin(g))] = True
            if A_new.tobytes() in seen:
                rule = "least_index"
            else:
                A = A_new
                continue
        # Murty least-index rule: flip the smallest-index violator only.
        k = int(np.flatnonzero(drop | add)[0])
        A = A.copy()
        A[k] = not A[k]
        if not A.any():
            A[int(np.argmin(g))] = True
    return MixedLCPResult(lam, t, g, A, int(max_iter), STATUS_MAX_ITER, rule, solver.factorizations, solver.last_rcond, history)
