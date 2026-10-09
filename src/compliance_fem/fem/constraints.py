"""Exact axial inextensibility constraints for a small-strain plate.

Each neighbouring pair of plate nodes contributes one independent row

    t_e · (u_{j+1} - u_j) = 0,

with ``t_e`` the element's own heel-to-toe unit tangent (one global ``t_p``
for a straight plate).

Rows are stored unscaled, matching the discrete kinematic constraint exactly.
The columns corresponding to plate rotation DOFs are identically zero.
Constant tangential translation of the plate lies in the constraint nullspace
and is not pinned.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def assemble_axial_constraints(
    n_q: int,
    u_dofs: np.ndarray,
    v_dofs: np.ndarray,
    t_p: np.ndarray,
    n_theta: int,
) -> sparse.csc_matrix:
    """Assemble B_p such that B_p q = 0 enforces plate inextensibility.

    Parameters
    ----------
    n_q
        Enlarged primal dimension (continuum + optional plate rotations).
    u_dofs, v_dofs
        Continuum displacement DOFs at ordered plate nodes (heel to toe).
    t_p
        Unit plate tangent (2,) for a straight plate, or one unit tangent per
        element (n_plate - 1, 2) for a curved plate.
    n_theta
        Number of plate rotation DOFs appended after the continuum block.
        Rotation columns of B_p are zero.
    """
    u_dofs = np.asarray(u_dofs, dtype=int)
    v_dofs = np.asarray(v_dofs, dtype=int)
    n_plate = int(u_dofs.size)
    if v_dofs.size != n_plate:
        raise ValueError("u_dofs and v_dofs must have the same length.")
    if n_plate < 2:
        raise ValueError("Need at least two plate nodes to form an axial constraint.")
    n_constraints = n_plate - 1
    t_arr = np.asarray(t_p, dtype=float)
    if t_arr.shape == (2,):
        tangents = np.broadcast_to(t_arr, (n_constraints, 2))
    elif t_arr.shape == (n_constraints, 2):
        tangents = t_arr
    else:
        raise ValueError(f"t_p must have shape (2,) or ({n_constraints}, 2); got {t_arr.shape}.")
    if n_q < int(np.max(np.concatenate([u_dofs, v_dofs]))) + 1:
        raise ValueError("n_q is smaller than a plate translational DOF index.")
    n_foam = n_q - n_theta
    if n_foam < 0:
        raise ValueError("n_theta cannot exceed n_q.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, j in enumerate(range(n_constraints)):
        jp1 = j + 1
        t_e = tangents[j]
        rows.extend([row, row, row, row])
        cols.extend([int(u_dofs[j]), int(v_dofs[j]), int(u_dofs[jp1]), int(v_dofs[jp1])])
        data.extend([-t_e[0], -t_e[1], t_e[0], t_e[1]])
    B = sparse.csc_matrix((data, (rows, cols)), shape=(n_constraints, n_q))
    if n_theta > 0 and sparse.linalg.norm(B[:, n_foam:]) != 0.0:
        raise ValueError("Axial constraints must not act on plate rotation DOFs.")
    return B


def constraint_rank(B: sparse.spmatrix, tol: float | None = None) -> int:
    """Return the numerical rank of B_p."""
    dense = B.toarray() if sparse.issparse(B) else np.asarray(B, dtype=float)
    return int(np.linalg.matrix_rank(dense, tol=tol))


def verify_constraint_rank(B: sparse.spmatrix, n_plate: int, tol: float | None = None) -> int:
    """Assert that B_p has n_plate - 1 independent rows."""
    expected = n_plate - 1
    if B.shape[0] != expected:
        raise ValueError(f"B_p has {B.shape[0]} rows; expected {expected}.")
    rank = constraint_rank(B, tol=tol)
    if rank != expected:
        raise ValueError(f"B_p has numerical rank {rank}; expected {expected}.")
    return rank
