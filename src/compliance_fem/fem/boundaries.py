"""Boundary DOF selection and component-major vector helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse


DOF_ORDERING_COMPONENT_MAJOR_UV = "component_major_uv"


def build_vector_selector(u_dofs: np.ndarray, v_dofs: np.ndarray, n_q: int) -> sparse.csc_matrix:
    """Build a component-major selector S with shape (2n, n_q): stacked [u; v].

    Plate-rotation columns of ``n_q`` remain zero because only translational
    foam DOFs are listed in ``u_dofs`` / ``v_dofs``.
    """
    u_dofs = np.asarray(u_dofs, dtype=int)
    v_dofs = np.asarray(v_dofs, dtype=int)
    if u_dofs.size != v_dofs.size:
        raise ValueError("u_dofs and v_dofs must have the same length.")
    n = int(u_dofs.size)
    rows = np.arange(2 * n)
    cols = np.concatenate([u_dofs, v_dofs])
    data = np.ones(2 * n, dtype=float)
    return sparse.csc_matrix((data, (rows, cols)), shape=(2 * n, n_q))


def component_major_indices(nodes: np.ndarray, n_nodes: int) -> np.ndarray:
    """Map node indices to stacked [u; v] indices in a component-major vector of length 2 n_nodes."""
    nodes = np.asarray(nodes, dtype=int)
    if nodes.size == 0:
        return np.zeros(0, dtype=int)
    return np.concatenate([nodes, nodes + int(n_nodes)])


def split_uv(stacked: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a component-major [u; v] array into its two blocks."""
    stacked = np.asarray(stacked, dtype=float)
    if stacked.shape[0] != 2 * n_nodes:
        raise ValueError(
            f"Stacked vector has leading dimension {stacked.shape[0]}; expected {2 * n_nodes}."
        )
    return stacked[:n_nodes], stacked[n_nodes:]


def yy_block(matrix: np.ndarray, n_row_nodes: int, n_col_nodes: int) -> np.ndarray:
    """Extract the vertical-vertical block of a component-major operator."""
    matrix = np.asarray(matrix, dtype=float)
    return matrix[n_row_nodes:, n_col_nodes:]


def vector_rigid_mode_matrix(
    x: np.ndarray,
    y: np.ndarray,
    xr: float,
    yr: float,
) -> np.ndarray:
    """Return (2n, 3) rigid rows: horizontal translation, vertical translation, rotation.

    Rotation is ``u = -(y - yr)``, ``v = (x - xr)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    n = int(x.size)
    R = np.zeros((2 * n, 3), dtype=float)
    R[:n, 0] = 1.0
    R[n:, 1] = 1.0
    R[:n, 2] = -(y - yr)
    R[n:, 2] = x - xr
    return R
