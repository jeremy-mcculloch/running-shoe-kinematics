"""Cubic Euler–Bernoulli / Hermite plate assembly along the foam interface.

Element frames
--------------
Plate nodes are ordered heel to toe. Element ``e`` joins nodes ``i -> j`` with

    L_e = |X_j - X_i|,   t_e = (X_j - X_i) / L_e,   n_e = (-t_e,y, t_e,x),

so ``n_e`` is ``t_e`` rotated +90 degrees (upward for a heel-to-toe plate).
The element deflection is ``w = n_e . (u, v)`` and the nodal rotation DOF is
``theta = dw/ds`` (counterclockwise positive), a frame-independent in-plane
rotation shared by the elements meeting at a node. A straight plate passes a
single ``(2,)`` normal; a curved plate passes one normal per element.
Curvature is ``kappa = d^2 w / ds^2`` and the bending moment ``M = EI kappa``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from skfem import Basis


def plate_element_frames(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lengths, tangents, normals)`` for consecutive plate nodes ``coords`` (2, n)."""
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2 or X.shape[0] != 2 or X.shape[1] < 2:
        raise ValueError("coords must have shape (2, n_plate) with n_plate >= 2.")
    delta = np.diff(X, axis=1)
    lengths = np.hypot(delta[0], delta[1])
    if np.any(lengths <= 0.0):
        raise ValueError("Plate element lengths must be positive.")
    tangents = (delta / lengths).T
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return lengths, tangents, normals


def _per_element(vec: np.ndarray, n_el: int, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape == (2,):
        return np.broadcast_to(arr, (n_el, 2))
    if arr.shape == (n_el, 2):
        return arr
    raise ValueError(f"{name} must have shape (2,) or ({n_el}, 2); got {arr.shape}.")


def hermite_bending_matrix(EI: float, length: float) -> np.ndarray:
    """Return the 4x4 Euler–Bernoulli bending stiffness for one plate element.

    DOF order is (w_i, theta_i, w_j, theta_j). The bilinear form is
    a(w, z) = EI ∫ w'' z'' ds, so q^T K q = EI ∫ (w'')^2 ds.
    """
    if length <= 0.0:
        raise ValueError(f"Plate element length must be positive, got {length}.")
    L = float(length)
    factor = float(EI) / L**3
    return factor * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ],
        dtype=float,
    )


def plate_translational_dofs(
    basis: Basis,
    node_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_dofs, v_dofs) for ordered plate mesh nodes."""
    nodes = np.asarray(node_indices, dtype=int)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("Plate node_indices must be a 1D array with at least two nodes.")
    n_nodal = basis.nodal_dofs.shape[1]
    if int(np.max(nodes)) >= n_nodal:
        raise ValueError("Plate node index exceeds the continuum nodal DOF table.")
    u_dofs = np.asarray(basis.nodal_dofs[0, nodes], dtype=int)
    v_dofs = np.asarray(basis.nodal_dofs[1, nodes], dtype=int)
    return u_dofs, v_dofs


def plate_normal_deflection(
    u: np.ndarray,
    v: np.ndarray,
    n_p: np.ndarray,
) -> np.ndarray:
    """Return w_j = n_p · u_j at plate nodes."""
    n_p = np.asarray(n_p, dtype=float).reshape(2)
    return n_p[0] * np.asarray(u, dtype=float) + n_p[1] * np.asarray(v, dtype=float)


def element_transformation(
    n_q: int,
    u_i: int,
    v_i: int,
    theta_i: int,
    u_j: int,
    v_j: int,
    theta_j: int,
    n_p: np.ndarray,
) -> sparse.csc_matrix:
    """Build T_e such that q_p^{(e)} = T_e q with q_p = (w_i, θ_i, w_j, θ_j)."""
    n_p = np.asarray(n_p, dtype=float).reshape(2)
    rows = np.array([0, 0, 1, 2, 2, 3], dtype=int)
    cols = np.array([u_i, v_i, theta_i, u_j, v_j, theta_j], dtype=int)
    data = np.array([n_p[0], n_p[1], 1.0, n_p[0], n_p[1], 1.0], dtype=float)
    T = sparse.csc_matrix((data, (rows, cols)), shape=(4, n_q))
    if T.shape != (4, n_q):
        raise ValueError(f"Unexpected transformation shape {T.shape}.")
    return T


def assemble_plate_bending(
    n_foam: int,
    u_dofs: np.ndarray,
    v_dofs: np.ndarray,
    coords: np.ndarray,
    n_p: np.ndarray,
    EI: float,
) -> sparse.csc_matrix:
    """Assemble the global plate bending matrix in the enlarged primal space.

    The enlarged vector is q = [d_foam, theta] with n_q = n_foam + n_plate.
    Endpoint rotations are left free (natural condition: zero generalized end
    moment). When EI == 0 the plate rotation DOFs are omitted (n_theta = 0)
    because they would otherwise form a spurious kernel. ``n_p`` is one unit
    normal (2,) for a straight plate or one per element (n_plate - 1, 2).
    """
    u_dofs = np.asarray(u_dofs, dtype=int)
    v_dofs = np.asarray(v_dofs, dtype=int)
    coords = np.asarray(coords, dtype=float)
    n_plate = int(u_dofs.size)
    normals = _per_element(n_p, max(n_plate - 1, 0), "n_p")
    if v_dofs.size != n_plate:
        raise ValueError("u_dofs and v_dofs must have the same length.")
    if coords.shape != (2, n_plate):
        raise ValueError(f"coords must have shape (2, n_plate), got {coords.shape}.")
    if n_plate < 2:
        raise ValueError("A plate mesh requires at least two interface nodes.")

    include_theta = float(EI) > 0.0
    n_theta = n_plate if include_theta else 0
    n_q = n_foam + n_theta
    if n_q < n_foam:
        raise ValueError("Enlarged primal dimension is smaller than the continuum size.")

    if not include_theta:
        return sparse.csc_matrix((n_q, n_q))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n_plate - 1):
        j = i + 1
        delta = coords[:, j] - coords[:, i]
        length = float(np.linalg.norm(delta))
        K_p = hermite_bending_matrix(EI, length)
        theta_i = n_foam + i
        theta_j = n_foam + j
        T = element_transformation(
            n_q,
            int(u_dofs[i]),
            int(v_dofs[i]),
            theta_i,
            int(u_dofs[j]),
            int(v_dofs[j]),
            theta_j,
            normals[i],
        )
        K_e = (T.T @ sparse.csc_matrix(K_p) @ T).tocoo()
        rows.extend(K_e.row.tolist())
        cols.extend(K_e.col.tolist())
        data.extend(K_e.data.tolist())
    return sparse.csc_matrix((data, (rows, cols)), shape=(n_q, n_q))


def enlarge_foam_stiffness(K_foam: sparse.spmatrix, n_theta: int) -> sparse.csc_matrix:
    """Embed K_foam into the enlarged primal space with zero rotation block."""
    K_foam = K_foam.tocsc()
    n_foam = K_foam.shape[0]
    if n_theta == 0:
        return K_foam
    return sparse.bmat(
        [
            [K_foam, None],
            [None, sparse.csc_matrix((n_theta, n_theta))],
        ],
        format="csc",
    )
