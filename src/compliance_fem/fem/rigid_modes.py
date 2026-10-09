"""Rigid-body mode construction and verification."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from skfem import Basis


def mesh_centroid(mesh) -> tuple[float, float]:
    """Return the area-weighted centroid of a 2D mesh."""
    x = mesh.p[0]
    y = mesh.p[1]
    if mesh.t.shape[0] == 3:
        elems = mesh.t.T
        x1, y1 = x[elems[:, 0]], y[elems[:, 0]]
        x2, y2 = x[elems[:, 1]], y[elems[:, 1]]
        x3, y3 = x[elems[:, 2]], y[elems[:, 2]]
        areas = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        xc = np.sum(areas * (x1 + x2 + x3) / 3.0) / np.sum(areas)
        yc = np.sum(areas * (y1 + y2 + y3) / 3.0) / np.sum(areas)
        return float(xc), float(yc)

    # Quad fallback: vertex average weighted by element area via determinant.
    elems = mesh.t.T
    xc_acc = 0.0
    yc_acc = 0.0
    area_acc = 0.0
    for quad in elems:
        pts = mesh.p[:, quad]
        area = 0.0
        cx = 0.0
        cy = 0.0
        for k in range(4):
            j = (k + 1) % 4
            cross = pts[0, k] * pts[1, j] - pts[0, j] * pts[1, k]
            area += cross
            cx += (pts[0, k] + pts[0, j]) * cross
            cy += (pts[1, k] + pts[1, j]) * cross
        area *= 0.5
        if area <= 0.0:
            continue
        xc_acc += cx / 6.0
        yc_acc += cy / 6.0
        area_acc += area
    return float(xc_acc / area_acc), float(yc_acc / area_acc)


def _component_dofs(basis: Basis, component: int) -> np.ndarray:
    """Return all global DOF indices for a vector component."""
    parts = [basis.nodal_dofs[component]]
    if basis.facet_dofs.size:
        parts.append(basis.facet_dofs[component])
    if basis.edge_dofs.size:
        parts.append(basis.edge_dofs[component])
    if basis.interior_dofs.size:
        parts.append(basis.interior_dofs[component])
    return np.unique(np.concatenate(parts))


def verify_rigid_modes(
    K: sparse.spmatrix,
    R: sparse.spmatrix,
    rtol: float = 1e-8,
) -> float:
    """Verify K @ R ≈ 0 and return the relative residual norm."""
    residual = K @ R
    rel = sparse.linalg.norm(residual) / max(sparse.linalg.norm(K), 1e-30)
    if rel > rtol:
        raise ValueError(f"Rigid modes do not lie in ker(K): relative residual {rel:.3e}.")
    return float(rel)


def build_enlarged_rigid_modes(
    basis: Basis,
    n_theta: int,
    rotation_theta: float = 1.0,
) -> tuple[sparse.csc_matrix, float, float]:
    """Construct orthonormal rigid modes in the enlarged primal space.

    Continuum components match the free-body translations and the unit
    counterclockwise rotation about the mesh centroid. Plate rotation entries
    are 0 for translations and `rotation_theta` for rigid rotation, which is
    +1 when θ = dw/ds with n_p = (-m_p, 1)/γ and t_p = (1, m_p)/γ.
    """
    mesh = basis.mesh
    xc, yc = mesh_centroid(mesh)
    n_foam = basis.N
    n_q = n_foam + n_theta

    mode_h = np.zeros(n_q)
    mode_v = np.zeros(n_q)
    mode_r = np.zeros(n_q)

    x_dofs = _component_dofs(basis, 0)
    y_dofs = _component_dofs(basis, 1)
    x_nodes = basis.doflocs[0, x_dofs]
    y_nodes = basis.doflocs[1, y_dofs]

    mode_h[x_dofs] = 1.0
    mode_v[y_dofs] = 1.0
    mode_r[x_dofs] = -(y_nodes - yc)
    mode_r[y_dofs] = x_nodes - xc
    if n_theta > 0:
        mode_r[n_foam:] = float(rotation_theta)

    R_dense = np.column_stack([mode_h, mode_v, mode_r])
    Q, _ = np.linalg.qr(R_dense, mode="reduced")
    R = sparse.csc_matrix(Q)
    if R.shape != (n_q, 3):
        raise ValueError(f"Enlarged rigid-mode matrix has shape {R.shape}, expected ({n_q}, 3).")
    return R, xc, yc


def verify_constraint_on_modes(
    B: sparse.spmatrix,
    R: sparse.spmatrix,
    rtol: float = 1e-8,
) -> float:
    """Verify B @ R ≈ 0 and return a relative residual."""
    residual = B @ R
    scale = max(sparse.linalg.norm(B) * sparse.linalg.norm(R), 1e-30)
    rel = sparse.linalg.norm(residual) / scale
    if rel > rtol:
        raise ValueError(
            f"Rigid modes do not lie in ker(B_p): relative residual {rel:.3e}."
        )
    return float(rel)
