"""Boundary DOF selection and boundary mass matrices."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse
from skfem import Basis, FacetBasis
from skfem.assembly import BilinearForm, LinearForm
from skfem.element import ElementQuad1, ElementQuad2, ElementTriP1, ElementTriP2


DOF_ORDERING_COMPONENT_MAJOR_UV = "component_major_uv"


def vertical_boundary_data(basis: Basis, boundary: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertical DOF indices and (x, y) coordinates sorted by x on a boundary."""
    dofs_view = basis.get_dofs(boundary)
    v_dofs = np.asarray(dofs_view.nodal["u^2"], dtype=int)
    x_coords = basis.doflocs[0, v_dofs]
    y_coords = basis.doflocs[1, v_dofs]
    order = np.argsort(x_coords)
    return v_dofs[order], x_coords[order], y_coords[order]


def vector_boundary_data(
    basis: Basis, boundary: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u_dofs, v_dofs, x, y) on a named boundary, nodes sorted by increasing x."""
    dofs_view = basis.get_dofs(boundary)
    u_dofs = np.asarray(dofs_view.nodal["u^1"], dtype=int)
    v_dofs = np.asarray(dofs_view.nodal["u^2"], dtype=int)
    x_u = basis.doflocs[0, u_dofs]
    y_u = basis.doflocs[1, u_dofs]
    x_v = basis.doflocs[0, v_dofs]
    y_v = basis.doflocs[1, v_dofs]
    order_u = np.argsort(x_u)
    order_v = np.argsort(x_v)
    u_dofs = u_dofs[order_u]
    v_dofs = v_dofs[order_v]
    x = x_u[order_u]
    y = y_u[order_u]
    if u_dofs.size != v_dofs.size:
        raise ValueError(
            f"Boundary '{boundary}' has {u_dofs.size} u DOFs and {v_dofs.size} v DOFs."
        )
    if not np.allclose(x, x_v[order_v]) or not np.allclose(y, y_v[order_v]):
        raise ValueError(f"Boundary '{boundary}' u and v nodes are not co-located after sorting.")
    return u_dofs, v_dofs, x, y


def build_selector(dof_indices: np.ndarray, n_dofs: int) -> sparse.csc_matrix:
    """Build a selector matrix S such that v = S @ d."""
    n = len(dof_indices)
    rows = np.arange(n)
    data = np.ones(n, dtype=float)
    return sparse.csc_matrix((data, (rows, dof_indices)), shape=(n, n_dofs))


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


def stacked_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Stack component-major [u; v] along axis 0."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.concatenate([u, v], axis=0)


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


def _boundary_facet_element(mesh, order: int):
    if mesh.t.shape[0] == 3:
        return ElementTriP1() if order == 1 else ElementTriP2()
    if mesh.t.shape[0] == 4:
        return ElementQuad1() if order == 1 else ElementQuad2()
    raise ValueError(f"Unsupported mesh topology with {mesh.t.shape[0]} nodes per element.")


def _scalar_boundary_indices(
    fbasis: FacetBasis,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Map sorted boundary coordinates to scalar facet-basis DOFs."""
    scalar_dofs = np.asarray(fbasis.get_dofs().nodal["u"], dtype=int)
    scalar_x = fbasis.doflocs[0, scalar_dofs]
    scalar_y = fbasis.doflocs[1, scalar_dofs]
    mapping: list[int] = []
    for target_x, target_y in zip(x_coords, y_coords):
        matches = np.where(
            np.isclose(scalar_x, target_x, rtol=0.0, atol=1e-12)
            & np.isclose(scalar_y, target_y, rtol=0.0, atol=1e-12)
        )[0]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely map boundary node at (x, y)=({target_x}, {target_y})."
            )
        mapping.append(int(scalar_dofs[matches[0]]))
    return np.asarray(mapping, dtype=int)


def assemble_boundary_mass_matrix(
    basis: Basis,
    boundary: str,
    order: int,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Assemble a 1D consistent boundary mass matrix on vertical DOFs."""
    mesh = basis.mesh
    facets = mesh.boundaries[boundary]
    fbasis = FacetBasis(mesh, _boundary_facet_element(mesh, order), facets=facets)

    @BilinearForm
    def mass_form(u, v, _):
        return u * v

    M_full = mass_form.assemble(fbasis).tocsc()
    _, x_coords, y_coords = vertical_boundary_data(basis, boundary)
    idx = _scalar_boundary_indices(fbasis, x_coords, y_coords)
    return M_full[idx][:, idx].tocsc(), x_coords


def assemble_traction_load(
    basis: Basis,
    boundary: str,
    traction: Callable[[np.ndarray], np.ndarray],
    order: int,
) -> np.ndarray:
    """Assemble consistent nodal forces F = ∫_Γ N^T f(x) ds for a traction callable."""
    mesh = basis.mesh
    facets = mesh.boundaries[boundary]
    fbasis = FacetBasis(mesh, _boundary_facet_element(mesh, order), facets=facets)

    @LinearForm
    def load_form(v, w):
        x = w.x[0]
        return traction(x) * v

    F_full = load_form.assemble(fbasis)
    _, x_coords, y_coords = vertical_boundary_data(basis, boundary)
    idx = _scalar_boundary_indices(fbasis, x_coords, y_coords)
    return np.asarray(F_full[idx], dtype=float)


def reaction_basis_rows(x_coords: np.ndarray, xc: float) -> np.ndarray:
    """Return an (n, 2) matrix with rows [1, x_i - x_c]."""
    return np.column_stack([np.ones_like(x_coords), x_coords - xc])
