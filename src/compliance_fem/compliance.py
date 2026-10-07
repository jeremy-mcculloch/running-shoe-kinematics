"""Compliance matrix computation via the rigid-mode saddle-point system."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from compliance_fem.assembly import (
    assemble_layered_foam_stiffness,
    assemble_region_foam_stiffness,
    assemble_stiffness,
    verify_stiffness_symmetry,
)
from compliance_fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    assemble_boundary_mass_matrix,
    build_vector_selector,
    vector_boundary_data,
    yy_block,
)
from compliance_fem.config import LayeredPlateConfig, MeasuredSoleConfig, ProblemConfig
from compliance_fem.constraints import (
    assemble_axial_constraints,
    verify_constant_tangential_nullspace,
    verify_constraint_rank,
)
from compliance_fem.geometry import generate_rectangular_mesh
from compliance_fem.layered_geometry import LayeredMeshData, generate_layered_mesh
from compliance_fem.plate import (
    assemble_plate_bending,
    enlarge_foam_stiffness,
    plate_element_frames,
    plate_translational_dofs,
)
from compliance_fem.rigid_modes import (
    build_enlarged_rigid_modes,
    build_rigid_modes,
    verify_constraint_on_modes,
    verify_rigid_modes,
)

COMPLIANCE_SCHEMA_VERSION_RECTANGLE = 3
COMPLIANCE_SCHEMA_VERSION_LAYERED = 3
COMPLIANCE_SCHEMA_VERSION_MEASURED = 4


@dataclass
class ComplianceResult:
    """Discrete compliance operators and diagnostics."""

    Cbt_force: np.ndarray
    Cbb_force: np.ndarray
    Ctb_force: np.ndarray
    Ctt_force: np.ndarray
    Cbt_traction: np.ndarray
    Cbb_traction: np.ndarray
    M_top: sparse.csc_matrix
    M_bottom: sparse.csc_matrix
    x_top: np.ndarray
    x_bottom: np.ndarray
    config: ProblemConfig | LayeredPlateConfig
    rigid_mode_error: float
    symmetry_errors: dict[str, float] = field(default_factory=dict)
    reciprocity_error: float = 0.0
    solve_residuals: dict[str, float] = field(default_factory=dict)
    K: sparse.csc_matrix | None = None
    basis: object | None = None
    factorization: splu | None = None
    Ctt_traction: np.ndarray | None = None
    Ctb_traction: np.ndarray | None = None
    y_top: np.ndarray | None = None
    y_bottom: np.ndarray | None = None
    x_plate: np.ndarray | None = None
    y_plate: np.ndarray | None = None
    rigid_constraint_residual: float = 0.0
    inextensibility_residuals: dict[str, float] = field(default_factory=dict)
    gauge_residuals: dict[str, float] = field(default_factory=dict)
    n_primal: int | None = None
    n_lambda: int = 0
    n_plate_nodes: int = 0
    n_plate_rotation_dofs: int = 0
    constraint_rank: int = 0
    Lambda_top: np.ndarray | None = None
    Lambda_bottom: np.ndarray | None = None
    B_p: sparse.csc_matrix | None = None
    compliance_schema_version: int = COMPLIANCE_SCHEMA_VERSION_RECTANGLE
    dof_ordering: str = DOF_ORDERING_COMPONENT_MAJOR_UV
    n_top_nodes: int = 0
    n_bottom_nodes: int = 0
    # Selector DOFs in heel-to-toe material order (component-major [u..., v...]).
    top_u_dofs: np.ndarray | None = None
    top_v_dofs: np.ndarray | None = None
    bottom_u_dofs: np.ndarray | None = None
    bottom_v_dofs: np.ndarray | None = None
    top_node_ids: np.ndarray | None = None
    bottom_node_ids: np.ndarray | None = None
    plate_node_ids: np.ndarray | None = None
    plate_element_tangents: np.ndarray | None = None
    plate_element_normals: np.ndarray | None = None
    shared_toe_node_id: int | None = None
    geometry_metadata: dict | None = None
    mesh_data: object | None = None

    @property
    def plate_element_ids(self) -> np.ndarray:
        n = 0 if self.plate_node_ids is None else max(int(len(self.plate_node_ids)) - 1, 0)
        return np.arange(n, dtype=int)

    @property
    def plate_u_dof_ids(self) -> np.ndarray:
        return np.asarray(self.basis.nodal_dofs[0, self.plate_node_ids], dtype=int)

    @property
    def plate_v_dof_ids(self) -> np.ndarray:
        return np.asarray(self.basis.nodal_dofs[1, self.plate_node_ids], dtype=int)

    @property
    def plate_rotation_dof_ids(self) -> np.ndarray:
        n_foam = int(self.basis.N)
        return n_foam + np.arange(int(self.n_plate_rotation_dofs), dtype=int)

    @property
    def plate_constraint_ids(self) -> np.ndarray:
        return np.arange(int(self.n_lambda), dtype=int)

    def selector_dofs(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return stored ``(u_dofs, v_dofs)`` for ``side`` in {"top", "bottom"}."""
        if side == "top":
            u, v = self.top_u_dofs, self.top_v_dofs
        elif side == "bottom":
            u, v = self.bottom_u_dofs, self.bottom_v_dofs
        else:
            raise ValueError(f"Unknown selector side {side!r}.")
        if u is None or v is None:
            u, v, _, _ = vector_boundary_data(self.basis, side)
        return np.asarray(u, dtype=int), np.asarray(v, dtype=int)


def build_augmented_matrix(
    K: sparse.csc_matrix,
    R: sparse.csc_matrix,
    B: sparse.csc_matrix | None = None,
) -> sparse.csc_matrix:
    """Build the symmetric saddle-point matrix.

    Rectangle (no plate constraints)::

        [[K, R], [R^T, 0]]

    Layered inextensible plate::

        [[K0, B^T, R], [B, 0, 0], [R^T, 0, 0]]
    """
    n = K.shape[0]
    n_r = R.shape[1]
    if K.shape[0] != K.shape[1]:
        raise ValueError("Stiffness matrix must be square.")
    if R.shape[0] != n:
        raise ValueError(f"R has {R.shape[0]} rows; expected {n}.")
    if B is None:
        return sparse.bmat(
            [[K, R], [R.T, sparse.csc_matrix((n_r, n_r))]],
            format="csc",
        )
    if B.shape[1] != n:
        raise ValueError(f"B_p has {B.shape[1]} columns; expected {n}.")
    n_l = B.shape[0]
    return sparse.bmat(
        [
            [K, B.T, R],
            [B, sparse.csc_matrix((n_l, n_l)), sparse.csc_matrix((n_l, n_r))],
            [R.T, sparse.csc_matrix((n_r, n_l)), sparse.csc_matrix((n_r, n_r))],
        ],
        format="csc",
    )


def _relative_residual(A: sparse.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    r = b - A @ x
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    return float(r_norm / max(b_norm, 1e-30))


def solve_multiple_rhs(
    lu: splu,
    rhs: np.ndarray,
    A: sparse.spmatrix | None = None,
    n_refine: int = 2,
) -> np.ndarray:
    """Solve A X = rhs for multiple right-hand sides.

    Optional iterative refinement reduces SuperLU roundoff on large
    symmetric-indefinite saddles, which otherwise shows up as reciprocity
    error in the extracted compliance blocks.
    """
    x = lu.solve(rhs)
    if A is None or n_refine <= 0:
        return x
    for _ in range(n_refine):
        x = x + lu.solve(rhs - A @ x)
    return x


def _solve_compliance_block(
    lu: splu,
    A: sparse.csc_matrix,
    S: sparse.csc_matrix,
    label: str,
    n_primal: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if n_primal is None:
        n_primal = A.shape[0] - 3
    if n_primal <= 0 or n_primal > A.shape[0]:
        raise ValueError(f"Invalid primal dimension n_primal={n_primal} for A of size {A.shape[0]}.")
    n_extra = A.shape[0] - n_primal
    if S.shape[1] != n_primal:
        raise ValueError(f"Selector has {S.shape[1]} columns; expected n_primal={n_primal}.")
    rhs = np.vstack([S.T.toarray(), np.zeros((n_extra, S.shape[0]))])
    solution = solve_multiple_rhs(lu, rhs, A=A, n_refine=2)
    U = solution[:n_primal, :]
    extras = solution[n_primal:, :]
    residual = max(_relative_residual(A, solution[:, j], rhs[:, j]) for j in range(rhs.shape[1]))
    if residual > 1e-8:
        # Very short plate elements (exact, unsnapped plate endpoints) make ||A|| large,
        # so the attainable ||r|| / ||b|| is roundoff-limited; accept a backward-stable solve.
        backward = _normwise_backward_error(A, solution, rhs)
        if backward > 1e-13:
            raise ValueError(
                f"Saddle-point solve residual for {label} is too large: {residual:.3e} "
                f"(normwise backward error {backward:.3e})."
            )
    return U, residual, extras


def _normwise_backward_error(A: sparse.spmatrix, X: np.ndarray, Bmat: np.ndarray) -> float:
    a_norm = float(sparse.linalg.norm(A, ord=np.inf))
    worst = 0.0
    for j in range(Bmat.shape[1]):
        r = Bmat[:, j] - A @ X[:, j]
        denom = a_norm * np.linalg.norm(X[:, j], np.inf) + np.linalg.norm(Bmat[:, j], np.inf)
        worst = max(worst, float(np.linalg.norm(r, np.inf) / max(denom, 1e-300)))
    return worst


def _symmetry_error(matrix: np.ndarray) -> float:
    err = np.linalg.norm(matrix - matrix.T)
    scale = np.linalg.norm(matrix)
    return float(err / max(scale, 1e-30))


def _reciprocity_errors(Ctt: np.ndarray, Cbb: np.ndarray, Cbt: np.ndarray, Ctb: np.ndarray) -> dict[str, float]:
    return {
        "Ctt": _symmetry_error(Ctt),
        "Cbb": _symmetry_error(Cbb),
        "Cbt_Ctb": float(np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30)),
    }


def _verify_selector_normalization(S: sparse.csc_matrix) -> None:
    if S.nnz != S.shape[0]:
        raise ValueError("Selector must contain exactly one unit entry per row.")
    row_sums = np.asarray(S.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Selector rows are not unit loads.")


def _finish_compliance_blocks(
    basis,
    order: int,
    S_top: sparse.csc_matrix,
    S_bottom: sparse.csc_matrix,
    U_top: np.ndarray,
    U_bottom: np.ndarray,
    x_top: np.ndarray,
    x_bottom: np.ndarray,
    y_top: np.ndarray,
    y_bottom: np.ndarray,
    mass_matrices: tuple[sparse.csc_matrix, sparse.csc_matrix] | None = None,
) -> dict:
    Ctt_force = np.asarray(S_top @ U_top)
    Ctb_force = np.asarray(S_top @ U_bottom)
    Cbt_force = np.asarray(S_bottom @ U_top)
    Cbb_force = np.asarray(S_bottom @ U_bottom)

    n_t = int(len(x_top))
    n_b = int(len(x_bottom))
    if Ctt_force.shape != (2 * n_t, 2 * n_t):
        raise ValueError(
            f"Ctt_force has shape {Ctt_force.shape}; expected ({2 * n_t}, {2 * n_t}) "
            "component-major vector block."
        )
    if Cbb_force.shape != (2 * n_b, 2 * n_b):
        raise ValueError(
            f"Cbb_force has shape {Cbb_force.shape}; expected ({2 * n_b}, {2 * n_b}) "
            "component-major vector block."
        )

    if mass_matrices is None:
        M_top, _ = assemble_boundary_mass_matrix(basis, "top", order)
        M_bottom, _ = assemble_boundary_mass_matrix(basis, "bottom", order)
    else:
        M_top, M_bottom = mass_matrices
    # Traction maps stay on the vertical-vertical block; lookup uses force blocks.
    Cbt_traction = yy_block(Cbt_force, n_b, n_t) @ M_top.toarray()
    Cbb_traction = yy_block(Cbb_force, n_b, n_b) @ M_bottom.toarray()
    Ctt_traction = yy_block(Ctt_force, n_t, n_t) @ M_top.toarray()
    Ctb_traction = yy_block(Ctb_force, n_t, n_b) @ M_bottom.toarray()

    reciprocity = _reciprocity_errors(Ctt_force, Cbb_force, Cbt_force, Ctb_force)
    return {
        "Ctt_force": Ctt_force,
        "Ctb_force": Ctb_force,
        "Cbt_force": Cbt_force,
        "Cbb_force": Cbb_force,
        "Ctt_traction": Ctt_traction,
        "Ctb_traction": Ctb_traction,
        "Cbt_traction": Cbt_traction,
        "Cbb_traction": Cbb_traction,
        "M_top": M_top,
        "M_bottom": M_bottom,
        "x_top": x_top,
        "x_bottom": x_bottom,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "symmetry_errors": {"Cbb_force": reciprocity["Cbb"], "Ctt_force": reciprocity["Ctt"]},
        "reciprocity_error": reciprocity["Cbt_Ctb"],
        "reciprocity_errors": reciprocity,
    }


def compute_rectangle_compliance(config: ProblemConfig, mesh_data=None) -> ComplianceResult:
    """Generate a rectangular mesh, assemble K, and compute compliance blocks."""
    if mesh_data is None:
        mesh_data = generate_rectangular_mesh(config)

    mesh = mesh_data.mesh
    K, basis = assemble_stiffness(mesh, config.E, config.nu, config.order)
    verify_stiffness_symmetry(K)

    R, _, _ = build_rigid_modes(basis)
    rigid_mode_error = verify_rigid_modes(K, R)

    u_top, v_top, x_top, y_top = vector_boundary_data(basis, "top")
    u_bottom, v_bottom, x_bottom, y_bottom = vector_boundary_data(basis, "bottom")
    S_top = build_vector_selector(u_top, v_top, basis.N)
    S_bottom = build_vector_selector(u_bottom, v_bottom, basis.N)
    _verify_selector_normalization(S_top)
    _verify_selector_normalization(S_bottom)

    A = build_augmented_matrix(K, R)
    lu = splu(A)

    U_top, res_top, _ = _solve_compliance_block(lu, A, S_top, "top forces", n_primal=basis.N)
    U_bottom, res_bottom, _ = _solve_compliance_block(
        lu, A, S_bottom, "bottom forces", n_primal=basis.N
    )
    blocks = _finish_compliance_blocks(
        basis,
        config.order,
        S_top,
        S_bottom,
        U_top,
        U_bottom,
        x_top,
        x_bottom,
        y_top,
        y_bottom,
    )
    return ComplianceResult(
        Cbt_force=blocks["Cbt_force"],
        Cbb_force=blocks["Cbb_force"],
        Ctb_force=blocks["Ctb_force"],
        Ctt_force=blocks["Ctt_force"],
        Cbt_traction=blocks["Cbt_traction"],
        Cbb_traction=blocks["Cbb_traction"],
        M_top=blocks["M_top"],
        M_bottom=blocks["M_bottom"],
        x_top=blocks["x_top"],
        x_bottom=blocks["x_bottom"],
        config=config,
        rigid_mode_error=rigid_mode_error,
        symmetry_errors=blocks["symmetry_errors"],
        reciprocity_error=blocks["reciprocity_error"],
        solve_residuals={"top": res_top, "bottom": res_bottom},
        K=K,
        basis=basis,
        factorization=lu,
        Ctt_traction=blocks["Ctt_traction"],
        Ctb_traction=blocks["Ctb_traction"],
        y_top=blocks["y_top"],
        y_bottom=blocks["y_bottom"],
        n_primal=basis.N,
        compliance_schema_version=COMPLIANCE_SCHEMA_VERSION_RECTANGLE,
        dof_ordering=DOF_ORDERING_COMPONENT_MAJOR_UV,
        n_top_nodes=int(len(x_top)),
        n_bottom_nodes=int(len(x_bottom)),
        top_u_dofs=np.asarray(u_top, dtype=int),
        top_v_dofs=np.asarray(v_top, dtype=int),
        bottom_u_dofs=np.asarray(u_bottom, dtype=int),
        bottom_v_dofs=np.asarray(v_bottom, dtype=int),
    )


def compute_layered_compliance(
    config: LayeredPlateConfig,
    mesh_data: LayeredMeshData | None = None,
) -> ComplianceResult:
    """Assemble the layered-plate operator and extract vector compliance blocks."""
    if mesh_data is None:
        mesh_data = generate_layered_mesh(config)

    mesh = mesh_data.mesh
    K_foam, basis, _, _ = assemble_layered_foam_stiffness(mesh, config)
    u_top, v_top, x_top, y_top = vector_boundary_data(basis, "top")
    u_bottom, v_bottom, x_bottom, y_bottom = vector_boundary_data(basis, "bottom")
    return _plate_compliance_core(
        config=config,
        K_foam=K_foam,
        basis=basis,
        plate_node_ids=np.asarray(mesh_data.plate_node_indices, dtype=int),
        plate_coords=np.vstack([mesh_data.x_plate, mesh_data.y_plate]),
        plate_normals=config.plate_normal,
        plate_tangents=config.plate_tangent,
        straight_tangent=config.plate_tangent,
        top=(u_top, v_top, x_top, y_top),
        bottom=(u_bottom, v_bottom, x_bottom, y_bottom),
        order=config.element_order,
        schema_version=COMPLIANCE_SCHEMA_VERSION_LAYERED,
    )


def _plate_compliance_core(
    *,
    config,
    K_foam: sparse.csc_matrix,
    basis,
    plate_node_ids: np.ndarray,
    plate_coords: np.ndarray,
    plate_normals: np.ndarray,
    plate_tangents: np.ndarray,
    straight_tangent: np.ndarray | None,
    top: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bottom: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    order: int,
    schema_version: int,
    top_node_ids: np.ndarray | None = None,
    bottom_node_ids: np.ndarray | None = None,
    mass_matrices: tuple[sparse.csc_matrix, sparse.csc_matrix] | None = None,
) -> ComplianceResult:
    """Free-body plate compliance shared by the layered and measured geometries.

    ``plate_normals`` / ``plate_tangents`` are one unit vector (straight plate)
    or one per plate element (curved plate). Plate rotation DOFs exist only for
    ``plate_node_ids``; the selectors act on translational foam DOFs only.
    """
    verify_stiffness_symmetry(K_foam)
    u_dofs, v_dofs = plate_translational_dofs(basis, plate_node_ids)
    n_plate = int(u_dofs.size)
    n_foam = basis.N
    n_theta = n_plate if config.EI_plate > 0.0 else 0
    n_q = n_foam + n_theta
    K_plate = assemble_plate_bending(
        n_foam,
        u_dofs,
        v_dofs,
        plate_coords,
        plate_normals,
        config.EI_plate,
    )
    if K_plate.shape != (n_q, n_q):
        raise ValueError(f"Plate stiffness has shape {K_plate.shape}, expected ({n_q}, {n_q}).")
    K0 = enlarge_foam_stiffness(K_foam, n_theta) + K_plate
    verify_stiffness_symmetry(K0)

    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, plate_tangents, n_theta)
    rank = verify_constraint_rank(B, n_plate)
    if straight_tangent is not None:
        verify_constant_tangential_nullspace(B, u_dofs, v_dofs, straight_tangent)

    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    rigid_mode_error = verify_rigid_modes(K0, R)
    rigid_constraint_residual = verify_constraint_on_modes(B, R)

    u_top, v_top, x_top, y_top = top
    u_bottom, v_bottom, x_bottom, y_bottom = bottom
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bottom, v_bottom, n_q)
    _verify_selector_normalization(S_top)
    _verify_selector_normalization(S_bottom)
    if S_top.shape[1] != n_q or S_bottom.shape[1] != n_q:
        raise ValueError("Boundary selectors must include zero columns for plate rotations.")
    if n_theta > 0:
        if sparse.linalg.norm(S_top[:, n_foam:]) != 0.0 or sparse.linalg.norm(S_bottom[:, n_foam:]) != 0.0:
            raise ValueError("Boundary selectors must not include plate rotation DOFs.")
    shared = np.intersect1d(np.concatenate([u_top, v_top]), np.concatenate([u_bottom, v_bottom]))
    if shared.size:
        raise ValueError(f"Top and bottom selectors share {shared.size} DOFs; they must be disjoint.")

    A = build_augmented_matrix(K0, R, B)
    lu = splu(A)

    U_top, res_top, extra_top = _solve_compliance_block(lu, A, S_top, "top forces", n_primal=n_q)
    U_bottom, res_bottom, extra_bottom = _solve_compliance_block(
        lu, A, S_bottom, "bottom forces", n_primal=n_q
    )
    n_lambda = B.shape[0]
    Lambda_top = np.asarray(extra_top[:n_lambda, :], dtype=float)
    Lambda_bottom = np.asarray(extra_bottom[:n_lambda, :], dtype=float)

    def _max_col_norm(mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        return float(max(np.linalg.norm(mat[:, j]) for j in range(mat.shape[1])))

    inext_top = _max_col_norm(np.asarray(B @ U_top))
    inext_bottom = _max_col_norm(np.asarray(B @ U_bottom))
    gauge_top = _max_col_norm(np.asarray(R.T @ U_top))
    gauge_bottom = _max_col_norm(np.asarray(R.T @ U_bottom))

    blocks = _finish_compliance_blocks(
        basis,
        order,
        S_top,
        S_bottom,
        U_top,
        U_bottom,
        x_top,
        x_bottom,
        y_top,
        y_bottom,
        mass_matrices=mass_matrices,
    )
    plate_coords = np.asarray(plate_coords, dtype=float)
    _, el_tangents, el_normals = plate_element_frames(plate_coords)
    return ComplianceResult(
        Cbt_force=blocks["Cbt_force"],
        Cbb_force=blocks["Cbb_force"],
        Ctb_force=blocks["Ctb_force"],
        Ctt_force=blocks["Ctt_force"],
        Cbt_traction=blocks["Cbt_traction"],
        Cbb_traction=blocks["Cbb_traction"],
        M_top=blocks["M_top"],
        M_bottom=blocks["M_bottom"],
        x_top=blocks["x_top"],
        x_bottom=blocks["x_bottom"],
        config=config,
        rigid_mode_error=rigid_mode_error,
        symmetry_errors=blocks["symmetry_errors"],
        reciprocity_error=blocks["reciprocity_error"],
        solve_residuals={"top": res_top, "bottom": res_bottom},
        K=K0,
        basis=basis,
        factorization=lu,
        Ctt_traction=blocks["Ctt_traction"],
        Ctb_traction=blocks["Ctb_traction"],
        y_top=blocks["y_top"],
        y_bottom=blocks["y_bottom"],
        x_plate=plate_coords[0].copy(),
        y_plate=plate_coords[1].copy(),
        rigid_constraint_residual=rigid_constraint_residual,
        inextensibility_residuals={"top": inext_top, "bottom": inext_bottom},
        gauge_residuals={"top": gauge_top, "bottom": gauge_bottom},
        n_primal=n_q,
        n_lambda=n_lambda,
        n_plate_nodes=n_plate,
        n_plate_rotation_dofs=n_theta,
        constraint_rank=rank,
        Lambda_top=Lambda_top,
        Lambda_bottom=Lambda_bottom,
        B_p=B,
        compliance_schema_version=schema_version,
        dof_ordering=DOF_ORDERING_COMPONENT_MAJOR_UV,
        n_top_nodes=int(len(x_top)),
        n_bottom_nodes=int(len(x_bottom)),
        top_u_dofs=np.asarray(u_top, dtype=int),
        top_v_dofs=np.asarray(v_top, dtype=int),
        bottom_u_dofs=np.asarray(u_bottom, dtype=int),
        bottom_v_dofs=np.asarray(v_bottom, dtype=int),
        top_node_ids=None if top_node_ids is None else np.asarray(top_node_ids, dtype=int),
        bottom_node_ids=None if bottom_node_ids is None else np.asarray(bottom_node_ids, dtype=int),
        plate_node_ids=np.asarray(plate_node_ids, dtype=int),
        plate_element_tangents=el_tangents,
        plate_element_normals=el_normals,
    )


def _boundary_mass_on_nodes(basis, facets: np.ndarray, nodes: np.ndarray) -> sparse.csc_matrix:
    """Consistent P1 boundary mass on ``facets`` restricted to the ordered ``nodes``."""
    from skfem import FacetBasis
    from skfem.assembly import BilinearForm
    from skfem.element import ElementTriP1

    fbasis = FacetBasis(basis.mesh, ElementTriP1(), facets=np.asarray(facets, dtype=int))

    @BilinearForm
    def mass_form(u, v, _):
        return u * v

    M = mass_form.assemble(fbasis).tocsc()
    idx = np.asarray(nodes, dtype=int)
    return M[idx][:, idx].tocsc()


def compute_measured_compliance(config, mesh_data=None) -> ComplianceResult:
    """Measured two-foam sole with a partial curved plate (same solver as layered).

    Selectors come from mesh tags in heel-to-toe material order; the shared toe
    vertex belongs to the bottom/contact selector only
    (``shared_toe_policy = bottom_contact_owns_toe_vertex``).
    """
    from compliance_fem.measured_geometry import REGION_LOWER, REGION_UPPER, geometry_metadata
    from compliance_fem.measured_mesh import generate_measured_mesh

    if mesh_data is None:
        mesh_data = generate_measured_mesh(config)
    mesh = mesh_data.mesh
    region_elements = {
        REGION_UPPER: mesh_data.region_elements(REGION_UPPER),
        REGION_LOWER: mesh_data.region_elements(REGION_LOWER),
    }
    K_foam, basis, _ = assemble_region_foam_stiffness(
        mesh, region_elements, config.region_materials(), config.L, config.element_order
    )
    top_nodes = np.asarray(mesh_data.top_selector_nodes, dtype=int)
    bottom_nodes = np.asarray(mesh_data.bottom_selector_nodes, dtype=int)
    if int(mesh_data.toe_node_id) in set(top_nodes.tolist()):
        raise ValueError("The shared toe vertex must not be in the top selector.")
    p = np.asarray(mesh.p, dtype=float)
    top = (
        np.asarray(basis.nodal_dofs[0, top_nodes], dtype=int),
        np.asarray(basis.nodal_dofs[1, top_nodes], dtype=int),
        p[0, top_nodes].copy(),
        p[1, top_nodes].copy(),
    )
    bottom = (
        np.asarray(basis.nodal_dofs[0, bottom_nodes], dtype=int),
        np.asarray(basis.nodal_dofs[1, bottom_nodes], dtype=int),
        p[0, bottom_nodes].copy(),
        p[1, bottom_nodes].copy(),
    )
    masses = (
        _boundary_mass_on_nodes(basis, mesh.boundaries["TOP_SURFACE"], top_nodes),
        _boundary_mass_on_nodes(basis, mesh.boundaries["BOTTOM_SURFACE"], bottom_nodes),
    )
    plate_nodes = np.asarray(mesh_data.plate_node_ids, dtype=int)
    plate_coords = p[:, plate_nodes]
    _, tangents, normals = plate_element_frames(plate_coords)
    result = _plate_compliance_core(
        config=config,
        K_foam=K_foam,
        basis=basis,
        plate_node_ids=plate_nodes,
        plate_coords=plate_coords,
        plate_normals=normals,
        plate_tangents=tangents,
        straight_tangent=None,
        top=top,
        bottom=bottom,
        order=config.element_order,
        schema_version=COMPLIANCE_SCHEMA_VERSION_MEASURED,
        top_node_ids=top_nodes,
        bottom_node_ids=bottom_nodes,
        mass_matrices=masses,
    )
    result.mesh_data = mesh_data
    result.shared_toe_node_id = int(mesh_data.toe_node_id)
    result.geometry_metadata = {
        **geometry_metadata(mesh_data.geometry, config),
        "shared_toe_node_id": int(mesh_data.toe_node_id),
        "mesh_quality": dict(mesh_data.quality),
    }
    return result


def compute_compliance(
    config: ProblemConfig | LayeredPlateConfig,
    mesh_data=None,
) -> ComplianceResult:
    """Generate mesh, assemble K, and compute nodal-force compliance blocks."""
    if isinstance(config, MeasuredSoleConfig):
        return compute_measured_compliance(config, mesh_data=mesh_data)
    if isinstance(config, LayeredPlateConfig):
        return compute_layered_compliance(config, mesh_data=mesh_data)
    return compute_rectangle_compliance(config, mesh_data=mesh_data)


def compliance_npz_payload(result: ComplianceResult) -> dict:
    """Build the NPZ dictionary, preserving existing rectangle keys."""
    cfg = result.config
    payload: dict = {
        "Cbt_force": result.Cbt_force,
        "Cbb_force": result.Cbb_force,
        "Ctb_force": result.Ctb_force,
        "Ctt_force": result.Ctt_force,
        "Cbt_traction": result.Cbt_traction,
        "Cbb_traction": result.Cbb_traction,
        "M_top": result.M_top.toarray(),
        "M_bottom": result.M_bottom.toarray(),
        "x_top": result.x_top,
        "x_bottom": result.x_bottom,
        "L": cfg.L,
        "H": cfg.H,
        "E": cfg.E,
        "nu": cfg.nu,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "order": cfg.order,
        "rigid_mode_error": result.rigid_mode_error,
        "reciprocity_error": result.reciprocity_error,
        "symmetry_error_Cbb": result.symmetry_errors.get("Cbb_force", 0.0),
        "symmetry_error_Ctt": result.symmetry_errors.get("Ctt_force", 0.0),
        "solve_residual_top": result.solve_residuals.get("top", 0.0),
        "solve_residual_bottom": result.solve_residuals.get("bottom", 0.0),
        "geometry_type": np.asarray(cfg.geometry_type),
        "compliance_schema_version": result.compliance_schema_version,
        "dof_ordering": np.asarray(result.dof_ordering),
        "n_top_nodes": result.n_top_nodes,
        "n_bottom_nodes": result.n_bottom_nodes,
        "sole_rocker_height": float(getattr(cfg, "sole_rocker_height", 0.0)),
        "sole_rocker_apex": float(getattr(cfg, "sole_rocker_apex", 0.5)),
    }
    if result.Ctt_traction is not None:
        payload["Ctt_traction"] = result.Ctt_traction
    if result.Ctb_traction is not None:
        payload["Ctb_traction"] = result.Ctb_traction
    if result.y_top is not None:
        payload["y_top"] = result.y_top
    if result.y_bottom is not None:
        payload["y_bottom"] = result.y_bottom
    if isinstance(cfg, LayeredPlateConfig):
        payload.update(
            {
                "h1_heel": cfg.h1_heel,
                "h1_toe": cfg.h1_toe,
                "h2_heel": cfg.h2_heel,
                "h2_toe": cfg.h2_toe,
                "E1": cfg.E1,
                "nu1": cfg.nu1,
                "E_heel": cfg.E_heel,
                "E_toe": cfg.E_toe,
                "nu2": cfg.nu2,
                "EI_plate": cfg.EI_plate,
                "ny1": cfg.ny1,
                "ny2": cfg.ny2,
                "element_order": cfg.element_order,
                "plate_axial_model": np.asarray("exact_inextensible"),
                "plate_length": cfg.plate_length,
                "plate_tangent": cfg.plate_tangent,
                "plate_normal": cfg.plate_normal,
                "number_of_plate_nodes": result.n_plate_nodes,
                "number_of_plate_rotation_dofs": result.n_plate_rotation_dofs,
                "number_of_plate_constraints": result.n_lambda,
                "constraint_rank": result.constraint_rank,
                "rigid_mode_residual": result.rigid_mode_error,
                "rigid_constraint_residual": result.rigid_constraint_residual,
                "compliance_reciprocity_errors": np.array(
                    [
                        result.symmetry_errors.get("Ctt_force", 0.0),
                        result.symmetry_errors.get("Cbb_force", 0.0),
                        result.reciprocity_error,
                    ],
                    dtype=float,
                ),
                "inextensibility_residual_top": result.inextensibility_residuals.get("top", 0.0),
                "inextensibility_residual_bottom": result.inextensibility_residuals.get(
                    "bottom", 0.0
                ),
            }
        )
        if result.x_plate is not None:
            payload["x_plate"] = result.x_plate
        if result.y_plate is not None:
            payload["y_plate"] = result.y_plate
        if result.Lambda_top is not None:
            payload["Lambda_top"] = result.Lambda_top
        if result.Lambda_bottom is not None:
            payload["Lambda_bottom"] = result.Lambda_bottom
    if isinstance(cfg, MeasuredSoleConfig):
        payload.update(measured_compliance_payload(result))
    return payload


def measured_compliance_payload(result: ComplianceResult) -> dict:
    """Measured-sole keys: geometry metadata, selectors, plate frames, materials."""
    import json

    cfg = result.config
    upper = cfg.material(cfg.upper_foam_material)
    lower = cfg.material(cfg.lower_foam_material)
    out = {
        "EI_plate": cfg.EI_plate,
        "element_order": cfg.element_order,
        "plate_axial_model": np.asarray("exact_inextensible"),
        "plate_tangent_model": np.asarray("per_element_chord"),
        "number_of_plate_nodes": result.n_plate_nodes,
        "number_of_plate_rotation_dofs": result.n_plate_rotation_dofs,
        "number_of_plate_constraints": result.n_lambda,
        "constraint_rank": result.constraint_rank,
        "rigid_mode_residual": result.rigid_mode_error,
        "rigid_constraint_residual": result.rigid_constraint_residual,
        "inextensibility_residual_top": result.inextensibility_residuals.get("top", 0.0),
        "inextensibility_residual_bottom": result.inextensibility_residuals.get("bottom", 0.0),
        "x_plate": result.x_plate,
        "y_plate": result.y_plate,
        "plate_element_tangents": result.plate_element_tangents,
        "plate_element_normals": result.plate_element_normals,
        "top_node_ids": result.top_node_ids,
        "bottom_node_ids": result.bottom_node_ids,
        "plate_node_ids": result.plate_node_ids,
        "shared_toe_node_id": int(result.shared_toe_node_id),
        "shoe_length_mm": float(cfg.shoe_length_mm),
        "upper_foam_material": np.asarray(upper.name),
        "lower_foam_material": np.asarray(lower.name),
        "upper_E_heel": upper.E_heel,
        "upper_E_toe": upper.E_toe,
        "upper_nu": upper.nu,
        "lower_E_heel": lower.E_heel,
        "lower_E_toe": lower.E_toe,
        "lower_nu": lower.nu,
        "ffturbo_E": cfg.ffturbo_E,
        "ffturbo_nu": cfg.ffturbo_nu,
        "ffleap_E_heel": cfg.ffleap_E_heel,
        "ffleap_E_toe": cfg.ffleap_E_toe,
        "ffleap_nu": cfg.ffleap_nu,
        "mesh_size": cfg.mesh_size,
        "toe_refinement": cfg.toe_refinement,
        "heel_corner_refinement": cfg.heel_corner_refinement,
        "interface_refinement": cfg.interface_refinement,
        "plate_end_refinement": cfg.plate_end_refinement,
        "geometry_metadata_json": np.asarray(json.dumps(result.geometry_metadata, sort_keys=True)),
        "normalized_geometry_json": np.asarray(
            json.dumps(cfg.normalized_geometry.to_payload(), sort_keys=True)
        ),
    }
    if result.Lambda_top is not None:
        out["Lambda_top"] = result.Lambda_top
    if result.Lambda_bottom is not None:
        out["Lambda_bottom"] = result.Lambda_bottom
    return out


def is_measured_compliance_npz(path) -> bool:
    """Return True when an NPZ was written from a measured-sole compliance solve."""
    from pathlib import Path

    data = np.load(Path(path))
    if "geometry_type" not in data.files:
        return False
    return str(np.asarray(data["geometry_type"]).reshape(-1)[0]) == "measured_sole"


def measured_config_from_compliance_npz(path) -> MeasuredSoleConfig:
    """Rebuild :class:`MeasuredSoleConfig` from the normalized geometry stored in the NPZ."""
    import json
    from pathlib import Path

    from compliance_fem.measured_geometry import NormalizedSoleGeometry, validate_normalized_geometry

    data = np.load(Path(path))
    required = (
        "normalized_geometry_json",
        "shoe_length_mm",
        "upper_foam_material",
        "lower_foam_material",
        "EI_plate",
        "mesh_size",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Measured compliance NPZ missing keys: {missing}")
    normalized = NormalizedSoleGeometry.from_payload(
        json.loads(str(np.asarray(data["normalized_geometry_json"]).reshape(-1)[0]))
    )
    validate_normalized_geometry(normalized)

    def _f(key: str, default: float) -> float:
        return float(data[key]) if key in data.files else default

    return MeasuredSoleConfig(
        shoe_length_mm=float(data["shoe_length_mm"]),
        normalized_geometry=normalized,
        upper_foam_material=str(np.asarray(data["upper_foam_material"]).reshape(-1)[0]),
        lower_foam_material=str(np.asarray(data["lower_foam_material"]).reshape(-1)[0]),
        EI_plate=float(data["EI_plate"]),
        ffturbo_E=_f("ffturbo_E", 2.6e5),
        ffturbo_nu=_f("ffturbo_nu", 0.113),
        ffleap_E_heel=_f("ffleap_E_heel", 3.54e5),
        ffleap_E_toe=_f("ffleap_E_toe", 2.07e5),
        ffleap_nu=_f("ffleap_nu", 0.113),
        mesh_size=float(data["mesh_size"]),
        toe_refinement=_f("toe_refinement", 0.4),
        heel_corner_refinement=_f("heel_corner_refinement", 0.5),
        interface_refinement=_f("interface_refinement", 0.6),
        plate_end_refinement=_f("plate_end_refinement", 0.4),
    )


def save_compliance_npz(result: ComplianceResult, path) -> None:
    """Write a compressed compliance NPZ file."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **compliance_npz_payload(result))


def is_layered_compliance_npz(path) -> bool:
    """Return True when an NPZ was written from a layered-plate compliance solve."""
    from pathlib import Path

    data = np.load(Path(path))
    if "EI_plate" in data.files and "h1_heel" in data.files and "E_heel" in data.files:
        return True
    if "geometry_type" in data.files:
        gt = str(np.asarray(data["geometry_type"]).reshape(-1)[0])
        return "layered" in gt.lower()
    return False


def layered_config_from_compliance_npz(path) -> LayeredPlateConfig:
    """Rebuild :class:`LayeredPlateConfig` from a layered compliance NPZ.

    Used by the contact-lookup CLI to recompute the FEM factorization in-process
    so plate_response fields can be recovered during lookup generation.
    """
    from pathlib import Path

    data = np.load(Path(path))
    required = (
        "L",
        "h1_heel",
        "h1_toe",
        "h2_heel",
        "h2_toe",
        "E1",
        "nu1",
        "E_heel",
        "E_toe",
        "nu2",
        "EI_plate",
        "nx",
        "ny1",
        "ny2",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Layered compliance NPZ missing keys: {missing}")
    element_order = (
        int(data["element_order"])
        if "element_order" in data.files
        else int(data["order"]) if "order" in data.files else 1
    )
    return LayeredPlateConfig(
        L=float(data["L"]),
        h1_heel=float(data["h1_heel"]),
        h1_toe=float(data["h1_toe"]),
        h2_heel=float(data["h2_heel"]),
        h2_toe=float(data["h2_toe"]),
        E1=float(data["E1"]),
        nu1=float(data["nu1"]),
        E_heel=float(data["E_heel"]),
        E_toe=float(data["E_toe"]),
        nu2=float(data["nu2"]),
        EI_plate=float(data["EI_plate"]),
        nx=int(data["nx"]),
        ny1=int(data["ny1"]),
        ny2=int(data["ny2"]),
        element_order=element_order,
        sole_rocker_height=(
            float(data["sole_rocker_height"]) if "sole_rocker_height" in data.files else 0.0
        ),
        sole_rocker_apex=(
            float(data["sole_rocker_apex"]) if "sole_rocker_apex" in data.files else 0.5
        ),
    )
