"""Compliance matrix computation via the rigid-mode saddle-point system."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Generic, TypeVar

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from compliance_fem.fem.assembly import (
    assemble_region_foam_stiffness,
    verify_stiffness_symmetry,
)
from compliance_fem.fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    build_vector_selector,
    yy_block,
)
from compliance_fem.contact.config import SoleConfig
from compliance_fem.fem.constraints import (
    assemble_axial_constraints,
    verify_constraint_rank,
)
from compliance_fem.fem.plate import (
    assemble_plate_bending,
    enlarge_foam_stiffness,
    plate_element_frames,
    plate_translational_dofs,
)
from compliance_fem.fem.rigid_modes import (
    build_enlarged_rigid_modes,
    verify_constraint_on_modes,
    verify_rigid_modes,
)

COMPLIANCE_SCHEMA_VERSION = 5
# SoleConfig settings saved in the NPZ under their field names.
SOLE_SCALAR_FIELDS = tuple(
    f.name
    for f in fields(SoleConfig)
    if f.name not in ("geometry_csv", "normalized_geometry", "upper_foam_material", "lower_foam_material")
)

T = TypeVar("T")


@dataclass(frozen=True)
class TopBottom(Generic[T]):
    """One value for the top (foot) boundary and one for the bottom (ground) boundary."""

    top: T
    bottom: T


@dataclass(frozen=True)
class BoundarySelection:
    """Translational DOFs and reference coordinates of one boundary, heel to toe."""

    u_dofs: np.ndarray
    v_dofs: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ReciprocityErrors:
    """Relative asymmetry of the compliance blocks (zero for an exact solve)."""

    Ctt: float = 0.0
    Cbb: float = 0.0
    Cbt_Ctb: float = 0.0


@dataclass(frozen=True)
class ComplianceBlocks:
    """Nodal-force compliance blocks, vertical traction maps and boundary masses."""

    Ctt_force: np.ndarray
    Ctb_force: np.ndarray
    Cbt_force: np.ndarray
    Cbb_force: np.ndarray
    Ctt_traction: np.ndarray
    Ctb_traction: np.ndarray
    Cbt_traction: np.ndarray
    Cbb_traction: np.ndarray
    M_top: sparse.csc_matrix
    M_bottom: sparse.csc_matrix
    reciprocity: ReciprocityErrors


def _zero_pair() -> TopBottom[float]:
    return TopBottom(0.0, 0.0)


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
    config: SoleConfig
    rigid_mode_error: float
    reciprocity: ReciprocityErrors = field(default_factory=ReciprocityErrors)
    solve_residuals: TopBottom[float] = field(default_factory=_zero_pair)
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
    inextensibility_residuals: TopBottom[float] = field(default_factory=_zero_pair)
    gauge_residuals: TopBottom[float] = field(default_factory=_zero_pair)
    n_primal: int | None = None
    n_lambda: int = 0
    n_plate_nodes: int = 0
    n_plate_rotation_dofs: int = 0
    constraint_rank: int = 0
    Lambda_top: np.ndarray | None = None
    Lambda_bottom: np.ndarray | None = None
    B_p: sparse.csc_matrix | None = None
    compliance_schema_version: int = COMPLIANCE_SCHEMA_VERSION
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
    def reciprocity_error(self) -> float:
        """Relative ``||Cbt - Ctb^T|| / ||Cbt||``."""
        return self.reciprocity.Cbt_Ctb

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
            raise ValueError(f"ComplianceResult has no stored {side} selector DOFs.")
        return np.asarray(u, dtype=int), np.asarray(v, dtype=int)


def build_augmented_matrix(
    K: sparse.csc_matrix,
    R: sparse.csc_matrix,
    B: sparse.csc_matrix | None = None,
) -> sparse.csc_matrix:
    """Build the symmetric saddle-point matrix.

    Without plate constraints::

        [[K, R], [R^T, 0]]

    With the inextensible plate::

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


def _reciprocity_errors(Ctt: np.ndarray, Cbb: np.ndarray, Cbt: np.ndarray, Ctb: np.ndarray) -> ReciprocityErrors:
    return ReciprocityErrors(
        Ctt=_symmetry_error(Ctt),
        Cbb=_symmetry_error(Cbb),
        Cbt_Ctb=float(np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30)),
    )


def _verify_selector_normalization(S: sparse.csc_matrix) -> None:
    if S.nnz != S.shape[0]:
        raise ValueError("Selector must contain exactly one unit entry per row.")
    row_sums = np.asarray(S.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Selector rows are not unit loads.")


def _finish_compliance_blocks(
    basis,
    order: int,
    S: TopBottom[sparse.csc_matrix],
    U: TopBottom[np.ndarray],
    n_t: int,
    n_b: int,
    mass_matrices: TopBottom[sparse.csc_matrix],
) -> ComplianceBlocks:
    Ctt_force = np.asarray(S.top @ U.top)
    Ctb_force = np.asarray(S.top @ U.bottom)
    Cbt_force = np.asarray(S.bottom @ U.top)
    Cbb_force = np.asarray(S.bottom @ U.bottom)

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

    M_top, M_bottom = mass_matrices.top, mass_matrices.bottom
    # Traction maps stay on the vertical-vertical block; lookup uses force blocks.
    return ComplianceBlocks(
        Ctt_force=Ctt_force,
        Ctb_force=Ctb_force,
        Cbt_force=Cbt_force,
        Cbb_force=Cbb_force,
        Ctt_traction=yy_block(Ctt_force, n_t, n_t) @ M_top.toarray(),
        Ctb_traction=yy_block(Ctb_force, n_t, n_b) @ M_bottom.toarray(),
        Cbt_traction=yy_block(Cbt_force, n_b, n_t) @ M_top.toarray(),
        Cbb_traction=yy_block(Cbb_force, n_b, n_b) @ M_bottom.toarray(),
        M_top=M_top,
        M_bottom=M_bottom,
        reciprocity=_reciprocity_errors(Ctt_force, Cbb_force, Cbt_force, Ctb_force),
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
    boundaries: TopBottom[BoundarySelection],
    order: int,
    node_ids: TopBottom[np.ndarray],
    mass_matrices: TopBottom[sparse.csc_matrix],
) -> ComplianceResult:
    """Free-body plate compliance of the two-foam sole.

    ``plate_normals`` / ``plate_tangents`` hold one unit vector per plate element. Plate rotation DOFs exist only for
    ``plate_node_ids``; the selectors act on translational foam DOFs only.
    """
    verify_stiffness_symmetry(K_foam)
    u_dofs, v_dofs = plate_translational_dofs(basis, plate_node_ids)
    n_plate = int(u_dofs.size)
    n_foam = basis.N
    n_theta = n_plate if config.EI_plate_Nm2_per_m > 0.0 else 0
    n_q = n_foam + n_theta
    K_plate = assemble_plate_bending(
        n_foam,
        u_dofs,
        v_dofs,
        plate_coords,
        plate_normals,
        config.EI_plate_Nm2_per_m,
    )
    if K_plate.shape != (n_q, n_q):
        raise ValueError(f"Plate stiffness has shape {K_plate.shape}, expected ({n_q}, {n_q}).")
    K0 = enlarge_foam_stiffness(K_foam, n_theta) + K_plate
    verify_stiffness_symmetry(K0)

    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, plate_tangents, n_theta)
    rank = verify_constraint_rank(B, n_plate)

    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    rigid_mode_error = verify_rigid_modes(K0, R)
    rigid_constraint_residual = verify_constraint_on_modes(B, R)

    top, bottom = boundaries.top, boundaries.bottom
    S = TopBottom(
        build_vector_selector(top.u_dofs, top.v_dofs, n_q),
        build_vector_selector(bottom.u_dofs, bottom.v_dofs, n_q),
    )
    _verify_selector_normalization(S.top)
    _verify_selector_normalization(S.bottom)
    if S.top.shape[1] != n_q or S.bottom.shape[1] != n_q:
        raise ValueError("Boundary selectors must include zero columns for plate rotations.")
    if n_theta > 0:
        if sparse.linalg.norm(S.top[:, n_foam:]) != 0.0 or sparse.linalg.norm(S.bottom[:, n_foam:]) != 0.0:
            raise ValueError("Boundary selectors must not include plate rotation DOFs.")
    shared = np.intersect1d(
        np.concatenate([top.u_dofs, top.v_dofs]), np.concatenate([bottom.u_dofs, bottom.v_dofs])
    )
    if shared.size:
        raise ValueError(f"Top and bottom selectors share {shared.size} DOFs; they must be disjoint.")

    A = build_augmented_matrix(K0, R, B)
    lu = splu(A)

    U_top, res_top, extra_top = _solve_compliance_block(lu, A, S.top, "top forces", n_primal=n_q)
    U_bottom, res_bottom, extra_bottom = _solve_compliance_block(
        lu, A, S.bottom, "bottom forces", n_primal=n_q
    )
    U = TopBottom(U_top, U_bottom)
    n_lambda = B.shape[0]

    def _max_col_norm(mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        return float(max(np.linalg.norm(mat[:, j]) for j in range(mat.shape[1])))

    def _residuals(operator) -> TopBottom[float]:
        return TopBottom(
            _max_col_norm(np.asarray(operator @ U.top)), _max_col_norm(np.asarray(operator @ U.bottom))
        )

    n_t, n_b = int(len(top.x)), int(len(bottom.x))
    blocks = _finish_compliance_blocks(basis, order, S, U, n_t, n_b, mass_matrices=mass_matrices)
    plate_coords = np.asarray(plate_coords, dtype=float)
    _, el_tangents, el_normals = plate_element_frames(plate_coords)
    return ComplianceResult(
        Cbt_force=blocks.Cbt_force,
        Cbb_force=blocks.Cbb_force,
        Ctb_force=blocks.Ctb_force,
        Ctt_force=blocks.Ctt_force,
        Cbt_traction=blocks.Cbt_traction,
        Cbb_traction=blocks.Cbb_traction,
        M_top=blocks.M_top,
        M_bottom=blocks.M_bottom,
        x_top=top.x,
        x_bottom=bottom.x,
        config=config,
        rigid_mode_error=rigid_mode_error,
        reciprocity=blocks.reciprocity,
        solve_residuals=TopBottom(res_top, res_bottom),
        K=K0,
        basis=basis,
        factorization=lu,
        Ctt_traction=blocks.Ctt_traction,
        Ctb_traction=blocks.Ctb_traction,
        y_top=top.y,
        y_bottom=bottom.y,
        x_plate=plate_coords[0].copy(),
        y_plate=plate_coords[1].copy(),
        rigid_constraint_residual=rigid_constraint_residual,
        inextensibility_residuals=_residuals(B),
        gauge_residuals=_residuals(R.T),
        n_primal=n_q,
        n_lambda=n_lambda,
        n_plate_nodes=n_plate,
        n_plate_rotation_dofs=n_theta,
        constraint_rank=rank,
        Lambda_top=np.asarray(extra_top[:n_lambda, :], dtype=float),
        Lambda_bottom=np.asarray(extra_bottom[:n_lambda, :], dtype=float),
        B_p=B,
        compliance_schema_version=COMPLIANCE_SCHEMA_VERSION,
        dof_ordering=DOF_ORDERING_COMPONENT_MAJOR_UV,
        n_top_nodes=n_t,
        n_bottom_nodes=n_b,
        top_u_dofs=np.asarray(top.u_dofs, dtype=int),
        top_v_dofs=np.asarray(top.v_dofs, dtype=int),
        bottom_u_dofs=np.asarray(bottom.u_dofs, dtype=int),
        bottom_v_dofs=np.asarray(bottom.v_dofs, dtype=int),
        top_node_ids=np.asarray(node_ids.top, dtype=int),
        bottom_node_ids=np.asarray(node_ids.bottom, dtype=int),
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


def compute_compliance(config: SoleConfig, mesh_data=None) -> ComplianceResult:
    """Mesh the two-foam sole with its partial curved plate and compute the nodal-force compliance blocks.

    Selectors come from mesh tags in heel-to-toe material order; the shared toe
    vertex belongs to the bottom/contact selector only
    (``shared_toe_policy = bottom_contact_owns_toe_vertex``).
    """
    from compliance_fem.geometry.profile import REGION_LOWER, REGION_UPPER, geometry_metadata
    from compliance_fem.geometry.mesh import generate_mesh

    if mesh_data is None:
        mesh_data = generate_mesh(config)
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

    def _selection(nodes: np.ndarray) -> BoundarySelection:
        return BoundarySelection(
            u_dofs=np.asarray(basis.nodal_dofs[0, nodes], dtype=int),
            v_dofs=np.asarray(basis.nodal_dofs[1, nodes], dtype=int),
            x=p[0, nodes].copy(),
            y=p[1, nodes].copy(),
        )

    masses = TopBottom(
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
        boundaries=TopBottom(_selection(top_nodes), _selection(bottom_nodes)),
        order=config.element_order,
        node_ids=TopBottom(top_nodes, bottom_nodes),
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


def compliance_npz_payload(result: ComplianceResult) -> dict:
    """Build the NPZ dictionary for a compliance result."""
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
        "rigid_mode_error": result.rigid_mode_error,
        "reciprocity_error": result.reciprocity_error,
        "symmetry_error_Cbb": result.reciprocity.Cbb,
        "symmetry_error_Ctt": result.reciprocity.Ctt,
        "solve_residual_top": result.solve_residuals.top,
        "solve_residual_bottom": result.solve_residuals.bottom,
        "compliance_schema_version": result.compliance_schema_version,
        "dof_ordering": np.asarray(result.dof_ordering),
        "n_top_nodes": result.n_top_nodes,
        "n_bottom_nodes": result.n_bottom_nodes,
    }
    if result.Ctt_traction is not None:
        payload["Ctt_traction"] = result.Ctt_traction
    if result.Ctb_traction is not None:
        payload["Ctb_traction"] = result.Ctb_traction
    if result.y_top is not None:
        payload["y_top"] = result.y_top
    if result.y_bottom is not None:
        payload["y_bottom"] = result.y_bottom
    payload.update(measured_compliance_payload(result))
    return payload


def measured_compliance_payload(result: ComplianceResult) -> dict:
    """Measured-sole keys: geometry metadata, selectors, plate frames, materials."""
    import json

    cfg = result.config
    upper = cfg.material(cfg.upper_foam_material)
    lower = cfg.material(cfg.lower_foam_material)
    out = {
        **{name: getattr(cfg, name) for name in SOLE_SCALAR_FIELDS},
        "plate_axial_model": np.asarray("exact_inextensible"),
        "plate_tangent_model": np.asarray("per_element_chord"),
        "number_of_plate_nodes": result.n_plate_nodes,
        "number_of_plate_rotation_dofs": result.n_plate_rotation_dofs,
        "number_of_plate_constraints": result.n_lambda,
        "constraint_rank": result.constraint_rank,
        "rigid_mode_residual": result.rigid_mode_error,
        "rigid_constraint_residual": result.rigid_constraint_residual,
        "inextensibility_residual_top": result.inextensibility_residuals.top,
        "inextensibility_residual_bottom": result.inextensibility_residuals.bottom,
        "x_plate": result.x_plate,
        "y_plate": result.y_plate,
        "plate_element_tangents": result.plate_element_tangents,
        "plate_element_normals": result.plate_element_normals,
        "top_node_ids": result.top_node_ids,
        "bottom_node_ids": result.bottom_node_ids,
        "plate_node_ids": result.plate_node_ids,
        "shared_toe_node_id": int(result.shared_toe_node_id),
        "upper_foam_material": np.asarray(upper.name),
        "lower_foam_material": np.asarray(lower.name),
        "upper_E_heel": upper.E_heel,
        "upper_E_toe": upper.E_toe,
        "upper_nu": upper.nu,
        "lower_E_heel": lower.E_heel,
        "lower_E_toe": lower.E_toe,
        "lower_nu": lower.nu,
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


def save_compliance_npz(result: ComplianceResult, path) -> None:
    """Write a compressed compliance NPZ file."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **compliance_npz_payload(result))
