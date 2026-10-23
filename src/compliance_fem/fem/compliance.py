"""Compliance matrix computation via the rigid-mode saddle-point system."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu
from skfem import CellBasis

from compliance_fem.fem.assembly import (
    assemble_region_foam_stiffness,
    verify_stiffness_symmetry,
)
from compliance_fem.fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    build_vector_selector,
)
from compliance_fem.contact.sole_config import SoleConfig
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
from compliance_fem.fem.skfem_types import dof_table
from pathlib import Path

if TYPE_CHECKING:
    from compliance_fem.geometry.mesh import SoleMeshData

COMPLIANCE_SCHEMA_VERSION = 7
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
class ReciprocityErrors:
    """Relative asymmetry of the compliance blocks (zero for an exact solve)."""

    Ctt: float = 0.0
    Cbb: float = 0.0
    Cbt_Ctb: float = 0.0


@dataclass
class ComplianceResult:
    """Free-body nodal-force compliance of the meshed sole and the FEM objects field recovery needs.

    The blocks are component-major ``[u...; v...]`` over ``top_node_ids`` / ``bottom_node_ids``
    (heel to toe; the shared toe vertex belongs to the bottom). Node ids and reference
    coordinates of the boundaries and the plate are read from ``mesh_data``.
    """

    Ctt_force: np.ndarray
    Ctb_force: np.ndarray
    Cbt_force: np.ndarray
    Cbb_force: np.ndarray
    config: SoleConfig
    mesh_data: SoleMeshData
    basis: CellBasis
    K: sparse.csc_matrix
    factorization: SuperLU
    plate_inextensibility_constraint: sparse.csc_matrix
    n_plate_rotation_dofs: int
    # Diagnostics.
    rigid_mode_error: float
    reciprocity: ReciprocityErrors
    solve_residuals: TopBottom[float]
    rigid_constraint_residual: float
    inextensibility_residuals: TopBottom[float]
    gauge_residuals: TopBottom[float]
    constraint_rank: int

    @property
    def top_node_ids(self) -> np.ndarray:
        return np.asarray(self.mesh_data.top_selector_nodes, dtype=int)

    @property
    def bottom_node_ids(self) -> np.ndarray:
        return np.asarray(self.mesh_data.bottom_selector_nodes, dtype=int)

    @property
    def plate_node_ids(self) -> np.ndarray:
        return np.asarray(self.mesh_data.plate_node_ids, dtype=int)

    def _coords(self, node_ids: np.ndarray, axis: int) -> np.ndarray:
        return np.asarray(self.mesh_data.mesh.p[axis, node_ids], dtype=float)

    @property
    def x_top(self) -> np.ndarray:
        return self._coords(self.top_node_ids, 0)

    @property
    def y_top(self) -> np.ndarray:
        return self._coords(self.top_node_ids, 1)

    @property
    def x_bottom(self) -> np.ndarray:
        return self._coords(self.bottom_node_ids, 0)

    @property
    def y_bottom(self) -> np.ndarray:
        return self._coords(self.bottom_node_ids, 1)

    @property
    def x_plate(self) -> np.ndarray:
        return self._coords(self.plate_node_ids, 0)

    @property
    def y_plate(self) -> np.ndarray:
        return self._coords(self.plate_node_ids, 1)

    @property
    def n_plate_nodes(self) -> int:
        return int(self.plate_node_ids.size)

    @property
    def n_primal(self) -> int:
        """Size of the enlarged primal vector ``[d_foam; theta_plate]``."""
        return int(self.basis.N) + int(self.n_plate_rotation_dofs)

    @property
    def n_lambda(self) -> int:
        """Number of plate inextensibility multipliers (one per plate element)."""
        return int(self.plate_inextensibility_constraint.shape[0])

    @property
    def reciprocity_error(self) -> float:
        """Relative ``||Cbt - Ctb^T|| / ||Cbt||``."""
        return self.reciprocity.Cbt_Ctb

    @property
    def geometry_metadata(self) -> dict:
        """JSON-ready geometry summary (lengths, regions, toe vertex, mesh quality)."""
        from compliance_fem.geometry.profile import geometry_metadata

        return {
            **geometry_metadata(self.mesh_data.geometry, self.config),
            "shared_toe_node_id": int(self.mesh_data.toe_node_id),
            "mesh_quality": dict(self.mesh_data.quality),
        }

    def node_dofs(self, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(u_dofs, v_dofs)`` of mesh vertices ``node_ids``."""
        nodal = dof_table(self.basis.nodal_dofs)
        ids = np.asarray(node_ids, dtype=int)
        return np.asarray(nodal[0, ids], dtype=int), np.asarray(nodal[1, ids], dtype=int)

    def selector_dofs(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """``(u_dofs, v_dofs)`` of the ``"top"`` or ``"bottom"`` selector, heel to toe."""
        if side == "top":
            return self.node_dofs(self.top_node_ids)
        if side == "bottom":
            return self.node_dofs(self.bottom_node_ids)
        raise ValueError(f"Unknown selector side {side!r}.")

    @property
    def plate_u_dof_ids(self) -> np.ndarray:
        return self.node_dofs(self.plate_node_ids)[0]

    @property
    def plate_v_dof_ids(self) -> np.ndarray:
        return self.node_dofs(self.plate_node_ids)[1]

    @property
    def plate_rotation_dof_ids(self) -> np.ndarray:
        """Plate rotation DOFs (appended after the foam DOFs); empty when ``EI = 0``."""
        return int(self.basis.N) + np.arange(int(self.n_plate_rotation_dofs), dtype=int)


def build_augmented_matrix(
    K: sparse.csc_matrix,
    R: sparse.csc_matrix,
    plate_inextensibility_constraint: sparse.csc_matrix | None = None,
) -> sparse.csc_matrix:
    """Build the symmetric saddle-point matrix.

    Without plate constraints::

        [[K, R], [R^T, 0]]

    With the inextensible plate (``B_p`` = ``plate_inextensibility_constraint``)::

        [[K0, B_p^T, R], [B_p, 0, 0], [R^T, 0, 0]]
    """
    n = K.shape[0]
    n_r = R.shape[1]
    if K.shape[0] != K.shape[1]:
        raise ValueError("Stiffness matrix must be square.")
    if R.shape[0] != n:
        raise ValueError(f"R has {R.shape[0]} rows; expected {n}.")
    # The stubs type bmat() as returning spmatrix whatever the format.
    if plate_inextensibility_constraint is None:
        return cast(
            sparse.csc_matrix,
            sparse.bmat([[K, R], [R.T, sparse.csc_matrix((n_r, n_r))]], format="csc"),
        )
    if plate_inextensibility_constraint.shape[1] != n:
        raise ValueError(f"plate_inextensibility_constraint has {plate_inextensibility_constraint.shape[1]} columns; expected {n}.")
    n_l = plate_inextensibility_constraint.shape[0]
    return cast(
        sparse.csc_matrix,
        sparse.bmat(
            [
                [K, plate_inextensibility_constraint.T, R],
                [plate_inextensibility_constraint, sparse.csc_matrix((n_l, n_l)), sparse.csc_matrix((n_l, n_r))],
                [R.T, sparse.csc_matrix((n_r, n_l)), sparse.csc_matrix((n_r, n_r))],
            ],
            format="csc",
        ),
    )


def _relative_residual(A: sparse.csc_matrix, x: np.ndarray, b: np.ndarray) -> float:
    r = b - A @ x
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    return float(r_norm / max(b_norm, 1e-30))


def solve_multiple_rhs(
    lu: SuperLU,
    rhs: np.ndarray,
    A: sparse.csc_matrix | None = None,
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
    lu: SuperLU,
    A: sparse.csc_matrix,
    S: sparse.csc_matrix,
    label: str,
    n_primal: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if n_primal is None:
        n_primal = A.shape[0] - 3
    if n_primal <= 0 or n_primal > A.shape[0]:
        raise ValueError(f"Invalid primal dimension n_primal={n_primal} for A of size {A.shape[0]}.")
    
    # Add extra zero rows to the rhs so that constraints are satisfied exactly. The extra rows are ignored in the solution
    n_extra = A.shape[0] - n_primal
    if S.shape[1] != n_primal:
        raise ValueError(f"Selector has {S.shape[1]} columns; expected n_primal={n_primal}.")
    rhs = np.vstack([S.T.toarray(), np.zeros((n_extra, S.shape[0]))])
    solution = solve_multiple_rhs(lu, rhs, A=A, n_refine=2)
    U = solution[:n_primal, :]
    extras = solution[n_primal:, :]
    # Verify residual is sufficiently small
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


def _normwise_backward_error(A: sparse.csc_matrix, X: np.ndarray, Bmat: np.ndarray) -> float:
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


def _plate_compliance_core(
    *,
    config: SoleConfig,
    K_foam: sparse.csc_matrix,
    basis: CellBasis,
    mesh_data: SoleMeshData,
) -> ComplianceResult:
    """Free-body plate compliance of the two-foam sole.

    Plate rotation DOFs exist only for the plate nodes (and only when ``EI > 0``); the
    selectors act on translational foam DOFs only.
    """
    verify_stiffness_symmetry(K_foam)
    node_coords = np.asarray(mesh_data.mesh.p, dtype=float)
    plate_node_ids = np.asarray(mesh_data.plate_node_ids, dtype=int)
    plate_coords = node_coords[:, plate_node_ids]
    _, plate_tangents, plate_normals = plate_element_frames(plate_coords)
    u_dofs, v_dofs = plate_translational_dofs(basis, plate_node_ids)
    n_plate = int(u_dofs.size)  # Number of plate nodes
    n_foam = basis.N  # Number of foam DOFs
    n_theta = n_plate if config.EI_plate_Nm2_per_m > 0.0 else 0
    n_q = n_foam + n_theta
    K_plate = assemble_plate_bending(n_foam, u_dofs, v_dofs, plate_coords, plate_normals, config.EI_plate_Nm2_per_m)
    if K_plate.shape != (n_q, n_q):
        raise ValueError(f"Plate stiffness has shape {K_plate.shape}, expected ({n_q}, {n_q}).")
    K0 = enlarge_foam_stiffness(K_foam, n_theta) + K_plate
    verify_stiffness_symmetry(K0)

    plate_inextensibility_constraint = assemble_axial_constraints(n_q, u_dofs, v_dofs, plate_tangents, n_theta)
    rank = verify_constraint_rank(plate_inextensibility_constraint, n_plate)

    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    rigid_mode_error = verify_rigid_modes(K0, R)
    rigid_constraint_residual = verify_constraint_on_modes(plate_inextensibility_constraint, R)

    nodal = dof_table(basis.nodal_dofs)
    top_nodes = np.asarray(mesh_data.top_selector_nodes, dtype=int)
    bottom_nodes = np.asarray(mesh_data.bottom_selector_nodes, dtype=int)
    u_top, v_top = np.asarray(nodal[0, top_nodes], dtype=int), np.asarray(nodal[1, top_nodes], dtype=int)
    u_bot, v_bot = np.asarray(nodal[0, bottom_nodes], dtype=int), np.asarray(nodal[1, bottom_nodes], dtype=int)
    S = TopBottom(build_vector_selector(u_top, v_top, n_q), build_vector_selector(u_bot, v_bot, n_q))
    _verify_selector_normalization(S.top)
    _verify_selector_normalization(S.bottom)
    if n_theta > 0:
        if sparse.linalg.norm(S.top[:, n_foam:]) != 0.0 or sparse.linalg.norm(S.bottom[:, n_foam:]) != 0.0:
            raise ValueError("Boundary selectors must not include plate rotation DOFs.")
    # The top and bottom selectors must be disjoint (no shared DOFs).
    shared = np.intersect1d(np.concatenate([u_top, v_top]), np.concatenate([u_bot, v_bot]))
    if shared.size:
        raise ValueError(f"Top and bottom selectors share {shared.size} DOFs; they must be disjoint.")

    # Build and factor the full system
    A = build_augmented_matrix(K0, R, plate_inextensibility_constraint)
    lu = splu(A)
    U_top, res_top, _ = _solve_compliance_block(lu, A, S.top, "top forces", n_primal=n_q)
    U_bottom, res_bottom, _ = _solve_compliance_block(lu, A, S.bottom, "bottom forces", n_primal=n_q)

    def _max_col_norm(mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        return float(max(np.linalg.norm(mat[:, j]) for j in range(mat.shape[1])))

    def _residuals(operator) -> TopBottom[float]:
        return TopBottom(_max_col_norm(np.asarray(operator @ U_top)), _max_col_norm(np.asarray(operator @ U_bottom)))

    Ctt_force = np.asarray(S.top @ U_top)
    Ctb_force = np.asarray(S.top @ U_bottom)
    Cbt_force = np.asarray(S.bottom @ U_top)
    Cbb_force = np.asarray(S.bottom @ U_bottom)
    return ComplianceResult(
        Ctt_force=Ctt_force,
        Ctb_force=Ctb_force,
        Cbt_force=Cbt_force,
        Cbb_force=Cbb_force,
        config=config,
        mesh_data=mesh_data,
        basis=basis,
        K=K0,
        factorization=lu,
        plate_inextensibility_constraint=plate_inextensibility_constraint,
        n_plate_rotation_dofs=n_theta,
        rigid_mode_error=rigid_mode_error,
        reciprocity=_reciprocity_errors(Ctt_force, Cbb_force, Cbt_force, Ctb_force),
        solve_residuals=TopBottom(res_top, res_bottom),
        rigid_constraint_residual=rigid_constraint_residual,
        inextensibility_residuals=_residuals(plate_inextensibility_constraint),
        gauge_residuals=_residuals(R.T),
        constraint_rank=rank,
    )


def compute_compliance(config: SoleConfig, mesh_path: Path | None = None) -> ComplianceResult:
    """Mesh the two-foam sole with its partial curved plate and compute the nodal-force compliance blocks.

    Selectors come from mesh tags in heel-to-toe material order; the shared toe
    vertex belongs to the bottom/contact selector only
    (``shared_toe_policy = bottom_contact_owns_toe_vertex``).
    """
    from compliance_fem.geometry.profile import REGION_LOWER, REGION_UPPER
    from compliance_fem.geometry.mesh import generate_mesh

    mesh_data = generate_mesh(config, output_msh=mesh_path)
    region_elements = {
        REGION_UPPER: mesh_data.region_elements(REGION_UPPER),
        REGION_LOWER: mesh_data.region_elements(REGION_LOWER),
    }
    K_foam, basis, _ = assemble_region_foam_stiffness(
        mesh_data.mesh, region_elements, config.region_materials(), config.shoe_length_m, config.element_order
    )
    if int(mesh_data.toe_node_id) in set(np.asarray(mesh_data.top_selector_nodes).tolist()):
        raise ValueError("The shared toe vertex must not be in the top selector.")
    return _plate_compliance_core(config=config, K_foam=K_foam, basis=basis, mesh_data=mesh_data)


def compliance_npz_payload(result: ComplianceResult) -> dict:
    """NPZ dictionary of a compliance result (the ``--save-fem`` debug output)."""
    import json

    cfg = result.config
    upper = cfg.material(cfg.upper_foam_material)
    lower = cfg.material(cfg.lower_foam_material)
    return {
        "compliance_schema_version": COMPLIANCE_SCHEMA_VERSION,
        "dof_ordering": np.asarray(DOF_ORDERING_COMPONENT_MAJOR_UV),
        "Cbt_force": result.Cbt_force,
        "Cbb_force": result.Cbb_force,
        "Ctb_force": result.Ctb_force,
        "Ctt_force": result.Ctt_force,
        "top_node_ids": result.top_node_ids,
        "bottom_node_ids": result.bottom_node_ids,
        "plate_node_ids": result.plate_node_ids,
        "x_top": result.x_top,
        "y_top": result.y_top,
        "x_bottom": result.x_bottom,
        "y_bottom": result.y_bottom,
        "x_plate": result.x_plate,
        "y_plate": result.y_plate,
        "L": cfg.shoe_length_m,
        "H": cfg.shoe_height_m,
        **{name: getattr(cfg, name) for name in SOLE_SCALAR_FIELDS},
        "upper_foam_material": np.asarray(upper.name),
        "lower_foam_material": np.asarray(lower.name),
        "upper_E_heel": upper.E_heel,
        "upper_E_toe": upper.E_toe,
        "upper_nu": upper.nu,
        "lower_E_heel": lower.E_heel,
        "lower_E_toe": lower.E_toe,
        "lower_nu": lower.nu,
        "plate_axial_model": np.asarray("exact_inextensible"),
        "plate_tangent_model": np.asarray("per_element_chord"),
        "number_of_plate_rotation_dofs": result.n_plate_rotation_dofs,
        "number_of_plate_constraints": result.n_lambda,
        "constraint_rank": result.constraint_rank,
        "rigid_mode_error": result.rigid_mode_error,
        "rigid_constraint_residual": result.rigid_constraint_residual,
        "reciprocity_error": result.reciprocity_error,
        "symmetry_error_Cbb": result.reciprocity.Cbb,
        "symmetry_error_Ctt": result.reciprocity.Ctt,
        "solve_residual_top": result.solve_residuals.top,
        "solve_residual_bottom": result.solve_residuals.bottom,
        "inextensibility_residual_top": result.inextensibility_residuals.top,
        "inextensibility_residual_bottom": result.inextensibility_residuals.bottom,
        "geometry_metadata_json": np.asarray(json.dumps(result.geometry_metadata, sort_keys=True)),
        "normalized_geometry_json": np.asarray(json.dumps(cfg.normalized_geometry.to_payload(), sort_keys=True)),
    }


def save_compliance_npz(result: ComplianceResult, path) -> None:
    """Write a compressed compliance NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **compliance_npz_payload(result))
