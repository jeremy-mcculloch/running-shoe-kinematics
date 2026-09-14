"""Compliance matrix computation via the rigid-mode saddle-point system."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from compliance_fem.assembly import (
    assemble_layered_foam_stiffness,
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
from compliance_fem.config import LayeredPlateConfig, ProblemConfig
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
        raise ValueError(f"Saddle-point solve residual for {label} is too large: {residual:.3e}.")
    return U, residual, extras


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

    M_top, _ = assemble_boundary_mass_matrix(basis, "top", order)
    M_bottom, _ = assemble_boundary_mass_matrix(basis, "bottom", order)
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
    verify_stiffness_symmetry(K_foam)

    u_dofs, v_dofs = plate_translational_dofs(basis, mesh_data.plate_node_indices)
    n_plate = int(u_dofs.size)
    n_foam = basis.N
    n_theta = n_plate if config.EI_plate > 0.0 else 0
    n_q = n_foam + n_theta
    coords = np.vstack([mesh_data.x_plate, mesh_data.y_plate])
    K_plate = assemble_plate_bending(
        n_foam,
        u_dofs,
        v_dofs,
        coords,
        config.plate_normal,
        config.EI_plate,
    )
    if K_plate.shape != (n_q, n_q):
        raise ValueError(f"Plate stiffness has shape {K_plate.shape}, expected ({n_q}, {n_q}).")
    K0 = enlarge_foam_stiffness(K_foam, n_theta) + K_plate
    verify_stiffness_symmetry(K0)

    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, config.plate_tangent, n_theta)
    rank = verify_constraint_rank(B, n_plate)
    verify_constant_tangential_nullspace(B, u_dofs, v_dofs, config.plate_tangent)

    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    rigid_mode_error = verify_rigid_modes(K0, R)
    rigid_constraint_residual = verify_constraint_on_modes(B, R)

    u_top, v_top, x_top, y_top = vector_boundary_data(basis, "top")
    u_bottom, v_bottom, x_bottom, y_bottom = vector_boundary_data(basis, "bottom")
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bottom, v_bottom, n_q)
    _verify_selector_normalization(S_top)
    _verify_selector_normalization(S_bottom)
    if S_top.shape[1] != n_q or S_bottom.shape[1] != n_q:
        raise ValueError("Boundary selectors must include zero columns for plate rotations.")
    if n_theta > 0:
        if sparse.linalg.norm(S_top[:, n_foam:]) != 0.0 or sparse.linalg.norm(S_bottom[:, n_foam:]) != 0.0:
            raise ValueError("Boundary selectors must not include plate rotation DOFs.")

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
        config.element_order,
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
        K=K0,
        basis=basis,
        factorization=lu,
        Ctt_traction=blocks["Ctt_traction"],
        Ctb_traction=blocks["Ctb_traction"],
        y_top=blocks["y_top"],
        y_bottom=blocks["y_bottom"],
        x_plate=np.asarray(mesh_data.x_plate, dtype=float),
        y_plate=np.asarray(mesh_data.y_plate, dtype=float),
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
        compliance_schema_version=COMPLIANCE_SCHEMA_VERSION_LAYERED,
        dof_ordering=DOF_ORDERING_COMPONENT_MAJOR_UV,
        n_top_nodes=int(len(x_top)),
        n_bottom_nodes=int(len(x_bottom)),
    )


def compute_compliance(
    config: ProblemConfig | LayeredPlateConfig,
    mesh_data=None,
) -> ComplianceResult:
    """Generate mesh, assemble K, and compute nodal-force compliance blocks."""
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
    return payload


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
    )
