"""Layered-plate compliance operator tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.assembly import assemble_layered_foam_stiffness
from compliance_fem.boundaries import build_vector_selector, vector_boundary_data
from compliance_fem.compliance import compute_compliance, save_compliance_npz
from compliance_fem.config import LayeredPlateConfig
from compliance_fem.constraints import assemble_axial_constraints
from compliance_fem.layered_geometry import generate_layered_mesh
from compliance_fem.plate import (
    assemble_plate_bending,
    enlarge_foam_stiffness,
    plate_translational_dofs,
)
from compliance_fem.rigid_modes import (
    build_enlarged_rigid_modes,
    verify_constraint_on_modes,
    verify_rigid_modes,
)


@pytest.fixture
def layered_config() -> LayeredPlateConfig:
    return LayeredPlateConfig(
        L=0.30,
        h1_heel=0.025,
        h1_toe=0.015,
        h2_heel=0.020,
        h2_toe=0.030,
        E1=2.0e6,
        nu1=0.30,
        E_heel=5.0e5,
        E_toe=1.5e6,
        nu2=0.30,
        EI_plate=10.0,
        nx=6,
        ny1=2,
        ny2=2,
        element_order=1,
    )


@pytest.fixture
def layered_result(layered_config: LayeredPlateConfig):
    return compute_compliance(layered_config)


def test_K0_and_Bp_on_rigid_modes(layered_config: LayeredPlateConfig) -> None:
    mesh_data = generate_layered_mesh(layered_config)
    K_foam, basis, _, _ = assemble_layered_foam_stiffness(mesh_data.mesh, layered_config)
    u_dofs, v_dofs = plate_translational_dofs(basis, mesh_data.plate_node_indices)
    n_plate = len(u_dofs)
    n_theta = n_plate
    n_q = basis.N + n_theta
    coords = np.vstack([mesh_data.x_plate, mesh_data.y_plate])
    K_plate = assemble_plate_bending(
        basis.N, u_dofs, v_dofs, coords, layered_config.plate_normal, layered_config.EI_plate
    )
    K0 = enlarge_foam_stiffness(K_foam, n_theta) + K_plate
    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, layered_config.plate_tangent, n_theta)
    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    assert verify_rigid_modes(K0, R) < 1e-8
    assert verify_constraint_on_modes(B, R) < 1e-8


def test_layered_compliance_residuals(layered_result) -> None:
    assert layered_result.solve_residuals["top"] < 1e-8
    assert layered_result.solve_residuals["bottom"] < 1e-8
    assert layered_result.inextensibility_residuals["top"] < 1e-8
    assert layered_result.inextensibility_residuals["bottom"] < 1e-8
    assert layered_result.gauge_residuals["top"] < 1e-8
    assert layered_result.gauge_residuals["bottom"] < 1e-8
    assert layered_result.rigid_mode_error < 1e-8
    assert layered_result.rigid_constraint_residual < 1e-8


def test_layered_reciprocity(layered_result) -> None:
    assert layered_result.symmetry_errors["Ctt_force"] < 1e-8
    assert layered_result.symmetry_errors["Cbb_force"] < 1e-8
    assert layered_result.reciprocity_error < 1e-8


def test_selectors_omit_plate_rotations(layered_result) -> None:
    basis = layered_result.basis
    n_q = layered_result.n_primal
    assert n_q is not None
    u_dofs, v_dofs, _, _ = vector_boundary_data(basis, "top")
    S = build_vector_selector(u_dofs, v_dofs, n_q)
    n_foam = basis.N
    assert S.shape[1] == n_q
    assert S.shape[0] == 2 * len(u_dofs)
    if n_q > n_foam:
        assert np.allclose(S[:, n_foam:].toarray(), 0.0)


def test_compliance_file_roundtrip(layered_result, tmp_path) -> None:
    path = tmp_path / "compliance_results.npz"
    save_compliance_npz(layered_result, path)
    data = np.load(path)
    for key in (
        "Ctt_force",
        "Ctb_force",
        "Cbt_force",
        "Cbb_force",
        "Cbt_traction",
        "Cbb_traction",
        "M_top",
        "M_bottom",
        "x_top",
        "x_bottom",
        "y_top",
        "y_bottom",
        "x_plate",
        "y_plate",
        "L",
        "H",
        "E",
        "nu",
        "geometry_type",
        "plate_axial_model",
        "EI_plate",
        "number_of_plate_nodes",
        "constraint_rank",
        "dof_ordering",
        "n_top_nodes",
        "n_bottom_nodes",
    ):
        assert key in data.files
    np.testing.assert_allclose(data["Ctt_force"], layered_result.Ctt_force)
    assert str(np.asarray(data["geometry_type"]).item()) == layered_result.config.geometry_type
    assert layered_result.n_plate_nodes >= 2
    assert layered_result.constraint_rank == layered_result.n_plate_nodes - 1
    assert layered_result.n_lambda == layered_result.n_plate_nodes - 1
