"""Axial inextensibility constraint tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.config import LayeredPlateConfig
from compliance_fem.constraints import (
    assemble_axial_constraints,
    constraint_rank,
    interface_tangential_mode,
    verify_constant_tangential_nullspace,
    verify_constraint_rank,
)
from compliance_fem.layered_geometry import generate_layered_mesh
from compliance_fem.plate import plate_translational_dofs
from compliance_fem.assembly import create_basis
from compliance_fem.rigid_modes import build_enlarged_rigid_modes, verify_constraint_on_modes


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


def _plate_system(config: LayeredPlateConfig):
    mesh_data = generate_layered_mesh(config)
    basis = create_basis(mesh_data.mesh, config.element_order)
    u_dofs, v_dofs = plate_translational_dofs(basis, mesh_data.plate_node_indices)
    n_plate = len(u_dofs)
    n_theta = n_plate
    n_q = basis.N + n_theta
    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, config.plate_tangent, n_theta)
    return mesh_data, basis, u_dofs, v_dofs, B, n_plate, n_theta, n_q


def test_constraint_count_and_rank(layered_config: LayeredPlateConfig) -> None:
    _, _, u_dofs, v_dofs, B, n_plate, n_theta, n_q = _plate_system(layered_config)
    assert B.shape == (n_plate - 1, n_q)
    assert verify_constraint_rank(B, n_plate) == n_plate - 1
    assert constraint_rank(B) == n_plate - 1
    if n_theta > 0:
        assert np.allclose(B[:, -n_theta:].toarray(), 0.0)


def test_constant_tangential_accepted(layered_config: LayeredPlateConfig) -> None:
    _, _, u_dofs, v_dofs, B, _, _, _ = _plate_system(layered_config)
    residual = verify_constant_tangential_nullspace(
        B, u_dofs, v_dofs, layered_config.plate_tangent
    )
    assert residual < 1e-12


def test_nonuniform_tangential_rejected(layered_config: LayeredPlateConfig) -> None:
    mesh_data, _, u_dofs, v_dofs, B, _, _, n_q = _plate_system(layered_config)
    t = layered_config.plate_tangent
    q = np.zeros(n_q)
    x = mesh_data.x_plate
    q[u_dofs] = t[0] * x
    q[v_dofs] = t[1] * x
    residual = float(np.linalg.norm(B @ q))
    assert residual > 1e-8


def test_no_mean_tangential_pin(layered_config: LayeredPlateConfig) -> None:
    _, _, u_dofs, v_dofs, B, n_plate, _, n_q = _plate_system(layered_config)
    assert B.shape[0] == n_plate - 1
    mean_row = interface_tangential_mode(n_q, u_dofs, v_dofs, layered_config.plate_tangent)
    B_pinned = np.vstack([B.toarray(), mean_row])
    assert np.linalg.matrix_rank(B_pinned) == n_plate
    assert constraint_rank(B) == n_plate - 1


def test_Bp_on_rigid_modes(layered_config: LayeredPlateConfig) -> None:
    _, basis, _, _, B, _, n_theta, _ = _plate_system(layered_config)
    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    err = verify_constraint_on_modes(B, R)
    assert err < 1e-10
