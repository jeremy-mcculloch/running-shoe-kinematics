"""Axial inextensibility constraint tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.fem.assembly import assemble_region_foam_stiffness
from compliance_fem.fem.constraints import assemble_axial_constraints, constraint_rank, verify_constraint_rank
from compliance_fem.geometry.profile import REGION_LOWER, REGION_UPPER
from compliance_fem.geometry.mesh import generate_mesh
from compliance_fem.fem.plate import plate_element_frames, plate_translational_dofs
from compliance_fem.fem.rigid_modes import build_enlarged_rigid_modes, verify_constraint_on_modes

from conftest import small_config


@pytest.fixture(scope="module")
def plate_system():
    config = small_config(rocker=0.008)
    mesh_data = generate_mesh(config)
    regions = {name: mesh_data.region_elements(name) for name in (REGION_UPPER, REGION_LOWER)}
    _, basis, _ = assemble_region_foam_stiffness(
        mesh_data.mesh, regions, config.region_materials(), config.L, config.element_order
    )
    plate_nodes = np.asarray(mesh_data.plate_node_ids, dtype=int)
    coords = np.asarray(mesh_data.mesh.p, dtype=float)[:, plate_nodes]
    _, tangents, _ = plate_element_frames(coords)
    u_dofs, v_dofs = plate_translational_dofs(basis, plate_nodes)
    n_plate = len(u_dofs)
    n_theta = n_plate
    n_q = basis.N + n_theta
    B = assemble_axial_constraints(n_q, u_dofs, v_dofs, tangents, n_theta)
    return coords, basis, u_dofs, v_dofs, B, n_plate, n_theta, n_q


def test_constraint_count_and_rank(plate_system) -> None:
    _, _, _, _, B, n_plate, n_theta, n_q = plate_system
    assert B.shape == (n_plate - 1, n_q)
    assert verify_constraint_rank(B, n_plate) == n_plate - 1
    assert constraint_rank(B) == n_plate - 1
    assert np.allclose(B[:, -n_theta:].toarray(), 0.0)


def test_stretching_along_the_plate_rejected(plate_system) -> None:
    coords, _, u_dofs, v_dofs, B, _, _, n_q = plate_system
    q = np.zeros(n_q)
    q[u_dofs] = coords[0]
    q[v_dofs] = coords[1]
    assert float(np.linalg.norm(B @ q)) > 1e-8


def test_Bp_on_rigid_modes(plate_system) -> None:
    _, basis, _, _, B, _, n_theta, _ = plate_system
    R, _, _ = build_enlarged_rigid_modes(basis, n_theta, rotation_theta=1.0)
    assert verify_constraint_on_modes(B, R) < 1e-10
