"""Stiffness assembly and rigid-mode tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.assembly import assemble_stiffness, verify_stiffness_symmetry
from compliance_fem.config import ProblemConfig
from compliance_fem.geometry import generate_rectangular_mesh
from compliance_fem.rigid_modes import build_rigid_modes, verify_rigid_modes


@pytest.fixture
def assembled_system():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=10, ny=5, order=1)
    mesh = generate_rectangular_mesh(config).mesh
    K, basis = assemble_stiffness(mesh, config.E, config.nu, config.order)
    R, _, _ = build_rigid_modes(basis)
    return K, R


def test_stiffness_symmetry(assembled_system) -> None:
    K, _ = assembled_system
    err = verify_stiffness_symmetry(K)
    assert err < 1e-12


def test_rigid_modes_in_kernel(assembled_system) -> None:
    K, R = assembled_system
    err = verify_rigid_modes(K, R)
    assert err < 1e-10


def test_rigid_mode_matrix_shape(assembled_system) -> None:
    _, R = assembled_system
    assert R.shape[1] == 3
