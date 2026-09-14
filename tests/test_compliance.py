"""Compliance matrix tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig


@pytest.fixture
def compliance_result():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=16, ny=8, order=1)
    return compute_compliance(config)


def test_saddle_point_residuals(compliance_result) -> None:
    assert compliance_result.solve_residuals["top"] < 1e-8
    assert compliance_result.solve_residuals["bottom"] < 1e-8


def test_compliance_symmetry(compliance_result) -> None:
    assert compliance_result.symmetry_errors["Cbb_force"] < 1e-10
    assert compliance_result.symmetry_errors["Ctt_force"] < 1e-10


def test_reciprocity(compliance_result) -> None:
    assert compliance_result.reciprocity_error < 1e-10


def test_compliance_scaling() -> None:
    base = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=12, ny=6, order=1)
    scaled = ProblemConfig(L=1.0, H=0.5, E=2.0e6, nu=0.3, nx=12, ny=6, order=1)
    C_base = compute_compliance(base)
    C_scaled = compute_compliance(scaled)
    ratio = np.linalg.norm(C_base.Cbt_force) / np.linalg.norm(C_scaled.Cbt_force)
    assert abs(ratio - 2.0) < 0.05


def test_vector_block_shapes(compliance_result) -> None:
    n_top = len(compliance_result.x_top)
    n_bottom = len(compliance_result.x_bottom)
    assert compliance_result.dof_ordering == "component_major_uv"
    assert compliance_result.Ctt_force.shape == (2 * n_top, 2 * n_top)
    assert compliance_result.Cbb_force.shape == (2 * n_bottom, 2 * n_bottom)
    assert compliance_result.Ctb_force.shape == (2 * n_top, 2 * n_bottom)
    assert compliance_result.Cbt_force.shape == (2 * n_bottom, 2 * n_top)
    assert compliance_result.Ctt_force.shape[0] % 2 == 0
    assert compliance_result.Cbt_traction.shape == (n_bottom, n_top)
    assert compliance_result.Cbb_traction.shape == (n_bottom, n_bottom)
    assert compliance_result.M_top.shape == (n_top, n_top)
    assert compliance_result.M_bottom.shape == (n_bottom, n_bottom)
