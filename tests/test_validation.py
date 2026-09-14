"""Analytical validation and mesh-convergence tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.boundaries import assemble_traction_load
from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.validation import (
    analytical_top_displacement,
    run_standard_validations,
    solve_bottom_contact,
    validate_traction_case,
)


@pytest.fixture
def fine_result():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=40, ny=20, order=1)
    return compute_compliance(config)


def test_uniform_traction_validation(fine_result) -> None:
    report = validate_traction_case(
        fine_result,
        "uniform",
        lambda x: np.full_like(x, 1000.0, dtype=float),
    )
    assert report.displacement_rel_error < 1e-3
    assert report.force_balance_error < 1e-10
    assert report.moment_balance_error < 1e-10


def test_affine_traction_validation(fine_result) -> None:
    report = validate_traction_case(
        fine_result,
        "affine",
        lambda x: 800.0 + 500.0 * x,
    )
    assert report.displacement_rel_error < 1e-3
    assert report.force_balance_error < 1e-10
    assert report.moment_balance_error < 1e-10


def test_standard_validation_suite(fine_result) -> None:
    reports = run_standard_validations(fine_result)
    assert len(reports) == 2
    for report in reports:
        assert report.displacement_rel_error < 1e-3


def test_smooth_traction_mesh_convergence() -> None:
    """Displacement under smooth traction should converge with mesh refinement."""
    traction = lambda x: 500.0 * np.sin(np.pi * x)

    def max_error(nx: int) -> float:
        config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=nx, ny=nx // 2, order=1)
        result = compute_compliance(config)
        F_top = assemble_traction_load(result.basis, "top", traction, config.order)
        _, _, _, v_top = solve_bottom_contact(result, F_top)
        v_exact = analytical_top_displacement(
            result.x_top,
            traction,
            config.E,
            config.nu,
            config.H,
        )
        return float(np.linalg.norm(v_top - v_exact) / np.linalg.norm(v_exact))

    err_coarse = max_error(12)
    err_fine = max_error(48)
    assert err_fine < err_coarse
