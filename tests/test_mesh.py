"""Mesh generation and boundary identification tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.config import ProblemConfig
from compliance_fem.geometry import (
    BOUNDARY_NAMES,
    boundary_node_coordinates,
    generate_rectangular_mesh,
    verify_mesh_dimensions,
    verify_positive_jacobians,
)


@pytest.fixture
def coarse_config() -> ProblemConfig:
    return ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=8, ny=4, order=1)


def test_mesh_dimensions(coarse_config: ProblemConfig) -> None:
    mesh_data = generate_rectangular_mesh(coarse_config)
    verify_mesh_dimensions(mesh_data.mesh, coarse_config)
    assert mesh_data.mesh.p.shape[0] == 2


def test_physical_boundaries_present(coarse_config: ProblemConfig) -> None:
    mesh = generate_rectangular_mesh(coarse_config).mesh
    for name in BOUNDARY_NAMES:
        assert name in mesh.boundaries


def test_top_bottom_matching_horizontal_coordinates(coarse_config: ProblemConfig) -> None:
    mesh = generate_rectangular_mesh(coarse_config).mesh
    x_top, _ = boundary_node_coordinates(mesh, "top")
    x_bottom, _ = boundary_node_coordinates(mesh, "bottom")
    np.testing.assert_allclose(x_top, x_bottom, atol=1e-12)


def test_positive_element_jacobians(coarse_config: ProblemConfig) -> None:
    mesh = generate_rectangular_mesh(coarse_config).mesh
    verify_positive_jacobians(mesh)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("element_type", ["quad", "tri"])
def test_mesh_generation_orders(order: int, element_type: str) -> None:
    config = ProblemConfig(
        L=1.0,
        H=0.5,
        E=1.0e6,
        nu=0.3,
        nx=4,
        ny=2,
        order=order,
        element_type=element_type,
    )
    mesh = generate_rectangular_mesh(config).mesh
    verify_mesh_dimensions(mesh, config)
    verify_positive_jacobians(mesh)
