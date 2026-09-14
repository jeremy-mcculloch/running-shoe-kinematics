"""Layered trapezoid geometry helpers and Gmsh mesh tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.config import LayeredPlateConfig
from compliance_fem.geometry import boundary_node_coordinates, verify_positive_jacobians
from compliance_fem.layered_geometry import (
    LAYERED_BOUNDARY_NAMES,
    generate_layered_mesh,
    ordered_interface_nodes,
    verify_layered_mesh,
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
        nx=8,
        ny1=3,
        ny2=3,
        element_order=1,
    )


def test_height_functions_at_ends(layered_config: LayeredPlateConfig) -> None:
    assert float(layered_config.h1(0.0)) == pytest.approx(layered_config.h1_heel)
    assert float(layered_config.h1(layered_config.L)) == pytest.approx(layered_config.h1_toe)
    assert float(layered_config.h2(0.0)) == pytest.approx(layered_config.h2_heel)
    assert float(layered_config.h2(layered_config.L)) == pytest.approx(layered_config.h2_toe)
    assert float(layered_config.y_plate(0.0)) == pytest.approx(layered_config.h2_heel)
    assert float(layered_config.y_top(layered_config.L)) == pytest.approx(
        layered_config.h2_toe + layered_config.h1_toe
    )


def test_polygon_vertices(layered_config: LayeredPlateConfig) -> None:
    assert layered_config.lower_vertices == (
        (0.0, 0.0),
        (layered_config.L, 0.0),
        (layered_config.L, layered_config.h2_toe),
        (0.0, layered_config.h2_heel),
    )
    assert layered_config.upper_vertices == (
        (0.0, layered_config.h2_heel),
        (layered_config.L, layered_config.h2_toe),
        (layered_config.L, layered_config.h2_toe + layered_config.h1_toe),
        (0.0, layered_config.h2_heel + layered_config.h1_heel),
    )


def test_plate_tangent_normal_length(layered_config: LayeredPlateConfig) -> None:
    m = layered_config.plate_slope
    gamma = layered_config.plate_gamma
    t = layered_config.plate_tangent
    n = layered_config.plate_normal
    np.testing.assert_allclose(gamma, np.sqrt(1.0 + m**2))
    np.testing.assert_allclose(layered_config.plate_length, gamma * layered_config.L)
    np.testing.assert_allclose(t, np.array([1.0, m]) / gamma)
    np.testing.assert_allclose(n, np.array([-m, 1.0]) / gamma)
    np.testing.assert_allclose(np.linalg.norm(t), 1.0)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)
    np.testing.assert_allclose(np.dot(t, n), 0.0, atol=1e-15)


def test_poisson_locking_warning() -> None:
    with pytest.warns(UserWarning, match="lock"):
        LayeredPlateConfig(
            L=0.3,
            h1_heel=0.02,
            h1_toe=0.02,
            h2_heel=0.02,
            h2_toe=0.02,
            E1=1.0e6,
            nu1=0.49,
            E_heel=1.0e6,
            E_toe=1.0e6,
            nu2=0.3,
            EI_plate=1.0,
            nx=2,
            ny1=1,
            ny2=1,
        )


def test_invalid_thickness_rejected() -> None:
    with pytest.raises(ValueError, match="h1_heel"):
        LayeredPlateConfig(
            L=0.3,
            h1_heel=0.0,
            h1_toe=0.02,
            h2_heel=0.02,
            h2_toe=0.02,
            E1=1.0e6,
            nu1=0.3,
            E_heel=1.0e6,
            E_toe=1.0e6,
            nu2=0.3,
            EI_plate=1.0,
            nx=2,
            ny1=1,
            ny2=1,
        )


def test_layered_mesh_groups_and_interface(layered_config: LayeredPlateConfig) -> None:
    mesh_data = generate_layered_mesh(layered_config)
    verify_layered_mesh(mesh_data)
    verify_positive_jacobians(mesh_data.mesh)
    mesh = mesh_data.mesh
    for name in ("upper_foam", "lower_foam"):
        assert name in mesh.subdomains
    for name in LAYERED_BOUNDARY_NAMES:
        assert name in mesh.boundaries

    nodes, x_plate, y_plate = ordered_interface_nodes(mesh, layered_config)
    np.testing.assert_array_equal(nodes, mesh_data.plate_node_indices)
    np.testing.assert_allclose(x_plate, np.sort(x_plate))
    np.testing.assert_allclose(y_plate, layered_config.y_plate(x_plate), atol=1e-8)
    assert np.all(np.diff(x_plate) > 0.0)


def test_conforming_shared_interface_nodes(layered_config: LayeredPlateConfig) -> None:
    mesh_data = generate_layered_mesh(layered_config)
    mesh = mesh_data.mesh
    upper = set(np.unique(mesh.t[:, mesh.subdomains["upper_foam"]].reshape(-1)))
    lower = set(np.unique(mesh.t[:, mesh.subdomains["lower_foam"]].reshape(-1)))
    assert set(mesh_data.plate_node_indices.tolist()) == upper & lower


def test_top_bottom_horizontal_coordinates(layered_config: LayeredPlateConfig) -> None:
    mesh = generate_layered_mesh(layered_config).mesh
    x_top, _ = boundary_node_coordinates(mesh, "top")
    x_bottom, _ = boundary_node_coordinates(mesh, "bottom")
    np.testing.assert_allclose(x_top, x_bottom, atol=1e-8)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("element_type", ["quad", "tri"])
def test_layered_mesh_orders(order: int, element_type: str) -> None:
    config = LayeredPlateConfig(
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
        nx=4,
        ny1=2,
        ny2=2,
        element_order=order,
        element_type=element_type,  # type: ignore[arg-type]
    )
    mesh_data = generate_layered_mesh(config)
    verify_layered_mesh(mesh_data)
    verify_positive_jacobians(mesh_data.mesh)
