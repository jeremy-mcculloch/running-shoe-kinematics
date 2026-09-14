"""Lower-foam spatially varying modulus and foam stiffness tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.assembly import (
    assemble_layered_foam_stiffness,
    exact_lower_modulus_integral,
    integrate_lower_modulus,
    verify_stiffness_symmetry,
)
from compliance_fem.config import LayeredPlateConfig
from compliance_fem.layered_geometry import generate_layered_mesh


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


def test_E2_end_values(layered_config: LayeredPlateConfig) -> None:
    assert float(layered_config.E2(0.0)) == pytest.approx(layered_config.E_heel)
    assert float(layered_config.E2(layered_config.L)) == pytest.approx(layered_config.E_toe)


def test_quadrature_variation_of_E2(layered_config: LayeredPlateConfig) -> None:
    mesh = generate_layered_mesh(layered_config).mesh
    assembled = integrate_lower_modulus(mesh, layered_config)
    exact = exact_lower_modulus_integral(layered_config)
    assert assembled == pytest.approx(exact, rel=1e-6, abs=1e-8)
    uniform = layered_config.E_heel * exact_lower_modulus_integral(
        LayeredPlateConfig(
            L=layered_config.L,
            h1_heel=layered_config.h1_heel,
            h1_toe=layered_config.h1_toe,
            h2_heel=layered_config.h2_heel,
            h2_toe=layered_config.h2_toe,
            E1=layered_config.E1,
            nu1=layered_config.nu1,
            E_heel=layered_config.E_heel,
            E_toe=layered_config.E_heel,
            nu2=layered_config.nu2,
            EI_plate=layered_config.EI_plate,
            nx=layered_config.nx,
            ny1=layered_config.ny1,
            ny2=layered_config.ny2,
        )
    )
    assert assembled != pytest.approx(uniform, rel=1e-3)


def test_foam_stiffness_symmetry(layered_config: LayeredPlateConfig) -> None:
    mesh = generate_layered_mesh(layered_config).mesh
    K, _, K1, K2 = assemble_layered_foam_stiffness(mesh, layered_config)
    verify_stiffness_symmetry(K)
    verify_stiffness_symmetry(K1)
    verify_stiffness_symmetry(K2)
    assert K.shape[0] == K.shape[1]
    assert K.shape == K1.shape == K2.shape
