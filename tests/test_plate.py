"""Hermite plate element tests (no volume mesh required)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.plate import (
    assemble_plate_bending,
    element_transformation,
    hermite_bending_matrix,
    plate_normal_deflection,
)


def test_hermite_matrix_symmetry() -> None:
    K = hermite_bending_matrix(EI=12.0, length=0.4)
    np.testing.assert_allclose(K, K.T, atol=1e-14)


def test_isolated_plate_rigid_modes() -> None:
    L = 0.5
    K = hermite_bending_matrix(EI=8.0, length=L)
    q_const = np.array([1.0, 0.0, 1.0, 0.0])
    q_linear = np.array([0.0, 1.0, L, 1.0])
    np.testing.assert_allclose(K @ q_const, 0.0, atol=1e-12)
    np.testing.assert_allclose(K @ q_linear, 0.0, atol=1e-12)
    assert q_const @ K @ q_const == pytest.approx(0.0, abs=1e-12)
    assert q_linear @ K @ q_linear == pytest.approx(0.0, abs=1e-12)


def test_zero_energy_constant_and_affine_normal_displacement() -> None:
    L = 0.25
    EI = 3.0
    K = hermite_bending_matrix(EI, L)
    q_const = np.array([2.0, 0.0, 2.0, 0.0])
    q_affine = np.array([-1.0, 4.0, -1.0 + 4.0 * L, 4.0])
    assert q_const @ K @ q_const == pytest.approx(0.0, abs=1e-12)
    assert q_affine @ K @ q_affine == pytest.approx(0.0, abs=1e-12)


def test_quadratic_and_cubic_bending_energy() -> None:
    L = 0.4
    EI = 5.0
    K = hermite_bending_matrix(EI, L)
    q_quad = np.array([0.0, 0.0, L**2, 2.0 * L])
    q_cubic = np.array([0.0, 0.0, L**3, 3.0 * L**2])
    assert q_quad @ K @ q_quad == pytest.approx(4.0 * EI * L, rel=1e-12)
    assert q_cubic @ K @ q_cubic == pytest.approx(12.0 * EI * L**3, rel=1e-12)


def test_normal_coupling_to_continuum_displacements() -> None:
    n_p = np.array([-0.3, 0.8])
    n_p = n_p / np.linalg.norm(n_p)
    u = np.array([0.1, -0.2])
    v = np.array([0.4, 0.5])
    w = plate_normal_deflection(u, v, n_p)
    np.testing.assert_allclose(w, n_p[0] * u + n_p[1] * v)

    n_foam = 4
    n_q = n_foam + 2
    T = element_transformation(n_q, 0, 1, 4, 2, 3, 5, n_p)
    q = np.array([u[0], v[0], u[1], v[1], 0.7, -0.2])
    qp = T @ q
    np.testing.assert_allclose(qp, np.array([w[0], 0.7, w[1], -0.2]))


def test_assembled_plate_uses_normal_not_vertical() -> None:
    n_p = np.array([-0.6, 0.8])
    coords = np.array([[0.0, 1.0], [0.0, 0.75]])
    u_dofs = np.array([0, 2])
    v_dofs = np.array([1, 3])
    K = assemble_plate_bending(4, u_dofs, v_dofs, coords, n_p, EI=2.0)
    q = np.zeros(6)
    q[0] = 1.0
    energy_u = float(q @ K @ q)
    q[:] = 0.0
    q[1] = 1.0
    energy_v = float(q @ K @ q)
    assert energy_u > 0.0
    assert energy_v > 0.0
    assert energy_u != pytest.approx(energy_v)
