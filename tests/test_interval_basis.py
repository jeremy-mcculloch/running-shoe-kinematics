"""Anchors, curved-sole closure and the five basis modes (spec tests 15-26)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.contact_basis import (
    AFFINE_COLUMN_NAMES,
    BASIS_ORDER,
    COL_ALPHA,
    COL_BRX,
    COL_BRY,
    COL_BX,
    COL_BY,
    COL_CONST,
    N_AFFINE_COLUMNS,
    build_contact_affine_matrix,
    build_contact_displacement_matrix,
    build_top_affine_matrix,
)
from compliance_fem.contact_lookup import CONTACT_ANCHOR_DEFINITION, SCALAR_FX, SCALAR_FY
from compliance_fem.corotation import (
    affine_coefficients,
    basis_coefficients,
    contact_displacement,
    contract_basis,
    rotation_coefficients,
    rotation_matrix,
)
from compliance_fem.force_control import Tolerances, reconstruct_rows


@pytest.fixture(scope="module")
def flat_lookup(flat_lookup_full):
    return flat_lookup_full


@pytest.fixture(scope="module")
def rocker_lookup(rocker_lookup_full):
    return rocker_lookup_full


@pytest.fixture(scope="module")
def asym_lookup(asym_lookup_full):
    return asym_lookup_full


def _rows(lookup, labels=("heel", "toe", "full", "interior")):
    rows = []
    for label in labels:
        rows.extend(lookup.rows_for(label).tolist())
    return np.array(sorted(rows))


def test_documented_anchor_for_every_interval(rocker_lookup) -> None:
    lk = rocker_lookup
    assert lk.metadata["contact_anchor_definition"] == CONTACT_ANCHOR_DEFINITION == "interval_midpoint"
    x_b, y_b = lk.x_bottom, lk.y_bottom
    for row in range(lk.n_records):
        i, j = int(lk.contact_start_index[row]), int(lk.contact_end_index[row])
        x_a = 0.5 * (x_b[i] + x_b[j])
        assert lk.contact_anchor_reference_x[row] == pytest.approx(x_a, abs=1e-15)
        assert lk.contact_anchor_reference_y[row] == pytest.approx(np.interp(x_a, x_b, y_b), abs=1e-15)
        # Physical edges stay distinct from the numerical anchor.
        assert lk.contact_start_x[row] == x_b[i] and lk.contact_end_x[row] == x_b[j]


def test_contact_rotation_zero_at_anchor(rocker_lookup) -> None:
    lk = rocker_lookup
    for i, j in [(2, 6), (0, 12), (5, 5), (4, 10)]:
        row = lk.row_of(i, j)
        x_a = lk.contact_anchor_reference_x[row]
        ids = np.arange(i, j + 1)
        ds = lk.x_bottom[ids] - x_a
        np.testing.assert_allclose(lk.bottom_u_basis[row, COL_BRX, ids], ds, atol=1e-15)
        np.testing.assert_allclose(lk.bottom_v_basis[row, COL_BRY, ids], ds, atol=1e-15)
        np.testing.assert_allclose(lk.bottom_v_basis[row, COL_BRX, ids], 0.0, atol=1e-15)
        np.testing.assert_allclose(lk.bottom_u_basis[row, COL_BRY, ids], 0.0, atol=1e-15)
        if (j - i) % 2 == 0:  # anchor coincides with node (i+j)/2 up to mesh round-off
            k = (i + j) // 2
            assert lk.bottom_u_basis[row, COL_BRX, k] == pytest.approx(0.0, abs=1e-9 * lk.L)
            assert lk.bottom_v_basis[row, COL_BRY, k] == pytest.approx(0.0, abs=1e-9 * lk.L)
    W = build_contact_displacement_matrix(np.array([0.1, 0.2, 0.3]), 0.2)
    n = 3
    assert W[1, 3] == 0.0 and W[n + 1, 4] == 0.0


@pytest.mark.parametrize("varphi_deg", [3.0, 25.0, -12.0])
def test_rotation_signs_on_both_sides_of_anchor(varphi_deg) -> None:
    varphi = np.deg2rad(varphi_deg)
    x = np.array([0.05, 0.10, 0.15])
    u, v = contact_displacement(x, 0.10, 0.0, 0.0, varphi)
    # Rotating the contact line by +varphi (toe up) in the fixed frame is -varphi in the local frame:
    # heel-side points (ds < 0) move up locally, toe-side points move down, the anchor stays.
    s = np.sign(varphi)
    assert v[1] == 0.0 and u[1] == 0.0
    assert np.sign(v[0]) == s and np.sign(v[2]) == -s
    assert u[0] > 0.0 and u[2] < 0.0  # cos - 1 < 0 shortens toward the anchor


@pytest.mark.parametrize("varphi_deg", [0.0, 1.0, 20.0, -35.0, 60.0])
def test_rotation_bases_match_exact_rotation_matrix(rocker_lookup, varphi_deg) -> None:
    lk = rocker_lookup
    varphi = np.deg2rad(varphi_deg)
    r_x, r_y = rotation_coefficients(varphi)
    assert r_x == pytest.approx(np.cos(varphi) - 1.0, abs=0.0)
    assert r_y == pytest.approx(-np.sin(varphi), abs=0.0)
    Q = rotation_matrix(varphi)
    for i, j in [(1, 7), (3, 12), (0, 0)]:
        row = lk.row_of(i, j)
        ids = np.arange(i, j + 1)
        x_a = lk.contact_anchor_reference_x[row]
        u = r_x * lk.bottom_u_basis[row, COL_BRX, ids] + r_y * lk.bottom_u_basis[row, COL_BRY, ids]
        v = r_x * lk.bottom_v_basis[row, COL_BRX, ids] + r_y * lk.bottom_v_basis[row, COL_BRY, ids]
        offsets = np.vstack([lk.x_bottom[ids] - x_a, np.zeros(ids.size)])
        exact = (Q.T - np.eye(2)) @ offsets
        np.testing.assert_allclose(u, exact[0], atol=1e-15)
        np.testing.assert_allclose(v, exact[1], atol=1e-15)


def test_flat_sole_has_zero_closure(flat_lookup) -> None:
    lk = flat_lookup
    assert lk.metadata["curved_sole"] is False
    assert np.all(lk.contact_anchor_reference_y == 0.0)
    for name in ("bottom_u_basis", "bottom_v_basis", "reaction_x_basis", "reaction_y_basis",
                 "top_force_x_basis", "top_force_y_basis"):
        assert np.max(np.abs(getattr(lk, name)[:, COL_CONST])) == 0.0, name
    assert np.max(np.abs(lk.scalar_lookup[:, COL_CONST])) == 0.0
    assert np.max(np.abs(lk.Q_alpha_shoe_on_foot_basis[:, COL_CONST])) == 0.0


def test_curved_sole_has_expected_nonzero_closure(rocker_lookup) -> None:
    lk = rocker_lookup
    assert lk.metadata["curved_sole"] is True
    for i, j in [(0, 12), (2, 9), (6, 6)]:
        row = lk.row_of(i, j)
        ids = np.arange(i, j + 1)
        y_a = lk.contact_anchor_reference_y[row]
        np.testing.assert_allclose(lk.bottom_v_basis[row, COL_CONST, ids], -(lk.y_bottom[ids] - y_a), atol=1e-15)
        np.testing.assert_allclose(lk.bottom_u_basis[row, COL_CONST, ids], 0.0, atol=0.0)
    full = lk.full_contact_row()
    assert np.max(np.abs(lk.reaction_y_basis[full, COL_CONST])) > 0.0
    assert abs(lk.scalar_lookup[full, COL_CONST, SCALAR_FY]) > 0.0


@pytest.mark.parametrize("lookup_name", ["flat_lookup", "rocker_lookup", "asym_lookup"])
def test_closure_with_zero_coordinates_puts_active_nodes_on_ground(request, lookup_name) -> None:
    lk = request.getfixturevalue(lookup_name)
    rows = lk.valid_rows
    gamma = np.zeros((rows.size, 5))
    rec = reconstruct_rows(lk, rows, gamma, 0.0, np.zeros(2), Tolerances(), full_fields=True)
    scale = max(float(np.max(np.abs(lk.y_bottom))), lk.L * 1e-12)
    assert np.max(rec["contact_ground_residual"]) <= 1e-12 * scale + 1e-18
    contact = lk.contact_mask[rows]
    assert np.max(np.abs(rec["full_bottom_gap"][contact])) <= 1e-12 * scale + 1e-18


def test_closure_term_is_included_exactly_once(rocker_lookup) -> None:
    lk = rocker_lookup
    row = lk.row_of(2, 10)
    gamma = basis_coefficients(0.07, 1e-4, -2e-4, np.deg2rad(4.0))
    B = lk.bottom_v_basis[row]
    manual = B[0] + sum(gamma[k] * B[k + 1] for k in range(5))
    np.testing.assert_allclose(contract_basis(B, gamma), manual, rtol=0, atol=1e-18)
    with pytest.raises(ValueError):
        affine_coefficients(np.concatenate([[2.0], gamma]))
    np.testing.assert_array_equal(affine_coefficients(np.concatenate([[1.0], gamma])), np.concatenate([[1.0], gamma]))
    rec = reconstruct_rows(lk, np.array([row]), gamma[None, :], np.deg2rad(4.0), np.zeros(2), Tolerances())
    np.testing.assert_allclose(rec["full_bottom_v"][0], manual, rtol=0, atol=1e-18)


@pytest.mark.parametrize("varphi_deg", [0.0, 7.0, -20.0])
def test_affine_superposition_matches_direct_vector_construction(rocker_lookup, varphi_deg) -> None:
    lk = rocker_lookup
    varphi = np.deg2rad(varphi_deg)
    alpha, d_ax, d_ay = 0.05, 3e-4, -1.5e-4
    gamma = basis_coefficients(alpha, d_ax, d_ay, varphi)
    for i, j in [(3, 9), (0, 5), (12, 12)]:
        row = lk.row_of(i, j)
        ids = np.arange(i, j + 1)
        x_a, y_a = lk.contact_anchor_reference_x[row], lk.contact_anchor_reference_y[row]
        u = contract_basis(lk.bottom_u_basis[row], gamma)[ids]
        v = contract_basis(lk.bottom_v_basis[row], gamma)[ids]
        u_ex, v_ex = contact_displacement(lk.x_bottom[ids], x_a, d_ax, d_ay, varphi, lk.y_bottom[ids], y_a)
        np.testing.assert_allclose(u, u_ex, atol=1e-16)
        np.testing.assert_allclose(v, v_ex, atol=1e-16)
        W_c = build_contact_affine_matrix(lk.x_bottom[ids], x_a, lk.y_bottom[ids], y_a)
        uv = W_c @ affine_coefficients(gamma)
        np.testing.assert_allclose(uv, np.concatenate([u_ex, v_ex]), atol=1e-16)
    W = build_top_affine_matrix(lk.x_top, lk.L, lk.softplus_a, lk.softplus_kappa)
    np.testing.assert_allclose(lk.basis_top_displacements, W)
    np.testing.assert_allclose(W[:, COL_CONST], 0.0)


def test_basis_ordering_is_preserved(flat_lookup) -> None:
    expected = [
        "top_shape_alpha",
        "contact_translation_x",
        "contact_translation_y",
        "contact_rotation_x",
        "contact_rotation_y",
    ]
    assert BASIS_ORDER == expected
    assert list(AFFINE_COLUMN_NAMES) == ["curved_sole_closure", *expected]
    assert (COL_CONST, COL_ALPHA, COL_BX, COL_BY, COL_BRX, COL_BRY) == (0, 1, 2, 3, 4, 5)
    assert flat_lookup.basis_order == expected
    assert flat_lookup.metadata["basis_order"] == expected
    assert flat_lookup.scalar_lookup.shape[1] == N_AFFINE_COLUMNS == 6


def _equilibrium(lk, row, col):
    n_t = lk.n_top_nodes
    ftx, fty = lk.top_force_x_basis[row, col], lk.top_force_y_basis[row, col]
    rx, ry = lk.reaction_x_basis[row, col], lk.reaction_y_basis[row, col]
    y_t = lk.y_top if lk.y_top is not None else np.full(n_t, lk.H)
    F = np.array([ftx.sum() + rx.sum(), fty.sum() + ry.sum()])
    M = (lk.x_top @ fty - y_t @ ftx) + (lk.x_bottom @ ry - lk.y_bottom @ rx)
    scale = max(np.abs(ftx).sum() + np.abs(fty).sum(), 1e-30)
    return F, M, scale


@pytest.mark.parametrize("lookup_name", ["rocker_lookup", "asym_lookup"])
def test_closure_solve_equilibrium(request, lookup_name) -> None:
    lk = request.getfixturevalue(lookup_name)
    for row in lk.valid_rows:
        F, M, scale = _equilibrium(lk, row, COL_CONST)
        assert np.max(np.abs(F)) <= 1e-9 * scale
        assert abs(M) <= 1e-9 * scale * lk.L


@pytest.mark.parametrize("lookup_name", ["flat_lookup", "rocker_lookup"])
def test_every_variable_basis_solve_equilibrium(request, lookup_name) -> None:
    lk = request.getfixturevalue(lookup_name)
    for row in lk.valid_rows:
        for col in range(1, N_AFFINE_COLUMNS):
            F, M, scale = _equilibrium(lk, row, col)
            assert np.max(np.abs(F)) <= 1e-9 * scale, (row, col)
            assert abs(M) <= 1e-9 * scale * lk.L, (row, col)
    scalars = lk.scalar_lookup[lk.valid_rows]
    totals = lk.contact_force_total[lk.valid_rows]
    np.testing.assert_allclose(
        scalars[:, :, [SCALAR_FX, SCALAR_FY]], -totals[:, :, :2],
        atol=1e-9 * float(np.max(np.abs(scalars[:, :, :2]))),
    )
