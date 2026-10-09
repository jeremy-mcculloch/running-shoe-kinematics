"""Interval lookup generation: boundary systems, residuals, compliance blocks and conventions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

from compliance_fem.fem.boundaries import DOF_ORDERING_COMPONENT_MAJOR_UV
from compliance_fem.contact.basis import (
    COL_ALPHA,
    COL_BX,
    COL_BY,
    N_AFFINE_COLUMNS,
    build_contact_affine_matrix,
    build_top_affine_matrix,
    softplus_w2,
)
from compliance_fem.contact.lookup import (
    LOOKUP_SCHEMA_VERSION,
    SCALAR_FX,
    SCALAR_FY,
    build_boundary_matrix,
    compute_toe_moment,
    get_compliance_block_matrix,
    generate_contact_lookup,
    interval_metadata,
    prepare_compliance_blocks,
    ramp_horizontal_lever,
    restricted_blocks,
    save_contact_lookup,
    solve_candidate,
)
from compliance_fem.contact.topology import ContactInterval
from compliance_fem.contact.force_control import evaluate_candidates, select_contact_candidate

from conftest import SMALL_KAPPA, SMALL_TOE_LENGTH


@pytest.fixture(scope="module")
def blocks(rocker_case):
    prepared, _ = prepare_compliance_blocks(get_compliance_block_matrix(rocker_case[0]))
    return prepared


def test_restricted_block_dimensions(blocks) -> None:
    iv = ContactInterval(3, 8, len(blocks.x_bottom))
    free, contact = iv.sets()
    rb = restricted_blocks(blocks, free, contact)
    n_t, n_c, n_f = 2 * len(blocks.x_top), 2 * contact.size, 2 * free.size
    assert rb.C_tc.shape == (n_t, n_c)
    assert rb.C_ct.shape == (n_c, n_t)
    assert rb.C_cc.shape == (n_c, n_c)
    assert rb.C_ft.shape == (n_f, n_t)
    assert rb.C_fc.shape == (n_f, n_c)


@pytest.mark.parametrize("interval", [(0, 4), (3, 8), (6, 6), (0, 12)])
def test_boundary_matrix_symmetry(blocks, interval) -> None:
    free, contact = ContactInterval(*interval, len(blocks.x_bottom)).sets()
    A = build_boundary_matrix(blocks, free, contact, x_r=0.5 * blocks.L).A
    n = 2 * len(blocks.x_top) + 2 * contact.size + 3
    assert A.shape == (n, n)
    assert np.linalg.norm(A - A.T) / np.linalg.norm(A) < 1e-12


def test_one_factorization_matches_separate_direct_solves(blocks) -> None:
    iv = ContactInterval(2, 9, len(blocks.x_bottom))
    W = build_top_affine_matrix(blocks.x_top, blocks.L, SMALL_TOE_LENGTH, SMALL_KAPPA)
    sol = solve_candidate(blocks, iv, W, 0.5 * blocks.L, toe_length=SMALL_TOE_LENGTH)
    A = sol.A
    free, contact = iv.sets()
    n_t = 2 * len(blocks.x_top)
    rhs = np.zeros((A.shape[0], N_AFFINE_COLUMNS))
    rhs[:n_t] = W
    rhs[n_t : n_t + 2 * contact.size] = build_contact_affine_matrix(
        blocks.x_bottom[contact], sol.x_anchor, blocks.y_bottom[contact], sol.y_anchor
    )
    for k in range(N_AFFINE_COLUMNS):
        x_k = linalg.solve(A, rhs[:, k])
        np.testing.assert_allclose(sol.F_t[:, k], x_k[:n_t], rtol=1e-9, atol=1e-9 * np.abs(x_k).max())


def test_prescribed_boundary_values_and_traction_free_bottom(rocker_lookup) -> None:
    lk = rocker_lookup
    assert np.nanmax(lk.top_displacement_residuals) < 1e-8
    assert np.nanmax(lk.contact_displacement_residuals) < 1e-8
    assert np.nanmax(lk.solve_residuals) < 1e-8
    fields = lk.record_fields(lk.valid_rows)
    for k, row in enumerate(lk.valid_rows):
        free, contact = lk.record_sets(row)
        np.testing.assert_array_equal(fields["reaction_x"][k][:, free], 0.0)
        np.testing.assert_array_equal(fields["reaction_y"][k][:, free], 0.0)
        # The top shape mode prescribes zero contact displacement.
        np.testing.assert_array_equal(fields["bottom_u"][k, COL_ALPHA][contact], 0.0)
        np.testing.assert_array_equal(fields["bottom_v"][k, COL_ALPHA][contact], 0.0)


def test_force_and_moment_equilibrium_residuals(rocker_lookup) -> None:
    lk = rocker_lookup
    assert np.nanmax(lk.force_equilibrium_residuals) < 1e-8
    assert np.nanmax(lk.moment_equilibrium_residuals) < 1e-8
    assert np.nanmax(lk.balance_residuals) < 1e-8


def test_kf_matrix_matches_translation_scalars(rocker_lookup) -> None:
    lk = rocker_lookup
    S = lk.scalar_lookup[lk.valid_rows]
    expected = np.stack(
        [np.stack([S[:, COL_BX, SCALAR_FX], S[:, COL_BY, SCALAR_FX]], -1),
         np.stack([S[:, COL_BX, SCALAR_FY], S[:, COL_BY, SCALAR_FY]], -1)], axis=1,
    )
    np.testing.assert_allclose(lk.kf_matrix[lk.valid_rows], expected)


def test_softplus_stable_large_kappa() -> None:
    x = np.linspace(0.0, 1.0, 21)
    w = softplus_w2(x, L=1.0, toe_length=0.6, kappa=1e6)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w, np.maximum(0.0, x - 0.4), atol=1e-4)


def test_metadata_records_conventions(rocker_lookup, tmp_path: Path) -> None:
    save_contact_lookup(rocker_lookup, tmp_path)
    meta = json.loads((tmp_path / "contact_lookup_metadata.json").read_text(encoding="utf-8"))
    for key, value in interval_metadata().items():
        assert key in meta, key
    assert meta["schema_version"] == LOOKUP_SCHEMA_VERSION == 11
    assert meta["contact_set_model"] == "single_contiguous_interval"
    assert meta["contact_anchor_definition"] == "interval_midpoint"
    assert meta["affine_columns"][0] == "curved_sole_closure"
    assert meta["n_intervals_theoretical"] == meta["n_records"] == 91
    assert meta["n_valid"] == 91 and meta["rejections"] == {}
    assert meta["label_counts"] == {"heel": 12, "full": 1, "toe": 12, "interior": 66}
    assert "infinitesimal strain" in rocker_lookup.metadata["strain_model"]
    assert np.isfinite(rocker_lookup.build_time_s) and rocker_lookup.build_time_s > 0.0


def test_progress_reporting(rocker_case) -> None:
    calls = []
    generate_contact_lookup(
        get_compliance_block_matrix(rocker_case[0]), toe_length=SMALL_TOE_LENGTH, kappa=SMALL_KAPPA,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls and calls[-1] == (91, 91)
    assert all(t == 91 for _, t in calls)


def test_xy_coupling_reciprocity(flat_case) -> None:
    r = flat_case[0]
    n_t, n_b = len(r.x_top), len(r.x_bottom)
    Ctt, Cbb, Ctb, Cbt = r.Ctt_force, r.Cbb_force, r.Ctb_force, r.Cbt_force
    assert np.linalg.norm(Ctt[:n_t, n_t:] - Ctt[n_t:, :n_t].T) / np.linalg.norm(Ctt[:n_t, n_t:]) < 1e-8
    assert np.linalg.norm(Cbt - Ctb.T) / np.linalg.norm(Cbt) < 1e-8
    assert np.linalg.norm(Cbb[:n_b, n_b:] - Cbb[n_b:, :n_b].T) / np.linalg.norm(Cbb[:n_b, n_b:]) < 1e-8
    assert r.solve_residuals.top < 1e-8 and r.rigid_mode_error < 1e-8


def test_flat_top_eta_is_zero() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = np.full(3, 0.2)
    np.testing.assert_allclose(ramp_horizontal_lever(x, y, 0.5, 0.2), 0.0)
    T, T_vert, H_mtp = compute_toe_moment(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 4.0]), x, y, 0.5)
    assert H_mtp == pytest.approx(0.2) and T == pytest.approx(T_vert) == pytest.approx(2.0)


def test_sloped_top_eta_enters_toe_moment() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.1, 0.2, 0.3])
    T, T_vert, H_mtp = compute_toe_moment(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 3.0]), x, y, 0.5)
    assert H_mtp == pytest.approx(0.2)
    assert T_vert == pytest.approx(1.5)
    assert T == pytest.approx(1.5 - 0.1 * 2.0)


def test_singular_kf_is_reported_not_regularized(flat_lookup) -> None:
    S = flat_lookup.scalar_lookup.copy()
    S[:, COL_BX, :2] = 0.0
    S[:, COL_BY, :2] = 0.0
    lk = replace(flat_lookup, scalar_lookup=S)
    ev = evaluate_candidates(lk, 0.0, -1.0e3, 0.0, 0.0)
    assert not np.any(ev.evaluated)
    assert np.all(ev.kf_illconditioned[lk.valid_rows])
    sel = select_contact_candidate(ev)
    assert sel.selected_row is None


def test_vector_blocks_and_interval_lookup(rocker_case) -> None:
    result, lookup = rocker_case
    n_t = len(result.x_top)
    assert result.dof_ordering == DOF_ORDERING_COMPONENT_MAJOR_UV
    assert result.Ctt_force.shape == (2 * n_t, 2 * n_t)
    assert result.inextensibility_residuals.top < 1e-8
    n_b = lookup.n_bottom_nodes
    assert lookup.n_records == n_b * (n_b + 1) // 2
    assert lookup.scalar_lookup.shape[1] == N_AFFINE_COLUMNS
    ev = evaluate_candidates(lookup, 0.0, -1.0e3, -2.0, 1.0)
    assert np.any(ev.evaluated)
