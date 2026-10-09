"""Co-rotating frame, angle conventions, center of effort, display transform and export."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.contact.basis import COL_ALPHA, COL_BRX, COL_BX, shape_mode_phi1
from compliance_fem.contact.lookup import (
    SCALAR_TOE,
    compute_toe_moment,
    get_compliance_block_matrix,
    generate_contact_lookup,
    solve_candidate,
)
from compliance_fem.contact.corotation import (
    SHAPE_MODE_SIGN,
    basis_coefficients,
    contract_basis,
    corotation_angle,
    reference_chord_angle,
    rotate_force_to_local,
    rotate_vector_to_fixed,
    rotation_coefficients,
    rotation_matrix,
    shape_amplitude,
    transform_to_fixed_frame,
)
from compliance_fem.contact.force_control import (
    Tolerances,
    angles_to_coefficients,
    evaluate_candidates,
    evaluate_from_angles,
    export_evaluation_csv,
    export_selected_result,
)
from compliance_fem.plotting.shape_render import build_shape_plot_data

from conftest import SMALL_KAPPA, SMALL_TOE_LENGTH, small_config


@pytest.fixture(scope="module")
def sloped_lookup():
    """Synthetic sole whose reference top chord is deliberately not horizontal."""
    result = compute_compliance(small_config(top_slope=0.02, EI_plate_Nm2_per_m=5.0, min_angle_deg=5.0))
    return generate_contact_lookup(
        get_compliance_block_matrix(result), toe_length=SMALL_TOE_LENGTH, kappa=30.0, fem_result=result
    )


def test_shape_mode_vanishes_at_endpoints_and_top_is_clamped(flat_lookup) -> None:
    for L, toe_length, kappa in ((1.0, 0.5, 30.0), (0.25, 0.10, 60.0), (2.0, 0.3, 8.0)):
        np.testing.assert_allclose(
            shape_mode_phi1(np.array([0.0, L]), L=L, toe_length=toe_length, kappa=kappa), 0.0, atol=1e-14
        )
    n_t = flat_lookup.n_top_nodes
    W = flat_lookup.basis_top_displacements
    assert flat_lookup.x_top[0] == 0.0 and abs(W[n_t, COL_ALPHA]) < 1e-14
    np.testing.assert_allclose(
        W[n_t:, COL_ALPHA],
        shape_mode_phi1(flat_lookup.x_top, flat_lookup.L, flat_lookup.softplus_toe_length, flat_lookup.softplus_kappa),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_array_equal(W[:, COL_BX:], 0.0)
    np.testing.assert_array_equal(W[:n_t, :], 0.0)


def test_lookup_generation_takes_no_angle_and_is_reproducible(flat_case) -> None:
    for fn in (generate_contact_lookup, solve_candidate):
        names = set(inspect.signature(fn).parameters)
        assert not (names & {"phi", "phi_deg", "varphi", "theta", "theta_deg"})
    blocks = get_compliance_block_matrix(flat_case[0])
    first = generate_contact_lookup(blocks, toe_length=SMALL_TOE_LENGTH, kappa=SMALL_KAPPA, store_fields=True)
    np.testing.assert_array_equal(first.scalar_lookup, flat_case[1].scalar_lookup)
    np.testing.assert_array_equal(first.reaction_x_basis, flat_case[1].reaction_x_basis)


def test_rotation_coefficients_vanish_at_zero_and_are_exact() -> None:
    assert rotation_coefficients(0.0) == (0.0, 0.0)
    for deg in (30.0, 75.0):
        v = np.deg2rad(deg)
        r_x, r_y = rotation_coefficients(v)
        assert r_x == np.cos(v) - 1.0 and r_y == -np.sin(v)
        assert abs(r_x - (-0.5 * v * v)) > 1e-4  # not a small-angle form


@pytest.mark.parametrize("varphi_deg", (-30.0, -5.0, 0.0, 12.0))
def test_force_rotation_to_local_and_back(varphi_deg) -> None:
    v = np.deg2rad(varphi_deg)
    F_loc = rotate_force_to_local(120.0, -900.0, v)
    np.testing.assert_allclose(F_loc, rotation_matrix(v).T @ np.array([120.0, -900.0]), atol=1e-12)
    fx, fy = rotate_vector_to_fixed(F_loc[0], F_loc[1], v)
    assert (fx, fy) == pytest.approx((120.0, -900.0), abs=1e-10)


def test_positive_theta_bends_the_toe_up() -> None:
    L, toe_length, kappa = 1.0, 0.5, 30.0
    x = np.linspace(0.0, L, 401)
    v = shape_amplitude(np.deg2rad(10.0)) * shape_mode_phi1(x, L=L, toe_length=toe_length, kappa=kappa)
    slope = np.gradient(v, x)
    assert float(np.mean(slope[x >= L - toe_length])) > 0.0
    assert float(np.mean(slope[x <= L - toe_length])) < 0.0
    assert SHAPE_MODE_SIGN == 1.0


def test_theta_at_ninety_degrees_is_rejected() -> None:
    with pytest.raises(ValueError, match="90 degrees"):
        shape_amplitude(0.5 * np.pi)
    with pytest.raises(ValueError, match="90 degrees"):
        angles_to_coefficients(0.0, 90.0, 0.0)


def test_gui_angle_conversion(flat_lookup) -> None:
    varphi, alpha, r_x, r_y = angles_to_coefficients(30.0, 15.0, 0.0)
    assert varphi == pytest.approx(np.pi / 6.0)
    assert alpha == pytest.approx(np.tan(np.pi / 12.0))
    assert (r_x, r_y) == pytest.approx((np.cos(np.pi / 6.0) - 1.0, -np.sin(np.pi / 6.0)))
    ev = evaluate_candidates(flat_lookup, 0.0, -1.0e3, 12.0, 9.0)
    assert ev.phi == pytest.approx(np.deg2rad(12.0)) and ev.alpha == pytest.approx(np.tan(np.deg2rad(9.0)))


def test_runtime_rejects_stale_lookup(flat_lookup) -> None:
    stale = dataclasses.replace(flat_lookup, schema_version=8)
    with pytest.raises(ValueError, match="schema_version=11"):
        evaluate_candidates(stale, 0.0, -1.0e3, 0.0, 0.0)


def test_sloped_top_reference_chord(sloped_lookup) -> None:
    x_top, y_top = sloped_lookup.x_top, sloped_lookup.y_top
    expected = np.arctan2(float(y_top[-1] - y_top[0]), float(x_top[-1] - x_top[0]))
    assert sloped_lookup.phi_ref == pytest.approx(expected) and abs(expected) > 1e-3
    phi_ref_deg = float(np.rad2deg(sloped_lookup.phi_ref))
    ev = evaluate_candidates(sloped_lookup, 0.0, -800.0, phi_ref_deg, 0.0)
    assert ev.varphi == pytest.approx(0.0, abs=1e-12)
    assert ev.Fy_star_local == pytest.approx(-800.0, rel=1e-12)
    assert reference_chord_angle(np.array([0.0, 1.0]), None) == 0.0
    assert corotation_angle(0.4, 0.1) == pytest.approx(0.3)


def test_toe_moment_superposes_and_uses_reference_levers(rocker_lookup) -> None:
    lk = rocker_lookup
    sel = evaluate_from_angles(lk, 50.0, -900.0, -1.0, 4.0)
    row = sel.selected_row
    gamma = basis_coefficients(sel.alpha, sel.d_ax, sel.d_ay, sel.varphi)
    assert sel.T_toe == pytest.approx(float(contract_basis(lk.scalar_lookup[row, :, SCALAR_TOE], gamma)), rel=1e-12)
    y_top = lk.y_top if lk.y_top is not None else np.full(lk.n_top_nodes, lk.H)
    T_direct, _, _ = compute_toe_moment(sel.top_force_x, sel.top_force_y, lk.x_top, y_top, lk.L - lk.softplus_toe_length)
    assert float(T_direct) == pytest.approx(sel.T_toe, rel=1e-9)


def test_center_of_effort_in_fixed_frame(rocker_lookup) -> None:
    lk = rocker_lookup
    sel = evaluate_from_angles(lk, 60.0, -900.0, -1.0, 3.0)
    y_top = lk.y_top if lk.y_top is not None else np.full(lk.n_top_nodes, lk.H)
    _, fty_F = rotate_vector_to_fixed(sel.top_force_x, sel.top_force_y, sel.varphi)
    x_top_F, _ = transform_to_fixed_frame(
        lk.x_top, y_top, sel.top_u, sel.top_v, sel.anchor_x, sel.d_ax, sel.d_ay, sel.varphi, y_anchor=sel.anchor_y
    )
    assert sel.x_cm == pytest.approx(float(np.sum(x_top_F * fty_F) / np.sum(fty_F)), rel=1e-10)
    assert sel.x_cm_rel == pytest.approx(sel.x_cm - sel.anchor_x, rel=1e-10, abs=1e-12)
    ev = evaluate_candidates(lk, 1.0e2, 0.0, 0.0, 2.0)
    assert np.all(np.isnan(ev.x_cm[ev.evaluated]))


def test_high_condition_number_is_flagged(flat_lookup) -> None:
    sel = evaluate_from_angles(flat_lookup, 0.0, -500.0, 0.3, 0.0, tolerances=Tolerances(kf_cond_warn=1.0 + 1e-12))
    assert sel.selected_row is not None and sel.kf_illconditioned
    assert "ill-conditioned" in sel.message


def test_shape_plot_scales_only_the_elastic_part(rocker_lookup) -> None:
    lk = rocker_lookup
    sel = evaluate_from_angles(lk, 0.0, -900.0, -2.0, 5.0)
    exact = build_shape_plot_data(lk, sel)
    scaled = build_shape_plot_data(lk, sel, scale_mode="manual", manual_scale=20.0)
    assert exact.scale == 1.0 and scaled.scale == 20.0
    Q = rotation_matrix(sel.varphi)
    x = exact.x_dense
    u_bot = np.interp(x, lk.x_bottom, sel.full_bottom_u)
    v_bot = np.interp(x, lk.x_bottom, sel.full_bottom_v)
    expected = 19.0 * (Q @ np.vstack([u_bot - sel.d_ax, v_bot - sel.d_ay]))
    np.testing.assert_allclose(scaled.x_bottom_def - exact.x_bottom_def, expected[0], atol=1e-14)
    np.testing.assert_allclose(scaled.y_bottom_def - exact.y_bottom_def, expected[1], atol=1e-14)
    assert exact.varphi_deg == pytest.approx(np.rad2deg(sel.varphi))


def test_export_csv_and_npz(rocker_lookup, tmp_path: Path) -> None:
    sel = evaluate_from_angles(rocker_lookup, 0.0, -500.0, 0.0, 0.0)
    csv_path = export_evaluation_csv(sel, tmp_path / "eval.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "contact_start_index" in text.splitlines()[0]
    npz = np.load(export_selected_result(sel, rocker_lookup, tmp_path / "sel.npz"), allow_pickle=True)
    assert int(npz["contact_start_index"]) == sel.contact_start_index
    assert str(npz["candidate_search_method"]) == sel.candidate_search_method
    assert npz["full_bottom_gap"].shape == (rocker_lookup.n_bottom_nodes,)


def test_rotation_basis_column_is_x_minus_anchor(rocker_lookup_full) -> None:
    row = rocker_lookup_full.row_of(2, 8)
    ids = np.arange(2, 9)
    np.testing.assert_allclose(
        rocker_lookup_full.bottom_u_basis[row, COL_BRX, ids],
        rocker_lookup_full.x_bottom[ids] - rocker_lookup_full.contact_anchor_reference_x[row],
    )
