"""Section 19 routing / runtime checks (items 11-23, 26, 27)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_basis import N_BASIS_MODES
from compliance_fem.contact_direct_fem import (
    compare_compliance_to_direct_fem,
    standard_comparison_specs,
)
from compliance_fem.contact_lookup import (
    CORNER_HEEL,
    CORNER_TOE,
    from_compliance_result,
    generate_contact_lookup,
)
from compliance_fem.contact_topology import ContactMode, ContactType
from compliance_fem.corotation import (
    basis_coefficients,
    fixed_frame_normal_component,
    rotate_vector_to_fixed,
    transform_to_fixed_frame,
)
from compliance_fem.force_control import (
    Tolerances,
    _route_families,
    evaluate_candidates,
    evaluate_from_angles,
    select_contact_candidate,
)


@pytest.fixture(scope="module")
def small_compliance():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=8, ny=4, order=1)
    return compute_compliance(config)


@pytest.fixture(scope="module")
def small_lookup(small_compliance):
    blocks = from_compliance_result(small_compliance)
    return generate_contact_lookup(blocks, a=0.6, kappa=30.0)


# --- 11–13: full-contact corner basis and rotation --------------------------


def test_full_contact_basis_stores_both_corner_force_components(small_lookup) -> None:
    full_row = small_lookup.full_contact_row()
    corners = np.asarray(small_lookup.corner_reactions_local[full_row], dtype=float)
    assert corners.shape == (N_BASIS_MODES, 2, 2)
    assert np.all(np.isfinite(corners))
    # Both local components at heel and toe for every basis mode.
    assert np.all(np.isfinite(corners[:, CORNER_HEEL, :]))
    assert np.all(np.isfinite(corners[:, CORNER_TOE, :]))
    # Partial records keep NaN corner bases.
    for kind in (ContactType.HEEL, ContactType.TOE):
        for row in small_lookup.rows_for(kind):
            assert np.all(np.isnan(small_lookup.corner_reactions_local[row]))


def test_runtime_corner_force_superposition_matches_direct(small_lookup) -> None:
    Fy = -1.0e4
    F_scale = abs(Fy)
    ev = evaluate_candidates(small_lookup, 300.0, Fy, phi_deg=-4.0, theta_deg=3.0)
    full_row = ev.full_contact_row
    assert ev.well_conditioned[full_row]
    gamma = basis_coefficients(ev.alpha, ev.d_ax[full_row], ev.d_ay[full_row], ev.varphi)
    expected = np.tensordot(
        gamma, small_lookup.corner_reactions_local[full_row], axes=([0], [0])
    )
    np.testing.assert_allclose(
        ev.corner_reaction_local[full_row],
        expected,
        rtol=1e-10,
        atol=1e-12 * F_scale,
    )


def test_rotated_fixed_frame_corner_reactions_agree_with_Q(small_lookup) -> None:
    Fy = -1.0e4
    F_scale = abs(Fy)
    ev = evaluate_candidates(small_lookup, 200.0, Fy, phi_deg=-5.0, theta_deg=4.0)
    full_row = ev.full_contact_row
    assert ev.well_conditioned[full_row]
    c_local = ev.corner_reaction_local[full_row]
    cfx, cfy = rotate_vector_to_fixed(c_local[:, 0], c_local[:, 1], ev.varphi)
    np.testing.assert_allclose(
        ev.corner_reaction_fixed[full_row, :, 0], cfx, rtol=1e-12, atol=1e-12 * F_scale
    )
    np.testing.assert_allclose(
        ev.corner_reaction_fixed[full_row, :, 1], cfy, rtol=1e-12, atol=1e-12 * F_scale
    )
    rn = fixed_frame_normal_component(c_local[:, 0], c_local[:, 1], ev.varphi)
    np.testing.assert_allclose(
        ev.corner_reaction_normal[full_row], rn, rtol=1e-12, atol=1e-12 * F_scale
    )
    assert ev.Rn_heel_corner[full_row] == pytest.approx(float(rn[CORNER_HEEL]), rel=1e-12)
    assert ev.Rn_toe_corner[full_row] == pytest.approx(float(rn[CORNER_TOE]), rel=1e-12)


# --- 14–18: full-contact validity and corner routing ------------------------


def test_full_contact_valid_when_both_corners_compressive(small_lookup) -> None:
    tau_R = 1.0e-3 * abs(-1.0e4)
    tol = Tolerances(tau_R=tau_R)
    ev = evaluate_candidates(small_lookup, 0.0, -1.0e4, 0.0, 0.0, tolerances=tol)
    full = ev.full_contact_row
    rn_h = float(ev.Rn_heel_corner[full])
    rn_t = float(ev.Rn_toe_corner[full])
    assert np.isfinite(rn_h) and np.isfinite(rn_t)
    # With compressive load at flat angles, corners should be compressive.
    assert rn_h >= -tau_R
    assert rn_t >= -tau_R
    assert bool(ev.full_contact_valid[full])
    routed, reason = _route_families(True, rn_h, rn_t, tau_R, True)
    assert routed == (ContactType.FULL,)
    assert "compressive" in reason


def test_full_contact_invalid_when_either_corner_tensile(small_lookup) -> None:
    tau_R = 0.0
    # Synthetic tensile corners: validity and routing must reject full contact.
    rn_h, rn_t = -10.0, 5.0
    assert rn_h < -tau_R
    full_valid = bool(rn_h >= -tau_R and rn_t >= -tau_R)
    assert not full_valid
    routed, reason = _route_families(True, rn_h, rn_t, tau_R, full_valid)
    assert ContactType.FULL not in routed
    assert "tensile" in reason

    # Mirror the evaluate_candidates predicate on a real reconstruction: choose
    # tau_R so rn >= -tau_R fails for the less-compressive corner.
    ev = evaluate_candidates(small_lookup, 0.0, -1.0e4, 0.0, 0.0, tolerances=Tolerances(tau_R=0.0))
    full = ev.full_contact_row
    rn_h = float(ev.Rn_heel_corner[full])
    rn_t = float(ev.Rn_toe_corner[full])
    assert np.isfinite(rn_h) and np.isfinite(rn_t)
    # Need rn < -tau_R for at least one corner ⇒ tau_R < -rn ⇒ tau_R = -max(rn) - 1.
    tau_fail = -max(rn_h, rn_t) - 1.0
    assert rn_h < -tau_fail or rn_t < -tau_fail
    ev_fail = evaluate_candidates(
        small_lookup, 0.0, -1.0e4, 0.0, 0.0, tolerances=Tolerances(tau_R=tau_fail)
    )
    assert not bool(ev_fail.full_contact_valid[full])


def test_tensile_heel_corner_routes_to_toe_family() -> None:
    routed, reason = _route_families(True, -5.0, 10.0, 1.0, False)
    assert routed == (ContactType.TOE,)
    assert "heel" in reason and "toe" in reason


def test_tensile_toe_corner_routes_to_heel_family() -> None:
    routed, reason = _route_families(True, 10.0, -5.0, 1.0, False)
    assert routed == (ContactType.HEEL,)
    assert "toe" in reason and "heel" in reason


def test_two_tensile_corners_route_to_both_partial_families() -> None:
    routed, reason = _route_families(True, -5.0, -8.0, 1.0, False)
    assert routed == (ContactType.HEEL, ContactType.TOE)
    assert "both" in reason
    routed_ill, reason_ill = _route_families(False, float("nan"), float("nan"), 1.0, False)
    assert routed_ill == (ContactType.HEEL, ContactType.TOE)
    assert "well-conditioned" in reason_ill or "searching heel and toe" in reason_ill


# --- 19–20: fixed-frame gap and reaction reconstruction ---------------------


def test_heel_and_toe_fixed_frame_gaps_agree_with_reconstruction(small_lookup) -> None:
    L = float(small_lookup.L)
    Fy = -1.0e4
    ev = evaluate_candidates(small_lookup, 150.0, Fy, phi_deg=-3.0, theta_deg=2.0)
    gap_tol = 1e-10 * max(L, 1.0)
    for kind in (ContactType.HEEL, ContactType.TOE):
        rows = [r for r in small_lookup.rows_for(kind) if ev.well_conditioned[r]]
        assert rows
        row = rows[len(rows) // 2]
        free, contact = small_lookup.record_sets(row)
        x_a = float(ev.anchor_x[row])
        dx = (small_lookup.x_bottom + ev.full_bottom_u[row]) - (x_a + ev.d_ax[row])
        dy = ev.full_bottom_v[row] - ev.d_ay[row]
        gap_direct = fixed_frame_normal_component(dx, dy, ev.varphi)
        np.testing.assert_allclose(
            ev.full_bottom_gap[row], gap_direct, rtol=1e-10, atol=gap_tol
        )
        # Contact nodes sit on the ground line.
        np.testing.assert_allclose(ev.full_bottom_gap[row][contact], 0.0, atol=gap_tol)
        # Direct fixed-frame transform of free nodes yields the same normal gap.
        if free.size:
            y_bot = (
                np.asarray(small_lookup.y_bottom, dtype=float)
                if small_lookup.y_bottom is not None
                else np.zeros(len(small_lookup.x_bottom))
            )
            x_F, y_F = transform_to_fixed_frame(
                small_lookup.x_bottom[free],
                y_bot[free],
                ev.full_bottom_u[row][free],
                ev.full_bottom_v[row][free],
                x_a,
                ev.d_ax[row],
                ev.d_ay[row],
                ev.varphi,
            )
            # Gauge r_a^F = (x_a, 0): normal gap is fixed-frame y.
            np.testing.assert_allclose(y_F, gap_direct[free], rtol=1e-10, atol=gap_tol)
            del x_F


def test_heel_and_toe_fixed_frame_reactions_agree_with_force_rotation(small_lookup) -> None:
    Fy = -1.0e4
    F_scale = abs(Fy)
    ev = evaluate_candidates(small_lookup, 100.0, Fy, phi_deg=-4.0, theta_deg=3.0)
    for kind in (ContactType.HEEL, ContactType.TOE):
        rows = [r for r in small_lookup.rows_for(kind) if ev.well_conditioned[r]]
        assert rows
        row = rows[len(rows) // 2]
        rx = ev.full_bottom_reaction_x[row]
        ry = ev.full_bottom_reaction_y[row]
        rn = fixed_frame_normal_component(rx, ry, ev.varphi)
        rtx, rty = rotate_vector_to_fixed(rx, ry, ev.varphi)
        np.testing.assert_allclose(
            ev.full_bottom_reaction_normal[row], rn, rtol=1e-12, atol=1e-12 * F_scale
        )
        # Tangential is the fixed-frame x component of the rotated local force.
        np.testing.assert_allclose(
            ev.full_bottom_reaction_tangential[row],
            rtx,
            rtol=1e-12,
            atol=1e-12 * F_scale,
        )
        del rty


# --- 21–23: partial admissibility and x_contact_rot -------------------------


def test_partial_candidates_reject_penetration_on_free_nodes(small_lookup) -> None:
    tol = Tolerances(tau_g=0.0, tau_R=1e9)
    ev = evaluate_candidates(small_lookup, 0.0, -1.0e4, -6.0, 5.0, tolerances=tol)
    for kind in (ContactType.HEEL, ContactType.TOE):
        for row in small_lookup.rows_for(kind):
            if not ev.well_conditioned[row]:
                continue
            free, _contact = small_lookup.record_sets(row)
            if free.size == 0:
                continue
            penetrated = bool(np.any(ev.full_bottom_gap[row][free] < -tol.tau_g))
            if penetrated:
                assert not bool(ev.admissible[row])


def test_partial_candidates_reject_tension_at_contact_nodes(small_lookup) -> None:
    tol = Tolerances(tau_g=1e9, tau_R=0.0)
    ev = evaluate_candidates(small_lookup, 0.0, -1.0e4, -6.0, 5.0, tolerances=tol)
    for kind in (ContactType.HEEL, ContactType.TOE):
        for row in small_lookup.rows_for(kind):
            if not ev.well_conditioned[row]:
                continue
            _free, contact = small_lookup.record_sets(row)
            if contact.size == 0:
                continue
            tensile = bool(np.any(ev.full_bottom_reaction_normal[row][contact] < -tol.tau_R))
            if tensile:
                assert not bool(ev.admissible[row])


def test_x_contact_rot_equals_l_plus_d_ax(small_lookup) -> None:
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)
    ev = evaluate_candidates(small_lookup, 50.0, -1.0e4, -2.0, 3.0)
    full = ev.full_contact_row
    assert np.isnan(ev.x_contact_rot[full])
    for kind in (ContactType.HEEL, ContactType.TOE):
        for row in small_lookup.rows_for(kind):
            if not ev.well_conditioned[row]:
                continue
            expected = float(ev.l[row]) + float(ev.d_ax[row])
            assert ev.x_contact_rot[row] == pytest.approx(expected, abs=abs_tol)
            assert ev.x_l_rot[row] == pytest.approx(expected, abs=abs_tol)


# --- 26–27: selection stability and direct FEM agreement --------------------


def test_auto_selection_stable_under_small_perturbations(small_lookup) -> None:
    Fy = -1.0e4
    F_scale = abs(Fy)
    L = float(small_lookup.L)
    base_tol = Tolerances(tau_g=1e-4 * L, tau_R=1e-4 * F_scale)
    sel0 = evaluate_from_angles(
        small_lookup, 0.0, Fy, -3.0, 2.0, tolerances=base_tol, mode=ContactMode.AUTO
    )
    assert sel0.selected_row is not None
    # Tiny relative force / angle / tolerance nudges should not flip the pick.
    for dFy, dphi, dtheta, dtau in (
        (1e-8 * F_scale, 0.0, 0.0, 0.0),
        (0.0, 1e-6, 0.0, 0.0),
        (0.0, 0.0, 1e-6, 0.0),
        (0.0, 0.0, 0.0, 1e-9 * F_scale),
    ):
        tol = Tolerances(
            tau_g=base_tol.tau_g,
            tau_R=base_tol.tau_R + dtau,
        )
        sel = evaluate_from_angles(
            small_lookup,
            0.0,
            Fy + dFy,
            -3.0 + dphi,
            2.0 + dtheta,
            tolerances=tol,
            mode=ContactMode.AUTO,
            previous_index=sel0.selected_index,
        )
        assert sel.selected_row == sel0.selected_row
        assert sel.contact_type == sel0.contact_type


def test_compliance_lookup_agrees_with_direct_fem(small_compliance) -> None:
    a, kappa = 0.6, 30.0
    specs = standard_comparison_specs(small_compliance.x_bottom, small_compliance.config.L)
    assert len(specs) == 3
    assert {s.contact_type for s in specs} == {
        ContactType.HEEL,
        ContactType.TOE,
        ContactType.FULL,
    }
    for spec in specs:
        for k in range(N_BASIS_MODES):
            comp = compare_compliance_to_direct_fem(small_compliance, spec, k, a, kappa)
            assert comp.top_force_rel_error < 1e-6
            assert comp.contact_reaction_rel_error < 1e-6
            assert comp.free_gap_rel_error < 1e-6
            # Relative aggregate errors are uninformative when a mode's resultant
            # is near null (e.g. pure horizontal contact translation).
            f_scale = max(float(np.linalg.norm(comp.f_t_fem)), 1e-30)
            n_t = len(comp.f_t_fem) // 2
            fy_f = float(np.sum(comp.f_t_fem[n_t:]))
            if abs(fy_f) > 1e-6 * f_scale:
                assert comp.Fy_rel_error < 1e-5
                assert comp.M_rel_error < 1e-5
                assert comp.T_toe_rel_error < 1e-5
