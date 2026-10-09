"""Edge quantities and all-node unilateral admissibility (spec tests 27-40)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compliance_fem.contact.basis import COL_CONST
from compliance_fem.contact.corotation import rotation_matrix, transform_to_fixed_frame
from compliance_fem.contact.force_control import (
    DISCONNECTED_CONTACT_WARNING,
    Tolerances,
    evaluate_candidates,
    mesh_spacing,
    reconstruct_rows,
    runtime_scales,
    select_contact_candidate,
)

@pytest.fixture(scope="module")
def flat_lookup(flat_lookup_full):
    """Stored fields so rejected rows also carry exact nodal gaps and reactions."""
    return flat_lookup_full


@pytest.fixture(scope="module")
def rocker_lookup(rocker_lookup_full):
    return rocker_lookup_full


EDGE_KEYS = (
    "heel_edge_reaction_local_x",
    "heel_edge_reaction_local_y",
    "toe_edge_reaction_local_x",
    "toe_edge_reaction_local_y",
)


def _selected(lookup, Fx, Fy, phi_deg, theta_deg=0.0, **kw):
    ev = evaluate_candidates(lookup, Fx, Fy, phi_deg, theta_deg)
    return ev, select_contact_candidate(ev, **kw)


def _gamma(sel):
    return np.array([[sel.alpha, sel.d_ax, sel.d_ay, np.cos(sel.varphi) - 1.0, -np.sin(sel.varphi)]])


def test_both_edge_force_components_stored_at_both_ends(rocker_lookup) -> None:
    lk = rocker_lookup
    er = lk.edge_responses
    for row in lk.valid_rows:
        i, j = int(lk.contact_start_index[row]), int(lk.contact_end_index[row])
        for key in EDGE_KEYS:
            assert er[key][row].shape == (6,)
            assert np.all(np.isfinite(er[key][row]))
        np.testing.assert_array_equal(er["heel_edge_reaction_local_x"][row], lk.reaction_x_basis[row, :, i])
        np.testing.assert_array_equal(er["heel_edge_reaction_local_y"][row], lk.reaction_y_basis[row, :, i])
        np.testing.assert_array_equal(er["toe_edge_reaction_local_x"][row], lk.reaction_x_basis[row, :, j])
        np.testing.assert_array_equal(er["toe_edge_reaction_local_y"][row], lk.reaction_y_basis[row, :, j])


@pytest.mark.parametrize("phi_deg", [0.0, 0.3, -0.3, 4.0])
def test_fixed_frame_edge_reactions_and_normals(rocker_lookup, phi_deg) -> None:
    lk = rocker_lookup
    ev, sel = _selected(lk, 20.0, -400.0, phi_deg)
    Q = rotation_matrix(sel.varphi)
    for side in ("heel", "toe"):
        local = np.array([getattr(sel, f"{side}_edge_reaction_local_x"), getattr(sel, f"{side}_edge_reaction_local_y")])
        fixed = np.array([getattr(sel, f"{side}_edge_reaction_fixed_x"), getattr(sel, f"{side}_edge_reaction_fixed_y")])
        np.testing.assert_allclose(fixed, Q @ local, rtol=1e-12, atol=1e-12 * np.abs(local).max())
        n_g = np.array([0.0, 1.0])
        assert getattr(sel, f"{side}_edge_normal_reaction") == pytest.approx(float(n_g @ fixed), rel=1e-12, abs=1e-12)
    i, j = sel.interval
    Rn = sel.full_bottom_reaction_normal
    assert sel.heel_edge_normal_reaction == pytest.approx(Rn[i], rel=1e-10, abs=1e-10)
    assert sel.toe_edge_normal_reaction == pytest.approx(Rn[j], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("phi_deg", [0.0, 0.5, -0.5, 6.0])
def test_adjacent_gaps_match_direct_coordinate_reconstruction(rocker_lookup, phi_deg) -> None:
    lk = rocker_lookup
    ev, sel = _selected(lk, 10.0, -300.0, phi_deg)
    rows = [r for r in lk.valid_rows if ev.evaluated[r]]
    for row in rows[:: max(1, len(rows) // 25)]:
        i, j = int(lk.contact_start_index[row]), int(lk.contact_end_index[row])
        x_a, y_a = lk.contact_anchor_reference_x[row], lk.contact_anchor_reference_y[row]
        u = ev.full_bottom_u[row]
        v = ev.full_bottom_v[row]
        _, yF = transform_to_fixed_frame(
            lk.x_bottom, lk.y_bottom, u, v, x_a, ev.d_ax[row], ev.d_ay[row], ev.varphi, y_anchor=y_a
        )
        if i > 0:
            assert ev.heel_adjacent_free_gap[row] == pytest.approx(yF[i - 1], rel=1e-9, abs=1e-14)
        else:
            assert np.isnan(ev.heel_adjacent_free_gap[row])
        if j < lk.n_bottom_nodes - 1:
            assert ev.toe_adjacent_free_gap[row] == pytest.approx(yF[j + 1], rel=1e-9, abs=1e-14)
        else:
            assert np.isnan(ev.toe_adjacent_free_gap[row])
        np.testing.assert_allclose(ev.full_bottom_gap[row], yF, rtol=1e-9, atol=1e-14)


@pytest.fixture(scope="module")
def heel_case(flat_lookup):
    """Admissible heel-attached state with many free nodes: flat sole, toe up 0.3 deg."""
    ev, sel = _selected(flat_lookup, 0.0, -500.0, 0.3)
    assert sel.exactly_admissible and sel.interval == (0, 4)
    return ev, sel


@pytest.fixture(scope="module")
def long_case(flat_lookup):
    """Admissible long heel-attached interval: flat sole, heavy load, toe up 0.1 deg."""
    ev, sel = _selected(flat_lookup, 0.0, -2000.0, 0.1)
    assert sel.exactly_admissible and sel.interval == (0, 11)
    return ev, sel


def test_penetration_at_any_free_node_rejects_and_endpoint_checks_do_not_hide_it(flat_lookup, heel_case) -> None:
    ev, sel = heel_case
    row = sel.selected_row
    k = 8  # a free node far from the toe-side edge (adjacent free node is 5)
    bv = flat_lookup.bottom_v_basis.copy()
    bv[row, COL_CONST, k] -= 10.0 * (sel.full_bottom_gap[k] + 1e-6)
    lk2 = replace(flat_lookup, bottom_v_basis=bv)
    rec = reconstruct_rows(lk2, np.array([row]), _gamma(sel), sel.varphi, np.array([0.0, -500.0]), Tolerances())
    assert rec["toe_adjacent_free_gap"][0] > 0.0  # endpoint check alone would pass
    assert rec["min_free_gap"][0] < 0.0 and rec["min_free_gap_node"][0] == k
    assert rec["max_free_penetration"][0] > 0.0
    assert not rec["admissible"][0]
    assert rec["gap_violation_term"][0] > 0.0


def test_tension_at_any_contact_node_rejects_and_endpoint_checks_do_not_hide_it(flat_lookup, long_case) -> None:
    ev, sel = long_case
    row = sel.selected_row
    k = 5  # interior contact node (edges are the selected interval ends)
    ry = flat_lookup.reaction_y_basis.copy()
    ry[row, COL_CONST, k] -= 10.0 * (abs(sel.full_bottom_reaction_normal[k]) + 1.0)
    lk2 = replace(flat_lookup, reaction_y_basis=ry)
    rec = reconstruct_rows(lk2, np.array([row]), _gamma(sel), sel.varphi, np.array([0.0, -2000.0]), Tolerances())
    assert rec["heel_edge_normal_reaction"][0] > 0.0 and rec["toe_edge_normal_reaction"][0] > 0.0
    assert rec["min_contact_reaction"][0] < 0.0 and rec["min_contact_reaction_node"][0] == k
    assert rec["max_contact_tension"][0] > 0.0
    assert rec["disconnected_contact_warning"][0]
    assert not rec["admissible"][0]
    assert rec["reaction_violation_term"][0] > 0.0


def test_physical_interior_tension_rejects_single_interval(rocker_lookup) -> None:
    """Forcing full contact on a rocker sole pulls the raised ends down: tensile reactions
    away from the apex make the single-interval candidate inadmissible and raise the warning."""
    ev, sel = _selected(rocker_lookup, 0.0, -50.0, 0.0, mode="full")
    assert sel.interval == (0, rocker_lookup.n_bottom_nodes - 1)
    assert not sel.exactly_admissible
    assert sel.disconnected_contact_warning
    assert DISCONNECTED_CONTACT_WARNING in sel.message
    Rn = sel.full_bottom_reaction_normal
    assert np.any(Rn[1:-1] < 0.0) and np.any(Rn[1:-1] > 0.0)
    auto = select_contact_candidate(ev)
    assert auto.exactly_admissible and auto.topology_label == "interior"


def test_full_contact_reports_both_outside_gaps_unavailable(flat_lookup) -> None:
    ev, sel = _selected(flat_lookup, 0.0, -50.0, 0.0, mode="full")
    assert np.isnan(sel.heel_adjacent_free_gap) and np.isnan(sel.toe_adjacent_free_gap)
    assert sel.min_free_gap_node == -1 and np.isinf(sel.min_free_gap)
    assert sel.max_free_penetration == 0.0


def test_heel_and_toe_attached_report_one_missing_outside_gap(flat_lookup) -> None:
    n = flat_lookup.n_bottom_nodes
    ev, heel = _selected(flat_lookup, 0.0, -500.0, 0.3, mode="specific", specific_interval=(0, 4))
    assert np.isnan(heel.heel_adjacent_free_gap) and np.isfinite(heel.toe_adjacent_free_gap)
    _, toe = _selected(flat_lookup, 0.0, -500.0, -0.3, mode="specific", specific_interval=(7, n - 1))
    assert np.isnan(toe.toe_adjacent_free_gap) and np.isfinite(toe.heel_adjacent_free_gap)
    _, interior = _selected(flat_lookup, 0.0, -500.0, 0.0, mode="specific", specific_interval=(3, 8))
    assert np.isfinite(interior.heel_adjacent_free_gap) and np.isfinite(interior.toe_adjacent_free_gap)


def test_tolerances_are_mesh_and_magnitude_scaled(flat_lookup) -> None:
    tol = Tolerances(tau_g=1e-9, tau_R=1e-6, gap_rel_tol=1e-6, reaction_rel_tol=1e-5)
    h = mesh_spacing(flat_lookup)
    assert h == pytest.approx(flat_lookup.L / (flat_lookup.n_bottom_nodes - 1))
    s1 = runtime_scales(flat_lookup, 0.0, -100.0, tol)
    s2 = runtime_scales(flat_lookup, 0.0, -1000.0, tol)
    assert s1.tau_g_eff == pytest.approx(1e-9 + 1e-6 * h)
    assert s1.g_s == pytest.approx(h)
    assert s1.R_ref == pytest.approx(100.0 * h / flat_lookup.L)
    assert s2.R_ref == pytest.approx(10.0 * s1.R_ref)
    assert s2.tau_R_eff - 1e-6 == pytest.approx(10.0 * (s1.tau_R_eff - 1e-6))
    # Below the force floor the reaction scale stays finite.
    s0 = runtime_scales(flat_lookup, 0.0, 0.0, tol)
    assert s0.R_ref == pytest.approx(tol.F_floor * h / flat_lookup.L)
