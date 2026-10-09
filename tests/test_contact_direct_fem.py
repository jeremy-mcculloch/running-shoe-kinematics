"""Direct sparse-FEM validation of interval lookup superposition (spec section 28).

Every affine column of representative intervals is checked against a direct FEM
solve, then complete runtime states (force-controlled, rotated, with the toe
mode and closure) are compared quantity by quantity.
"""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.contact.basis import N_AFFINE_COLUMNS
from compliance_fem.contact.direct_fem import (
    run_standard_direct_comparisons,
    solve_direct_fem_contact,
    standard_validation_intervals,
)
from compliance_fem.contact.lookup import (
    lookup_q_alpha_basis,
    lookup_rearfoot_geometry,
)
from compliance_fem.contact.corotation import contract_basis
from compliance_fem.contact.force_control import Tolerances, evaluate_candidates, reconstruct_rows, select_contact_candidate
from compliance_fem.gait.passive_toe import pick_instant_best_passive, solve_passive_toe_candidates
from compliance_fem.contact.toe_spring import (
    ToeSpringConfig,
    q_alpha_shoe_on_foot,
    rearfoot_toe_mode,
    spring_dU_dalpha,
    top_shape_mode_vertical,
)

TOL = 1e-8
WIDTH = 0.1


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


@pytest.mark.parametrize("case_name", ["flat_case", "rocker_case", "asym_case"])
def test_every_affine_column_agrees_with_direct_fem(request, case_name) -> None:
    result, lookup = request.getfixturevalue(case_name)
    comps = run_standard_direct_comparisons(result, lookup.softplus_toe_length, lookup.softplus_kappa)
    names = set(standard_validation_intervals(result.x_bottom, result.config.L))
    assert set(comps) == names
    for name, cols in comps.items():
        assert [c.column_index for c in cols] == list(range(N_AFFINE_COLUMNS))
        for c in cols:
            assert c.top_force_rel_error < TOL, (name, c.column_name)
            assert c.contact_reaction_rel_error < TOL, (name, c.column_name)
            assert c.free_gap_rel_error < TOL, (name, c.column_name)
            assert c.T_toe_rel_error < TOL and c.M_rel_error < TOL and c.Fy_rel_error < TOL
            assert c.free_traction_rel < 1e-6


def _validate_state(result, lk, row, gamma, varphi, F_star, *, toe_alpha=None):
    """Compare a combined runtime state against a direct FEM solve; return relative errors."""
    iv = lk.interval(row)
    fem = solve_direct_fem_contact(result, iv, 0, lk.softplus_toe_length, lk.softplus_kappa, gamma=gamma)
    u = fem.u
    n_t = lk.n_top_nodes
    rec = reconstruct_rows(lk, np.array([row]), np.asarray(gamma)[None, :], varphi, F_star, Tolerances())
    errs: dict[str, float] = {}

    ftx = rec["top_force_x"][0]
    fty = rec["top_force_y"][0]
    errs["top_force"] = _rel(np.concatenate([ftx, fty]), fem.f_t)
    contact, free = iv.contact_node_ids, iv.free_node_ids
    rx = rec["full_bottom_reaction_x"][0][contact]
    ry = rec["full_bottom_reaction_y"][0][contact]
    errs["contact_force"] = _rel(np.concatenate([rx, ry]), fem.r_c)

    # Fixed-frame gaps at all bottom nodes from the direct displacement field.
    u_bot, v_bot = result.selector_dofs("bottom")
    x_b, y_b = result.x_bottom, result.y_bottom
    np.testing.assert_allclose(x_b, lk.x_bottom, atol=1e-14)
    x_a, y_a = lk.contact_anchor_reference_x[row], lk.contact_anchor_reference_y[row]
    d_ax, d_ay = gamma[1], gamma[2]
    c, s = np.cos(varphi), np.sin(varphi)
    g_direct = s * (x_b - x_a + u[u_bot] - d_ax) + c * (y_b - y_a + u[v_bot] - d_ay)
    g_lookup = rec["full_bottom_gap"][0]
    disp_scale = max(np.max(np.abs(u[u_bot])), np.max(np.abs(u[v_bot])))
    gap_scale = max(np.max(np.abs(g_direct)), disp_scale, 1e-300)
    errs["free_gap"] = float(np.max(np.abs(g_lookup[free] - g_direct[free])) / gap_scale) if free.size else 0.0
    errs["contact_ground"] = float(np.max(np.abs(g_direct[contact])) / gap_scale)

    r_scale = max(np.max(np.abs(fem.r_c)), 1e-300)
    n_c = contact.size
    fem_rx, fem_ry = fem.r_c[:n_c], fem.r_c[n_c:]
    errs["heel_edge"] = float(max(abs(rec["heel_edge_reaction_local_x"][0] - fem_rx[0]),
                                  abs(rec["heel_edge_reaction_local_y"][0] - fem_ry[0])) / r_scale)
    errs["toe_edge"] = float(max(abs(rec["toe_edge_reaction_local_x"][0] - fem_rx[-1]),
                                 abs(rec["toe_edge_reaction_local_y"][0] - fem_ry[-1])) / r_scale)
    for side, k in (("heel", iv.heel_adjacent_free_node_id), ("toe", iv.toe_adjacent_free_node_id)):
        val = rec[f"{side}_adjacent_free_gap"][0]
        if k < 0:
            assert np.isnan(val)
            errs[f"{side}_adjacent_gap"] = 0.0
        else:
            errs[f"{side}_adjacent_gap"] = float(abs(val - g_direct[k]) / gap_scale)

    geometry = lookup_rearfoot_geometry(lk)
    psi = rearfoot_toe_mode(top_shape_mode_vertical(lk.basis_top_displacements, n_t), lk.x_top,
                            geometry.L, geometry.toe_length, geometry.phi1_mtp)
    fem_fty = fem.f_t[n_t:]
    Q_direct = float(q_alpha_shoe_on_foot(fem_fty, psi))
    Q_lookup = float(contract_basis(lookup_q_alpha_basis(lk)[row], gamma))
    errs["toe_generalized_force"] = abs(Q_lookup - Q_direct) / max(abs(Q_direct), np.sum(np.abs(fem_fty)) * lk.L, 1e-300)
    if toe_alpha is not None:
        dU = float(spring_dU_dalpha(toe_alpha, ToeSpringConfig().toe_stiffness_Nm_per_rad))
        errs["toe_equilibrium_residual"] = abs(WIDTH * Q_direct - dU) / max(abs(dU), 1e-6)

    if lk.has_plate_response:
        pu = contract_basis(lk.plate_u_local_basis[row], gamma)
        pv = contract_basis(lk.plate_v_local_basis[row], gamma)
        pth = contract_basis(lk.plate_rotation_local_basis[row], gamma)
        errs["plate_displacement"] = _rel(np.concatenate([pu, pv]),
                                          np.concatenate([u[lk.plate_u_dof_ids], u[lk.plate_v_dof_ids]]))
        errs["plate_rotation"] = _rel(pth, u[lk.plate_rotation_dof_ids])

    f_tx_fem, f_ty_fem = fem.f_t[:n_t], fem.f_t[n_t:]
    y_t = lk.y_top if lk.y_top is not None else np.full(n_t, lk.H)
    F_tot = np.array([f_tx_fem.sum() + fem_rx.sum(), f_ty_fem.sum() + fem_ry.sum()])
    M_tot = (lk.x_top @ f_ty_fem - y_t @ f_tx_fem) + (lk.x_bottom[contact] @ fem_ry - lk.y_bottom[contact] @ fem_rx)
    f_scale = np.sum(np.abs(fem.f_t))
    errs["force_equilibrium"] = float(np.max(np.abs(F_tot)) / f_scale)
    errs["moment_equilibrium"] = float(abs(M_tot) / (f_scale * lk.L))
    return errs, rec


def _sel_state(lk, Fx, Fy, phi_deg, theta_deg=0.0, **kw):
    ev = evaluate_candidates(lk, Fx, Fy, phi_deg, theta_deg)
    sel = select_contact_candidate(ev, **kw)
    gamma = np.array([sel.alpha, sel.d_ax, sel.d_ay, np.cos(sel.varphi) - 1.0, -np.sin(sel.varphi)])
    return sel, gamma


CASES = {
    "flat_heel": ("flat_case", dict(Fx=0.0, Fy=-500.0, phi_deg=0.3), "heel"),
    "flat_toe": ("flat_case", dict(Fx=0.0, Fy=-500.0, phi_deg=-0.3), "toe"),
    "flat_full": ("flat_case", dict(Fx=0.0, Fy=-50.0, phi_deg=0.0), "full"),
    "symmetric_concave_interior": ("rocker_case", dict(Fx=0.0, Fy=-50.0, phi_deg=0.0), "interior"),
    "asymmetric_interior": ("asym_case", dict(Fx=10.0, Fy=-300.0, phi_deg=0.0), "interior"),
    "one_free_node_each_side": ("rocker_case", dict(Fx=0.0, Fy=-2000.0, phi_deg=0.0, mode="specific",
                                                    specific_interval=(1, 11)), "interior"),
    "tensile_interior": ("rocker_case", dict(Fx=0.0, Fy=-50.0, phi_deg=0.0, mode="full"), "full"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_direct_validation_of_runtime_states(request, name) -> None:
    case_name, kwargs, label = CASES[name]
    result, lk = request.getfixturevalue(case_name)
    kw = dict(kwargs)
    Fx, Fy, phi = kw.pop("Fx"), kw.pop("Fy"), kw.pop("phi_deg")
    theta = kw.pop("theta_deg", 0.0)
    sel, gamma = _sel_state(lk, Fx, Fy, phi, theta, **kw)
    if label is not None:
        assert sel.topology_label == label
    if name == "asymmetric_interior":
        i, j = sel.interval
        assert i + j != lk.n_bottom_nodes - 1  # not centred on mid-length
    if name == "tensile_interior":
        assert sel.disconnected_contact_warning and not sel.exactly_admissible
    elif kw.get("mode") is None:
        assert sel.exactly_admissible
    errs, rec = _validate_state(result, lk, sel.selected_row, gamma, sel.varphi, np.array([Fx, Fy]))
    assert rec["force_residual"][0] <= 1e-8 * abs(Fy)
    for key, val in errs.items():
        assert val < TOL, (name, key, val)
    if lk.has_plate_response:
        assert "plate_displacement" in errs and "plate_rotation" in errs


def test_direct_validation_of_gait_sample_with_passive_toe(rocker_case) -> None:
    result, lk = rocker_case
    Fx, Fy, phi = 40.0, -2000.0, np.deg2rad(1.0)
    cands = solve_passive_toe_candidates(
        lk, Fx_star=Fx, Fy_star=Fy, Mz_meas=0.0, phi_rad=phi, config=ToeSpringConfig(), width_m=WIDTH,
    )
    best = pick_instant_best_passive(cands)
    assert best is not None and best.admissible and best.toe.toe_equilibrium_stable
    gamma = np.asarray(best.gamma)
    varphi = lookup_rearfoot_geometry(lk).chord_rotation(phi, best.alpha)
    errs, rec = _validate_state(result, lk, best.row, gamma, varphi, np.array([Fx, Fy]), toe_alpha=best.alpha)
    for key, val in errs.items():
        assert val < TOL, (key, val)
    assert rec["admissible"][0]


def test_validation_intervals_cover_required_cases(flat_case) -> None:
    result, _ = flat_case
    specs = standard_validation_intervals(result.x_bottom, result.config.L)
    n = len(result.x_bottom)
    assert specs["flat_heel"].topology_label.value == "heel"
    assert specs["flat_toe"].topology_label.value == "toe"
    assert (specs["full"].start, specs["full"].end) == (0, n - 1)
    sym = specs["symmetric_interior"]
    assert sym.start + sym.end == n - 1 and sym.topology_label.value == "interior"
    asym = specs["asymmetric_interior"]
    assert asym.start + asym.end != n - 1 and asym.topology_label.value == "interior"
    one = specs["one_free_node_each_side"]
    assert (one.start, one.end) == (1, n - 2)
    assert specs["single_node"].is_single_node
