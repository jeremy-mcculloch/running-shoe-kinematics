"""Runtime affine force control and exact passive-toe equilibrium (spec tests 41-53)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.contact_basis import COL_ALPHA, COL_BRX, COL_BRY, COL_BX, COL_BY, COL_CONST
from compliance_fem.contact_lookup import (
    SCALAR_FX,
    SCALAR_FY,
    lookup_q_alpha_basis,
    lookup_rearfoot_geometry,
    recompute_q_alpha_basis,
)
from compliance_fem.corotation import contract_basis, rotate_force_to_local, rotate_vector_to_fixed
from compliance_fem.force_control import (
    angles_to_coefficients,
    evaluate_candidates,
    select_contact_candidate,
    solve_force_control,
)
from compliance_fem.gait.passive_toe import pick_instant_best_passive, solve_passive_toe_candidates
from compliance_fem.toe_spring import (
    ToeSpringConfig,
    rearfoot_toe_model,
    relaxed_root_index,
    solve_toe_equilibrium,
    solve_toe_equilibrium_model,
    spring_dU_dalpha,
)

WIDTH = 0.1


def _manual_translation(S, alpha, varphi, F_local):
    r_x, r_y = np.cos(varphi) - 1.0, -np.sin(varphi)
    K = np.array([[S[COL_BX, SCALAR_FX], S[COL_BY, SCALAR_FX]], [S[COL_BX, SCALAR_FY], S[COL_BY, SCALAR_FY]]])
    known = S[COL_CONST, :2] + alpha * S[COL_ALPHA, :2] + r_x * S[COL_BRX, :2] + r_y * S[COL_BRY, :2]
    return np.linalg.solve(K, F_local - known), K, known


def test_closure_on_the_correct_side_of_force_control(rocker_lookup) -> None:
    lk = rocker_lookup
    phi, theta = 1.5, 3.0
    varphi, alpha, r_x, r_y = angles_to_coefficients(phi, theta, lk.phi_ref)
    F_local = rotate_force_to_local(15.0, -600.0, varphi)
    row = lk.row_of(3, 9)
    S = lk.scalar_lookup[row]
    assert np.max(np.abs(S[COL_CONST, :2])) > 0.0
    d, K, known = _manual_translation(S, alpha, varphi, F_local)
    # K_F d = F* - F_closure - alpha F_alpha - r_x F_rx - r_y F_ry
    np.testing.assert_allclose(K @ d + known, F_local, rtol=1e-12, atol=1e-9)
    gamma = np.array([alpha, d[0], d[1], r_x, r_y])
    F_rec = contract_basis(S[:, :2], gamma)
    np.testing.assert_allclose(F_rec, F_local, rtol=1e-10, atol=1e-9)
    # Moving the closure to the other side breaks the force balance.
    d_wrong = np.linalg.solve(K, F_local - known + 2.0 * S[COL_CONST, :2])
    F_wrong = contract_basis(S[:, :2], np.array([alpha, *d_wrong, r_x, r_y]))
    assert np.linalg.norm(F_wrong - F_local) > 1.0


@pytest.mark.parametrize("lookup_name", ["flat_lookup", "rocker_lookup", "asym_lookup"])
def test_force_reconstruction_and_direct_dense_solve(request, lookup_name) -> None:
    lk = request.getfixturevalue(lookup_name)
    Fx, Fy, phi, theta = 30.0, -800.0, -2.0, 5.0
    ev = evaluate_candidates(lk, Fx, Fy, phi, theta)
    rows = np.flatnonzero(ev.evaluated)
    assert rows.size == lk.valid_rows.size
    assert np.max(ev.force_residual[rows]) <= 1e-8 * 800.0
    varphi, alpha, _, _ = angles_to_coefficients(phi, theta, lk.phi_ref)
    F_local = rotate_force_to_local(Fx, Fy, varphi)
    for row in rows[:: max(1, rows.size // 15)]:
        d, _, _ = _manual_translation(lk.scalar_lookup[row], alpha, varphi, F_local)
        assert ev.d_ax[row] == pytest.approx(d[0], rel=1e-9, abs=1e-15)
        assert ev.d_ay[row] == pytest.approx(d[1], rel=1e-9, abs=1e-15)
    fc = solve_force_control(lk, rows, alpha, varphi, F_local, 0.0)
    np.testing.assert_allclose(fc["d"][:, 0], ev.d_ax[rows], rtol=1e-12, atol=1e-18)
    assert np.all(np.isfinite(fc["cond"]))


def test_q_alpha_includes_closure(rocker_lookup) -> None:
    lk = rocker_lookup
    q = lookup_q_alpha_basis(lk)
    assert q.shape == (lk.n_records, 6)
    valid = lk.valid_rows
    assert np.max(np.abs(q[valid, COL_CONST])) > 0.0
    np.testing.assert_allclose(recompute_q_alpha_basis(lk)[valid], q[valid], rtol=1e-12, atol=1e-15)


def test_default_spring_parameters() -> None:
    cfg = ToeSpringConfig()
    assert cfg.toe_stiffness_Nm_per_rad == 25.0
    assert cfg.toe_neutral_angle_rad == 0.0
    assert cfg.toe_damping_Nms_per_rad == 0.0
    assert cfg.to_provenance()["damping_enters_equation"] is False


def test_spring_force_is_exact_not_small_angle() -> None:
    k = 25.0
    for theta in (0.05, 0.6, 1.1):
        alpha = np.tan(theta)
        assert spring_dU_dalpha(alpha, k) == pytest.approx(k * theta / (1.0 + alpha**2), rel=1e-14)
    # A small-angle form k * alpha differs substantially at 1.1 rad.
    assert abs(spring_dU_dalpha(np.tan(1.1), k) - k * np.tan(1.1)) > 10.0


@pytest.fixture(scope="module")
def passive_candidates(rocker_lookup):
    cfg = ToeSpringConfig()
    out = {}
    for Fy, phi_deg in [(-500.0, 0.0), (-2000.0, 2.0), (-4000.0, -3.0)]:
        out[(Fy, phi_deg)] = solve_passive_toe_candidates(
            rocker_lookup, Fx_star=0.0, Fy_star=Fy, Mz_meas=0.0,
            phi_rad=np.deg2rad(phi_deg), config=cfg, width_m=WIDTH,
        )
    return out


def test_passive_equilibrium_matches_direct_residual(rocker_lookup, passive_candidates) -> None:
    lk = rocker_lookup
    cfg = ToeSpringConfig()
    q_all = lookup_q_alpha_basis(lk)
    geometry = lookup_rearfoot_geometry(lk)
    for (Fy, phi_deg), cands in passive_candidates.items():
        assert cands, (Fy, phi_deg)
        for c in cands:
            assert c.admissible
            toe = c.toe
            gamma = np.asarray(c.gamma)
            Q_direct = WIDTH * float(contract_basis(q_all[c.row], gamma))
            assert toe.Q_alpha_shoe_on_foot == pytest.approx(Q_direct, rel=1e-9, abs=1e-12)
            residual = Q_direct - float(spring_dU_dalpha(c.alpha, cfg.toe_stiffness_Nm_per_rad))
            scale = max(abs(Q_direct), 1e-3)
            assert abs(residual) <= 1e-8 * scale + 1e-9
            assert toe.toe_equilibrium_residual_Nm == pytest.approx(residual, abs=1e-9)
            # gamma rotation entries are evaluated at the exact chord rotation (no small angle).
            varphi = geometry.chord_rotation(np.deg2rad(phi_deg), c.alpha)
            assert gamma[3] == pytest.approx(np.cos(varphi) - 1.0, abs=1e-15)
            assert gamma[4] == pytest.approx(-np.sin(varphi), abs=1e-15)
            # Force control holds at the toe solution.
            S = lk.scalar_lookup[c.row]
            F_loc = contract_basis(S[:, :2], gamma)
            fx, fy = rotate_vector_to_fixed(F_loc[0], F_loc[1], varphi)
            assert np.hypot(fx - 0.0, fy - Fy) <= 1e-8 * abs(Fy)
            assert c.force_residual <= 1e-8 * abs(Fy)


def test_multiple_roots_and_stability_handling() -> None:
    cfg = ToeSpringConfig()
    sol = solve_toe_equilibrium(0.0, 10.0, cfg)  # 0 < q1 < k: three roots
    assert sol.status == "multiple_roots" and len(sol.roots) == 3
    stable = [r.stable for r in sol.roots]
    assert stable == [False, True, False]
    assert sol.roots[0].theta_deg == pytest.approx(-sol.roots[2].theta_deg, rel=1e-9)
    assert relaxed_root_index(sol.roots) == 1
    single = solve_toe_equilibrium(0.0, 50.0, cfg)  # q1 > k: only the unstable neutral root
    assert len(single.roots) == 1 and not single.roots[0].stable


def test_low_load_returns_relaxed_solution(rocker_lookup) -> None:
    cands = solve_passive_toe_candidates(
        rocker_lookup, Fx_star=0.0, Fy_star=-1e-6, Mz_meas=0.0, phi_rad=0.0,
        config=ToeSpringConfig(), width_m=WIDTH, low_load=True,
    )
    best = pick_instant_best_passive(cands)
    assert best is not None
    assert abs(best.theta_deg) < 1e-3
    assert best.toe.low_load


def test_changing_interval_changes_passive_equilibrium(rocker_lookup) -> None:
    lk = rocker_lookup
    cfg = ToeSpringConfig()
    q_all = lookup_q_alpha_basis(lk)
    geometry = lookup_rearfoot_geometry(lk)
    F = np.array([50.0, -3000.0])
    thetas = {}
    for interval in [(0, 4), (4, 8), (8, 12)]:
        row = lk.row_of(*interval)
        model = rearfoot_toe_model(lk.scalar_lookup[row], q_all[row], F, 0.0, geometry, WIDTH)
        sol = solve_toe_equilibrium_model(model, cfg)
        assert sol.roots
        thetas[interval] = sol.roots[relaxed_root_index(sol.roots)].theta_rad
    vals = np.array(list(thetas.values()))
    assert np.ptp(vals) > 1e-4


def test_no_fem_solve_when_angle_or_interval_changes(rocker_lookup, monkeypatch) -> None:
    import compliance_fem.compliance as compliance
    import compliance_fem.contact_lookup as contact_lookup
    import scipy.sparse.linalg as spla

    def boom(*args, **kwargs):
        raise AssertionError("runtime FEM solve attempted")

    monkeypatch.setattr(compliance, "compute_compliance", boom)
    monkeypatch.setattr(contact_lookup, "solve_candidate", boom)
    monkeypatch.setattr(contact_lookup, "generate_contact_lookup", boom)
    monkeypatch.setattr(spla, "spsolve", boom)
    lk = rocker_lookup
    for theta in (0.0, 4.0, -6.0):
        ev = evaluate_candidates(lk, 0.0, -700.0, 1.0, theta)
        for interval in [(0, 3), (4, 8), (5, 12)]:
            select_contact_candidate(ev, mode="specific", specific_interval=interval)
        select_contact_candidate(ev)
    solve_passive_toe_candidates(
        lk, Fx_star=0.0, Fy_star=-700.0, Mz_meas=0.0, phi_rad=0.01,
        config=ToeSpringConfig(), width_m=WIDTH, previous_interval=(4, 8),
    )
