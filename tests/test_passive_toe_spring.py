"""Passive toe-joint spring: exact equilibrium, sign, schema, runtime, and GUI/export."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.contact.basis import COL_ALPHA, MODE_PHI1, shape_mode_phi1
from compliance_fem.contact.direct_fem import solve_direct_fem_contact
from compliance_fem.contact.lookup import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    get_compliance_block_matrix,
    SCALAR_TOE,
    generate_contact_lookup,
    load_contact_lookup,
    lookup_q_alpha_basis,
    lookup_rearfoot_geometry,
    recompute_q_alpha_basis,
    save_contact_lookup,
)
from compliance_fem.contact.topology import ContactType
from compliance_fem.contact.corotation import (
    contract_basis,
    rotate_force_to_local,
    rotation_coefficients,
)
from compliance_fem.contact.force_control import evaluate_candidates
from compliance_fem.gait.passive_toe import pick_instant_best_passive, solve_passive_toe_candidates
from compliance_fem.gait.replay import replay_stance, resolve_toe_config, save_replay_result
from compliance_fem.gait.candidate import unilateral_check
from compliance_fem.contact.toe_spring import (
    DEFAULT_TOE_MODEL,
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    ToeSpringConfig,
    potential_curvature,
    q_alpha_foot_on_shoe,
    q_alpha_shoe_on_foot,
    q_theta_from_q_alpha,
    rearfoot_toe_mode,
    rearfoot_toe_model,
    solve_toe_equilibrium,
    solve_toe_equilibrium_model,
    spring_d2U_dalpha2,
    spring_dU_dalpha,
    spring_energy,
    spring_generalized_force_alpha,
    spring_moment_theta,
    toe_affine_model,
    toe_residual,
    top_shape_mode_vertical,
    total_potential,
)

from conftest import small_config

K = 25.0
TOE_LENGTH_SOFT = 0.4
KAPPA = 30.0


@pytest.fixture(scope="module")
def fem():
    return compute_compliance(
        small_config(
            L=1.0,
            height=0.05,
            ffturbo_E_Pa=1.0e6,
            ffturbo_nu=0.3,
            ffleap_E_heel_Pa=1.0e6,
            ffleap_E_toe_Pa=1.0e6,
            ffleap_nu=0.3,
            EI_plate_Nm2_per_m=1.0,
        )
    )


@pytest.fixture(scope="module")
def lookup(fem):
    return generate_contact_lookup(get_compliance_block_matrix(fem), toe_length=TOE_LENGTH_SOFT, kappa=KAPPA, fem_result=fem, store_fields=True)


def _rows(lookup, kind: ContactType) -> np.ndarray:
    return lookup.rows_for(kind)


def _representative_rows(lookup) -> dict[str, int]:
    heel = _rows(lookup, ContactType.HEEL)
    toe = _rows(lookup, ContactType.TOE)
    return {"heel": int(heel[len(heel) // 2]), "full": lookup.full_contact_row(), "toe": int(toe[len(toe) // 2])}


def _phi1(lookup) -> np.ndarray:
    return top_shape_mode_vertical(lookup.basis_top_displacements, len(lookup.x_top))


def _psi(lookup) -> np.ndarray:
    geo = lookup_rearfoot_geometry(lookup)
    return rearfoot_toe_mode(_phi1(lookup), lookup.x_top, geo.L, geo.toe_length, geo.phi1_mtp)


def _model(lookup, row, Fx, Fy, phi, width=1.0):
    q_basis = lookup_q_alpha_basis(lookup)
    return rearfoot_toe_model(
        lookup.scalar_lookup[row], q_basis[row], np.array([Fx, Fy]), phi, lookup_rearfoot_geometry(lookup), width
    )


def _linear_W(lookup) -> np.ndarray:
    """The five linear top modes; the closure column is zero on the top."""
    W = np.asarray(lookup.basis_top_displacements)
    assert W.shape[1] == 6
    np.testing.assert_array_equal(W[:, 0], 0.0)
    return W[:, 1:]


def _row_root(lookup, row, Fx, Fy, phi, width=1.0, cfg=None):
    """Exact passive-toe solve for one interval row, independent of candidate search."""
    cfg = cfg or ToeSpringConfig()
    m = _model(lookup, row, Fx, Fy, phi, width)
    sol = solve_toe_equilibrium_model(m, cfg)
    return m, sol


def _within_toe_tol(c, cfg) -> bool:
    scale = max(abs(c.toe.Q_alpha_shoe_on_foot), abs(float(spring_dU_dalpha(c.alpha, cfg.toe_stiffness_Nm_per_rad))))
    tol = cfg.toe_equilibrium_abs_tol_Nm + cfg.toe_equilibrium_rel_tol * max(scale, abs(c.toe.q0_Nm))
    return abs(c.toe.toe_equilibrium_residual_Nm) <= 10.0 * tol


# 1. Shape-mode derivative.
def test_shape_mode_derivative_wrt_alpha(lookup) -> None:
    W = _linear_W(lookup)
    n_t = len(lookup.x_top)
    gamma0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
    eps = 1e-6
    u0 = W @ gamma0
    u1 = W @ (gamma0 + np.array([eps, 0, 0, 0, 0]))
    du = (u1 - u0) / eps
    np.testing.assert_allclose(du[:n_t], 0.0, atol=1e-12)
    np.testing.assert_allclose(du[n_t:], shape_mode_phi1(lookup.x_top, lookup.L, TOE_LENGTH_SOFT, KAPPA), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(_phi1(lookup), W[n_t:, MODE_PHI1])
    np.testing.assert_allclose(_phi1(lookup), np.asarray(lookup.basis_top_displacements)[n_t:, COL_ALPHA])


# 2. Virtual work vs nodal contraction.
def test_q_alpha_virtual_work_matches_nodal_contraction(lookup) -> None:
    rng = np.random.default_rng(0)
    W = _linear_W(lookup)
    n_t = len(lookup.x_top)
    for row in _representative_rows(lookup).values():
        gamma = rng.normal(size=5) * np.array([0.1, 1e-3, 1e-3, 1e-2, 1e-2])
        f_tx = contract_basis(lookup.top_force_x_basis[row], gamma)
        f_ty = contract_basis(lookup.top_force_y_basis[row], gamma)
        f_t = np.concatenate([f_tx, f_ty])
        eps = 1e-7
        du = (W @ (gamma + np.array([eps, 0, 0, 0, 0])) - W @ gamma) / eps
        q_vw = float(f_t @ du)
        q_chord = float(q_alpha_foot_on_shoe(f_ty, _phi1(lookup)))
        assert abs(q_vw - q_chord) / max(abs(q_chord), 1e-12) < 1e-6
        assert np.all(du[:n_t] == 0.0)
        # Rearfoot-fixed virtual displacement psi = phi1 + (rigid rotation about the heel).
        q_rf = float(q_alpha_foot_on_shoe(f_ty, _psi(lookup)))
        geo = lookup_rearfoot_geometry(lookup)
        q_rf_mv = q_chord - (geo.phi1_mtp / (geo.L - geo.toe_length)) * float(lookup.x_top @ f_ty)
        q_basis = -float(contract_basis(lookup_q_alpha_basis(lookup)[row], gamma))
        scale = max(abs(q_rf), 1e-12)
        assert abs(q_rf - q_rf_mv) / scale < 1e-12
        assert abs(q_basis - q_rf) / scale < 1e-12


def test_rearfoot_mode_vanishes_at_heel_and_mtp(lookup) -> None:
    geo = lookup_rearfoot_geometry(lookup)
    x = np.array([0.0, geo.L - geo.toe_length, lookup.L])
    phi1 = shape_mode_phi1(x, lookup.L, TOE_LENGTH_SOFT, KAPPA)
    psi = rearfoot_toe_mode(phi1, x, geo.L, geo.toe_length, geo.phi1_mtp)
    assert psi[0] == pytest.approx(0.0, abs=1e-15)
    assert psi[1] == pytest.approx(0.0, abs=1e-15)
    # Toe tip rises relative to the rearfoot line by ~ toe_length per unit alpha.
    assert psi[2] > 0.0
    assert psi[2] == pytest.approx(geo.toe_length, rel=0.1)
    assert np.all(_psi(lookup)[np.asarray(lookup.x_top) >= geo.L - geo.toe_length] >= -1e-12)


# 3. Units: phi1 is a length, so Q = phi1^T f is force x length.
def test_q_alpha_units_force_times_length(lookup) -> None:
    x = np.linspace(0.0, 1.0, 17)
    s = 3.7
    np.testing.assert_allclose(shape_mode_phi1(s * x, s * 1.0, s * TOE_LENGTH_SOFT, KAPPA), s * shape_mode_phi1(x, 1.0, TOE_LENGTH_SOFT, KAPPA), rtol=1e-12)
    f = np.linspace(-2.0, 1.0, 17)
    assert q_alpha_foot_on_shoe(f, s * shape_mode_phi1(x, 1.0, TOE_LENGTH_SOFT, KAPPA)) == pytest.approx(
        s * q_alpha_foot_on_shoe(f, shape_mode_phi1(x, 1.0, TOE_LENGTH_SOFT, KAPPA))
    )
    cfg = ToeSpringConfig(toe_stiffness_Nm_per_rad=0.0)
    kw = dict(Fx_star=0.0, Fy_star=-2.0e3, Mz_meas=0.0, phi_rad=0.0, config=cfg)
    c1 = solve_passive_toe_candidates(lookup, width_m=1.0, **kw)
    c2 = solve_passive_toe_candidates(lookup, width_m=0.1, **kw)
    for a, b in zip(c1, c2):
        if a.toe.converged and b.toe.converged:
            assert b.toe.q0_Nm == pytest.approx(0.1 * a.toe.q0_Nm, rel=1e-12, abs=1e-12)


# 4. Stored vs recomputed.
def test_stored_basis_matches_recomputation(lookup) -> None:
    stored = np.asarray(lookup.Q_alpha_shoe_on_foot_basis)
    assert stored.shape == (lookup.n_records, 6)
    valid = lookup.valid_mask
    np.testing.assert_allclose(stored[valid], recompute_q_alpha_basis(lookup)[valid], rtol=1e-13, atol=0.0)


# 5. Opposite signs + stiffness sign proof from strain energy.
def test_foot_on_shoe_and_shoe_on_foot_signs(fem, lookup) -> None:
    f = np.array([1.0, -3.0, 2.0])
    phi = np.array([0.0, -0.2, 0.0])
    assert q_alpha_shoe_on_foot(f, phi) == pytest.approx(-q_alpha_foot_on_shoe(f, phi))
    phi_rf = lookup_rearfoot_geometry(lookup).reference_angle
    for name, row in _representative_rows(lookup).items():
        # Zero resultant, rearfoot at its reference angle: bending the toe stores
        # strain energy E ~ 1/2 (-q1) alpha^2 > 0, so the shoe resists (q1 < 0).
        m = _model(lookup, row, 0.0, 0.0, phi_rf)
        assert m.q0 == pytest.approx(0.0, abs=1e-12)
        assert m.q1 < 0.0, name
        alpha = 1e-3
        gamma = m.gamma(alpha)
        direct = solve_direct_fem_contact(fem, lookup.interval(row), 0, TOE_LENGTH_SOFT, KAPPA, gamma=gamma)
        u = np.asarray(direct.u)
        strain_energy = 0.5 * float(u @ (fem.K @ u))
        assert strain_energy > 0.0
        assert strain_energy == pytest.approx(-float(m.work(alpha)), rel=1e-3), name
        assert strain_energy == pytest.approx(0.5 * (-m.q1) * alpha**2, rel=5e-3), name


# 6-7. Spring opposes displacement; zero at neutral.
def test_spring_moment_opposes_displacement_and_vanishes_at_neutral() -> None:
    for th in (0.3, -0.3):
        a = np.tan(th)
        assert np.sign(spring_moment_theta(a, K)) == -np.sign(th)
        assert np.sign(spring_generalized_force_alpha(a, K)) == -np.sign(th)
    assert spring_moment_theta(0.0, K) == 0.0
    assert spring_dU_dalpha(0.0, K) == 0.0
    th0 = 0.2
    assert spring_moment_theta(np.tan(th0), K, th0) == pytest.approx(0.0, abs=1e-15)
    assert spring_dU_dalpha(np.tan(th0), K, th0) == pytest.approx(0.0, abs=1e-15)


# 8. Exact spring generalized force.
def test_exact_spring_generalized_force() -> None:
    a = np.array([-3.0, -1.0, -0.1, 0.0, 0.2, 1.0, 2.5])
    np.testing.assert_allclose(spring_dU_dalpha(a, K), K * np.arctan(a) / (1.0 + a * a), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        spring_d2U_dalpha2(a, K), K * (1.0 - 2.0 * a * np.arctan(a)) / (1.0 + a * a) ** 2, atol=1e-14
    )


# 9. arctan(alpha) is never replaced by alpha.
def test_no_small_angle_substitution() -> None:
    assert spring_dU_dalpha(1.0, K) == pytest.approx(K * np.pi / 8.0)
    assert abs(spring_dU_dalpha(1.0, K) - K * 1.0 / 2.0) > 2.0
    q1 = -5.0
    theta_star = np.deg2rad(50.0)
    a_star = np.tan(theta_star)
    q0 = K * theta_star / (1 + a_star**2) - q1 * a_star
    sol = solve_toe_equilibrium(q0, q1, ToeSpringConfig())
    assert sol.n_roots == 1
    assert sol.roots[0].theta_rad == pytest.approx(theta_star, abs=1e-10)
    # A linearized kα spring would put the root elsewhere.
    a_lin = q0 / (K - q1)
    assert abs(np.arctan(a_lin) - theta_star) > np.deg2rad(5.0)


# 10. d(alpha) is affine and matches the 2x2 force-controlled solve.
def test_force_controlled_translations_affine(lookup) -> None:
    Fx, Fy, phi_deg = 300.0, -4.0e3, 4.0
    varphi = np.deg2rad(phi_deg) - lookup.phi_ref
    r_x, r_y = rotation_coefficients(varphi)
    F_local = rotate_force_to_local(Fx, Fy, varphi)
    q_basis = lookup_q_alpha_basis(lookup)
    for row in _representative_rows(lookup).values():
        m = toe_affine_model(lookup.scalar_lookup[row], q_basis[row], F_local, r_x, r_y)
        for theta_deg in (-20.0, 0.0, 7.5, 30.0):
            ev = evaluate_candidates(lookup, Fx, Fy, phi_deg, theta_deg)
            d = m.displacement(np.tan(np.deg2rad(theta_deg)))
            scale = max(abs(ev.d_ax[row]), abs(ev.d_ay[row]), 1e-12)
            assert abs(d[0] - ev.d_ax[row]) / scale < 1e-9
            assert abs(d[1] - ev.d_ay[row]) / scale < 1e-9


# 11. q0/q1 vs direct superposition (fixed frame) and the rearfoot model vs superposition.
def test_q0_q1_match_direct_superposition(lookup) -> None:
    Fx, Fy, phi_deg = -150.0, -6.0e3, -3.0
    varphi = np.deg2rad(phi_deg) - lookup.phi_ref
    r_x, r_y = rotation_coefficients(varphi)
    F_local = rotate_force_to_local(Fx, Fy, varphi)
    q_basis = lookup_q_alpha_basis(lookup)
    psi = _psi(lookup)
    for row in _representative_rows(lookup).values():
        m = toe_affine_model(lookup.scalar_lookup[row], q_basis[row], F_local, r_x, r_y)
        for alpha in (-0.4, 0.0, 0.25):
            ev = evaluate_candidates(lookup, Fx, Fy, phi_deg, np.rad2deg(np.arctan(alpha)))
            q_direct = float(q_alpha_shoe_on_foot(ev.top_force_y[row], psi))
            assert m.generalized_force(alpha) == pytest.approx(q_direct, rel=1e-9, abs=1e-9 * abs(m.q1))


def test_rearfoot_model_matches_superposition_at_rotated_chord(lookup) -> None:
    Fx, Fy, phi = -150.0, -6.0e3, np.deg2rad(-3.0)
    width = 0.1
    psi = _psi(lookup)
    for row in _representative_rows(lookup).values():
        m = _model(lookup, row, Fx, Fy, phi, width)
        for alpha in (-0.4, 0.0, 0.25, 0.9):
            chord_phi_deg = np.rad2deg(float(m.varphi(alpha)) + lookup.phi_ref)
            ev = evaluate_candidates(lookup, Fx, Fy, chord_phi_deg, np.rad2deg(np.arctan(alpha)))
            q_direct = width * float(q_alpha_shoe_on_foot(ev.top_force_y[row], psi))
            assert float(m.generalized_force(alpha)) == pytest.approx(q_direct, rel=1e-9, abs=1e-9)
            g = m.gamma(alpha)
            assert g[1] == pytest.approx(ev.d_ax[row], rel=1e-9, abs=1e-15)
            assert g[2] == pytest.approx(ev.d_ay[row], rel=1e-9, abs=1e-15)


def test_rearfoot_model_derivative_and_work(lookup) -> None:
    row = _representative_rows(lookup)["toe"]
    m = _model(lookup, row, 300.0, -5.0e3, 0.2, 0.1)
    h = 1e-6
    for alpha in (-0.8, -0.1, 0.0, 0.3, 1.5):
        num = (m.generalized_force(alpha + h) - m.generalized_force(alpha - h)) / (2 * h)
        assert float(m.generalized_force_derivative(alpha)) == pytest.approx(float(num), rel=1e-6, abs=1e-8)
        numw = (m.work(alpha + h) - m.work(alpha - h)) / (2 * h)
        assert float(numw) == pytest.approx(float(m.generalized_force(alpha)), rel=1e-7, abs=1e-8)


# The measured pitch is the heel -> MTP line: at every solved root the deformed
# rearfoot line of the model lies exactly along the measured angle.
def test_solved_configuration_keeps_measured_rearfoot_angle(lookup) -> None:
    phi = 0.12
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=500.0, Fy_star=-6.0e3, Mz_meas=0.0, phi_rad=phi, config=ToeSpringConfig(), width_m=0.1
    )
    geo = lookup_rearfoot_geometry(lookup)
    x = np.array([0.0, geo.L - geo.toe_length])
    phi1 = shape_mode_phi1(x, lookup.L, TOE_LENGTH_SOFT, KAPPA)
    y_top = np.interp(x, lookup.x_top, lookup.y_top) if lookup.y_top is not None else np.zeros(2)
    n = 0
    for c in cands:
        if not c.toe.converged:
            continue
        v = y_top + c.alpha * phi1
        rearfoot_local = np.arctan2(v[1] - v[0], x[1] - x[0])
        assert c.toe.varphi_rad + rearfoot_local == pytest.approx(phi, abs=1e-12)
        n += 1
    assert n > 0


@pytest.fixture(scope="module")
def sharp_lookup(fem):
    """Softplus transition narrower than the node spacing: psi ~ max(0, x - (L - toe_length)) at nodes."""
    return generate_contact_lookup(get_compliance_block_matrix(fem), toe_length=TOE_LENGTH_SOFT, kappa=400.0)


# Sign convention: a negative foot-on-shoe toe moment about the MTP point (load
# under the toes) gives a positive toe angle, and vice versa.
def test_negative_toe_moment_gives_positive_theta(sharp_lookup) -> None:
    lk = sharp_lookup
    cfg = ToeSpringConfig()
    phi = lookup_rearfoot_geometry(lk).reference_angle
    psi = _psi(lk)
    ramp = np.maximum(0.0, np.asarray(lk.x_top) - (lk.L - TOE_LENGTH_SOFT))
    dev = float(np.max(np.abs(psi - ramp)))
    assert dev < 0.01 * lk.L
    checked = {+1: 0, -1: 0}
    for Fx, Fy in ((0.0, -6.0e3), (800.0, -4.0e3), (-500.0, -8.0e3)):
        cands = solve_passive_toe_candidates(
            lk, Fx_star=Fx, Fy_star=Fy, Mz_meas=0.0, phi_rad=phi, config=cfg, width_m=0.1
        )
        for c in cands:
            if not (c.toe.converged and c.toe.toe_equilibrium_stable):
                continue
            g0 = _model(lk, c.row, Fx, Fy, phi, 0.1).gamma(0.0)
            m_toe0 = float(contract_basis(lk.scalar_lookup[c.row][:, SCALAR_TOE], g0))
            f_ty0 = contract_basis(lk.record_fields([c.row])["top_force_y"][0], g0)
            bound = dev * float(np.sum(np.abs(f_ty0)))
            assert abs(c.toe.q0_Nm / 0.1 + m_toe0) <= bound + 1e-9
            if abs(m_toe0) <= bound:
                continue
            assert np.sign(c.theta_deg) == -np.sign(m_toe0), (c.row, m_toe0, c.theta_deg)
            assert np.sign(c.toe.q0_Nm) == -np.sign(m_toe0)
            checked[int(-np.sign(m_toe0))] += 1
    assert checked[+1] > 0


def _synthetic(theta_deg: float, q1: float, k: float = K) -> tuple[float, float, float]:
    th = np.deg2rad(theta_deg)
    a = np.tan(th)
    return float(spring_dU_dalpha(a, k) - q1 * a), q1, th


# 12-13. Known roots of both signs.
@pytest.mark.parametrize("theta_deg", [22.0, -35.0, 0.0])
def test_synthetic_root_recovered_with_sign(theta_deg: float) -> None:
    q0, q1, th = _synthetic(theta_deg, -40.0)
    sol = solve_toe_equilibrium(q0, q1, ToeSpringConfig())
    assert sol.status == "ok" and sol.n_roots == 1
    r = sol.roots[0]
    assert r.theta_rad == pytest.approx(th, abs=1e-11)
    assert np.sign(round(r.theta_deg, 9)) == np.sign(theta_deg)
    assert r.stable and r.converged


# 14-15. Multiple roots and stability from potential curvature.
def test_multiple_roots_and_stability() -> None:
    q1 = 5.0  # negative-stiffness shoe response: three roots
    sol = solve_toe_equilibrium(0.0, q1, ToeSpringConfig())
    assert sol.status == "multiple_roots"
    assert sol.n_roots == 3
    mid = min(sol.roots, key=lambda r: abs(r.theta_rad))
    assert mid.theta_rad == pytest.approx(0.0, abs=1e-12)
    assert mid.stable
    for r in sol.roots:
        h = 1e-5
        num = (
            total_potential(r.alpha + h, 0.0, q1, K) - 2 * total_potential(r.alpha, 0.0, q1, K)
            + total_potential(r.alpha - h, 0.0, q1, K)
        ) / h**2
        assert r.tangent == pytest.approx(num, rel=1e-4, abs=1e-4)
        assert r.stable == (r.tangent > 0.0)
        if r is not mid:
            assert not r.stable


# 16. Root outside the bounds is rejected.
def test_root_outside_bounds_rejected() -> None:
    q0, q1, _ = _synthetic(50.0, -40.0)
    cfg = ToeSpringConfig(toe_angle_min_deg=-10.0, toe_angle_max_deg=40.0)
    sol = solve_toe_equilibrium(q0, q1, cfg)
    assert sol.n_roots == 0
    assert sol.status == "no_root_in_bounds"
    assert sol.diagnostic is not None and not sol.diagnostic.converged


# 17. No-root candidate gets an explicit failure status.
def test_no_root_candidate_status(lookup) -> None:
    cfg = ToeSpringConfig(toe_angle_min_deg=-0.5, toe_angle_max_deg=0.5)
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.0, Fy_star=-2.0e5, Mz_meas=0.0, phi_rad=0.15, config=cfg, width_m=1.0
    )
    failed = [c for c in cands if "no_root_in_bounds" in c.status]
    assert failed
    for c in failed:
        assert not c.admissible
        assert not c.toe.converged
        assert c.toe.toe_solve_status.startswith("no_root_in_bounds")
        assert "diagnostic_min_residual" in c.status


# 18. Zero load gives theta = 0 and d = 0.
def test_zero_load_returns_zero(lookup) -> None:
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.0, Fy_star=0.0, Mz_meas=0.0, phi_rad=lookup.phi_ref,
        config=ToeSpringConfig(), width_m=0.1,
    )
    ok = [c for c in cands if c.toe.converged]
    assert ok
    for c in ok:
        assert c.theta_deg == pytest.approx(0.0, abs=1e-10)
        assert c.d_ax == pytest.approx(0.0, abs=1e-14)
        assert c.d_ay == pytest.approx(0.0, abs=1e-14)
        assert c.toe.toe_spring_energy_J == pytest.approx(0.0, abs=1e-20)


# 19. Low load does not give a spurious large angle and keeps the relaxed root.
def test_low_load_no_spurious_angle(lookup) -> None:
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.1, Fy_star=-1.0, Mz_meas=0.0, phi_rad=lookup.phi_ref,
        config=ToeSpringConfig(), width_m=1.0, low_load=True,
    )
    for c in cands:
        if c.toe.converged:
            assert abs(c.theta_deg) < 0.1
            assert c.toe.low_load and "low_load" in c.status
    # Synthetic three-root case: low-load keeps only the branch through theta0.
    q0, q1 = 1e-6, 5.0
    sol = solve_toe_equilibrium(q0, q1, ToeSpringConfig())
    from compliance_fem.contact.toe_spring import relaxed_root_index

    assert sol.n_roots == 3
    assert abs(sol.roots[relaxed_root_index(sol.roots)].theta_deg) < 1e-3


# 20. Continuity under small force perturbations.
def test_selected_root_continuous_under_perturbation(lookup) -> None:
    t = np.linspace(0.0, 0.2, 41)
    Fy = -8.0e3 * np.sin(np.pi * t / 0.2) ** 2 - 1.0e3
    syn = {"times": t, "Fx": 0.03 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": np.deg2rad(2.0) * np.ones_like(t)}
    r = replay_stance(lookup, synthetic=syn, shoe_width_m=1.0)
    th = r.theta_deg
    assert np.all(np.isfinite(th))
    jumps = np.abs(np.diff(th))
    dF = np.abs(np.diff(Fy)) / np.max(np.abs(Fy))
    assert np.max(jumps) < 2.0
    assert np.all(jumps <= 50.0 * dF + 0.05)


# 21. Each interval keeps its own root (no averaging across intervals).
def test_roots_retained_per_topology(lookup) -> None:
    cfg = ToeSpringConfig()
    Fx, Fy, phi = 200.0, -5.0e3, 0.05
    rows = _representative_rows(lookup)
    thetas = {}
    for name, row in rows.items():
        assert lookup.contact_type(row).value == name
        _, sol = _row_root(lookup, row, Fx, Fy, phi, cfg=cfg)
        assert sol.n_roots >= 1, name
        thetas[name] = sol.roots[0].theta_deg
    assert len({round(v, 6) for v in thetas.values()}) == 3
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=Fx, Fy_star=Fy, Mz_meas=0.0, phi_rad=phi, config=cfg, width_m=1.0
    )
    assert cands
    for c in cands:
        assert _within_toe_tol(c, cfg)
        _, own = _row_root(lookup, c.row, Fx, Fy, phi, cfg=cfg)
        assert min(abs(r.theta_deg - c.theta_deg) for r in own.roots) < 1e-9


# 22. Both interval edge nodes stay on the ground after solving alpha.
def test_transition_node_in_contact_after_toe_solve(lookup) -> None:
    from compliance_fem.contact.corotation import contact_displacement, fixed_frame_normal_component

    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.0, Fy_star=-5.0e3, Mz_meas=0.0, phi_rad=0.08, config=ToeSpringConfig(), width_m=1.0
    )
    y_b = np.zeros_like(lookup.x_bottom) if lookup.y_bottom is None else np.asarray(lookup.y_bottom)
    n = 0
    for c in cands:
        if not c.toe.converged:
            continue
        edges = np.array([lookup.contact_start_index[c.row], lookup.contact_end_index[c.row]], dtype=int)
        assert np.all(lookup.contact_mask[c.row][edges])
        varphi = c.toe.varphi_rad
        x_a = float(lookup.anchor_reference_x[c.row])
        y_a = float(lookup.anchor_reference_y[c.row])
        u_c, v_c = contact_displacement(
            lookup.x_bottom[edges], x_a, c.d_ax, c.d_ay, varphi, y_contact=y_b[edges], y_anchor=y_a
        )
        gap = fixed_frame_normal_component(
            (lookup.x_bottom[edges] - x_a) + u_c - c.d_ax, (y_b[edges] - y_a) + v_c - c.d_ay, varphi
        )
        np.testing.assert_allclose(gap, 0.0, atol=1e-12 * lookup.L)
        n += 1
    assert n > 0


# 23-24. Force reconstruction and toe residual within scaled tolerance.
def test_force_and_toe_residuals_within_tolerance(lookup) -> None:
    cfg = ToeSpringConfig()
    Fx, Fy = 400.0, -7.0e3
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=Fx, Fy_star=Fy, Mz_meas=0.0, phi_rad=-0.04, config=cfg, width_m=1.0
    )
    F_scale = np.hypot(Fx, Fy)
    for c in cands:
        if not c.toe.converged:
            continue
        assert c.force_residual <= 1e-9 * F_scale
        assert _within_toe_tol(c, cfg)
        assert c.toe.toe_equilibrium_relative_residual < 1e-9


# 25. Unilateral checks use the solved alpha.
def test_unilateral_checks_after_alpha(lookup) -> None:
    from compliance_fem.contact.force_control import Tolerances

    phi = 0.1
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.0, Fy_star=-4.0e3, Mz_meas=0.0, phi_rad=phi, config=ToeSpringConfig(), width_m=1.0
    )
    n_diff = 0
    for c in cands:
        if not c.toe.converged:
            continue
        varphi = c.toe.varphi_rad
        assert c.gamma[0] == c.alpha
        uc = unilateral_check(lookup, c.row, c.gamma, varphi, Tolerances())
        assert uc.max_free_penetration == pytest.approx(c.max_free_penetration, abs=1e-15)
        g0 = c.gamma.copy()
        g0[0] = 0.0
        uc0 = unilateral_check(lookup, c.row, g0, varphi, Tolerances())
        assert uc.min_contact_reaction == pytest.approx(c.min_contact_reaction, rel=1e-9, abs=1e-9)
        n_diff += int(
            abs(uc0.max_free_penetration - uc.max_free_penetration) > 1e-12
            or abs(uc0.min_contact_reaction - uc.min_contact_reaction) > 1e-9 * abs(uc.min_contact_reaction)
        )
    assert n_diff > 0


# 26. Runtime superposition matches a direct FEM reconstruction (heel/full/toe).
def test_runtime_superposition_matches_direct_fem(fem, lookup) -> None:
    cfg = ToeSpringConfig()
    width = 0.2
    Fx, Fy, phi = 250.0, -6.0e3, 0.03
    psi = _psi(lookup)
    n_t = len(lookup.x_top)
    for name, row in _representative_rows(lookup).items():
        m, sol = _row_root(lookup, row, Fx, Fy, phi, width, cfg)
        root = next(r for r in sol.roots if r.converged)
        gamma = m.gamma(root.alpha)
        F_local = rotate_force_to_local(Fx, Fy, float(m.varphi(root.alpha)))
        direct = solve_direct_fem_contact(fem, lookup.interval(row), 0, TOE_LENGTH_SOFT, KAPPA, gamma=gamma)
        f_t = np.asarray(direct.f_t)
        f_tx, f_ty = f_t[:n_t], f_t[n_t:]
        np.testing.assert_allclose(f_ty, contract_basis(lookup.top_force_y_basis[row], gamma), rtol=1e-7, atol=1e-7 * np.max(np.abs(f_ty)))
        assert np.array([f_tx.sum(), f_ty.sum()]) == pytest.approx(F_local, rel=1e-7, abs=1e-6 * abs(Fy))
        Q_fem = width * float(q_alpha_shoe_on_foot(f_ty, psi))
        assert Q_fem == pytest.approx(float(m.generalized_force(root.alpha)), rel=1e-7, abs=1e-9)
        assert Q_fem == pytest.approx(float(spring_dU_dalpha(root.alpha, K)), rel=1e-6, abs=1e-8), name


# 27-28. Spring energy and its derivative.
def test_spring_energy_and_numerical_derivative() -> None:
    for th in (-0.7, -0.1, 0.0, 0.4, 1.1):
        a = np.tan(th)
        assert spring_energy(a, K) == pytest.approx(0.5 * K * th**2, abs=1e-15)
        h = 1e-6
        num = (spring_energy(a + h, K) - spring_energy(a - h, K)) / (2 * h)
        assert num == pytest.approx(spring_dU_dalpha(a, K), rel=1e-7, abs=1e-9)
        numP = (total_potential(a + h, 1.3, -7.0, K) - total_potential(a - h, 1.3, -7.0, K)) / (2 * h)
        assert numP == pytest.approx(-toe_residual(a, 1.3, -7.0, K), rel=1e-7, abs=1e-9)
        assert q_theta_from_q_alpha(2.0, a) == pytest.approx(2.0 / np.cos(th) ** 2)
    assert potential_curvature(0.0, -7.0, K) == pytest.approx(K + 7.0)


# 29-31. Defaults.
def test_defaults() -> None:
    cfg = ToeSpringConfig()
    assert cfg.toe_stiffness_Nm_per_rad == 25.0
    assert cfg.toe_neutral_angle_rad == 0.0
    assert cfg.toe_damping_Nms_per_rad == 0.0
    assert cfg.toe_model == TOE_MODEL_PASSIVE_SPRING == DEFAULT_TOE_MODEL
    assert resolve_toe_config().toe_model == TOE_MODEL_PASSIVE_SPRING
    with pytest.raises(ValueError):
        ToeSpringConfig(toe_stiffness_Nm_per_rad=-1.0)
    with pytest.raises(ValueError):
        ToeSpringConfig(toe_model="prescribed_legacy")


def test_replay_default_is_passive_and_damping_ignored(lookup) -> None:
    t = np.linspace(0.0, 0.1, 6)
    syn = {"times": t, "Fx": np.zeros(6), "Fy": -3.0e3 * np.ones(6), "Mz": np.zeros(6), "phi": np.zeros(6)}
    r = replay_stance(lookup, synthetic=syn, shoe_width_m=1.0)
    assert r.toe_model == TOE_MODEL_PASSIVE_SPRING
    assert r.model_info["toe_config"]["toe_stiffness_Nm_per_rad"] == 25.0
    r2 = replay_stance(
        lookup, synthetic=syn, shoe_width_m=1.0, toe_config=ToeSpringConfig(toe_damping_Nms_per_rad=3.0)
    )
    np.testing.assert_array_equal(r.theta_deg, r2.theta_deg)
    assert any("does not enter" in w for w in r2.warnings)


# 32. Viscoelastic rejection / explicit approximation.
def test_viscoelastic_rejected_for_passive_spring(lookup) -> None:
    t = np.linspace(0.0, 0.1, 6)
    syn = {"times": t, "Fx": np.zeros(6), "Fy": -3.0e3 * np.ones(6), "Mz": np.zeros(6), "phi": np.zeros(6)}
    sls = {"g_inf": 0.5, "tau_r": 0.05}
    with pytest.raises(ValueError, match="visco_model='elastic'"):
        replay_stance(lookup, synthetic=syn, shoe_width_m=1.0, visco_model="sls", visco_config=sls)
    r = replay_stance(
        lookup, synthetic=syn, shoe_width_m=1.0, visco_model="sls", visco_config=sls,
        toe_model=TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    )
    assert r.toe_model == TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT
    assert any("approximation" in w for w in r.warnings)


# 33. Serialization preserves the generalized-force basis.
def test_serialization_preserves_q_basis(lookup, tmp_path: Path) -> None:
    save_contact_lookup(lookup, tmp_path)
    loaded = load_contact_lookup(tmp_path)
    assert loaded.schema_version == LOOKUP_SCHEMA_VERSION == 11
    assert loaded.toe_generalized_force_source == "stored"
    np.testing.assert_array_equal(loaded.Q_alpha_shoe_on_foot_basis, lookup.Q_alpha_shoe_on_foot_basis)
    data = dict(np.load(tmp_path / "contact_lookup.npz", allow_pickle=True))
    data["Q_alpha_shoe_on_foot_basis"] = -data["Q_alpha_shoe_on_foot_basis"]
    np.savez_compressed(tmp_path / "tampered.npz", **data)
    with pytest.raises(ValueError, match="disagrees"):
        load_contact_lookup(tmp_path / "tampered.npz")


# 34. Old schemas are rejected: heel/toe/full records cannot be migrated to intervals.
def test_old_schema_rejected(lookup, tmp_path: Path) -> None:
    save_contact_lookup(lookup, tmp_path)
    data = dict(np.load(tmp_path / "contact_lookup.npz", allow_pickle=True))
    for version in (5, 6, 7, 8, 9):
        old = dict(data)
        old["schema_version"] = np.asarray(version)
        np.savez_compressed(tmp_path / f"v{version}.npz", **old)
        with pytest.raises(ValueError, match="Regenerate"):
            load_contact_lookup(tmp_path / f"v{version}.npz")
    assert "schema_version=11" in REGENERATE_LOOKUP_MESSAGE


# 35. GUI and exported results show the same solved theta.
def test_gui_and_export_show_same_theta(lookup, tmp_path: Path) -> None:
    from compliance_fem.gui.gait_page import toe_equilibrium_figure, toe_stance_figure
    from compliance_fem.gui.gait_plots import selection_proxy_for_frame

    t = np.linspace(0.0, 0.1, 11)
    Fy = -5.0e3 * np.sin(np.pi * t / 0.1) ** 2
    syn = {"times": t, "Fx": 0.05 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": 0.02 * np.ones_like(t)}
    r = replay_stance(lookup, synthetic=syn, shoe_width_m=1.0)
    paths = save_replay_result(r, tmp_path)
    npz = np.load(paths["npz"], allow_pickle=True)
    np.testing.assert_array_equal(npz["theta_deg"], r.theta_deg)
    np.testing.assert_array_equal(npz["theta"], r.theta_deg)
    np.testing.assert_allclose(npz["theta_rad"], np.deg2rad(r.theta_deg))
    np.testing.assert_allclose(np.tan(npz["theta_rad"]), npz["alpha"], rtol=1e-12, atol=1e-15)
    header = paths["csv"].read_text().splitlines()[0].split(",")
    for key in ("theta_deg", "Q_alpha_shoe_on_foot", "toe_equilibrium_residual_Nm", "toe_solve_status", "toe_model"):
        assert key in header
    i = int(np.argmax(np.abs(r.theta_deg)))
    fig = toe_equilibrium_figure(r, i, ToeSpringConfig())
    vline = [s for s in fig.layout.shapes if s.type == "line"][0]
    assert vline.x0 == pytest.approx(float(r.theta_deg[i]))
    roots_trace = [d for d in fig.data if d.name == "all roots"][0]
    assert np.min(np.abs(np.asarray(roots_trace.x) - r.theta_deg[i])) < 1e-9
    stance = toe_stance_figure(r)
    np.testing.assert_array_equal(np.asarray(stance.data[0].y), r.theta_deg)
    proxy = selection_proxy_for_frame(lookup, r, i)
    assert proxy.alpha == pytest.approx(np.tan(np.deg2rad(r.theta_deg[i])))


def test_instant_best_prefers_converged_stable_and_continuity(lookup) -> None:
    cands = solve_passive_toe_candidates(
        lookup, Fx_star=0.0, Fy_star=-3.0e3, Mz_meas=0.0, phi_rad=0.0, config=ToeSpringConfig(), width_m=1.0
    )
    best = pick_instant_best_passive(cands, previous_theta_deg=None)
    assert best is not None and best.toe.converged and best.toe.toe_equilibrium_stable
    bad = dataclasses.replace(best, admissible=False)
    assert pick_instant_best_passive([bad, best]) is best
