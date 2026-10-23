"""Passive toe-joint spring: spring law, rearfoot geometry, replay defaults and GUI/export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.contact.basis import shape_mode_toe_bend
from compliance_fem.contact.contact_model import ContactModel, build_contact_model
from compliance_fem.contact.continuous_toe import solve_toe_contact
from compliance_fem.contact.toe_spring import (
    DEFAULT_TOE_MODEL,
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    ToeSpringConfig,
    q_theta_from_q_alpha,
    rearfoot_geometry,
    rearfoot_toe_mode,
    spring_d2U_dalpha2,
    spring_dU_dalpha,
    spring_energy,
    spring_moment_theta,
)
from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.gait.replay import replay_stance, resolve_toe_config, save_replay_result

from conftest import small_config

K = 25.0
TOE_LENGTH_SOFT = 0.4
TOE_BEND_SCALE_FACTOR = 30.0


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
def model(fem) -> ContactModel:
    return build_contact_model(fem, toe_length=TOE_LENGTH_SOFT, kappa=TOE_BEND_SCALE_FACTOR)


def test_rearfoot_mode_vanishes_at_heel_and_mtp(model) -> None:
    op = model.contact_operator
    geo = rearfoot_geometry(op.x_top, op.y_top, model.L, TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR)
    assert geo == op.rearfoot
    x = np.array([0.0, geo.L - geo.toe_length, model.L])
    phi1 = shape_mode_toe_bend(x, model.L, TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR)
    psi = rearfoot_toe_mode(phi1, x, geo.L, geo.toe_length, geo.y_mtp_shape_mode)
    assert psi[0] == pytest.approx(0.0, abs=1e-15)
    assert psi[1] == pytest.approx(0.0, abs=1e-15)
    # Toe tip rises relative to the rearfoot line by ~ toe_length per unit alpha.
    assert psi[2] > 0.0
    assert psi[2] == pytest.approx(geo.toe_length, rel=0.1)
    psi_top = rearfoot_toe_mode(
        shape_mode_toe_bend(op.x_top, model.L, TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR), op.x_top, geo.L, geo.toe_length, geo.y_mtp_shape_mode
    )
    np.testing.assert_allclose(op.toe_shape_mode_flat_rearfoot, psi_top, atol=1e-14)
    assert np.all(op.toe_shape_mode_flat_rearfoot[op.x_top >= geo.L - geo.toe_length] >= -1e-12)


# Units: phi1 is a length (scales with the sole), so Q = -psi^T f is force x length.
def test_shape_mode_scales_as_a_length() -> None:
    x = np.linspace(0.0, 1.0, 17)
    s = 3.7
    np.testing.assert_allclose(
        shape_mode_toe_bend(s * x, s * 1.0, s * TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR),
        s * shape_mode_toe_bend(x, 1.0, TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR),
        rtol=1e-12,
    )


def test_spring_moment_opposes_displacement_and_vanishes_at_neutral() -> None:
    for th in (0.3, -0.3):
        a = np.tan(th)
        assert np.sign(spring_moment_theta(a, K)) == -np.sign(th)
        assert np.sign(spring_dU_dalpha(a, K)) == np.sign(th)
    assert spring_moment_theta(0.0, K) == 0.0
    assert spring_dU_dalpha(0.0, K) == 0.0
    th0 = 0.2
    assert spring_moment_theta(np.tan(th0), K, th0) == pytest.approx(0.0, abs=1e-15)
    assert spring_dU_dalpha(np.tan(th0), K, th0) == pytest.approx(0.0, abs=1e-15)


def test_exact_spring_generalized_force() -> None:
    a = np.array([-3.0, -1.0, -0.1, 0.0, 0.2, 1.0, 2.5])
    np.testing.assert_allclose(spring_dU_dalpha(a, K), K * np.arctan(a) / (1.0 + a * a), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        spring_d2U_dalpha2(a, K), K * (1.0 - 2.0 * a * np.arctan(a)) / (1.0 + a * a) ** 2, atol=1e-14
    )


# arctan(alpha) is never replaced by alpha, in the spring law or in the solved frame.
def test_no_small_angle_substitution(model) -> None:
    assert spring_dU_dalpha(1.0, K) == pytest.approx(K * np.pi / 8.0)
    assert abs(spring_dU_dalpha(1.0, K) - K * 1.0 / 2.0) > 2.0
    cfg = ToeSpringConfig()
    phi = model.contact_operator.rearfoot.reference_angle
    fr = solve_toe_contact(model.contact_operator, np.array([500.0, -6.0e3]), phi, cfg, 0.1)
    assert fr.converged
    assert abs(fr.theta_rad) > 1e-3
    assert fr.alpha == pytest.approx(np.tan(fr.theta_rad), rel=1e-14)
    k = cfg.toe_stiffness_Nm_per_rad
    assert fr.toe_spring_moment_Nm == pytest.approx(-k * fr.theta_rad, rel=1e-12)
    # Generalized spring force on alpha is -dU/dalpha with the exact arctan law.
    assert fr.toe_spring_generalized_force_alpha == pytest.approx(
        -float(spring_dU_dalpha(fr.alpha, k)), rel=1e-12, abs=1e-15
    )
    assert fr.Q_alpha_shoe_on_foot - float(spring_dU_dalpha(fr.alpha, k)) == pytest.approx(
        fr.toe_equilibrium_residual_Nm, rel=1e-9, abs=1e-12
    )
    assert fr.Q_theta_shoe_on_foot == pytest.approx(
        float(q_theta_from_q_alpha(fr.Q_alpha_shoe_on_foot, fr.alpha)), rel=1e-12, abs=1e-15
    )


# The measured pitch is the heel -> MTP line: the solved deformed rearfoot line
# lies exactly along the prescribed angle.
def test_solved_configuration_keeps_measured_rearfoot_angle(model) -> None:
    phi = 0.12
    op = model.contact_operator
    fr = solve_toe_contact(op, np.array([500.0, -6.0e3]), phi, ToeSpringConfig(), 0.1)
    assert fr.converged
    geo = op.rearfoot
    x = np.array([0.0, geo.L - geo.toe_length])
    phi1 = shape_mode_toe_bend(x, model.L, TOE_LENGTH_SOFT, TOE_BEND_SCALE_FACTOR)
    v = np.interp(x, op.x_top, op.y_top) + fr.alpha * phi1
    rearfoot_local = np.arctan2(v[1] - v[0], x[1] - x[0])
    assert fr.varphi + rearfoot_local == pytest.approx(phi, abs=1e-12)


def test_spring_energy_and_numerical_derivative() -> None:
    for th in (-0.7, -0.1, 0.0, 0.4, 1.1):
        a = np.tan(th)
        assert spring_energy(a, K) == pytest.approx(0.5 * K * th**2, abs=1e-15)
        h = 1e-6
        num = (spring_energy(a + h, K) - spring_energy(a - h, K)) / (2 * h)
        assert num == pytest.approx(spring_dU_dalpha(a, K), rel=1e-7, abs=1e-9)
        num2 = (spring_dU_dalpha(a + h, K) - spring_dU_dalpha(a - h, K)) / (2 * h)
        assert num2 == pytest.approx(spring_d2U_dalpha2(a, K), rel=1e-6, abs=1e-8)
        assert q_theta_from_q_alpha(2.0, a) == pytest.approx(2.0 / np.cos(th) ** 2)


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


def test_replay_default_is_passive_and_damping_ignored(model) -> None:
    t = np.linspace(0.0, 0.1, 6)
    syn = {"times": t, "Fx": np.zeros(6), "Fy": -3.0e3 * np.ones(6), "Mz": np.zeros(6), "phi": np.zeros(6)}
    r = replay_stance(model, synthetic=syn, shoe_width_m=1.0)
    assert r.toe_model == TOE_MODEL_PASSIVE_SPRING
    assert r.model_info["toe_config"]["toe_stiffness_Nm_per_rad"] == 25.0
    r2 = replay_stance(
        model, synthetic=syn, shoe_width_m=1.0, toe_config=ToeSpringConfig(toe_damping_Nms_per_rad=3.0)
    )
    np.testing.assert_array_equal(r.theta_deg, r2.theta_deg)
    assert any("does not enter" in w for w in r2.warnings)


def test_viscoelastic_rejected_for_passive_spring(model) -> None:
    t = np.linspace(0.0, 0.1, 6)
    syn = {"times": t, "Fx": np.zeros(6), "Fy": -3.0e3 * np.ones(6), "Mz": np.zeros(6), "phi": np.zeros(6)}
    sls = {"g_inf": 0.5, "tau_r": 0.05}
    with pytest.raises(ValueError, match="visco_model='elastic'"):
        replay_stance(model, synthetic=syn, shoe_width_m=1.0, visco_model="sls", visco_config=sls)
    r = replay_stance(
        model, synthetic=syn, shoe_width_m=1.0, visco_model="sls", visco_config=sls,
        toe_model=TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    )
    assert r.toe_model == TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT
    assert any("approximation" in w for w in r.warnings)


def test_theta_continuous_under_smooth_load(model) -> None:
    t = np.linspace(0.0, 0.2, 41)
    Fy = -8.0e3 * np.sin(np.pi * t / 0.2) ** 2 - 1.0e3
    syn = {"times": t, "Fx": 0.03 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": np.deg2rad(2.0) * np.ones_like(t)}
    r = replay_stance(model, synthetic=syn, shoe_width_m=1.0)
    th = r.theta_deg
    assert np.all(np.isfinite(th))
    jumps = np.abs(np.diff(th))
    dF = np.abs(np.diff(Fy)) / np.max(np.abs(Fy))
    assert np.max(jumps) < 2.0
    assert np.all(jumps <= 50.0 * dF + 0.05)


# GUI and exported results show the same solved theta.
def test_gui_and_export_show_same_theta(model, tmp_path: Path) -> None:
    from compliance_fem.gui.gait_page import toe_equilibrium_figure, toe_stance_figure
    from compliance_fem.gui.gait_plots import selection_proxy_for_frame

    t = np.linspace(0.0, 0.1, 11)
    Fy = -5.0e3 * np.sin(np.pi * t / 0.1) ** 2
    syn = {"times": t, "Fx": 0.05 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": 0.02 * np.ones_like(t)}
    r = replay_stance(model, synthetic=syn, shoe_width_m=1.0)
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
    fig = toe_equilibrium_figure(model, r, i, ToeSpringConfig())
    vline = [s for s in fig.layout.shapes if s.type == "line"][0]  # pyright: ignore[reportAttributeAccessIssue]
    assert vline.x0 == pytest.approx(float(r.theta_deg[i]))
    # plotly leaves Figure.data untyped.
    roots_trace = [d for d in fig.data if d.name == "all roots"][0]  # pyright: ignore[reportAttributeAccessIssue]
    assert np.min(np.abs(np.asarray(roots_trace.x) - r.theta_deg[i])) < 1e-9  # pyright: ignore[reportAttributeAccessIssue]
    stance = toe_stance_figure(r)
    np.testing.assert_array_equal(np.asarray(stance.data[0].y), r.theta_deg)  # pyright: ignore[reportAttributeAccessIssue]
    proxy = selection_proxy_for_frame(model, r, i)
    assert proxy.alpha == pytest.approx(np.tan(np.deg2rad(r.theta_deg[i])))
