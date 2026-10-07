"""Unit tests for gait replay (synthetic Wang-like fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.contact_basis import COL_ALPHA, COL_BX
from compliance_fem.contact_lookup import SCALAR_FX, SCALAR_FY, SCALAR_MZ
from compliance_fem.contact_topology import ContactType
from compliance_fem.corotation import contract_basis
from compliance_fem.gait.phi_markers import estimate_phi, select_chord_markers
from compliance_fem.gait.replay import replay_stance, save_replay_result
from compliance_fem.gait.sagittal import (
    lab_grf_to_model_top_wrench,
    moment_about_origin,
    translate_moment,
)
from compliance_fem.gait.stance import detect_stances
from compliance_fem.gait.temporal import TemporalWeights, viterbi_select
from compliance_fem.gait.wang_io import (
    discover_trials,
    parse_wang_trial_name,
    read_opensim_mot,
    read_opensim_trc,
)
from compliance_fem.gait.wrench_control import (
    assemble_Kw,
    pick_instant_best,
    solve_wrench_control,
)
from compliance_fem.viscoelasticity import ElasticConfig, SLSConfig, create_material


def test_prescribed_toe_angle_relu_neg_phi():
    from compliance_fem.gait.toe_angle import prescribed_toe_angle_deg

    # φ = −10° → relu(−φ)=10°; at peak load factor → 10°
    phi = np.deg2rad([-10.0, 5.0, -20.0])
    fy = np.array([-1000.0, -500.0, 0.0])  # compressive top force
    fy_max = 1000.0
    theta = prescribed_toe_angle_deg(phi, fy, fy_max, theta_min_deg=0.0, theta_max_deg=45.0)
    assert theta[0] == pytest.approx(10.0)
    # Positive φ → relu(−φ)=0
    assert theta[1] == pytest.approx(0.0)
    # Zero load → load factor 0
    assert theta[2] == pytest.approx(0.0)
    # Mid load: ratio=0.5 → 1-(0.5)^4 = 1-0.0625 = 0.9375
    theta_mid = prescribed_toe_angle_deg(
        np.deg2rad(-10.0), -500.0, 1000.0, theta_min_deg=0.0, theta_max_deg=45.0
    )
    assert float(theta_mid) == pytest.approx(10.0 * (1.0 - 0.5**4))


def test_parse_wang_trial_name():
    s, c, t = parse_wang_trial_name("P01_pr1_01.trc")
    assert s == "P01" and c == "pr1" and t == "01"
    s, c, t = parse_wang_trial_name("P4 pr1 01.trc")
    assert s == "P4" and c == "pr1" and t == "01"
    s, c, t = parse_wang_trial_name("P4 pr1 01_force.mot")
    assert s == "P4" and c == "pr1" and t == "01"
    s, c, t = parse_wang_trial_name("P4 -130r2 01_force.mot")
    assert s == "P4" and c == "-130r2" and t == "01"
    with pytest.raises(ValueError):
        parse_wang_trial_name("not_a_trial")


def test_discover_trials_pairs_force_mot(tmp_path: Path):
    trc_dir = tmp_path / "trc"
    mot_dir = tmp_path / "MOT"
    trc_dir.mkdir()
    mot_dir.mkdir()
    (trc_dir / "P4 pr1 01.trc").write_text("dummy", encoding="utf-8")
    (mot_dir / "P4 pr1 01_force.mot").write_text("dummy", encoding="utf-8")
    trials = discover_trials(tmp_path)
    assert len(trials) == 1
    assert trials[0].stem == "P4 pr1 01"
    assert trials[0].trc_path is not None
    assert trials[0].mot_path is not None
    assert trials[0].is_preferred_r1


def test_unit_conversions_moment_and_cop(tmp_path: Path):
    # Minimal MOT with N·mm moments and mm COP.
    mot = tmp_path / "P01_pr1_01.mot"
    mot.write_text(
        "\n".join(
            [
                "ground_reaction",
                "version=1",
                "nRows=3",
                "nColumns=4",
                "inDegrees=no",
                "endheader",
                "time ground_force_vx ground_force_vy ground_force_px ground_force_mz",
                "0.0 0.0 0.0 0.0 0.0",
                "0.001 10.0 800.0 150.0 12000.0",
                "0.002 5.0 700.0 160.0 11000.0",
            ]
        ),
        encoding="utf-8",
    )
    data = read_opensim_mot(mot)
    # COP mm → m
    assert data["columns"]["ground_force_px"][1] == pytest.approx(0.15)
    # Moment N·mm → N·m
    assert data["columns"]["ground_force_mz"][1] == pytest.approx(12.0)


def test_trc_mm_to_metres(tmp_path: Path):
    trc = tmp_path / "P01_pr1_01.trc"
    trc.write_text(
        "\n".join(
            [
                "PathFileType\t4\t(X/Y/Z)\tP01_pr1_01.trc",
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
                "200\t200\t2\t2\tmm\t200\t1\t2",
                "Frame#\tTime\tRHEEL\t\t\tRTOE\t\t",
                "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
                "1\t0.0\t0.0\t0.0\t0.0\t290.0\t0.0\t0.0",
                "2\t0.005\t0.0\t0.0\t0.0\t290.0\t10.0\t0.0",
            ]
        ),
        encoding="utf-8",
    )
    data = read_opensim_trc(trc)
    assert data["markers"]["RTOE"][0, 0] == pytest.approx(0.29)
    assert data["markers"]["RTOE"][1, 1] == pytest.approx(0.01)


def test_discover_trials(tmp_path: Path):
    (tmp_path / "P02_pr1_03.trc").write_text("x", encoding="utf-8")
    (tmp_path / "P02_pr1_03.mot").write_text("x", encoding="utf-8")
    trials = discover_trials(tmp_path)
    assert len(trials) == 1
    assert trials[0].is_preferred_r1


def test_stance_hysteresis_20N():
    t = np.linspace(0, 1, 1001)
    fy = np.zeros_like(t)
    fy[200:500] = 800.0
    fy[199] = 15.0
    fy[500] = 10.0
    st = detect_stances(t, fy, threshold=20.0, hysteresis=5.0)
    assert len(st) == 1
    assert st[0].peak_fy == pytest.approx(800.0)


def test_force_sign_compression():
    t = np.array([0.0, 0.1])
    wrench, _ = lab_grf_to_model_top_wrench(
        times=t,
        fx_lab=np.array([0.0, 0.0]),
        fy_lab=np.array([0.0, 1000.0]),
        fz_lab=np.array([0.0, 0.0]),
        cop_x_lab=np.array([0.0, 0.1]),
        cop_y_lab=np.array([0.0, 0.0]),
    )
    assert wrench.Fy[1] == pytest.approx(-1000.0)


def test_wrench_translation_invariance():
    fx, fy = 10.0, -800.0
    mz0 = moment_about_origin(fx, fy, 0.12, 0.0, x_o=0.0, y_o=0.0)
    mz1 = translate_moment(mz0, fx, fy, from_xy=(0.0, 0.0), to_xy=(0.05, 0.0))
    mz1_direct = moment_about_origin(fx, fy, 0.12, 0.0, x_o=0.05, y_o=0.0)
    assert float(mz1) == pytest.approx(float(mz1_direct), rel=1e-12)


def test_phi_from_markers_and_unwrap():
    n = 50
    t = np.linspace(0, 1, n)
    ang = np.linspace(-0.2, 0.2, n)
    heel = np.zeros((n, 3))
    toe = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(n)]) * 0.3
    markers = {"RHEEL": heel, "RTOE": toe}
    series = estimate_phi(t, markers, foot="right")
    np.testing.assert_allclose(series.phi_rad, ang, atol=1e-9)
    # unwrap across pi
    ang2 = np.linspace(3.0, 3.4, n)
    toe2 = np.column_stack([np.cos(ang2), np.sin(ang2), np.zeros(n)])
    s2 = estimate_phi(t, {"RHEEL": heel, "RTOE": toe2}, foot="right")
    assert np.all(np.diff(s2.phi_rad) > -np.pi)


def test_wang_marker_aliases_for_phi():
    labels = [
        "RASIS",
        "RHeel",
        "RM5",
        "RM1",
        "RMid",
        "LHeel",
        "LM5",
        "LM1",
        "LMid",
    ]
    heel, fore = select_chord_markers(labels, foot="right")
    assert heel == "RHeel"
    assert fore == "MID(RM1+RM5)"
    n = 5
    t = np.linspace(0, 1, n)
    markers = {
        "RHeel": np.zeros((n, 3)),
        "RM1": np.tile([0.25, 0.0, 0.02], (n, 1)),
        "RM5": np.tile([0.25, 0.0, -0.02], (n, 1)),
        "RMid": np.tile([0.28, 0.0, 0.0], (n, 1)),
    }
    series = estimate_phi(t, markers, foot="right")
    assert series.heel_label == "RHeel"
    assert series.fore_label == "MID(RM1+RM5)"
    assert np.all(np.isfinite(series.phi_rad))


def _wrench_row(lookup) -> int:
    """A valid interval row whose 3x3 wrench map [dx, dy, alpha] -> [Fx, Fy, Mz] is well conditioned."""
    best, best_cond = -1, np.inf
    for row in lookup.valid_rows:
        cond = np.linalg.cond(assemble_Kw(lookup.scalar_lookup[int(row)]))
        if cond < best_cond:
            best, best_cond = int(row), cond
    assert best >= 0 and best_cond < 1e8
    return best


def _wrench_from_gamma(lookup, row: int, gamma: np.ndarray) -> np.ndarray:
    """Local-frame (Fx, Fy, Mz) of the affine state, closure column included once."""
    S = np.asarray(lookup.scalar_lookup[row])
    W = np.stack([S[:, SCALAR_FX], S[:, SCALAR_FY], S[:, SCALAR_MZ]])
    return contract_basis(W, gamma, mode_axis=1)


def _row_candidate(cands, row: int):
    mine = [c for c in cands if c.row == row]
    assert len(mine) == 1
    return mine[0]


def test_Kw_assembly_and_recovery(flat_lookup):
    lookup = flat_lookup
    row = _wrench_row(lookup)
    Kw = assemble_Kw(lookup.scalar_lookup[row])
    assert Kw.shape == (3, 3)
    S = lookup.scalar_lookup[row]
    np.testing.assert_array_equal(Kw[:, 0], S[COL_BX, [SCALAR_FX, SCALAR_FY, SCALAR_MZ]])
    np.testing.assert_array_equal(Kw[:, 2], S[COL_ALPHA, [SCALAR_FX, SCALAR_FY, SCALAR_MZ]])
    dx, dy, alpha = 1e-6, -2e-5, np.tan(np.deg2rad(8.0))
    w = _wrench_from_gamma(lookup, row, np.array([alpha, dx, dy, 0.0, 0.0]))
    cands = solve_wrench_control(
        lookup,
        Fx_star=float(w[0]),
        Fy_star=float(w[1]),
        Mz_star=float(w[2]),
        phi_rad=float(lookup.phi_ref),
        theta_min_deg=-20,
        theta_max_deg=40,
    )
    c = _row_candidate(cands, row)
    iv = lookup.interval(row)
    assert c.interval == (iv.start, iv.end)
    assert c.d_ax == pytest.approx(dx, rel=1e-7, abs=1e-12)
    assert c.d_ay == pytest.approx(dy, rel=1e-7)
    assert c.alpha == pytest.approx(alpha, rel=1e-7)
    assert c.theta_deg == pytest.approx(8.0, abs=1e-6)
    assert c.force_residual < 1e-8 * np.hypot(w[0], w[1])


def _row_wrench_at_theta(lookup, row: int, theta_deg: float) -> np.ndarray:
    return _wrench_from_gamma(lookup, row, np.array([np.tan(np.deg2rad(theta_deg)), 0.0, -2e-5, 0.0, 0.0]))


def test_theta_bounds_clamp_nonnegative(flat_lookup):
    lookup = flat_lookup
    row = _wrench_row(lookup)
    w = _row_wrench_at_theta(lookup, row, 30.0)
    cands = solve_wrench_control(
        lookup, Fx_star=float(w[0]), Fy_star=float(w[1]), Mz_star=float(w[2]),
        phi_rad=float(lookup.phi_ref), theta_min_deg=0.0, theta_max_deg=1.0,
    )
    assert cands
    assert all(0.0 <= c.theta_deg <= 1.0 + 1e-9 for c in cands if np.isfinite(c.theta_deg))
    c = _row_candidate(cands, row)
    assert "theta_clamped" in c.status
    assert c.theta_deg == pytest.approx(1.0)

    w = _row_wrench_at_theta(lookup, row, -20.0)
    cands_neg = solve_wrench_control(
        lookup, Fx_star=float(w[0]), Fy_star=float(w[1]), Mz_star=float(w[2]),
        phi_rad=float(lookup.phi_ref), theta_min_deg=0.0, theta_max_deg=45.0,
    )
    assert all(c.theta_deg >= -1e-9 for c in cands_neg if np.isfinite(c.theta_deg))
    c = _row_candidate(cands_neg, row)
    assert "theta_clamped" in c.status
    assert c.theta_deg == pytest.approx(0.0, abs=1e-12)


def test_theta_prior_used_when_free_alpha_oob(flat_lookup):
    """Force-phi prior replaces nearest-bound clamp when free alpha is out of range."""
    lookup = flat_lookup
    row = _wrench_row(lookup)
    prior = 12.0
    w = _row_wrench_at_theta(lookup, row, -20.0)
    cands = solve_wrench_control(
        lookup, Fx_star=float(w[0]), Fy_star=float(w[1]), Mz_star=float(w[2]),
        phi_rad=float(lookup.phi_ref), theta_min_deg=0.0, theta_max_deg=45.0, theta_prior_deg=prior,
    )
    c = _row_candidate(cands, row)
    assert "theta_prior_fallback" in c.status
    assert c.theta_deg == pytest.approx(prior, abs=1e-9)
    assert all(
        abs(c.theta_deg - prior) < 1e-9 for c in cands if np.isfinite(c.theta_deg) and "fallback" in c.status
    )


def test_viterbi_continuity_and_forefoot():
    from compliance_fem.gait.wrench_control import WrenchCandidate

    n_b = 13
    spans = {ContactType.HEEL: (0, 4), ContactType.FULL: (0, n_b - 1), ContactType.TOE: (8, n_b - 1)}

    def cand(ctype, theta, row=0, adm=True):
        i, j = spans[ctype]
        return WrenchCandidate(
            row=row,
            contact_type=ctype,
            contact_start_index=i,
            contact_end_index=j,
            contact_start_x=float(i),
            contact_end_x=float(j),
            anchor_x=0.5 * (i + j),
            d_ax=0.0,
            d_ay=0.0,
            alpha=np.tan(np.deg2rad(theta)),
            theta_deg=theta,
            Fx_pred=0.0,
            Fy_pred=-100.0,
            Mz_pred=0.0,
            force_residual=0.1 if adm else 5.0,
            moment_residual=0.1,
            cop_residual=0.0,
            cond=1.0,
            max_free_penetration=0.0,
            min_contact_reaction=1.0,
            admissible=adm,
            status="ok",
            gamma=np.zeros(5),
        )

    frames = [
        [cand(ContactType.TOE, 5.0), cand(ContactType.HEEL, 5.0, adm=False)],
        [cand(ContactType.TOE, 6.0), cand(ContactType.FULL, 20.0)],
        [cand(ContactType.TOE, 7.0), cand(ContactType.HEEL, 0.0)],
    ]
    path = viterbi_select(frames, weights=TemporalWeights(topology_jump=0.1))
    assert all(c.contact_type is ContactType.TOE for c in path)
    assert all(c.interval == spans[ContactType.TOE] for c in path)


def test_visco_three_components_chronological():
    m = create_material(SLSConfig(g_inf=0.5, tau_r=0.2, n_components=3))
    t = np.linspace(0, 0.5, 20)
    F = np.column_stack([np.sin(t), -800 * t, 10 * t])
    Fe = m.evaluate_history(t, F, reset=True)
    assert Fe.shape == (20, 3)
    # Chronological ≠ shuffled
    m2 = create_material(SLSConfig(g_inf=0.5, tau_r=0.2, n_components=3))
    order = np.arange(20)[::-1]
    # reverse time must raise
    with pytest.raises(ValueError):
        m2.evaluate_history(t[order], F[order], reset=True)


def test_elastic_identity_wrench():
    m = create_material(ElasticConfig(n_components=3))
    t = np.array([0.0, 0.1, 0.2])
    F = np.array([[0, 0, 0], [1, -2, 3], [1, -2, 3]], dtype=float)
    Fe = m.evaluate_history(t, F)
    np.testing.assert_allclose(Fe, F)


def test_shoe_width_scales_forces(flat_lookup):
    """Experimental totals / width -> plane-strain per-metre loads."""
    lookup = flat_lookup
    t = np.array([0.0, 0.05, 0.1])
    syn = {
        "times": t,
        "Fx": np.array([0.0, 10.0, 10.0]),
        "Fy": np.array([0.0, -100.0, -100.0]),
        "Mz": np.array([0.0, 2.0, 2.0]),
        "phi": np.zeros_like(t),
    }
    r1 = replay_stance(
        lookup, synthetic=syn, theta_mode="fit-cop", shoe_width_m=1.0, cop_fit_min_force=1.0
    )
    r01 = replay_stance(
        lookup, synthetic=syn, theta_mode="fit-cop", shoe_width_m=0.1, cop_fit_min_force=1.0
    )
    np.testing.assert_allclose(r01.elastic_Fx, r1.elastic_Fx / 0.1)
    np.testing.assert_allclose(r01.elastic_Fy, r1.elastic_Fy / 0.1)
    np.testing.assert_allclose(r01.elastic_Mz, r1.elastic_Mz / 0.1)
    assert r01.provenance["shoe_width_m"] == 0.1


def test_fit_cop_low_force_defaults_theta_to_zero(flat_lookup):
    """When |Fy| is below the COP-fit threshold, theta is fixed at 0 deg."""
    t = np.array([0.0, 0.05, 0.1])
    syn = {
        "times": t,
        "Fx": np.zeros(3),
        "Fy": np.array([0.0, -1.0, -1.0]),
        "Mz": np.array([0.0, 5.0, 5.0]),
        "phi": np.zeros(3),
    }
    result = replay_stance(
        flat_lookup,
        synthetic=syn,
        theta_mode="fit-cop",
        shoe_width_m=1.0,
        cop_fit_min_force=50.0,
    )
    np.testing.assert_allclose(result.theta_deg, 0.0, atol=1e-12)
    assert result.provenance["toe_angle_method"] == "fit-cop-3x3-wrench"
    assert all("low_force_fallback" in s for s in result.selection_status)


def test_end_to_end_synthetic_replay(flat_lookup, tmp_path: Path):
    lookup = flat_lookup
    row = _wrench_row(lookup)
    t = np.linspace(0.0, 0.2, 40)
    w = _wrench_from_gamma(lookup, row, np.array([np.tan(np.deg2rad(5.0)), 1e-6, -2e-5, 0.0, 0.0]))
    ramp = np.clip(t / 0.02, 0, 1)
    syn = {
        "times": t,
        "Fx": w[0] * ramp,
        "Fy": w[1] * ramp,
        "Mz": w[2] * ramp,
        "cop_x": np.zeros_like(t),
        "cop_y": np.zeros_like(t),
        "phi": np.full_like(t, float(lookup.phi_ref)),
    }
    result = replay_stance(
        lookup,
        synthetic=syn,
        visco_model="elastic",
        theta_mode="fit-cop",
        theta_min_deg=-10,
        theta_max_deg=30,
        cop_fit_min_force=1.0,
        shoe_width_m=1.0,
    )
    paths = save_replay_result(result, tmp_path)
    assert paths["csv"].exists()
    F = float(np.hypot(w[0], w[1]))
    assert np.nanmax(result.force_residual[10:]) < 1e-8 * F
    assert np.all(result.contact_start_index[10:] >= 0)
    assert np.all(result.contact_end_index[10:] >= result.contact_start_index[10:])
    same = result.record_index[10:] == row
    if np.any(same):
        np.testing.assert_allclose(result.theta_deg[10:][same], 5.0, atol=1e-6)
