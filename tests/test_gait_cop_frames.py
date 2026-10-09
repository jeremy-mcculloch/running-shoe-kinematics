"""Tests for Wang COP frames and units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.gait.cop_frames import (
    cop_foot_to_lab,
    cop_lab_to_foot,
    cop_validity_mask,
    force_induced_moment_about_point,
    foot_vector_to_lab,
    lab_vector_to_foot,
    model_cop_from_wrench,
    translate_moment_between_points,
)
from compliance_fem.gait.sagittal import lab_grf_to_model_top_wrench
from compliance_fem.gait.units import resolve_position_units, resolve_units
from compliance_fem.gait.wang_io import read_opensim_mot, read_wang_forces_csv


def test_mot_cop_columns_are_global_lab_axes(tmp_path: Path):
    mot = tmp_path / "P01_pr1_01_force.mot"
    mot.write_text(
        "\n".join(
            [
                "nRows=2",
                "nColumns=10",
                "endheader",
                "time 1_force_vx 1_force_vy 1_force_vz 1_force_px 1_force_py 1_force_pz "
                "1_torque_x 1_torque_y 1_torque_z",
                "0.0 1.0 800.0 0.1 1.2 0.0 -0.3 0.0 0.0 0.5",
                "0.001 2.0 700.0 0.2 1.3 0.0 -0.25 0.0 0.0 0.4",
            ]
        ),
        encoding="utf-8",
    )
    data = read_opensim_mot(mot, position_units="m", moment_units="N-m")
    assert data["axis_mapping"]["force_vx / px"] == "anterior"
    assert data["axis_mapping"]["cop_frame"] == "global_laboratory_ground"
    assert data["columns"]["1_force_px"][0] == pytest.approx(1.2)
    assert data["source_position_units"] == "m"


def test_csv_cx_cy_cz_mapping(tmp_path: Path):
    csv = tmp_path / "P4 pr1 01.csv"
    csv.write_text(
        "\n".join(
            [
                "Devices",
                "1000",
                ",,Kistler - Force,,,Kistler - Moment,,,Kistler - CoP,,,",
                "Frame,Sub Frame,Fx,Fy,Fz,Mx,My,Mz,Cx,Cy,Cz",
                ",,N,N,N,mm.N,mm.N,mm.N,mm,mm,mm",
                # Cx=ML=100mm, Cy=AP=250mm, Cz=0
                "1,0,10,20,800,0,0,1000,100,250,0",
                "2,0,11,21,700,0,0,900,110,260,0",
            ]
        ),
        encoding="utf-8",
    )
    data = read_wang_forces_csv(csv, position_units="mm", moment_units="N-mm")
    assert data["axis_mapping"]["csv_Cx"].startswith("mediolateral")
    assert data["columns"]["1_force_px"][0] == pytest.approx(0.25)  # Cy mm→m AP
    assert data["columns"]["1_force_pz"][0] == pytest.approx(0.10)  # Cx ML
    assert data["columns"]["1_force_py"][0] == pytest.approx(0.0)
    assert data["columns"]["1_torque_z"][0] == pytest.approx(1.0)  # 1000 N·mm


def test_mm_and_nmm_convert_once():
    u = resolve_units(
        position_mode="mm",
        moment_mode="N-mm",
        sample_positions=np.array([250.0]),
        sample_moments=np.array([12000.0]),
    )
    assert u.position_scale == pytest.approx(1e-3)
    assert u.moment_scale == pytest.approx(1e-3)
    # Applying scale once
    assert 250.0 * u.position_scale == pytest.approx(0.25)
    assert 12000.0 * u.moment_scale == pytest.approx(12.0)


def test_auto_units_reject_ambiguous_position():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_position_units(
            declared=None, sample_positions=np.array([30.0]), mode="auto"
        )


def test_global_cop_not_heel_relative():
    # Global COP far from heel; foot-frame x is relative.
    cop = cop_lab_to_foot(
        cop_lab_x=np.array([-1.2]),
        cop_lab_y=np.array([0.0]),
        heel_lab_x=np.array([-1.4]),
        heel_lab_y=np.array([0.05]),
        phi=np.array([0.0]),
    )
    assert cop.x[0] == pytest.approx(0.2)
    assert abs(cop.x[0] - (-1.2)) > 0.5  # not the raw global coordinate


def test_pure_translation_invariant_foot_cop():
    phi = np.deg2rad(-20.0)
    heel = np.array([1.0, 0.1])
    cop = np.array([1.15, 0.02])
    a = cop_lab_to_foot(cop[0], cop[1], heel[0], heel[1], phi)
    shift = np.array([3.5, -2.0])
    b = cop_lab_to_foot(
        cop[0] + shift[0],
        cop[1] + shift[1],
        heel[0] + shift[0],
        heel[1] + shift[1],
        phi,
    )
    assert a.x == pytest.approx(b.x)
    assert a.y == pytest.approx(b.y)


def test_known_rotation_foot_cop():
    phi = np.deg2rad(30.0)
    # Lab offset from heel: (0.2, 0) → foot: Q^T [0.2,0]
    foot = cop_lab_to_foot(0.2, 0.0, 0.0, 0.0, phi)
    c, s = np.cos(phi), np.sin(phi)
    assert foot.x == pytest.approx(c * 0.2)
    assert foot.y == pytest.approx(-s * 0.2)


def test_inverse_recovers_global_cop():
    phi = np.deg2rad(-15.0)
    heel_x, heel_y = -1.5, 0.08
    cop_x, cop_y = -1.35, 0.01
    foot = cop_lab_to_foot(cop_x, cop_y, heel_x, heel_y, phi)
    lab = cop_foot_to_lab(foot.x, foot.y, heel_x, heel_y, phi)
    assert lab.x == pytest.approx(cop_x)
    assert lab.y == pytest.approx(cop_y)


def test_force_rotation_matches_cop_Q():
    phi = np.deg2rad(25.0)
    fx, fy = 100.0, 800.0
    f_foot = lab_vector_to_foot(fx, fy, phi)
    back = foot_vector_to_lab(*f_foot, phi)
    assert back[0] == pytest.approx(fx)
    assert back[1] == pytest.approx(fy)


def test_upward_grf_produces_compressive_top_force():
    t = np.array([0.0, 0.01])
    wrench, _ = lab_grf_to_model_top_wrench(
        times=t,
        fx_lab=np.zeros(2),
        fy_lab=np.array([0.0, 800.0]),  # upward lab GRF
        fz_lab=np.zeros(2),
        cop_x_lab=np.zeros(2),
        cop_y_lab=np.zeros(2),
        mz_lab=np.zeros(2),
    )
    assert wrench.Fy[1] < 0.0  # compressive top load


def test_moment_translation_and_rotation_invariance():
    fx, fy = 50.0, -800.0
    # Force through (0.2, 0) about origin
    m0 = force_induced_moment_about_point(fx, fy, 0.2, 0.0, 0.0, 0.0)
    m_a = translate_moment_between_points(
        m0, fx, fy, from_xy=(0.0, 0.0), to_xy=(0.1, 0.0)
    )
    m_a_direct = force_induced_moment_about_point(fx, fy, 0.2, 0.0, 0.1, 0.0)
    assert m_a == pytest.approx(m_a_direct)
    # Horizontal force with vertical lever: r=(0,0.05), F=(fx,0) → M = -0.05*fx
    m_h = force_induced_moment_about_point(fx, 0.0, 0.0, 0.05, 0.0, 0.0)
    assert m_h == pytest.approx(-0.05 * fx)
    # Rigid rotation of the whole configuration leaves M about the transformed point
    phi = np.deg2rad(40.0)
    r_lab = np.array([0.2, 0.05])
    f_lab = np.array([fx, fy])
    m_lab = r_lab[0] * f_lab[1] - r_lab[1] * f_lab[0]
    r_f = lab_vector_to_foot(r_lab[0], r_lab[1], phi)
    f_f = lab_vector_to_foot(f_lab[0], f_lab[1], phi)
    m_f = r_f[0] * f_f[1] - r_f[1] * f_f[0]
    assert m_f == pytest.approx(m_lab)


def test_free_torque_distinguished_from_force_moment():
    fx, fy = 0.0, -800.0
    m_force = force_induced_moment_about_point(fx, fy, 0.15, 0.0, 0.0, 0.0)
    tz_free = 0.5
    m_total = m_force + tz_free
    assert m_force == pytest.approx(0.15 * fy)
    assert m_total == pytest.approx(m_force + 0.5)


def test_model_cop_from_wrench_and_low_force_invalid():
    mz = np.array([80.0, 10.0])
    fy = np.array([-800.0, -5.0])
    x = model_cop_from_wrench(mz, fy, fy_min=20.0)
    assert x[0] == pytest.approx(80.0 / -800.0)
    assert np.isnan(x[1])
    valid, oos = cop_validity_mask(
        fy_lab_vertical=np.array([800.0, 5.0]),
        cop_foot_x=np.array([0.1, 0.1]),
        shoe_length=0.3,
        fy_min=20.0,
    )
    assert bool(valid[0]) is True
    assert bool(valid[1]) is False


def test_placeholder_zero_cop_outside_stance_invalid():
    valid, _ = cop_validity_mask(
        fy_lab_vertical=np.array([0.0]),
        cop_foot_x=np.array([0.0]),
        shoe_length=0.3,
        fy_min=20.0,
    )
    assert bool(valid[0]) is False


def test_out_of_shoe_flagged_not_clipped():
    valid, oos = cop_validity_mask(
        fy_lab_vertical=np.array([500.0]),
        cop_foot_x=np.array([2.0]),
        shoe_length=0.3,
        fy_min=20.0,
        x_tol=0.05,
    )
    assert bool(valid[0]) is True
    assert bool(oos[0]) is True


@pytest.mark.skipif(
    not Path("data/wang/trc/P4/P4 pr1 01.trc").exists()
    or not Path("outputs/contact_lookup/contact_lookup.npz").exists(),
    reason="Wang data / lookup not present",
)
def test_real_wang_stance_cop_frames():
    from compliance_fem.contact.lookup import load_contact_lookup
    from compliance_fem.gait.replay import replay_stance

    lookup = load_contact_lookup("outputs/contact_lookup")
    result = replay_stance(
        lookup,
        trc_path="data/wang/trc/P4/P4 pr1 01.trc",
        mot_path="data/wang/MOT/P4/P4 pr1 01_force.mot",
        shoe_width_m=0.10,
        position_units="m",
        moment_units="N-m",
    )
    # Global COP must differ from heel-relative COP
    assert result.cop_lab_x.size == result.times.size
    assert np.nanmax(np.abs(result.cop_lab_x - result.cop_foot_x)) > 0.1
    # Heel-relative COP should be nearer shoe scale than lab origin distance
    valid = result.cop_valid
    assert np.count_nonzero(valid) > 10
    x = result.cop_foot_x[valid]
    assert float(np.nanmin(x)) > -0.5
    assert float(np.nanmax(x)) < float(lookup.L) + 0.5
