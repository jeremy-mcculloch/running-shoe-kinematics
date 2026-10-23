"""2.5D out-of-plane tangent stiffness: formulas, invariances, non-interference, integration."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from compliance_fem.api import SoleModel
from compliance_fem.contact import out_of_plane as oop_module
from compliance_fem.contact.continuous_contact import _normal_operator, solve_contact_state
from compliance_fem.contact.out_of_plane import (
    STATUS_ILL_CONDITIONED,
    STATUS_NO_CONTACT,
    STATUS_OK,
    STATUS_SINGULAR,
    OutOfPlaneSettings,
    compute_out_of_plane_stiffness,
    lateral_ratio_from_poisson,
)
from compliance_fem.contact.toe_spring import ToeSpringConfig
from compliance_fem.gait.replay import load_replay_arrays, replay_stance, save_replay_result
from compliance_fem.plotting.shape_render import build_shape_plot_data

B = 0.1
ETA = 0.5


def _settings(**kw) -> OutOfPlaneSettings:
    base: dict[str, Any] = dict(shoe_width_m=B, lateral_to_normal_stiffness_ratio=ETA)
    base.update(kw)
    return OutOfPlaneSettings(**base)


def _state(model, F=(0.0, -2000.0), varphi=0.0, alpha=0.0):
    return solve_contact_state(model.contact_operator, varphi, alpha, np.array(F))


def _K_active(model, sol) -> np.ndarray:
    """Explicit per-width normal tangent of the active set (test reference only)."""
    op = model.contact_operator
    c, s = np.cos(sol.varphi), np.sin(sol.varphi)
    a = np.flatnonzero(sol.active)
    H = _normal_operator(op, c, s, s, c)[np.ix_(a, a)]
    return np.linalg.inv(0.5 * (H + H.T))


def _x_fixed(model, sol) -> np.ndarray:
    op = model.contact_operator
    n = op.n_bottom
    c, s = np.cos(sol.varphi), np.sin(sol.varphi)
    return sol.T[0] + c * (op.x_bottom + sol.d_b[:n]) - s * (op.y_bottom + sol.d_b[n:])


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# 1. No contact: zeros and status no_contact.
def test_no_contact_returns_zero(rocker_model) -> None:
    r = compute_out_of_plane_stiffness(rocker_model.contact_operator, _state(rocker_model, (0.0, 0.0)), _settings())
    assert r.status == STATUS_NO_CONTACT
    assert (r.lateral_translation_stiffness, r.roll_stiffness, r.yaw_stiffness) == (0.0, 0.0, 0.0)
    assert r.active_contact_node_count == 0


# 2. A stable single contact region: finite, nonnegative.
@pytest.mark.parametrize("sole", ["flat_model", "rocker_model", "asym_model"])
def test_single_region_finite_nonnegative(request, sole) -> None:
    model = request.getfixturevalue(sole)
    sol = _state(model)
    r = compute_out_of_plane_stiffness(model.contact_operator, sol, _settings())
    assert r.status == STATUS_OK and sol.diagnostics["n_patches"] == 1
    assert np.all(np.isfinite(r.diagonal)) and np.all(r.diagonal > 0.0)
    assert r.min_eigenvalue > 0.0 and r.symmetry_error < 1e-10 and not r.augmented and not r.regularized


# 3-5. eta_t scales k_z and k_yaw linearly and leaves k_roll unchanged.
def test_eta_scaling(rocker_model) -> None:
    sol = _state(rocker_model)
    r1 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(lateral_to_normal_stiffness_ratio=0.3))
    r2 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(lateral_to_normal_stiffness_ratio=0.9))
    assert _rel(r2.lateral_translation_stiffness, 3.0 * r1.lateral_translation_stiffness) < 1e-12
    assert _rel(r2.yaw_stiffness, 3.0 * r1.yaw_stiffness) < 1e-12
    assert _rel(r2.roll_stiffness, r1.roll_stiffness) < 1e-14


# 6-7. Constant-width roll: b^2/12 times the vertical stiffness; x4 at fixed physical tangent.
def test_roll_constant_width(rocker_model) -> None:
    sol = _state(rocker_model)
    K = _K_active(rocker_model, sol)
    one = np.ones(K.shape[0])
    for b in (0.08, 0.16):
        r = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(shoe_width_m=b))
        k_vertical = b * float(one @ K @ one)  # physical vertical tangent, N/m
        assert _rel(r.roll_stiffness, b**2 / 12.0 * k_vertical) < 1e-9
    r1 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(shoe_width_m=0.08))
    r2 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(shoe_width_m=0.16))
    kv1, kv2 = 0.08 * float(one @ K @ one), 0.16 * float(one @ K @ one)
    # Holding the physical vertical tangent fixed, doubling b gives x4; the per-metre model also doubles it (x8 total).
    assert _rel((r2.roll_stiffness / kv2) / (r1.roll_stiffness / kv1), 4.0) < 1e-9
    assert _rel(r2.roll_stiffness / r1.roll_stiffness, 8.0) < 1e-9


# 8-9. k_z = 1^T K_t 1 and k_yaw = r^T K_t r with K_t = eta b K (constant width).
def test_lateral_and_yaw_match_quadratic_forms(rocker_model) -> None:
    sol = _state(rocker_model, F=(150.0, -2500.0), varphi=0.03)
    K_t = ETA * B * _K_active(rocker_model, sol)
    r = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    one = np.ones(K_t.shape[0])
    rx = _x_fixed(rocker_model, sol)[sol.active] - r.yaw_reference_x
    assert _rel(r.lateral_translation_stiffness, float(one @ K_t @ one)) < 1e-9
    assert _rel(r.yaw_stiffness, float(rx @ K_t @ rx)) < 1e-9
    assert r.yaw_reference_source == "center_of_pressure"
    assert r.yaw_reference_x == pytest.approx(sol.diagnostics["cop_fixed_x"], abs=1e-14)


# 10. Translating all x and the yaw reference together leaves k_yaw unchanged.
def test_yaw_translation_invariance(rocker_model) -> None:
    sol = _state(rocker_model)
    shifted = dataclasses.replace(sol, T=sol.T + np.array([0.37, 0.0]))
    for ref in ("center_of_pressure", "shoe_reference"):
        a = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(yaw_reference=ref))
        b = compute_out_of_plane_stiffness(rocker_model.contact_operator, shifted, _settings(yaw_reference=ref))
        assert _rel(a.yaw_stiffness, b.yaw_stiffness) < 1e-9
        assert b.yaw_reference_x == pytest.approx(a.yaw_reference_x + 0.37, abs=1e-14)


# 11-12. Yaw stiffness vs reference: quadratic in the offset, symmetric about the stiffness centroid.
def test_yaw_reference_dependence_and_symmetry(rocker_model) -> None:
    sol = _state(rocker_model)
    op = rocker_model.contact_operator
    K_t = ETA * B * _K_active(rocker_model, sol)
    xa = _x_fixed(rocker_model, sol)[sol.active]
    one = np.ones(xa.size)
    centroid = float(one @ K_t @ xa / (one @ K_t @ one))
    c = np.cos(sol.varphi)

    def k_at(x_fixed: float) -> float:
        x_mat = (x_fixed - sol.T[0]) / c
        return compute_out_of_plane_stiffness(
            op, sol, _settings(yaw_reference="shoe_reference", fixed_yaw_reference_x=x_mat)
        ).yaw_stiffness

    for x0 in (0.05, 0.12):
        r = xa - x0
        assert _rel(k_at(x0), float(r @ K_t @ r)) < 1e-9
    for d in (0.01, 0.03):
        assert _rel(k_at(centroid + d), k_at(centroid - d)) < 1e-9
        assert k_at(centroid + d) > k_at(centroid)


# 13-14. A different converged active set changes the values; the calculation does not change it.
def test_active_set_dependence_and_no_mutation(rocker_model) -> None:
    light, heavy = _state(rocker_model, (0.0, -800.0)), _state(rocker_model, (0.0, -6000.0))
    assert light.active.sum() != heavy.active.sum()
    a = compute_out_of_plane_stiffness(rocker_model.contact_operator, light, _settings())
    b = compute_out_of_plane_stiffness(rocker_model.contact_operator, heavy, _settings())
    assert a.active_contact_node_count != b.active_contact_node_count
    assert not np.allclose(a.diagonal, b.diagonal)


# 15-17. Sagittal forces, toe angle and displacements are untouched (SoleModel with and without 2.5D).
def test_step_sagittal_results_unchanged(rocker_model) -> None:
    on = SoleModel(rocker_model, shoe_width_m=B)
    off = SoleModel(rocker_model, shoe_width_m=B, out_of_plane=OutOfPlaneSettings(enabled=False, shoe_width_m=B))
    for F, phi in (((-20.0, -300.0), 2.0), ((10.0, -150.0), -1.0)):
        s1, s2 = on.step(*F, phi), off.step(*F, phi)
        assert s1.theta_deg == s2.theta_deg
        np.testing.assert_array_equal(s1.contact_mask, s2.contact_mask)
        for name in ("top_force_N", "bottom_reaction_N", "top_xy_m", "bottom_xy_m", "bottom_displacement_m"):
            np.testing.assert_array_equal(getattr(s1, name), getattr(s2, name))
        assert (s1.Fx_N, s1.Fy_N, s1.Mz_shoe_Nm) == (s2.Fx_N, s2.Fy_N, s2.Mz_shoe_Nm)
        assert s2.out_of_plane_stiffness is not None and s2.out_of_plane_stiffness.status == "disabled"


def test_solution_object_not_modified(rocker_model) -> None:
    sol = _state(rocker_model, (100.0, -3000.0), 0.02, 0.1)
    snap = {k: np.array(v, copy=True) for k, v in vars(sol).items() if isinstance(v, np.ndarray)}
    compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    for k, v in snap.items():
        np.testing.assert_array_equal(getattr(sol, k), v)


# 18, 29. The deformed-sole plot is unchanged, and the stiffness does not depend on the plot scale.
def test_plot_independence(rocker_model) -> None:
    from compliance_fem.contact.contact_model import selection_proxy

    sol = _state(rocker_model)
    before = build_shape_plot_data(rocker_model, selection_proxy(rocker_model, sol), scale_mode="manual", manual_scale=7.0)
    k0 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    after = build_shape_plot_data(rocker_model, selection_proxy(rocker_model, sol), scale_mode="manual", manual_scale=7.0)
    for name in ("x_top_def", "y_top_def", "x_bottom_def", "y_bottom_def", "bottom_node_x", "bottom_node_y"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    build_shape_plot_data(rocker_model, selection_proxy(rocker_model, sol), scale_mode="manual", manual_scale=50.0)
    k1 = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    np.testing.assert_array_equal(k0.diagonal, k1.diagonal)


# 19. No FEM or contact re-solve happens inside the calculation.
def test_no_additional_solves(rocker_model, monkeypatch) -> None:
    from compliance_fem.contact import active_set, continuous_contact
    from compliance_fem.fem import compliance

    sol = _state(rocker_model)

    def forbid(*_a, **_k):
        raise AssertionError("solve called during the 2.5D stiffness calculation")

    monkeypatch.setattr(compliance, "compute_compliance", forbid)
    monkeypatch.setattr(continuous_contact, "solve_contact_state", forbid)
    monkeypatch.setattr(active_set, "solve_mixed_lcp", forbid)
    monkeypatch.setattr(oop_module.np.linalg, "inv", forbid)
    r = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    assert r.status == STATUS_OK


# 20. SoleModel.step returns all three values, with timings.
def test_step_returns_stiffness(rocker_model) -> None:
    st = SoleModel(rocker_model, shoe_width_m=B).step(-20.0, -300.0, 2.0)
    o = st.out_of_plane_stiffness
    assert o is not None and o.status in (STATUS_OK, STATUS_ILL_CONDITIONED)
    assert np.all(np.isfinite(o.diagonal))
    assert o.lateral_ratio_source.startswith("assumed thin-layer")
    assert o.lateral_to_normal_stiffness_ratio == pytest.approx(lateral_ratio_from_poisson(0.113))
    d = st.diagnostics
    assert d["out_of_plane_stiffness_time_s"] < d["sagittal_solve_time_s"] + 0.05
    assert d["total_step_time_s"] >= d["sagittal_solve_time_s"]
    st.to_dict()


# 21. The GUI section shows the same values as SoleModel.step for the same state and settings.
def test_gui_values_match_step(rocker_model) -> None:
    import streamlit as stl

    from compliance_fem.gui import app

    from compliance_fem.contact.continuous_toe import solve_toe_contact

    state = SoleModel(rocker_model, shoe_width_m=B, out_of_plane=_settings()).step(-20.0, -300.0, 2.0)
    # The GUI evaluates the same pose with the same (cold-started) coupled solve.

    fr = solve_toe_contact(rocker_model.contact_operator, np.array([-20.0, -300.0]) / B, np.deg2rad(2.0), ToeSpringConfig(), B)
    app._out_of_plane_section(rocker_model, fr.contact, _settings())
    gui = stl.session_state["gui_oop_result"]
    assert state.out_of_plane_stiffness is not None
    np.testing.assert_allclose(gui.diagonal, state.out_of_plane_stiffness.diagonal, rtol=1e-9)


# 22-25. Replay: one value per frame, time / stance axes, NaN for invalid frames, CSV with units and status.
@pytest.fixture(scope="module")
def replay(rocker_model):
    t = np.linspace(0.0, 0.2, 81)
    Fy = -3000.0 * np.sin(np.pi * t / 0.2) ** 2 - 20.0
    syn = {"times": t, "Fx": 0.1 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": np.deg2rad(4.0) * (t / 0.2 - 0.5)}
    return replay_stance(rocker_model, synthetic=syn, shoe_width_m=B, out_of_plane=_settings())


def test_replay_stores_one_value_per_frame(replay, rocker_model) -> None:
    n = replay.times.size
    for name in ("lateral_translation_stiffness", "roll_stiffness", "yaw_stiffness", "oop_yaw_reference_x"):
        assert getattr(replay, name).shape == (n,)
    assert len(replay.oop_stiffness_status) == n
    ok = replay.frame_valid
    assert np.all(np.isfinite(replay.roll_stiffness[ok]))
    # The stored value is the one computed from that frame's converged state.
    i = int(np.flatnonzero(ok)[ok.sum() // 2])
    ref = compute_out_of_plane_stiffness(rocker_model.contact_operator, replay.frames[i].contact, _settings())
    assert ref.roll_stiffness == pytest.approx(replay.roll_stiffness[i], rel=1e-12)
    assert ref.yaw_stiffness == pytest.approx(replay.yaw_stiffness[i], rel=1e-12)


def test_replay_plot_axes(replay) -> None:
    from compliance_fem.gui.gait_plots import out_of_plane_stiffness_figure

    for axis, ref in (("time", replay.times), ("stance", replay.stance_percent)):
        fig = out_of_plane_stiffness_figure(replay, x_axis=axis)
        np.testing.assert_array_equal(np.asarray(fig.data[0].x), ref)  # pyright: ignore[reportAttributeAccessIssue]
        y = np.asarray(fig.data[0].y, dtype=float)  # pyright: ignore[reportAttributeAccessIssue]
        assert np.array_equal(np.isnan(y), np.isnan(replay.lateral_translation_stiffness))


def test_invalid_frames_are_nan(rocker_model) -> None:
    t = np.linspace(0.0, 0.1, 11)
    Fy = np.full_like(t, -3000.0)
    syn = {"times": t, "Fx": 0.1 * Fy, "Fy": Fy, "Mz": np.zeros_like(t), "phi": np.zeros_like(t)}
    # A toe window that excludes the equilibrium: no root -> unconverged frames.
    cfg = ToeSpringConfig(toe_angle_min_deg=60.0, toe_angle_max_deg=61.0, toe_root_scan_points=5)
    r = replay_stance(rocker_model, synthetic=syn, shoe_width_m=B, toe_config=cfg, out_of_plane=_settings())
    bad = ~r.frame_valid
    assert bad.any()
    assert np.all(np.isnan(r.lateral_translation_stiffness[bad]))
    assert all(r.oop_stiffness_status[i] == "invalid_frame" for i in np.flatnonzero(bad))


def test_csv_export_has_units_and_status(replay, tmp_path) -> None:
    from compliance_fem.gui.gait_plots import out_of_plane_csv

    text = out_of_plane_csv(replay)
    header = text.splitlines()[0].split(",")
    for col in ("time_s", "lateral_translation_stiffness_N_per_m", "roll_stiffness_Nm_per_rad",
                "yaw_stiffness_Nm_per_rad", "oop_stiffness_status", "oop_yaw_reference_x_m"):
        assert col in header
    assert len(text.splitlines()) == replay.times.size + 1
    paths = save_replay_result(replay, tmp_path)
    assert "roll_stiffness_Nm_per_rad" in paths["csv"].read_text().splitlines()[0]


# 26-27. Uniform profile == constant width; the width enters exactly once.
def test_width_profile_and_single_scaling(rocker_model) -> None:
    sol = _state(rocker_model)
    const = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    prof = compute_out_of_plane_stiffness(
        rocker_model.contact_operator, sol, _settings(shoe_width_x=(0.0, 0.1, 0.25), shoe_width_values=(B, B, B))
    )
    np.testing.assert_allclose(prof.diagonal, const.diagonal, rtol=1e-12)
    assert prof.shoe_width_summary["mode"] == "profile"
    double = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(shoe_width_m=2 * B))
    assert _rel(double.lateral_translation_stiffness, 2.0 * const.lateral_translation_stiffness) < 1e-12
    assert _rel(double.yaw_stiffness, 2.0 * const.yaw_stiffness) < 1e-12
    with pytest.raises(ValueError):
        OutOfPlaneSettings(shoe_width_x=(0.0, 0.0), shoe_width_values=(0.1, 0.1))
    with pytest.raises(ValueError):
        OutOfPlaneSettings(shoe_width_m=-0.1)
    with pytest.raises(ValueError):
        OutOfPlaneSettings(lateral_to_normal_stiffness_ratio=-1.0)


# 28. Ill-conditioning: warning above the threshold, NaN when numerically singular.
def test_conditioning_policy(rocker_model, monkeypatch) -> None:
    sol = _state(rocker_model)
    warn = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings(condition_number_warning=1.0001))
    assert warn.status == STATUS_ILL_CONDITIONED and warn.warnings and np.all(np.isfinite(warn.diagonal))
    monkeypatch.setattr(oop_module, "SINGULAR_CONDITION", 1.0)
    sing = compute_out_of_plane_stiffness(rocker_model.contact_operator, sol, _settings())
    assert sing.status == STATUS_SINGULAR and np.all(np.isnan(sing.diagonal))


# 30. Old replay files without 2.5D columns stay loadable; stiffness reads N/A.
def test_old_replay_without_stiffness_loads(replay, tmp_path) -> None:
    save_replay_result(replay, tmp_path)
    data = dict(np.load(tmp_path / "gait_replay.npz", allow_pickle=True))
    for k in [k for k in data if "stiffness" in k or k.startswith("oop_")]:
        del data[k]
    np.savez_compressed(tmp_path / "gait_replay.npz", **data)
    loaded = load_replay_arrays(tmp_path)
    assert np.all(np.isnan(loaded["roll_stiffness_Nm_per_rad"]))
    assert set(loaded["oop_stiffness_status"].tolist()) == {"not_available"}
    np.testing.assert_array_equal(loaded["theta"], replay.theta_deg)


def test_matrix_structure(rocker_model) -> None:
    r = compute_out_of_plane_stiffness(rocker_model.contact_operator, _state(rocker_model), _settings())
    m = r.as_matrix()
    np.testing.assert_array_equal(np.diag(m), r.diagonal)
    assert np.all(np.isnan(m[~np.eye(3, dtype=bool)]))
