"""External-codebase interface: SoleModel.step with the passive toe spring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from compliance_fem.api import SoleModel
from compliance_fem.contact_lookup import save_contact_lookup
from compliance_fem.gait.passive_toe import pick_instant_best_passive, solve_passive_toe_candidates
from compliance_fem.toe_spring import TOE_MODEL_PRESCRIBED_LEGACY, ToeSpringConfig

WIDTH = 0.10
LOAD = (-8.0, -80.0)  # N per shoe, foot on shoe (compressive)


@pytest.fixture()
def model(rocker_lookup):
    return SoleModel(rocker_lookup, shoe_width_m=WIDTH)


def test_step_matches_passive_toe_pipeline(model, rocker_lookup) -> None:
    st = model.step(*LOAD, 2.0)
    cands = solve_passive_toe_candidates(
        rocker_lookup, Fx_star=LOAD[0] / WIDTH, Fy_star=LOAD[1] / WIDTH, Mz_meas=float("nan"),
        phi_rad=np.deg2rad(2.0), config=ToeSpringConfig(), width_m=WIDTH,
    )
    best = pick_instant_best_passive(cands)
    assert st.valid and st.contact_interval == best.interval
    assert st.theta_deg == pytest.approx(best.theta_deg, abs=1e-12)
    assert st.Fx_N == pytest.approx(best.Fx_pred * WIDTH, rel=1e-12)


def test_force_balance_and_ground_contact(model) -> None:
    st = model.step(*LOAD, 2.0)
    assert st.admissible
    np.testing.assert_allclose(st.top_force_N.sum(axis=1), LOAD, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(st.bottom_reaction_N.sum(axis=1), -np.array(LOAD), rtol=1e-8, atol=1e-9)
    assert (st.Fx_N, st.Fy_N) == pytest.approx(LOAD, rel=1e-8)
    assert np.max(np.abs(st.bottom_xy_m[1, st.contact_mask])) < 1e-12
    free = ~st.contact_mask
    if free.any():
        assert st.bottom_xy_m[1, free].min() > -1e-9
    assert np.all(st.bottom_reaction_N[:, free] == 0.0)


def test_shapes_and_moments(model) -> None:
    st = model.step(*LOAD, 2.0)
    assert st.top_xy_m.shape == (2, model.n_top) == st.top_force_N.shape == st.top_displacement_m.shape
    assert st.bottom_xy_m.shape == (2, model.n_bottom) == st.bottom_reaction_N.shape
    c, s = np.cos(st.chord_rotation_rad), np.sin(st.chord_rotation_rad)
    f_local = np.array([[c, s], [-s, c]]) @ st.top_force_N
    Mz = float(np.sum(model.x_top * f_local[1] - model.y_top * f_local[0]))
    assert st.Mz_shoe_Nm == pytest.approx(Mz, rel=1e-8, abs=1e-12)
    fixed = float(np.sum(st.top_xy_m[0] * st.top_force_N[1] - st.top_xy_m[1] * st.top_force_N[0]))
    assert st.Mz_fixed_Nm == pytest.approx(fixed, rel=1e-12)


def test_continuity_state_and_reset(model) -> None:
    assert model._previous_interval is None
    st = model.step(*LOAD, 2.0)
    assert model._previous_interval == st.contact_interval
    assert model._previous_theta_deg == pytest.approx(st.theta_deg)
    model.reset()
    assert model._previous_interval is None and model._previous_theta_deg is None


def test_to_dict_is_json_serializable(model) -> None:
    d = model.step(*LOAD, 2.0).to_dict()
    json.dumps(d)
    assert len(d["top_xy_m"]) == 2 and len(d["top_xy_m"][0]) == model.n_top


def test_loads_from_path(rocker_lookup, tmp_path) -> None:
    save_contact_lookup(rocker_lookup, tmp_path)
    m = SoleModel(tmp_path, shoe_width_m=WIDTH)
    st = m.step(*LOAD, 2.0)
    ref = SoleModel(rocker_lookup, shoe_width_m=WIDTH).step(*LOAD, 2.0)
    assert st.contact_interval == ref.contact_interval
    assert st.theta_deg == pytest.approx(ref.theta_deg, abs=1e-10)


def test_rejects_bad_configuration(rocker_lookup) -> None:
    with pytest.raises(ValueError, match="shoe_width_m"):
        SoleModel(rocker_lookup, shoe_width_m=0.0)
    with pytest.raises(ValueError, match="passive"):
        SoleModel(rocker_lookup, toe_config=ToeSpringConfig(toe_model=TOE_MODEL_PRESCRIBED_LEGACY))
