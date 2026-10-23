"""External-codebase interface: SoleModel.step with continuous contact and the passive toe spring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from compliance_fem.api import SoleModel
from compliance_fem.contact.contact_model import save_contact_model
from compliance_fem.contact.continuous_toe import solve_toe_contact
from compliance_fem.contact.toe_spring import ToeSpringConfig

WIDTH = 0.10
LOAD = (-20.0, -300.0)  # N per shoe, foot on shoe (compressive)


@pytest.fixture()
def model(rocker_model):
    return SoleModel(rocker_model, shoe_width_m=WIDTH)


def test_step_matches_coupled_solver(model, rocker_model) -> None:
    st = model.step(*LOAD, 2.0)
    ref = solve_toe_contact(rocker_model.contact_operator, np.array(LOAD) / WIDTH, np.deg2rad(2.0), ToeSpringConfig(), WIDTH)
    assert st.valid and st.admissible
    assert st.theta_deg == pytest.approx(ref.theta_deg, abs=1e-12)
    np.testing.assert_array_equal(st.contact_mask, ref.contact.active)
    first, last = st.contact_interval or (-1, -1)
    assert (first, last) == (int(np.flatnonzero(ref.contact.active)[0]), int(np.flatnonzero(ref.contact.active)[-1]))


def test_force_balance_and_ground_contact(model) -> None:
    st = model.step(*LOAD, 2.0)
    np.testing.assert_allclose(st.top_force_N.sum(axis=1), LOAD, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(st.bottom_reaction_N.sum(axis=1), -np.array(LOAD), rtol=1e-9, atol=1e-9)
    assert (st.Fx_N, st.Fy_N) == pytest.approx(LOAD, rel=1e-9)
    assert np.max(np.abs(st.bottom_xy_m[1, st.contact_mask])) < 1e-12
    assert st.bottom_xy_m[1].min() > -1e-12
    assert np.all(st.bottom_reaction_N[:, ~st.contact_mask] == 0.0)
    np.testing.assert_allclose(st.bottom_gap_m, st.bottom_xy_m[1], atol=1e-15)


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
    lam = st.bottom_reaction_N[1]
    assert st.centre_of_pressure_x_m == pytest.approx(float(lam @ st.bottom_xy_m[0] / lam.sum()), rel=1e-12)


def test_warm_start_state_and_reset(model) -> None:
    assert model._warm is None
    st = model.step(*LOAD, 2.0)
    assert model._warm is not None and model._warm.theta_rad == pytest.approx(np.deg2rad(st.theta_deg))
    again = model.step(*LOAD, 2.0)
    assert again.theta_deg == pytest.approx(st.theta_deg, abs=1e-11)
    assert again.diagnostics["toe_iterations"] < 20
    model.reset()
    assert model._warm is None


def test_to_dict_is_json_serializable(model) -> None:
    d = model.step(*LOAD, 2.0).to_dict()
    json.dumps(d)
    assert len(d["top_xy_m"]) == 2 and len(d["top_xy_m"][0]) == model.n_top


def test_loads_from_path(rocker_model, tmp_path) -> None:
    save_contact_model(rocker_model, tmp_path)
    st = SoleModel(tmp_path, shoe_width_m=WIDTH).step(*LOAD, 2.0)
    ref = SoleModel(rocker_model, shoe_width_m=WIDTH).step(*LOAD, 2.0)
    assert st.contact_interval == ref.contact_interval
    assert st.theta_deg == pytest.approx(ref.theta_deg, abs=1e-10)


def test_rejects_bad_configuration(rocker_model) -> None:
    with pytest.raises(ValueError, match="shoe_width_m"):
        SoleModel(rocker_model, shoe_width_m=0.0)
