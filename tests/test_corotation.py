"""Co-rotating frame, angle conventions and the display transform."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from compliance_fem.contact.basis import shape_mode_phi1
from compliance_fem.contact.contact_model import build_contact_model, selection_proxy
from compliance_fem.contact.continuous_contact import solve_contact_state
from compliance_fem.contact.corotation import (
    SHAPE_MODE_SIGN,
    reference_chord_angle,
    rotate_vector_to_fixed,
    transform_to_fixed_frame,
)
from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.plotting.shape_render import build_shape_plot_data

from conftest import SMALL_TOE_LENGTH, small_config


def _Q(v: float) -> np.ndarray:
    c, s = np.cos(v), np.sin(v)
    return np.array([[c, -s], [s, c]])


def test_shape_mode_vanishes_at_endpoints() -> None:
    for L, toe_length, kappa in ((1.0, 0.5, 30.0), (0.25, 0.10, 60.0), (2.0, 0.3, 8.0)):
        np.testing.assert_allclose(
            shape_mode_phi1(np.array([0.0, L]), L=L, toe_length=toe_length, kappa=kappa), 0.0, atol=1e-14
        )


def test_model_build_takes_no_angle() -> None:
    names = set(inspect.signature(build_contact_model).parameters)
    assert not (names & {"phi", "phi_deg", "varphi", "theta", "theta_deg"})


@pytest.mark.parametrize("varphi_deg", (-30.0, -5.0, 0.0, 12.0))
def test_vector_rotation_to_fixed_and_back(varphi_deg) -> None:
    v = np.deg2rad(varphi_deg)
    fx, fy = rotate_vector_to_fixed(np.array(120.0), np.array(-900.0), v)
    np.testing.assert_allclose([fx, fy], _Q(v) @ np.array([120.0, -900.0]), atol=1e-12)
    bx, by = rotate_vector_to_fixed(fx, fy, -v)
    assert (float(bx), float(by)) == pytest.approx((120.0, -900.0), abs=1e-10)


def test_positive_theta_bends_the_toe_up() -> None:
    L, toe_length, kappa = 1.0, 0.5, 30.0
    x = np.linspace(0.0, L, 401)
    v = SHAPE_MODE_SIGN * np.tan(np.deg2rad(10.0)) * shape_mode_phi1(x, L=L, toe_length=toe_length, kappa=kappa)
    slope = np.gradient(v, x)
    assert float(np.mean(slope[x >= L - toe_length])) > 0.0
    assert float(np.mean(slope[x <= L - toe_length])) < 0.0
    assert SHAPE_MODE_SIGN == 1.0


def test_sloped_top_reference_chord() -> None:
    """A sole whose reference top chord is deliberately not horizontal."""
    result = compute_compliance(small_config(top_slope=0.02, EI_plate_Nm2_per_m=5.0, min_angle_deg=5.0))
    model = build_contact_model(result, toe_length=SMALL_TOE_LENGTH, kappa=30.0)
    x_top, y_top = model.x_top, model.y_top
    expected = np.arctan2(float(y_top[-1] - y_top[0]), float(x_top[-1] - x_top[0]))
    assert model.phi_ref == pytest.approx(expected) and abs(expected) > 1e-3
    assert reference_chord_angle(np.array([0.0, 1.0]), None) == 0.0


def test_transform_to_fixed_frame_is_exact_rotation() -> None:
    x = np.array([0.0, 0.1, 0.2])
    y = np.array([0.0, 0.01, 0.03])
    u = np.array([1e-3, -2e-3, 0.5e-3])
    v = np.array([-1e-3, 0.0, 2e-3])
    varphi = np.deg2rad(25.0)
    xf, yf = transform_to_fixed_frame(x, y, u, v, 0.1, u[1], v[1], varphi, anchor=(0.3, -0.2), y_anchor=0.01)
    expected = np.array([[0.3], [-0.2]]) + _Q(varphi) @ np.vstack([x + u - (0.1 + u[1]), y + v - (0.01 + v[1])])
    np.testing.assert_allclose(np.vstack([xf, yf]), expected, atol=1e-14)


def test_shape_plot_scales_only_the_elastic_part(rocker_model) -> None:
    op = rocker_model.operator
    sol = solve_contact_state(op, np.deg2rad(-2.0), np.tan(np.deg2rad(5.0)), np.array([0.0, -900.0]))
    state = selection_proxy(rocker_model, sol)
    exact = build_shape_plot_data(rocker_model, state)
    scaled = build_shape_plot_data(rocker_model, state, scale_mode="manual", manual_scale=20.0)
    assert exact.scale == 1.0 and scaled.scale == 20.0
    x = exact.x_dense
    u_bot = np.interp(x, rocker_model.x_bottom, state.full_bottom_u)
    v_bot = np.interp(x, rocker_model.x_bottom, state.full_bottom_v)
    expected = 19.0 * (_Q(state.varphi) @ np.vstack([u_bot - state.d_ax, v_bot - state.d_ay]))
    np.testing.assert_allclose(scaled.x_bottom_def - exact.x_bottom_def, expected[0], atol=1e-14)
    np.testing.assert_allclose(scaled.y_bottom_def - exact.y_bottom_def, expected[1], atol=1e-14)
    assert exact.varphi_deg == pytest.approx(np.rad2deg(state.varphi))
