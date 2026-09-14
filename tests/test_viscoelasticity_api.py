"""General viscoelastic mapper API tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.viscoelasticity import (
    FractionalConfig,
    FungConfig,
    SLSConfig,
    create_material,
)


@pytest.fixture
def sls():
    return create_material(SLSConfig(g_inf=0.5, tau_r=1.0))


def test_zero_force_history_gives_zero_elastic(sls):
    t = np.linspace(0.0, 2.0, 50)
    FVE = np.zeros((t.size, 2))
    FE = sls.evaluate_history(t, FVE)
    np.testing.assert_allclose(FE, 0.0, atol=1e-15)


def test_zero_timestep_handling(sls):
    sls.reset()
    fe0 = sls.update([0.0, 0.0], 0.0)
    np.testing.assert_allclose(fe0, 0.0)
    # Repeated identical stamp is a no-op.
    fe1 = sls.update([0.0, 0.0], 0.0)
    np.testing.assert_allclose(fe1, 0.0)
    # Advance, then dt=0 with force change uses glassy J(0)=1 jump.
    sls.update([1.0, -0.5], 0.1)
    fe_before = sls.FE.copy()
    fe_same_t = sls.update([1.2, -0.4], 0.1)
    np.testing.assert_allclose(fe_same_t - fe_before, [0.2, 0.1], rtol=1e-12)


def test_non_monotone_time_rejected(sls):
    sls.reset()
    sls.update([0.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        sls.update([0.0, 0.0], 0.5)


def test_reset_reproduces_identical_results(sls):
    t = np.linspace(0.0, 1.0, 40)
    FVE = np.column_stack([np.sin(3 * t), np.cos(2 * t) - 1.0])
    a = sls.evaluate_history(t, FVE, reset=True)
    b = sls.evaluate_history(t, FVE, reset=True)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_components_process_independently(sls):
    t = np.linspace(0.0, 1.0, 60)
    fx = np.sin(5 * t)
    fy = 0.3 * t
    both = sls.evaluate_history(t, np.column_stack([fx, fy]), reset=True)
    x_only = create_material(SLSConfig(g_inf=0.5, tau_r=1.0, n_components=1))
    y_only = create_material(SLSConfig(g_inf=0.5, tau_r=1.0, n_components=1))
    fe_x = x_only.evaluate_history(t, fx.reshape(-1, 1), reset=True)[:, 0]
    fe_y = y_only.evaluate_history(t, fy.reshape(-1, 1), reset=True)[:, 0]
    np.testing.assert_allclose(both[:, 0], fe_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(both[:, 1], fe_y, rtol=1e-12, atol=1e-12)


def test_factory_dict_and_validation():
    m = create_material({"model": "sls", "g_inf": 0.25, "tau_r": 2.0})
    assert m.parameter_summary()["model"] == "sls"
    with pytest.raises(ValueError):
        create_material(SLSConfig(g_inf=0.0, tau_r=1.0))
    with pytest.raises(ValueError):
        create_material(FractionalConfig(E0=1.0, T=1.0, alpha=1.0))
    with pytest.raises(ValueError):
        create_material(FungConfig(C=0.1, tau1=1.0, tau2=0.5))


def test_evaluate_history_uses_update_path(sls):
    t = np.array([0.0, 0.1, 0.25, 0.5])
    FVE = np.array([[0.0, 0.0], [1.0, 0.5], [1.0, 0.5], [0.2, -0.1]])
    batch = sls.evaluate_history(t, FVE, reset=True)
    sls.reset()
    manual = np.array([sls.update(FVE[i], float(t[i])) for i in range(len(t))])
    np.testing.assert_allclose(batch, manual, rtol=0.0, atol=0.0)
