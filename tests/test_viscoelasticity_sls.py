"""SLS viscoelastic mapper tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.viscoelasticity import SLSConfig, create_material
from compliance_fem.viscoelasticity.sls import SLSMaterial


def test_sls_step_response_asymptote():
    g_inf = 0.4
    mat = SLSMaterial(g_inf=g_inf, tau_r=0.5, n_components=1)
    # Hold a step after a short ramp from 0.
    t = np.concatenate([[0.0], np.linspace(1e-6, 20.0, 400)])
    FVE = np.ones((t.size, 1))
    FVE[0, 0] = 0.0
    FE = mat.evaluate_history(t, FVE)
    assert FE[-1, 0] == pytest.approx(1.0 / g_inf, rel=1e-4)


def test_sls_analytical_creep_after_step():
    """After an ideal step in F_VE at t=0+, Fe(t) = J(t) F_VE."""
    g_inf = 0.5
    tau_r = 1.0
    tau_c = tau_r / g_inf
    mat = SLSMaterial(g_inf=g_inf, tau_r=tau_r, n_components=1)
    F0 = 2.0
    # Approximate step with a very short ramp, then hold.
    t_ramp = 1e-8
    t = np.concatenate([[0.0, t_ramp], np.linspace(t_ramp, 8.0, 300)[1:]])
    FVE = np.full((t.size, 1), F0)
    FVE[0, 0] = 0.0
    FE = mat.evaluate_history(t, FVE)[:, 0]
    # Analytic J(t) * F0 for t >= t_ramp (relative to step at 0).
    J = 1.0 + (1.0 / g_inf - 1.0) * (1.0 - np.exp(-t / tau_c))
    # Skip the artificial ramp point.
    mask = t >= t_ramp
    np.testing.assert_allclose(FE[mask], J[mask] * F0, rtol=2e-3, atol=2e-3)


def test_sls_ramp_response():
    """Constant-rate F_VE = r t => closed form via Laplace / convolution."""
    g_inf = 0.5
    tau_r = 1.0
    tau_c = tau_r / g_inf
    beta = 1.0 / g_inf - 1.0
    r = 3.0
    mat = SLSMaterial(g_inf=g_inf, tau_r=tau_r, n_components=1)
    t = np.linspace(0.0, 5.0, 501)
    FVE = (r * t).reshape(-1, 1)
    FE = mat.evaluate_history(t, FVE)[:, 0]
    # J * r  (since dFVE/dt = r): Fe = r ∫_0^t J(s) ds
    # ∫ J = ∫ [1 + β(1-e^{-s/τc})] = t + β(t - τc(1-e^{-t/τc}))
    analytic = r * (t + beta * (t - tau_c * (1.0 - np.exp(-t / tau_c))))
    np.testing.assert_allclose(FE, analytic, rtol=1e-6, atol=1e-6)


def test_sls_timestep_refinement():
    g_inf = 0.6
    tau_r = 0.25
    t_end = 2.0
    # Smooth FVE
    def run(n):
        t = np.linspace(0.0, t_end, n)
        FVE = np.column_stack([np.sin(2 * np.pi * t), 0.5 * t])
        mat = create_material(SLSConfig(g_inf=g_inf, tau_r=tau_r))
        return t, mat.evaluate_history(t, FVE)

    _, fe_coarse = run(40)
    t_fine, fe_fine = run(640)
    # Interpolate fine onto coarse grid endpoints comparison at shared times.
    t_c = np.linspace(0.0, t_end, 40)
    fe_f_on_c = np.vstack(
        [np.interp(t_c, t_fine, fe_fine[:, j]) for j in range(2)]
    ).T
    err_coarse = np.max(np.abs(fe_coarse - fe_f_on_c))
    _, fe_mid = run(160)
    t_m = np.linspace(0.0, t_end, 160)
    fe_f_on_m = np.vstack(
        [np.interp(t_m, t_fine, fe_fine[:, j]) for j in range(2)]
    ).T
    err_mid = np.max(np.abs(fe_mid - fe_f_on_m))
    assert err_mid < err_coarse


def test_sls_extreme_dt_over_tau():
    mat = SLSMaterial(g_inf=0.3, tau_r=1.0, n_components=1)
    # Very small Δt/τ
    t_small = np.array([0.0, 1e-12, 2e-12])
    FVE = np.array([[0.0], [1.0], [1.0]])
    FE = mat.evaluate_history(t_small, FVE)
    assert np.all(np.isfinite(FE))
    # Very large Δt/τ: should approach 1/g_inf asymptote after one huge hold.
    mat.reset()
    t_large = np.array([0.0, 1e-9, 1e6])
    FVE2 = np.array([[0.0], [2.0], [2.0]])
    FE2 = mat.evaluate_history(t_large, FVE2)
    assert FE2[-1, 0] == pytest.approx(2.0 / 0.3, rel=1e-10)


def test_sls_forward_inverse_roundtrip():
    g_inf = 0.55
    tau_r = 0.8
    mat = SLSMaterial(g_inf=g_inf, tau_r=tau_r, n_components=2)
    t = np.linspace(0.0, 4.0, 400)
    FE_true = np.column_stack([np.sin(t), 0.2 * t**2])
    FVE = mat.forward_history(t, FE_true)
    FE_rec = mat.evaluate_history(t, FVE, reset=True)
    np.testing.assert_allclose(FE_rec, FE_true, rtol=1e-5, atol=1e-5)
