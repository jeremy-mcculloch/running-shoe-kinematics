"""Fung viscoelastic mapper tests."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.viscoelasticity import FungConfig, create_material
from compliance_fem.viscoelasticity.exponential import fung_prony_weights
from compliance_fem.viscoelasticity.fung import (
    FungMaterial,
    fung_G_analytic,
    fung_G_prony,
)


def test_fung_G0_normalization():
    C, tau1, tau2 = 0.2, 1e-3, 1e2
    mat = FungMaterial(C=C, tau1=tau1, tau2=tau2, num_modes=40, n_components=1)
    G0 = mat.g_inf + float(np.sum(mat.spectrum.weight))
    assert G0 == pytest.approx(1.0, rel=1e-12)
    assert fung_G_analytic(0.0, C, tau1, tau2) == pytest.approx(1.0)


def test_fung_discrete_spectrum_matches_analytic_G():
    C, tau1, tau2 = 0.15, 1e-4, 1e3
    t = np.geomspace(1e-5, 1e2, 80)
    G_ref = fung_G_analytic(t, C, tau1, tau2)
    errs = []
    for M in (8, 16, 32, 64):
        g_inf, h, tau = fung_prony_weights(C, tau1, tau2, M)
        G_approx = fung_G_prony(t, g_inf, h, tau)
        errs.append(float(np.max(np.abs(G_approx - G_ref))))
    assert errs[-1] < errs[0]
    assert errs[-1] < 0.02


def test_fung_decades_span():
    mat = FungMaterial(C=0.1, tau1=1e-6, tau2=1e4, num_modes=48, n_components=1)
    assert mat.spectrum.tau[0] == pytest.approx(1e-6)
    assert mat.spectrum.tau[-1] == pytest.approx(1e4)
    assert mat.g_inf == pytest.approx(1.0 / (1.0 + 0.1 * np.log(1e4 / 1e-6)))


def test_fung_forward_inverse_roundtrip():
    mat = FungMaterial(C=0.12, tau1=1e-3, tau2=10.0, num_modes=48, n_components=2)
    t = np.linspace(0.0, 5.0, 500)
    FE_true = np.column_stack([np.sin(1.3 * t), 0.4 * (1.0 - np.cos(0.7 * t))])
    FVE = mat.forward_history(t, FE_true)
    FE_rec = mat.evaluate_history(t, FVE, reset=True)
    np.testing.assert_allclose(FE_rec, FE_true, rtol=1e-4, atol=1e-4)


def test_fung_zero_history():
    mat = create_material(FungConfig(C=0.2, tau1=0.01, tau2=10.0, num_modes=16))
    t = np.linspace(0.0, 1.0, 25)
    FE = mat.evaluate_history(t, np.zeros((t.size, 2)))
    np.testing.assert_allclose(FE, 0.0, atol=1e-14)


def test_fung_step_long_time_asymptote():
    """As t→∞, J→1/g_inf so Fe → FVE/g_inf for a held step."""
    C, tau1, tau2 = 0.25, 1e-3, 1.0
    mat = FungMaterial(C=C, tau1=tau1, tau2=tau2, num_modes=40, n_components=1)
    t = np.concatenate([[0.0], np.linspace(1e-8, 50.0 * tau2, 600)])
    FVE = np.ones((t.size, 1))
    FVE[0] = 0.0
    FE = mat.evaluate_history(t, FVE)
    assert FE[-1, 0] == pytest.approx(1.0 / mat.g_inf, rel=2e-3)
