"""Fractional viscoelastic mapper tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn

from compliance_fem.viscoelasticity import FractionalConfig, create_material
from compliance_fem.viscoelasticity.fractional import FractionalMaterial


def _analytic_power_law_Fe(t, A, p, E0, T, alpha):
    """Fe for FVE = A t^p using the Beta-function identity."""
    coeff = (alpha * A / (E0 * T**alpha)) * beta_fn(alpha, p + 1.0)
    return coeff * np.power(t, alpha + p)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_fractional_power_law_against_analytic(alpha):
    E0, T, A, p = 1.0, 1.0, 1.5, 1.0
    mat = FractionalMaterial(
        E0=E0,
        T=T,
        alpha=alpha,
        num_modes=80,
        tau_min=1e-6,
        tau_max=1e5,
        n_components=1,
    )
    t = np.linspace(0.0, 2.0, 400)
    FVE = (A * np.power(t, p)).reshape(-1, 1)
    FE = mat.evaluate_history(t, FVE)[:, 0]
    analytic = _analytic_power_law_Fe(t, A, p, E0, T, alpha)
    # Relative error away from t=0 (kernel singularity / early transient).
    mask = t > 0.2
    rel = np.max(np.abs(FE[mask] - analytic[mask]) / np.maximum(np.abs(analytic[mask]), 1e-12))
    assert rel < 0.05


def test_fractional_mode_convergence():
    E0, T, alpha, A, p = 1.0, 1.0, 0.5, 1.0, 1.0
    t = np.linspace(0.0, 1.5, 300)
    FVE = (A * t**p).reshape(-1, 1)
    analytic = _analytic_power_law_Fe(t, A, p, E0, T, alpha)
    mask = t > 0.25
    errs = []
    for M in (8, 16, 32, 64):
        mat = FractionalMaterial(
            E0=E0, T=T, alpha=alpha, num_modes=M, tau_min=1e-6, tau_max=1e5, n_components=1
        )
        FE = mat.evaluate_history(t, FVE)[:, 0]
        errs.append(float(np.max(np.abs(FE[mask] - analytic[mask]))))
    assert errs[-1] < errs[0]
    assert errs[-1] < 0.05 * np.max(np.abs(analytic[mask]))


def test_fractional_creep_scaling_t_alpha():
    """For a unit step in FVE, Fe ~ (1/E0)(t/T)^α (approx via short ramp)."""
    E0, T, alpha = 2.0, 1.0, 0.4
    mat = FractionalMaterial(
        E0=E0, T=T, alpha=alpha, num_modes=48, tau_min=1e-5, tau_max=1e4, n_components=1
    )
    t_ramp = 1e-8
    t = np.concatenate([[0.0, t_ramp], np.geomspace(1e-4, 5.0, 200)])
    FVE = np.ones((t.size, 1))
    FVE[0, 0] = 0.0
    FE = mat.evaluate_history(t, FVE)[:, 0]
    analytic = (1.0 / E0) * (t / T) ** alpha
    mask = (t > 1e-3) & (t < 2.0)
    # Compare shape via ratio to t^α (allow SoE bias as nearly constant factor).
    ratio = FE[mask] / np.maximum(analytic[mask], 1e-30)
    assert np.std(ratio) / np.mean(ratio) < 0.15


def test_fractional_zero_and_components():
    mat = create_material(
        FractionalConfig(E0=1.0, T=1.0, alpha=0.5, num_modes=16, n_components=2)
    )
    t = np.linspace(0.0, 1.0, 20)
    FE = mat.evaluate_history(t, np.zeros((t.size, 2)))
    np.testing.assert_allclose(FE, 0.0, atol=1e-14)


def test_fractional_gamma_form_consistency():
    """Cross-check Beta form vs Γ(1+α) I^α definition for p=0 step density."""
    alpha = 0.5
    # For FVE = A (constant after 0), integral form with Beta:
    # Fe = α A /(E0 T^α) * t^α * B(α, 1) = A/(E0 T^α) t^α
    # since α B(α,1) = α Γ(α)Γ(1)/Γ(α+1) = Γ(α+1)/Γ(α+1) = 1.
    A = 1.0
    t = 2.5
    E0 = T = 1.0
    assert _analytic_power_law_Fe(np.array([t]), A, 0.0, E0, T, alpha)[0] == pytest.approx(
        (A / (E0 * T**alpha)) * t**alpha, rel=1e-12
    )
    # Γ form: Γ(1+α)/(E0 T^α) * A t^α / Γ(1+α) = same.
    assert (gamma_fn(1 + alpha) / (E0 * T**alpha)) * (
        A * t**alpha / gamma_fn(1 + alpha)
    ) == pytest.approx((A / (E0 * T**alpha)) * t**alpha)
