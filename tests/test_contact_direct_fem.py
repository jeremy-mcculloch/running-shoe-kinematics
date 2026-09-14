"""Direct FEM cross-checks for contact-edge boundary solutions."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import LayeredPlateConfig, ProblemConfig
from compliance_fem.contact_basis import N_BASIS_MODES
from compliance_fem.contact_direct_fem import run_standard_direct_comparisons

N_EXPECTED = 3 * N_BASIS_MODES


def _assert_fy_agrees(c, atol_rel: float = 1e-8) -> None:
    """Relative Fy error is ill-defined when both sides are ~0 (e.g. full Bx/Bry)."""
    n_t = c.f_t_compliance.size // 2
    fy_c = float(np.sum(c.f_t_compliance[n_t:]))
    fy_f = float(np.sum(c.f_t_fem[n_t:]))
    scale = max(abs(fy_c), abs(fy_f), 1.0)
    assert abs(fy_c - fy_f) / scale < atol_rel


@pytest.fixture(scope="module")
def compliance_result():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=16, ny=8, order=1)
    return compute_compliance(config)


def test_layered_direct_fem_agrees_with_compliance() -> None:
    config = LayeredPlateConfig(
        L=0.30,
        h1_heel=0.025,
        h1_toe=0.015,
        h2_heel=0.020,
        h2_toe=0.030,
        E1=2.0e6,
        nu1=0.30,
        E_heel=5.0e5,
        E_toe=1.5e6,
        nu2=0.30,
        EI_plate=10.0,
        nx=6,
        ny1=2,
        ny2=2,
        element_order=1,
    )
    result = compute_compliance(config)
    comps = run_standard_direct_comparisons(result, a=0.18, kappa=30.0)
    assert len(comps) == N_EXPECTED
    for c in comps:
        assert c.top_force_rel_error < 1e-6
        assert c.contact_reaction_rel_error < 1e-6
        assert c.free_gap_rel_error < 1e-6
        assert c.free_traction_rel < 1e-5


def test_direct_fem_agrees_with_compliance(compliance_result) -> None:
    comps = run_standard_direct_comparisons(compliance_result, a=0.6, kappa=30.0)
    assert len(comps) == N_EXPECTED
    for c in comps:
        assert c.top_force_rel_error < 1e-8
        assert c.contact_reaction_rel_error < 1e-8
        assert c.free_gap_rel_error < 1e-8
        _assert_fy_agrees(c, atol_rel=1e-8)
        assert c.M_rel_error < 1e-8
        assert c.T_toe_rel_error < 1e-8
        assert c.free_traction_rel < 1e-6


def test_all_five_basis_modes_are_compared(compliance_result) -> None:
    comps = run_standard_direct_comparisons(compliance_result, a=0.6, kappa=30.0)
    assert sorted({c.basis_index for c in comps}) == list(range(N_BASIS_MODES))
