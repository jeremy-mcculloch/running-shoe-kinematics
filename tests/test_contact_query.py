"""Tests for contact-lookup superposition and admissibility."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_basis import N_BASIS_MODES
from compliance_fem.contact_lookup import (
    SCALAR_FY,
    SCALAR_MV,
    from_compliance_result,
    generate_contact_lookup,
)
from compliance_fem.contact_query import (
    center_of_effort,
    evaluate_admissibility,
    select_candidate,
    superpose,
)


@pytest.fixture(scope="module")
def lookup():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=16, ny=8, order=1)
    result = compute_compliance(config)
    return generate_contact_lookup(from_compliance_result(result), a=0.6, kappa=30.0)


def test_exact_algebraic_superposition(lookup) -> None:
    gamma = np.array([0.7, -0.2, 0.4, -0.005, 0.05])
    profile = superpose(lookup, gamma)
    expected_fy = sum(gamma[k] * lookup.scalar_lookup[:, k, SCALAR_FY] for k in range(N_BASIS_MODES))
    expected_m = sum(gamma[k] * lookup.scalar_lookup[:, k, SCALAR_MV] for k in range(N_BASIS_MODES))
    expected_g = sum(gamma[k] * lookup.gap_basis[:, k, :] for k in range(N_BASIS_MODES))
    expected_r = sum(gamma[k] * lookup.reaction_basis[:, k, :] for k in range(N_BASIS_MODES))
    np.testing.assert_allclose(profile.Fy, expected_fy)
    np.testing.assert_allclose(profile.M, expected_m)
    np.testing.assert_allclose(profile.gap, expected_g)
    np.testing.assert_allclose(profile.reaction, expected_r)


def test_superpose_rejects_wrong_coefficient_count(lookup) -> None:
    with pytest.raises(ValueError, match="five entries|5 entries"):
        superpose(lookup, [1.0, 0.0, 0.0, 0.0])


def test_combined_center_of_effort(lookup) -> None:
    gamma = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    profile = superpose(lookup, gamma)
    expected = profile.M / profile.Fy
    np.testing.assert_allclose(profile.x_ce, expected)
    gamma2 = np.array([1.0, 0.5, 0.2, 0.0, 0.1])
    p2 = superpose(lookup, gamma2)
    naive = np.zeros_like(p2.Fy)
    for k in range(N_BASIS_MODES):
        fy_k = lookup.scalar_lookup[:, k, SCALAR_FY]
        m_k = lookup.scalar_lookup[:, k, SCALAR_MV]
        naive += gamma2[k] * (m_k / fy_k)
    assert not np.allclose(p2.x_ce, naive)


def test_nan_center_of_effort_when_fy_zero() -> None:
    assert np.isnan(center_of_effort(0.0, 1.0))
    out = center_of_effort(np.array([0.0, 2.0]), np.array([1.0, 4.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(2.0)


def test_admissibility_and_selection(lookup) -> None:
    # Pure upward contact translation typically opens gaps / may not be admissible;
    # still returns a least-violating candidate.
    gamma = [0.0, 0.0, 1.0, 0.0, 0.0]
    selected = select_candidate(lookup, gamma, tau_g=1e-8, tau_R=1e-8)
    assert selected.candidate_row >= 0
    assert selected.gap.shape == (len(lookup.x_bottom),)
    assert selected.reaction.shape == (len(lookup.x_bottom),)
    assert np.isfinite(selected.violation)

    profile = evaluate_admissibility(lookup, superpose(lookup, gamma))
    assert profile.violation.shape == (len(lookup.candidate_indices),)
    assert profile.admissible.dtype == bool


def test_violation_score_nonnegative(lookup) -> None:
    profile = evaluate_admissibility(lookup, superpose(lookup, [0.2, 0.1, 0.5, -0.01, 0.1]))
    assert np.all(profile.violation >= -1e-15)
