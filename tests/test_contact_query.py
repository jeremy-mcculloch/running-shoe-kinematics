"""Raw-coefficient superposition over interval records (debug query path)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.contact_lookup import SCALAR_FY, SCALAR_MV
from compliance_fem.contact_query import center_of_effort, select_candidate, superpose, varphi_from_gamma
from compliance_fem.corotation import basis_coefficients, contract_basis


def test_exact_algebraic_superposition(rocker_lookup) -> None:
    lk = rocker_lookup
    gamma = basis_coefficients(0.03, 1e-4, -3e-4, np.deg2rad(2.0))
    prof = superpose(lk, gamma)
    for row in (0, lk.row_of(4, 8), lk.n_records - 1):
        S = lk.scalar_lookup[row]
        assert prof.Fy[row] == pytest.approx(float(contract_basis(S[:, SCALAR_FY], gamma)), rel=1e-12)
        assert prof.M[row] == pytest.approx(float(contract_basis(S[:, SCALAR_MV], gamma)), rel=1e-12)
    assert prof.gap.shape == (lk.n_records, lk.n_bottom_nodes)


def test_superpose_rejects_wrong_coefficient_count(rocker_lookup) -> None:
    with pytest.raises(ValueError):
        superpose(rocker_lookup, [0.0, 0.0, 0.0, 0.0])


def test_varphi_roundtrip() -> None:
    for deg in (-40.0, 0.0, 7.5, 80.0):
        gamma = basis_coefficients(0.0, 0.0, 0.0, np.deg2rad(deg))
        assert varphi_from_gamma(gamma) == pytest.approx(np.deg2rad(deg), abs=1e-15)


def test_center_of_effort_nan_when_fy_zero() -> None:
    assert np.isnan(center_of_effort(0.0, 1.0))
    np.testing.assert_allclose(center_of_effort(np.array([2.0, 0.0]), np.array([1.0, 1.0])), [0.5, np.nan])


def test_selection_reports_interval_and_admissibility(rocker_lookup) -> None:
    lk = rocker_lookup
    gamma = basis_coefficients(0.0, 0.0, -1e-4, 0.0)
    sel = select_candidate(lk, gamma)
    assert sel.contact_start_index <= sel.contact_end_index
    assert sel.contact_type in ("heel", "toe", "full", "interior")
    assert sel.violation >= 0.0
    assert np.all(sel.profile.violation[lk.valid_rows] >= 0.0)
    if sel.admissible_rows.size:
        assert sel.exactly_admissible and sel.violation == 0.0
