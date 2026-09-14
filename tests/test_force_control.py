"""Tests for force-controlled evaluation and contact selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_basis import N_BASIS_MODES, shape_mode_phi1
from compliance_fem.contact_lookup import (
    N_SCALAR_FIELDS,
    SCALAR_GAP,
    SCALAR_MV,
    SCALAR_RY,
    SCALAR_TOE,
    SCALAR_TOE_VERT,
    compute_toe_basis,
    from_compliance_result,
    generate_contact_lookup,
    ramp_virtual_displacement,
)
from compliance_fem.corotation import basis_coefficients
from compliance_fem.force_control import (
    Tolerances,
    angles_to_coefficients,
    evaluate_candidates,
    evaluate_from_angles,
    export_evaluation_csv,
    export_selected_result,
    select_contact_candidate,
    top_displacement_vector,
)
from compliance_fem.shape_render import build_shape_plot_data, choose_display_scale


@pytest.fixture(scope="module")
def lookup():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=16, ny=8, order=1)
    result = compute_compliance(config)
    return generate_contact_lookup(from_compliance_result(result), a=0.6, kappa=30.0)


def test_angles_to_coefficients() -> None:
    varphi, alpha, r_x, r_y = angles_to_coefficients(0.0, 0.0)
    assert varphi == pytest.approx(0.0)
    assert alpha == pytest.approx(0.0)
    assert r_x == pytest.approx(0.0)
    assert r_y == pytest.approx(0.0)

    varphi, alpha, r_x, r_y = angles_to_coefficients(45.0, 45.0)
    assert varphi == pytest.approx(0.25 * np.pi)
    assert alpha == pytest.approx(1.0)
    assert r_x == pytest.approx(np.cos(0.25 * np.pi) - 1.0)
    assert r_y == pytest.approx(-np.sin(0.25 * np.pi))

    # phi_ref shifts only the rotation, never the shape amplitude.
    varphi, alpha, _, _ = angles_to_coefficients(30.0, 30.0, phi_ref=np.deg2rad(10.0))
    assert varphi == pytest.approx(np.deg2rad(20.0))
    assert alpha == pytest.approx(1.0 / np.sqrt(3.0))


def test_toe_moment_ramp_dot_product() -> None:
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    a = 0.5
    rho = ramp_virtual_displacement(x, a)
    np.testing.assert_allclose(rho, [0.0, 0.0, 0.0, 0.25, 0.5])
    f = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    assert compute_toe_basis(f, x, a) == pytest.approx(0.25 * 2.0 + 0.5 * 3.0)


def test_lookup_schema_has_toe(lookup) -> None:
    assert lookup.schema_version == 6
    assert lookup.scalar_lookup.shape[-1] == N_SCALAR_FIELDS
    assert lookup.top_force_y_basis is not None
    assert lookup.top_force_y_basis.shape == (
        len(lookup.candidate_indices),
        N_BASIS_MODES,
        len(lookup.x_top),
    )
    for row in range(len(lookup.candidate_indices)):
        for k in range(N_BASIS_MODES):
            expected = compute_toe_basis(
                lookup.top_force_y_basis[row, k],
                lookup.x_top,
                lookup.softplus_a,
            )
            assert lookup.scalar_lookup[row, k, SCALAR_TOE_VERT] == pytest.approx(expected)


def test_translation_and_force_reconstruction(lookup) -> None:
    Fy_star = -2.0e4
    Fx_star = 5.0e2
    ev = evaluate_candidates(lookup, Fx_star, Fy_star, phi_deg=-3.0, theta_deg=5.0)
    assert np.any(ev.well_conditioned)
    for row in np.flatnonzero(ev.well_conditioned):
        assert ev.Fy[row] == pytest.approx(Fy_star, rel=1e-8, abs=1e-6)
        assert ev.Fx[row] == pytest.approx(Fx_star, rel=1e-8, abs=1e-6)
        assert np.isfinite(ev.d_lx[row])
        assert np.isfinite(ev.d_ly[row])


def test_reconstructed_scalars_and_fields(lookup) -> None:
    ev = evaluate_candidates(lookup, 0.0, -1.5e4, phi_deg=-2.0, theta_deg=4.0)
    row = int(np.flatnonzero(ev.well_conditioned)[len(ev.l) // 2])
    gamma = basis_coefficients(ev.alpha, ev.d_lx[row], ev.d_ly[row], ev.varphi)
    np.testing.assert_allclose(gamma, [ev.alpha, ev.d_lx[row], ev.d_ly[row], ev.r_x, ev.r_y])

    assert ev.M[row] == pytest.approx(lookup.scalar_lookup[row, :, SCALAR_MV] @ gamma)
    assert ev.T_toe[row] == pytest.approx(lookup.scalar_lookup[row, :, SCALAR_TOE] @ gamma)
    # Local per-node fields are pure superpositions of the stored bases.
    np.testing.assert_allclose(
        ev.full_bottom_reaction_y[row], gamma @ lookup.reaction_y_basis[row], atol=1e-12
    )
    np.testing.assert_allclose(
        ev.full_bottom_reaction_x[row], gamma @ lookup.reaction_x_basis[row], atol=1e-12
    )
    np.testing.assert_allclose(
        ev.top_v[row],
        lookup.basis_top_displacements[len(lookup.x_top) :, :] @ gamma,
        atol=1e-14,
    )
    np.testing.assert_allclose(ev.top_u[row], 0.0, atol=1e-14)
    # The stored scalar edge values are the local (rotating-frame) components.
    free_last = int(lookup.candidate_indices[row]) - 1
    assert lookup.scalar_lookup[row, :, SCALAR_GAP] @ gamma == pytest.approx(
        ev.full_bottom_v[row][free_last]
    )
    assert lookup.scalar_lookup[row, :, SCALAR_RY] @ gamma == pytest.approx(
        ev.full_bottom_reaction_y[row][int(lookup.candidate_indices[row])]
    )


def test_x_cm_and_nan_when_fy_zero(lookup) -> None:
    ev = evaluate_candidates(lookup, 0.0, -1.0e4, 0.0, 0.0)
    row = int(np.flatnonzero(ev.well_conditioned)[0])
    # Center of effort is formed from fixed-frame top nodal forces/positions.
    assert np.isfinite(ev.x_cm[row])
    ev0 = evaluate_candidates(lookup, 1.0e3, 0.0, 3.0, 1.0)
    assert np.all(np.isnan(ev0.x_cm))


def test_small_singular_value_excluded(lookup) -> None:
    tol = Tolerances(F_abs_tol=1e20, F_rel_tol=0.0)
    ev = evaluate_candidates(lookup, 0.0, -1.0e4, 0.0, 0.0, tolerances=tol)
    assert not np.any(ev.well_conditioned)
    sel = select_contact_candidate(ev, tol)
    assert sel.selected_row is None


def test_selection_ignores_moment_and_tangential_reaction(lookup) -> None:
    """Admissibility uses only fixed-frame normal gap / normal reaction."""
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -5.0, 2.0)
    assert sel.selected_row is not None
    ev = sel.evaluation
    ev.M[:] = 1e9
    ev.edge_contact_reaction_tangential[:] = 1e9
    ev.T_toe[:] = -1e9
    sel2 = select_contact_candidate(ev)
    assert sel2.selected_row == sel.selected_row


def test_violation_score_normalized(lookup) -> None:
    ev = evaluate_candidates(lookup, 0.0, -1.0e4, 0.0, 0.0)
    assert np.all(ev.violation_score[ev.well_conditioned] >= -1e-15)
    assert ev.g_scale > 0 and ev.R_scale > 0


def test_multiple_and_none_admissible_behavior(lookup) -> None:
    tol = Tolerances(tau_g=1e6, tau_R=1e6)
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, 0.0, 0.0, tolerances=tol)
    assert sel.n_admissible >= 1
    assert sel.exactly_admissible
    assert not sel.approximate

    tol2 = Tolerances(tau_g=0.0, tau_R=0.0)
    sel2 = evaluate_from_angles(lookup, 0.0, 1.0e4, 0.0, 0.0, tolerances=tol2)
    if sel2.n_admissible == 0:
        assert sel2.approximate
        assert sel2.selected_row is not None


def test_export_csv_and_npz(lookup, tmp_path: Path) -> None:
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -4.0, 3.0)
    csv_path = export_evaluation_csv(sel, tmp_path / "table.csv")
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    for column in ("T_toe", "well_conditioned", "kf_cond", "x_l_rot", "d_lx", "d_ly"):
        assert column in text
    npz_path = export_selected_result(sel, lookup, tmp_path / "selected.npz")
    data = np.load(npz_path)
    for key in ("top_v", "bottom_gap", "d_lx", "d_ly", "x_l_rot", "varphi", "phi_ref"):
        assert key in data.files


def test_shape_plot_data_without_streamlit(lookup) -> None:
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -5.0, 2.0)
    shape = build_shape_plot_data(lookup, sel, scale_mode="true")
    assert shape.scale == 1.0
    assert shape.y_top_def.shape == shape.x_dense.shape
    assert shape.x_top_def.shape == shape.x_dense.shape
    assert shape.chord_x.shape == (2,)
    s, label = choose_display_scale(lookup.H, np.array([1e-3]), np.array([1e-3]), "auto", 1.0)
    assert s >= 1.0
    assert "auto" in label or "exaggerated" in label


def test_top_displacement_vector_matches_basis(lookup) -> None:
    alpha = 0.15
    u, v = top_displacement_vector(
        lookup.x_top, alpha, lookup.L, lookup.softplus_a, lookup.softplus_kappa
    )
    gamma = basis_coefficients(alpha, 0.0, 0.0, 0.0)
    expected = lookup.basis_top_displacements[len(lookup.x_top) :, :] @ gamma
    np.testing.assert_allclose(v, expected, atol=1e-15)
    np.testing.assert_allclose(u, 0.0, atol=0.0)
    np.testing.assert_allclose(
        v,
        alpha * shape_mode_phi1(lookup.x_top, lookup.L, lookup.softplus_a, lookup.softplus_kappa),
        atol=1e-15,
    )
