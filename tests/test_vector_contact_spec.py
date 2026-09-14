"""Spec tests for vector compliance, five-mode lookup, and 2x2 force control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    yy_block,
)
from compliance_fem.compliance import compute_compliance
from compliance_fem.config import LayeredPlateConfig, ProblemConfig
from compliance_fem.contact_direct_fem import compare_compliance_to_direct_fem, pick_candidate_near
from compliance_fem.contact_basis import N_BASIS_MODES
from compliance_fem.contact_topology import ContactRecordSpec, ContactType
from compliance_fem.contact_lookup import (
    MODE_BY,
    SCALAR_FY,
    SCALAR_GAP,
    SCALAR_RX,
    SCALAR_RY,
    compute_toe_moment,
    from_compliance_result,
    generate_contact_lookup,
    load_force_compliance,
    ramp_horizontal_lever,
)
from compliance_fem.corotation import basis_coefficients
from compliance_fem.force_control import Tolerances, evaluate_candidates, select_contact_candidate


@pytest.fixture(scope="module")
def rectangle_result():
    return compute_compliance(ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=12, ny=6, order=1))


@pytest.fixture(scope="module")
def rectangle_lookup(rectangle_result):
    return generate_contact_lookup(from_compliance_result(rectangle_result), a=0.6, kappa=30.0)


def test_xy_coupling_reciprocity(rectangle_result) -> None:
    n_t = len(rectangle_result.x_top)
    Ctt = rectangle_result.Ctt_force
    Cbb = rectangle_result.Cbb_force
    Ctb = rectangle_result.Ctb_force
    Cbt = rectangle_result.Cbt_force
    Ctt_xy = Ctt[:n_t, n_t:]
    Ctt_yx = Ctt[n_t:, :n_t]
    assert np.linalg.norm(Ctt_xy - Ctt_yx.T) / max(np.linalg.norm(Ctt_xy), 1e-30) < 1e-8
    assert np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30) < 1e-8
    n_b = len(rectangle_result.x_bottom)
    Cbb_xy = Cbb[:n_b, n_b:]
    Cbb_yx = Cbb[n_b:, :n_b]
    assert np.linalg.norm(Cbb_xy - Cbb_yx.T) / max(np.linalg.norm(Cbb_xy), 1e-30) < 1e-8


def test_gauge_and_plate_constraints_on_compliance_columns(rectangle_result) -> None:
    assert rectangle_result.solve_residuals["top"] < 1e-8
    assert rectangle_result.rigid_mode_error < 1e-8


def test_fx_fy_reconstruction_and_superposition(rectangle_lookup) -> None:
    ev = evaluate_candidates(rectangle_lookup, 400.0, -8.0e3, phi_deg=-3.0, theta_deg=2.0)
    well = np.flatnonzero(ev.well_conditioned)
    assert well.size
    row = int(well[len(well) // 2])
    assert ev.Fx[row] == pytest.approx(400.0, rel=1e-8, abs=1e-6)
    assert ev.Fy[row] == pytest.approx(-8.0e3, rel=1e-8, abs=1e-6)
    gamma = basis_coefficients(ev.alpha, ev.d_lx[row], ev.d_ly[row], ev.varphi)
    np.testing.assert_allclose(
        ev.full_bottom_reaction_y[row],
        gamma @ rectangle_lookup.reaction_y_basis[row],
        rtol=1e-10,
        atol=1e-12,
    )


def test_gap_and_reaction_edge_scalars(rectangle_lookup) -> None:
    row = len(rectangle_lookup.candidate_indices) // 2
    for k in range(N_BASIS_MODES):
        g = rectangle_lookup.scalar_lookup[row, k, SCALAR_GAP]
        rx = rectangle_lookup.scalar_lookup[row, k, SCALAR_RX]
        ry = rectangle_lookup.scalar_lookup[row, k, SCALAR_RY]
        assert np.isfinite(g) and np.isfinite(rx) and np.isfinite(ry)


def test_selection_ignores_tangential_reaction(rectangle_lookup) -> None:
    ev = evaluate_candidates(rectangle_lookup, 0.0, -5.0e3, phi_deg=-4.0, theta_deg=0.0)
    sel = select_contact_candidate(ev)
    assert sel.selected_row is not None
    ev.edge_contact_reaction_tangential[:] = np.nan
    ev.full_bottom_reaction_tangential[:] = np.nan
    ev.T_toe[:] = 1e12
    sel2 = select_contact_candidate(ev)
    assert sel2.selected_row == sel.selected_row


def test_flat_top_eta_is_zero() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = np.full(3, 0.2)
    a = 0.5
    eta = ramp_horizontal_lever(x, y, a, 0.2)
    np.testing.assert_allclose(eta, 0.0)
    f_tx = np.array([1.0, 2.0, 3.0])
    f_ty = np.array([0.0, 0.0, 4.0])
    T, T_vert, H_a = compute_toe_moment(f_tx, f_ty, x, y, a)
    assert H_a == pytest.approx(0.2)
    assert T == pytest.approx(T_vert)
    assert T == pytest.approx(0.5 * 4.0)


def test_sloped_top_eta_enters_toe_moment() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.1, 0.2, 0.3])
    a = 0.5
    f_tx = np.array([0.0, 0.0, 2.0])
    f_ty = np.array([0.0, 0.0, 3.0])
    T, T_vert, H_a = compute_toe_moment(f_tx, f_ty, x, y, a)
    assert H_a == pytest.approx(0.2)
    assert T_vert == pytest.approx(0.5 * 3.0)
    assert T == pytest.approx(0.5 * 3.0 - 0.1 * 2.0)


def test_illconditioned_2x2_detection(rectangle_lookup) -> None:
    original = rectangle_lookup.kf_matrix
    try:
        rectangle_lookup.kf_matrix = np.zeros_like(original)
        ev = evaluate_candidates(rectangle_lookup, 0.0, -1.0e4, phi_deg=0.0, theta_deg=0.0)
        assert not np.any(ev.well_conditioned)
        assert np.all(ev.kf_illconditioned)
        sel = select_contact_candidate(ev)
        assert sel.selected_row is None
    finally:
        rectangle_lookup.kf_matrix = original


def test_load_force_compliance_rejects_vertical_only(tmp_path: Path, rectangle_result) -> None:
    n_t = len(rectangle_result.x_top)
    n_b = len(rectangle_result.x_bottom)
    path = tmp_path / "old_vertical.npz"
    np.savez_compressed(
        path,
        Ctt_force=np.eye(n_t),
        Ctb_force=np.zeros((n_t, n_b)),
        Cbt_force=np.zeros((n_b, n_t)),
        Cbb_force=np.eye(n_b),
        x_top=rectangle_result.x_top,
        x_bottom=rectangle_result.x_bottom,
        L=1.0,
        H=0.5,
        E=1.0e6,
        nu=0.3,
    )
    with pytest.raises(ValueError, match="vector"):
        load_force_compliance(path)


def test_direct_fem_vector_stick_and_free_traction(rectangle_result) -> None:
    i = pick_candidate_near(rectangle_result.x_bottom, 0.5 * rectangle_result.config.L)
    spec = ContactRecordSpec(ContactType.TOE, i)
    for k in range(N_BASIS_MODES):
        comp = compare_compliance_to_direct_fem(rectangle_result, spec, k, a=0.6, kappa=30.0)
        assert comp.top_force_rel_error < 1e-7
        assert comp.contact_reaction_rel_error < 1e-7
        assert comp.free_gap_rel_error < 1e-7
        assert comp.free_traction_rel < 1e-6
        assert comp.T_toe_rel_error < 1e-7


def test_layered_vector_blocks_and_lookup() -> None:
    cfg = LayeredPlateConfig(
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
    result = compute_compliance(cfg)
    n_t = len(result.x_top)
    n_b = len(result.x_bottom)
    assert result.dof_ordering == DOF_ORDERING_COMPONENT_MAJOR_UV
    assert result.Ctt_force.shape == (2 * n_t, 2 * n_t)
    assert result.inextensibility_residuals["top"] < 1e-8
    assert result.gauge_residuals["top"] < 1e-8
    lookup = generate_contact_lookup(from_compliance_result(result), a=0.18, kappa=30.0)
    assert lookup.scalar_lookup.shape[1] == N_BASIS_MODES
    ev = evaluate_candidates(
        lookup, 0.0, -1.0e3, phi_deg=-2.0, theta_deg=1.0, tolerances=Tolerances()
    )
    assert np.any(ev.well_conditioned)
    assert yy_block(result.Ctt_force, n_t, n_t).shape == (n_t, n_t)
    assert yy_block(result.Cbb_force, n_b, n_b).shape == (n_b, n_b)


def test_fx_star_zero_need_not_imply_zero_horizontal_translation(rectangle_lookup) -> None:
    ev = evaluate_candidates(rectangle_lookup, 0.0, -1.0e4, phi_deg=0.0, theta_deg=0.0)
    well = np.flatnonzero(ev.well_conditioned)
    assert well.size
    kf = rectangle_lookup.kf_matrix[well[0]]
    # Horizontal/vertical coupling: a purely vertical requested force still
    # needs a nonzero d_lx whenever the off-diagonal coupling is significant.
    if abs(kf[0, 1]) > 1e-12 * max(abs(kf[1, 1]), 1.0):
        assert np.nanmax(np.abs(ev.d_lx[well])) > 0.0
    # And a purely vertical contact translation generally produces some Fx.
    assert np.any(np.abs(rectangle_lookup.scalar_lookup[:, MODE_BY, SCALAR_FY]) > 0.0)
