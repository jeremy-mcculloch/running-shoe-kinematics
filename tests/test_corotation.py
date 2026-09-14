"""Tests for the top-attached co-rotating frame and the five-mode basis."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance, compute_layered_compliance
from compliance_fem.config import LayeredPlateConfig, ProblemConfig, RuntimeAngleConfig
from compliance_fem.contact_basis import (
    BASIS_MODE_NAMES,
    MODE_BRX,
    MODE_BRY,
    MODE_BX,
    MODE_BY,
    MODE_PHI1,
    N_BASIS_MODES,
    build_contact_displacement_matrix,
    build_top_displacement_matrix,
    shape_mode_phi1,
)
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    N_SCALAR_FIELDS,
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_TOE,
    build_boundary_matrix,
    free_contact_sets,
    from_compliance_result,
    generate_contact_lookup,
    prepare_compliance_blocks,
    solve_candidate,
)
from compliance_fem.contact_topology import ContactMode, ContactRecordSpec, ContactType
from compliance_fem.corotation import (
    SHAPE_MODE_SIGN,
    basis_coefficients,
    contact_displacement,
    contract_basis,
    corotation_angle,
    fixed_frame_normal_component,
    reference_chord_angle,
    rotate_force_to_local,
    rotate_vector_to_fixed,
    rotation_coefficients,
    rotation_matrix,
    shape_amplitude,
    transform_to_fixed_frame,
)
from compliance_fem.force_control import (
    Tolerances,
    angles_to_coefficients,
    evaluate_candidates,
    evaluate_from_angles,
    select_contact_candidate,
)

A_SOFT = 0.5
KAPPA = 30.0
ANGLES_DEG = (-12.0, -5.0, -1.0, 0.0, 1.0, 5.0, 12.0)


@pytest.fixture(scope="module")
def compliance_result():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=12, ny=6, order=1)
    return compute_compliance(config)


@pytest.fixture(scope="module")
def lookup(compliance_result):
    blocks = from_compliance_result(compliance_result)
    return generate_contact_lookup(blocks, a=A_SOFT, kappa=KAPPA)


@pytest.fixture(scope="module")
def sloped_lookup():
    """Layered geometry whose reference top chord is deliberately not horizontal."""
    config = LayeredPlateConfig(
        L=0.25,
        h1_heel=0.025,
        h1_toe=0.010,
        h2_heel=0.020,
        h2_toe=0.030,
        E1=2.0e6,
        nu1=0.3,
        E_heel=4.0e5,
        E_toe=8.0e5,
        nu2=0.3,
        EI_plate=5.0,
        nx=10,
        ny1=3,
        ny2=3,
    )
    result = compute_layered_compliance(config)
    blocks = from_compliance_result(result)
    return generate_contact_lookup(blocks, a=0.15, kappa=30.0, reciprocity_tol=1e-6)


# 1. Shape mode vanishes at both endpoints.


def test_shape_mode_vanishes_at_endpoints() -> None:
    for L, a, kappa in ((1.0, 0.5, 30.0), (0.25, 0.15, 60.0), (2.0, 1.7, 8.0)):
        phi1 = shape_mode_phi1(np.array([0.0, L]), L=L, a=a, kappa=kappa)
        np.testing.assert_allclose(phi1, 0.0, atol=1e-14)


def test_top_displacement_matrix_endpoints_are_zero(lookup) -> None:
    n_t = len(lookup.x_top)
    W = lookup.basis_top_displacements
    v = W[n_t:, MODE_PHI1]
    assert abs(v[0]) < 1e-14
    assert abs(v[-1]) < 1e-14
    # Only the shape mode prescribes top displacement.
    np.testing.assert_allclose(W[:, MODE_BX:], 0.0, atol=0.0)
    # The top is clamped horizontally in the rotating frame.
    np.testing.assert_allclose(W[:n_t, :], 0.0, atol=0.0)


# 2. Exactly five basis modes in the documented order.


def test_five_basis_modes_in_documented_order(lookup) -> None:
    assert N_BASIS_MODES == 5
    assert BASIS_MODE_NAMES == (
        "top_shape_phi1",
        "contact_translation_x",
        "contact_translation_y",
        "contact_rotation_x",
        "contact_rotation_y",
    )
    assert (MODE_PHI1, MODE_BX, MODE_BY, MODE_BRX, MODE_BRY) == (0, 1, 2, 3, 4)
    assert lookup.scalar_lookup.shape == (
        len(lookup.candidate_indices),
        N_BASIS_MODES,
        N_SCALAR_FIELDS,
    )
    assert lookup.metadata["basis_order"] == list(BASIS_MODE_NAMES)
    assert lookup.metadata["n_shape_modes"] == 1
    assert lookup.metadata["shape_mode_normalization"] == "none"


def test_contact_mode_matrix_columns(lookup) -> None:
    x_c = np.linspace(0.4, 1.0, 7)
    l_i = 0.4
    W_c = build_contact_displacement_matrix(x_c, l_i)
    n = x_c.size
    assert W_c.shape == (2 * n, N_BASIS_MODES)
    np.testing.assert_allclose(W_c[:, MODE_PHI1], 0.0)
    np.testing.assert_allclose(W_c[:n, MODE_BX], 1.0)
    np.testing.assert_allclose(W_c[n:, MODE_BX], 0.0)
    np.testing.assert_allclose(W_c[:n, MODE_BY], 0.0)
    np.testing.assert_allclose(W_c[n:, MODE_BY], 1.0)
    np.testing.assert_allclose(W_c[:n, MODE_BRX], x_c - l_i)
    np.testing.assert_allclose(W_c[n:, MODE_BRX], 0.0)
    np.testing.assert_allclose(W_c[:n, MODE_BRY], 0.0)
    np.testing.assert_allclose(W_c[n:, MODE_BRY], x_c - l_i)


# 3. Lookup generation does not depend on phi.


def test_lookup_generation_takes_no_angle(compliance_result) -> None:
    import inspect

    names = set(inspect.signature(generate_contact_lookup).parameters)
    assert not (names & {"phi", "phi_deg", "varphi", "theta", "theta_deg"})
    names_solve = set(inspect.signature(solve_candidate).parameters)
    assert not (names_solve & {"phi", "phi_deg", "varphi", "theta", "theta_deg"})


def test_lookup_generation_is_reproducible(compliance_result) -> None:
    blocks = from_compliance_result(compliance_result)
    first = generate_contact_lookup(blocks, a=A_SOFT, kappa=KAPPA)
    second = generate_contact_lookup(blocks, a=A_SOFT, kappa=KAPPA)
    np.testing.assert_array_equal(first.scalar_lookup, second.scalar_lookup)
    np.testing.assert_array_equal(first.gap_u_basis, second.gap_u_basis)
    np.testing.assert_array_equal(first.reaction_x_basis, second.reaction_x_basis)


# 4. Rotation coefficients vanish at varphi = 0.


def test_rotation_coefficients_vanish_at_zero() -> None:
    r_x, r_y = rotation_coefficients(0.0)
    assert r_x == 0.0
    assert r_y == 0.0
    gamma = basis_coefficients(0.3, 1e-3, -2e-3, 0.0)
    np.testing.assert_allclose(gamma, [0.3, 1e-3, -2e-3, 0.0, 0.0])
    np.testing.assert_allclose(rotation_matrix(0.0), np.eye(2))


# 5 and 6. Contact displacement matches d_l + (Q^T - I)(X - X_l) and vanishes at l_i.


@pytest.mark.parametrize("varphi_deg", ANGLES_DEG)
def test_superposed_contact_displacement_matches_exact_rotation(varphi_deg: float) -> None:
    varphi = np.deg2rad(varphi_deg)
    l_i = 0.35
    x_c = np.linspace(l_i, 1.0, 9)
    d_l = np.array([1.3e-3, -4.1e-3])
    alpha = 0.21

    gamma = basis_coefficients(alpha, d_l[0], d_l[1], varphi)
    W_c = build_contact_displacement_matrix(x_c, l_i)
    superposed = contract_basis(W_c, gamma, mode_axis=1)
    n = x_c.size
    u_super, v_super = superposed[:n], superposed[n:]

    Q = rotation_matrix(varphi)
    for j, x in enumerate(x_c):
        expected = d_l + (Q.T - np.eye(2)) @ np.array([x - l_i, 0.0])
        assert abs(u_super[j] - expected[0]) < 1e-14
        assert abs(v_super[j] - expected[1]) < 1e-14

    u_helper, v_helper = contact_displacement(x_c, l_i, d_l[0], d_l[1], varphi)
    np.testing.assert_allclose(u_helper, u_super, atol=1e-15)
    np.testing.assert_allclose(v_helper, v_super, atol=1e-15)


@pytest.mark.parametrize("varphi_deg", ANGLES_DEG)
def test_rotation_displacement_is_zero_at_the_contact_edge(varphi_deg: float) -> None:
    varphi = np.deg2rad(varphi_deg)
    l_i = 0.35
    u, v = contact_displacement(np.array([l_i]), l_i, 0.0, 0.0, varphi)
    assert abs(float(u[0])) < 1e-16
    assert abs(float(v[0])) < 1e-16
    # And the full contact displacement at x = l_i is exactly d_l.
    u, v = contact_displacement(np.array([l_i]), l_i, 7e-4, -2e-3, varphi)
    assert float(u[0]) == pytest.approx(7e-4, abs=1e-16)
    assert float(v[0]) == pytest.approx(-2e-3, abs=1e-16)


def test_rotation_terms_are_not_small_angle_forms() -> None:
    varphi = np.deg2rad(20.0)
    r_x, r_y = rotation_coefficients(varphi)
    # Exact finite rotation, not -varphi**2/2 and -varphi.
    assert r_x == pytest.approx(np.cos(varphi) - 1.0, rel=0, abs=0)
    assert r_y == pytest.approx(-np.sin(varphi), rel=0, abs=0)
    assert abs(r_x - (-0.5 * varphi**2)) > 1e-5
    assert abs(r_y - (-varphi)) > 1e-4


# 7. Runtime force rotation.


@pytest.mark.parametrize("varphi_deg", ANGLES_DEG)
def test_force_rotation_to_local_frame(varphi_deg: float) -> None:
    varphi = np.deg2rad(varphi_deg)
    F_fixed = np.array([1234.0, -9876.0])
    F_local = rotate_force_to_local(F_fixed[0], F_fixed[1], varphi)
    Q = rotation_matrix(varphi)
    np.testing.assert_allclose(F_local, Q.T @ F_fixed, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        F_local,
        [
            np.cos(varphi) * F_fixed[0] + np.sin(varphi) * F_fixed[1],
            -np.sin(varphi) * F_fixed[0] + np.cos(varphi) * F_fixed[1],
        ],
        atol=1e-12,
    )
    # Round trip.
    np.testing.assert_allclose(Q @ F_local, F_fixed, atol=1e-9)


# 8. The 2x2 solve reproduces the requested fixed-frame forces.


@pytest.mark.parametrize("phi_deg", (-8.0, -3.0, 0.0, 3.0, 8.0))
@pytest.mark.parametrize("theta_deg", (-6.0, 0.0, 6.0))
def test_two_by_two_solve_reproduces_requested_force(lookup, phi_deg, theta_deg) -> None:
    Fx_star, Fy_star = 850.0, -1.2e4
    ev = evaluate_candidates(lookup, Fx_star, Fy_star, phi_deg, theta_deg)
    well = ev.well_conditioned
    assert np.any(well)
    scale = max(abs(Fx_star), abs(Fy_star))
    np.testing.assert_allclose(ev.Fx[well], Fx_star, atol=1e-6 * scale)
    np.testing.assert_allclose(ev.Fy[well], Fy_star, atol=1e-6 * scale)
    np.testing.assert_allclose(ev.Fx_local[well], ev.Fx_star_local, atol=1e-6 * scale)
    np.testing.assert_allclose(ev.Fy_local[well], ev.Fy_star_local, atol=1e-6 * scale)


def test_known_force_contribution_uses_documented_combination(lookup) -> None:
    phi_deg, theta_deg = 4.0, 7.0
    varphi, alpha, r_x, r_y = angles_to_coefficients(phi_deg, theta_deg, lookup.phi_ref)
    ev = evaluate_candidates(lookup, 0.0, -1.0e4, phi_deg, theta_deg)
    s = lookup.scalar_lookup
    row = int(np.flatnonzero(ev.well_conditioned)[len(np.flatnonzero(ev.well_conditioned)) // 2])
    F_known = np.array(
        [
            alpha * s[row, MODE_PHI1, SCALAR_FX]
            + r_x * s[row, MODE_BRX, SCALAR_FX]
            + r_y * s[row, MODE_BRY, SCALAR_FX],
            alpha * s[row, MODE_PHI1, SCALAR_FY]
            + r_x * s[row, MODE_BRX, SCALAR_FY]
            + r_y * s[row, MODE_BRY, SCALAR_FY],
        ]
    )
    K_F = np.array(
        [
            [s[row, MODE_BX, SCALAR_FX], s[row, MODE_BY, SCALAR_FX]],
            [s[row, MODE_BX, SCALAR_FY], s[row, MODE_BY, SCALAR_FY]],
        ]
    )
    F_star_local = np.array([ev.Fx_star_local, ev.Fy_star_local])
    expected = np.linalg.solve(K_F, F_star_local - F_known)
    np.testing.assert_allclose([ev.d_lx[row], ev.d_ly[row]], expected, rtol=1e-9)


# 9. Direct phi-dependent boundary solve agrees with five-mode superposition.


@pytest.mark.parametrize("phi_deg", (-7.0, 0.0, 7.0))
def test_direct_phi_dependent_solve_matches_superposition(compliance_result, phi_deg) -> None:
    from scipy import linalg as sla

    blocks, _ = prepare_compliance_blocks(from_compliance_result(compliance_result))
    n_b = len(blocks.x_bottom)
    n_t = len(blocks.x_top)
    x_r = 0.5 * blocks.L
    i = 7
    l_i = float(blocks.x_bottom[i])
    free, contact = free_contact_sets(i, n_b)

    varphi = np.deg2rad(phi_deg)
    alpha = np.tan(np.deg2rad(6.0))
    d_lx, d_ly = 2.5e-4, -3.1e-3
    gamma = basis_coefficients(alpha, d_lx, d_ly, varphi)

    # Assemble the actual phi-dependent boundary data and solve once.
    W_t = build_top_displacement_matrix(blocks.x_top, blocks.L, A_SOFT, KAPPA)
    u_c, v_c = contact_displacement(blocks.x_bottom[contact], l_i, d_lx, d_ly, varphi)
    A, _R_t, _R_f, _rb = build_boundary_matrix(blocks, free, contact, x_r)
    rhs = np.zeros(A.shape[0], dtype=float)
    rhs[: 2 * n_t] = contract_basis(W_t, gamma, mode_axis=1)
    rhs[2 * n_t : 2 * n_t + contact.size] = u_c
    rhs[2 * n_t + contact.size : 2 * n_t + 2 * contact.size] = v_c
    direct = sla.solve(A, rhs, assume_a="sym")

    sol = solve_candidate(blocks, ContactRecordSpec(ContactType.TOE, i), W_t, x_r, a=A_SOFT)
    superposed_Ft = contract_basis(np.asarray(sol["F_t"]), gamma, mode_axis=1)
    superposed_Fc = contract_basis(np.asarray(sol["F_c"]), gamma, mode_axis=1)
    superposed_Gf = contract_basis(np.asarray(sol["G_f"]), gamma, mode_axis=1)

    scale = max(np.linalg.norm(direct[: 2 * n_t]), 1.0)
    np.testing.assert_allclose(direct[: 2 * n_t], superposed_Ft, atol=1e-8 * scale)
    np.testing.assert_allclose(
        direct[2 * n_t : 2 * n_t + 2 * contact.size], superposed_Fc, atol=1e-8 * scale
    )
    assert superposed_Gf.size == 2 * free.size


# 10. Fixed-frame gaps agree with direct coordinate reconstruction.


@pytest.mark.parametrize("phi_deg", (-6.0, 0.0, 6.0))
def test_fixed_frame_gaps_match_coordinate_reconstruction(lookup, phi_deg) -> None:
    sel = evaluate_from_angles(lookup, 300.0, -1.0e4, phi_deg, 5.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    l_i = float(sel.l)
    varphi = ev.varphi
    u = ev.full_bottom_u[row]
    v = ev.full_bottom_v[row]

    x_fixed, y_fixed = transform_to_fixed_frame(
        lookup.x_bottom,
        np.zeros_like(lookup.x_bottom),
        u,
        v,
        l_i,
        sel.d_lx,
        sel.d_ly,
        varphi,
    )
    # The fixed-frame height above the ground line is exactly the signed gap.
    np.testing.assert_allclose(y_fixed, ev.full_bottom_gap[row], atol=1e-14)
    # And the contact edge sits at its gauge position with zero gap.
    i = int(sel.selected_index)
    assert abs(float(ev.full_bottom_gap[row][i])) < 1e-15
    assert x_fixed[i] == pytest.approx(l_i, abs=1e-14)

    # Contact nodes are on the ground by construction.
    _free, contact = free_contact_sets(i, len(lookup.x_bottom))
    np.testing.assert_allclose(ev.full_bottom_gap[row][contact], 0.0, atol=1e-14)


def test_gap_uses_both_components_when_rotated(lookup) -> None:
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -6.0, 5.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    i = int(sel.selected_index)
    free, _ = free_contact_sets(i, len(lookup.x_bottom))
    assert free.size > 0
    dx = (lookup.x_bottom + ev.full_bottom_u[row]) - (sel.l + sel.d_lx)
    dy = ev.full_bottom_v[row] - sel.d_ly
    np.testing.assert_allclose(
        ev.full_bottom_gap[row], fixed_frame_normal_component(dx, dy, ev.varphi), atol=1e-15
    )
    # The local vertical displacement alone is not the ground-normal gap.
    assert np.max(np.abs(ev.full_bottom_gap[row][free] - dy[free])) > 1e-9


# 11. Fixed-frame reactions agree with direct force rotation.


@pytest.mark.parametrize("phi_deg", (-6.0, 0.0, 6.0))
def test_fixed_frame_reactions_match_force_rotation(lookup, phi_deg) -> None:
    sel = evaluate_from_angles(lookup, 250.0, -1.0e4, phi_deg, 4.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    rx = ev.full_bottom_reaction_x[row]
    ry = ev.full_bottom_reaction_y[row]
    rt_expected, rn_expected = rotate_vector_to_fixed(rx, ry, ev.varphi)
    np.testing.assert_allclose(ev.full_bottom_reaction_normal[row], rn_expected, atol=1e-12)
    np.testing.assert_allclose(ev.full_bottom_reaction_tangential[row], rt_expected, atol=1e-12)
    i = int(sel.selected_index)
    assert sel.edge_contact_reaction_normal == pytest.approx(float(rn_expected[i]), rel=1e-12)
    assert sel.edge_contact_reaction_tangential == pytest.approx(float(rt_expected[i]), rel=1e-12)


def test_normal_reaction_differs_from_local_vertical_when_rotated(lookup) -> None:
    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -6.0, 4.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    i = int(sel.selected_index)
    _free, contact = free_contact_sets(i, len(lookup.x_bottom))
    diff = np.abs(
        ev.full_bottom_reaction_normal[row][contact] - ev.full_bottom_reaction_y[row][contact]
    )
    assert np.max(diff) > 1e-6


# 12. Candidate selection uses fixed-frame normal gaps and reactions.


def test_selection_uses_fixed_frame_quantities(lookup) -> None:
    ev = evaluate_candidates(lookup, 0.0, -1.0e4, -5.0, 4.0)
    for row in np.flatnonzero(ev.well_conditioned):
        free, contact = lookup.record_sets(row)
        g = ev.full_bottom_gap[row]
        r = ev.full_bottom_reaction_normal[row]
        gap_ok = True if free.size == 0 else bool(np.all(g[free] >= 0.0))
        reac_ok = True if contact.size == 0 else bool(np.all(r[contact] >= 0.0))
        assert bool(ev.admissible[row]) == (gap_ok and reac_ok)
        if free.size:
            assert ev.min_free_gap[row] == pytest.approx(float(np.min(g[free])))
        if contact.size:
            assert ev.min_contact_reaction[row] == pytest.approx(float(np.min(r[contact])))


def test_selection_ignores_tangential_reaction_toe_and_center_of_effort(lookup) -> None:
    ev = evaluate_candidates(lookup, 400.0, -1.0e4, -5.0, 4.0)
    base = select_contact_candidate(ev)
    import dataclasses

    perturbed = dataclasses.replace(
        ev,
        full_bottom_reaction_tangential=-1e6 * np.ones_like(ev.full_bottom_reaction_tangential),
        edge_contact_reaction_tangential=-1e6 * np.ones_like(ev.edge_contact_reaction_tangential),
        T_toe=-1e9 * np.ones_like(ev.T_toe),
        x_cm=np.full_like(ev.x_cm, np.nan),
        x_cm_rel=np.full_like(ev.x_cm_rel, np.nan),
        M=np.full_like(ev.M, np.nan),
    )
    assert select_contact_candidate(perturbed).selected_row == base.selected_row


# 13. Positive theta gives the documented toe-up shape.


def test_positive_theta_bends_the_toe_up() -> None:
    L, a, kappa = 1.0, 0.5, 30.0
    x = np.linspace(0.0, L, 401)
    phi1 = shape_mode_phi1(x, L=L, a=a, kappa=kappa)
    alpha = shape_amplitude(np.deg2rad(10.0))
    assert alpha > 0.0
    v = alpha * phi1

    distal = x >= a
    slope = np.gradient(v, x)
    # The toe segment rises relative to the (fixed-endpoint) chord.
    assert float(np.mean(slope[distal])) > 0.0
    assert v[-1] == pytest.approx(0.0, abs=1e-14)
    # The kink sags below the chord, so the toe is up relative to the heel side.
    assert float(np.min(v)) < 0.0
    assert float(np.mean(slope[x <= a])) < 0.0
    # Negative theta mirrors it.
    assert float(np.mean(np.gradient(-v, x)[distal])) < 0.0
    assert SHAPE_MODE_SIGN == 1.0


# 14. Rotated contact coordinate.


@pytest.mark.parametrize("phi_deg", (-6.0, 0.0, 6.0))
def test_rotated_contact_coordinate(lookup, phi_deg) -> None:
    ev = evaluate_candidates(lookup, 150.0, -1.0e4, phi_deg, 3.0)
    well = ev.well_conditioned & np.isfinite(ev.l)
    np.testing.assert_allclose(ev.x_l_rot[well], ev.l[well] + ev.d_lx[well], atol=1e-15)
    np.testing.assert_allclose(ev.y_l_rot[well], ev.d_ly[well], atol=1e-15)
    sel = select_contact_candidate(ev, mode=ContactMode.TOE)
    assert sel.x_l_rot == pytest.approx(sel.l + sel.d_lx, abs=1e-15)
    # The offline table still stores the material coordinate for partial records.
    partial = np.isfinite(lookup.candidate_l)
    np.testing.assert_allclose(
        lookup.candidate_l[partial], lookup.x_bottom[lookup.candidate_indices[partial]]
    )


# 15. Toe moment stays referenced to the top point and superposes.


def test_toe_moment_referenced_to_top_point(lookup) -> None:
    assert "(a, H_a)" in lookup.metadata["toe_moment"]
    sel = evaluate_from_angles(lookup, 200.0, -1.0e4, -4.0, 6.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    gamma = basis_coefficients(ev.alpha, sel.d_lx, sel.d_ly, ev.varphi)
    expected = float(lookup.scalar_lookup[row, :, SCALAR_TOE] @ gamma)
    assert sel.T_toe == pytest.approx(expected, rel=1e-12)


def test_toe_moment_uses_reference_levers_so_scalar_superposition_is_valid(lookup) -> None:
    from compliance_fem.contact_lookup import compute_toe_moment

    n_t = len(lookup.x_top)
    y_top = (
        lookup.y_top if lookup.y_top is not None else np.full(n_t, lookup.H)
    )
    sel = evaluate_from_angles(lookup, 200.0, -1.0e4, -4.0, 6.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    T_direct, _T_vert, _H_a = compute_toe_moment(
        ev.top_force_x[row], ev.top_force_y[row], lookup.x_top, y_top, lookup.softplus_a
    )
    assert float(T_direct) == pytest.approx(sel.T_toe, rel=1e-9)


# 16. Old lookup tables are rejected explicitly.


def test_old_schema_is_rejected(lookup, tmp_path) -> None:
    from compliance_fem.contact_lookup import (
        REGENERATE_LOOKUP_MESSAGE,
        load_contact_lookup,
        save_contact_lookup,
    )

    save_contact_lookup(lookup, tmp_path)
    path = tmp_path / "contact_lookup.npz"
    for stale in (1, 2, 3, 4, 5):
        data = dict(np.load(path, allow_pickle=True))
        data["schema_version"] = stale
        np.savez_compressed(tmp_path / "stale.npz", **data)
        with pytest.raises(ValueError, match="schema_version=6"):
            load_contact_lookup(tmp_path / "stale.npz")
    assert "five-mode co-rotating basis" in REGENERATE_LOOKUP_MESSAGE
    assert "cannot be migrated" in REGENERATE_LOOKUP_MESSAGE


def test_wrong_basis_order_is_rejected(lookup, tmp_path) -> None:
    from compliance_fem.contact_lookup import load_contact_lookup, save_contact_lookup

    save_contact_lookup(lookup, tmp_path)
    path = tmp_path / "contact_lookup.npz"
    data = dict(np.load(path, allow_pickle=True))
    data["basis_mode_names"] = np.asarray(list(reversed(BASIS_MODE_NAMES)), dtype=object)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="basis order"):
        load_contact_lookup(path)


def test_runtime_rejects_stale_lookup(lookup) -> None:
    import dataclasses

    from compliance_fem.contact_lookup import REGENERATE_LOOKUP_MESSAGE

    stale = dataclasses.replace(lookup, schema_version=4)
    with pytest.raises(ValueError, match="schema_version=6"):
        evaluate_candidates(stale, 0.0, -1.0e4, 0.0, 0.0)
    assert REGENERATE_LOOKUP_MESSAGE


# 17. Degree/radian conversion.


def test_gui_angle_conversion(lookup) -> None:
    varphi, alpha, r_x, r_y = angles_to_coefficients(30.0, 15.0, 0.0)
    assert varphi == pytest.approx(np.pi / 6.0)
    assert alpha == pytest.approx(np.tan(np.pi / 12.0))
    assert r_x == pytest.approx(np.cos(np.pi / 6.0) - 1.0)
    assert r_y == pytest.approx(-np.sin(np.pi / 6.0))

    ev = evaluate_candidates(lookup, 0.0, -1.0e4, 12.0, 9.0)
    assert ev.phi == pytest.approx(np.deg2rad(12.0))
    assert ev.theta == pytest.approx(np.deg2rad(9.0))
    assert ev.alpha == pytest.approx(np.tan(np.deg2rad(9.0)))

    cfg = RuntimeAngleConfig()
    assert cfg.angle_display_units == "deg"
    assert cfg.lookup_schema_version == LOOKUP_SCHEMA_VERSION
    assert cfg.phi_min_deg <= cfg.phi_default_deg <= cfg.phi_max_deg


def test_theta_at_ninety_degrees_is_rejected() -> None:
    with pytest.raises(ValueError, match="90 degrees"):
        shape_amplitude(0.5 * np.pi)
    with pytest.raises(ValueError, match="90 degrees"):
        angles_to_coefficients(0.0, 90.0, 0.0)


# 18. Non-horizontal reference top surfaces use varphi = phi - phi_ref.


def test_reference_chord_angle_of_sloped_top(sloped_lookup) -> None:
    x_top = sloped_lookup.x_top
    y_top = sloped_lookup.y_top
    expected = np.arctan2(float(y_top[-1] - y_top[0]), float(x_top[-1] - x_top[0]))
    assert sloped_lookup.phi_ref == pytest.approx(expected)
    assert abs(sloped_lookup.phi_ref) > 1e-3  # genuinely non-horizontal


def test_corotation_angle_subtracts_reference_chord(sloped_lookup) -> None:
    phi_ref = sloped_lookup.phi_ref
    phi_deg = 3.0
    varphi, _alpha, r_x, r_y = angles_to_coefficients(phi_deg, 2.0, phi_ref)
    assert varphi == pytest.approx(np.deg2rad(phi_deg) - phi_ref)
    assert r_x == pytest.approx(np.cos(varphi) - 1.0)
    assert r_y == pytest.approx(-np.sin(varphi))

    # Requesting phi == phi_ref leaves the mesh unrotated.
    phi_ref_deg = float(np.rad2deg(phi_ref))
    ev = evaluate_candidates(sloped_lookup, 0.0, -800.0, phi_ref_deg, 0.0)
    assert ev.varphi == pytest.approx(0.0, abs=1e-12)
    assert ev.r_x == pytest.approx(0.0, abs=1e-12)
    assert ev.r_y == pytest.approx(0.0, abs=1e-12)
    assert ev.Fx_star_local == pytest.approx(0.0, abs=1e-9)
    assert ev.Fy_star_local == pytest.approx(-800.0, rel=1e-12)


def test_flat_top_has_zero_reference_angle(lookup) -> None:
    assert lookup.phi_ref == pytest.approx(0.0, abs=1e-15)
    assert reference_chord_angle(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.0
    assert reference_chord_angle(np.array([0.0, 1.0]), None) == 0.0
    assert corotation_angle(0.4, 0.1) == pytest.approx(0.3)


# 19. Ill-conditioned K_F is reported, not regularized.


def test_ill_conditioned_kf_is_reported_not_regularized(lookup) -> None:
    import dataclasses

    singular = np.zeros_like(lookup.kf_matrix)
    singular[:, 0, 0] = 1.0  # rank one for every candidate
    broken = dataclasses.replace(lookup, kf_matrix=singular)
    ev = evaluate_candidates(broken, 0.0, -1.0e4, 0.0, 3.0)
    assert not np.any(ev.well_conditioned)
    assert np.all(ev.kf_illconditioned)
    assert np.all(np.isnan(ev.d_lx))
    assert np.all(np.isnan(ev.d_ly))
    selection = select_contact_candidate(ev)
    assert selection.selected_row is None
    assert "singular" in selection.message


def test_high_condition_number_is_flagged(lookup) -> None:
    ev = evaluate_candidates(
        lookup, 0.0, -1.0e4, 0.0, 3.0, tolerances=Tolerances(kf_cond_warn=1.0 + 1e-12)
    )
    assert np.all(ev.kf_illconditioned[np.isfinite(ev.kf_cond)])
    selection = select_contact_candidate(ev)
    assert selection.selected_row is not None
    assert selection.kf_illconditioned
    assert (
        "ill-conditioned" in selection.message
        or selection.contact_type is ContactType.FULL
    )


def test_kf_diagnostics_are_stored(lookup) -> None:
    assert lookup.kf_matrix.shape == (len(lookup.candidate_indices), 2, 2)
    assert lookup.kf_svals.shape == (len(lookup.candidate_indices), 2)
    finite = np.isfinite(lookup.kf_cond)
    assert np.all(lookup.kf_cond[finite] >= 1.0)
    for row in np.flatnonzero(finite):
        det = float(np.linalg.det(lookup.kf_matrix[row]))
        assert lookup.kf_det[row] == pytest.approx(det, rel=1e-10)
        svals = np.linalg.svd(lookup.kf_matrix[row], compute_uv=False)
        np.testing.assert_allclose(lookup.kf_svals[row], svals, rtol=1e-12)


# Center of effort and the visualization gauge.


def test_center_of_effort_is_computed_in_the_fixed_frame(lookup) -> None:
    sel = evaluate_from_angles(lookup, 300.0, -1.0e4, -5.0, 4.0, mode=ContactMode.TOE)
    ev = sel.evaluation
    row = sel.selected_row
    n_t = len(lookup.x_top)
    y_top = lookup.y_top if lookup.y_top is not None else np.full(n_t, lookup.H)

    _ftx_F, fty_F = rotate_vector_to_fixed(ev.top_force_x[row], ev.top_force_y[row], ev.varphi)
    x_top_F, _y = transform_to_fixed_frame(
        lookup.x_top, y_top, ev.top_u[row], ev.top_v[row], sel.l, sel.d_lx, sel.d_ly, ev.varphi
    )
    expected = float(np.sum(x_top_F * fty_F) / np.sum(fty_F))
    assert sel.x_cm == pytest.approx(expected, rel=1e-10)
    assert sel.x_cm_rel == pytest.approx(sel.x_cm - sel.l, rel=1e-10, abs=1e-12)
    # The fixed-frame vertical force uses both local components.
    assert float(np.sum(fty_F)) == pytest.approx(sel.Fy, rel=1e-8)


def test_center_of_effort_undefined_for_zero_vertical_force(lookup) -> None:
    ev = evaluate_candidates(lookup, 1.0e3, 0.0, 0.0, 2.0)
    assert np.all(np.isnan(ev.x_cm))
    assert np.all(np.isnan(ev.x_cm_rel))
    assert ev.diagnostics["fy_zero"]


# Shape plotting transform.


def test_shape_plot_uses_full_transform_without_scaling_rotation(lookup) -> None:
    from compliance_fem.shape_render import build_shape_plot_data

    sel = evaluate_from_angles(lookup, 0.0, -1.0e4, -6.0, 5.0, mode=ContactMode.TOE)
    exact = build_shape_plot_data(lookup, sel, scale_mode="true")
    scaled = build_shape_plot_data(lookup, sel, scale_mode="manual", manual_scale=20.0)
    assert exact.scale == 1.0
    assert scaled.scale == 20.0
    for shape in (exact, scaled):
        assert shape.varphi_deg == pytest.approx(np.rad2deg(sel.varphi))
        assert shape.x_contact_rot == pytest.approx(sel.x_contact_rot)
        assert shape.anchor_x == pytest.approx(sel.anchor_x)

    # Raising the scale must change only the rotated elastic part, so the
    # difference between two scales is Q(varphi) (s2 - s1) (d - d_l): the rigid
    # rotation itself carries no scale factor.
    Q = rotation_matrix(sel.varphi)
    x = exact.x_dense
    u_bot = np.interp(x, lookup.x_bottom, sel.full_bottom_u)
    v_bot = np.interp(x, lookup.x_bottom, sel.full_bottom_v)
    elastic = np.vstack([u_bot - sel.d_lx, v_bot - sel.d_ly])
    expected = (20.0 - 1.0) * (Q @ elastic)
    np.testing.assert_allclose(
        scaled.x_bottom_def - exact.x_bottom_def, expected[0], atol=1e-14
    )
    np.testing.assert_allclose(
        scaled.y_bottom_def - exact.y_bottom_def, expected[1], atol=1e-14
    )

    # At true scale the reference geometry maps exactly onto the fixed frame.
    i_edge = int(np.argmin(np.abs(x - sel.l)))
    assert abs(x[i_edge] - sel.l) < 1e-3
    assert abs(float(exact.y_bottom_def[i_edge])) < 1e-4


def test_auto_scale_keeps_the_contact_interval_on_the_ground(lookup) -> None:
    from compliance_fem.shape_render import build_shape_plot_data

    # Under rotation the contact displacement is dominated by the rigid term
    # (Q^T - I)(X - X_l); auto scale must not exaggerate it off the ground.
    for phi_deg in (-8.0, -4.0, 0.0, 4.0):
        sel = evaluate_from_angles(
            lookup, 200.0, -1.0e4, phi_deg, 5.0, mode=ContactMode.TOE
        )
        if sel.selected_row is None:
            continue
        shape = build_shape_plot_data(lookup, sel, scale_mode="auto")
        assert shape.scale >= 1.0
        lo, hi = shape.contact_span
        in_contact = (shape.x_dense >= lo - 1e-12) & (shape.x_dense <= hi + 1e-12)
        assert np.max(np.abs(shape.y_bottom_def[in_contact])) < 0.05 * lookup.H


def test_auto_scale_still_exaggerates_without_rotation(lookup) -> None:
    from compliance_fem.shape_render import build_shape_plot_data

    sel = evaluate_from_angles(lookup, 0.0, -1.0e2, 0.0, 0.0, mode=ContactMode.TOE)
    shape = build_shape_plot_data(lookup, sel, scale_mode="auto")
    assert shape.scale > 1.0
