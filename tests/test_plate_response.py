"""Plate-response recovery, Hermite postprocess, and runtime lookup checks."""

from __future__ import annotations

import time

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import LayeredPlateConfig, ProblemConfig
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    from_compliance_result,
    generate_contact_lookup,
    load_contact_lookup,
    save_contact_lookup,
    solve_candidate,
)
from compliance_fem.contact_topology import ContactMode, ContactType
from compliance_fem.corotation import (
    basis_coefficients,
    contract_basis,
    transform_to_fixed_frame,
)
from compliance_fem.force_control import evaluate_from_angles
from compliance_fem.plate_response import (
    COLOR_UNITS,
    axial_force_from_multiplier,
    extract_plate_mesh_info,
    extract_plate_nodal_fields,
    hermite_curvature,
    hermite_deflection,
    hermite_rotation,
    hermite_shear,
    plate_mesh_from_lookup,
    plate_state_for_selection,
    recover_plate_basis_for_record,
    recover_primal_from_boundary_forces,
    rigid_primal_from_alpha,
    sample_plate_centerline,
)
from compliance_fem.shape_render import build_shape_plot_data


# ---------------------------------------------------------------------------
# Fixtures and mesh-/magnitude-scaled tolerances
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layered_config() -> LayeredPlateConfig:
    return LayeredPlateConfig(
        L=0.3,
        h1_heel=0.025,
        h1_toe=0.015,
        h2_heel=0.02,
        h2_toe=0.03,
        E1=2e6,
        nu1=0.3,
        E_heel=5e5,
        E_toe=1.5e6,
        nu2=0.3,
        EI_plate=10.0,
        nx=6,
        ny1=2,
        ny2=2,
    )


@pytest.fixture(scope="module")
def fem_result(layered_config: LayeredPlateConfig):
    return compute_compliance(layered_config)


@pytest.fixture(scope="module")
def lookup(fem_result):
    blocks = from_compliance_result(fem_result)
    return generate_contact_lookup(
        blocks,
        a=0.18,
        kappa=30.0,
        fem_result=fem_result,
        require_plate=True,
    )


@pytest.fixture(scope="module")
def plate_mesh(fem_result):
    return extract_plate_mesh_info(fem_result)


@pytest.fixture(scope="module")
def rectangle_lookup():
    cfg = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=8, ny=4, order=1)
    result = compute_compliance(cfg)
    return generate_contact_lookup(from_compliance_result(result), a=0.6, kappa=30.0)


def _mesh_h(cfg: LayeredPlateConfig) -> float:
    return float(cfg.L) / float(cfg.nx)


def _disp_atol(cfg: LayeredPlateConfig, magnitude: float, rtol: float = 1e-8) -> float:
    """Absolute tolerance scaled by mesh size and response magnitude."""
    return rtol * max(abs(float(magnitude)), 1.0) + 1e-9 * _mesh_h(cfg)


def _rel_atol(ref: np.ndarray, cfg: LayeredPlateConfig, rtol: float = 1e-8) -> float:
    mag = float(np.linalg.norm(np.ravel(ref)))
    return _disp_atol(cfg, mag, rtol=rtol)


def _pick_row(lookup, kind: ContactType) -> int:
    rows = lookup.rows_for(kind)
    assert rows.size >= 1
    if kind is ContactType.FULL:
        return int(rows[0])
    # Prefer an interior partial-contact edge when available.
    mid = int(rows[len(rows) // 2])
    return mid


def _selection_for(lookup, mode: ContactMode):
    # Moderate compressive load with a small angle so every forced mode finds a row.
    return evaluate_from_angles(
        lookup,
        Fx=50.0,
        Fy=-2.0e3,
        phi_deg=-2.0,
        theta_deg=1.5,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# 1. Plate DOF indices
# ---------------------------------------------------------------------------


def test_plate_dof_indices_identify_continuum_and_rotations(fem_result, plate_mesh) -> None:
    n_foam = plate_mesh.n_foam
    n_plate = plate_mesh.n_nodes
    assert n_foam == int(fem_result.basis.N)
    assert np.all(plate_mesh.u_dof_ids >= 0)
    assert np.all(plate_mesh.u_dof_ids < n_foam)
    assert np.all(plate_mesh.v_dof_ids >= 0)
    assert np.all(plate_mesh.v_dof_ids < n_foam)
    np.testing.assert_array_equal(
        plate_mesh.rotation_dof_ids,
        np.arange(n_foam, n_foam + n_plate, dtype=int),
    )
    assert plate_mesh.n_primal == n_foam + n_plate


# ---------------------------------------------------------------------------
# 2–5. Lookup plate recovery, Bp, dimensions, serialization
# ---------------------------------------------------------------------------


def test_recovered_basis_reproduces_prescribed_boundary_displacements(
    fem_result, lookup, plate_mesh, layered_config
) -> None:
    """Recovered full primal matches W_t / W_c (generation residuals + direct recover)."""
    assert lookup.has_plate_response
    assert lookup.plate_bp_residual is not None
    # Generation already enforced plate BC residuals; confirm they stayed small.
    bp = np.asarray(lookup.plate_bp_residual, dtype=float)
    assert float(np.max(bp)) < 1e-7

    blocks = from_compliance_result(fem_result)
    from compliance_fem.contact_lookup import build_top_displacement_matrix, prepare_compliance_blocks
    from compliance_fem.contact_lookup import SOLVER_YR

    prepared, _ = prepare_compliance_blocks(blocks)
    W = build_top_displacement_matrix(prepared.x_top, prepared.L, a=0.18, kappa=30.0)
    x_r = 0.5 * prepared.L

    for kind in (ContactType.HEEL, ContactType.FULL, ContactType.TOE):
        row = _pick_row(lookup, kind)
        spec = lookup.record_spec(row)
        sol = solve_candidate(prepared, spec, W, x_r, a=0.18, y_r=SOLVER_YR)
        contact = np.asarray(sol["contact"], dtype=int)
        plate = recover_plate_basis_for_record(
            fem_result,
            plate_mesh,
            np.asarray(sol["F_t"], dtype=float),
            np.asarray(sol["F_c"], dtype=float),
            contact,
            W,
            np.asarray(sol["W_c"], dtype=float),
            prepared.x_top,
            prepared.x_bottom[contact] if contact.size else np.zeros(0),
            Alpha=np.asarray(sol["Alpha"], dtype=float),
            x_r=x_r,
            y_r=SOLVER_YR,
        )
        atol_bc = _disp_atol(layered_config, 1.0, rtol=1e-5)
        assert float(np.max(plate.top_bc_residual)) < max(1e-5, atol_bc)
        assert float(np.max(plate.contact_bc_residual)) < max(1e-5, atol_bc)
        np.testing.assert_allclose(
            plate.u_local,
            lookup.plate_u_local_basis[row],
            atol=_rel_atol(plate.u_local, layered_config),
        )
        np.testing.assert_allclose(
            plate.v_local,
            lookup.plate_v_local_basis[row],
            atol=_rel_atol(plate.v_local, layered_config),
        )
        np.testing.assert_allclose(
            plate.rotation_local,
            lookup.plate_rotation_local_basis[row],
            atol=_rel_atol(plate.rotation_local, layered_config),
        )


def test_every_basis_solution_satisfies_Bp_q(lookup, layered_config) -> None:
    """3. Every stored basis solution satisfies B_p q ≈ 0."""
    assert lookup.plate_bp_residual is not None
    residual = np.asarray(lookup.plate_bp_residual, dtype=float)
    # Absolute residual; mesh-scaled floor from generation plate_bp_tol=1e-7.
    assert float(np.max(np.abs(residual))) < 1e-7 + 1e-12 * _mesh_h(layered_config)


def test_plate_array_dimensions_consistent_across_contact_types(lookup, plate_mesh) -> None:
    """4. Heel / full / toe share the same n_plate and array ranks."""
    n_plate = plate_mesh.n_nodes
    n_el = plate_mesh.n_elements
    n_rec = lookup.n_records
    assert lookup.plate_u_local_basis.shape == (n_rec, 5, n_plate)
    assert lookup.plate_v_local_basis.shape == (n_rec, 5, n_plate)
    assert lookup.plate_rotation_local_basis.shape == (n_rec, 5, n_plate)
    assert lookup.plate_constraint_multiplier_basis.shape == (n_rec, 5, n_el)
    assert lookup.plate_axial_force_basis.shape == (n_rec, 5, n_el)
    assert lookup.plate_bp_residual.shape == (n_rec, 5)
    for kind in (ContactType.HEEL, ContactType.FULL, ContactType.TOE):
        row = _pick_row(lookup, kind)
        assert lookup.plate_u_local_basis[row].shape[1] == n_plate
        assert lookup.plate_axial_force_basis[row].shape[1] == n_el


def test_serialization_preserves_plate_fields(lookup, tmp_path, layered_config) -> None:
    """5. save / load roundtrip preserves plate arrays."""
    out = tmp_path / "lookup_plate"
    save_contact_lookup(lookup, out)
    reloaded = load_contact_lookup(out / "contact_lookup.npz")
    assert reloaded.has_plate_response
    assert reloaded.schema_version == LOOKUP_SCHEMA_VERSION
    np.testing.assert_allclose(
        reloaded.plate_u_local_basis,
        lookup.plate_u_local_basis,
        atol=_rel_atol(lookup.plate_u_local_basis, layered_config),
    )
    np.testing.assert_allclose(
        reloaded.plate_v_local_basis,
        lookup.plate_v_local_basis,
        atol=_rel_atol(lookup.plate_v_local_basis, layered_config),
    )
    np.testing.assert_allclose(
        reloaded.plate_rotation_local_basis,
        lookup.plate_rotation_local_basis,
        atol=_rel_atol(lookup.plate_rotation_local_basis, layered_config),
    )
    np.testing.assert_allclose(
        reloaded.plate_axial_force_basis,
        lookup.plate_axial_force_basis,
        atol=_rel_atol(lookup.plate_axial_force_basis, layered_config),
    )
    np.testing.assert_allclose(reloaded.plate_reference_x, lookup.plate_reference_x)
    np.testing.assert_allclose(reloaded.plate_reference_y, lookup.plate_reference_y)
    assert reloaded.EI_plate == pytest.approx(lookup.EI_plate)


def test_schema_5_load_rejected(lookup, tmp_path) -> None:
    """Schema rejection: load with schema_version=5 fails."""
    out = tmp_path / "lookup_v5"
    npz_path = save_contact_lookup(lookup, out)
    data = dict(np.load(npz_path, allow_pickle=True))
    data["schema_version"] = np.asarray(5)
    bad = tmp_path / "contact_lookup_v5.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError, match="schema_version=6|Regenerate"):
        load_contact_lookup(bad)


# ---------------------------------------------------------------------------
# 6–7. Runtime superposition and FEM reconstruction
# ---------------------------------------------------------------------------


def test_runtime_superposition_matches_five_term_sum(lookup, layered_config) -> None:
    """6. Runtime plate nodal values agree with direct five-term vector sum."""
    sel = _selection_for(lookup, ContactMode.FULL)
    if sel.selected_row is None:
        pytest.skip("No full-contact selection for this load.")
    row = int(sel.selected_row)
    gamma = basis_coefficients(sel.alpha, sel.d_ax, sel.d_ay, sel.varphi)
    u_rt = contract_basis(lookup.plate_u_local_basis[row], gamma, mode_axis=0)
    v_rt = contract_basis(lookup.plate_v_local_basis[row], gamma, mode_axis=0)
    th_rt = contract_basis(lookup.plate_rotation_local_basis[row], gamma, mode_axis=0)

    u_sum = sum(gamma[k] * lookup.plate_u_local_basis[row, k] for k in range(5))
    v_sum = sum(gamma[k] * lookup.plate_v_local_basis[row, k] for k in range(5))
    th_sum = sum(gamma[k] * lookup.plate_rotation_local_basis[row, k] for k in range(5))
    np.testing.assert_allclose(u_rt, u_sum, atol=_rel_atol(u_sum, layered_config, rtol=1e-12))
    np.testing.assert_allclose(v_rt, v_sum, atol=_rel_atol(v_sum, layered_config, rtol=1e-12))
    np.testing.assert_allclose(th_rt, th_sum, atol=_rel_atol(th_sum, layered_config, rtol=1e-12))

    state = plate_state_for_selection(lookup, sel)
    assert state is not None
    np.testing.assert_allclose(state.u_local, u_sum, atol=_rel_atol(u_sum, layered_config))
    np.testing.assert_allclose(state.v_local, v_sum, atol=_rel_atol(v_sum, layered_config))
    np.testing.assert_allclose(
        state.rotation_local, th_sum, atol=_rel_atol(th_sum, layered_config)
    )


def test_runtime_plate_agrees_with_direct_fem_reconstruction(
    fem_result, lookup, plate_mesh, layered_config
) -> None:
    """7. Lookup+gamma plate matches recover from F_t,F_c for heel/full/toe."""
    blocks = from_compliance_result(fem_result)
    from compliance_fem.contact_lookup import (
        SOLVER_YR,
        build_top_displacement_matrix,
        prepare_compliance_blocks,
    )

    prepared, _ = prepare_compliance_blocks(blocks)
    W = build_top_displacement_matrix(prepared.x_top, prepared.L, a=0.18, kappa=30.0)
    x_r = 0.5 * prepared.L

    for mode, kind in (
        (ContactMode.HEEL, ContactType.HEEL),
        (ContactMode.FULL, ContactType.FULL),
        (ContactMode.TOE, ContactType.TOE),
    ):
        sel = _selection_for(lookup, mode)
        if sel.selected_row is None:
            pytest.skip(f"No {kind.value} selection for this load.")
        row = int(sel.selected_row)
        assert lookup.contact_type(row) is kind
        gamma = basis_coefficients(sel.alpha, sel.d_ax, sel.d_ay, sel.varphi)

        u_lookup = contract_basis(lookup.plate_u_local_basis[row], gamma, mode_axis=0)
        v_lookup = contract_basis(lookup.plate_v_local_basis[row], gamma, mode_axis=0)
        th_lookup = contract_basis(
            lookup.plate_rotation_local_basis[row], gamma, mode_axis=0
        )

        spec = lookup.record_spec(row)
        sol = solve_candidate(prepared, spec, W, x_r, a=0.18, y_r=SOLVER_YR)
        contact = np.asarray(sol["contact"], dtype=int)
        F_t = np.asarray(sol["F_t"], dtype=float)
        F_c = np.asarray(sol["F_c"], dtype=float)
        Alpha = np.asarray(sol["Alpha"], dtype=float)
        F_t_s = contract_basis(F_t, gamma, mode_axis=1)
        F_c_s = (
            contract_basis(F_c, gamma, mode_axis=1)
            if F_c.size
            else np.zeros(0, dtype=float)
        )
        Alpha_s = contract_basis(Alpha, gamma, mode_axis=1)
        q, lam, _ = recover_primal_from_boundary_forces(fem_result, F_t_s, F_c_s, contact)
        q = q + rigid_primal_from_alpha(fem_result, plate_mesh, Alpha_s, x_r, SOLVER_YR)
        u, v, th, _, _ = extract_plate_nodal_fields(q, lam, plate_mesh)

        np.testing.assert_allclose(
            u_lookup, u, atol=_rel_atol(u, layered_config, rtol=1e-6)
        )
        np.testing.assert_allclose(
            v_lookup, v, atol=_rel_atol(v, layered_config, rtol=1e-6)
        )
        np.testing.assert_allclose(
            th_lookup, th, atol=_rel_atol(th, layered_config, rtol=1e-6)
        )


# ---------------------------------------------------------------------------
# 8–14, 23–24. Pure Hermite mathematics
# ---------------------------------------------------------------------------


def test_hermite_reproduces_endpoint_displacements() -> None:
    """8. Hermite interpolation reproduces both endpoint displacements."""
    L = 0.4
    w_i, th_i, w_j, th_j = 0.2, -0.5, -0.1, 1.2
    xi = np.array([0.0, 1.0])
    w = hermite_deflection(xi, L, w_i, th_i, w_j, th_j)
    np.testing.assert_allclose(w, [w_i, w_j], atol=1e-14)


def test_hermite_reproduces_endpoint_rotations() -> None:
    """9. Hermite interpolation reproduces both endpoint rotations."""
    L = 0.4
    w_i, th_i, w_j, th_j = 0.2, -0.5, -0.1, 1.2
    xi = np.array([0.0, 1.0])
    th = hermite_rotation(xi, L, w_i, th_i, w_j, th_j)
    np.testing.assert_allclose(th, [th_i, th_j], atol=1e-14)


def test_constant_normal_displacement_zero_curvature() -> None:
    """10. Constant normal displacement → zero curvature."""
    L = 0.35
    c = 1.7
    xi = np.linspace(0.0, 1.0, 9)
    kappa = hermite_curvature(xi, L, c, 0.0, c, 0.0)
    np.testing.assert_allclose(kappa, 0.0, atol=1e-14)


def test_linear_normal_with_compatible_rotations_zero_curvature() -> None:
    """11. Linear w with θ = slope → zero curvature."""
    L = 0.5
    slope = 0.8
    w_i, w_j = 0.0, slope * L
    xi = np.linspace(0.0, 1.0, 11)
    kappa = hermite_curvature(xi, L, w_i, slope, w_j, slope)
    np.testing.assert_allclose(kappa, 0.0, atol=1e-13)


def test_quadratic_cubic_curvature_matches_analytical() -> None:
    """12. Known quadratic/cubic field produces expected analytical curvature."""
    L = 0.4
    xi = np.linspace(0.0, 1.0, 17)
    # w = ξ² → w_i=0, θ_i=0, w_j=1, θ_j=2/L; κ = 2/L² constant
    kappa_q = hermite_curvature(xi, L, 0.0, 0.0, 1.0, 2.0 / L)
    np.testing.assert_allclose(kappa_q, 2.0 / L**2, atol=1e-12)
    # w = ξ³ → κ = 6ξ/L²
    kappa_c = hermite_curvature(xi, L, 0.0, 0.0, 1.0, 3.0 / L)
    np.testing.assert_allclose(kappa_c, 6.0 * xi / L**2, atol=1e-12)


def test_bending_moment_is_EI_times_curvature() -> None:
    """13. M = EI κ."""
    L = 0.25
    EI = 10.0
    xi = np.linspace(0.0, 1.0, 8)
    kappa = hermite_curvature(xi, L, 0.0, 0.0, 1.0, 2.0 / L)
    # sample_plate_centerline stores M = EI * kappa; check relation directly
    np.testing.assert_allclose(EI * kappa, EI * (2.0 / L**2), atol=1e-12)


def test_shear_is_EI_times_w_triple_prime() -> None:
    """14. V = EI w'''."""
    L = 0.4
    EI = 7.0
    # Cubic w=ξ³ has w''' = 6/L³
    V = hermite_shear(L, 0.0, 0.0, 1.0, 3.0 / L, EI)
    assert V == pytest.approx(EI * 6.0 / L**3, rel=1e-12)


def test_curvature_sign_convention_for_w_equals_xi_squared() -> None:
    """23. κ = d²w/ds² for w=ξ² (positive upward w, upward n_p)."""
    L = 0.3
    # Positive upward parabola opening upward: w=ξ² ≥ 0, κ = +2/L² > 0
    kappa = hermite_curvature(np.array([0.5]), L, 0.0, 0.0, 1.0, 2.0 / L)
    assert float(kappa[0]) == pytest.approx(2.0 / L**2, rel=1e-12)
    assert float(kappa[0]) > 0.0


def test_no_duplicate_sample_points_at_element_boundaries(plate_mesh) -> None:
    """24. Unique consecutive samples; len == 1 + n_el*(samples-1)."""
    n_el = plate_mesh.n_elements
    samples = 7
    u = np.linspace(0.0, 0.01, plate_mesh.n_nodes)
    v = np.linspace(0.0, -0.02, plate_mesh.n_nodes)
    theta = np.linspace(0.0, 0.05, plate_mesh.n_nodes)
    axial = np.zeros(n_el)
    out = sample_plate_centerline(plate_mesh, u, v, theta, axial, samples_per_element=samples)
    expected = 1 + n_el * (samples - 1)
    assert out["s"].size == expected
    # Strictly increasing arc length (no duplicated boundary points).
    assert np.all(np.diff(out["s"]) > 0.0)


# ---------------------------------------------------------------------------
# 15–16. Axial constraint / multiplier convention
# ---------------------------------------------------------------------------


def test_infinite_ea_tangential_differences_below_tolerance(
    lookup, layered_config
) -> None:
    """15. Infinite-EA tangential displacement differences below tolerance."""
    sel = _selection_for(lookup, ContactMode.FULL)
    if sel.selected_row is None:
        pytest.skip("No selection.")
    state = plate_state_for_selection(lookup, sel)
    assert state is not None
    # Mesh-scaled: inextensibility residual should be near machine/mesh zero.
    assert state.maximum_plate_axial_displacement_difference < 1e-6 * max(
        1.0, _mesh_h(layered_config)
    ) + 1e-8 * max(1.0, float(np.max(np.abs(state.u_tangential))))


def test_axial_force_from_multiplier_is_identity() -> None:
    """16. Multiplier-to-axial-force is identity for unscaled rows."""
    lam = np.array([1.5, -0.2, 3.0])
    np.testing.assert_array_equal(axial_force_from_multiplier(lam), lam)
    assert axial_force_from_multiplier(lam) is not lam  # returns a copy


# ---------------------------------------------------------------------------
# 17–19. Fixed-frame transform and elastic scale
# ---------------------------------------------------------------------------


def test_fixed_frame_plate_coordinates_agree_with_foam_transform(
    lookup, layered_config
) -> None:
    """17. Fixed-frame plate coordinates agree with common foam transform."""
    sel = _selection_for(lookup, ContactMode.HEEL)
    if sel.selected_row is None:
        pytest.skip("No heel selection.")
    state = plate_state_for_selection(lookup, sel, elastic_scale=1.0)
    assert state is not None
    mesh = state.mesh
    # Reconstruct local sample displacements and re-apply the shared transform.
    x_ref = np.zeros_like(state.sample_x_local)
    y_ref = np.zeros_like(state.sample_y_local)
    for idx, e in enumerate(state.sample_element_id):
        i = int(mesh.element_connectivity[e, 0])
        L_e = float(mesh.element_lengths[e])
        t_e = mesh.element_tangents[e]
        s0 = float(mesh.reference_arc_length[i])
        xi = (float(state.sample_s[idx]) - s0) / L_e if L_e > 0 else 0.0
        X_i = np.array([mesh.reference_x[i], mesh.reference_y[i]])
        X = X_i + xi * L_e * t_e
        x_ref[idx] = X[0]
        y_ref[idx] = X[1]
    u_samp = state.sample_x_local - x_ref
    v_samp = state.sample_y_local - y_ref
    x_f, y_f = transform_to_fixed_frame(
        x_ref,
        y_ref,
        u_samp,
        v_samp,
        float(sel.anchor_x),
        float(sel.d_ax),
        float(sel.d_ay),
        float(sel.varphi),
        elastic_scale=1.0,
    )
    np.testing.assert_allclose(
        state.sample_x_fixed, x_f, atol=_rel_atol(x_f, layered_config, rtol=1e-12)
    )
    np.testing.assert_allclose(
        state.sample_y_fixed, y_f, atol=_rel_atol(y_f, layered_config, rtol=1e-12)
    )


def test_plate_and_foam_interface_nodes_coincide_after_transform(
    fem_result, lookup, plate_mesh, layered_config
) -> None:
    """18. Plate and foam interface nodes coincide after transformation."""
    sel = _selection_for(lookup, ContactMode.FULL)
    if sel.selected_row is None:
        pytest.skip("No selection.")
    row = int(sel.selected_row)
    gamma = basis_coefficients(sel.alpha, sel.d_ax, sel.d_ay, sel.varphi)
    u_p = contract_basis(lookup.plate_u_local_basis[row], gamma, mode_axis=0)
    v_p = contract_basis(lookup.plate_v_local_basis[row], gamma, mode_axis=0)

    # Foam interface: same continuum DOFs as plate translations.
    blocks = from_compliance_result(fem_result)
    from compliance_fem.contact_lookup import (
        SOLVER_YR,
        build_top_displacement_matrix,
        prepare_compliance_blocks,
    )

    prepared, _ = prepare_compliance_blocks(blocks)
    W = build_top_displacement_matrix(prepared.x_top, prepared.L, a=0.18, kappa=30.0)
    sol = solve_candidate(
        prepared, lookup.record_spec(row), W, 0.5 * prepared.L, a=0.18, y_r=SOLVER_YR
    )
    contact = np.asarray(sol["contact"], dtype=int)
    F_t_s = contract_basis(np.asarray(sol["F_t"]), gamma, mode_axis=1)
    F_c = np.asarray(sol["F_c"], dtype=float)
    F_c_s = contract_basis(F_c, gamma, mode_axis=1) if F_c.size else np.zeros(0)
    Alpha_s = contract_basis(np.asarray(sol["Alpha"]), gamma, mode_axis=1)
    q, lam, _ = recover_primal_from_boundary_forces(fem_result, F_t_s, F_c_s, contact)
    q = q + rigid_primal_from_alpha(
        fem_result, plate_mesh, Alpha_s, 0.5 * prepared.L, SOLVER_YR
    )
    u_foam = q[plate_mesh.u_dof_ids]
    v_foam = q[plate_mesh.v_dof_ids]
    np.testing.assert_allclose(u_p, u_foam, atol=_rel_atol(u_foam, layered_config, rtol=1e-6))
    np.testing.assert_allclose(v_p, v_foam, atol=_rel_atol(v_foam, layered_config, rtol=1e-6))

    x_a, d_ax, d_ay, varphi = (
        float(sel.anchor_x),
        float(sel.d_ax),
        float(sel.d_ay),
        float(sel.varphi),
    )
    xp, yp = transform_to_fixed_frame(
        plate_mesh.reference_x, plate_mesh.reference_y, u_p, v_p, x_a, d_ax, d_ay, varphi
    )
    xf, yf = transform_to_fixed_frame(
        plate_mesh.reference_x,
        plate_mesh.reference_y,
        u_foam,
        v_foam,
        x_a,
        d_ax,
        d_ay,
        varphi,
    )
    np.testing.assert_allclose(xp, xf, atol=_rel_atol(xf, layered_config, rtol=1e-6))
    np.testing.assert_allclose(yp, yf, atol=_rel_atol(yf, layered_config, rtol=1e-6))


def test_rigid_rotation_varphi_not_multiplied_by_elastic_scale() -> None:
    """19. Rigid rotation varphi is not multiplied by elastic scale."""
    x_ref = np.array([0.0, 0.15, 0.3])
    y_ref = np.array([0.02, 0.025, 0.03])
    x_a, d_ax, d_ay, varphi = 0.15, 0.01, -0.002, 0.2
    # Pure rigid gauge: elastic relative displacement (u - d_a) is zero.
    u_rigid = np.full_like(x_ref, d_ax, dtype=float)
    v_rigid = np.full_like(y_ref, d_ay, dtype=float)
    x1, y1 = transform_to_fixed_frame(
        x_ref, y_ref, u_rigid, v_rigid, x_a, d_ax, d_ay, varphi, elastic_scale=1.0
    )
    x2, y2 = transform_to_fixed_frame(
        x_ref, y_ref, u_rigid, v_rigid, x_a, d_ax, d_ay, varphi, elastic_scale=2.0
    )
    np.testing.assert_allclose(x1, x2, atol=1e-15)
    np.testing.assert_allclose(y1, y2, atol=1e-15)

    # Nonzero elastic: difference between scales equals scaled elastic only.
    u_e = u_rigid + np.array([0.001, -0.002, 0.003])
    v_e = v_rigid + np.array([0.004, 0.0, -0.001])
    x_s1, y_s1 = transform_to_fixed_frame(
        x_ref, y_ref, u_e, v_e, x_a, d_ax, d_ay, varphi, elastic_scale=1.0
    )
    x_s2, y_s2 = transform_to_fixed_frame(
        x_ref, y_ref, u_e, v_e, x_a, d_ax, d_ay, varphi, elastic_scale=2.0
    )
    x_r, y_r = transform_to_fixed_frame(
        x_ref, y_ref, u_e, v_e, x_a, d_ax, d_ay, varphi, elastic_scale=0.0
    )
    np.testing.assert_allclose(x_s2 - x_r, 2.0 * (x_s1 - x_r), atol=1e-14)
    np.testing.assert_allclose(y_s2 - y_r, 2.0 * (y_s1 - y_r), atol=1e-14)


# ---------------------------------------------------------------------------
# 20–22. GUI / contact-type switching / color map
# ---------------------------------------------------------------------------


def test_gui_handles_lookup_without_plate_fields(rectangle_lookup) -> None:
    """20. plate_state_for_selection returns None; shape_render with rectangle lookup."""
    assert not rectangle_lookup.has_plate_response
    sel = evaluate_from_angles(
        rectangle_lookup, Fx=100.0, Fy=-5.0e3, phi_deg=-3.0, theta_deg=2.0
    )
    assert plate_state_for_selection(rectangle_lookup, sel) is None
    if sel.selected_row is None:
        pytest.skip("No rectangle contact selection.")
    shape = build_shape_plot_data(rectangle_lookup, sel, scale_mode="true", show_plate=True)
    assert shape is not None
    # Rectangle has no plate profile.
    assert shape.y_plate_ref is None or not getattr(rectangle_lookup, "has_plate_response", False)


def test_changing_contact_type_updates_plate_without_fem_solve(
    lookup, layered_config
) -> None:
    """21. HEEL vs TOE vs FULL yield different plate fields (no FEM re-solve)."""
    fields = {}
    t0 = time.perf_counter()
    for mode, kind in (
        (ContactMode.HEEL, ContactType.HEEL),
        (ContactMode.TOE, ContactType.TOE),
        (ContactMode.FULL, ContactType.FULL),
    ):
        sel = _selection_for(lookup, mode)
        if sel.selected_row is None:
            pytest.skip(f"No {kind.value} selection.")
        state = plate_state_for_selection(lookup, sel)
        assert state is not None
        fields[kind] = (state.u_local.copy(), state.v_local.copy(), state.rotation_local.copy())
    elapsed = time.perf_counter() - t0
    # Three runtime contractions should be far cheaper than a FEM solve.
    assert elapsed < 2.0

    heel_u, heel_v, heel_th = fields[ContactType.HEEL]
    toe_u, toe_v, toe_th = fields[ContactType.TOE]
    full_u, full_v, full_th = fields[ContactType.FULL]
    # At least one nodal field differs between topologies.
    diffs = [
        np.linalg.norm(heel_u - toe_u) + np.linalg.norm(heel_v - toe_v),
        np.linalg.norm(heel_u - full_u) + np.linalg.norm(heel_v - full_v),
        np.linalg.norm(toe_u - full_u) + np.linalg.norm(toe_v - full_v),
        np.linalg.norm(heel_th - toe_th),
        np.linalg.norm(heel_th - full_th),
    ]
    assert max(diffs) > _disp_atol(layered_config, 1e-3, rtol=1e-3)


def test_plate_color_quantities_and_units(lookup) -> None:
    """22. COLOR_UNITS / build with each color quantity."""
    sel = _selection_for(lookup, ContactMode.FULL)
    if sel.selected_row is None:
        pytest.skip("No selection.")
    expected = {
        "none": "",
        "normal_displacement": "length",
        "tangential_displacement": "length",
        "rotation": "rad",
        "curvature": "1/length",
        "bending_moment": "force·length",
        "shear_force": "force",
        "axial_constraint_force": "force",
    }
    assert COLOR_UNITS == expected
    for color, units in expected.items():
        state = plate_state_for_selection(lookup, sel, color_quantity=color)
        assert state is not None
        assert state.color_quantity == color
        assert state.color_units == units
        if color == "none":
            assert state.color_values is None
        else:
            assert state.color_values is not None
            assert state.color_values.shape == state.sample_s.shape


def test_plate_mesh_from_lookup_roundtrip(lookup) -> None:
    mesh = plate_mesh_from_lookup(lookup)
    assert mesh is not None
    assert mesh.n_nodes == int(lookup.plate_node_ids.size)
    assert mesh.EI_plate == pytest.approx(lookup.EI_plate)
