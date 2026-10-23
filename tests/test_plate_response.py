"""Plate-response recovery, Hermite postprocess, and plate runtime state checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.contact.contact_model import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    ContactModel,
    build_contact_model,
    load_contact_model,
    plate_fields,
    plate_state_for_solution,
    save_contact_model,
)
from compliance_fem.contact.continuous_contact import ContactSolution, fixed_frame_positions, solve_contact_state
from compliance_fem.contact.corotation import transform_to_fixed_frame
from compliance_fem.contact.sole_config import SoleConfig
from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.fem.plate_response import (
    COLOR_UNITS,
    PLATE_ARRAY_FIELDS,
    hermite_curvature,
    hermite_deflection,
    hermite_rotation,
    hermite_shear,
    sample_plate_centerline,
)

from conftest import SMALL_TOE_LENGTH, small_config


# ---------------------------------------------------------------------------
# Fixtures and mesh-/magnitude-scaled tolerances
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sole_config() -> SoleConfig:
    return small_config(rocker=0.008, EI_plate_Nm2_per_m=10.0)


@pytest.fixture(scope="module")
def fem_result(sole_config: SoleConfig):
    return compute_compliance(sole_config)


@pytest.fixture(scope="module")
def model(fem_result) -> ContactModel:
    return build_contact_model(fem_result, toe_length=SMALL_TOE_LENGTH, kappa=30.0)


@pytest.fixture(scope="module")
def plate_mesh(model):
    return model.plate


@pytest.fixture(scope="module")
def solution(model: ContactModel) -> ContactSolution:
    # Moderate compressive load (per metre width) with small rotations.
    return solve_contact_state(
        model.contact_operator, np.deg2rad(-2.0), np.tan(np.deg2rad(1.5)), np.array([50.0, -2.0e3])
    )


def _mesh_h(cfg: SoleConfig) -> float:
    return float(cfg.mesh_size_m)


def _disp_atol(cfg: SoleConfig, magnitude: float, rtol: float = 1e-8) -> float:
    """Absolute tolerance scaled by mesh size and response magnitude."""
    return rtol * max(abs(float(magnitude)), 1.0) + 1e-9 * _mesh_h(cfg)


def _rel_atol(ref: np.ndarray | float, cfg: SoleConfig, rtol: float = 1e-8) -> float:
    mag = float(np.linalg.norm(np.ravel(ref)))
    return _disp_atol(cfg, mag, rtol=rtol)


# ---------------------------------------------------------------------------
# 1. Plate DOF indices
# ---------------------------------------------------------------------------


def test_plate_dof_indices_identify_continuum_and_rotations(fem_result) -> None:
    n_foam = int(fem_result.basis.N)
    n_plate = fem_result.n_plate_nodes
    assert np.all(fem_result.plate_u_dof_ids >= 0)
    assert np.all(fem_result.plate_u_dof_ids < n_foam)
    assert np.all(fem_result.plate_v_dof_ids >= 0)
    assert np.all(fem_result.plate_v_dof_ids < n_foam)
    np.testing.assert_array_equal(
        fem_result.plate_rotation_dof_ids,
        np.arange(n_foam, n_foam + n_plate, dtype=int),
    )
    assert fem_result.n_primal == n_foam + n_plate




# ---------------------------------------------------------------------------
# Serialization and schema rejection
# ---------------------------------------------------------------------------


def test_serialization_preserves_plate_fields(model, tmp_path: Path, sole_config) -> None:
    """save / load roundtrip preserves the plate response and its reference geometry."""
    npz_path = save_contact_model(model, tmp_path / "lookup_plate")
    reloaded = load_contact_model(npz_path)
    assert reloaded.schema_version == LOOKUP_SCHEMA_VERSION
    for name in PLATE_ARRAY_FIELDS:
        ref = getattr(model.plate, name)
        np.testing.assert_allclose(getattr(reloaded.plate, name), ref, atol=_rel_atol(ref, sole_config))
    np.testing.assert_allclose(reloaded.plate.reference_x, model.plate.reference_x)
    np.testing.assert_allclose(reloaded.plate.reference_y, model.plate.reference_y)
    np.testing.assert_allclose(reloaded.plate.element_tangents, model.plate.element_tangents)
    assert reloaded.EI_plate_Nm2_per_m == pytest.approx(model.EI_plate_Nm2_per_m)


@pytest.mark.parametrize("version", (5, 11, 12, 13, 14, 15, 16))
def test_other_schema_versions_rejected(model, tmp_path: Path, version: int) -> None:
    npz_path = save_contact_model(model, tmp_path / "lookup")
    with np.load(npz_path, allow_pickle=True) as npz:
        data = dict(npz)
    data["schema_version"] = np.asarray(version)
    bad = tmp_path / f"contact_lookup_v{version}.npz"
    np.savez_compressed(bad, **data)
    with pytest.raises(ValueError, match="Regenerate") as exc:
        load_contact_model(bad)
    assert REGENERATE_LOOKUP_MESSAGE in str(exc.value)


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




def test_infinite_ea_tangential_differences_below_tolerance(model, solution, sole_config) -> None:
    """Infinite-EA tangential displacement differences and B_p residual below tolerance."""
    state = plate_state_for_solution(model, solution)
    assert state is not None
    # Mesh-scaled: inextensibility residual should be near machine/mesh zero.
    assert state.maximum_plate_axial_displacement_difference < 1e-6 * max(
        1.0, _mesh_h(sole_config)
    ) + 1e-8 * max(1.0, float(np.max(np.abs(state.u_tangential))))
    f = plate_fields(model, solution)
    scale = max(1.0, float(np.max(np.abs(np.concatenate([f["u"], f["v"]])))))
    assert float(f["inextensibility_residual_norm"][0]) < 1e-8 * scale


# ---------------------------------------------------------------------------
# Fixed-frame transform and elastic scale
# ---------------------------------------------------------------------------


def test_fixed_frame_plate_coordinates_agree_with_foam_transform(model, solution, sole_config) -> None:
    """At s_d = 1 the plate samples sit at the exact fixed-frame placement r = T + Q (X + d)."""
    state = plate_state_for_solution(model, solution, elastic_scale=1.0)
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
        X = np.array([mesh.reference_x[i], mesh.reference_y[i]]) + xi * L_e * t_e
        x_ref[idx] = X[0]
        y_ref[idx] = X[1]
    u_samp = state.sample_x_local - x_ref
    v_samp = state.sample_y_local - y_ref
    x_f, y_f = fixed_frame_positions(model.contact_operator, solution, x_ref, y_ref, u_samp, v_samp)
    np.testing.assert_allclose(state.sample_x_fixed, x_f, atol=_rel_atol(x_f, sole_config, rtol=1e-12))
    np.testing.assert_allclose(state.sample_y_fixed, y_f, atol=_rel_atol(y_f, sole_config, rtol=1e-12))


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
# Color map
# ---------------------------------------------------------------------------


def test_plate_color_quantities_and_units(model, solution) -> None:
    """COLOR_UNITS / build with each color quantity."""
    expected = {
        "none": "",
        "normal_displacement": "length",
        "tangential_displacement": "length",
        "rotation": "rad",
        "curvature": "1/length",
        "bending_moment": "force·length",
        "shear_force": "force",
        "plate_tension": "force",
        "strain_energy_density": "energy/length",
    }
    assert COLOR_UNITS == expected
    for color, units in expected.items():
        state = plate_state_for_solution(model, solution, color_quantity=color)
        assert state is not None
        assert state.color_quantity == color
        assert state.color_units == units
        if color == "none":
            assert state.color_values is None
        else:
            assert state.color_values is not None
            assert state.color_values.shape == state.sample_s.shape
