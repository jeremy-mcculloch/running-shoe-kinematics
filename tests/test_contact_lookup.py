"""Tests for contact-edge lookup generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_basis import (
    MODE_PHI1,
    N_BASIS_MODES,
    build_contact_displacement_matrix,
    build_top_displacement_matrix,
    softplus_w2,
)
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    N_SCALAR_FIELDS,
    SCALAR_EDGE_GAP_U,
    SCALAR_EDGE_GAP_V,
    SCALAR_EDGE_RY,
    REGENERATE_LOOKUP_MESSAGE,
    build_boundary_matrix,
    candidate_indices,
    contact_records,
    free_contact_sets,
    from_compliance_result,
    generate_contact_lookup,
    load_contact_lookup,
    prepare_compliance_blocks,
    restricted_blocks,
    save_contact_lookup,
)
from compliance_fem.contact_topology import (
    ContactRecordSpec,
    ContactType,
    edge_node_pair,
)


@pytest.fixture(scope="module")
def compliance_result():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=16, ny=8, order=1)
    return compute_compliance(config)


@pytest.fixture(scope="module")
def lookup(compliance_result):
    blocks = from_compliance_result(compliance_result)
    return generate_contact_lookup(blocks, a=0.6, kappa=30.0)


def test_candidate_free_contact_sets(compliance_result) -> None:
    n_b = len(compliance_result.x_bottom)
    indices = candidate_indices(n_b, include_endpoints=False)
    assert indices[0] == 1
    assert indices[-1] == n_b - 2
    for i in indices:
        free, contact = free_contact_sets(int(i), n_b)
        assert i in contact
        assert i - 1 in free
        assert free.size == i
        assert contact.size == n_b - i
        assert np.all(free < i)
        assert np.all(contact >= i)


def test_restricted_block_dimensions(compliance_result) -> None:
    blocks, _ = prepare_compliance_blocks(from_compliance_result(compliance_result))
    i = 5
    free, contact = free_contact_sets(i, len(blocks.x_bottom))
    rb = restricted_blocks(blocks, free, contact)
    n_t = 2 * len(blocks.x_top)
    n_c = 2 * contact.size
    n_f = 2 * free.size
    assert rb["C_tc"].shape == (n_t, n_c)
    assert rb["C_ct"].shape == (n_c, n_t)
    assert rb["C_cc"].shape == (n_c, n_c)
    assert rb["C_ft"].shape == (n_f, n_t)
    assert rb["C_fc"].shape == (n_f, n_c)


def test_boundary_matrix_symmetry(compliance_result) -> None:
    blocks, _ = prepare_compliance_blocks(from_compliance_result(compliance_result))
    free, contact = free_contact_sets(5, len(blocks.x_bottom))
    A, _, _, _ = build_boundary_matrix(blocks, free, contact, x_r=0.5 * blocks.L)
    n_t = 2 * len(blocks.x_top)
    n_c = 2 * contact.size
    assert A.shape == (n_t + n_c + 3, n_t + n_c + 3)
    assert np.linalg.norm(A - A.T) / np.linalg.norm(A) < 1e-12


def test_multi_rhs_matches_separate_solves(compliance_result) -> None:
    blocks, _ = prepare_compliance_blocks(from_compliance_result(compliance_result))
    W = build_top_displacement_matrix(blocks.x_top, blocks.L, a=0.6, kappa=30.0)
    x_r = 0.5 * blocks.L
    i = 6
    free, contact = free_contact_sets(i, len(blocks.x_bottom))
    A, _, _, _ = build_boundary_matrix(blocks, free, contact, x_r)
    n_t_vec = 2 * len(blocks.x_top)
    W_c = build_contact_displacement_matrix(blocks.x_bottom[contact], float(blocks.x_bottom[i]))
    rhs = np.zeros((A.shape[0], N_BASIS_MODES))
    rhs[:n_t_vec, :] = W
    rhs[n_t_vec : n_t_vec + 2 * contact.size, :] = W_c
    X_multi = linalg.solve(A, rhs, assume_a="sym")
    for k in range(N_BASIS_MODES):
        X_k = linalg.solve(A, rhs[:, k], assume_a="sym")
        np.testing.assert_allclose(X_multi[:, k], X_k, rtol=1e-10, atol=1e-12)


def test_prescribed_top_and_contact_displacements(lookup) -> None:
    assert np.max(lookup.top_displacement_residuals) < 1e-8
    assert np.max(lookup.contact_displacement_residuals) < 1e-8
    for row in range(lookup.n_records):
        free, contact = lookup.record_sets(row)
        # The stored free-surface bases carry no contact-node entries.
        np.testing.assert_allclose(lookup.gap_basis[row][:, contact], 0.0, atol=1e-12)
        np.testing.assert_allclose(lookup.gap_u_basis[row][:, contact], 0.0, atol=1e-12)
        # The free bottom is traction free.
        np.testing.assert_allclose(lookup.reaction_basis[row][:, free], 0.0, atol=1e-12)
        np.testing.assert_allclose(lookup.reaction_x_basis[row][:, free], 0.0, atol=1e-12)
        # The top shape mode prescribes zero contact displacement.
        if contact.size:
            W_c = build_contact_displacement_matrix(
                lookup.x_bottom[contact], float(lookup.anchor_reference_x[row])
            )
            np.testing.assert_allclose(W_c[:, MODE_PHI1], 0.0, atol=0.0)


def test_force_and_moment_equilibrium(lookup) -> None:
    assert np.max(lookup.force_equilibrium_residuals) < 1e-8
    assert np.max(lookup.moment_equilibrium_residuals) < 1e-8


def test_edge_scalar_extraction_uses_topology_specific_nodes(lookup) -> None:
    n_b = len(lookup.x_bottom)
    for row in range(lookup.n_records):
        spec = lookup.record_spec(row)
        edge_contact, edge_free = edge_node_pair(spec, n_b)
        assert int(lookup.edge_contact_node_ids[row]) == (
            -1 if edge_contact is None else edge_contact
        )
        assert int(lookup.edge_free_node_ids[row]) == (-1 if edge_free is None else edge_free)
        for k in range(N_BASIS_MODES):
            if edge_free is None:
                assert np.isnan(lookup.scalar_lookup[row, k, SCALAR_EDGE_GAP_V])
                assert np.isnan(lookup.scalar_lookup[row, k, SCALAR_EDGE_GAP_U])
            else:
                np.testing.assert_allclose(
                    lookup.scalar_lookup[row, k, SCALAR_EDGE_GAP_V],
                    lookup.gap_basis[row, k, edge_free],
                )
                np.testing.assert_allclose(
                    lookup.scalar_lookup[row, k, SCALAR_EDGE_GAP_U],
                    lookup.gap_u_basis[row, k, edge_free],
                )
            if edge_contact is None:
                assert np.isnan(lookup.scalar_lookup[row, k, SCALAR_EDGE_RY])
            else:
                np.testing.assert_allclose(
                    lookup.scalar_lookup[row, k, SCALAR_EDGE_RY],
                    lookup.reaction_basis[row, k, edge_contact],
                )


def test_softplus_stable_large_kappa() -> None:
    x = np.linspace(0.0, 1.0, 21)
    w = softplus_w2(x, L=1.0, a=0.4, kappa=1e6)
    assert np.all(np.isfinite(w))
    expected = np.maximum(0.0, x - 0.4)
    np.testing.assert_allclose(w, expected, atol=1e-4)


def test_endpoint_candidates_use_nan(compliance_result, tmp_path: Path) -> None:
    blocks = from_compliance_result(compliance_result)
    lookup = generate_contact_lookup(blocks, a=0.5, kappa=20.0, include_endpoints=True)
    n_b = len(lookup.x_bottom)
    assert lookup.n_records == 2 * n_b - 1
    assert lookup.candidate_indices[-1] == -1
    assert np.isnan(lookup.candidate_l[-1])
    assert lookup.contact_type(lookup.n_records - 1) is ContactType.FULL
    assert np.isnan(lookup.scalar_lookup[-1, :, SCALAR_EDGE_GAP_V]).all()
    assert np.isnan(lookup.scalar_lookup[-1, :, SCALAR_EDGE_GAP_U]).all()
    assert np.isnan(lookup.scalar_lookup[-1, :, SCALAR_EDGE_RY]).all()
    assert lookup.status[-1] == "full_contact"


def test_serialization_roundtrip(lookup, tmp_path: Path) -> None:
    save_contact_lookup(lookup, tmp_path)
    loaded = load_contact_lookup(tmp_path / "contact_lookup.npz")
    np.testing.assert_allclose(loaded.scalar_lookup, lookup.scalar_lookup)
    np.testing.assert_allclose(loaded.gap_basis, lookup.gap_basis)
    np.testing.assert_allclose(loaded.reaction_basis, lookup.reaction_basis)
    np.testing.assert_allclose(loaded.candidate_l, lookup.candidate_l)
    np.testing.assert_allclose(loaded.top_force_y_basis, lookup.top_force_y_basis)
    np.testing.assert_allclose(loaded.kf_matrix, lookup.kf_matrix)
    np.testing.assert_allclose(loaded.kf_svals, lookup.kf_svals)
    assert loaded.schema_version == lookup.schema_version
    assert loaded.phi_ref == pytest.approx(lookup.phi_ref)
    assert loaded.status == lookup.status
    assert (tmp_path / "contact_lookup.csv").exists()
    assert (tmp_path / "contact_lookup_metadata.json").exists()


def test_scalar_lookup_has_five_modes_and_ten_fields(lookup) -> None:
    assert lookup.schema_version == LOOKUP_SCHEMA_VERSION
    assert lookup.scalar_lookup.shape == (
        len(lookup.candidate_indices),
        N_BASIS_MODES,
        N_SCALAR_FIELDS,
    )
    assert (N_BASIS_MODES, N_SCALAR_FIELDS) == (5, 10)


def test_metadata_records_conventions(lookup, tmp_path: Path) -> None:
    import json

    save_contact_lookup(lookup, tmp_path)
    meta = json.loads((tmp_path / "contact_lookup_metadata.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == LOOKUP_SCHEMA_VERSION
    assert meta["n_shape_modes"] == 1
    assert meta["shape_mode_normalization"] == "none"
    assert meta["force_frame"] == "rotating_local"
    assert meta["phi_ref"] == pytest.approx(lookup.phi_ref)
    assert len(meta["basis_order"]) == N_BASIS_MODES
    assert "infinitesimal strain" in lookup.metadata["strain_model"]
    assert "phi_ref" in lookup.metadata["angle_conventions"] or "phi" in lookup.metadata[
        "angle_conventions"
    ]


def test_load_rejects_old_schema(tmp_path: Path, lookup) -> None:
    save_contact_lookup(lookup, tmp_path)
    path = tmp_path / "contact_lookup.npz"
    data = dict(np.load(path, allow_pickle=True))
    data["schema_version"] = 4
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="schema_version=6"):
        load_contact_lookup(path)
    assert "Regenerate" in REGENERATE_LOOKUP_MESSAGE
