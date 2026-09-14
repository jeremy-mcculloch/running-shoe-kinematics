"""Section 19 topology / basis checks (items 1-10, 24, 25)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_basis import (
    MODE_BRX,
    MODE_BRY,
    N_BASIS_MODES,
    build_contact_displacement_matrix,
    build_top_displacement_matrix,
)
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    SCALAR_EDGE_GAP_U,
    SCALAR_EDGE_GAP_V,
    SCALAR_EDGE_RX,
    SCALAR_EDGE_RY,
    from_compliance_result,
    generate_contact_lookup,
    load_contact_lookup,
    save_contact_lookup,
    solve_candidate,
)
from compliance_fem.contact_topology import (
    ContactType,
    full_sets,
    heel_sets,
    n_records,
    toe_sets,
)


@pytest.fixture(scope="module")
def small_compliance():
    config = ProblemConfig(L=1.0, H=0.5, E=1.0e6, nu=0.3, nx=8, ny=4, order=1)
    return compute_compliance(config)


@pytest.fixture(scope="module")
def small_lookup(small_compliance):
    blocks = from_compliance_result(small_compliance)
    return generate_contact_lookup(blocks, a=0.6, kappa=30.0)


# --- 1–5: node sets and full-contact storage ---------------------------------


def test_heel_candidate_node_sets() -> None:
    n_b = 9
    for i in range(0, n_b - 1):
        contact, free = heel_sets(i, n_b)
        np.testing.assert_array_equal(contact, np.arange(0, i + 1))
        np.testing.assert_array_equal(free, np.arange(i + 1, n_b))
        assert i in contact
        assert np.all(contact <= i)
        assert np.all(free > i)


def test_toe_candidate_node_sets() -> None:
    n_b = 9
    for i in range(1, n_b):
        contact, free = toe_sets(i, n_b)
        np.testing.assert_array_equal(contact, np.arange(i, n_b))
        np.testing.assert_array_equal(free, np.arange(0, i))
        assert i in contact
        assert np.all(contact >= i)
        assert np.all(free < i)


def test_full_contact_contains_every_bottom_node() -> None:
    n_b = 9
    contact, free = full_sets(n_b)
    np.testing.assert_array_equal(contact, np.arange(n_b))
    assert free.size == 0


def test_transition_node_belongs_to_contact_set() -> None:
    n_b = 9
    for i in range(0, n_b - 1):
        contact, _free = heel_sets(i, n_b)
        assert i in contact
    for i in range(1, n_b):
        contact, _free = toe_sets(i, n_b)
        assert i in contact


def test_full_contact_stored_exactly_once(small_lookup) -> None:
    n_b = len(small_lookup.x_bottom)
    assert small_lookup.n_records == n_records(n_b)
    assert small_lookup.n_records == 2 * n_b - 1
    assert callable(small_lookup.contact_types)
    types = small_lookup.contact_types()
    assert sum(1 for t in types if t is ContactType.FULL) == 1
    assert int(np.count_nonzero(small_lookup.contact_type_codes == 2)) == 1
    full_row = small_lookup.full_contact_row()
    assert small_lookup.contact_type(full_row) is ContactType.FULL
    assert np.isnan(small_lookup.candidate_l[full_row])
    free, contact = small_lookup.record_sets(full_row)
    assert free.size == 0
    np.testing.assert_array_equal(contact, np.arange(n_b))


# --- 6–10: five-mode anchors and rotation displacement signs ----------------


def test_five_basis_modes_use_correct_anchor(small_lookup, small_compliance) -> None:
    blocks = from_compliance_result(small_compliance)
    W = build_top_displacement_matrix(
        blocks.x_top, blocks.L, small_lookup.softplus_a, small_lookup.softplus_kappa
    )
    x_r = 0.5 * blocks.L
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)

    for row in range(small_lookup.n_records):
        spec = small_lookup.record_spec(row)
        x_anchor = float(small_lookup.anchor_reference_x[row])
        if spec.contact_type is ContactType.FULL:
            assert x_anchor == pytest.approx(0.5 * L, abs=abs_tol)
        else:
            l_i = float(blocks.x_bottom[int(spec.edge_node_id)])
            assert x_anchor == pytest.approx(l_i, abs=abs_tol)

        sol = solve_candidate(blocks, spec, W, x_r, a=small_lookup.softplus_a)
        free, contact = spec.sets(len(blocks.x_bottom))
        W_c = build_contact_displacement_matrix(blocks.x_bottom[contact], x_anchor)
        np.testing.assert_allclose(sol["W_c"], W_c, rtol=0.0, atol=abs_tol)
        assert float(sol["x_anchor"]) == pytest.approx(x_anchor, abs=abs_tol)
        ds = blocks.x_bottom[contact] - x_anchor
        np.testing.assert_allclose(W_c[: contact.size, MODE_BRX], ds, atol=abs_tol)
        np.testing.assert_allclose(W_c[contact.size :, MODE_BRY], ds, atol=abs_tol)


def test_heel_rotation_modes_use_negative_x_minus_l(small_lookup) -> None:
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)
    for row in small_lookup.rows_for(ContactType.HEEL):
        i = int(small_lookup.candidate_indices[row])
        contact, _ = heel_sets(i, len(small_lookup.x_bottom))
        l_i = float(small_lookup.x_bottom[i])
        W_c = build_contact_displacement_matrix(small_lookup.x_bottom[contact], l_i)
        ds = W_c[: contact.size, MODE_BRX]
        assert np.all(ds <= abs_tol)
        assert np.min(ds) < -abs_tol or contact.size == 1


def test_toe_rotation_modes_use_positive_x_minus_l(small_lookup) -> None:
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)
    for row in small_lookup.rows_for(ContactType.TOE):
        i = int(small_lookup.candidate_indices[row])
        contact, _ = toe_sets(i, len(small_lookup.x_bottom))
        l_i = float(small_lookup.x_bottom[i])
        W_c = build_contact_displacement_matrix(small_lookup.x_bottom[contact], l_i)
        ds = W_c[: contact.size, MODE_BRX]
        assert np.all(ds >= -abs_tol)
        assert np.max(ds) > abs_tol or contact.size == 1


def test_partial_contact_rotation_displacement_zero_at_edge(small_lookup) -> None:
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)
    for kind in (ContactType.HEEL, ContactType.TOE):
        for row in small_lookup.rows_for(kind):
            i = int(small_lookup.candidate_indices[row])
            l_i = float(small_lookup.x_bottom[i])
            W_c = build_contact_displacement_matrix(np.array([l_i]), l_i)
            assert W_c[0, MODE_BRX] == pytest.approx(0.0, abs=abs_tol)
            assert W_c[1, MODE_BRY] == pytest.approx(0.0, abs=abs_tol)


def test_full_contact_rotation_displacement_zero_at_midspan(small_lookup) -> None:
    L = float(small_lookup.L)
    abs_tol = 1e-12 * max(L, 1.0)
    x_a = 0.5 * L
    W_c = build_contact_displacement_matrix(np.array([x_a]), x_a)
    assert W_c[0, MODE_BRX] == pytest.approx(0.0, abs=abs_tol)
    assert W_c[1, MODE_BRY] == pytest.approx(0.0, abs=abs_tol)
    full_row = small_lookup.full_contact_row()
    assert float(small_lookup.anchor_reference_x[full_row]) == pytest.approx(x_a, abs=abs_tol)


# --- 24–25: full-contact NaN edge scalars and v4 rejection --------------------


def test_full_contact_edge_scalars_are_nan(small_lookup) -> None:
    full_row = small_lookup.full_contact_row()
    edge_fields = (SCALAR_EDGE_GAP_V, SCALAR_EDGE_GAP_U, SCALAR_EDGE_RX, SCALAR_EDGE_RY)
    for k in range(N_BASIS_MODES):
        for field in edge_fields:
            assert np.isnan(small_lookup.scalar_lookup[full_row, k, field])
    assert not bool(small_lookup.has_free_edge[full_row])
    assert int(small_lookup.edge_contact_node_ids[full_row]) == -1
    assert int(small_lookup.edge_free_node_ids[full_row]) == -1


def test_old_v4_lookup_schema_is_rejected(small_lookup, tmp_path: Path) -> None:
    save_contact_lookup(small_lookup, tmp_path)
    path = tmp_path / "contact_lookup.npz"
    data = dict(np.load(path, allow_pickle=True))
    data["schema_version"] = 4
    # Drop topology keys so a stale v4-shaped file is unmistakable.
    for key in (
        "contact_type_codes",
        "edge_node_ids",
        "anchor_reference_x",
        "contact_mask",
        "corner_reactions_local",
        "has_free_edge",
    ):
        data.pop(key, None)
    stale = tmp_path / "stale_v4.npz"
    np.savez_compressed(stale, **data)
    with pytest.raises(ValueError, match="schema_version=6") as exc_info:
        load_contact_lookup(stale)
    message = str(exc_info.value)
    assert "Schema v4" in message or "cannot be migrated" in message
    assert "Regenerate" in REGENERATE_LOOKUP_MESSAGE
    assert LOOKUP_SCHEMA_VERSION == 6
    assert "cannot be migrated" in REGENERATE_LOOKUP_MESSAGE
