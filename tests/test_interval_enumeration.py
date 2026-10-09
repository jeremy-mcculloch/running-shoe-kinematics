"""Interval enumeration, record structure and structured rejection (spec tests 1-14)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import compliance_fem.contact.lookup as contact_lookup
from compliance_fem.contact.lookup import (
    REJECT_RANK_DEFICIENT,
    get_compliance_block_matrix,
    generate_contact_lookup,
    record_counts,
)
from compliance_fem.contact.topology import (
    ABSENT_NODE_ID,
    CONTACT_SET_MODEL,
    ContactInterval,
    ContactType,
    enumerate_intervals,
    interval_arrays,
    interval_label,
    interval_row,
    labels_from_arrays,
    n_intervals,
    validate_interval,
)

from conftest import SMALL_KAPPA, SMALL_TOE_LENGTH


@pytest.mark.parametrize("n", [1, 2, 3, 7, 13])
def test_every_interval_generated_exactly_once(n) -> None:
    pairs = [(iv.start, iv.end) for iv in enumerate_intervals(n)]
    expected = {(i, j) for i in range(n) for j in range(i, n)}
    assert len(pairs) == len(set(pairs)) == len(expected)
    assert set(pairs) == expected
    for row, (i, j) in enumerate(pairs):
        assert interval_row(i, j, n) == row
    starts, ends = interval_arrays(n)
    assert list(zip(starts.tolist(), ends.tolist())) == pairs


@pytest.mark.parametrize("n", [1, 2, 5, 13])
def test_total_interval_count(n) -> None:
    assert n_intervals(n) == n * (n + 1) // 2
    counts = record_counts(n)
    assert counts["total"] == n * (n + 1) // 2
    assert counts["heel"] + counts["toe"] + counts["full"] + counts["interior"] == counts["total"]


def test_lookup_stores_all_intervals_when_supported(flat_lookup) -> None:
    n = flat_lookup.n_bottom_nodes
    assert flat_lookup.n_records == n * (n + 1) // 2
    assert int(np.count_nonzero(flat_lookup.valid_mask)) == n * (n + 1) // 2
    assert flat_lookup.metadata["contact_set_model"] == CONTACT_SET_MODEL
    assert flat_lookup.metadata["n_intervals_theoretical"] == n * (n + 1) // 2


@pytest.mark.parametrize("i,j", [(0, 0), (0, 4), (3, 7), (5, 5), (8, 12), (0, 12)])
def test_node_sets_of_an_interval(i, j) -> None:
    n = 13
    iv = ContactInterval(i, j, n)
    assert iv.contact_node_ids.tolist() == list(range(i, j + 1))
    assert iv.heel_free_node_ids.tolist() == list(range(0, i))
    assert iv.toe_free_node_ids.tolist() == list(range(j + 1, n))
    assert i in iv.contact_node_ids and j in iv.contact_node_ids
    assert iv.heel_contact_edge_node_id == i and iv.toe_contact_edge_node_id == j
    free, contact = iv.sets()
    assert np.intersect1d(free, contact).size == 0
    assert np.union1d(free, contact).tolist() == list(range(n))


def test_labels_derived_from_endpoints() -> None:
    n = 9
    intervals = enumerate_intervals(n)
    full = [iv for iv in intervals if iv.topology_label is ContactType.FULL]
    assert [(iv.start, iv.end) for iv in full] == [(0, n - 1)]
    for iv in intervals:
        label = iv.topology_label
        if label is ContactType.HEEL:
            assert iv.start == 0 and iv.end < n - 1
        elif label is ContactType.TOE:
            assert iv.end == n - 1 and iv.start > 0
        elif label is ContactType.INTERIOR:
            assert iv.heel_free_node_ids.size > 0 and iv.toe_free_node_ids.size > 0
        assert interval_label(iv.start, iv.end, n) is label
    starts, ends = interval_arrays(n)
    codes = labels_from_arrays(starts, ends, n)
    assert codes.tolist() == [contact_lookup.contact_type_code(iv.topology_label) for iv in intervals]


def test_adjacent_free_node_ids(flat_lookup) -> None:
    n = flat_lookup.n_bottom_nodes
    for row in range(flat_lookup.n_records):
        i = int(flat_lookup.contact_start_index[row])
        j = int(flat_lookup.contact_end_index[row])
        h = int(flat_lookup.heel_adjacent_free_node_id[row])
        t = int(flat_lookup.toe_adjacent_free_node_id[row])
        assert h == (i - 1 if i > 0 else ABSENT_NODE_ID)
        assert t == (j + 1 if j < n - 1 else ABSENT_NODE_ID)


def test_missing_adjacent_nodes_consistent(flat_lookup) -> None:
    n = flat_lookup.n_bottom_nodes
    heel_absent = flat_lookup.contact_start_index == 0
    toe_absent = flat_lookup.contact_end_index == n - 1
    assert np.all(flat_lookup.heel_adjacent_free_node_id[heel_absent] == ABSENT_NODE_ID)
    assert np.all(flat_lookup.toe_adjacent_free_node_id[toe_absent] == ABSENT_NODE_ID)
    er = flat_lookup.edge_responses
    for name in ("heel_adjacent_free_u", "heel_adjacent_free_v"):
        assert np.all(np.isnan(er[name][heel_absent]))
        assert np.all(np.isfinite(er[name][~heel_absent]))
    for name in ("toe_adjacent_free_u", "toe_adjacent_free_v"):
        assert np.all(np.isnan(er[name][toe_absent]))
        assert np.all(np.isfinite(er[name][~toe_absent]))


def test_validate_interval_rejects_bad_endpoints() -> None:
    validate_interval(0, 0, 1)
    for i, j in [(-1, 2), (3, 2), (0, 5), (5, 5)]:
        with pytest.raises(ValueError):
            validate_interval(i, j, 5)


def test_one_node_interval_does_not_double_count_edge_force(flat_lookup_full) -> None:
    n = flat_lookup_full.n_bottom_nodes
    for k in (0, n // 2, n - 1):
        row = flat_lookup_full.row_of(k, k)
        assert flat_lookup_full.valid_mask[row]
        assert flat_lookup_full.status[row] == "single_contact_node"
        er = flat_lookup_full.edge_responses
        rx = flat_lookup_full.reaction_x_basis[row]
        ry = flat_lookup_full.reaction_y_basis[row]
        # The single node is both heel and toe edge; each field reports its force once.
        np.testing.assert_allclose(er["heel_edge_reaction_local_x"][row], rx[:, k])
        np.testing.assert_allclose(er["toe_edge_reaction_local_x"][row], rx[:, k])
        np.testing.assert_allclose(er["heel_edge_reaction_local_y"][row], ry[:, k])
        np.testing.assert_allclose(er["toe_edge_reaction_local_y"][row], ry[:, k])
        # Total contact force is the nodal force, not heel + toe.
        total = flat_lookup_full.contact_force_total[row]
        np.testing.assert_allclose(total[:, 0], rx[:, k], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(total[:, 1], ry[:, k], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rx.sum(axis=1), rx[:, k])
        # Top force balances the single nodal force.
        S = flat_lookup_full.scalar_lookup[row]
        scale = np.max(np.abs(S[:, :2]))
        assert np.max(np.abs(S[:, 0] + rx[:, k])) <= 1e-9 * scale
        assert np.max(np.abs(S[:, 1] + ry[:, k])) <= 1e-9 * scale


def test_rank_deficient_interval_rejected_with_structured_reason(flat_case, monkeypatch) -> None:
    result, _ = flat_case
    blocks = get_compliance_block_matrix(result)
    n = len(blocks.x_bottom)
    target_contact = 3  # contact count that identifies the sabotaged interval (2, 4)
    real = contact_lookup.build_boundary_matrix

    def sabotaged(blocks_, free, contact, x_r, y_r=0.0):
        system = real(blocks_, free, contact, x_r, y_r=y_r)
        if contact.size == target_contact and int(contact[0]) == 2:
            A = system.A.copy()
            A[-1, :] = A[-2, :]  # duplicate a row: exactly singular
            return dataclasses.replace(system, A=A)
        return system

    monkeypatch.setattr(contact_lookup, "build_boundary_matrix", sabotaged)
    lookup = generate_contact_lookup(blocks, toe_length=SMALL_TOE_LENGTH, kappa=SMALL_KAPPA)
    row = lookup.row_of(2, 4)
    assert not lookup.valid_mask[row]
    assert lookup.rejection_reason[row].startswith(REJECT_RANK_DEFICIENT)
    assert lookup.boundary_matrix_rank[row] < lookup.boundary_matrix_size[row]
    assert lookup.status[row] == "rejected"
    assert np.all(np.isnan(lookup.scalar_lookup[row]))
    assert int(np.count_nonzero(~lookup.valid_mask)) == 1
    assert lookup.rejection_summary() == {lookup.rejection_reason[row]: 1}
    assert lookup.metadata["rejection_counts"] == {REJECT_RANK_DEFICIENT: 1}
    assert lookup.metadata["n_intervals_valid"] == n * (n + 1) // 2 - 1
