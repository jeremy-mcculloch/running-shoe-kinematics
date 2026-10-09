"""Serialization, schema handling and interval selection (spec tests 54-65)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from compliance_fem.contact.lookup import (
    EDGE_RESPONSE_FIELDS,
    LOOKUP_SCHEMA_VERSION,
    load_contact_lookup,
    save_contact_lookup,
)
from compliance_fem.contact.topology import CONTACT_SET_MODEL, ContactMode, parse_contact_mode
from compliance_fem.contact.force_control import (
    SEARCH_EXPANDED,
    SEARCH_FALLBACK,
    SEARCH_FORCED,
    SEARCH_GLOBAL,
    SEARCH_LOCAL,
    SelectionConfig,
    evaluate_candidates,
    select_contact_candidate,
)

ARRAY_FIELDS = (
    "contact_start_index", "contact_end_index", "contact_type_codes", "contact_start_x", "contact_end_x",
    "contact_anchor_reference_x", "contact_anchor_reference_y", "heel_contact_edge_node_id",
    "toe_contact_edge_node_id", "heel_adjacent_free_node_id", "toe_adjacent_free_node_id",
    "contact_mask", "record_valid", "scalar_lookup", "bottom_u_basis", "bottom_v_basis",
    "reaction_x_basis", "reaction_y_basis", "top_force_x_basis", "top_force_y_basis",
    "contact_force_total", "Q_alpha_shoe_on_foot_basis", "kf_matrix", "kf_cond",
    "basis_top_displacements", "x_bottom", "y_bottom", "x_top",
)


@pytest.fixture(scope="module")
def saved(rocker_lookup_full, tmp_path_factory):
    out = tmp_path_factory.mktemp("lookup")
    path = save_contact_lookup(rocker_lookup_full, out)
    return path, load_contact_lookup(path)


def test_interval_records_roundtrip_without_loss(rocker_lookup_full, saved) -> None:
    path, loaded = saved
    assert loaded.schema_version == LOOKUP_SCHEMA_VERSION == 11
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(loaded, name)), np.asarray(getattr(rocker_lookup_full, name)), err_msg=name)
    for name in EDGE_RESPONSE_FIELDS:
        np.testing.assert_array_equal(loaded.edge_responses[name], rocker_lookup_full.edge_responses[name], err_msg=name)
    assert loaded.rejection_reason == rocker_lookup_full.rejection_reason
    assert loaded.status == rocker_lookup_full.status
    for row in (0, 7, rocker_lookup_full.n_records - 1):
        np.testing.assert_array_equal(loaded.contact_node_ids(row), rocker_lookup_full.contact_node_ids(row))
        np.testing.assert_array_equal(loaded.free_node_ids(row), rocker_lookup_full.free_node_ids(row))
    for key in ("contact_set_model", "ground_geometry", "contact_law", "contact_anchor_definition",
                "curved_sole_closure_convention", "force_sign_convention", "gap_sign_convention",
                "normal_reaction_sign_convention", "basis_order", "schema_version"):
        assert loaded.metadata[key] == rocker_lookup_full.metadata[key], key
    assert loaded.metadata["contact_set_model"] == CONTACT_SET_MODEL
    assert (path.parent / "contact_lookup.csv").is_file()
    assert (path.parent / "contact_lookup_metadata.json").is_file()


def test_affine_constant_response_survives_roundtrip(rocker_lookup_full, saved) -> None:
    path, loaded = saved
    data = np.load(path, allow_pickle=True)
    np.testing.assert_array_equal(data["affine_constant_response"], rocker_lookup_full.scalar_lookup[:, 0, :])
    np.testing.assert_array_equal(data["basis_response"], rocker_lookup_full.scalar_lookup[:, 1:, :])
    assert np.max(np.abs(loaded.scalar_lookup[loaded.valid_rows, 0])) > 0.0
    np.testing.assert_array_equal(loaded.bottom_v_basis[:, 0], rocker_lookup_full.bottom_v_basis[:, 0])
    for key in ("contact_node_ids_flat", "free_node_ids_flat", "heel_free_node_ids_flat", "toe_free_node_ids_flat"):
        assert key in data.files


def _rewrite(path: Path, out: Path, **changes) -> Path:
    data = dict(np.load(path, allow_pickle=True))
    for k, v in changes.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    out.mkdir(parents=True, exist_ok=True)
    target = out / "contact_lookup.npz"
    np.savez_compressed(target, **data)
    return target


@pytest.mark.parametrize("old", [1, 6, 7, 8])
def test_old_schemas_are_rejected_clearly(saved, tmp_path, old) -> None:
    path, _ = saved
    target = _rewrite(path, tmp_path / f"v{old}", schema_version=old)
    with pytest.raises(ValueError, match="Regenerate"):
        load_contact_lookup(target)


def test_truncated_interval_records_are_rejected(saved, tmp_path) -> None:
    path, _ = saved
    data = np.load(path, allow_pickle=True)
    n = 5  # a heel/toe/full-only table has 2 N_b - 1 records
    with pytest.raises(ValueError):
        load_contact_lookup(_rewrite(
            path, tmp_path / "c",
            contact_start_index=np.asarray(data["contact_start_index"])[:n],
            contact_end_index=np.asarray(data["contact_end_index"])[:n],
        ))


def test_auto_selection_finds_known_admissible_interval(rocker_lookup, asym_lookup) -> None:
    ev = evaluate_candidates(rocker_lookup, 0.0, -50.0, 0.0, 0.0)
    sel = select_contact_candidate(ev)
    assert sel.exactly_admissible and sel.interval == (5, 7) and sel.topology_label == "interior"
    assert sel.candidate_search_method == SEARCH_GLOBAL
    np.testing.assert_array_equal(np.flatnonzero(ev.admissible), [rocker_lookup.row_of(5, 7)])
    ev2 = evaluate_candidates(asym_lookup, 0.0, -50.0, 0.0, 0.0)
    sel2 = select_contact_candidate(ev2)
    assert sel2.exactly_admissible and sel2.topology_label == "interior"
    # Off-centre: the asymmetric apex (0.35 L) pulls the interval toward the heel.
    assert 0.5 * (sel2.contact_start_x + sel2.contact_end_x) < 0.5 * asym_lookup.L


@pytest.mark.parametrize(
    "mode,check",
    [
        ("heel", lambda i, j, n: i == 0),
        ("toe", lambda i, j, n: j == n - 1),
        ("interior", lambda i, j, n: i > 0 and j < n - 1),
        ("full", lambda i, j, n: (i, j) == (0, n - 1)),
    ],
)
def test_forced_filters_evaluate_only_their_intervals(flat_lookup, mode, check) -> None:
    n = flat_lookup.n_bottom_nodes
    for Fy, phi in [(-500.0, 0.3), (-500.0, -0.3), (-2000.0, 0.0)]:
        ev = evaluate_candidates(flat_lookup, 0.0, Fy, phi, 0.0)
        sel = select_contact_candidate(ev, mode=mode)
        if sel.selected_row is None:
            continue
        assert check(*sel.interval, n)
        if mode != "full":
            for r in sel.admissible_rows:
                assert check(int(ev.contact_start_index[r]), int(ev.contact_end_index[r]), n)
        if mode == "full":
            assert sel.candidate_search_method == SEARCH_FORCED


def test_specific_interval_mode(flat_lookup) -> None:
    n = flat_lookup.n_bottom_nodes
    ev = evaluate_candidates(flat_lookup, 0.0, -500.0, 0.0, 0.0)
    for interval in [(0, 0), (3, 8), (6, 6), (2, n - 1)]:
        sel = select_contact_candidate(ev, mode="specific", specific_interval=interval)
        assert sel.interval == interval and sel.candidate_search_method == SEARCH_FORCED
    for bad in [(5, 4), (-1, 3), (0, n)]:
        with pytest.raises(ValueError):
            select_contact_candidate(ev, mode="specific", specific_interval=bad)
    with pytest.raises(ValueError):
        select_contact_candidate(ev, mode="specific")


def test_contact_mode_parses_values_and_gui_labels() -> None:
    assert parse_contact_mode("Auto") is ContactMode.AUTO
    assert parse_contact_mode("heel") is ContactMode.HEEL
    assert parse_contact_mode("Toe-attached intervals") is ContactMode.TOE
    assert parse_contact_mode("Full contact") is ContactMode.FULL
    assert parse_contact_mode("Interior intervals") is ContactMode.INTERIOR
    assert parse_contact_mode("Specific interval") is ContactMode.SPECIFIC


def test_local_and_global_search_agree_when_unique(rocker_lookup) -> None:
    ev = evaluate_candidates(rocker_lookup, 0.0, -500.0, 0.0, 0.0)
    glob = select_contact_candidate(ev)
    assert glob.candidate_search_method == SEARCH_GLOBAL and ev.admissible.sum() == 1
    i, j = glob.interval
    local = select_contact_candidate(ev, previous_interval=(i, j + 1))
    assert local.candidate_search_method == SEARCH_LOCAL and local.interval == glob.interval
    far = select_contact_candidate(ev, previous_interval=(0, 2))
    assert far.candidate_search_method in (SEARCH_EXPANDED, SEARCH_GLOBAL)
    assert far.interval == glob.interval


def test_continuity_never_preserves_inadmissible_candidate(rocker_lookup) -> None:
    ev = evaluate_candidates(rocker_lookup, 0.0, -50.0, 0.0, 0.0)
    n = rocker_lookup.n_bottom_nodes
    full_row = rocker_lookup.row_of(0, n - 1)
    assert not ev.admissible[full_row]
    for cfg in (SelectionConfig(), SelectionConfig(topology_change_penalty=1e9)):
        sel = select_contact_candidate(ev, previous_interval=(0, n - 1), selection_config=cfg)
        assert sel.exactly_admissible and sel.interval == (5, 7)
        assert sel.candidate_search_method != SEARCH_FALLBACK


def test_selection_is_order_independent(rocker_lookup) -> None:
    rng = np.random.default_rng(3)
    base = select_contact_candidate(evaluate_candidates(rocker_lookup, 5.0, -800.0, 0.7, 2.0))
    for _ in range(3):
        rows = rng.permutation(rocker_lookup.valid_rows)
        sel = select_contact_candidate(evaluate_candidates(rocker_lookup, 5.0, -800.0, 0.7, 2.0, rows=rows))
        assert sel.selected_row == base.selected_row
        assert sel.candidate_search_method == base.candidate_search_method


def test_small_perturbations_do_not_oscillate(flat_lookup) -> None:
    previous = None
    chosen = []
    for k in range(12):
        eps = 1e-9 * (-1) ** k
        ev = evaluate_candidates(flat_lookup, 0.0, -500.0 * (1.0 + eps), 0.3 * (1.0 - eps), 0.0)
        sel = select_contact_candidate(ev, previous_interval=previous)
        chosen.append(sel.interval)
        previous = sel.interval
    assert len(set(chosen)) == 1
