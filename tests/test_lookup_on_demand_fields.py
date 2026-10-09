"""Compact lookups: nodal fields recomputed on demand match the stored bases exactly."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.contact.lookup import (
    NODAL_FIELD_ARRAY_KEYS,
    NODAL_FIELD_NAMES,
    PLATE_BASIS_ARRAY_KEYS,
    PLATE_FIELD_NAMES,
    get_compliance_block_matrix,
    generate_contact_lookup,
    load_contact_lookup,
    save_contact_lookup,
)
from compliance_fem.contact.force_control import evaluate_candidates, evaluate_from_angles, refine_top_k
from compliance_fem.gait.passive_toe import solve_passive_toe_candidates
from compliance_fem.fem.plate_response import plate_state_for_selection
from compliance_fem.contact.toe_spring import ToeSpringConfig

from conftest import SMALL_TOE_LENGTH, small_config

LOADS = list(itertools.product((-200.0, 0.0, 300.0), (-3000.0, -800.0, -50.0), (-20.0, 0.0, 15.0)))


def _assert_fields_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


@pytest.fixture(scope="module")
def saved_pair(asym_lookup, asym_lookup_full, tmp_path_factory):
    compact_path = save_contact_lookup(asym_lookup, tmp_path_factory.mktemp("compact"))
    full_path = save_contact_lookup(asym_lookup_full, tmp_path_factory.mktemp("full"))
    return compact_path, full_path


def test_default_lookup_is_compact(asym_lookup, asym_lookup_full) -> None:
    assert not asym_lookup.nodal_fields_stored and asym_lookup.has_nodal_fields
    assert asym_lookup.metadata["nodal_fields_stored"] is False
    assert all(getattr(asym_lookup, key) is None for key in NODAL_FIELD_ARRAY_KEYS)
    assert asym_lookup_full.nodal_fields_stored
    np.testing.assert_array_equal(asym_lookup.scalar_lookup, asym_lookup_full.scalar_lookup)
    np.testing.assert_array_equal(asym_lookup.Q_alpha_shoe_on_foot_basis, asym_lookup_full.Q_alpha_shoe_on_foot_basis)


def test_record_fields_bitwise_equal_to_stored(asym_lookup, asym_lookup_full) -> None:
    rows = np.arange(asym_lookup.n_records)
    _assert_fields_equal(asym_lookup.record_fields(rows), asym_lookup_full.record_fields(rows))
    assert set(asym_lookup.record_fields([0])) == set(NODAL_FIELD_NAMES)


def test_record_fields_are_cached(rocker_lookup) -> None:
    solver = rocker_lookup.field_solver
    solver.clear_cache()
    before = solver.n_solves
    rocker_lookup.record_fields([3, 5, 3])
    rocker_lookup.record_fields([5])
    assert solver.n_solves - before == 2


def test_compact_file_round_trip(saved_pair, asym_lookup_full) -> None:
    compact_path, full_path = saved_pair
    with np.load(compact_path, allow_pickle=True) as data:
        assert not set(NODAL_FIELD_ARRAY_KEYS) & set(data.files)
        assert "field_solver_compliance" in data.files and not bool(data["nodal_fields_stored"])
    assert compact_path.stat().st_size < 0.5 * full_path.stat().st_size
    compact, full = load_contact_lookup(compact_path), load_contact_lookup(full_path)
    assert not compact.nodal_fields_stored and full.nodal_fields_stored
    rows = np.arange(compact.n_records)
    _assert_fields_equal(compact.record_fields(rows), asym_lookup_full.record_fields(rows))
    _assert_fields_equal(full.record_fields(rows), asym_lookup_full.record_fields(rows))


def test_save_needs_fields_or_solver(asym_lookup) -> None:
    bare = dataclasses.replace(asym_lookup, field_solver=None)
    assert not bare.has_nodal_fields
    with pytest.raises(ValueError, match="nodal fields"):
        save_contact_lookup(bare, "unused")


def test_evaluation_bounds_and_admissibility(asym_lookup, asym_lookup_full) -> None:
    for Fx, Fy, phi in LOADS[::3]:
        lazy = evaluate_candidates(asym_lookup, Fx, Fy, phi, 0.0)
        full = evaluate_candidates(asym_lookup_full, Fx, Fy, phi, 0.0)
        np.testing.assert_array_equal(lazy.admissible, full.admissible)
        assert np.all(full.fields_exact[full.evaluated])
        ex = lazy.fields_exact
        assert np.all(lazy.admissible <= ex)
        np.testing.assert_array_equal(lazy.violation_score[ex], full.violation_score[ex])
        rest = lazy.evaluated & ~ex & np.isfinite(full.violation_score)
        assert np.all(lazy.violation_score[rest] <= full.violation_score[rest])
        g_lazy, g_full = lazy.min_free_gap[rest], full.min_free_gap[rest]
        assert np.all((g_lazy == g_full) | (g_lazy >= g_full - 1e-12 * np.abs(g_full)))


def test_selection_matches_stored_lookup(asym_lookup, asym_lookup_full) -> None:
    for Fx, Fy, phi in LOADS:
        for theta in (0.0, 15.0):
            a = evaluate_from_angles(asym_lookup_full, Fx, Fy, phi, theta)
            b = evaluate_from_angles(asym_lookup, Fx, Fy, phi, theta)
            assert b.selected_row == a.selected_row and b.candidate_search_method == a.candidate_search_method
            assert b.message.split(" Interior contact")[0] == a.message.split(" Interior contact")[0]
            if a.selected_row is not None:
                for k in ("full_bottom_gap", "violation_score", "min_free_gap", "top_force_y"):
                    np.testing.assert_array_equal(np.asarray(getattr(b, k)), np.asarray(getattr(a, k)), err_msg=k)


def _same(a, b) -> None:
    if a is None or b is None:
        assert a is b
        return
    for attr in ("row", "theta_deg", "admissible", "status", "max_free_penetration", "min_contact_reaction"):
        va, vb = getattr(a, attr), getattr(b, attr)
        if isinstance(va, float) and np.isnan(va):
            assert np.isnan(vb), attr
        else:
            assert va == vb, attr


def test_passive_toe_matches_stored_lookup(asym_lookup, asym_lookup_full) -> None:
    cfg = ToeSpringConfig(toe_stiffness_Nm_per_rad=2.0)
    for Fx, Fy, phi in LOADS:
        kw = dict(Fx_star=Fx, Fy_star=Fy, Mz_meas=0.0, phi_rad=np.deg2rad(phi), config=cfg, width_m=0.1)
        ca = solve_passive_toe_candidates(asym_lookup_full, **kw)
        cb = solve_passive_toe_candidates(asym_lookup, **kw)
        assert len(ca) == len(cb)
        for x, y in zip(ca, cb):
            _same(x, y)


def test_refine_top_k_orders_exactly() -> None:
    exact_v = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    bound = np.array([0.5, 0.9, 0.1, 2.0, 4.0])
    exact = np.array([False, False, False, True, True])
    current = bound.copy()
    refined = []

    def refine(idx):
        refined.extend(int(i) for i in idx)
        current[idx] = exact_v[idx]
        exact[idx] = True

    order = refine_top_k(np.arange(5), lambda p: (current[p],), lambda p: exact[p], refine, k=2)
    np.testing.assert_array_equal(order[:2], [1, 3])
    assert 4 not in refined


@pytest.fixture(scope="module")
def plate_pair():
    fem = compute_compliance(small_config(rocker=0.008))
    blocks = get_compliance_block_matrix(fem)
    kw = dict(toe_length=SMALL_TOE_LENGTH, kappa=30.0, fem_result=fem, require_plate=True)
    return generate_contact_lookup(blocks, **kw), generate_contact_lookup(blocks, store_fields=True, **kw)


def test_plate_fields_on_demand(plate_pair, tmp_path) -> None:
    lazy, full = plate_pair
    assert lazy.has_plate_response and all(getattr(lazy, k) is None for k in PLATE_BASIS_ARRAY_KEYS)
    rows = np.arange(lazy.n_records)
    a, b = lazy.record_fields(rows, plate=True), full.record_fields(rows, plate=True)
    assert set(PLATE_FIELD_NAMES) <= set(a)
    _assert_fields_equal(a, b)
    reloaded = load_contact_lookup(save_contact_lookup(lazy, tmp_path))
    assert reloaded.has_plate_response
    _assert_fields_equal(reloaded.record_fields(rows, plate=True), b)
    sel_full = evaluate_from_angles(full, 0.0, -800.0, 0.0, 0.0)
    sel_lazy = evaluate_from_angles(lazy, 0.0, -800.0, 0.0, 0.0)
    assert sel_lazy.selected_row == sel_full.selected_row is not None
    pa, pb = plate_state_for_selection(full, sel_full), plate_state_for_selection(reloaded, sel_lazy)
    for k in ("u_local", "v_local", "rotation_local", "axial_force", "sample_x_fixed", "sample_y_fixed"):
        np.testing.assert_array_equal(getattr(pb, k), getattr(pa, k), err_msg=k)
