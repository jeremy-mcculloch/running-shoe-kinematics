"""GUI geometry and diagnostics for the selected interval (spec tests 66-70)."""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.contact.force_control import NOT_AVAILABLE, evaluate_candidates, format_optional, select_contact_candidate
from compliance_fem.plotting.shape_render import build_shape_plot_data


def _shape(lookup, Fx, Fy, phi, theta=0.0, **kw):
    ev = evaluate_candidates(lookup, Fx, Fy, phi, theta)
    sel = select_contact_candidate(ev, **kw)
    return sel, build_shape_plot_data(lookup, sel)


@pytest.mark.parametrize("lookup_name", ["flat_lookup", "rocker_lookup", "asym_lookup"])
def test_shading_matches_reference_interval(request, lookup_name) -> None:
    lk = request.getfixturevalue(lookup_name)
    sel, shape = _shape(lk, 0.0, -500.0, 0.0)
    i, j = sel.interval
    assert shape.contact_span == (lk.x_bottom[i], lk.x_bottom[j])
    assert (shape.contact_start_index, shape.contact_end_index) == (i, j)
    assert shape.contact_curve_x.size == j - i + 1
    # The contact curve follows the deformed curved bottom of nodes i..j and lies on the ground.
    assert np.max(np.abs(shape.contact_curve_y)) <= 1e-9 * lk.L
    assert shape.scale == 1.0
    assert (shape.contact_curve_x[0], shape.contact_curve_y[0]) == shape.heel_edge_xy
    assert (shape.contact_curve_x[-1], shape.contact_curve_y[-1]) == shape.toe_edge_xy
    n_free = lk.n_bottom_nodes - (j - i + 1)
    assert shape.free_node_x.size == n_free
    assert len(shape.free_curves) == int(i > 0) + int(j < lk.n_bottom_nodes - 1)


def test_edge_markers_distinct_from_numerical_anchor(rocker_lookup) -> None:
    from compliance_fem.gui.app import _draw_shape

    sel, shape = _shape(rocker_lookup, 0.0, -2000.0, 0.0)
    i, j = sel.interval
    assert j > i
    assert shape.heel_edge_xy != shape.toe_edge_xy
    assert shape.anchor_xy[0] == pytest.approx(0.5 * (rocker_lookup.x_bottom[i] + rocker_lookup.x_bottom[j]))
    assert shape.anchor_xy != shape.heel_edge_xy and shape.anchor_xy != shape.toe_edge_xy
    fig = _draw_shape(shape)
    traces = {t.name: t for t in fig.data if t.name}
    heel = next(t for n, t in traces.items() if n.startswith("heel contact edge"))
    toe = next(t for n, t in traces.items() if n.startswith("toe contact edge"))
    anchor = next(t for n, t in traces.items() if n.startswith("neutral anchor"))
    symbols = {heel.marker.symbol, toe.marker.symbol, anchor.marker.symbol}
    assert len(symbols) == 3
    assert "numerical" in anchor.name
    assert anchor.marker.color != heel.marker.color
    contact = next(t for n, t in traces.items() if "contact {" in n)
    free = traces["free bottom"]
    assert contact.line.dash is None and free.line.dash == "dash"
    assert "ground" in traces


def test_missing_outside_gaps_display_as_na(flat_lookup) -> None:
    sel, _ = _shape(flat_lookup, 0.0, -50.0, 0.0, mode="full")
    assert format_optional(sel.heel_adjacent_free_gap) == NOT_AVAILABLE == "N/A"
    assert format_optional(sel.toe_adjacent_free_gap) == "N/A"
    assert format_optional(None) == "N/A"
    assert format_optional(1.25e-4, ".3e") == "1.250e-04"
    heel, shape = _shape(flat_lookup, 0.0, -500.0, 0.3)
    assert heel.interval[0] == 0
    assert format_optional(heel.heel_adjacent_free_gap) == "N/A"
    assert format_optional(heel.toe_adjacent_free_gap) != "N/A"
    assert shape.heel_adjacent_xy is None and shape.toe_adjacent_xy is not None


def test_gui_update_does_not_trigger_fem_solve(rocker_lookup, monkeypatch) -> None:
    import compliance_fem.fem.compliance as compliance
    import compliance_fem.contact.lookup as contact_lookup
    import scipy.sparse.linalg as spla
    from compliance_fem.gui.app import _draw_shape

    def boom(*args, **kwargs):
        raise AssertionError("runtime FEM solve attempted")

    monkeypatch.setattr(compliance, "compute_compliance", boom)
    monkeypatch.setattr(contact_lookup, "solve_candidate", boom)
    monkeypatch.setattr(spla, "spsolve", boom)
    for phi, theta, kw in [(0.0, 0.0, {}), (2.0, 5.0, {}), (0.0, 0.0, {"mode": "specific", "specific_interval": (2, 9)})]:
        _, shape = _shape(rocker_lookup, 0.0, -900.0, phi, theta, **kw)
        _draw_shape(shape)
