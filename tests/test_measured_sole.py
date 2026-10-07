"""Measured carbon-plated sole: geometry, mesh, partial plate, contact, serialization, GUI.

Numbered comments follow the measured-sole specification (tests 1-67). The
full-resolution 270 mm model is used for geometry / mesh / plate checks; a
subsampled copy of the same CSV with a coarse mesh keeps the lookup tests fast.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from compliance_fem.compliance import (
    compute_compliance,
    compute_measured_compliance,
    is_measured_compliance_npz,
    measured_config_from_compliance_npz,
    save_compliance_npz,
)
from compliance_fem.config import MeasuredSoleConfig, validate_shoe_length_mm
from compliance_fem.constraints import assemble_axial_constraints
from compliance_fem.contact_direct_fem import run_standard_direct_comparisons
from compliance_fem.contact_lookup import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MZ,
    from_compliance_result,
    generate_contact_lookup,
    load_contact_lookup,
    save_contact_lookup,
)
from compliance_fem.contact_topology import ContactInterval, n_intervals
from compliance_fem.force_control import evaluate_candidates, select_contact_candidate
from compliance_fem.measured_geometry import (
    BOUNDARY_TAGS,
    FOAM_REGION_TAGS,
    SHARED_TOE_POLICY,
    GeometryFileError,
    build_measured_geometry,
    distance_to_polyline,
    load_normalized_sole_csv,
    point_in_polygon,
    polyline_self_intersections,
)
from compliance_fem.measured_mesh import MeshQualityError, generate_measured_mesh
from compliance_fem.plate import assemble_plate_bending, plate_element_frames
from compliance_fem.shape_render import build_shape_plot_data

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "geometry" / "sole_geometry_normalized.csv"
SHOE_MM = 270.0
L_M = SHOE_MM / 1000.0
A_SOFT = 0.78 * L_M
KAPPA = 160.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["record_type", "name", "index", "x_over_length", "y_over_length"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _subsampled_rows(step: int = 5) -> list[dict]:
    """Every ``step``-th sample (bottom kept dense near the toe, where the interface meets it)."""
    rows = _read_rows(CSV_PATH)
    points = [r for r in rows if r["record_type"] == "point"]
    out = list(points)
    by_curve: dict[str, list[dict]] = {}
    for r in rows:
        if r["record_type"] == "curve":
            by_curve.setdefault(r["name"], []).append(r)
    for name, rs in by_curve.items():
        rs = sorted(rs, key=lambda r: int(r["index"]))
        n = len(rs)
        keep = [
            r for i, r in enumerate(rs)
            if i % step == 0 or i == n - 1 or (name == "bottom_surface" and i >= n - 6)
        ]
        out.extend({**r, "index": str(j)} for j, r in enumerate(keep))
    return out


@pytest.fixture(scope="module")
def meas_cfg() -> MeasuredSoleConfig:
    return MeasuredSoleConfig(shoe_length_mm=SHOE_MM, geometry_csv=str(CSV_PATH))


@pytest.fixture(scope="module")
def meas_geom(meas_cfg):
    return build_measured_geometry(meas_cfg.normalized_geometry, SHOE_MM)


@pytest.fixture(scope="module")
def meas_fem(meas_cfg):
    return compute_measured_compliance(meas_cfg)


@pytest.fixture(scope="module")
def meas_mesh(meas_fem):
    return meas_fem.mesh_data


@pytest.fixture(scope="module")
def coarse_csv(tmp_path_factory) -> Path:
    return _write_rows(tmp_path_factory.mktemp("coarse") / "coarse_sole.csv", _subsampled_rows())


@pytest.fixture(scope="module")
def coarse_cfg(coarse_csv) -> MeasuredSoleConfig:
    return MeasuredSoleConfig(shoe_length_mm=SHOE_MM, geometry_csv=str(coarse_csv), mesh_size=0.010)


@pytest.fixture(scope="module")
def coarse_fem(coarse_cfg):
    return compute_measured_compliance(coarse_cfg)


@pytest.fixture(scope="module")
def coarse_lookup(coarse_fem):
    with pytest.warns(UserWarning):
        return generate_contact_lookup(from_compliance_result(coarse_fem), a=A_SOFT, kappa=KAPPA, fem_result=coarse_fem, store_fields=True)


@pytest.fixture(scope="module")
def saved_lookup(coarse_lookup, tmp_path_factory):
    out = tmp_path_factory.mktemp("meas_lookup")
    path = save_contact_lookup(coarse_lookup, out)
    return path, load_contact_lookup(path)


def _selection(lookup, phi_deg=0.0, theta_deg=0.0, Fx=0.0, Fy=-1500.0):
    ev = evaluate_candidates(lookup, Fx, Fy, phi_deg, theta_deg)
    return ev, select_contact_candidate(ev)


# ---------------------------------------------------------------------------
# Coordinate conversion (1-9)
# ---------------------------------------------------------------------------


# 1. shoe_length_mm is required.
def test_shoe_length_required() -> None:
    with pytest.raises(ValueError, match="shoe_length_mm is required"):
        MeasuredSoleConfig(shoe_length_mm=None, geometry_csv=str(CSV_PATH))


# 2. Nonpositive / nonfinite lengths are rejected.
@pytest.mark.parametrize("bad", [0.0, -270.0, float("nan"), float("inf"), "abc"])
def test_bad_shoe_length_rejected(bad) -> None:
    with pytest.raises(ValueError):
        validate_shoe_length_mm(bad)
    with pytest.raises(ValueError):
        MeasuredSoleConfig(shoe_length_mm=bad, geometry_csv=str(CSV_PATH))


# 3-4. Toe maps to L mm, heel-bottom to the origin.
@pytest.mark.parametrize("length_mm", [240.0, 270.0, 310.0])
def test_toe_and_heel_mapping(meas_cfg, length_mm) -> None:
    g = build_measured_geometry(meas_cfg.normalized_geometry, length_mm)
    assert g.landmarks["toe_tip"][0] == pytest.approx(length_mm / 1000.0, rel=1e-12)
    np.testing.assert_allclose(g.landmarks["heel_bottom"], [0.0, 0.0], atol=1e-15)


# 5-6. One common scale for both axes; aspect ratio preserved.
def test_uniform_scale_and_aspect(meas_cfg, meas_geom) -> None:
    norm = meas_cfg.normalized_geometry
    s = SHOE_MM / 1000.0
    for name, p in norm.landmarks.items():
        np.testing.assert_allclose(meas_geom.landmarks[name], s * np.asarray(p), rtol=0, atol=1e-15)
    top_n = np.asarray(norm.curves["top_surface"])
    ratio_n = np.ptp(top_n[:, 1]) / np.ptp(top_n[:, 0])
    ratio_p = np.ptp(meas_geom.top[:, 1]) / np.ptp(meas_geom.top[:, 0])
    assert ratio_p == pytest.approx(ratio_n, rel=1e-12)


# 7. Coordinates are metres before assembly.
def test_metres_before_assembly(meas_mesh) -> None:
    p = meas_mesh.mesh.p
    assert p[0].max() == pytest.approx(L_M, rel=1e-12)
    assert 0.01 < np.ptp(p[1]) < 0.1


# 8-9. No sign flip, no second reversal.
def test_no_flip_no_reversal(meas_geom) -> None:
    g = meas_geom
    np.testing.assert_array_equal(g.top[0], g.landmarks["heel_top"])
    np.testing.assert_array_equal(g.top[-1], g.landmarks["toe_tip"])
    np.testing.assert_array_equal(g.bottom[0], g.landmarks["heel_bottom"])
    np.testing.assert_array_equal(g.bottom[-1], g.landmarks["toe_tip"])
    for curve in (g.top, g.bottom, g.interface):
        assert np.all(np.diff(curve[:, 0]) > 0.0)
    x = np.linspace(0.05, 0.9, 20) * L_M
    assert np.all(np.interp(x, g.top[:, 0], g.top[:, 1]) > np.interp(x, g.bottom[:, 0], g.bottom[:, 1]))
    assert g.total_area > 0.0
    assert g.landmarks["toe_tip"][1] > 0.0


# ---------------------------------------------------------------------------
# Geometry topology (10-20)
# ---------------------------------------------------------------------------


# 10. Records are found by name (row order is irrelevant).
def test_parse_by_name_not_row(tmp_path) -> None:
    rows = _read_rows(CSV_PATH)
    rng = np.random.default_rng(3)
    shuffled = [rows[i] for i in rng.permutation(len(rows))]
    a = load_normalized_sole_csv(CSV_PATH)
    b = load_normalized_sole_csv(_write_rows(tmp_path / "shuffled.csv", shuffled))
    for name in a.landmarks:
        np.testing.assert_array_equal(a.landmarks[name], b.landmarks[name])
    for name in a.curves:
        np.testing.assert_array_equal(a.curves[name], b.curves[name])


# 11. Indices must be consecutive; duplicates, nonfinite and non-monotone x are rejected.
@pytest.mark.parametrize("defect", ["gap", "duplicate", "nonfinite", "nonmonotone", "missing"])
def test_curve_validation(tmp_path, defect) -> None:
    rows = [dict(r) for r in _read_rows(CSV_PATH)]
    top = [r for r in rows if r["name"] == "top_surface"]
    if defect == "gap":
        top[10]["index"] = "1000"
    elif defect == "duplicate":
        rows.append(dict(top[10]))
    elif defect == "nonfinite":
        top[10]["y_over_length"] = "nan"
    elif defect == "nonmonotone":
        top[10]["x_over_length"], top[11]["x_over_length"] = top[11]["x_over_length"], top[10]["x_over_length"]
    elif defect == "missing":
        rows = [r for r in rows if r["name"] != "plate_toe_endpoint"]
    with pytest.raises(GeometryFileError):
        load_normalized_sole_csv(_write_rows(tmp_path / f"{defect}.csv", rows))


# 12. Curves are heel-to-toe; 13. the exterior closes exactly.
def test_exterior_closes(meas_geom) -> None:
    g = meas_geom
    np.testing.assert_array_equal(g.bottom[-1], g.top[-1])
    np.testing.assert_array_equal(g.top[0], g.heel_edge[0])
    np.testing.assert_array_equal(g.heel_edge[-1], g.bottom[0])
    n_ext = len(g.bottom) + len(g.top) - 1 + len(g.heel_edge) - 2
    assert len(g.exterior) == n_ext


# 14. Heel edge is the straight heel_top -> heel_bottom segment.
def test_heel_edge_straight(meas_geom) -> None:
    g = meas_geom
    a, b = g.landmarks["heel_top"], g.landmarks["heel_bottom"]
    np.testing.assert_array_equal(g.heel_edge[0], a)
    np.testing.assert_array_equal(g.heel_edge[-1], b)
    d = b - a
    for p in g.heel_edge:
        cross = abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / np.linalg.norm(d)
        assert cross <= 1e-9 * L_M


# 15. Top and bottom share exactly one vertex, the toe.
def test_single_toe_vertex(meas_geom) -> None:
    shared = {tuple(p) for p in meas_geom.top} & {tuple(p) for p in meas_geom.bottom}
    assert shared == {tuple(meas_geom.landmarks["toe_tip"])}


# 16. No self-intersection.
def test_exterior_simple(meas_geom) -> None:
    assert polyline_self_intersections(meas_geom.exterior, closed=True, eps=1e-14 * L_M**2) == []


# 17-18. All three corners convex; names like corner_2_interior are not interpreted.
def test_corners_convex_and_names_ignored(tmp_path) -> None:
    rows = _read_rows(CSV_PATH)
    rows.append({"record_type": "point", "name": "corner_2_interior", "index": "", "x_over_length": "0.5", "y_over_length": "0.5"})
    norm = load_normalized_sole_csv(_write_rows(tmp_path / "extra.csv", rows))
    assert "corner_2_interior" in norm.ignored_names
    g = build_measured_geometry(norm, SHOE_MM)
    assert set(g.corner_angles_deg) == {"heel_bottom", "toe_tip", "heel_top"}
    assert all(0.0 < ang < 180.0 for ang in g.corner_angles_deg.values())
    ref = build_measured_geometry(load_normalized_sole_csv(CSV_PATH), SHOE_MM)
    np.testing.assert_array_equal(g.exterior, ref.exterior)


# 19. Interface endpoints lie on the exterior.
def test_interface_endpoints_on_exterior(meas_geom) -> None:
    closed = np.vstack([meas_geom.exterior, meas_geom.exterior[:1]])
    d = distance_to_polyline(closed, meas_geom.interface[[0, -1]])
    assert np.all(d <= 1e-12 * L_M)


# 20. Exactly two subdomains.
def test_two_subdomains(meas_geom, meas_mesh) -> None:
    assert set(meas_geom.region_polygons) == set(FOAM_REGION_TAGS)
    assert sum(meas_geom.region_areas.values()) == pytest.approx(meas_geom.total_area, rel=1e-9)
    regions = np.unique(meas_mesh.element_region)
    assert regions.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Materials and mesh (21-28)
# ---------------------------------------------------------------------------


def _centroids(mesh) -> np.ndarray:
    return mesh.p[:, mesh.t].mean(axis=1).T


# 21. Each element gets exactly one material.
def test_one_material_per_element(meas_mesh) -> None:
    up = meas_mesh.region_elements("upper_foam")
    lo = meas_mesh.region_elements("lower_foam")
    assert np.intersect1d(up, lo).size == 0
    assert up.size + lo.size == meas_mesh.mesh.t.shape[1]


# 22. FFTurbo above the interface, FFLeap below.
def test_material_regions(meas_cfg, meas_mesh) -> None:
    mats = meas_cfg.region_materials()
    assert mats["upper_foam"].name == "FFTurbo" and mats["lower_foam"].name == "FFLeap"
    c = _centroids(meas_mesh.mesh)
    iface = meas_mesh.geometry.interface
    y_if = np.interp(c[:, 0], iface[:, 0], iface[:, 1])
    inside = (c[:, 0] > iface[0, 0]) & (c[:, 0] < iface[-1, 0])
    up = meas_mesh.element_region == 0
    assert np.all(c[up & inside, 1] > y_if[up & inside] - 1e-9)
    assert np.all(c[~up & inside, 1] < y_if[~up & inside] + 1e-9)


# 23. Interface nodes are shared by both regions.
def test_interface_nodes_shared(meas_mesh) -> None:
    t = meas_mesh.mesh.t
    up_nodes = set(np.unique(t[:, meas_mesh.element_region == 0]).tolist())
    lo_nodes = set(np.unique(t[:, meas_mesh.element_region == 1]).tolist())
    assert up_nodes & lo_nodes == set(int(n) for n in meas_mesh.interface_nodes)


# 24. Assignment follows geometry, not element numbering.
def test_material_independent_of_numbering(meas_mesh) -> None:
    c = _centroids(meas_mesh.mesh)
    in_upper = point_in_polygon(meas_mesh.geometry.region_polygons["upper_foam"], c)
    np.testing.assert_array_equal(in_upper, meas_mesh.element_region == 0)
    perm = np.random.default_rng(0).permutation(c.shape[0])
    np.testing.assert_array_equal(
        point_in_polygon(meas_mesh.geometry.region_polygons["upper_foam"], c[perm]),
        (meas_mesh.element_region == 0)[perm],
    )


# 25. Boundary tags survive meshing.
def test_boundary_tags(meas_mesh) -> None:
    mesh = meas_mesh.mesh
    for tag, nodes in (
        ("TOP_SURFACE", meas_mesh.top_curve_nodes),
        ("BOTTOM_SURFACE", meas_mesh.bottom_curve_nodes),
        ("HEEL_EDGE", meas_mesh.heel_edge_nodes),
        ("FOAM_INTERFACE", meas_mesh.interface_nodes),
        ("PLATE_SEGMENT", meas_mesh.plate_node_ids),
    ):
        facets = mesh.boundaries[tag]
        assert facets.size > 0
        assert set(np.unique(mesh.facets[:, facets]).tolist()) == set(int(n) for n in nodes)


# 26. Positive element areas.
def test_positive_areas(meas_mesh) -> None:
    p, t = meas_mesh.mesh.p, meas_mesh.mesh.t
    a, b, c = p[:, t[0]], p[:, t[1]], p[:, t[2]]
    area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    assert np.all(area > 1e-6 * meas_mesh.config.mesh_size**2)


# 27. Mesh quality is enforced near the point toe.
def test_toe_quality_enforced(meas_mesh, meas_cfg) -> None:
    q = meas_mesh.quality
    assert q["toe_elements"] >= 1
    assert q["toe_min_angle_deg"] >= meas_cfg.min_angle_deg
    assert q["toe_min_angle_deg"] <= meas_mesh.geometry.corner_angles_deg["toe_tip"] + 1e-9
    strict = dataclasses.replace(meas_cfg, min_angle_deg=58.0)
    with pytest.raises(MeshQualityError):
        generate_measured_mesh(strict)


# 28. Unequal top / bottom node counts.
def test_unequal_node_counts(meas_fem, coarse_fem) -> None:
    for fem in (meas_fem, coarse_fem):
        assert fem.n_top_nodes != fem.n_bottom_nodes


# ---------------------------------------------------------------------------
# Partial plate (29-38)
# ---------------------------------------------------------------------------


# 29-30. Plate endpoints are projected (not snapped) and inserted.
def test_plate_endpoints_projected(meas_geom, meas_cfg) -> None:
    g = meas_geom
    for label in ("plate_heel_endpoint", "plate_toe_endpoint"):
        proj = g.landmarks[f"{label}_projected"]
        assert g.projection_distances[label] <= 1e-5 * L_M
        assert np.linalg.norm(proj - g.landmarks[label]) == pytest.approx(g.projection_distances[label], abs=1e-15)
    np.testing.assert_array_equal(g.interface[g.plate_heel_index], g.landmarks["plate_heel_endpoint_projected"])
    np.testing.assert_array_equal(g.interface[g.plate_toe_index], g.landmarks["plate_toe_endpoint_projected"])
    raw = np.asarray(meas_cfg.normalized_geometry.curves["foam_interface"]) * L_M
    d_samples = np.min(np.hypot(*(raw - g.landmarks["plate_toe_endpoint_projected"]).T))
    assert d_samples > 1e-6 * L_M  # inserted, not snapped onto an existing sample


# 31. Plate elements lie between the endpoints only.
def test_plate_elements_within_span(meas_fem, meas_geom) -> None:
    p = meas_fem.basis.mesh.p[:, meas_fem.plate_node_ids]
    np.testing.assert_allclose(p[:, 0], meas_geom.landmarks["plate_heel_endpoint_projected"], atol=1e-12 * L_M)
    np.testing.assert_allclose(p[:, -1], meas_geom.landmarks["plate_toe_endpoint_projected"], atol=1e-12 * L_M)
    assert np.all(np.diff(p[0]) > 0.0)
    assert meas_fem.plate_element_ids.size == meas_fem.n_plate_nodes - 1


# 32-33. No plate stiffness / rotation DOFs outside the span.
def test_no_plate_terms_outside_span(meas_fem) -> None:
    r = meas_fem
    n_foam = r.basis.N
    assert r.n_plate_rotation_dofs == r.n_plate_nodes
    assert r.n_primal == n_foam + r.n_plate_nodes
    coords = r.basis.mesh.p[:, r.plate_node_ids]
    _, _, normals = plate_element_frames(coords)
    Kp = assemble_plate_bending(n_foam, r.plate_u_dof_ids, r.plate_v_dof_ids, coords, normals, 2.0).tocoo()
    allowed = set(np.concatenate([r.plate_u_dof_ids, r.plate_v_dof_ids, r.plate_rotation_dof_ids]).tolist())
    assert set(Kp.row.tolist()) <= allowed and set(Kp.col.tolist()) <= allowed


# 34. Inextensibility rows only on plate elements.
def test_constraints_only_on_plate(meas_fem) -> None:
    B = meas_fem.B_p.tocoo()
    assert meas_fem.B_p.shape[0] == meas_fem.n_plate_nodes - 1 == meas_fem.constraint_rank
    allowed = set(np.concatenate([meas_fem.plate_u_dof_ids, meas_fem.plate_v_dof_ids]).tolist())
    assert set(B.col.tolist()) <= allowed
    assert meas_fem.inextensibility_residuals["top"] <= 1e-12 * L_M


# 35. Element tangents / normals.
def test_element_frames(meas_fem) -> None:
    coords = meas_fem.basis.mesh.p[:, meas_fem.plate_node_ids]
    lengths, t, n = plate_element_frames(coords)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, rtol=1e-14)
    np.testing.assert_allclose(n, np.column_stack([-t[:, 1], t[:, 0]]), atol=0)
    np.testing.assert_allclose(t * lengths[:, None], np.diff(coords, axis=1).T, rtol=1e-12, atol=1e-15)
    assert np.all(t[:, 0] > 0.0) and np.all(n[:, 1] > 0.0)
    np.testing.assert_allclose(meas_fem.plate_element_tangents, t)
    assert np.ptp(np.arctan2(t[:, 1], t[:, 0])) > np.deg2rad(5.0)  # the plate is curved


# 36. Curved-plate assembly equals direct local-to-global transformation.
def test_curved_plate_matches_direct_assembly(meas_fem) -> None:
    r = meas_fem
    n_foam = r.basis.N
    coords = r.basis.mesh.p[:, r.plate_node_ids]
    lengths, t, n = plate_element_frames(coords)
    EI = 2.0
    K = assemble_plate_bending(n_foam, r.plate_u_dof_ids, r.plate_v_dof_ids, coords, n, EI).toarray()
    K_ref = np.zeros_like(K)
    u, v, th = r.plate_u_dof_ids, r.plate_v_dof_ids, r.plate_rotation_dof_ids
    for e, L in enumerate(lengths):
        k = EI / L**3 * np.array(
            [[12, 6 * L, -12, 6 * L], [6 * L, 4 * L**2, -6 * L, 2 * L**2],
             [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L**2, -6 * L, 4 * L**2]]
        )
        dofs = [u[e], v[e], th[e], u[e + 1], v[e + 1], th[e + 1]]
        T = np.zeros((4, 6))
        T[0, 0:2] = n[e]
        T[1, 2] = 1.0
        T[2, 3:5] = n[e]
        T[3, 5] = 1.0
        K_ref[np.ix_(dofs, dofs)] += T.T @ k @ T
    np.testing.assert_allclose(K, K_ref, rtol=1e-12, atol=1e-12 * np.abs(K_ref).max())


# 37. Natural (free) end-rotation condition.
def test_natural_end_rotation(meas_fem) -> None:
    r = meas_fem
    rot = r.plate_rotation_dof_ids
    assert r.B_p.tocsc()[:, rot].nnz == 0
    for side in ("top", "bottom"):
        assert np.intersect1d(np.concatenate(r.selector_dofs(side)), rot).size == 0
    K = r.K.tocsr()
    first = set(K[rot[0]].indices.tolist())
    elem0 = {r.plate_u_dof_ids[0], r.plate_v_dof_ids[0], rot[0], r.plate_u_dof_ids[1], r.plate_v_dof_ids[1], rot[1]}
    assert first <= elem0 and K[rot[0], rot[0]] > 0.0


# ---------------------------------------------------------------------------
# Shared toe (39-44)
# ---------------------------------------------------------------------------


# 39. One physical toe node.
def test_one_toe_node(meas_mesh) -> None:
    toe = meas_mesh.geometry.landmarks["toe_tip"]
    d = np.hypot(*(meas_mesh.mesh.p - toe[:, None]))
    assert np.count_nonzero(d <= 1e-9 * L_M) == 1
    assert int(np.argmin(d)) == meas_mesh.toe_node_id


# 40-42. Toe is in the bottom selector only; selectors are disjoint.
def test_toe_selector_ownership(meas_fem, meas_mesh) -> None:
    toe = meas_mesh.toe_node_id
    assert int(meas_fem.bottom_node_ids[-1]) == toe
    assert toe not in set(meas_fem.top_node_ids.tolist())
    top = np.concatenate(meas_fem.selector_dofs("top"))
    bot = np.concatenate(meas_fem.selector_dofs("bottom"))
    assert np.intersect1d(top, bot).size == 0
    assert meas_fem.geometry_metadata["shared_toe_policy"] == SHARED_TOE_POLICY


# 43. Top resultants sum over the top selector only (toe not double-counted).
def test_top_force_no_double_count(coarse_lookup, coarse_fem) -> None:
    n_t = coarse_fem.n_top_nodes
    assert n_t == len(coarse_fem.mesh_data.top_curve_nodes) - 1
    rows = coarse_lookup.valid_rows[:50]
    fx = coarse_lookup.top_force_x_basis[rows]
    fy = coarse_lookup.top_force_y_basis[rows]
    assert fx.shape[-1] == n_t
    np.testing.assert_allclose(coarse_lookup.scalar_lookup[rows, :, SCALAR_FX], fx.sum(axis=-1), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(coarse_lookup.scalar_lookup[rows, :, SCALAR_FY], fy.sum(axis=-1), rtol=1e-12, atol=1e-9)
    total = coarse_lookup.contact_force_total[rows][..., :2]
    top_total = coarse_lookup.scalar_lookup[rows][..., [SCALAR_FX, SCALAR_FY]]
    scale = np.abs(top_total).max()
    np.testing.assert_allclose(top_total + total, 0.0, atol=1e-8 * scale)


# 44. Candidates containing the toe are well posed.
def test_toe_candidates_valid(coarse_lookup) -> None:
    n_b = coarse_lookup.n_bottom_nodes
    toe_rows = np.flatnonzero(coarse_lookup.contact_end_index == n_b - 1)
    assert toe_rows.size == n_b
    assert np.all(coarse_lookup.valid_mask[toe_rows])
    assert np.all(np.isfinite(coarse_lookup.condition_estimates[toe_rows]))


# ---------------------------------------------------------------------------
# Compliance and contact (45-57)
# ---------------------------------------------------------------------------


# 45. Blocks support unequal node counts.
def test_block_shapes(coarse_fem) -> None:
    n_t, n_b = coarse_fem.n_top_nodes, coarse_fem.n_bottom_nodes
    assert coarse_fem.Ctt_force.shape == (2 * n_t, 2 * n_t)
    assert coarse_fem.Cbb_force.shape == (2 * n_b, 2 * n_b)
    assert coarse_fem.Ctb_force.shape == (2 * n_t, 2 * n_b)
    assert coarse_fem.reciprocity_error < 1e-8


# 46-47. Resultants are nodal sums; Mz uses both components and actual coordinates.
def test_moment_uses_both_components(coarse_lookup) -> None:
    y_top = np.asarray(coarse_lookup.y_top)
    assert np.ptp(y_top) > 0.01 * coarse_lookup.L
    row = coarse_lookup.full_contact_row()
    fx = coarse_lookup.top_force_x_basis[row]
    fy = coarse_lookup.top_force_y_basis[row]
    mz = coarse_lookup.x_top @ fy.T - y_top @ fx.T
    np.testing.assert_allclose(coarse_lookup.scalar_lookup[row, :, SCALAR_MZ], mz, rtol=1e-12, atol=1e-12 * np.abs(mz).max())


# 48-49. Curved bottom ordered heel to toe; interval enumeration.
def test_bottom_order_and_enumeration(coarse_lookup) -> None:
    n_b = coarse_lookup.n_bottom_nodes
    assert np.all(np.diff(coarse_lookup.x_bottom) > 0.0)
    assert coarse_lookup.n_records == n_intervals(n_b) == n_b * (n_b + 1) // 2
    starts, ends = coarse_lookup.contact_start_index, coarse_lookup.contact_end_index
    assert np.all(starts <= ends)
    assert len({(int(i), int(j)) for i, j in zip(starts, ends)}) == coarse_lookup.n_records


# 50. The closure column places active nodes on the ground.
def test_closure_places_contact_on_ground(coarse_lookup) -> None:
    y_b = np.asarray(coarse_lookup.y_bottom)
    assert y_b.min() < 0.0 < y_b.max()  # genuinely curved, partly below the heel datum
    for row in coarse_lookup.valid_rows[::97]:
        iv = coarse_lookup.interval(row)
        c = iv.contact_node_ids
        y_a = coarse_lookup.anchor_reference_y[row]
        np.testing.assert_allclose(
            y_b[c] + coarse_lookup.bottom_v_basis[row, 0, c] - y_a, 0.0, atol=1e-12 * coarse_lookup.L
        )


def _fixed_frame_state(lookup, sel):
    from compliance_fem.corotation import affine_coefficients

    row = sel.selected_row
    varphi = float(sel.varphi)
    gamma = np.array([sel.alpha, sel.d_ax, sel.d_ay, np.cos(varphi) - 1.0, -np.sin(varphi)])
    coef = affine_coefficients(gamma)
    u = coef @ lookup.bottom_u_basis[row]
    v = coef @ lookup.bottom_v_basis[row]
    rx = coef @ lookup.reaction_x_basis[row]
    ry = coef @ lookup.reaction_y_basis[row]
    c, s = np.cos(varphi), np.sin(varphi)
    xa, ya = lookup.anchor_reference_x[row], lookup.anchor_reference_y[row]
    gap = s * (lookup.x_bottom - xa + u - sel.d_ax) + c * (lookup.y_bottom - ya + v - sel.d_ay)
    rn = s * rx + c * ry
    return lookup.interval(row), gap, rn


# 51-54. Full fixed-frame gaps on every free node; rotated reactions on every contact node.
@pytest.mark.parametrize("phi,theta", [(0.0, 0.0), (12.0, 5.0), (-14.0, 20.0)])
def test_fixed_frame_gap_and_reaction(coarse_lookup, phi, theta) -> None:
    _, sel = _selection(coarse_lookup, phi, theta)
    iv, gap, rn = _fixed_frame_state(coarse_lookup, sel)
    free, contact = iv.sets()
    if free.size:
        assert sel.min_free_gap == pytest.approx(gap[free].min(), abs=1e-12 * coarse_lookup.L)
        assert sel.max_free_penetration == pytest.approx(max(0.0, -gap[free].min()), abs=1e-12 * coarse_lookup.L)
    np.testing.assert_allclose(gap[contact], 0.0, atol=1e-10 * coarse_lookup.L)
    assert sel.min_contact_reaction == pytest.approx(rn[contact].min(), rel=1e-9, abs=1e-9)
    shape = build_shape_plot_data(coarse_lookup, sel)
    np.testing.assert_allclose(shape.contact_curve_y, 0.0, atol=1e-9 * coarse_lookup.L)


# 55. Lookup superposition agrees with direct FEM (heel, interior, toe, full).
def test_lookup_matches_direct_fem(coarse_fem) -> None:
    n_b = coarse_fem.n_bottom_nodes
    specs = {
        "heel": ContactInterval(0, n_b // 3, n_b),
        "interior": ContactInterval(n_b // 4, (3 * n_b) // 4, n_b),
        "toe": ContactInterval((2 * n_b) // 3, n_b - 1, n_b),
        "full": ContactInterval(0, n_b - 1, n_b),
    }
    out = run_standard_direct_comparisons(coarse_fem, A_SOFT, KAPPA, intervals=specs)
    for name, comps in out.items():
        worst = max(c.max_rel_error for c in comps)
        assert worst < 1e-6, (name, worst)
        assert max(c.free_traction_rel for c in comps) < 1e-8


# 56. Passive toe equilibrium with k = 25 N m/rad.
def test_passive_toe_exact(coarse_lookup) -> None:
    from compliance_fem.gait.passive_toe import solve_passive_toe_candidates
    from compliance_fem.toe_spring import ToeSpringConfig

    cfg = ToeSpringConfig()
    assert cfg.toe_stiffness_Nm_per_rad == 25.0
    cands = solve_passive_toe_candidates(
        coarse_lookup, Fx_star=150.0, Fy_star=-1500.0, Mz_meas=0.0, phi_rad=-0.05, config=cfg, width_m=0.1
    )
    converged = [c for c in cands if c.toe.converged]
    assert converged
    for c in converged:
        assert c.toe.toe_equilibrium_relative_residual < 1e-9
        assert c.force_residual <= 1e-9 * np.hypot(150.0, 1500.0)


# 57. No runtime FEM solve when gait inputs change.
def test_no_runtime_fem_solve(coarse_lookup, monkeypatch) -> None:
    import scipy.sparse.linalg as spla

    import compliance_fem.compliance as comp

    def _forbidden(*_a, **_k):
        raise AssertionError("FEM solve during runtime evaluation")

    for mod, name in ((spla, "splu"), (spla, "spsolve"), (comp, "splu"), (comp, "compute_compliance")):
        monkeypatch.setattr(mod, name, _forbidden)
    for phi, theta, fy in ((0.0, 0.0, -800.0), (8.0, 3.0, -1500.0), (-12.0, 15.0, -2500.0)):
        _, sel = _selection(coarse_lookup, phi, theta, Fy=fy)
        build_shape_plot_data(coarse_lookup, sel)


# ---------------------------------------------------------------------------
# Serialization and GUI (58-67)
# ---------------------------------------------------------------------------


# 58-61. Geometry metadata, shoe length, tags and the toe convention survive serialization.
def test_lookup_serialization(saved_lookup, coarse_lookup) -> None:
    _, lk = saved_lookup
    assert lk.schema_version == LOOKUP_SCHEMA_VERSION and lk.is_measured
    gm, ref = lk.geometry_metadata, coarse_lookup.geometry_metadata
    assert gm["shoe_length_mm"] == SHOE_MM
    for key in ("normalized_landmarks", "physical_landmarks", "plate_start_reference_coordinate",
                "plate_end_reference_coordinate", "top_surface_reference_coordinates",
                "bottom_surface_reference_coordinates", "foam_interface_reference_coordinates"):
        np.testing.assert_allclose(np.asarray(_flat(gm[key])), np.asarray(_flat(ref[key])), rtol=0, atol=0)
    assert gm["foam_region_tags"] == list(FOAM_REGION_TAGS)
    assert gm["boundary_tags"] == list(BOUNDARY_TAGS)
    assert gm["shared_toe_policy"] == SHARED_TOE_POLICY
    assert gm["shared_toe_node_id"] == int(coarse_lookup.measured_render["bottom_curve_node_ids"][-1])
    for key, arr in coarse_lookup.measured_render.items():
        np.testing.assert_array_equal(lk.measured_render[key], arr)
    np.testing.assert_array_equal(lk.plate_node_ids, coarse_lookup.plate_node_ids)
    np.testing.assert_array_equal(lk.rigid_alpha_basis, coarse_lookup.rigid_alpha_basis)


def _flat(obj):
    if isinstance(obj, dict):
        return [v for k in sorted(obj) for v in _flat(obj[k])]
    return np.asarray(obj, dtype=float).ravel().tolist()


def test_compliance_npz_roundtrip(coarse_fem, coarse_cfg, tmp_path) -> None:
    path = tmp_path / "compliance_results.npz"
    save_compliance_npz(coarse_fem, path)
    assert is_measured_compliance_npz(path)
    cfg = measured_config_from_compliance_npz(path)
    for f in ("shoe_length_mm", "upper_foam_material", "lower_foam_material", "EI_plate", "mesh_size",
              "ffturbo_E", "ffleap_E_heel", "ffleap_E_toe", "toe_refinement"):
        assert getattr(cfg, f) == getattr(coarse_cfg, f), f
    assert cfg.normalized_geometry.to_payload() == coarse_cfg.normalized_geometry.to_payload()


# 62. Rectangle and layered modes are unchanged.
def test_simple_modes_unchanged() -> None:
    from conftest import small_config

    from compliance_fem.boundaries import vector_boundary_data
    from compliance_fem.config import LayeredPlateConfig

    rect = compute_compliance(small_config())
    assert rect.compliance_schema_version == 3 and rect.plate_node_ids is None
    u, v, _, _ = vector_boundary_data(rect.basis, "bottom")
    np.testing.assert_array_equal(rect.selector_dofs("bottom")[0], u)
    lay_cfg = LayeredPlateConfig(
        L=0.30, h1_heel=0.025, h1_toe=0.015, h2_heel=0.020, h2_toe=0.030, E1=2.0e6, nu1=0.30,
        E_heel=5.0e5, E_toe=1.5e6, nu2=0.30, EI_plate=10.0, nx=8, ny1=2, ny2=2,
    )
    lay = compute_compliance(lay_cfg)
    assert lay.compliance_schema_version == 3 and lay.geometry_metadata is None
    coords = np.vstack([lay.x_plate, lay.y_plate])
    _, t, n = plate_element_frames(coords)
    K_global = assemble_plate_bending(lay.basis.N, lay.plate_u_dof_ids, lay.plate_v_dof_ids, coords, lay_cfg.plate_normal, 2.0)
    K_local = assemble_plate_bending(lay.basis.N, lay.plate_u_dof_ids, lay.plate_v_dof_ids, coords, n, 2.0)
    assert abs(K_global - K_local).max() <= 1e-12 * abs(K_global).max()
    B1 = assemble_axial_constraints(lay.n_primal, lay.plate_u_dof_ids, lay.plate_v_dof_ids, lay_cfg.plate_tangent, lay.n_plate_rotation_dofs)
    B2 = assemble_axial_constraints(lay.n_primal, lay.plate_u_dof_ids, lay.plate_v_dof_ids, t, lay.n_plate_rotation_dofs)
    assert abs(B1 - B2).max() <= 1e-14


# 63. Changing the shoe length scales the whole geometry uniformly.
def test_gui_scaling_uniform(meas_cfg) -> None:
    g1 = build_measured_geometry(meas_cfg.normalized_geometry, 250.0)
    g2 = build_measured_geometry(meas_cfg.normalized_geometry, 300.0)
    k = 300.0 / 250.0
    for a, b in ((g1.top, g2.top), (g1.bottom, g2.bottom), (g1.interface, g2.interface), (g1.exterior, g2.exterior)):
        np.testing.assert_allclose(b, k * a, rtol=1e-12, atol=1e-15)
    for name in g1.region_polygons:
        np.testing.assert_allclose(g2.region_polygons[name], k * g1.region_polygons[name], rtol=1e-12, atol=1e-15)
    assert g2.total_area == pytest.approx(k**2 * g1.total_area, rel=1e-12)


# 64. GUI colours follow the material of each region.
def test_gui_material_colors(coarse_lookup) -> None:
    from compliance_fem.app import MATERIAL_FILL_COLORS, _draw_shape

    _, sel = _selection(coarse_lookup)
    for upper, lower in (("FFTurbo", "FFLeap"), ("FFLeap", "FFTurbo")):
        lk = dataclasses.replace(
            coarse_lookup,
            geometry_metadata={**coarse_lookup.geometry_metadata, "upper_foam_material": upper, "lower_foam_material": lower},
        )
        fig = _draw_shape(build_shape_plot_data(lk, sel))
        fills = {tr.name: tr.fillcolor for tr in fig.data if getattr(tr, "fill", None) == "toself"}
        assert fills[f"upper foam ({upper})"] == MATERIAL_FILL_COLORS[upper]
        assert fills[f"lower foam ({lower})"] == MATERIAL_FILL_COLORS[lower]


# 38/65. The rendered plate is the partial span and stays attached to the interface.
def test_gui_plate_partial_span(coarse_lookup) -> None:
    _, sel = _selection(coarse_lookup, 6.0, 4.0)
    sh = build_shape_plot_data(coarse_lookup, sel, show_plate_nodes=True)
    gm = coarse_lookup.geometry_metadata
    np.testing.assert_allclose(
        [coarse_lookup.plate_reference_x[0], coarse_lookup.plate_reference_y[0]],
        gm["plate_start_reference_coordinate"], atol=1e-12 * L_M,
    )
    np.testing.assert_allclose(
        [coarse_lookup.plate_reference_x[-1], coarse_lookup.plate_reference_y[-1]],
        gm["plate_end_reference_coordinate"], atol=1e-12 * L_M,
    )
    tol = 1e-9 * L_M
    assert sh.plate_x_def[0] == pytest.approx(sh.plate_node_x[0], abs=tol)
    assert sh.plate_x_def[-1] == pytest.approx(sh.plate_node_x[-1], abs=tol)
    sec = coarse_lookup.measured_render
    iface = np.asarray(sec["interface_node_ids"])
    on_plate = np.isin(iface, sec["plate_curve_node_ids"])
    np.testing.assert_allclose(sh.interface_def[0][on_plate], sh.plate_node_x, atol=1e-7 * L_M)
    np.testing.assert_allclose(sh.interface_def[1][on_plate], sh.plate_node_y, atol=1e-7 * L_M)
    assert not np.all(on_plate)


# 66. Contact shading follows the deformed curved bottom.
def test_gui_contact_follows_bottom(coarse_lookup) -> None:
    _, sel = _selection(coarse_lookup, -8.0, 10.0)
    sh = build_shape_plot_data(coarse_lookup, sel)
    i, j = sel.interval
    np.testing.assert_allclose(sh.contact_curve_x, sh.x_bottom_def[i : j + 1], atol=1e-9 * L_M)
    np.testing.assert_allclose(sh.contact_curve_y, sh.y_bottom_def[i : j + 1], atol=1e-9 * L_M)
    assert sh.is_measured and set(sh.region_def) == set(FOAM_REGION_TAGS)


# 67. Incompatible old schemas are identified clearly.
def test_old_schema_messages(saved_lookup, tmp_path) -> None:
    path, _ = saved_lookup
    data = dict(np.load(path, allow_pickle=True))
    for version, match in ((8, "schema_version=8"), (9, "cannot contain measured-sole")):
        old = dict(data)
        old["schema_version"] = np.asarray(version)
        np.savez_compressed(tmp_path / f"v{version}.npz", **old)
        with pytest.raises(ValueError, match=match):
            load_contact_lookup(tmp_path / f"v{version}.npz")
    assert "schema_version=10" in REGENERATE_LOOKUP_MESSAGE
    broken = {k: v for k, v in data.items() if k != "render_influence_loads"}
    np.savez_compressed(tmp_path / "broken.npz", **broken)
    with pytest.raises(ValueError, match="missing geometry arrays"):
        load_contact_lookup(tmp_path / "broken.npz")
