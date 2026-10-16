"""Measured carbon-plated sole: geometry, mesh, partial plate, contact, serialization, GUI.

Numbered comments follow the measured-sole specification (tests 1-67). The
full-resolution 270 mm model is used for geometry / mesh / plate checks; a
subsampled copy of the same CSV with a coarse mesh keeps the contact tests fast.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from compliance_fem.fem.compliance import (
    SOLE_SCALAR_FIELDS,
    compute_compliance,
    save_compliance_npz,
)
from compliance_fem.contact.sole_config import SoleConfig, validate_shoe_length_mm
from compliance_fem.contact.contact_model import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    ContactModel,
    build_contact_model,
    load_contact_model,
    save_contact_model,
    selection_proxy,
)
from compliance_fem.contact.continuous_contact import ContactSolution, solve_contact_state
from compliance_fem.contact.continuous_toe import solve_toe_contact
from compliance_fem.contact.toe_spring import ToeSpringConfig
from compliance_fem.geometry.profile import (
    BOUNDARY_TAGS,
    FOAM_REGION_TAGS,
    SHARED_TOE_POLICY,
    GeometryFileError,
    build_sole_geometry,
    distance_to_polyline,
    load_normalized_sole_csv,
)
from compliance_fem.geometry.mesh import MeshQualityError, generate_mesh
from compliance_fem.fem.plate import assemble_plate_bending, plate_element_frames
from compliance_fem.plotting.shape_render import build_shape_plot_data

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "geometry" / "sole_geometry.csv"
SHOE_MM = 270.0
L_M = SHOE_MM / 1000.0
TOE_LENGTH_SOFT = 0.22 * L_M
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
    out = [r for r in rows if r["record_type"] == "plate"]
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
def meas_cfg() -> SoleConfig:
    return SoleConfig(shoe_length_mm=SHOE_MM, geometry_csv=str(CSV_PATH))


@pytest.fixture(scope="module")
def meas_geom(meas_cfg):
    return build_sole_geometry(meas_cfg.normalized_geometry, SHOE_MM)


@pytest.fixture(scope="module")
def meas_fem(meas_cfg):
    return compute_compliance(meas_cfg)


@pytest.fixture(scope="module")
def meas_mesh(meas_fem):
    return meas_fem.mesh_data


@pytest.fixture(scope="module")
def coarse_csv(tmp_path_factory) -> Path:
    return _write_rows(tmp_path_factory.mktemp("coarse") / "coarse_sole.csv", _subsampled_rows())


@pytest.fixture(scope="module")
def coarse_cfg(coarse_csv) -> SoleConfig:
    return SoleConfig(shoe_length_mm=SHOE_MM, geometry_csv=str(coarse_csv), mesh_size_m=0.010)


@pytest.fixture(scope="module")
def coarse_fem(coarse_cfg):
    return compute_compliance(coarse_cfg)




@pytest.fixture(scope="module")
def coarse_model(coarse_fem) -> ContactModel:
    with pytest.warns(UserWarning):
        return build_contact_model(coarse_fem, toe_length=TOE_LENGTH_SOFT, kappa=KAPPA)


@pytest.fixture(scope="module")
def saved_model(coarse_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("meas_lookup")
    path = save_contact_model(coarse_model, out)
    return path, load_contact_model(path)


def _solve(model: ContactModel, phi_deg=0.0, theta_deg=0.0, Fx=0.0, Fy=-1500.0) -> ContactSolution:
    """Contact state at absolute chord angle ``phi_deg`` and toe angle ``theta_deg`` (per-metre loads)."""
    varphi = float(np.deg2rad(phi_deg)) - model.phi_ref
    return solve_contact_state(model.operator, varphi, float(np.tan(np.deg2rad(theta_deg))), np.array([Fx, Fy]))


# ---------------------------------------------------------------------------
# Coordinate conversion (1-9)
# ---------------------------------------------------------------------------


# 1. shoe_length_mm is required.
def test_shoe_length_required() -> None:
    with pytest.raises(ValueError, match="shoe_length_mm is required"):
        SoleConfig(shoe_length_mm=None, geometry_csv=str(CSV_PATH))  # pyright: ignore[reportArgumentType]  (deliberately invalid)


# 2. Nonpositive / nonfinite lengths are rejected.
@pytest.mark.parametrize("bad", [0.0, -270.0, float("nan"), float("inf"), "abc"])
def test_bad_shoe_length_rejected(bad) -> None:
    with pytest.raises(ValueError):
        validate_shoe_length_mm(bad)
    with pytest.raises(ValueError):
        SoleConfig(shoe_length_mm=bad, geometry_csv=str(CSV_PATH))


# 3-4. Toe maps to L mm, heel-bottom to the origin.
@pytest.mark.parametrize("length_mm", [240.0, 270.0, 310.0])
def test_toe_and_heel_mapping(meas_cfg, length_mm) -> None:
    g = build_sole_geometry(meas_cfg.normalized_geometry, length_mm)
    assert g.toe_tip[0] == pytest.approx(length_mm / 1000.0, rel=1e-12)
    np.testing.assert_allclose(g.heel_bottom, [0.0, 0.0], atol=1e-15)


# 5-6. One common scale for both axes; aspect ratio preserved.
def test_uniform_scale_and_aspect(meas_cfg, meas_geom) -> None:
    norm = meas_cfg.normalized_geometry
    s = SHOE_MM / 1000.0
    # Every supplied sample appears, scaled by s, among the SI curve vertices.
    for name, curve in (("top_surface", meas_geom.top), ("bottom_surface", meas_geom.bottom)):
        for q in s * np.asarray(norm.curves[name]):
            assert np.min(np.hypot(*(curve - q).T)) <= 1e-15
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
    np.testing.assert_array_equal(g.top[0], g.heel_top)
    np.testing.assert_array_equal(g.top[-1], g.toe_tip)
    np.testing.assert_array_equal(g.bottom[0], g.heel_bottom)
    np.testing.assert_array_equal(g.bottom[-1], g.toe_tip)
    for curve in (g.top, g.bottom, g.interface):
        assert np.all(np.diff(curve[:, 0]) > 0.0)
    x = np.linspace(0.05, 0.9, 20) * L_M
    assert np.all(np.interp(x, g.top[:, 0], g.top[:, 1]) > np.interp(x, g.bottom[:, 0], g.bottom[:, 1]))
    assert g.total_area > 0.0
    assert g.toe_tip[1] > 0.0


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
    assert a.plate_x == b.plate_x
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
        rows = [r for r in rows if r["name"] != "toe_end"]
    with pytest.raises(GeometryFileError):
        load_normalized_sole_csv(_write_rows(tmp_path / f"{defect}.csv", rows))


# 12. Curves are heel-to-toe; 13. the exterior closes exactly.
def test_exterior_closes(meas_geom) -> None:
    g = meas_geom
    np.testing.assert_array_equal(g.bottom[-1], g.top[-1])
    np.testing.assert_array_equal(g.top[0], g.heel_edge[0])
    np.testing.assert_array_equal(g.heel_edge[-1], g.bottom[0])
    loop = np.vstack([g.bottom, g.top[::-1][1:], g.heel_edge[1:-1]])
    assert len({tuple(p) for p in loop}) == len(loop)  # closes without repeated vertices


# 14. Heel edge is the straight heel_top -> heel_bottom segment.
def test_heel_edge_straight(meas_geom) -> None:
    g = meas_geom
    a, b = g.heel_top, g.heel_bottom
    np.testing.assert_array_equal(g.heel_edge[0], a)
    np.testing.assert_array_equal(g.heel_edge[-1], b)
    d = b - a
    for p in g.heel_edge:
        cross = abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / np.linalg.norm(d)
        assert cross <= 1e-9 * L_M


# 15. Top and bottom share exactly one vertex, the toe.
def test_single_toe_vertex(meas_geom) -> None:
    shared = {tuple(p) for p in meas_geom.top} & {tuple(p) for p in meas_geom.bottom}
    assert shared == {tuple(meas_geom.toe_tip)}


# 16. The exterior cannot fold over: a top surface dipping below the bottom is rejected.
def test_top_below_bottom_rejected(meas_cfg) -> None:
    norm = meas_cfg.normalized_geometry
    curves = {k: np.array(v, dtype=float) for k, v in norm.curves.items()}
    curves["top_surface"][50, 1] = -0.1
    bad = dataclasses.replace(norm, curves=curves)
    with pytest.raises(GeometryFileError, match="above bottom_surface"):
        build_sole_geometry(bad, SHOE_MM)


# 17-18. All three corners convex; old-format landmark point rows are rejected.
def test_corners_convex_and_point_rows_rejected(meas_geom, tmp_path) -> None:
    assert set(meas_geom.corner_angles_deg) == {"heel_bottom", "toe_tip", "heel_top"}
    assert all(0.0 < ang < 180.0 for ang in meas_geom.corner_angles_deg.values())
    rows = _read_rows(CSV_PATH)
    rows.append({"record_type": "point", "name": "toe_tip", "index": "", "x_over_length": "1", "y_over_length": "0.19"})
    with pytest.raises(GeometryFileError, match="no longer used"):
        load_normalized_sole_csv(_write_rows(tmp_path / "old_format.csv", rows))


# 19. Interface endpoints lie on the exterior.
def test_interface_endpoints_on_exterior(meas_geom) -> None:
    g = meas_geom
    closed = np.vstack([g.bottom, g.top[::-1][1:], g.heel_edge[1:]])
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
    upper = MplPath(meas_mesh.geometry.region_polygons["upper_foam"])
    in_upper = upper.contains_points(c)
    np.testing.assert_array_equal(in_upper, meas_mesh.element_region == 0)
    perm = np.random.default_rng(0).permutation(c.shape[0])
    np.testing.assert_array_equal(
        upper.contains_points(c[perm]),
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
    assert np.all(area > 1e-6 * meas_mesh.config.mesh_size_m**2)


# 27. Mesh quality is enforced near the point toe.
def test_toe_quality_enforced(meas_mesh, meas_cfg) -> None:
    q = meas_mesh.quality
    assert q["toe_elements"] >= 1
    assert q["toe_min_angle_deg"] >= meas_cfg.min_angle_deg
    assert q["toe_min_angle_deg"] <= meas_mesh.geometry.corner_angles_deg["toe_tip"] + 1e-9
    strict = dataclasses.replace(meas_cfg, min_angle_deg=58.0)
    with pytest.raises(MeshQualityError):
        generate_mesh(strict)


# 28. Unequal top / bottom node counts.
def test_unequal_node_counts(meas_fem, coarse_fem) -> None:
    for fem in (meas_fem, coarse_fem):
        assert fem.n_top_nodes != fem.n_bottom_nodes


# ---------------------------------------------------------------------------
# Partial plate (29-38)
# ---------------------------------------------------------------------------


# 29-30. Plate endpoints sit on the interface at the configured x and are inserted.
def test_plate_endpoints_inserted(meas_geom, meas_cfg) -> None:
    g = meas_geom
    raw = np.asarray(meas_cfg.normalized_geometry.curves["foam_interface"]) * L_M
    for idx, x in zip((g.plate_heel_index, g.plate_toe_index), meas_cfg.normalized_geometry.plate_x):
        end = g.interface[idx]
        assert end[0] == pytest.approx(x * L_M, rel=1e-12)
        assert end[1] == pytest.approx(np.interp(end[0], raw[:, 0], raw[:, 1]), abs=1e-12 * L_M)
        assert np.min(np.hypot(*(raw - end).T)) > 1e-6 * L_M  # inserted, not snapped onto a sample


# 31. Plate elements lie between the endpoints only.
def test_plate_elements_within_span(meas_fem, meas_geom) -> None:
    p = meas_fem.basis.mesh.p[:, meas_fem.plate_node_ids]
    np.testing.assert_allclose(p[:, 0], meas_geom.interface[meas_geom.plate_heel_index], atol=1e-12 * L_M)
    np.testing.assert_allclose(p[:, -1], meas_geom.interface[meas_geom.plate_toe_index], atol=1e-12 * L_M)
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
    assert meas_fem.inextensibility_residuals.top <= 1e-12 * L_M


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
    toe = meas_mesh.geometry.toe_tip
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
def test_top_force_no_double_count(coarse_model, coarse_fem) -> None:
    n_t = coarse_fem.n_top_nodes
    assert n_t == len(coarse_fem.mesh_data.top_curve_nodes) - 1
    assert coarse_model.n_top_nodes == n_t
    sol = _solve(coarse_model, 4.0, 3.0, Fx=150.0)
    assert sol.F_t.shape == (2 * n_t,)
    d = sol.diagnostics
    assert d["Fx_local"] == pytest.approx(float(sol.F_t[:n_t].sum()), rel=1e-12)
    assert d["Fy_local"] == pytest.approx(float(sol.F_t[n_t:].sum()), rel=1e-12)
    # Top resultant balances the bottom reactions.
    assert d["equilibrium_force_residual"] <= 1e-8 * np.hypot(150.0, 1500.0)


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


# 48. Curved bottom ordered heel to toe.
def test_bottom_order(coarse_model) -> None:
    assert np.all(np.diff(coarse_model.x_bottom) > 0.0)
    assert np.ptp(coarse_model.y_top) > 0.01 * coarse_model.L


# 51-54, 66. Fixed-frame gaps on every node; contact shading follows the deformed bottom on the ground.
@pytest.mark.parametrize("phi,theta", [(0.0, 0.0), (12.0, 5.0), (-14.0, 20.0)])
def test_fixed_frame_gap_and_reaction(coarse_model, phi, theta) -> None:
    sol = _solve(coarse_model, phi, theta)
    L = coarse_model.L
    act = sol.active
    assert act.any()
    np.testing.assert_allclose(sol.gap[act], 0.0, atol=1e-9 * L)
    assert np.all(sol.gap[~act] >= -1e-9 * L)
    assert np.all(sol.lam[act] >= -1e-9 * np.abs(sol.lam).max())
    np.testing.assert_array_equal(sol.lam[~act], 0.0)
    assert sol.diagnostics["min_gap"] == pytest.approx(float(sol.gap.min()), abs=1e-15)
    sh = build_shape_plot_data(coarse_model, selection_proxy(coarse_model, sol))
    assert sh.contact_curves and len(sh.contact_curves) == len(sh.patches)
    for (cx, cy), p in zip(sh.contact_curves, sh.patches):
        np.testing.assert_allclose(cy, 0.0, atol=1e-9 * L)
        np.testing.assert_allclose(cx, sh.bottom_node_x[p.start : p.end + 1], atol=1e-12 * L)
    np.testing.assert_allclose(sh.bottom_gap, sol.gap)
    assert sh.has_render_mesh and set(sh.region_def) == set(FOAM_REGION_TAGS)


# 56. Passive toe equilibrium with k = 25 N m/rad.
def test_passive_toe_exact(coarse_model) -> None:
    cfg = ToeSpringConfig()
    assert cfg.toe_stiffness_Nm_per_rad == 25.0
    F = np.array([150.0, -1500.0])
    fr = solve_toe_contact(coarse_model.operator, F, -0.05, cfg, 0.1)
    assert fr.converged
    assert fr.toe_equilibrium_relative_residual < 1e-9
    assert fr.contact.diagnostics["force_residual"] <= 1e-9 * np.hypot(*F)


# 57. No runtime FEM solve when gait inputs change.
def test_no_runtime_fem_solve(coarse_model, monkeypatch) -> None:
    import scipy.sparse.linalg as spla

    import compliance_fem.fem.compliance as comp

    def _forbidden(*_a, **_k):
        raise AssertionError("FEM solve during runtime evaluation")

    for mod, name in ((spla, "splu"), (spla, "spsolve"), (comp, "splu"), (comp, "compute_compliance")):
        monkeypatch.setattr(mod, name, _forbidden)
    for phi, fy in ((0.0, -800.0), (0.14, -1500.0), (-0.2, -2500.0)):
        fr = solve_toe_contact(coarse_model.operator, np.array([0.1 * abs(fy), fy]), phi, ToeSpringConfig(), 0.1)
        build_shape_plot_data(coarse_model, selection_proxy(coarse_model, fr))


# ---------------------------------------------------------------------------
# Serialization and GUI (58-67)
# ---------------------------------------------------------------------------


# 58-61. Geometry metadata, shoe length, tags and the toe convention survive serialization.
def test_lookup_serialization(saved_model, coarse_model) -> None:
    _, lk = saved_model
    assert lk.schema_version == LOOKUP_SCHEMA_VERSION
    gm, ref = lk.geometry_metadata, coarse_model.geometry_metadata
    assert gm["shoe_length_mm"] == SHOE_MM
    for key in ("plate_x_over_length", "physical_landmarks", "plate_start_reference_coordinate",
                "plate_end_reference_coordinate", "top_surface_reference_coordinates",
                "bottom_surface_reference_coordinates", "foam_interface_reference_coordinates"):
        np.testing.assert_allclose(np.asarray(_flat(gm[key])), np.asarray(_flat(ref[key])), rtol=0, atol=0)
    assert gm["foam_region_tags"] == list(FOAM_REGION_TAGS)
    assert gm["boundary_tags"] == list(BOUNDARY_TAGS)
    assert gm["shared_toe_policy"] == SHARED_TOE_POLICY
    assert gm["shared_toe_node_id"] == int(coarse_model.render_section["bottom_curve_node_ids"][-1])
    for key, arr in coarse_model.render_section.items():
        np.testing.assert_array_equal(lk.render_section[key], arr)
    np.testing.assert_array_equal(lk.plate_mesh.node_ids, coarse_model.plate_mesh.node_ids)
    np.testing.assert_array_equal(lk.operator.G_bb, coarse_model.operator.G_bb)
    np.testing.assert_array_equal(lk.operator.psi_top, coarse_model.operator.psi_top)


def _flat(obj):
    if isinstance(obj, dict):
        return [v for k in sorted(obj) for v in _flat(obj[k])]
    return np.asarray(obj, dtype=float).ravel().tolist()


def test_compliance_npz_stores_settings(coarse_fem, coarse_cfg, tmp_path) -> None:
    path = tmp_path / "compliance_results.npz"
    save_compliance_npz(coarse_fem, path)
    data = np.load(path)
    for name in SOLE_SCALAR_FIELDS:
        assert data[name].item() == getattr(coarse_cfg, name), name
    stored = json.loads(str(data["normalized_geometry_json"]))
    assert stored == json.loads(json.dumps(coarse_cfg.normalized_geometry.to_payload()))


# 63. Changing the shoe length scales the whole geometry uniformly.
def test_gui_scaling_uniform(meas_cfg) -> None:
    g1 = build_sole_geometry(meas_cfg.normalized_geometry, 250.0)
    g2 = build_sole_geometry(meas_cfg.normalized_geometry, 300.0)
    k = 300.0 / 250.0
    for a, b in ((g1.top, g2.top), (g1.bottom, g2.bottom), (g1.interface, g2.interface)):
        np.testing.assert_allclose(b, k * a, rtol=1e-12, atol=1e-15)
    for name in g1.region_polygons:
        np.testing.assert_allclose(g2.region_polygons[name], k * g1.region_polygons[name], rtol=1e-12, atol=1e-15)
    assert g2.total_area == pytest.approx(k**2 * g1.total_area, rel=1e-12)


# 64. GUI colours follow the material of each region.
def test_gui_material_colors(coarse_model) -> None:
    from compliance_fem.gui.app import MATERIAL_FILL_COLORS, _draw_shape

    sol = _solve(coarse_model)
    for upper, lower in (("FFTurbo", "FFLeap"), ("FFLeap", "FFTurbo")):
        lk = dataclasses.replace(
            coarse_model,
            geometry_metadata={**coarse_model.geometry_metadata, "upper_foam_material": upper, "lower_foam_material": lower},
        )
        fig = _draw_shape(build_shape_plot_data(lk, selection_proxy(lk, sol)))
        fills = {tr.name: tr.fillcolor for tr in fig.data if getattr(tr, "fill", None) == "toself"}  # pyright: ignore[reportAttributeAccessIssue]  (plotly leaves Figure.data untyped)
        assert fills[f"upper foam ({upper})"] == MATERIAL_FILL_COLORS[upper]
        assert fills[f"lower foam ({lower})"] == MATERIAL_FILL_COLORS[lower]


# 38/65. The rendered plate is the partial span and stays attached to the interface.
def test_gui_plate_partial_span(coarse_model) -> None:
    sol = _solve(coarse_model, 6.0, 4.0)
    sh = build_shape_plot_data(coarse_model, selection_proxy(coarse_model, sol), show_plate_nodes=True)
    gm = coarse_model.geometry_metadata
    mesh = coarse_model.plate_mesh
    np.testing.assert_allclose(
        [mesh.reference_x[0], mesh.reference_y[0]], gm["plate_start_reference_coordinate"], atol=1e-12 * L_M
    )
    np.testing.assert_allclose(
        [mesh.reference_x[-1], mesh.reference_y[-1]], gm["plate_end_reference_coordinate"], atol=1e-12 * L_M
    )
    tol = 1e-9 * L_M
    assert sh.plate_x_def is not None and sh.plate_node_x is not None and sh.plate_node_y is not None
    assert sh.interface_def is not None
    assert sh.plate_x_def[0] == pytest.approx(sh.plate_node_x[0], abs=tol)
    assert sh.plate_x_def[-1] == pytest.approx(sh.plate_node_x[-1], abs=tol)
    sec = coarse_model.render_section
    iface = np.asarray(sec["interface_node_ids"])
    on_plate = np.isin(iface, sec["plate_curve_node_ids"])
    np.testing.assert_allclose(sh.interface_def[0][on_plate], sh.plate_node_x, atol=1e-7 * L_M)
    np.testing.assert_allclose(sh.interface_def[1][on_plate], sh.plate_node_y, atol=1e-7 * L_M)
    assert not np.all(on_plate)


# 67. Incompatible old schemas are identified clearly.
def test_old_schema_messages(saved_model, tmp_path) -> None:
    path, _ = saved_model
    with np.load(path, allow_pickle=True) as npz:
        data = dict(npz)
    for version in (8, 9, 11, 12):
        old = dict(data)
        old["schema_version"] = np.asarray(version)
        np.savez_compressed(tmp_path / f"v{version}.npz", **old)
        with pytest.raises(ValueError, match=f"schema_version={version}"):
            load_contact_model(tmp_path / f"v{version}.npz")
    assert f"schema_version={LOOKUP_SCHEMA_VERSION}" in REGENERATE_LOOKUP_MESSAGE
    assert LOOKUP_SCHEMA_VERSION == 13
    broken = {k: v for k, v in data.items() if k != "render_influence_loads"}
    np.savez_compressed(tmp_path / "broken.npz", **broken)
    with pytest.raises(ValueError, match="Render section is missing"):
        load_contact_model(tmp_path / "broken.npz")
