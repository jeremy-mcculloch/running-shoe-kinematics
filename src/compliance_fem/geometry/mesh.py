"""Unstructured constrained triangulation of the measured two-foam sole.

Every sampled and inserted vertex of the exterior boundary, the foam interface,
and both plate endpoints becomes a Gmsh geometry point, so all of them are mesh
vertices. The two foam regions are separate plane surfaces sharing the interface
lines, which makes the interface conforming (one translational node per
interface location, shared by both foams). Boundary and material tags come
straight from the Gmsh entities; no coordinate equality test is used.

The output is the mesh / region-tag / boundary-tag / plate-tag structure
consumed by the compliance solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTri

from compliance_fem.geometry.gmsh_util import gmsh_session, initialize_gmsh
from compliance_fem.geometry.profile import (
    REGION_LOWER,
    REGION_UPPER,
    SHARED_TOE_POLICY,
    TAG_BOTTOM,
    TAG_HEEL,
    TAG_INTERFACE,
    TAG_PLATE,
    TAG_TOP,
    SoleGeometry,
    distance_to_polyline,
    geometry_from_config,
    turning_angles,
)

REGION_CODES = {REGION_UPPER: 0, REGION_LOWER: 1}


class MeshQualityError(RuntimeError):
    """The measured-sole mesh failed a quality or topology check."""


@dataclass
class SoleMeshData:
    """Measured-sole mesh with explicit, heel-to-toe ordered tag node lists."""

    mesh: MeshTri
    config: object
    geometry: SoleGeometry
    element_region: np.ndarray  # (n_el,) REGION_CODES
    top_curve_nodes: np.ndarray  # heel_top -> toe_tip (includes the toe)
    bottom_curve_nodes: np.ndarray  # heel_bottom -> toe_tip (includes the toe)
    heel_edge_nodes: np.ndarray  # heel_top -> heel_bottom
    interface_nodes: np.ndarray  # heel -> toe along the foam interface
    plate_node_ids: np.ndarray  # heel -> toe, subset of interface_nodes
    toe_node_id: int
    region_loops: dict[str, np.ndarray]
    quality: dict = field(default_factory=dict)
    msh_path: Path | None = None

    # Shared-toe convention: the bottom/contact selector owns the toe vertex.
    shared_toe_policy: str = SHARED_TOE_POLICY

    @property
    def top_selector_nodes(self) -> np.ndarray:
        nodes = np.asarray(self.top_curve_nodes, dtype=int)
        return nodes[nodes != int(self.toe_node_id)]

    @property
    def bottom_selector_nodes(self) -> np.ndarray:
        return np.asarray(self.bottom_curve_nodes, dtype=int)

    @property
    def x_plate(self) -> np.ndarray:
        return np.asarray(self.mesh.p[0, self.plate_node_ids], dtype=float)

    @property
    def y_plate(self) -> np.ndarray:
        return np.asarray(self.mesh.p[1, self.plate_node_ids], dtype=float)

    @property
    def plate_element_ids(self) -> np.ndarray:
        return np.arange(int(self.plate_node_ids.size) - 1, dtype=int)

    def region_elements(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.element_region == REGION_CODES[name])


def _local_sizes(geom: SoleGeometry, config) -> dict[tuple[float, float], float]:
    """Characteristic length per geometry vertex (curvature + landmark refinement)."""
    h = float(config.mesh_size_m)
    max_turn = np.radians(float(config.curvature_max_turn_deg))
    sizes: dict[tuple[float, float], float] = {}

    def put(p, lc):
        key = (float(p[0]), float(p[1]))
        sizes[key] = min(sizes.get(key, h), float(lc))

    for poly in (geom.top, geom.bottom, geom.interface, geom.heel_edge):
        seg = np.hypot(*np.diff(poly, axis=0).T)
        turn = np.abs(turning_angles(poly))
        for k, p in enumerate(poly):
            put(p, h)
            if 0 < k < len(poly) - 1 and turn[k] > 0.0:
                ds = 0.5 * (seg[k - 1] + seg[k])
                kappa = turn[k] / max(ds, 1e-300)
                put(p, max_turn / kappa)
    lm = geom.landmarks
    put(lm["toe_tip"], config.toe_refinement * h)
    put(lm["heel_top"], config.heel_corner_refinement * h)
    put(lm["heel_bottom"], config.heel_corner_refinement * h)
    put(geom.interface[0], config.interface_refinement * h)
    put(geom.interface[-1], config.interface_refinement * h)
    put(geom.interface[geom.plate_heel_index], config.plate_end_refinement * h)
    put(geom.interface[geom.plate_toe_index], config.plate_end_refinement * h)
    return sizes


def _chain(edges: np.ndarray, start: int, end: int) -> np.ndarray:
    """Order the nodes of an open chain of 2-node edges from ``start`` to ``end``."""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) > 2 for v in adj.values()):
        raise MeshQualityError("Tagged curve mesh is branched.")
    order = [int(start)]
    prev = -1
    while order[-1] != int(end):
        nxt = [n for n in adj.get(order[-1], []) if n != prev]
        if not nxt:
            raise MeshQualityError("Tagged curve mesh is not a connected chain between its endpoints.")
        prev = order[-1]
        order.append(nxt[0])
        if len(order) > len(adj) + 1:
            raise MeshQualityError("Tagged curve mesh contains a cycle.")
    if len(order) != len(adj):
        raise MeshQualityError("Tagged curve mesh has nodes outside the endpoint chain.")
    return np.asarray(order, dtype=int)


def _triangle_quality(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed areas (input orientation) and minimum interior angles (deg)."""
    a, b, c = p[:, t[0]], p[:, t[1]], p[:, t[2]]
    area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

    def ang(u, v):
        cosang = np.einsum("ij,ij->j", u, v) / (np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    angles = np.vstack([ang(b - a, c - a), ang(a - b, c - b), ang(a - c, b - c)])
    return area, angles.min(axis=0)


def _components(t: np.ndarray, n_nodes: int) -> int:
    """Number of edge-connected element components."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_el = t.shape[1]
    rows = np.repeat(np.arange(n_el), 3)
    cols = t.T.reshape(-1)
    incidence = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_el, n_nodes)).tocsr()
    adj = (incidence @ incidence.T) >= 2  # share an edge
    n, _ = connected_components(adj, directed=False)
    return int(n)


def generate_mesh(config, output_msh: Path | None = None) -> SoleMeshData:
    """Mesh the measured sole with conforming foams and an inserted plate segment."""
    geom = geometry_from_config(config)
    sizes = _local_sizes(geom, config)
    h = float(config.mesh_size_m)

    with gmsh_session():
        try:
            initialize_gmsh()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("measured_sole")
            point_tag: dict[tuple[float, float], int] = {}

            def pt(p) -> int:
                key = (float(p[0]), float(p[1]))
                if key not in point_tag:
                    point_tag[key] = gmsh.model.geo.addPoint(key[0], key[1], 0.0, sizes.get(key, h))
                return point_tag[key]

            line_of: dict[tuple[int, int], int] = {}

            def polyline(poly) -> list[int]:
                tags = [pt(p) for p in poly]
                out = []
                for a, b in zip(tags[:-1], tags[1:]):
                    if (a, b) in line_of:
                        out.append(line_of[(a, b)])
                    elif (b, a) in line_of:
                        out.append(-line_of[(b, a)])
                    else:
                        line_of[(a, b)] = gmsh.model.geo.addLine(a, b)
                        out.append(line_of[(a, b)])
                return out

            top_lines = polyline(geom.top)
            bottom_lines = polyline(geom.bottom)
            heel_lines = polyline(geom.heel_edge)
            iface_lines = polyline(geom.interface)
            plate_lines = iface_lines[geom.plate_heel_index : geom.plate_toe_index]

            def loop_lines(poly) -> list[int]:
                closed = np.vstack([poly, poly[:1]])
                return polyline(closed)

            surfaces = {}
            for name in (REGION_UPPER, REGION_LOWER):
                loop = gmsh.model.geo.addCurveLoop(loop_lines(geom.region_polygons[name]))
                surfaces[name] = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            special = [
                (geom.landmarks["toe_tip"], config.toe_refinement),
                (geom.landmarks["heel_top"], config.heel_corner_refinement),
                (geom.landmarks["heel_bottom"], config.heel_corner_refinement),
                (geom.interface[0], config.interface_refinement),
                (geom.interface[-1], config.interface_refinement),
                (geom.interface[geom.plate_heel_index], config.plate_end_refinement),
                (geom.interface[geom.plate_toe_index], config.plate_end_refinement),
            ]
            fields = []
            for p, factor in special:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "PointsList", [pt(p)])
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", float(factor) * h)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", float(factor) * h)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", 4.0 * h)
                fields.append(ft)
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)

            gmsh.model.addPhysicalGroup(2, [surfaces[REGION_UPPER]], name=REGION_UPPER)
            gmsh.model.addPhysicalGroup(2, [surfaces[REGION_LOWER]], name=REGION_LOWER)
            for name, lines in (
                (TAG_TOP, top_lines),
                (TAG_BOTTOM, bottom_lines),
                (TAG_HEEL, heel_lines),
                (TAG_INTERFACE, iface_lines),
                (TAG_PLATE, plate_lines),
            ):
                gmsh.model.addPhysicalGroup(1, [abs(x) for x in lines], name=name)
            gmsh.model.mesh.generate(2)
            if output_msh is not None:
                output_msh = Path(output_msh)
                output_msh.parent.mkdir(parents=True, exist_ok=True)
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.write(str(output_msh))

            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            node_tags = np.asarray(node_tags, dtype=np.int64)
            xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
            tag_to_raw = {int(t): k for k, t in enumerate(node_tags)}

            tri_by_region: dict[str, np.ndarray] = {}
            for name, surf in surfaces.items():
                etypes, _, enodes = gmsh.model.mesh.getElements(2, surf)
                blocks = [np.asarray(n, dtype=np.int64).reshape(-1, 3) for et, n in zip(etypes, enodes) if et == 2]
                if len(blocks) != len(etypes):
                    raise MeshQualityError("Measured-sole mesh contains non-triangular elements.")
                tri_by_region[name] = np.vstack(blocks)

            def line_edges(lines: list[int]) -> np.ndarray:
                out = []
                for lt in lines:
                    etypes, _, enodes = gmsh.model.mesh.getElements(1, abs(lt))
                    for et, n in zip(etypes, enodes):
                        if et != 1:
                            raise MeshQualityError("Unexpected curve element type in measured mesh.")
                        out.append(np.asarray(n, dtype=np.int64).reshape(-1, 2))
                return np.vstack(out)

            curve_edges = {
                "top": line_edges(top_lines),
                "bottom": line_edges(bottom_lines),
                "heel": line_edges(heel_lines),
                "interface": line_edges(iface_lines),
                "plate": line_edges(plate_lines),
            }

            def point_node(p) -> int:
                tags_, _, _ = gmsh.model.mesh.getNodes(0, pt(p))
                return int(tags_[0])

            landmark_nodes = {
                "toe_tip": point_node(geom.landmarks["toe_tip"]),
                "heel_top": point_node(geom.landmarks["heel_top"]),
                "heel_bottom": point_node(geom.landmarks["heel_bottom"]),
                "iface_heel": point_node(geom.interface[0]),
                "iface_toe": point_node(geom.interface[-1]),
                "plate_heel": point_node(geom.interface[geom.plate_heel_index]),
                "plate_toe": point_node(geom.interface[geom.plate_toe_index]),
            }
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    # Compact node numbering to the nodes used by triangles.
    all_tri = np.vstack([tri_by_region[REGION_UPPER], tri_by_region[REGION_LOWER]])
    used_tags = np.unique(all_tri)
    new_index = {int(t): k for k, t in enumerate(used_tags)}
    remap = np.vectorize(lambda t: new_index[int(t)], otypes=[np.int64])
    p = xyz[[tag_to_raw[int(t)] for t in used_tags], :2].T.copy()
    t_upper = remap(tri_by_region[REGION_UPPER]).T
    t_lower = remap(tri_by_region[REGION_LOWER]).T
    t_all = np.hstack([t_upper, t_lower])
    element_region = np.concatenate(
        [np.full(t_upper.shape[1], REGION_CODES[REGION_UPPER]), np.full(t_lower.shape[1], REGION_CODES[REGION_LOWER])]
    ).astype(int)
    lm = {k: new_index[v] for k, v in landmark_nodes.items()}
    edges = {k: remap(v) for k, v in curve_edges.items()}

    top_nodes = _chain(edges["top"], lm["heel_top"], lm["toe_tip"])
    bottom_nodes = _chain(edges["bottom"], lm["heel_bottom"], lm["toe_tip"])
    heel_nodes = _chain(edges["heel"], lm["heel_top"], lm["heel_bottom"])
    iface_nodes = _chain(edges["interface"], lm["iface_heel"], lm["iface_toe"])
    plate_nodes = _chain(edges["plate"], lm["plate_heel"], lm["plate_toe"])

    area, min_angle = _triangle_quality(p, t_all)
    sign = np.sign(area)
    if np.any(sign == 0) or not (np.all(sign > 0) or np.all(sign < 0)):
        raise MeshQualityError("Measured-sole mesh has degenerate or inverted triangles.")
    if np.all(sign < 0):
        t_all = t_all[[0, 2, 1]]
        area = -area

    mesh = MeshTri(np.ascontiguousarray(p), np.ascontiguousarray(t_all), sort_t=False)
    upper_el = np.flatnonzero(element_region == REGION_CODES[REGION_UPPER])
    lower_el = np.flatnonzero(element_region == REGION_CODES[REGION_LOWER])
    facet_index = {(int(a), int(b)): k for k, (a, b) in enumerate(np.sort(mesh.facets, axis=0).T)}

    def facets_of(edge_array: np.ndarray) -> np.ndarray:
        out = []
        for a, b in edge_array:
            key = (int(min(a, b)), int(max(a, b)))
            if key not in facet_index:
                raise MeshQualityError("A tagged curve edge is not a mesh facet.")
            out.append(facet_index[key])
        return np.asarray(sorted(out), dtype=int)

    boundaries = {
        TAG_TOP: facets_of(edges["top"]),
        TAG_BOTTOM: facets_of(edges["bottom"]),
        TAG_HEEL: facets_of(edges["heel"]),
        TAG_INTERFACE: facets_of(edges["interface"]),
        TAG_PLATE: facets_of(edges["plate"]),
    }
    boundaries["top"] = boundaries[TAG_TOP]
    boundaries["bottom"] = boundaries[TAG_BOTTOM]
    boundaries["plate_interface"] = boundaries[TAG_PLATE]
    mesh = mesh.with_subdomains({REGION_UPPER: upper_el, REGION_LOWER: lower_el})
    mesh = mesh.with_boundaries(boundaries)

    def loop_nodes(*parts: np.ndarray) -> np.ndarray:
        seq: list[int] = []
        for part in parts:
            for n in part:
                if not seq or seq[-1] != int(n):
                    seq.append(int(n))
        if seq[0] == seq[-1]:
            seq.pop()
        return np.asarray(seq, dtype=int)

    region_loops = _region_loops(geom, top_nodes, bottom_nodes, heel_nodes, iface_nodes, loop_nodes)

    data = SoleMeshData(
        mesh=mesh,
        config=config,
        geometry=geom,
        element_region=element_region,
        top_curve_nodes=top_nodes,
        bottom_curve_nodes=bottom_nodes,
        heel_edge_nodes=heel_nodes,
        interface_nodes=iface_nodes,
        plate_node_ids=plate_nodes,
        toe_node_id=int(lm["toe_tip"]),
        region_loops=region_loops,
        msh_path=output_msh,
    )
    data.quality = verify_mesh(data, min_area=0.0, min_angle_deg=float(config.min_angle_deg))
    return data


def _region_loops(geom, top_nodes, bottom_nodes, heel_nodes, iface_nodes, loop_nodes) -> dict[str, np.ndarray]:
    """Counterclockwise boundary node loops of the two foam regions."""
    iface_toe, iface_heel = int(iface_nodes[-1]), int(iface_nodes[0])
    ext = loop_nodes(bottom_nodes, top_nodes[::-1], heel_nodes)
    k_h = int(np.flatnonzero(ext == iface_heel)[0])
    k_t = int(np.flatnonzero(ext == iface_toe)[0])

    def arc(i0, i1):
        idx = [i0]
        while idx[-1] != i1:
            idx.append((idx[-1] + 1) % ext.size)
        return ext[idx]

    lower = loop_nodes(arc(k_h, k_t), iface_nodes[::-1])
    upper = loop_nodes(arc(k_t, k_h), iface_nodes)
    names = geom.region_polygons
    # Assign by which loop carries the top surface (same rule as the geometry).
    top_set = set(int(n) for n in top_nodes[1:-1])
    if len(top_set & set(lower.tolist())) > len(top_set & set(upper.tolist())):
        lower, upper = upper, lower
    del names
    return {REGION_UPPER: upper, REGION_LOWER: lower}


def verify_mesh(data: SoleMeshData, min_area: float = 0.0, min_angle_deg: float = 12.0) -> dict:
    """Quality and topology checks; raises :class:`MeshQualityError` on failure."""
    mesh = data.mesh
    geom = data.geometry
    p = np.asarray(mesh.p, dtype=float)
    t = np.asarray(mesh.t, dtype=int)
    L = float(geom.shoe_length_m)
    area, min_ang = _triangle_quality(p, t)
    abs_area = np.abs(area)
    report: dict = {
        "n_nodes": int(p.shape[1]),
        "n_elements": int(t.shape[1]),
        "n_elements_upper": int(np.count_nonzero(data.element_region == REGION_CODES[REGION_UPPER])),
        "n_elements_lower": int(np.count_nonzero(data.element_region == REGION_CODES[REGION_LOWER])),
        "min_element_area_m2": float(abs_area.min()),
        "min_angle_deg": float(min_ang.min()),
        "mean_min_angle_deg": float(min_ang.mean()),
    }
    if abs_area.min() <= max(min_area, 1e-14 * L * L):
        raise MeshQualityError(f"Degenerate element (area {abs_area.min():.3e} m^2).")
    if min_ang.min() < min_angle_deg:
        k = int(np.argmin(min_ang))
        near_toe = int(data.toe_node_id) in t[:, k]
        raise MeshQualityError(
            f"Minimum element angle {min_ang.min():.2f} deg < {min_angle_deg} deg"
            + (
                " at the point toe. Refine toe_refinement; the toe is never rounded automatically "
                "(an explicit toe regularization would be required)."
                if near_toe
                else "."
            )
        )
    toe_elems = np.flatnonzero(np.any(t == int(data.toe_node_id), axis=0))
    report["toe_elements"] = int(toe_elems.size)
    report["toe_min_angle_deg"] = float(min_ang[toe_elems].min()) if toe_elems.size else float("nan")
    if toe_elems.size == 0:
        raise MeshQualityError("The toe-tip node is not attached to any element.")

    tree = cKDTree(p.T)
    pairs = tree.query_pairs(1e-9 * L)
    if pairs:
        raise MeshQualityError(f"Duplicate mesh nodes: {sorted(pairs)[:5]}.")
    n_comp = _components(t, p.shape[1])
    report["n_connected_components"] = n_comp
    if n_comp != 1:
        raise MeshQualityError(f"Mesh has {n_comp} disconnected components.")
    for name, code in REGION_CODES.items():
        sub = t[:, data.element_region == code]
        if sub.shape[1] == 0:
            raise MeshQualityError(f"Foam region {name} has no elements.")
        if _components(sub, p.shape[1]) != 1:
            raise MeshQualityError(f"Foam region {name} is disconnected.")
    for name, poly_area in geom.region_areas.items():
        mesh_area = float(abs_area[data.element_region == REGION_CODES[name]].sum())
        report[f"area_{name}_m2"] = mesh_area
        if abs(mesh_area - poly_area) > 1e-9 * geom.total_area:
            raise MeshQualityError(f"Region {name} mesh area {mesh_area} differs from geometry {poly_area}.")

    upper_nodes = set(np.unique(t[:, data.element_region == 0]).tolist())
    lower_nodes = set(np.unique(t[:, data.element_region == 1]).tolist())
    shared = upper_nodes & lower_nodes
    iface = set(int(n) for n in data.interface_nodes)
    if shared != iface:
        raise MeshQualityError(
            f"Foam regions share {len(shared)} nodes but the interface has {len(iface)}; not conforming."
        )
    if not set(int(n) for n in data.plate_node_ids) <= iface:
        raise MeshQualityError("Plate nodes are not a subset of the foam interface.")

    dev = {}
    for name, nodes, poly in (
        ("top", data.top_curve_nodes, geom.top),
        ("bottom", data.bottom_curve_nodes, geom.bottom),
        ("heel_edge", data.heel_edge_nodes, geom.heel_edge),
        ("interface", data.interface_nodes, geom.interface),
    ):
        d = distance_to_polyline(poly, p[:, nodes].T)
        dev[name] = float(d.max())
        if d.max() > 1e-9 * L:
            raise MeshQualityError(f"{name} nodes deviate {d.max():.3e} m from the supplied curve.")
        for v in poly:
            if np.min(np.hypot(*(p[:, nodes].T - v).T)) > 1e-9 * L:
                raise MeshQualityError(f"{name} curve vertex {v.tolist()} is not a mesh node.")
    report["max_boundary_deviation_m"] = dev
    pe = data.geometry.interface[[data.geometry.plate_heel_index, data.geometry.plate_toe_index]]
    plate_xy = p[:, data.plate_node_ids]
    if np.max(np.hypot(*(plate_xy[:, [0, -1]].T - pe).T)) > 1e-12 * L:
        raise MeshQualityError("Plate end nodes do not coincide with the inserted plate endpoints.")
    if int(data.top_curve_nodes[-1]) != int(data.toe_node_id) or int(data.bottom_curve_nodes[-1]) != int(data.toe_node_id):
        raise MeshQualityError("Top and bottom curves do not share the single toe-tip node.")
    common = set(int(n) for n in data.top_curve_nodes) & set(int(n) for n in data.bottom_curve_nodes)
    if common != {int(data.toe_node_id)}:
        raise MeshQualityError(f"Top and bottom curves share nodes {sorted(common)}; expected only the toe.")
    report["n_top_curve_nodes"] = int(data.top_curve_nodes.size)
    report["n_top_selector_nodes"] = int(data.top_selector_nodes.size)
    report["n_bottom_nodes"] = int(data.bottom_curve_nodes.size)
    report["n_interface_nodes"] = int(data.interface_nodes.size)
    report["n_plate_nodes"] = int(data.plate_node_ids.size)
    report["n_plate_elements"] = int(data.plate_node_ids.size - 1)
    return report
