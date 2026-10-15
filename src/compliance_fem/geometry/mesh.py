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
from typing import Any

import gmsh
import numpy as np
from skfem import MeshTri

from compliance_fem.fem.skfem_types import mesh_facets

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
    build_sole_geometry,
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
    # Point sizes also grade the adjacent curve segments; the distance fields in
    # generate_mesh refine the surrounding area. Both are needed.
    for p, factor in _refined_points(geom, config):
        put(p, factor * h)
    return sizes


def _refined_points(geom: SoleGeometry, config) -> list[tuple[np.ndarray, float]]:
    """Corner, interface-end and plate-end points with their size factor (of mesh_size_m)."""
    return [
        (geom.toe_tip, config.toe_refinement),
        (geom.heel_top, config.heel_corner_refinement),
        (geom.heel_bottom, config.heel_corner_refinement),
        (geom.interface[0], config.interface_refinement),
        (geom.interface[-1], config.interface_refinement),
        (geom.interface[geom.plate_heel_index], config.plate_end_refinement),
        (geom.interface[geom.plate_toe_index], config.plate_end_refinement),
    ]


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
    # scipy-stubs does not type sparse-vs-scalar comparison, which scipy supports.
    adj = (incidence @ incidence.T) >= 2  # share an edge  # pyright: ignore[reportOperatorIssue]
    n, _ = connected_components(adj, directed=False)
    return int(n)


def generate_mesh(config, output_msh: Path | None = None) -> SoleMeshData:
    print("rebuilding measured-sole mesh...")
    """Mesh the measured sole with conforming foams and an inserted plate segment."""
    geom = build_sole_geometry(config.normalized_geometry, config.shoe_length_mm, config.landmark_tolerance)
    sizes = _local_sizes(geom, config)
    h = float(config.mesh_size_m)

    with gmsh_session():
        try:
            initialize_gmsh()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("measured_sole")
            point_ids: dict[tuple[float, float], int] = {}

            def pt(p) -> int:
                key = (float(p[0]), float(p[1]))
                if key not in point_ids:
                    point_ids[key] = gmsh.model.geo.addPoint(key[0], key[1], 0.0, sizes.get(key, h))
                return point_ids[key]

            line_ids: dict[tuple[int, int], int] = {}

            def polyline(points) -> list[int]:
                pt_ids = [pt(p) for p in points]
                out = []
                for a, b in zip(pt_ids[:-1], pt_ids[1:]):
                    if (a, b) in line_ids:
                        out.append(line_ids[(a, b)])
                    elif (b, a) in line_ids:
                        out.append(-line_ids[(b, a)])
                    else:
                        line_ids[(a, b)] = gmsh.model.geo.addLine(a, b)
                        out.append(line_ids[(a, b)])
                return out

            # Define interior and exterior curves as polylines
            top_lines = polyline(geom.top)
            bottom_lines = polyline(geom.bottom)
            heel_lines = polyline(geom.heel_edge)
            iface_lines = polyline(geom.interface)
            plate_lines = iface_lines[geom.plate_heel_index : geom.plate_toe_index]

            def polyline_loop(points) -> list[int]:
                closed = np.vstack([points, points[:1]])
                return polyline(closed)

            # Define the two foam regions as plane surfaces
            surface_ids = {}
            for name in (REGION_UPPER, REGION_LOWER):
                loop = gmsh.model.geo.addCurveLoop(polyline_loop(geom.region_polygons[name]))
                surface_ids[name] = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            # Refine the mesh near the corners, interface ends, and plate ends using distance fields.
            fields = []
            for p, factor in _refined_points(geom, config):
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "PointsList", [pt(p)])
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", float(factor) * h)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", float(factor) * h)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", 4.0 * h)
                fields.append(ft)

            # Define global mesh size and recombination settings, and generate the mesh.
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)

            # Add physical groups for the foam regions and the boundary curves.
            gmsh.model.addPhysicalGroup(2, [surface_ids[REGION_UPPER]], name=REGION_UPPER)
            gmsh.model.addPhysicalGroup(2, [surface_ids[REGION_LOWER]], name=REGION_LOWER)
            for name, lines in (
                (TAG_TOP, top_lines),
                (TAG_BOTTOM, bottom_lines),
                (TAG_HEEL, heel_lines),
                (TAG_INTERFACE, iface_lines),
                (TAG_PLATE, plate_lines),
            ):
                gmsh.model.addPhysicalGroup(1, [abs(x) for x in lines], name=name)

            # Generate mesh and write to disc
            gmsh.model.mesh.generate(2)
            if output_msh is not None:
                output_msh = Path(output_msh)
                output_msh.parent.mkdir(parents=True, exist_ok=True)
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.write(str(output_msh))

            tri_by_region: dict[str, np.ndarray] = {}
            for name, surf in surface_ids.items():
                etypes, _, enodes = gmsh.model.mesh.getElements(2, surf)
                if list(etypes) != [2]:
                    raise MeshQualityError("Measured-sole mesh contains non-triangular elements.")
                tri_by_region[name] = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 3)

            def curve_nodes(points: np.ndarray) -> np.ndarray:
                """Gmsh node tags along the polyline through ``points``, in order."""
                chain: list[int] = []
                for a, b in zip(points[:-1], points[1:]):
                    chain.append(int(gmsh.model.mesh.getNodes(0, pt(a))[0][0]))
                    line = line_ids.get((pt(a), pt(b))) or line_ids[(pt(b), pt(a))]
                    tags, xyz, _ = gmsh.model.mesh.getNodes(1, line)
                    # Lines are straight, so interior nodes order by distance from ``a``.
                    along = (np.asarray(xyz).reshape(-1, 3)[:, :2] - a) @ (b - a)
                    chain.extend(int(t) for t in np.asarray(tags)[np.argsort(along)])
                chain.append(int(gmsh.model.mesh.getNodes(0, pt(points[-1]))[0][0]))
                return np.asarray(chain, dtype=np.int64)

            ih, it = geom.plate_heel_index, geom.plate_toe_index
            chains = {
                TAG_TOP: curve_nodes(geom.top),
                TAG_BOTTOM: curve_nodes(geom.bottom),
                TAG_HEEL: curve_nodes(geom.heel_edge),
                TAG_INTERFACE: curve_nodes(geom.interface),
                TAG_PLATE: curve_nodes(geom.interface[ih : it + 1]),
            }
            # Counterclockwise region loops, closed polygon minus the repeated start node.
            loops = {
                name: curve_nodes(np.vstack([poly, poly[:1]]))[:-1] for name, poly in geom.region_polygons.items()
            }
            node_ids, node_xyz, _ = gmsh.model.mesh.getNodes()
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    # Renumber the gmsh node tags used by triangles to 0..n-1 (``used`` is sorted).
    used = np.unique(np.vstack(list(tri_by_region.values())))
    xy_of = dict(zip(np.asarray(node_ids).tolist(), np.asarray(node_xyz, dtype=float).reshape(-1, 3)[:, :2]))
    p = np.array([xy_of[t] for t in used.tolist()]).T

    def renumber(tags: np.ndarray) -> np.ndarray:
        return np.searchsorted(used, tags)

    t_upper = renumber(tri_by_region[REGION_UPPER]).T
    t_lower = renumber(tri_by_region[REGION_LOWER]).T
    t_all = np.hstack([t_upper, t_lower])
    element_region = np.repeat(
        [REGION_CODES[REGION_UPPER], REGION_CODES[REGION_LOWER]], [t_upper.shape[1], t_lower.shape[1]]
    )
    chains = {k: renumber(v) for k, v in chains.items()}
    loops = {k: renumber(v) for k, v in loops.items()}

    area, _ = _triangle_quality(p, t_all)
    sign = np.sign(area)
    if np.any(sign == 0) or not (np.all(sign > 0) or np.all(sign < 0)):
        raise MeshQualityError("Measured-sole mesh has degenerate or inverted triangles.")
    if np.all(sign < 0):
        t_all = t_all[[0, 2, 1]]

    mesh = MeshTri(np.ascontiguousarray(p), np.ascontiguousarray(t_all), sort_t=False)
    facet_index = {(int(a), int(b)): k for k, (a, b) in enumerate(np.sort(mesh_facets(mesh), axis=0).T)}

    def facets_of(chain: np.ndarray) -> np.ndarray:
        """Facet ids of the consecutive node pairs of ``chain``."""
        keys = [(int(min(a, b)), int(max(a, b))) for a, b in zip(chain[:-1], chain[1:])]
        if any(k not in facet_index for k in keys):
            raise MeshQualityError("A tagged curve edge is not a mesh facet.")
        return np.sort([facet_index[k] for k in keys])

    # Annotated loosely: MeshTri.with_boundaries takes an invariant Dict of arrays or callables.
    boundaries: dict[str, Any] = {tag: facets_of(chain) for tag, chain in chains.items()}
    mesh = mesh.with_subdomains(
        {name: np.flatnonzero(element_region == code) for name, code in REGION_CODES.items()}
    ).with_boundaries(boundaries)

    data = SoleMeshData(
        mesh=mesh,
        config=config,
        geometry=geom,
        element_region=element_region,
        top_curve_nodes=chains[TAG_TOP],
        bottom_curve_nodes=chains[TAG_BOTTOM],
        heel_edge_nodes=chains[TAG_HEEL],
        interface_nodes=chains[TAG_INTERFACE],
        plate_node_ids=chains[TAG_PLATE],
        toe_node_id=int(chains[TAG_BOTTOM][-1]),
        region_loops=loops,
        msh_path=output_msh,
    )
    data.quality = verify_mesh(data, min_area=0.0, min_angle_deg=float(config.min_angle_deg))
    print("rebuilt")
    return data


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
