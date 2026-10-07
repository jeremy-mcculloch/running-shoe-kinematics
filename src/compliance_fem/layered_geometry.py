"""Gmsh geometry for two conforming foam trapezoids sharing a plate interface."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gmsh
import meshio
import numpy as np
from skfem import Mesh
from skfem.io import from_meshio

from compliance_fem.config import LayeredPlateConfig
from compliance_fem.geometry import MeshData, map_bottom_profile, verify_positive_jacobians
from compliance_fem.gmsh_util import gmsh_session, initialize_gmsh

LAYERED_BOUNDARY_NAMES = (
    "top",
    "bottom",
    "left_upper",
    "left_lower",
    "right_upper",
    "right_lower",
    "left",
    "right",
    "plate_interface",
)
LAYERED_SUBDOMAIN_NAMES = ("upper_foam", "lower_foam")


@dataclass(frozen=True)
class LayeredMeshData:
    """Conforming layered mesh with ordered plate-interface nodes."""

    mesh: Mesh
    config: LayeredPlateConfig
    plate_node_indices: np.ndarray
    x_plate: np.ndarray
    y_plate: np.ndarray
    msh_path: Path | None = None

    @property
    def as_mesh_data(self) -> MeshData:
        return MeshData(mesh=self.mesh, msh_path=self.msh_path)


def _element_centroids(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    pts = mesh.p[:, mesh.t]
    centroids = pts.mean(axis=1)
    return np.asarray(centroids[0], dtype=float), np.asarray(centroids[1], dtype=float)


def _interface_facets(mesh: Mesh) -> np.ndarray:
    """Return interior facets shared by upper_foam and lower_foam elements."""
    subdomains = getattr(mesh, "subdomains", None) or {}
    if "upper_foam" not in subdomains or "lower_foam" not in subdomains:
        raise KeyError("Need upper_foam and lower_foam subdomains to locate the plate interface.")
    upper = set(np.asarray(subdomains["upper_foam"], dtype=int).tolist())
    lower = set(np.asarray(subdomains["lower_foam"], dtype=int).tolist())
    f2t = mesh.f2t
    facets = [
        f
        for f in range(f2t.shape[1])
        if int(f2t[1, f]) >= 0
        and (
            (int(f2t[0, f]) in upper and int(f2t[1, f]) in lower)
            or (int(f2t[0, f]) in lower and int(f2t[1, f]) in upper)
        )
    ]
    if not facets:
        raise RuntimeError("No interior facets found between the upper and lower foam blocks.")
    return np.asarray(facets, dtype=int)


def _ensure_layered_groups(mesh: Mesh, config: LayeredPlateConfig, atol: float = 1e-8) -> Mesh:
    """Attach named subdomains and boundaries, including the interior plate interface."""
    cx, cy = _element_centroids(mesh)
    yp = np.asarray(config.y_plate(cx), dtype=float)
    lower = cy <= yp + atol
    upper = ~lower
    if not np.any(lower) or not np.any(upper):
        raise RuntimeError("Failed to classify upper and lower foam elements.")

    subdomains = dict(getattr(mesh, "subdomains", None) or {})
    if "upper_foam" not in subdomains or "lower_foam" not in subdomains:
        mesh = mesh.with_subdomains(
            {
                "upper_foam": lambda x: np.asarray(x[1])
                > np.asarray(config.y_plate(x[0])) + atol,
                "lower_foam": lambda x: np.asarray(x[1])
                <= np.asarray(config.y_plate(x[0])) + atol,
            }
        )

    h2_heel = config.h2_heel
    h2_toe = config.h2_toe
    mesh = mesh.with_boundaries(
        {
            "top": lambda x: np.isclose(
                x[1], np.asarray(config.y_top(x[0])), atol=atol, rtol=0.0
            ),
            "left_lower": lambda x: np.isclose(x[0], 0.0, atol=atol)
            & (x[1] <= h2_heel + atol)
            & (x[1] >= -atol),
            "left_upper": lambda x: np.isclose(x[0], 0.0, atol=atol)
            & (x[1] >= h2_heel - atol),
            "right_lower": lambda x: np.isclose(x[0], config.L, atol=atol)
            & (x[1] <= h2_toe + atol)
            & (x[1] >= -atol),
            "right_upper": lambda x: np.isclose(x[0], config.L, atol=atol)
            & (x[1] >= h2_toe - atol),
            "left": lambda x: np.isclose(x[0], 0.0, atol=atol),
            "right": lambda x: np.isclose(x[0], config.L, atol=atol),
        },
        boundaries_only=True,
    )
    return mesh.with_boundaries(
        {"bottom": _bottom_facets(mesh, config, atol), "plate_interface": _interface_facets(mesh)}
    )


def _bottom_facets(mesh: Mesh, config: LayeredPlateConfig, atol: float) -> np.ndarray:
    """Boundary facets whose vertices all lie on ``y = y_b(x)``.

    Vertices, not facet midpoints, are tested: on a curved (rocker) sole a
    straight facet's midpoint lies off the profile by O(h^2).
    """
    boundary = np.flatnonzero(mesh.f2t[1] < 0)
    pts = mesh.p[:, mesh.facets[:, boundary]]
    on = np.isclose(pts[1], np.asarray(config.y_bottom(pts[0]), dtype=float), atol=atol, rtol=0.0)
    facets = boundary[np.all(on, axis=0)]
    if facets.size == 0:
        raise RuntimeError("No boundary facets found on the bottom profile.")
    return facets.astype(int)


def ordered_interface_nodes(mesh: Mesh, config: LayeredPlateConfig, atol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return heel-to-toe plate node indices and coordinates."""
    if "plate_interface" not in mesh.boundaries:
        raise KeyError("Mesh is missing the plate_interface physical group.")
    facets = mesh.boundaries["plate_interface"]
    if len(facets) == 0:
        raise ValueError("plate_interface physical group contains no facets.")
    nodes = np.unique(mesh.facets[:, facets].reshape(-1))
    if nodes.size == 0:
        raise ValueError("No nodes found on plate_interface.")
    x = mesh.p[0, nodes]
    y = mesh.p[1, nodes]
    order = np.argsort(x, kind="mergesort")
    nodes = np.asarray(nodes[order], dtype=int)
    x = np.asarray(x[order], dtype=float)
    y = np.asarray(y[order], dtype=float)
    y_expected = np.asarray(config.y_plate(x), dtype=float)
    if np.max(np.abs(y - y_expected)) > atol:
        raise ValueError(
            "Interface nodes do not lie on the plate line y_p(x) within mesh tolerance."
        )
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Plate interface nodes are not strictly ordered by increasing x.")
    return nodes, x, y


def generate_layered_mesh(
    config: LayeredPlateConfig,
    output_msh: Path | None = None,
) -> LayeredMeshData:
    """Create a conforming two-block trapezoidal mesh with a shared plate interface."""
    L = config.L
    h1_heel, h1_toe = config.h1_heel, config.h1_toe
    h2_heel, h2_toe = config.h2_heel, config.h2_toe
    nx, ny1, ny2 = config.nx, config.ny1, config.ny2
    order = config.element_order
    use_quads = config.element_type == "quad"

    with gmsh_session():
        try:
            initialize_gmsh()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.RecombineAll", int(use_quads))
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.model.add("layered_trapezoids")

            p_bl = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p_br = gmsh.model.geo.addPoint(L, 0.0, 0.0)
            p_il = gmsh.model.geo.addPoint(0.0, h2_heel, 0.0)
            p_ir = gmsh.model.geo.addPoint(L, h2_toe, 0.0)
            p_tl = gmsh.model.geo.addPoint(0.0, h2_heel + h1_heel, 0.0)
            p_tr = gmsh.model.geo.addPoint(L, h2_toe + h1_toe, 0.0)

            l_bottom = gmsh.model.geo.addLine(p_bl, p_br)
            l_right_lower = gmsh.model.geo.addLine(p_br, p_ir)
            l_interface = gmsh.model.geo.addLine(p_il, p_ir)
            l_left_lower = gmsh.model.geo.addLine(p_il, p_bl)
            l_right_upper = gmsh.model.geo.addLine(p_ir, p_tr)
            l_top = gmsh.model.geo.addLine(p_tr, p_tl)
            l_left_upper = gmsh.model.geo.addLine(p_tl, p_il)

            loop_lower = gmsh.model.geo.addCurveLoop(
                [l_bottom, l_right_lower, -l_interface, l_left_lower]
            )
            loop_upper = gmsh.model.geo.addCurveLoop(
                [l_interface, l_right_upper, l_top, l_left_upper]
            )
            s_lower = gmsh.model.geo.addPlaneSurface([loop_lower])
            s_upper = gmsh.model.geo.addPlaneSurface([loop_upper])
            gmsh.model.geo.synchronize()

            for curve, count in (
                (l_bottom, nx),
                (l_interface, nx),
                (l_top, nx),
                (l_left_lower, ny2),
                (l_right_lower, ny2),
                (l_left_upper, ny1),
                (l_right_upper, ny1),
            ):
                gmsh.model.mesh.setTransfiniteCurve(curve, count + 1)
            gmsh.model.mesh.setTransfiniteSurface(s_lower)
            gmsh.model.mesh.setTransfiniteSurface(s_upper)
            if use_quads:
                gmsh.model.mesh.setRecombine(2, s_lower)
                gmsh.model.mesh.setRecombine(2, s_upper)

            gmsh.model.addPhysicalGroup(2, [s_upper], tag=1, name="upper_foam")
            gmsh.model.addPhysicalGroup(2, [s_lower], tag=2, name="lower_foam")
            gmsh.model.addPhysicalGroup(1, [l_top], tag=3, name="top")
            gmsh.model.addPhysicalGroup(1, [l_bottom], tag=4, name="bottom")
            gmsh.model.addPhysicalGroup(1, [l_interface], tag=5, name="plate_interface")
            gmsh.model.addPhysicalGroup(1, [l_left_upper], tag=6, name="left_upper")
            gmsh.model.addPhysicalGroup(1, [l_left_lower], tag=7, name="left_lower")
            gmsh.model.addPhysicalGroup(1, [l_right_upper], tag=8, name="right_upper")
            gmsh.model.addPhysicalGroup(1, [l_right_lower], tag=9, name="right_lower")
            gmsh.model.addPhysicalGroup(
                1, [l_left_upper, l_left_lower], tag=10, name="left"
            )
            gmsh.model.addPhysicalGroup(
                1, [l_right_upper, l_right_lower], tag=11, name="right"
            )

            gmsh.model.mesh.generate(2)
            if order > 1:
                gmsh.model.mesh.setOrder(order)

            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            if output_msh is not None:
                output_msh.parent.mkdir(parents=True, exist_ok=True)
                gmsh.write(str(output_msh))
                read_path = str(output_msh)
                msh_path = output_msh
            else:
                tmp_fd, read_path = tempfile.mkstemp(suffix=".msh")
                os.close(tmp_fd)
                gmsh.write(read_path)
                msh_path = None
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    meshio_data = meshio.read(read_path)
    if msh_path is None:
        os.remove(read_path)

    mesh = from_meshio(meshio_data)
    if mesh is None:
        raise RuntimeError("Failed to import layered Gmsh mesh via meshio/scikit-fem.")
    if config.sole_rocker_height > 0.0:
        mesh = map_bottom_profile(mesh, config.y_bottom, config.y_plate)
    mesh = _ensure_layered_groups(mesh, config)
    verify_positive_jacobians(mesh)

    plate_nodes, x_plate, y_plate = ordered_interface_nodes(mesh, config)
    return LayeredMeshData(
        mesh=mesh,
        config=config,
        plate_node_indices=plate_nodes,
        x_plate=x_plate,
        y_plate=y_plate,
        msh_path=msh_path,
    )


def verify_layered_mesh(mesh_data: LayeredMeshData, atol: float = 1e-8) -> None:
    """Check extents, named groups, and conforming interface nodes."""
    mesh = mesh_data.mesh
    config = mesh_data.config
    xmin, xmax = float(mesh.p[0].min()), float(mesh.p[0].max())
    ymin, ymax = float(mesh.p[1].min()), float(mesh.p[1].max())
    if not np.isclose(xmin, 0.0, atol=atol) or not np.isclose(xmax, config.L, atol=atol):
        raise ValueError(f"Mesh x-extent [{xmin}, {xmax}] does not match L={config.L}.")
    ymin_expected = float(np.min(config.y_bottom(mesh.p[0])))
    if not np.isclose(ymin, ymin_expected, atol=atol):
        raise ValueError(f"Mesh minimum y={ymin} does not match the bottom profile minimum.")
    if not np.isclose(ymax, config.H, atol=atol):
        # H is the max outer height; ymax should match the taller end.
        y_top_ends = (float(config.y_top(0.0)), float(config.y_top(config.L)))
        if not np.isclose(ymax, max(y_top_ends), atol=atol):
            raise ValueError(f"Mesh maximum y={ymax} does not match the outer top.")

    for name in LAYERED_SUBDOMAIN_NAMES:
        if name not in (mesh.subdomains or {}):
            raise KeyError(f"Missing subdomain physical group '{name}'.")
    for name in ("top", "bottom", "plate_interface", "left_upper", "left_lower", "right_upper", "right_lower"):
        if name not in (mesh.boundaries or {}):
            raise KeyError(f"Missing boundary physical group '{name}'.")

    upper = set(np.unique(mesh.t[:, mesh.subdomains["upper_foam"]].reshape(-1)))
    lower = set(np.unique(mesh.t[:, mesh.subdomains["lower_foam"]].reshape(-1)))
    shared = np.array(sorted(upper & lower), dtype=int)
    plate = np.asarray(mesh_data.plate_node_indices, dtype=int)
    if set(plate.tolist()) != set(shared.tolist()):
        raise ValueError("Plate interface nodes do not match the shared upper/lower mesh nodes.")
