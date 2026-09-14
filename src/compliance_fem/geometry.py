"""Gmsh geometry and mesh generation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh
import meshio
import numpy as np
from skfem import Mesh
from skfem.io import from_meshio

from compliance_fem.gmsh_util import gmsh_session, initialize_gmsh

if TYPE_CHECKING:
    from compliance_fem.config import ProblemConfig


BOUNDARY_NAMES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class MeshData:
    """Container for a generated mesh and metadata."""

    mesh: Mesh
    msh_path: Path | None = None


def _ensure_boundaries(mesh: Mesh, L: float, H: float) -> Mesh:
    """Attach named boundaries when meshio import omits physical tags."""
    if hasattr(mesh, "boundaries") and mesh.boundaries:
        return mesh
    return mesh.with_boundaries(
        {
            "top": lambda x: np.isclose(x[1], H),
            "bottom": lambda x: np.isclose(x[1], 0.0),
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], L),
        }
    )


def generate_rectangular_mesh(
    config: ProblemConfig,
    output_msh: Path | None = None,
) -> MeshData:
    """Create a structured rectangular mesh with named physical groups."""
    L, H = config.L, config.H
    nx, ny = config.nx, config.ny
    order = config.order
    use_quads = config.element_type == "quad"

    with gmsh_session():
        try:
            initialize_gmsh()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.RecombineAll", int(use_quads))
            gmsh.model.add("rectangle")

            p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p2 = gmsh.model.geo.addPoint(L, 0.0, 0.0)
            p3 = gmsh.model.geo.addPoint(L, H, 0.0)
            p4 = gmsh.model.geo.addPoint(0.0, H, 0.0)

            l_bottom = gmsh.model.geo.addLine(p1, p2)
            l_right = gmsh.model.geo.addLine(p2, p3)
            l_top = gmsh.model.geo.addLine(p3, p4)
            l_left = gmsh.model.geo.addLine(p4, p1)
            curve_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])
            gmsh.model.geo.synchronize()

            for curve, count in ((l_bottom, nx), (l_right, ny), (l_top, nx), (l_left, ny)):
                gmsh.model.mesh.setTransfiniteCurve(curve, count + 1)
            gmsh.model.mesh.setTransfiniteSurface(surface)
            if use_quads:
                gmsh.model.mesh.setRecombine(2, surface)

            gmsh.model.mesh.generate(2)
            if order > 1:
                gmsh.model.mesh.setOrder(order)

            gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="domain")
            gmsh.model.addPhysicalGroup(1, [l_top], tag=2, name="top")
            gmsh.model.addPhysicalGroup(1, [l_bottom], tag=3, name="bottom")
            gmsh.model.addPhysicalGroup(1, [l_left], tag=4, name="left")
            gmsh.model.addPhysicalGroup(1, [l_right], tag=5, name="right")

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
        raise RuntimeError("Failed to import Gmsh mesh via meshio/scikit-fem.")
    mesh = _ensure_boundaries(mesh, L, H)
    return MeshData(mesh=mesh, msh_path=msh_path)


def boundary_node_coordinates(mesh: Mesh, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted (x, y) coordinates of nodes on a named boundary."""
    facets = mesh.boundaries[boundary]
    nodes = np.unique(mesh.facets[:, facets].reshape(-1))
    x = mesh.p[0, nodes]
    y = mesh.p[1, nodes]
    order = np.argsort(x)
    return x[order], y[order]


def verify_mesh_dimensions(mesh: Mesh, config: ProblemConfig, atol: float = 1e-10) -> None:
    """Check that the mesh spans the requested rectangle."""
    xmin, xmax = mesh.p[0].min(), mesh.p[0].max()
    ymin, ymax = mesh.p[1].min(), mesh.p[1].max()
    if not np.isclose(xmin, 0.0, atol=atol) or not np.isclose(xmax, config.L, atol=atol):
        raise ValueError(f"Mesh x-extent [{xmin}, {xmax}] does not match L={config.L}.")
    if not np.isclose(ymin, 0.0, atol=atol) or not np.isclose(ymax, config.H, atol=atol):
        raise ValueError(f"Mesh y-extent [{ymin}, {ymax}] does not match H={config.H}.")


def verify_positive_jacobians(mesh: Mesh) -> None:
    """Raise if any element has a non-positive Jacobian determinant."""
    if hasattr(mesh, "jacobian"):
        det = mesh.jacobian()
        if np.any(det <= 0.0):
            raise ValueError("Mesh contains elements with non-positive Jacobian determinants.")
        return

    if mesh.t.shape[0] == 3:
        for element in mesh.t.T:
            pts = mesh.p[:, element]
            area = 0.5 * np.linalg.det(
                np.column_stack(
                    [
                        np.ones(3),
                        pts[0, :],
                        pts[1, :],
                    ]
                )
            )
            if abs(area) <= 1e-14:
                raise ValueError("Mesh contains degenerate triangular elements.")
        return

    if mesh.t.shape[0] == 4:
        for quad in mesh.t.T:
            pts = mesh.p[:, quad]
            area = 0.0
            for k in range(4):
                j = (k + 1) % 4
                area += pts[0, k] * pts[1, j] - pts[0, j] * pts[1, k]
            if abs(area) <= 1e-14:
                raise ValueError("Mesh contains degenerate quadrilateral elements.")
        return

    raise ValueError("Unsupported mesh topology for Jacobian verification.")
