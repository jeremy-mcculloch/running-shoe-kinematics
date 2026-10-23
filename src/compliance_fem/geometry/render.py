"""Lookup-side rendering data for the measured two-foam sole.

The lookup stores the reference mesh, the boundary node chains of each foam
region, and a render influence matrix so the GUI can draw both deformed
regions without any FEM solve:

    d_render = G [F_t ; F_bottom] + R_render alpha_rigid

The render nodes are every mesh vertex (in mesh order), so the GUI can draw the
deformed mesh and colour each element by its strain energy density
(:meth:`RenderSection.strain_energy_density`) with the per-element plane-strain
moduli stored alongside (``mesh_element_E``, ``mesh_element_nu``; ``E`` at the
element centroid for a graded foam).

``G`` (``influence_loads``, ``2 n_vertices x (2 n_t + 2 n_b)``, component-major rows
``[u..., v...]``) is sliced from the shared unit-boundary-load solve in
``contact_model.build_field_recovery``, which also builds the plate response;
``R_render`` (``influence_rigid``) is the rigid-mode matrix about the solver
reference ``(x_r, y_r)``. ``F_t``, ``F_bottom`` and the rigid amplitudes come from
one continuous contact solution (``contact_model.render_vertex_displacements``).
Displacements are rotating-frame local values; the fixed-frame transform is
applied by the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult

RENDER_KEY_PREFIX = "render_"
_INT_FIELDS = (
    "mesh_triangles",
    "mesh_element_region",
    "top_curve_node_ids",
    "bottom_curve_node_ids",
    "heel_edge_node_ids",
    "interface_node_ids",
    "upper_foam_loop",
    "lower_foam_loop",
)


@dataclass
class RenderSection:
    """Reference mesh, region chains and the all-vertex displacement influence (local frame)."""

    mesh_points: np.ndarray  # (2, n_vertices)
    mesh_triangles: np.ndarray  # (3, n_el)
    mesh_element_region: np.ndarray  # (n_el,) REGION_CODES
    mesh_element_E: np.ndarray  # (n_el,)
    mesh_element_nu: np.ndarray  # (n_el,)
    top_curve_node_ids: np.ndarray
    bottom_curve_node_ids: np.ndarray
    heel_edge_node_ids: np.ndarray
    interface_node_ids: np.ndarray
    upper_foam_loop: np.ndarray
    lower_foam_loop: np.ndarray
    influence_loads: np.ndarray  # (2 n_vertices, 2 n_t + 2 n_b)
    influence_rigid: np.ndarray  # (2 n_vertices, 3)

    @property
    def n_vertices(self) -> int:
        return int(self.mesh_points.shape[1])

    def uv_rows(self, node_ids: np.ndarray) -> np.ndarray:
        """Rows ``[u...; v...]`` of the influence matrices for mesh vertices ``node_ids``."""
        ids = np.asarray(node_ids, dtype=int)
        return np.concatenate([ids, self.n_vertices + ids])

    def displacements(self, loads: np.ndarray, rigid_amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local ``(u, v)`` at every vertex for stacked boundary ``loads`` and rigid amplitudes."""
        d = self.influence_loads @ loads + self.influence_rigid @ rigid_amplitudes
        return d[: self.n_vertices], d[self.n_vertices :]

    def element_areas(self) -> np.ndarray:
        """Reference area (m^2) of every mesh triangle."""
        x, y = self.mesh_points[0, self.mesh_triangles], self.mesh_points[1, self.mesh_triangles]
        return 0.5 * np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def strain_energy_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Plane-strain energy density per mesh triangle (J/m^3) from vertex displacements.

        ``u``, ``v`` are the displacements at every mesh vertex. Strain is the constant
        P1 strain of each triangle (exact for ``element_order = 1``); rigid motion carries
        no linear strain, so rotating-frame displacements can be used as they are.
        ``W = lambda/2 tr(eps)^2 + mu eps:eps``.
        """
        t = self.mesh_triangles
        x, y = self.mesh_points[0, t], self.mesh_points[1, t]  # (3, n_el)
        u_e, v_e = np.asarray(u, dtype=float)[t], np.asarray(v, dtype=float)[t]
        # Gradients of the linear shape functions: dN_i/dx = b_i / 2A, dN_i/dy = c_i / 2A.
        b = np.stack([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.stack([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        two_A = x[0] * b[0] + x[1] * b[1] + x[2] * b[2]
        e_xx = np.sum(b * u_e, axis=0) / two_A
        e_yy = np.sum(c * v_e, axis=0) / two_A
        e_xy = 0.5 * np.sum(c * u_e + b * v_e, axis=0) / two_A
        E, nu = self.mesh_element_E, self.mesh_element_nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        tr = e_xx + e_yy
        return 0.5 * lam * tr**2 + mu * (e_xx**2 + e_yy**2 + 2.0 * e_xy**2)

    def to_payload(self) -> dict[str, np.ndarray]:
        return {RENDER_KEY_PREFIX + f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_payload(cls, data) -> RenderSection:
        files = set(getattr(data, "files", data.keys()))
        missing = [f.name for f in fields(cls) if RENDER_KEY_PREFIX + f.name not in files]
        if missing:
            raise KeyError(f"render section arrays missing: {missing}")
        return cls(
            **{
                f.name: np.asarray(data[RENDER_KEY_PREFIX + f.name], dtype=int if f.name in _INT_FIELDS else float)
                for f in fields(cls)
            }
        )


def element_moduli(fem_result: ComplianceResult) -> tuple[np.ndarray, np.ndarray]:
    """Per-element ``(E, nu)``; a graded foam's ``E(x)`` is taken at the element centroid."""
    from compliance_fem.geometry.profile import REGION_LOWER, REGION_UPPER

    md = fem_result.mesh_data
    config = fem_result.config
    p = np.asarray(md.mesh.p, dtype=float)
    t = np.asarray(md.mesh.t, dtype=int)
    x_c = p[0, t].mean(axis=0)
    L = float(config.shoe_length_m)
    E = np.full(t.shape[1], np.nan)
    nu = np.full(t.shape[1], np.nan)
    for name, mat in config.region_materials().items():
        if name not in (REGION_UPPER, REGION_LOWER):
            continue
        idx = md.region_elements(name)
        E[idx] = mat.E_heel + (mat.E_toe - mat.E_heel) * (x_c[idx] / L)
        nu[idx] = mat.nu
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(nu))):
        raise ValueError("Every mesh element must belong to a foam region with a material.")
    return E, nu
