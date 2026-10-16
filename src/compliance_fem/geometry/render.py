"""Lookup-side rendering data for the measured two-foam sole.

The lookup stores the reference mesh, the boundary node chains of each foam
region, and a render influence matrix so the GUI can draw both deformed
regions without any FEM solve:

    d_render = G [F_t ; F_bottom] + R_render alpha_rigid

The render nodes are every mesh vertex, so the GUI can draw the deformed
mesh and colour each element by its strain energy density
(:func:`element_strain_energy_density`) with the per-element plane-strain
moduli stored alongside (``mesh_element_E``, ``mesh_element_nu``; ``E`` at the
element centroid for a graded foam).

``G`` (``2 n_r x (2 n_t + 2 n_b)``, component-major rows ``[u..., v...]``)
comes from one multi-RHS solve of the factorized free-body operator, exactly
like the plate influence; ``R_render`` is the rigid-mode matrix about the
solver reference ``(x_r, y_r)``. ``F_t``, ``F_bottom`` and the rigid amplitudes
come from one continuous contact solution
(``contact_model.render_vertex_displacements``). Displacements are
rotating-frame local values; the fixed-frame transform is applied by the renderer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeGuard

import numpy as np
from scipy import sparse

from compliance_fem.fem.boundaries import build_vector_selector

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult

RENDER_INT_KEYS = (
    "mesh_triangles",
    "mesh_element_region",
    "render_node_ids",
    "top_curve_node_ids",
    "bottom_curve_node_ids",
    "heel_edge_node_ids",
    "interface_node_ids",
    "plate_curve_node_ids",
    "upper_foam_loop",
    "lower_foam_loop",
)
RENDER_FLOAT_KEYS = (
    "mesh_points",
    "mesh_element_E",
    "mesh_element_nu",
    "render_influence_loads",
    "render_influence_rigid",
)
RENDER_KEYS = RENDER_INT_KEYS + RENDER_FLOAT_KEYS


def has_render_inputs(fem_result: ComplianceResult | None) -> TypeGuard[ComplianceResult]:
    return fem_result is not None and getattr(fem_result, "geometry_metadata", None) is not None and (
        getattr(fem_result, "mesh_data", None) is not None
    )


def build_render_section(fem_result, *, x_r: float, y_r: float) -> dict:
    """Mesh, region chains and render influence for a measured ComplianceResult."""
    if fem_result.factorization is None or fem_result.basis is None or fem_result.n_primal is None:
        raise ValueError("Measured render influence needs the retained factorization and basis.")
    md = fem_result.mesh_data
    mesh = md.mesh
    p = np.asarray(mesh.p, dtype=float)
    loops = {name: np.asarray(ids, dtype=int) for name, ids in md.region_loops.items()}
    render_ids = np.arange(p.shape[1], dtype=int)
    element_E, element_nu = _element_moduli(fem_result)

    n_q = int(fem_result.n_primal)
    n_lambda = int(fem_result.n_lambda or 0)
    u_top, v_top = fem_result.selector_dofs("top")
    u_bot, v_bot = fem_result.selector_dofs("bottom")
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bot, v_bot, n_q)
    loads = sparse.hstack([S_top.T, S_bottom.T]).tocsc()
    rhs = np.zeros((n_q + n_lambda + 3, loads.shape[1]), dtype=float)
    rhs[:n_q] = loads.toarray()
    sol = np.asarray(fem_result.factorization.solve(rhs), dtype=float)
    u_r = np.asarray(fem_result.basis.nodal_dofs[0, render_ids], dtype=int)
    v_r = np.asarray(fem_result.basis.nodal_dofs[1, render_ids], dtype=int)
    G = np.vstack([sol[u_r], sol[v_r]])

    xr = p[0, render_ids] - float(x_r)
    yr = p[1, render_ids] - float(y_r)
    n_r = render_ids.size
    R = np.zeros((2 * n_r, 3), dtype=float)
    R[:n_r, 0] = 1.0
    R[n_r:, 1] = 1.0
    R[:n_r, 2] = -yr
    R[n_r:, 2] = xr

    return {
        "mesh_points": p.copy(),
        "mesh_triangles": np.asarray(mesh.t, dtype=int).copy(),
        "mesh_element_region": np.asarray(md.element_region, dtype=int).copy(),
        "mesh_element_E": element_E,
        "mesh_element_nu": element_nu,
        "render_node_ids": render_ids,
        "top_curve_node_ids": np.asarray(md.top_curve_nodes, dtype=int),
        "bottom_curve_node_ids": np.asarray(md.bottom_curve_nodes, dtype=int),
        "heel_edge_node_ids": np.asarray(md.heel_edge_nodes, dtype=int),
        "interface_node_ids": np.asarray(md.interface_nodes, dtype=int),
        "plate_curve_node_ids": np.asarray(md.plate_node_ids, dtype=int),
        "upper_foam_loop": loops["upper_foam"],
        "lower_foam_loop": loops["lower_foam"],
        "render_influence_loads": G,
        "render_influence_rigid": R,
    }


def _element_moduli(fem_result) -> tuple[np.ndarray, np.ndarray]:
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


def element_strain_energy_density(lookup, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plane-strain energy density per mesh triangle (J/m^3) from vertex displacements.

    ``u``, ``v`` are the displacements at every mesh vertex (``render_node_ids``
    order, which is all vertices). Strain is the constant P1 strain of each
    triangle's vertices (exact for ``element_order = 1``, the vertex
    interpolant for order 2); rigid motion carries no linear strain, so the
    rotating-frame displacements can be used as they are.
    ``W = lambda/2 tr(eps)^2 + mu eps:eps``.
    """
    sec = lookup.render_section
    p = np.asarray(sec["mesh_points"], dtype=float)
    t = np.asarray(sec["mesh_triangles"], dtype=int)
    if np.asarray(sec["render_node_ids"]).size != p.shape[1]:
        raise ValueError("Render section does not cover every mesh vertex; regenerate the lookup.")
    x, y = p[0, t], p[1, t]  # (3, n_el)
    u_e, v_e = np.asarray(u, dtype=float)[t], np.asarray(v, dtype=float)[t]
    # Gradients of the linear shape functions: dN_i/dx = b_i / 2A, dN_i/dy = c_i / 2A.
    b = np.stack([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.stack([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    two_A = x[0] * b[0] + x[1] * b[1] + x[2] * b[2]
    e_xx = np.sum(b * u_e, axis=0) / two_A
    e_yy = np.sum(c * v_e, axis=0) / two_A
    e_xy = 0.5 * np.sum(c * u_e + b * v_e, axis=0) / two_A
    E = np.asarray(sec["mesh_element_E"], dtype=float)
    nu = np.asarray(sec["mesh_element_nu"], dtype=float)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    tr = e_xx + e_yy
    return 0.5 * lam * tr**2 + mu * (e_xx**2 + e_yy**2 + 2.0 * e_xy**2)


def element_areas(lookup) -> np.ndarray:
    """Reference area (m^2) of every mesh triangle."""
    sec = lookup.render_section
    p = np.asarray(sec["mesh_points"], dtype=float)
    t = np.asarray(sec["mesh_triangles"], dtype=int)
    x, y = p[0, t], p[1, t]
    return 0.5 * np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))


def render_section_to_payload(section: dict, geometry_metadata: dict) -> dict:
    payload = {key: np.asarray(section[key]) for key in RENDER_KEYS}
    payload["geometry_metadata_json"] = np.asarray(json.dumps(geometry_metadata, sort_keys=True))
    return payload


def render_index(lookup, node_ids: np.ndarray) -> np.ndarray:
    """Positions of mesh ``node_ids`` inside ``render_node_ids``."""
    ids = np.asarray(lookup.render_section["render_node_ids"], dtype=int)
    idx = np.searchsorted(ids, np.asarray(node_ids, dtype=int))
    if np.any(idx >= ids.size) or np.any(ids[np.minimum(idx, ids.size - 1)] != node_ids):
        raise ValueError("Requested node is not a render node.")
    return idx
