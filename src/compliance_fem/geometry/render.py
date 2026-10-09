"""Lookup-side rendering data for the measured two-foam sole.

The lookup stores the reference mesh, the boundary node chains of each foam
region, and a render influence matrix so the GUI can draw both deformed
regions without any FEM solve:

    d_render = G [F_t ; F_bottom] + R_render alpha_rigid

``G`` (``2 n_r x (2 n_t + 2 n_b)``, component-major rows ``[u..., v...]``)
comes from one multi-RHS solve of the factorized free-body operator, exactly
like the plate influence; ``R_render`` is the rigid-mode matrix about the
solver reference ``(x_r, y_r)``. Per record, ``F_t`` / ``F_bottom`` and the
rigid amplitudes are contracted from the stored six-column affine bases with
``z = z_closure + sum_k gamma_k z_k``. Displacements are rotating-frame local
values; the fixed-frame transform is applied by the renderer.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from compliance_fem.fem.boundaries import build_vector_selector

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
    "render_influence_loads",
    "render_influence_rigid",
)
RENDER_KEYS = RENDER_INT_KEYS + RENDER_FLOAT_KEYS


def has_render_inputs(fem_result) -> bool:
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
    render_ids = np.unique(np.concatenate([loops["upper_foam"], loops["lower_foam"]]))

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


def render_section_to_payload(section: dict, geometry_metadata: dict) -> dict:
    payload = {key: np.asarray(section[key]) for key in RENDER_KEYS}
    payload["geometry_metadata_json"] = np.asarray(json.dumps(geometry_metadata, sort_keys=True))
    return payload


def render_node_displacements(lookup, row: int, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local (u, v) at ``render_node_ids`` for record ``row`` and affine coefficients (6,)."""
    sec = lookup.render_section
    if not sec:
        raise ValueError("Lookup has no measured render section.")
    c = np.asarray(coefficients, dtype=float).reshape(-1)
    r = int(row)
    f = lookup.record_fields([r])
    f_tx = c @ np.asarray(f["top_force_x"][0], dtype=float)
    f_ty = c @ np.asarray(f["top_force_y"][0], dtype=float)
    r_x = np.nan_to_num(c @ np.asarray(f["reaction_x"][0], dtype=float))
    r_y = np.nan_to_num(c @ np.asarray(f["reaction_y"][0], dtype=float))
    alpha = c @ np.asarray(lookup.rigid_alpha_basis[r], dtype=float)
    loads = np.concatenate([f_tx, f_ty, r_x, r_y])
    d = np.asarray(sec["render_influence_loads"]) @ loads + np.asarray(sec["render_influence_rigid"]) @ alpha
    n_r = d.size // 2
    return d[:n_r], d[n_r:]


def render_index(lookup, node_ids: np.ndarray) -> np.ndarray:
    """Positions of mesh ``node_ids`` inside ``render_node_ids``."""
    ids = np.asarray(lookup.render_section["render_node_ids"], dtype=int)
    idx = np.searchsorted(ids, np.asarray(node_ids, dtype=int))
    if np.any(idx >= ids.size) or np.any(ids[np.minimum(idx, ids.size - 1)] != node_ids):
        raise ValueError("Requested node is not a render node.")
    return idx
