"""Stiffness matrix assembly for plane-strain linear elasticity."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from skfem import Basis, Mesh
from skfem.assembly import BilinearForm
from skfem.element import ElementQuad1, ElementQuad2, ElementTriP1, ElementTriP2
from skfem.element import ElementVector
from skfem.helpers import ddot, div, sym_grad


def _vector_element(mesh: Mesh, order: int):
    if mesh.t.shape[0] == 3:
        scalar = ElementTriP1() if order == 1 else ElementTriP2()
    elif mesh.t.shape[0] == 4:
        scalar = ElementQuad1() if order == 1 else ElementQuad2()
    else:
        raise ValueError(f"Unsupported element connectivity with {mesh.t.shape[0]} nodes.")
    return ElementVector(scalar)


def verify_stiffness_symmetry(K: sparse.spmatrix, rtol: float = 1e-10) -> float:
    """Return the relative symmetry error ||K-K^T|| / ||K||."""
    diff = K - K.T
    norm = sparse.linalg.norm(K)
    rel = sparse.linalg.norm(diff) / max(norm, 1e-30)
    if rel > rtol:
        raise ValueError(f"Stiffness matrix is not symmetric: relative error {rel:.3e}.")
    return float(rel)


def plane_strain_moduli(E: float | np.ndarray, nu: float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return (lambda, G) for plane-strain isotropic elasticity."""
    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, G


def spatially_varying_plane_strain(E_heel: float, E_toe: float, L: float, nu: float):
    """Bilinear form with E(x) evaluated at quadrature coordinates."""

    @BilinearForm
    def form(u, v, w):
        x = w.x[0]
        E = E_heel + (E_toe - E_heel) * (x / L)
        lam, G = plane_strain_moduli(E, nu)
        return 2.0 * G * ddot(sym_grad(u), sym_grad(v)) + lam * div(u) * div(v)

    return form


def assemble_region_foam_stiffness(
    mesh: Mesh,
    region_elements: dict[str, np.ndarray],
    region_materials: dict,
    L: float,
    order: int = 1,
) -> tuple[sparse.csc_matrix, Basis, dict[str, sparse.csc_matrix]]:
    """Plane-strain ``K = sum_r K_r``, each region with its own material.

    ``region_materials[name]`` is a :class:`compliance_fem.contact.config.FoamMaterial`;
    its modulus ``E(x)`` is evaluated at quadrature points (constant unless the
    material is graded heel to toe). Every element must belong to exactly one region.
    """
    element = _vector_element(mesh, order)
    basis = Basis(mesh, element)
    n_el = int(mesh.t.shape[1])
    owner = np.full(n_el, -1, dtype=int)
    for k, name in enumerate(region_elements):
        idx = np.asarray(region_elements[name], dtype=int)
        if np.any(owner[idx] >= 0):
            raise ValueError(f"Elements are assigned to more than one foam region ({name}).")
        owner[idx] = k
    if np.any(owner < 0):
        raise ValueError(f"{int(np.count_nonzero(owner < 0))} elements belong to no foam region.")
    blocks: dict[str, sparse.csc_matrix] = {}
    K = None
    for name, idx in region_elements.items():
        mat = region_materials[name]
        sub = Basis(mesh, element, elements=np.asarray(idx, dtype=int))
        K_r = spatially_varying_plane_strain(mat.E_heel, mat.E_toe, L, mat.nu).assemble(sub).tocsc()
        blocks[name] = K_r
        K = K_r if K is None else K + K_r
    return K.tocsc(), basis, blocks
