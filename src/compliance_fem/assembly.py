"""Stiffness matrix assembly for plane-strain linear elasticity."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse
from skfem import Basis, Mesh
from skfem.assembly import BilinearForm, Functional
from skfem.element import ElementQuad1, ElementQuad2, ElementTriP1, ElementTriP2
from skfem.element import ElementVector
from skfem.helpers import ddot, div, sym_grad
from skfem.models.elasticity import lame_parameters, linear_elasticity

from compliance_fem.config import LayeredPlateConfig


def _vector_element(mesh: Mesh, order: int):
    if mesh.t.shape[0] == 3:
        scalar = ElementTriP1() if order == 1 else ElementTriP2()
    elif mesh.t.shape[0] == 4:
        scalar = ElementQuad1() if order == 1 else ElementQuad2()
    else:
        raise ValueError(f"Unsupported element connectivity with {mesh.t.shape[0]} nodes.")
    return ElementVector(scalar)


def create_basis(mesh: Mesh, order: int) -> Basis:
    """Create a vector displacement basis on the mesh."""
    element = _vector_element(mesh, order)
    return Basis(mesh, element)


def assemble_stiffness(
    mesh: Mesh,
    E: float,
    nu: float,
    order: int,
) -> Tuple[sparse.csc_matrix, Basis]:
    """Assemble the free-body plane-strain stiffness matrix K."""
    basis = create_basis(mesh, order)
    lam, mu = lame_parameters(E, nu)
    stiffness = linear_elasticity(lam, mu)
    K = stiffness.assemble(basis).tocsc()
    return K, basis


def verify_stiffness_symmetry(K: sparse.spmatrix, rtol: float = 1e-10, atol: float = 1e-12) -> float:
    """Return the relative symmetry error ||K-K^T|| / ||K||."""
    diff = K - K.T
    norm = sparse.linalg.norm(K)
    rel = sparse.linalg.norm(diff) / max(norm, 1e-30)
    if rel > rtol:
        raise ValueError(f"Stiffness matrix is not symmetric: relative error {rel:.3e}.")
    return float(rel)


def _scalar_element(mesh: Mesh, order: int):
    if mesh.t.shape[0] == 3:
        return ElementTriP1() if order == 1 else ElementTriP2()
    if mesh.t.shape[0] == 4:
        return ElementQuad1() if order == 1 else ElementQuad2()
    raise ValueError(f"Unsupported element connectivity with {mesh.t.shape[0]} nodes.")


def subdomain_elements(mesh: Mesh, name: str) -> np.ndarray:
    """Return element indices for a named subdomain."""
    subdomains = getattr(mesh, "subdomains", None) or {}
    if name not in subdomains:
        raise KeyError(f"Mesh is missing subdomain '{name}'.")
    return np.asarray(subdomains[name], dtype=int)


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


def assemble_layered_foam_stiffness(
    mesh: Mesh,
    config: LayeredPlateConfig,
) -> tuple[sparse.csc_matrix, Basis, sparse.csc_matrix, sparse.csc_matrix]:
    """Assemble K_foam = K1 + K2 with quadrature-true E2(x) on the lower block."""
    order = config.element_order
    element = _vector_element(mesh, order)
    upper = subdomain_elements(mesh, "upper_foam")
    lower = subdomain_elements(mesh, "lower_foam")
    basis = Basis(mesh, element)
    basis_upper = Basis(mesh, element, elements=upper)
    basis_lower = Basis(mesh, element, elements=lower)

    lam, mu = lame_parameters(config.E1, config.nu1)
    K1 = linear_elasticity(lam, mu).assemble(basis_upper).tocsc()
    K2 = spatially_varying_plane_strain(
        config.E_heel, config.E_toe, config.L, config.nu2
    ).assemble(basis_lower).tocsc()
    if K1.shape != K2.shape or K1.shape[0] != basis.N:
        raise ValueError(
            f"Foam stiffness blocks have incompatible shapes {K1.shape}, {K2.shape}, n={basis.N}."
        )
    K = (K1 + K2).tocsc()
    return K, basis, K1, K2


def integrate_lower_modulus(mesh: Mesh, config: LayeredPlateConfig) -> float:
    """Return ∫_{Ω2} E2(x) dA using the same quadrature as stiffness assembly."""
    lower = subdomain_elements(mesh, "lower_foam")
    scalar = _scalar_element(mesh, config.element_order)
    basis = Basis(mesh, scalar, elements=lower)

    @Functional
    def integrand(w):
        return config.E2(w.x[0])

    return float(integrand.assemble(basis))


def exact_lower_modulus_integral(config: LayeredPlateConfig) -> float:
    """Exact ∫_0^L E2(x) h2(x) dx for linearly varying thickness and modulus."""
    L = config.L
    a_e, b_e = config.E_heel, (config.E_toe - config.E_heel) / L
    a_h, b_h = config.h2_heel, (config.h2_toe - config.h2_heel) / L
    # ∫ (a_e + b_e x)(a_h + b_h x) dx from 0 to L
    return float(a_e * a_h * L + 0.5 * (a_e * b_h + b_e * a_h) * L**2 + (b_e * b_h / 3.0) * L**3)
