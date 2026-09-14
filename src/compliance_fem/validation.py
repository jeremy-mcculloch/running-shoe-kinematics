"""Validation against analytical plane-strain solutions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from compliance_fem.boundaries import assemble_traction_load, reaction_basis_rows, yy_block
from compliance_fem.compliance import ComplianceResult
from compliance_fem.rigid_modes import mesh_centroid


@dataclass(frozen=True)
class ValidationReport:
    """Numerical validation metrics for a traction load case."""

    case_name: str
    displacement_rel_error: float
    force_balance_error: float
    moment_balance_error: float
    v_top_computed: np.ndarray
    v_top_analytical: np.ndarray


def analytical_top_displacement(
    x: np.ndarray,
    traction: Callable[[np.ndarray], np.ndarray],
    E: float,
    nu: float,
    H: float,
) -> np.ndarray:
    """Exact top displacement for affine traction with full frictionless bottom support."""
    factor = (1.0 - nu**2) * H / E
    return factor * traction(x)


def solve_bottom_contact(
    result: ComplianceResult,
    F_top: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Solve for bottom reactions and rigid motion given top nodal forces."""
    x_top = result.x_top
    x_bottom = result.x_bottom
    n_t = len(x_top)
    n_b = len(x_bottom)
    C_bb = yy_block(result.Cbb_force, n_b, n_b)
    C_bt = yy_block(result.Cbt_force, n_b, n_t)
    C_tt = yy_block(result.Ctt_force, n_t, n_t)
    C_tb = yy_block(result.Ctb_force, n_t, n_b)

    xc, _ = mesh_centroid(result.basis.mesh)
    R_b = reaction_basis_rows(x_bottom, xc)
    R_t = reaction_basis_rows(x_top, xc)

    A = np.zeros((n_b + 2, n_b + 2))
    rhs = np.zeros(n_b + 2)
    A[:n_b, :n_b] = C_bb
    A[:n_b, n_b:] = R_b
    A[n_b:, :n_b] = np.vstack([np.ones(n_b), x_bottom])
    rhs[:n_b] = -C_bt @ F_top
    rhs[n_b] = -np.sum(F_top)
    rhs[n_b + 1] = -np.dot(x_top, F_top)

    sol = np.linalg.solve(A, rhs)
    F_bottom = sol[:n_b]
    a, b = sol[n_b], sol[n_b + 1]
    v_top = C_tt @ F_top + C_tb @ F_bottom + R_t @ np.array([a, b])
    return F_bottom, a, b, v_top


def validate_traction_case(
    result: ComplianceResult,
    case_name: str,
    traction: Callable[[np.ndarray], np.ndarray],
) -> ValidationReport:
    """Validate reconstructed top displacement against the analytical solution."""
    basis = result.basis
    if basis is None:
        raise ValueError("ComplianceResult must include basis for traction assembly.")

    F_top = assemble_traction_load(basis, "top", traction, result.config.order)
    F_bottom, _, _, v_top = solve_bottom_contact(result, F_top)

    cfg = result.config
    v_analytical = analytical_top_displacement(
        result.x_top,
        traction,
        cfg.E,
        cfg.nu,
        cfg.H,
    )

    disp_err = np.linalg.norm(v_top - v_analytical) / max(np.linalg.norm(v_analytical), 1e-30)
    force_balance = np.sum(F_top) + np.sum(F_bottom)
    moment_balance = np.dot(result.x_top, F_top) + np.dot(result.x_bottom, F_bottom)

    return ValidationReport(
        case_name=case_name,
        displacement_rel_error=float(disp_err),
        force_balance_error=float(abs(force_balance)),
        moment_balance_error=float(abs(moment_balance)),
        v_top_computed=v_top,
        v_top_analytical=v_analytical,
    )


def run_standard_validations(result: ComplianceResult) -> list[ValidationReport]:
    """Run uniform and affine traction validation cases."""
    c0, c1 = 1000.0, 0.0
    uniform = validate_traction_case(result, "uniform", lambda x, c0=c0: np.full_like(x, c0, dtype=float))
    c0, c1 = 800.0, 500.0
    affine = validate_traction_case(result, "affine", lambda x, c0=c0, c1=c1: c0 + c1 * x)
    return [uniform, affine]
