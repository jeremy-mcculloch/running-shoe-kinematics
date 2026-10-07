"""Lookup-level superposition and unilateral admissibility for a raw coefficient vector.

This is a debug path that bypasses the force-controlled solve: it superposes
``z = z_0 + sum_k gamma_k z_k`` for every valid interval record with a given
``gamma = (alpha, d_ax, d_ay, r_x, r_y)`` and applies the same fixed-frame
all-node checks as :func:`compliance_fem.force_control.reconstruct_rows`. The
normal runtime entry point is ``force_control.evaluate_from_angles``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compliance_fem.contact_lookup import SCALAR_FY, SCALAR_MV, ContactLookupResult
from compliance_fem.corotation import N_BASIS_COEFFICIENTS
from compliance_fem.force_control import EXACT_ALL, Tolerances, reconstruct_rows


@dataclass(frozen=True)
class SuperposedProfile:
    """Superposed responses for every interval record (NaN on rejected records)."""

    coefficients: np.ndarray
    varphi: float
    Fy: np.ndarray
    M: np.ndarray
    x_ce: np.ndarray
    gap: np.ndarray
    reaction: np.ndarray
    violation: np.ndarray
    admissible: np.ndarray


@dataclass(frozen=True)
class SelectedCandidate:
    """Least-violating interval for a raw coefficient vector."""

    candidate_row: int
    contact_start_index: int
    contact_end_index: int
    contact_type: str
    anchor_x: float
    exactly_admissible: bool
    Fy: float
    M: float
    x_ce: float
    gap: np.ndarray
    reaction: np.ndarray
    violation: float
    admissible_rows: np.ndarray
    profile: SuperposedProfile


def center_of_effort(Fy: np.ndarray | float, M: np.ndarray | float, tol: float = 1e-14) -> np.ndarray | float:
    """Return M/Fy with NaN where |Fy| is below tolerance."""
    Fy_arr = np.asarray(Fy, dtype=float)
    M_arr = np.asarray(M, dtype=float)
    out = np.full_like(Fy_arr, np.nan, dtype=float)
    mask = np.abs(Fy_arr) > tol
    out[mask] = M_arr[mask] / Fy_arr[mask]
    if np.ndim(Fy) == 0:
        return float(out)
    return out


def varphi_from_gamma(gamma: np.ndarray) -> float:
    """Recover the rotation from ``r_x = cos(varphi) - 1`` and ``r_y = -sin(varphi)``."""
    g = np.asarray(gamma, dtype=float)
    return float(np.arctan2(-g[4], 1.0 + g[3]))


def superpose(
    lookup: ContactLookupResult,
    coefficients: np.ndarray | list[float],
    fy_tol: float = 1e-14,
    tolerances: Tolerances | None = None,
) -> SuperposedProfile:
    """Superpose every valid interval record for the raw coefficient vector ``gamma``.

    ``gap`` is the fixed-frame normal gap of every bottom node and ``reaction``
    the fixed-frame normal nodal reaction (zero on free nodes).
    """
    c = np.asarray(coefficients, dtype=float).reshape(-1)
    if c.size != N_BASIS_COEFFICIENTS:
        raise ValueError(
            "coefficients must be gamma = (alpha, d_ax, d_ay, r_x, r_y) with "
            f"{N_BASIS_COEFFICIENTS} entries."
        )
    tol = tolerances or Tolerances()
    varphi = varphi_from_gamma(c)
    rows = lookup.valid_rows
    n, n_b = lookup.n_records, int(lookup.n_bottom_nodes)
    rec = reconstruct_rows(
        lookup, rows, np.tile(c, (rows.size, 1)), varphi, np.zeros(2), tol, force_threshold=np.inf,
        exact=EXACT_ALL,
    )
    S6 = np.asarray(lookup.scalar_lookup)[rows]
    G6 = np.concatenate([[1.0], c])
    Fy = np.full(n, np.nan)
    M = np.full(n, np.nan)
    Fy[rows] = S6[:, :, SCALAR_FY] @ G6
    M[rows] = S6[:, :, SCALAR_MV] @ G6
    gap = np.full((n, n_b), np.nan)
    reaction = np.full((n, n_b), np.nan)
    gap[rows] = rec["full_bottom_gap"]
    reaction[rows] = rec["full_bottom_reaction_normal"]
    violation = np.full(n, np.inf)
    sc = rec["scales"]
    violation[rows] = rec["gap_violation_term"] + rec["reaction_violation_term"]
    admissible = np.zeros(n, dtype=bool)
    admissible[rows] = (rec["min_free_gap_interpolated"] >= -sc.tau_g_eff) & (
        rec["min_contact_reaction"] >= -sc.tau_R_eff
    )
    return SuperposedProfile(
        coefficients=c,
        varphi=varphi,
        Fy=Fy,
        M=M,
        x_ce=np.asarray(center_of_effort(Fy, M, tol=fy_tol), dtype=float),
        gap=gap,
        reaction=reaction,
        violation=violation,
        admissible=admissible,
    )


def select_candidate(
    lookup: ContactLookupResult,
    coefficients: np.ndarray | list[float],
    tau_g: float = 0.0,
    tau_R: float = 0.0,
    fy_tol: float = 1e-14,
) -> SelectedCandidate:
    """All admissible intervals and the least-violating one (ties -> lowest row)."""
    profile = superpose(lookup, coefficients, fy_tol=fy_tol, tolerances=Tolerances(tau_g=tau_g, tau_R=tau_R))
    admissible_rows = np.flatnonzero(profile.admissible)
    best_row = int(np.argmin(profile.violation))
    return SelectedCandidate(
        candidate_row=best_row,
        contact_start_index=int(lookup.contact_start_index[best_row]),
        contact_end_index=int(lookup.contact_end_index[best_row]),
        contact_type=lookup.contact_type(best_row).value,
        anchor_x=float(lookup.contact_anchor_reference_x[best_row]),
        exactly_admissible=bool(profile.admissible[best_row]),
        Fy=float(profile.Fy[best_row]),
        M=float(profile.M[best_row]),
        x_ce=float(profile.x_ce[best_row]),
        gap=profile.gap[best_row].copy(),
        reaction=profile.reaction[best_row].copy(),
        violation=float(profile.violation[best_row]),
        admissible_rows=admissible_rows,
        profile=profile,
    )
