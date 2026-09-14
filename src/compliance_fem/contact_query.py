"""Superposition, center-of-effort, and unilateral admissibility for contact lookup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compliance_fem.contact_lookup import (
    SCALAR_FY,
    SCALAR_MV,
    ContactLookupResult,
)
from compliance_fem.contact_topology import free_contact_sets
from compliance_fem.corotation import N_BASIS_COEFFICIENTS, contract_basis


@dataclass(frozen=True)
class SuperposedProfile:
    """Weighted top-profile reconstruction over all candidates."""

    coefficients: np.ndarray
    Fy: np.ndarray
    M: np.ndarray
    x_ce: np.ndarray
    gap: np.ndarray
    reaction: np.ndarray
    violation: np.ndarray
    admissible: np.ndarray


@dataclass(frozen=True)
class SelectedCandidate:
    """Result of selecting an admissible / least-violating contact edge."""

    candidate_row: int
    candidate_index: int
    contact_type: str
    l: float
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


def _sets_for_row(
    lookup: ContactLookupResult,
    candidate_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(free, contact)`` node indices for one record row.

    Schema v5 stores membership in ``contact_mask``. Older fixtures without a
    mask fall back to the historical toe-family ``free_contact_sets`` helper.
    """
    if lookup.contact_mask is None:
        i = int(lookup.candidate_indices[candidate_row])
        return free_contact_sets(i, len(lookup.x_bottom))
    mask = np.asarray(lookup.contact_mask[candidate_row], dtype=bool)
    contact = np.flatnonzero(mask)
    free = np.flatnonzero(~mask)
    return free, contact


def superpose(
    lookup: ContactLookupResult,
    coefficients: np.ndarray | list[float],
    fy_tol: float = 1e-14,
) -> SuperposedProfile:
    """Superpose basis responses for the raw runtime coefficient vector gamma.

    ``gamma = (alpha, d_lx, d_ly, r_x, r_y)`` in the saved basis order. This is
    a lookup-level debug path that bypasses the force-controlled solve; the
    normal runtime entry point is ``force_control.evaluate_from_angles``.

    All quantities are local (rotating-frame): ``Fy``, ``Mv``, the bottom
    vertical displacement, and the vertical nodal reaction. Combined center of
    effort uses Mv/Fy of the weighted totals, never the average of individual
    x_ce^(k) values.
    """
    c = np.asarray(coefficients, dtype=float).reshape(-1)
    if c.size != N_BASIS_COEFFICIENTS:
        raise ValueError(
            "coefficients must be gamma = (alpha, d_lx, d_ly, r_x, r_y) with "
            f"{N_BASIS_COEFFICIENTS} entries."
        )
    Fy = contract_basis(lookup.scalar_lookup[:, :, SCALAR_FY], c, mode_axis=1)
    M = contract_basis(lookup.scalar_lookup[:, :, SCALAR_MV], c, mode_axis=1)
    gap = contract_basis(lookup.gap_basis, c, mode_axis=1)
    reaction = contract_basis(lookup.reaction_basis, c, mode_axis=1)
    x_ce = center_of_effort(Fy, M, tol=fy_tol)
    return SuperposedProfile(
        coefficients=c,
        Fy=Fy,
        M=M,
        x_ce=np.asarray(x_ce, dtype=float),
        gap=gap,
        reaction=reaction,
        violation=np.full(len(lookup.candidate_indices), np.nan),
        admissible=np.zeros(len(lookup.candidate_indices), dtype=bool),
    )


def violation_score(
    lookup: ContactLookupResult,
    gap: np.ndarray,
    reaction: np.ndarray,
    candidate_row: int,
    weights: np.ndarray | None = None,
    g_scale: float | None = None,
    R_scale: float | None = None,
    scale_floor: float = 1e-30,
) -> float:
    """Normalized unilateral violation score J for one record row.

    Free and contact sets come from the record's own stored mask, so heel, full,
    and toe records are all handled without inspecting ``l``.
    """
    free, contact = _sets_for_row(lookup, candidate_row)
    g = gap[candidate_row]
    r = reaction[candidate_row]
    if weights is None:
        weights = np.ones(len(lookup.x_bottom), dtype=float)
    weights = np.asarray(weights, dtype=float)

    g_pen = np.minimum(g[free], 0.0) if free.size else np.array([])
    r_pen = np.minimum(r[contact], 0.0) if contact.size else np.array([])

    if g_scale is None:
        g_scale = max(float(np.max(np.abs(g))), scale_floor)
    if R_scale is None:
        R_scale = max(float(np.max(np.abs(r))), scale_floor)
    g_scale = max(float(g_scale), scale_floor)
    R_scale = max(float(R_scale), scale_floor)

    gap_term = float(np.sum(weights[free] * g_pen**2) / g_scale**2) if free.size else 0.0
    reac_term = float(np.sum(weights[contact] * r_pen**2) / R_scale**2) if contact.size else 0.0
    return gap_term + reac_term


def is_admissible(
    lookup: ContactLookupResult,
    gap: np.ndarray,
    reaction: np.ndarray,
    candidate_row: int,
    tau_g: float = 0.0,
    tau_R: float = 0.0,
) -> bool:
    """Return True if free gaps and contact reactions satisfy unilateral inequalities."""
    free, contact = _sets_for_row(lookup, candidate_row)
    g = gap[candidate_row]
    r = reaction[candidate_row]
    gap_ok = True if free.size == 0 else bool(np.all(g[free] >= -tau_g))
    reac_ok = True if contact.size == 0 else bool(np.all(r[contact] >= -tau_R))
    return gap_ok and reac_ok


def evaluate_admissibility(
    lookup: ContactLookupResult,
    profile: SuperposedProfile,
    tau_g: float = 0.0,
    tau_R: float = 0.0,
    weights: np.ndarray | None = None,
) -> SuperposedProfile:
    """Fill violation scores and admissibility flags on a superposed profile."""
    n = len(lookup.candidate_indices)
    violation = np.zeros(n, dtype=float)
    admissible = np.zeros(n, dtype=bool)
    for row in range(n):
        violation[row] = violation_score(
            lookup,
            profile.gap,
            profile.reaction,
            row,
            weights=weights,
        )
        admissible[row] = is_admissible(
            lookup,
            profile.gap,
            profile.reaction,
            row,
            tau_g=tau_g,
            tau_R=tau_R,
        )
    return SuperposedProfile(
        coefficients=profile.coefficients,
        Fy=profile.Fy,
        M=profile.M,
        x_ce=profile.x_ce,
        gap=profile.gap,
        reaction=profile.reaction,
        violation=violation,
        admissible=admissible,
    )


def select_candidate(
    lookup: ContactLookupResult,
    coefficients: np.ndarray | list[float],
    tau_g: float = 0.0,
    tau_R: float = 0.0,
    fy_tol: float = 1e-14,
    weights: np.ndarray | None = None,
) -> SelectedCandidate:
    """Select all admissible candidates and the minimum-J candidate."""
    profile = evaluate_admissibility(
        lookup,
        superpose(lookup, coefficients, fy_tol=fy_tol),
        tau_g=tau_g,
        tau_R=tau_R,
        weights=weights,
    )
    admissible_rows = np.flatnonzero(profile.admissible)
    best_row = int(np.argmin(profile.violation))
    exactly = bool(profile.admissible[best_row])
    return SelectedCandidate(
        candidate_row=best_row,
        candidate_index=int(lookup.candidate_indices[best_row]),
        contact_type=lookup.contact_type(best_row).value,
        l=float(lookup.candidate_l[best_row]),
        exactly_admissible=exactly,
        Fy=float(profile.Fy[best_row]),
        M=float(profile.M[best_row]),
        x_ce=float(profile.x_ce[best_row]),
        gap=profile.gap[best_row].copy(),
        reaction=profile.reaction[best_row].copy(),
        violation=float(profile.violation[best_row]),
        admissible_rows=admissible_rows,
        profile=profile,
    )
