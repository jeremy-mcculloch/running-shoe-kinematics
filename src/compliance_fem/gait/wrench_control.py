"""3×3 wrench-controlled COP fit: solve [dx, dy, α] from prescribed (Fx, Fy, Mz).

Every candidate is one stored contact interval ``I_ij``. The interval's affine
closure column (curved-sole term, zero for a flat sole) is part of the known
right-hand side; the unilateral checks use the same vectorized reconstruction
as :mod:`compliance_fem.force_control`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.contact_basis import COL_ALPHA, COL_BRX, COL_BRY, COL_BX, COL_BY, COL_CONST
from compliance_fem.contact_lookup import (
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MZ,
    ContactLookupResult,
    contact_type_from_code,
)
from compliance_fem.contact_topology import ContactType
from compliance_fem.corotation import (
    contract_basis,
    corotation_angle,
    rotate_force_to_local,
    rotate_vector_to_fixed,
)
from compliance_fem.force_control import (
    EXACT_ALL,
    Tolerances,
    reconstruct_rows,
    refine_reconstruction,
    refine_top_k,
)

# Per-candidate diagnostics copied from ``reconstruct_rows`` (§9–§12, §15).
CANDIDATE_DIAGNOSTIC_FIELDS = (
    "min_free_gap",
    "min_free_gap_node",
    "max_free_penetration",
    "min_contact_reaction",
    "min_contact_reaction_node",
    "max_contact_tension",
    "heel_edge_reaction_local_x",
    "heel_edge_reaction_local_y",
    "toe_edge_reaction_local_x",
    "toe_edge_reaction_local_y",
    "heel_edge_normal_reaction",
    "toe_edge_normal_reaction",
    "heel_adjacent_free_gap",
    "toe_adjacent_free_gap",
    "complementarity_score",
    "gap_violation_term",
    "reaction_violation_term",
    "force_violation_term",
    "penalty_term",
    "violation_score",
    "disconnected_contact_warning",
)


@dataclass(frozen=True)
class WrenchCandidate:
    row: int
    contact_type: ContactType
    contact_start_index: int
    contact_end_index: int
    contact_start_x: float
    contact_end_x: float
    anchor_x: float
    d_ax: float
    d_ay: float
    alpha: float
    theta_deg: float
    Fx_pred: float
    Fy_pred: float
    Mz_pred: float
    force_residual: float
    moment_residual: float
    cop_residual: float
    cond: float
    max_free_penetration: float
    min_contact_reaction: float
    admissible: bool
    status: str
    gamma: NDArray[np.float64]
    toe: Any = None
    anchor_y: float = 0.0
    search_method: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def interval(self) -> tuple[int, int]:
        return int(self.contact_start_index), int(self.contact_end_index)

    @property
    def violation_score(self) -> float:
        return float(self.diagnostics.get("violation_score", np.inf))


@dataclass(frozen=True)
class UnilateralCheck:
    max_free_penetration: float
    min_contact_reaction: float
    admissible: bool
    checked: bool
    diagnostics: dict = field(default_factory=dict)


def _candidate_geometry(lookup: ContactLookupResult, row: int) -> dict:
    return {
        "contact_type": contact_type_from_code(int(lookup.contact_type_codes[row])),
        "contact_start_index": int(lookup.contact_start_index[row]),
        "contact_end_index": int(lookup.contact_end_index[row]),
        "contact_start_x": float(lookup.contact_start_x[row]),
        "contact_end_x": float(lookup.contact_end_x[row]),
        "anchor_x": float(lookup.contact_anchor_reference_x[row]),
        "anchor_y": float(lookup.contact_anchor_reference_y[row]),
    }


def diagnostics_from_reconstruction(rec: dict, k: int) -> dict:
    out = {}
    for name in CANDIDATE_DIAGNOSTIC_FIELDS:
        v = rec[name][k]
        out[name] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    return out


def unilateral_check(
    lookup: ContactLookupResult,
    row: int,
    gamma: NDArray[np.float64],
    varphi: float,
    tol: Tolerances,
    F_star_fixed: NDArray[np.float64] | None = None,
) -> UnilateralCheck:
    """All-node fixed-frame nonpenetration and compression-only checks for one interval.

    Free nodes need ``g >= -tau_g``; contact nodes need ``R_n >= -tau_R`` (the
    edge nodes and every interior node). Force reconstruction is not part of this
    check; use ``force_residual`` separately.
    """
    gamma = np.asarray(gamma, dtype=float).reshape(1, 5)
    if not np.all(np.isfinite(gamma)) or not bool(lookup.valid_mask[int(row)]):
        return UnilateralCheck(float("nan"), float("nan"), False, False)
    F = np.zeros(2) if F_star_fixed is None else np.asarray(F_star_fixed, dtype=float)
    rec = reconstruct_rows(
        lookup, np.array([int(row)]), gamma, float(varphi), F, tol,
        force_threshold=np.inf, full_fields=False, exact=EXACT_ALL,
    )
    sc = rec["scales"]
    min_gap = float(rec["min_free_gap_interpolated"][0])
    min_R = float(rec["min_contact_reaction"][0])
    ok = bool(rec["finite"][0]) and min_gap >= -sc.tau_g_eff and min_R >= -sc.tau_R_eff
    return UnilateralCheck(
        float(rec["max_free_penetration"][0]), min_R, ok, True,
        diagnostics_from_reconstruction(rec, 0),
    )


def _wrench_modes(scalars_row: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (3, 6) affine matrix: rows (Fx, Fy, Mz), columns = affine columns."""
    S = np.asarray(scalars_row, dtype=float)
    return np.stack([S[:, SCALAR_FX], S[:, SCALAR_FY], S[:, SCALAR_MZ]])


def assemble_Kw(scalars_row: NDArray[np.float64]) -> NDArray[np.float64]:
    """K_W columns map z=[dx, dy, α] → wrench (local-frame scalars)."""
    W = _wrench_modes(scalars_row)
    return np.column_stack([W[:, COL_BX], W[:, COL_BY], W[:, COL_ALPHA]])


def _fix_alpha_solve_translations(
    Kw: NDArray[np.float64],
    rhs: NDArray[np.float64],
    theta_deg: float,
) -> tuple[NDArray[np.float64] | None, float]:
    """Fix α = tan(θ); solve translations from Fx, Fy only (moment stays a residual)."""
    alpha_fix = float(np.tan(np.deg2rad(theta_deg)))
    Kf = Kw[:2, :2]
    rhs_f = rhs[:2] - alpha_fix * Kw[:2, 2]
    try:
        dxy = np.linalg.solve(Kf, rhs_f)
    except np.linalg.LinAlgError:
        return None, alpha_fix
    return np.array([float(dxy[0]), float(dxy[1]), alpha_fix], dtype=np.float64), alpha_fix


def _nan_candidate(lookup, row: int, cond: float, status: str) -> WrenchCandidate:
    nan = float("nan")
    return WrenchCandidate(
        row=row, **_candidate_geometry(lookup, row),
        d_ax=nan, d_ay=nan, alpha=nan, theta_deg=nan, Fx_pred=nan, Fy_pred=nan, Mz_pred=nan,
        force_residual=float("inf"), moment_residual=float("inf"), cop_residual=nan, cond=cond,
        max_free_penetration=nan, min_contact_reaction=nan, admissible=False, status=status,
        gamma=np.full(5, np.nan),
    )


def solve_wrench_control(
    lookup: ContactLookupResult,
    *,
    Fx_star: float,
    Fy_star: float,
    Mz_star: float,
    phi_rad: float,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
    cond_warn: float = 1.0e8,
    tolerances: Tolerances | None = None,
    low_force_fallback: bool = False,
    theta_prior_deg: float | None = None,
) -> list[WrenchCandidate]:
    """Evaluate every valid interval record for a prescribed fixed-frame wrench (legacy fit-cop).

    Forces are rotated into the local frame for the linear solve. When
    ``theta_prior_deg`` is set it seeds θ for low-force / ill-conditioned frames
    and replaces out-of-bounds α; fixed-α solves match Fx, Fy only.
    """
    tol = tolerances or Tolerances()
    scalars = np.asarray(lookup.scalar_lookup, dtype=float)
    phi_ref = float(getattr(lookup, "phi_ref", 0.0))
    varphi = corotation_angle(phi_rad, phi_ref)
    r_x = float(np.cos(varphi) - 1.0)
    r_y = float(-np.sin(varphi))
    F_local = rotate_force_to_local(Fx_star, Fy_star, varphi)
    w_star = np.array([F_local[0], F_local[1], float(Mz_star)], dtype=np.float64)
    prior_clipped: float | None = None
    if theta_prior_deg is not None and np.isfinite(theta_prior_deg):
        prior_clipped = float(np.clip(theta_prior_deg, theta_min_deg, theta_max_deg))

    out: list[WrenchCandidate] = []
    solved: list[tuple[int, NDArray[np.float64], float, str, float]] = []
    for row in lookup.valid_rows:
        row = int(row)
        W = _wrench_modes(scalars[row])
        if not np.all(np.isfinite(W)):
            continue
        Kw = assemble_Kw(scalars[row])
        rhs = w_star - W[:, COL_CONST] - r_x * W[:, COL_BRX] - r_y * W[:, COL_BRY]
        status = "ok"
        svals = np.linalg.svd(Kw, compute_uv=False)
        cond = float(svals[0] / max(float(svals[-1]), 1e-300))
        ill = not np.isfinite(cond) or cond > cond_warn or float(svals[-1]) <= 0.0
        if low_force_fallback and prior_clipped is not None:
            z, _ = _fix_alpha_solve_translations(Kw, rhs, prior_clipped)
            status = "low_force_fallback" if z is not None else "ill_conditioned"
        elif ill:
            z = None
            status = "ill_conditioned"
            if prior_clipped is not None:
                z, _ = _fix_alpha_solve_translations(Kw, rhs, prior_clipped)
                status = "theta_prior_fallback" if z is not None else "ill_conditioned"
        else:
            z = np.linalg.solve(Kw, rhs)
        if z is None or not np.all(np.isfinite(z)):
            out.append(_nan_candidate(lookup, row, cond, status))
            continue
        d_ax, d_ay, alpha = float(z[0]), float(z[1]), float(z[2])
        theta_deg = float(np.rad2deg(np.arctan(alpha)))
        if theta_deg < theta_min_deg or theta_deg > theta_max_deg:
            if prior_clipped is not None and status not in {"low_force_fallback", "theta_prior_fallback"}:
                theta_deg = prior_clipped
                fallback_status = "theta_prior_fallback"
            else:
                theta_deg = float(np.clip(theta_deg, theta_min_deg, theta_max_deg))
                fallback_status = "theta_clamped"
            z_fix, alpha_fix = _fix_alpha_solve_translations(Kw, rhs, theta_deg)
            if z_fix is not None:
                d_ax, d_ay = float(z_fix[0]), float(z_fix[1])
                alpha = alpha_fix
                status = fallback_status if status == "ok" else f"{status}+{fallback_status}"
            else:
                status = "theta_out_of_bounds" if status == "ok" else f"{status}+theta_out_of_bounds"
        gamma = np.array([alpha, d_ax, d_ay, r_x, r_y], dtype=np.float64)
        solved.append((row, gamma, theta_deg, status, cond))

    if solved:
        rows = np.array([s[0] for s in solved], dtype=int)
        gam = np.stack([s[1] for s in solved])
        rec = reconstruct_rows(
            lookup, rows, gam, varphi, np.array([Fx_star, Fy_star], dtype=float), tol,
            force_threshold=np.inf, full_fields=False,
        )
        sc = rec["scales"]
        if not np.all(rec["fields_exact"]):
            _refine_best_candidate(lookup, rec, solved, scalars, varphi, Fx_star, Fy_star, Mz_star,
                                   theta_min_deg, theta_max_deg)
    for k, (row, gamma, theta_deg, status, cond) in enumerate(solved):
        w_pred = contract_basis(_wrench_modes(scalars[row]), gamma, mode_axis=1)
        fx_f, fy_f = rotate_vector_to_fixed(w_pred[0], w_pred[1], varphi)
        force_res = float(np.hypot(fx_f - Fx_star, fy_f - Fy_star))
        cop_res = float("nan")
        if abs(F_local[1]) > 1e-12 and abs(w_pred[1]) > 1e-12:
            cop_res = float(abs(w_pred[2] / w_pred[1] - Mz_star / F_local[1]))
        min_gap = float(rec["min_free_gap_interpolated"][k])
        min_R = float(rec["min_contact_reaction"][k])
        unilateral_ok = bool(rec["finite"][k]) and min_gap >= -sc.tau_g_eff and min_R >= -sc.tau_R_eff
        theta_ok = theta_min_deg <= theta_deg <= theta_max_deg
        admissible = unilateral_ok and theta_ok and status != "ill_conditioned"
        if not theta_ok and "theta_out_of_bounds" not in status:
            status = "theta_out_of_bounds" if status == "ok" else f"{status}+theta_out_of_bounds"
        out.append(
            WrenchCandidate(
                row=row, **_candidate_geometry(lookup, row),
                d_ax=float(gamma[1]), d_ay=float(gamma[2]), alpha=float(gamma[0]), theta_deg=theta_deg,
                Fx_pred=float(fx_f), Fy_pred=float(fy_f), Mz_pred=float(w_pred[2]),
                force_residual=force_res, moment_residual=float(abs(w_pred[2] - Mz_star)),
                cop_residual=cop_res, cond=cond,
                max_free_penetration=float(rec["max_free_penetration"][k]),
                min_contact_reaction=min_R, admissible=admissible, status=status, gamma=gamma,
                diagnostics=diagnostics_from_reconstruction(rec, k),
            )
        )
    return out


def _refine_best_candidate(
    lookup, rec: dict, solved: list, scalars, varphi: float, Fx_star: float, Fy_star: float, Mz_star: float,
    theta_min_deg: float, theta_max_deg: float,
) -> None:
    """Make the :func:`pick_instant_best` winner exact when only edge-only bounds
    are known for some records (lookups without stored nodal fields)."""
    sc = rec["scales"]
    res = np.empty(len(solved))
    gate = np.empty(len(solved), dtype=bool)
    for k, (row, gamma, theta_deg, status, _cond) in enumerate(solved):
        w = contract_basis(_wrench_modes(scalars[row]), gamma, mode_axis=1)
        fx_f, fy_f = rotate_vector_to_fixed(w[0], w[1], varphi)
        res[k] = float(np.hypot(fx_f - Fx_star, fy_f - Fy_star)) + 0.25 * float(abs(w[2] - Mz_star))
        gate[k] = theta_min_deg <= theta_deg <= theta_max_deg and status != "ill_conditioned"

    def keys(p: np.ndarray) -> tuple:
        ok = (
            rec["finite"][p]
            & (rec["min_free_gap_interpolated"][p] >= -sc.tau_g_eff)
            & (rec["min_contact_reaction"][p] >= -sc.tau_R_eff)
            & gate[p]
        )
        pen = rec["max_free_penetration"][p]
        min_R = rec["min_contact_reaction"][p]
        return (
            rec["rows"][p],
            np.where(np.isfinite(min_R), -min_R, 0.0),
            res[p],
            np.where(np.isfinite(pen), pen, 1e30),
            (~ok).astype(int),
        )

    refine_top_k(
        np.arange(len(solved)), keys, lambda p: rec["fields_exact"][p],
        lambda p: refine_reconstruction(lookup, rec, p),
    )


def pick_instant_best(cands: list[WrenchCandidate]) -> WrenchCandidate | None:
    if not cands:
        return None

    def key(c: WrenchCandidate) -> tuple:
        return (
            0 if c.admissible else 1,
            c.max_free_penetration if np.isfinite(c.max_free_penetration) else 1e30,
            c.force_residual + 0.25 * c.moment_residual,
            -c.min_contact_reaction if np.isfinite(c.min_contact_reaction) else 0.0,
            c.row,
        )

    return min(cands, key=key)
