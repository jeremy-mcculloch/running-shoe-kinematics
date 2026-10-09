"""Force-controlled interval contact evaluation in a top-attached co-rotating frame.

Runtime inputs are the fixed-frame top resultants ``Fx``, ``Fy``, the absolute
heel-to-toe chord angle ``phi``, and the toe-bending shape angle ``theta``.

Every stored interval record ``I_ij`` is reconstructed the same way:

1. Rotate the requested fixed-frame force into the local frame,
   ``F^{T,*} = Q(varphi)^T F^{F,*}``.
2. Solve ``K_F [d_ax, d_ay]^T = F^{T,*} - F_0 - alpha F_alpha - r_x F_rx - r_y F_ry``
   for the interval-anchor translation (``F_0`` is the curved-sole closure).
3. Superpose the affine responses ``z = z_0 + sum_k gamma_k z_k`` with
   ``gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)]``.
4. Transform every bottom node to the fixed frame and evaluate the signed
   ground gap ``g_k = n_g . r_k`` (free nodes) and the normal reaction
   ``R_n,k = n_g . Q R_k`` (contact nodes), plus the edge diagnostics.
5. Apply the all-node unilateral checks and the violation score.

All records are evaluated with vectorized contractions; there is no runtime FEM
solve. Interval selection (:func:`select_contact_candidate`) searches near the
previous interval first, then expands, then searches globally. Temporal
continuity is only a tie-breaker among admissible candidates.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from compliance_fem.contact.basis import (
    BASIS_ORDER,
    COL_ALPHA,
    COL_BRX,
    COL_BRY,
    COL_BX,
    COL_BY,
    COL_CONST,
    N_AFFINE_COLUMNS,
    shape_mode_phi1,
)
from compliance_fem.contact.lookup import (
    LOOKUP_SCHEMA_VERSION,
    REGENERATE_LOOKUP_MESSAGE,
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    ContactLookupResult,
    contact_type_from_code,
)
from compliance_fem.contact.topology import (
    ContactMode,
    ContactType,
    contact_mask_from_arrays,
    interval_distance,
    mode_mask,
    parse_contact_mode,
    validate_interval,
)
from compliance_fem.contact.corotation import (
    corotation_angle,
    rotate_force_to_local,
    rotation_coefficients,
    shape_amplitude,
)

if TYPE_CHECKING:
    from compliance_fem.fem.plate_response import PlateRuntimeState

__all__ = (
    "ContactMode",
    "ContactType",
    "Tolerances",
    "SelectionConfig",
    "CandidateEvaluation",
    "ContactSelection",
    "SEARCH_METHODS",
    "angles_to_coefficients",
    "reconstruct_rows",
    "evaluate_candidates",
    "select_contact_candidate",
    "evaluate_from_angles",
    "top_displacement_vector",
    "evaluation_to_csv_rows",
    "export_evaluation_csv",
    "export_selected_result",
)

SEARCH_LOCAL = "local_interval_search"
SEARCH_EXPANDED = "expanded_interval_search"
SEARCH_GLOBAL = "global_interval_search"
SEARCH_FORCED = "forced_interval"
SEARCH_FALLBACK = "least_violating_fallback"
SEARCH_METHODS = (SEARCH_LOCAL, SEARCH_EXPANDED, SEARCH_GLOBAL, SEARCH_FORCED, SEARCH_FALLBACK)

DISCONNECTED_CONTACT_WARNING = (
    "Interior contact nodes carry tensile normal reactions: the true contact set may be "
    "disconnected (two separated regions). A multi-interval contact model is required."
)
INSUFFICIENT_TOPOLOGY_MESSAGE = (
    "No admissible single-interval contact state exists for this load; showing the "
    "least-violating interval."
)


@dataclass(frozen=True)
class Tolerances:
    """Force conditioning, unilateral admissibility, and violation-score settings.

    Effective unilateral tolerances are mesh- and magnitude-scaled:

        tau_g,eff = tau_g + gap_rel_tol * h,          h = median bottom node spacing
        tau_R,eff = tau_R + reaction_rel_tol * R_ref,  R_ref = max(|F*|, F_floor) h / L

    The violation score uses the normalizing scales ``g_s = h``, ``R_s = R_ref``,
    ``F_s = max(|F*|, F_floor)``.
    """

    tau_g: float = 0.0
    tau_R: float = 0.0
    gap_rel_tol: float = 1.0e-9
    reaction_rel_tol: float = 1.0e-9
    F_abs_tol: float = 1e-8
    F_rel_tol: float = 1e-8
    F_floor: float = 1.0
    fy_zero_tol: float = 1e-14
    g_floor: float = 1e-30
    R_floor: float = 1e-30
    kf_cond_warn: float = 1e8
    w_g: float = 1.0
    w_R: float = 1.0
    w_F: float = 1.0
    flag_penalty: float = 1.0e6

    def __post_init__(self) -> None:
        for name in ("tau_g", "tau_R", "gap_rel_tol", "reaction_rel_tol", "w_g", "w_R", "w_F"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True)
class SelectionConfig:
    """Interval search and temporal-continuity settings.

    ``topology_change_penalty`` multiplies ``|i - i_prev| + |j - j_prev|`` and is
    added to the complementarity score of *admissible* candidates only, so it can
    never keep a penetrating or tensile candidate.
    """

    use_temporal_continuity: bool = True
    topology_change_penalty: float = 0.0
    local_radius: int = 1
    expansion_radii: tuple[int, ...] = (2, 4, 8, 16)
    score_tie_window: float = 1e-9

    def __post_init__(self) -> None:
        if self.topology_change_penalty < 0.0:
            raise ValueError("topology_change_penalty must be non-negative.")
        if self.local_radius < 1:
            raise ValueError("local_radius must be >= 1.")


def angles_to_coefficients(
    phi_deg: float,
    theta_deg: float,
    phi_ref: float = 0.0,
) -> tuple[float, float, float, float]:
    """Convert GUI degrees to ``(varphi, alpha, r_x, r_y)``."""
    phi = float(np.deg2rad(float(phi_deg)))
    theta = float(np.deg2rad(float(theta_deg)))
    varphi = corotation_angle(phi, phi_ref)
    alpha = shape_amplitude(theta)
    r_x, r_y = rotation_coefficients(varphi)
    return varphi, alpha, r_x, r_y


def _require_interval_lookup(lookup: ContactLookupResult) -> None:
    if int(lookup.schema_version) != LOOKUP_SCHEMA_VERSION:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    s = np.asarray(lookup.scalar_lookup)
    if s.ndim != 3 or s.shape[1] != N_AFFINE_COLUMNS:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    if lookup.contact_start_index is None:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Missing contact_start_index.")
    if not lookup.has_nodal_fields:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Missing the per-node bases and the field solver.")


def mesh_spacing(lookup: ContactLookupResult) -> float:
    """Median bottom-node spacing ``h`` (mesh scale of the gap tolerance)."""
    x = np.asarray(lookup.x_bottom, dtype=float)
    if x.size < 2:
        return float(lookup.L)
    return float(np.median(np.diff(x)))


@dataclass(frozen=True)
class Scales:
    h: float
    g_s: float
    R_ref: float
    F_s: float
    tau_g_eff: float
    tau_R_eff: float
    force_threshold: float


def runtime_scales(lookup: ContactLookupResult, Fx_star: float, Fy_star: float, tol: Tolerances) -> Scales:
    h = mesh_spacing(lookup)
    F_mag = max(float(np.hypot(Fx_star, Fy_star)), float(tol.F_floor))
    R_ref = F_mag * h / float(lookup.L)
    kf = lookup.kf_matrix
    kf_max = float(np.nanmax(np.abs(kf))) if kf is not None and np.any(np.isfinite(kf)) else 1.0
    F_scale = max(abs(Fx_star), abs(Fy_star), kf_max, 1.0)
    return Scales(
        h=h,
        g_s=max(h, tol.g_floor),
        R_ref=max(R_ref, tol.R_floor),
        F_s=F_mag,
        tau_g_eff=float(tol.tau_g) + float(tol.gap_rel_tol) * h,
        tau_R_eff=float(tol.tau_R) + float(tol.reaction_rel_tol) * R_ref,
        force_threshold=float(tol.F_abs_tol) + float(tol.F_rel_tol) * F_scale,
    )


EXACT_AUTO = "auto"
EXACT_ALL = "all"
# Relative slack on edge-only rejections so round-off can never reject a record
# that the all-node check would accept.
_EDGE_REJECT_SLACK = 1.0e-9


def reconstruct_rows(
    lookup: ContactLookupResult,
    rows: np.ndarray,
    gamma: np.ndarray,
    varphi: np.ndarray | float,
    F_star_fixed: np.ndarray,
    tol: Tolerances,
    *,
    extra_penalty: np.ndarray | None = None,
    kf_cond: np.ndarray | None = None,
    force_threshold: float | None = None,
    full_fields: bool = True,
    exact: str = EXACT_AUTO,
) -> dict[str, np.ndarray]:
    """Vectorized affine reconstruction and unilateral checks for selected rows.

    ``gamma`` has shape ``(m, 5)`` (one coefficient vector per row) and ``varphi``
    is a scalar or ``(m,)``. The closure column is applied exactly once.
    ``extra_penalty`` adds per-row flag penalties to the violation score (e.g. a
    failed or unstable toe root).

    With stored per-node bases, or ``exact="all"``, every row is checked at every
    bottom node. For a lookup without stored bases, ``exact="auto"`` first applies
    the exact checks that need no nodal fields (force match, heel/toe edge
    reactions, adjacent free gaps); only rows passing them are re-solved and
    checked at every node. The remaining rows are provably inadmissible and carry
    edge-only values (``fields_exact = False``): ``violation_score``,
    ``max_free_penetration`` and the violation terms are lower bounds,
    ``min_free_gap`` / ``min_contact_reaction`` upper bounds of the all-node
    values, and nodal fields are NaN. :func:`refine_reconstruction` makes rows exact.
    """
    if exact not in (EXACT_AUTO, EXACT_ALL):
        raise ValueError(f"exact must be {EXACT_AUTO!r} or {EXACT_ALL!r}, got {exact!r}.")
    rows = np.asarray(rows, dtype=int)
    m = rows.size
    G = np.asarray(gamma, dtype=float).reshape(m, 5)
    G6 = np.concatenate([np.ones((m, 1)), G], axis=1)
    vphi = np.broadcast_to(np.asarray(varphi, dtype=float), (m,)).astype(float)
    c = np.cos(vphi)[:, None]
    s = np.sin(vphi)[:, None]
    Fstar = np.asarray(F_star_fixed, dtype=float).reshape(2)
    sc = runtime_scales(lookup, float(Fstar[0]), float(Fstar[1]), tol)
    threshold = sc.force_threshold if force_threshold is None else float(force_threshold)

    n_b = int(lookup.n_bottom_nodes)
    x_b = np.asarray(lookup.x_bottom, dtype=float)
    y_b = (
        np.asarray(lookup.y_bottom, dtype=float)
        if lookup.y_bottom is not None
        else np.zeros(n_b)
    )
    x_a = np.asarray(lookup.contact_anchor_reference_x, dtype=float)[rows][:, None]
    y_a = np.asarray(lookup.contact_anchor_reference_y, dtype=float)[rows][:, None]
    starts = np.asarray(lookup.contact_start_index, dtype=int)[rows]
    ends = np.asarray(lookup.contact_end_index, dtype=int)[rows]
    contact = contact_mask_from_arrays(starts, ends, n_b)
    free = ~contact
    d_ax = G[:, 1:2]
    d_ay = G[:, 2:3]

    S = np.einsum("rk,rkf->rf", G6, np.asarray(lookup.scalar_lookup)[rows])
    Fx_loc = S[:, SCALAR_FX]
    Fy_loc = S[:, SCALAR_FY]
    Fx_fix = c[:, 0] * Fx_loc - s[:, 0] * Fy_loc
    Fy_fix = s[:, 0] * Fx_loc + c[:, 0] * Fy_loc
    force_err = np.hypot(Fx_fix - Fstar[0], Fy_fix - Fstar[1])
    has_free = np.any(free, axis=1)
    tau_g, tau_R = sc.tau_g_eff, sc.tau_R_eff

    # Edge reactions (local -> fixed -> normal) and adjacent free gaps.
    def edge(name: str) -> np.ndarray:
        arr = np.asarray(lookup.edge_responses[name])[rows]
        return np.einsum("rk,rk->r", G6, arr)

    cc, ss = c[:, 0], s[:, 0]
    h_rx, h_ry = edge("heel_edge_reaction_local_x"), edge("heel_edge_reaction_local_y")
    t_rx, t_ry = edge("toe_edge_reaction_local_x"), edge("toe_edge_reaction_local_y")
    h_fx, h_fy = cc * h_rx - ss * h_ry, ss * h_rx + cc * h_ry
    t_fx, t_fy = cc * t_rx - ss * t_ry, ss * t_rx + cc * t_ry
    heel_adj = np.where(starts > 0, starts - 1, -1)
    toe_adj = np.where(ends < n_b - 1, ends + 1, -1)
    xa, ya = x_a[:, 0], y_a[:, 0]
    with np.errstate(invalid="ignore"):
        hu, hv = edge("heel_adjacent_free_u"), edge("heel_adjacent_free_v")
        tu, tv = edge("toe_adjacent_free_u"), edge("toe_adjacent_free_v")
        hx = np.where(heel_adj >= 0, x_b[np.maximum(heel_adj, 0)], np.nan)
        hy = np.where(heel_adj >= 0, y_b[np.maximum(heel_adj, 0)], np.nan)
        tx = np.where(toe_adj >= 0, x_b[np.maximum(toe_adj, 0)], np.nan)
        ty = np.where(toe_adj >= 0, y_b[np.maximum(toe_adj, 0)], np.nan)
        heel_gap = ss * (hx - xa + hu - d_ax[:, 0]) + cc * (hy - ya + hv - d_ay[:, 0])
        toe_gap = ss * (tx - xa + tu - d_ax[:, 0]) + cc * (ty - ya + tv - d_ay[:, 0])
    heel_gap = np.where(heel_adj >= 0, heel_gap, np.nan)
    toe_gap = np.where(toe_adj >= 0, toe_gap, np.nan)
    h_Rn = h_fy
    t_Rn = t_fy
    comp_h = np.abs(h_Rn * heel_gap) / (sc.R_ref * sc.g_s)
    comp_t = np.abs(t_Rn * toe_gap) / (sc.R_ref * sc.g_s)
    comp_score = np.nan_to_num(comp_h, nan=0.0) + np.nan_to_num(comp_t, nan=0.0)

    F_term = float(tol.w_F) * (force_err / sc.F_s) ** 2
    finite_S = np.isfinite(force_err) & np.isfinite(S).all(axis=1)
    force_ok = force_err <= threshold
    kfc = np.full(m, np.nan) if kf_cond is None else np.asarray(kf_cond, dtype=float)
    ill = np.isfinite(kfc) & (kfc >= tol.kf_cond_warn)
    extra = np.zeros(m) if extra_penalty is None else np.asarray(extra_penalty, dtype=float)
    ar = np.arange(m)

    # Edge-only values (bounds of the all-node values) for every row.
    adj_gap = np.column_stack([heel_gap, toe_gap])
    adj_g = np.where(np.isnan(adj_gap), np.inf, adj_gap)
    k_adj = np.argmin(adj_g, axis=1)
    min_gap = np.where(has_free, adj_g[ar, k_adj], np.inf)
    min_gap_node = np.where(has_free, np.column_stack([heel_adj, toe_adj])[ar, k_adj], -1)
    edge_R = np.where(np.isnan(np.column_stack([h_Rn, t_Rn])), np.inf, np.column_stack([h_Rn, t_Rn]))
    k_R = np.argmin(edge_R, axis=1)
    min_R = edge_R[ar, k_R]
    min_R_node = np.where(k_R == 0, starts, ends)
    contact_ground_residual = np.zeros(m)
    interior_tension = np.zeros(m, dtype=bool)
    lb = 1.0 - _EDGE_REJECT_SLACK
    gap_terms = np.maximum(0.0, (-np.where(np.isnan(adj_gap), np.inf, adj_gap) - tau_g) / sc.g_s)
    gap_term = lb * float(tol.w_g) * np.sum(gap_terms**2, axis=1)
    R_terms = np.maximum(0.0, (-edge_R - tau_R) / sc.R_ref)
    R_terms[:, 1] = np.where(starts == ends, 0.0, R_terms[:, 1])
    R_term = lb * float(tol.w_R) * np.sum(R_terms**2, axis=1)
    finite = finite_S.copy()
    admissible = np.zeros(m, dtype=bool)
    fields_exact = np.zeros(m, dtype=bool)

    if exact == EXACT_ALL or lookup.nodal_fields_stored:
        ex = ar
    else:
        def _edge_ok(v: np.ndarray, tau: float) -> np.ndarray:
            return ~(v < -tau - _EDGE_REJECT_SLACK * (np.abs(v) + tau))

        ex = np.flatnonzero(
            finite_S & force_ok
            & _edge_ok(heel_gap, tau_g) & _edge_ok(toe_gap, tau_g)
            & _edge_ok(h_Rn, tau_R) & _edge_ok(t_Rn, tau_R)
        )

    if full_fields:
        n_t = int(lookup.n_top_nodes)
        nodal = {
            name: np.full((m, n_t if name.startswith("top_force") else n_b), np.nan)
            for name in (
                "full_bottom_gap", "full_bottom_u", "full_bottom_v", "full_bottom_reaction_x",
                "full_bottom_reaction_y", "full_bottom_reaction_normal",
                "full_bottom_reaction_tangential", "top_force_x", "top_force_y",
            )
        }

    if ex.size:
        fields = lookup.record_fields(rows[ex])
        Ge = G6[ex]

        def contract(name: str) -> np.ndarray:
            return np.einsum("rk,rkn->rn", Ge, fields[name])

        ce, se = c[ex], s[ex]
        cont, fr, hf = contact[ex], free[ex], has_free[ex]
        are = np.arange(ex.size)
        u_b = contract("bottom_u")
        v_b = contract("bottom_v")
        dx = (x_b[None, :] - x_a[ex]) + (u_b - d_ax[ex])
        dy = (y_b[None, :] - y_a[ex]) + (v_b - d_ay[ex])
        gap = se * dx + ce * dy
        rx = contract("reaction_x")
        ry = contract("reaction_y")
        Rn = se * rx + ce * ry
        Rt = ce * rx - se * ry

        g_free = np.where(fr, gap, np.inf)
        mg_node = np.argmin(g_free, axis=1)
        mg = g_free[are, mg_node]
        mg_node = np.where(hf, mg_node, -1)
        R_contact = np.where(cont, Rn, np.inf)
        mR_node = np.argmin(R_contact, axis=1)
        mR = R_contact[are, mR_node]
        # Interior (non-edge) tension: candidate of a disconnected contact set.
        interior = cont.copy()
        interior[are, starts[ex]] = False
        interior[are, ends[ex]] = False

        gt = np.where(fr, np.maximum(0.0, (-gap - tau_g) / sc.g_s), 0.0)
        g_term = float(tol.w_g) * np.sum(gt**2, axis=1)
        rt = np.where(cont, np.maximum(0.0, (-Rn - tau_R) / sc.R_ref), 0.0)
        fin = finite_S[ex] & np.all(np.isfinite(gap), axis=1) & np.all(np.isfinite(Rn), axis=1)

        min_gap[ex] = mg
        min_gap_node[ex] = mg_node
        min_R[ex] = mR
        min_R_node[ex] = mR_node
        contact_ground_residual[ex] = np.max(np.where(cont, np.abs(gap), 0.0), axis=1)
        interior_tension[ex] = np.any(interior & (Rn < -tau_R), axis=1)
        gap_term[ex] = g_term
        R_term[ex] = float(tol.w_R) * np.sum(rt**2, axis=1)
        finite[ex] = fin
        admissible[ex] = fin & force_ok[ex] & (~hf | (mg >= -tau_g)) & (mR >= -tau_R)
        fields_exact[ex] = True

        if full_fields:
            for name, value in (
                ("full_bottom_gap", gap), ("full_bottom_u", u_b), ("full_bottom_v", v_b),
                ("full_bottom_reaction_x", rx), ("full_bottom_reaction_y", ry),
                ("full_bottom_reaction_normal", np.where(cont, Rn, 0.0)),
                ("full_bottom_reaction_tangential", np.where(cont, Rt, 0.0)),
                ("top_force_x", contract("top_force_x")), ("top_force_y", contract("top_force_y")),
            ):
                nodal[name][ex] = value

    min_gap = np.where(has_free, min_gap, np.inf)
    max_pen = np.where(has_free, np.maximum(0.0, -min_gap), 0.0)
    max_tension = np.maximum(0.0, -min_R)
    penalty = float(tol.flag_penalty) * (
        (~finite).astype(float) + (~force_ok).astype(float) + ill.astype(float)
    ) + extra
    V = np.where(finite, gap_term + R_term + F_term, np.inf) + penalty

    out = {
        "rows": rows,
        "d_ax": d_ax[:, 0].copy(),
        "d_ay": d_ay[:, 0].copy(),
        "Fx_local": Fx_loc,
        "Fy_local": Fy_loc,
        "Fx": Fx_fix,
        "Fy": Fy_fix,
        "M": S[:, SCALAR_MV],
        "T_toe": S[:, SCALAR_TOE],
        "force_residual": force_err,
        "min_free_gap": np.where(has_free, min_gap, np.inf),
        "min_free_gap_node": min_gap_node,
        "max_free_penetration": max_pen,
        "min_contact_reaction": min_R,
        "min_contact_reaction_node": min_R_node,
        "max_contact_tension": max_tension,
        "contact_ground_residual": contact_ground_residual,
        "disconnected_contact_warning": interior_tension,
        "heel_edge_reaction_local_x": h_rx,
        "heel_edge_reaction_local_y": h_ry,
        "toe_edge_reaction_local_x": t_rx,
        "toe_edge_reaction_local_y": t_ry,
        "heel_edge_reaction_fixed_x": h_fx,
        "heel_edge_reaction_fixed_y": h_fy,
        "toe_edge_reaction_fixed_x": t_fx,
        "toe_edge_reaction_fixed_y": t_fy,
        "heel_edge_normal_reaction": h_Rn,
        "toe_edge_normal_reaction": t_Rn,
        "heel_adjacent_free_gap": heel_gap,
        "toe_adjacent_free_gap": toe_gap,
        "complementarity_heel": comp_h,
        "complementarity_toe": comp_t,
        "complementarity_score": comp_score,
        "gap_violation_term": gap_term,
        "reaction_violation_term": R_term,
        "force_violation_term": F_term,
        "penalty_term": penalty,
        "violation_score": V,
        "admissible": admissible,
        "finite": finite,
        "force_ok": force_ok,
        "kf_illconditioned": ill,
        "fields_exact": fields_exact,
        "gamma": G,
        "varphi": vphi,
        "extra_penalty": extra,
        "kf_cond": kfc,
        "force_threshold": float(threshold),
        "F_star_fixed": (float(Fstar[0]), float(Fstar[1])),
        "tolerances": tol,
    }
    if full_fields:
        W = np.asarray(lookup.basis_top_displacements, dtype=float)
        U_top = G6 @ W.T
        top_u = U_top[:, :n_t]
        top_v = U_top[:, n_t:]
        ftx = nodal["top_force_x"]
        fty = nodal["top_force_y"]
        x_t = np.asarray(lookup.x_top, dtype=float)
        y_t = (
            np.asarray(lookup.y_top, dtype=float)
            if lookup.y_top is not None
            else np.full(n_t, float(lookup.H))
        )
        px = (x_t[None, :] - x_a) + (top_u - d_ax)
        py = (y_t[None, :] - y_a) + (top_v - d_ay)
        x_top_F = x_a + c * px - s * py
        fty_F = s * ftx + c * fty
        fy_total = np.sum(fty_F, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cm = np.sum(x_top_F * fty_F, axis=1) / fy_total
        ok_cm = (abs(Fstar[1]) > tol.fy_zero_tol) & (np.abs(fy_total) > threshold)
        x_cm = np.where(ok_cm, x_cm, np.nan)
        out.update(nodal)
        out.update(top_u=top_u, top_v=top_v, x_cm=x_cm, x_cm_rel=x_cm - xa)
    out["scales"] = sc
    return out


def refine_reconstruction(lookup: ContactLookupResult, rec: dict, idx) -> np.ndarray:
    """Recompute positions ``idx`` of a :func:`reconstruct_rows` result with exact
    all-node checks, in place. Returns the positions that were refined."""
    idx = np.atleast_1d(np.asarray(idx, dtype=int))
    idx = idx[~np.asarray(rec["fields_exact"])[idx]]
    if idx.size == 0:
        return idx
    sub = reconstruct_rows(
        lookup,
        np.asarray(rec["rows"])[idx],
        np.asarray(rec["gamma"])[idx],
        np.asarray(rec["varphi"])[idx],
        np.asarray(rec["F_star_fixed"], dtype=float),
        rec["tolerances"],
        extra_penalty=np.asarray(rec["extra_penalty"])[idx],
        kf_cond=np.asarray(rec["kf_cond"])[idx],
        force_threshold=rec["force_threshold"],
        full_fields="top_u" in rec,
        exact=EXACT_ALL,
    )
    for key, value in sub.items():
        target = rec.get(key)
        if isinstance(value, np.ndarray) and isinstance(target, np.ndarray) and value.ndim and value.shape[0] == idx.size:
            target[idx] = value
    return idx


def refine_top_k(pool, sort_keys: Callable, is_exact: Callable, refine: Callable, k: int = 1) -> np.ndarray:
    """Order ``pool`` by ``np.lexsort(sort_keys(pool))`` with the first ``k`` exact.

    ``sort_keys`` returns lexsort keys (primary last) whose values for inexact
    entries are lower bounds of their exact values; ``refine(entries)`` makes
    entries exact. Once the current first ``k`` are all exact no other entry can
    precede them, so the returned order agrees with an all-exact ordering there.
    """
    pool = np.asarray(pool, dtype=int)
    k = max(1, int(k))
    while True:
        order = pool[np.lexsort(sort_keys(pool))] if pool.size else pool
        top = order[:k]
        pending = top[~np.asarray(is_exact(top), dtype=bool)]
        if pending.size == 0:
            return order
        refine(pending)


def solve_force_control(
    lookup: ContactLookupResult,
    rows: np.ndarray,
    alpha: float,
    varphi: float,
    F_local_star: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Batched ``K_F [d_ax, d_ay] = F* - F_0 - alpha F_alpha - r_x F_rx - r_y F_ry``.

    Singular or numerically singular ``K_F`` (smallest singular value below the
    force threshold) is reported, never regularized.
    """
    rows = np.asarray(rows, dtype=int)
    S = np.asarray(lookup.scalar_lookup, dtype=float)[rows]
    r_x, r_y = rotation_coefficients(varphi)
    K = np.stack(
        [
            np.stack([S[:, COL_BX, SCALAR_FX], S[:, COL_BY, SCALAR_FX]], axis=-1),
            np.stack([S[:, COL_BX, SCALAR_FY], S[:, COL_BY, SCALAR_FY]], axis=-1),
        ],
        axis=1,
    )
    known = (
        S[:, COL_CONST, :2]
        + float(alpha) * S[:, COL_ALPHA, :2]
        + r_x * S[:, COL_BRX, :2]
        + r_y * S[:, COL_BRY, :2]
    )
    rhs = np.asarray(F_local_star, dtype=float)[None, :] - known
    finite = np.all(np.isfinite(K), axis=(1, 2)) & np.all(np.isfinite(rhs), axis=1)
    svals = np.full((rows.size, 2), np.nan)
    if np.any(finite):
        svals[finite] = np.linalg.svd(K[finite], compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(svals[:, 1] > 0.0, svals[:, 0] / svals[:, 1], np.inf)
    det = np.where(finite, np.linalg.det(np.where(finite[:, None, None], K, 0.0)), np.nan)
    well = finite & (svals[:, 1] > threshold) & np.isfinite(cond)
    d = np.full((rows.size, 2), np.nan)
    if np.any(well):
        d[well] = np.linalg.solve(K[well], rhs[well][:, :, None])[:, :, 0]
    return {"d": d, "cond": cond, "det": det, "svals": svals, "well": well}


@dataclass
class CandidateEvaluation:
    """Force-controlled reconstruction for every interval record.

    Per-record arrays have length ``n_records`` (2-D fields ``(n_records, n_nodes)``);
    rows that were not evaluated (rejected intervals or outside the evaluated
    subset) hold NaN and ``evaluated = False``.
    """

    contact_start_index: np.ndarray
    contact_end_index: np.ndarray
    contact_type_codes: np.ndarray
    contact_start_x: np.ndarray
    contact_end_x: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    record_valid: np.ndarray
    evaluated: np.ndarray
    phi: float
    theta: float
    phi_ref: float
    varphi: float
    alpha: float
    r_x: float
    r_y: float
    Fx_star: float
    Fy_star: float
    Fx_star_local: float
    Fy_star_local: float
    fields: dict[str, np.ndarray]
    well_conditioned: np.ndarray
    kf_cond: np.ndarray
    kf_det: np.ndarray
    kf_svals: np.ndarray
    scales: Scales
    n_bottom: int
    diagnostics: dict = field(default_factory=dict)
    refiner: Callable[[np.ndarray], None] | None = field(default=None, repr=False, compare=False)

    def ensure_exact(self, rows) -> None:
        """Make the all-node fields of evaluated ``rows`` exact (compact lookups)."""
        rows = np.atleast_1d(np.asarray(rows, dtype=int))
        exact = self.__dict__["fields"]["fields_exact"]
        need = rows[self.evaluated[rows] & ~exact[rows]]
        if need.size and self.refiner is not None:
            self.refiner(need)

    def __getattr__(self, name: str):
        fields = self.__dict__.get("fields")
        if fields is not None and name in fields:
            return fields[name]
        raise AttributeError(name)

    @property
    def n_records(self) -> int:
        return int(self.contact_start_index.size)

    @property
    def x_anchor_rot(self) -> np.ndarray:
        return self.anchor_x + self.fields["d_ax"]

    def contact_type(self, row: int) -> ContactType:
        return contact_type_from_code(int(self.contact_type_codes[int(row)]))

    def rows_for(self, contact_type: ContactType | str) -> np.ndarray:
        from compliance_fem.contact.lookup import contact_type_code

        code = contact_type_code(contact_type)
        return np.flatnonzero(np.asarray(self.contact_type_codes, dtype=int) == code)

    def row_of(self, start: int, end: int) -> int:
        from compliance_fem.contact.topology import interval_row

        return interval_row(start, end, self.n_bottom)

    @property
    def full_contact_row(self) -> int:
        return self.row_of(0, self.n_bottom - 1)


_ROW_FIELDS_1D = (
    "d_ax", "d_ay", "Fx_local", "Fy_local", "Fx", "Fy", "M", "T_toe", "force_residual",
    "min_free_gap", "min_free_gap_node", "max_free_penetration",
    "min_contact_reaction", "min_contact_reaction_node", "max_contact_tension",
    "contact_ground_residual", "disconnected_contact_warning",
    "heel_edge_reaction_local_x", "heel_edge_reaction_local_y",
    "toe_edge_reaction_local_x", "toe_edge_reaction_local_y",
    "heel_edge_reaction_fixed_x", "heel_edge_reaction_fixed_y",
    "toe_edge_reaction_fixed_x", "toe_edge_reaction_fixed_y",
    "heel_edge_normal_reaction", "toe_edge_normal_reaction",
    "heel_adjacent_free_gap", "toe_adjacent_free_gap",
    "complementarity_heel", "complementarity_toe", "complementarity_score",
    "gap_violation_term", "reaction_violation_term", "force_violation_term", "penalty_term",
    "violation_score", "admissible", "finite", "force_ok", "kf_illconditioned", "x_cm", "x_cm_rel",
    "fields_exact",
)
_ROW_FIELDS_BOTTOM = (
    "full_bottom_gap", "full_bottom_u", "full_bottom_v", "full_bottom_reaction_x",
    "full_bottom_reaction_y", "full_bottom_reaction_normal", "full_bottom_reaction_tangential",
)
_ROW_FIELDS_TOP = ("top_u", "top_v", "top_force_x", "top_force_y")
_BOOL_FIELDS = (
    "disconnected_contact_warning", "admissible", "finite", "force_ok", "kf_illconditioned", "fields_exact",
)
_INT_FIELDS = ("min_free_gap_node", "min_contact_reaction_node")


def evaluate_candidates(
    lookup: ContactLookupResult,
    Fx: float,
    Fy: float,
    phi_deg: float,
    theta_deg: float,
    tolerances: Tolerances | None = None,
    rows: np.ndarray | None = None,
) -> CandidateEvaluation:
    """Reconstruct interval records for prescribed fixed-frame ``Fx, Fy`` and angles.

    By default every valid record is evaluated (one vectorized pass); ``rows``
    restricts the pass to a subset. Selection is a separate step.
    """
    _require_interval_lookup(lookup)
    tol = tolerances or Tolerances()
    n = lookup.n_records
    n_b = int(lookup.n_bottom_nodes)
    n_t = int(lookup.n_top_nodes)
    phi = float(np.deg2rad(float(phi_deg)))
    theta = float(np.deg2rad(float(theta_deg)))
    varphi, alpha, r_x, r_y = angles_to_coefficients(phi_deg, theta_deg, lookup.phi_ref)
    F_local = rotate_force_to_local(float(Fx), float(Fy), varphi)
    sc = runtime_scales(lookup, float(Fx), float(Fy), tol)

    valid = lookup.valid_mask
    target = np.flatnonzero(valid) if rows is None else np.asarray(rows, dtype=int)
    target = target[valid[target]]
    fc = solve_force_control(lookup, target, alpha, varphi, F_local, sc.force_threshold)
    good = target[fc["well"]]
    d = fc["d"][fc["well"]]
    gamma = np.column_stack(
        [np.full(good.size, alpha), d[:, 0], d[:, 1], np.full(good.size, r_x), np.full(good.size, r_y)]
    )
    rec = reconstruct_rows(
        lookup,
        good,
        gamma,
        varphi,
        np.array([float(Fx), float(Fy)]),
        tol,
        kf_cond=fc["cond"][fc["well"]],
        force_threshold=sc.force_threshold,
    )

    fields: dict[str, np.ndarray] = {}
    for name in _ROW_FIELDS_1D:
        if name in _BOOL_FIELDS:
            arr = np.zeros(n, dtype=bool)
        elif name in _INT_FIELDS:
            arr = np.full(n, -1, dtype=int)
        else:
            arr = np.full(n, np.nan)
        arr[good] = rec[name]
        fields[name] = arr
    for name in _ROW_FIELDS_BOTTOM:
        arr = np.full((n, n_b), np.nan)
        arr[good] = rec[name]
        fields[name] = arr
    for name in _ROW_FIELDS_TOP:
        arr = np.full((n, n_t), np.nan)
        arr[good] = rec[name]
        fields[name] = arr
    fields["violation_score"][~np.isin(np.arange(n), good)] = np.inf
    kill = np.zeros(n, dtype=bool)
    kill[target] = fc["well"]
    kf_cond = np.full(n, np.nan)
    kf_det = np.full(n, np.nan)
    kf_sv = np.full((n, 2), np.nan)
    kf_cond[target] = fc["cond"]
    kf_det[target] = fc["det"]
    kf_sv[target] = fc["svals"]
    fields["kf_illconditioned"][target] |= ~fc["well"]
    evaluated = np.zeros(n, dtype=bool)
    evaluated[good] = True
    position = np.full(n, -1, dtype=int)
    position[good] = np.arange(good.size)

    def refiner(record_rows: np.ndarray) -> None:
        pos = refine_reconstruction(lookup, rec, position[np.asarray(record_rows, dtype=int)])
        sel = good[pos]
        for name in (*_ROW_FIELDS_1D, *_ROW_FIELDS_BOTTOM, *_ROW_FIELDS_TOP):
            fields[name][sel] = rec[name][pos]

    return CandidateEvaluation(
        contact_start_index=np.asarray(lookup.contact_start_index, dtype=int),
        contact_end_index=np.asarray(lookup.contact_end_index, dtype=int),
        contact_type_codes=np.asarray(lookup.contact_type_codes, dtype=int),
        contact_start_x=np.asarray(lookup.contact_start_x, dtype=float),
        contact_end_x=np.asarray(lookup.contact_end_x, dtype=float),
        anchor_x=np.asarray(lookup.contact_anchor_reference_x, dtype=float),
        anchor_y=np.asarray(lookup.contact_anchor_reference_y, dtype=float),
        record_valid=valid.copy(),
        evaluated=evaluated,
        phi=phi,
        theta=theta,
        phi_ref=float(lookup.phi_ref),
        varphi=varphi,
        alpha=alpha,
        r_x=r_x,
        r_y=r_y,
        Fx_star=float(Fx),
        Fy_star=float(Fy),
        Fx_star_local=float(F_local[0]),
        Fy_star_local=float(F_local[1]),
        fields=fields,
        well_conditioned=kill,
        kf_cond=kf_cond,
        kf_det=kf_det,
        kf_svals=kf_sv,
        scales=sc,
        n_bottom=n_b,
        diagnostics={
            "smin_threshold": sc.force_threshold,
            "n_evaluated": int(good.size),
            "n_singular_kf": int(np.count_nonzero(~fc["well"])),
            "basis_order": list(BASIS_ORDER),
            "gauge": "r_a^F = (x_a, 0) with X_a = (x_a, y_a) the interval-midpoint anchor",
            "tau_g_eff": sc.tau_g_eff,
            "tau_R_eff": sc.tau_R_eff,
            "V_definition": (
                "w_g sum_free max(0,(-g-tau_g)/g_s)^2 + w_R sum_contact max(0,(-R_n-tau_R)/R_s)^2 "
                "+ w_F (|dF|/F_s)^2 + flag penalties"
            ),
            "nodal_fields_stored": bool(lookup.nodal_fields_stored),
        },
        refiner=refiner,
    )


def _admissible_choice(
    rows: np.ndarray,
    evaluation: CandidateEvaluation,
    previous: tuple[int, int] | None,
    cfg: SelectionConfig,
) -> int:
    """Pick among admissible rows: keep the previous interval if it is admissible,
    else minimal complementarity score (+ optional topology-change penalty),
    preferring well-conditioned K_F, ties broken by interval distance then row."""
    starts = evaluation.contact_start_index[rows]
    ends = evaluation.contact_end_index[rows]
    use_prev = cfg.use_temporal_continuity and previous is not None
    dist = interval_distance(starts, ends, previous if use_prev else None).astype(float)
    if use_prev:
        same = np.flatnonzero(dist == 0)
        if same.size:
            return int(rows[same[0]])
    score = np.nan_to_num(evaluation.complementarity_score[rows], nan=np.inf)
    score = score + cfg.topology_change_penalty * dist
    ill = evaluation.kf_illconditioned[rows].astype(int)
    best_ill = int(np.min(ill))
    pool = np.flatnonzero(ill == best_ill)
    best = float(np.min(score[pool]))
    window = cfg.score_tie_window * max(1.0, abs(best))
    tied = pool[score[pool] <= best + window]
    key = np.lexsort((rows[tied], dist[tied]))
    return int(rows[tied[key[0]]])


def _fallback_choice(
    rows: np.ndarray,
    evaluation: CandidateEvaluation,
    previous: tuple[int, int] | None,
    cfg: SelectionConfig,
) -> int:
    exact = evaluation.fields["fields_exact"]
    if not np.all(exact[rows]):
        V_all = evaluation.violation_score
        order = refine_top_k(rows, lambda p: (p, V_all[p]), lambda p: exact[p], evaluation.ensure_exact)
        best = float(V_all[order[0]])
        if np.isfinite(best):
            limit = best + cfg.score_tie_window * max(1.0, abs(best))
            while True:
                pending = rows[~exact[rows] & (V_all[rows] <= limit)]
                if pending.size == 0:
                    break
                evaluation.ensure_exact(pending)
    V = evaluation.violation_score[rows]
    starts = evaluation.contact_start_index[rows]
    ends = evaluation.contact_end_index[rows]
    use_prev = cfg.use_temporal_continuity and previous is not None
    dist = interval_distance(starts, ends, previous if use_prev else None)
    finite = np.isfinite(V)
    if not np.any(finite):
        return int(rows[0])
    best = float(np.min(V[finite]))
    window = cfg.score_tie_window * max(1.0, abs(best))
    tied = np.flatnonzero(finite & (V <= best + window))
    key = np.lexsort((rows[tied], dist[tied]))
    return int(rows[tied[key[0]]])


def staged_interval_search(
    evaluation: CandidateEvaluation,
    allowed: np.ndarray,
    previous: tuple[int, int] | None,
    cfg: SelectionConfig,
) -> tuple[int | None, str, np.ndarray]:
    """Local -> expanded -> global search over admissible rows, then the least-violating fallback.

    Returns ``(row, search_method, admissible_rows_in_allowed)``.
    """
    adm = allowed & evaluation.admissible
    adm_rows = np.flatnonzero(adm)
    if previous is not None and cfg.use_temporal_continuity:
        dist = interval_distance(evaluation.contact_start_index, evaluation.contact_end_index, previous)
        stages = [(SEARCH_LOCAL, int(cfg.local_radius))]
        stages += [(SEARCH_EXPANDED, int(r)) for r in cfg.expansion_radii if r > cfg.local_radius]
        for method, radius in stages:
            cand = np.flatnonzero(adm & (dist <= radius))
            if cand.size:
                return _admissible_choice(cand, evaluation, previous, cfg), method, adm_rows
    if adm_rows.size:
        return _admissible_choice(adm_rows, evaluation, previous, cfg), SEARCH_GLOBAL, adm_rows
    pool = np.flatnonzero(allowed & evaluation.evaluated)
    if pool.size == 0:
        return None, SEARCH_FALLBACK, adm_rows
    return _fallback_choice(pool, evaluation, previous, cfg), SEARCH_FALLBACK, adm_rows


class ContactSelection:
    """Selected interval plus the full record evaluation.

    Per-record evaluation fields are available as attributes for the selected
    row (``selection.min_free_gap``, ``selection.heel_adjacent_free_gap``, ...);
    they are ``None`` when nothing was selected.
    """

    def __init__(
        self,
        evaluation: CandidateEvaluation,
        contact_mode: ContactMode,
        selected_row: int | None,
        exactly_admissible: bool,
        approximate: bool,
        admissible_rows: np.ndarray,
        message: str,
        candidate_search_method: str,
        plate: PlateRuntimeState | None = None,
    ) -> None:
        self.evaluation = evaluation
        self.contact_mode = contact_mode
        self.selected_row = None if selected_row is None else int(selected_row)
        self.exactly_admissible = bool(exactly_admissible)
        self.approximate = bool(approximate)
        self.admissible_rows = np.asarray(admissible_rows, dtype=int)
        self.n_admissible = int(self.admissible_rows.size)
        self.message = message
        self.candidate_search_method = candidate_search_method
        self.plate = plate

    def __getattr__(self, name: str):
        ev = self.__dict__.get("evaluation")
        if ev is None:
            raise AttributeError(name)
        if name in ev.fields:
            row = self.__dict__.get("selected_row")
            if row is None:
                return None
            value = ev.fields[name][row]
            if np.ndim(value) == 0:
                if name in _BOOL_FIELDS:
                    return bool(value)
                if name in _INT_FIELDS:
                    return int(value)
                return float(value)
            return np.array(value, copy=True)
        if name in ("phi", "theta", "varphi", "alpha", "Fx_star", "Fy_star", "phi_ref"):
            return getattr(ev, name)
        raise AttributeError(name)

    def _row_value(self, arr) -> float | int | None:
        if self.selected_row is None:
            return None
        return arr[self.selected_row]

    @property
    def interval(self) -> tuple[int, int] | None:
        if self.selected_row is None:
            return None
        return int(self.contact_start_index), int(self.contact_end_index)

    @property
    def contact_start_index(self) -> int | None:
        v = self._row_value(self.evaluation.contact_start_index)
        return None if v is None else int(v)

    @property
    def contact_end_index(self) -> int | None:
        v = self._row_value(self.evaluation.contact_end_index)
        return None if v is None else int(v)

    @property
    def contact_start_x(self) -> float | None:
        v = self._row_value(self.evaluation.contact_start_x)
        return None if v is None else float(v)

    @property
    def contact_end_x(self) -> float | None:
        v = self._row_value(self.evaluation.contact_end_x)
        return None if v is None else float(v)

    @property
    def anchor_x(self) -> float | None:
        v = self._row_value(self.evaluation.anchor_x)
        return None if v is None else float(v)

    @property
    def anchor_y(self) -> float | None:
        v = self._row_value(self.evaluation.anchor_y)
        return None if v is None else float(v)

    @property
    def contact_type(self) -> ContactType | None:
        if self.selected_row is None:
            return None
        return self.evaluation.contact_type(self.selected_row)

    @property
    def topology_label(self) -> str | None:
        kind = self.contact_type
        return None if kind is None else kind.value

    @property
    def contact_node_ids(self) -> np.ndarray | None:
        if self.selected_row is None:
            return None
        return np.arange(self.contact_start_index, self.contact_end_index + 1)

    @property
    def free_node_ids(self) -> np.ndarray | None:
        if self.selected_row is None:
            return None
        nodes = np.arange(self.evaluation.n_bottom)
        mask = (nodes < self.contact_start_index) | (nodes > self.contact_end_index)
        return nodes[mask]

    @property
    def x_anchor_rot(self) -> float | None:
        if self.selected_row is None:
            return None
        return float(self.anchor_x + self.d_ax)

    @property
    def kf_cond(self) -> float | None:
        v = self._row_value(self.evaluation.kf_cond)
        return None if v is None else float(v)

    def contact_span(self, L: float | None = None) -> tuple[float, float] | None:
        """Reference material contact interval ``[x_i, x_j]`` of the selected record."""
        del L
        if self.selected_row is None:
            return None
        return float(self.contact_start_x), float(self.contact_end_x)


def select_contact_candidate(
    evaluation: CandidateEvaluation,
    tolerances: Tolerances | None = None,
    mode: ContactMode | str = ContactMode.AUTO,
    previous_index: int | None = None,
    *,
    previous_interval: tuple[int, int] | None = None,
    specific_interval: tuple[int, int] | None = None,
    selection_config: SelectionConfig | None = None,
) -> ContactSelection:
    """Select an interval from fixed-frame gaps / normal reactions only.

    ``AUTO`` searches every interval; ``HEEL`` / ``TOE`` / ``INTERIOR``
    restrict the search to intervals attached to the heel, attached to the toe,
    or free at both ends; ``FULL`` and ``SPECIFIC`` force one interval
    (``specific_interval=(i, j)`` is validated). ``previous_index`` (a row) or
    ``previous_interval`` enables local-first search and continuity tie-breaking.
    """
    del tolerances
    mode = parse_contact_mode(mode)
    cfg = selection_config or SelectionConfig()
    n_b = evaluation.n_bottom
    previous = previous_interval
    if previous is None and previous_index is not None:
        previous = (
            int(evaluation.contact_start_index[int(previous_index)]),
            int(evaluation.contact_end_index[int(previous_index)]),
        )
    if previous is not None:
        validate_interval(previous[0], previous[1], n_b)

    if mode in (ContactMode.FULL, ContactMode.SPECIFIC):
        if mode is ContactMode.FULL:
            target = (0, n_b - 1)
        else:
            if specific_interval is None:
                raise ValueError("Specific interval mode requires specific_interval=(start, end).")
            target = (int(specific_interval[0]), int(specific_interval[1]))
            validate_interval(target[0], target[1], n_b)
        row = evaluation.row_of(*target)
        adm_rows = np.flatnonzero(evaluation.admissible)
        if not evaluation.evaluated[row]:
            reason = (
                "rejected during lookup generation"
                if not evaluation.record_valid[row]
                else "K_F is singular for this interval"
            )
            return ContactSelection(
                evaluation, mode, None, False, True, adm_rows,
                f"Interval {target} cannot be evaluated: {reason}.", SEARCH_FORCED,
            )
        evaluation.ensure_exact(row)
        ok = bool(evaluation.admissible[row])
        msg = f"Interval {target} forced; " + (
            "it is admissible." if ok else "it violates the unilateral conditions (approximate)."
        )
        if bool(evaluation.disconnected_contact_warning[row]):
            msg += " " + DISCONNECTED_CONTACT_WARNING
        return ContactSelection(evaluation, mode, row, ok, not ok, adm_rows, msg, SEARCH_FORCED)

    allowed = mode_mask(evaluation.contact_start_index, evaluation.contact_end_index, n_b, mode)
    allowed &= evaluation.evaluated
    row, method, adm_rows = staged_interval_search(evaluation, allowed, previous, cfg)
    if row is None:
        return ContactSelection(
            evaluation, mode, None, False, True, adm_rows,
            "No evaluable interval: K_F is singular for every allowed record.", method,
        )
    evaluation.ensure_exact(row)
    ok = method != SEARCH_FALLBACK
    if ok:
        msg = f"{adm_rows.size} admissible interval(s); selected by {method}."
    else:
        msg = INSUFFICIENT_TOPOLOGY_MESSAGE
        if np.any(allowed & evaluation.disconnected_contact_warning):
            msg += " " + DISCONNECTED_CONTACT_WARNING
    if bool(evaluation.kf_illconditioned[row]):
        msg += " Warning: K_F is ill-conditioned."
    return ContactSelection(evaluation, mode, row, ok, not ok, adm_rows, msg, method)


def evaluate_from_angles(
    lookup: ContactLookupResult,
    Fx: float,
    Fy: float,
    phi_deg: float,
    theta_deg: float,
    tolerances: Tolerances | None = None,
    mode: ContactMode | str = ContactMode.AUTO,
    previous_index: int | None = None,
    *,
    previous_interval: tuple[int, int] | None = None,
    specific_interval: tuple[int, int] | None = None,
    selection_config: SelectionConfig | None = None,
) -> ContactSelection:
    """Fixed-frame (Fx, Fy, phi, theta) -> interval evaluation -> selected interval."""
    evaluation = evaluate_candidates(lookup, Fx, Fy, phi_deg, theta_deg, tolerances=tolerances)
    return select_contact_candidate(
        evaluation,
        tolerances=tolerances,
        mode=mode,
        previous_index=previous_index,
        previous_interval=previous_interval,
        specific_interval=specific_interval,
        selection_config=selection_config,
    )


def top_displacement_vector(
    x: np.ndarray,
    alpha: float,
    L: float,
    toe_length: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotating-frame top displacement: u = 0, v = alpha * phi1(x)."""
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x, dtype=float), float(alpha) * shape_mode_phi1(x, L=L, toe_length=toe_length, kappa=kappa)


def _f(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


NOT_AVAILABLE = "N/A"


def format_optional(value, fmt: str = ".6g") -> str:
    """Format a diagnostic, showing ``N/A`` for missing or non-finite values
    (e.g. the outside gap beyond a domain endpoint)."""
    if value is None:
        return NOT_AVAILABLE
    v = float(value)
    return NOT_AVAILABLE if not np.isfinite(v) else format(v, fmt)


_CSV_FIELDS = (
    "d_ax", "d_ay", "Fx", "Fy", "Fx_local", "Fy_local", "M", "x_cm", "T_toe", "force_residual",
    "min_free_gap", "max_free_penetration", "min_contact_reaction", "max_contact_tension",
    "heel_edge_normal_reaction", "toe_edge_normal_reaction",
    "heel_adjacent_free_gap", "toe_adjacent_free_gap",
    "complementarity_score", "violation_score",
)


def evaluation_to_csv_rows(selection: ContactSelection) -> list[dict]:
    """CSV row dicts for every evaluated interval."""
    ev = selection.evaluation
    out = []
    for i in np.flatnonzero(ev.evaluated):
        rec = {
            "row": int(i),
            "contact_start_index": int(ev.contact_start_index[i]),
            "contact_end_index": int(ev.contact_end_index[i]),
            "topology_label": ev.contact_type(i).value,
            "contact_start_x": float(ev.contact_start_x[i]),
            "contact_end_x": float(ev.contact_end_x[i]),
            "anchor_x": float(ev.anchor_x[i]),
            "phi_deg": float(np.rad2deg(ev.phi)),
            "theta_deg": float(np.rad2deg(ev.theta)),
            "varphi_deg": float(np.rad2deg(ev.varphi)),
            "alpha": float(ev.alpha),
            "Fx_star": ev.Fx_star,
            "Fy_star": ev.Fy_star,
        }
        for name in _CSV_FIELDS:
            rec[name] = _f(ev.fields[name][i])
        rec["kf_cond"] = _f(ev.kf_cond[i])
        rec["admissible"] = bool(ev.admissible[i])
        rec["disconnected_contact_warning"] = bool(ev.disconnected_contact_warning[i])
        rec["selected"] = selection.selected_row is not None and int(i) == selection.selected_row
        out.append(rec)
    return out


def export_evaluation_csv(selection: ContactSelection, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = evaluation_to_csv_rows(selection)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_selected_result(
    selection: ContactSelection,
    lookup: ContactLookupResult,
    path: Path | str,
) -> Path:
    """Save the selected interval result to a compressed NPZ."""
    if selection.selected_row is None:
        raise ValueError("No selected record to export.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = selection.selected_row
    payload = {
        "contact_start_index": selection.contact_start_index,
        "contact_end_index": selection.contact_end_index,
        "contact_start_x": selection.contact_start_x,
        "contact_end_x": selection.contact_end_x,
        "contact_anchor_x": selection.anchor_x,
        "contact_anchor_y": selection.anchor_y,
        "topology_label": selection.topology_label,
        "contact_mode": selection.contact_mode.value,
        "candidate_search_method": selection.candidate_search_method,
        "phi": selection.phi,
        "theta": selection.theta,
        "varphi": selection.varphi,
        "phi_ref": lookup.phi_ref,
        "alpha": selection.alpha,
        "Fx_star": selection.Fx_star,
        "Fy_star": selection.Fy_star,
        "kf_cond": selection.kf_cond,
        "x_top": lookup.x_top,
        "x_bottom": lookup.x_bottom,
        "contact_mask": lookup.contact_mask[row],
        "exactly_admissible": selection.exactly_admissible,
        "approximate": selection.approximate,
        "softplus_toe_length": lookup.softplus_toe_length,
        "softplus_kappa": lookup.softplus_kappa,
        "L": lookup.L,
        "H": lookup.H,
        "gauge": "r_a^F = (x_a, 0)",
    }
    for name in _ROW_FIELDS_1D + _ROW_FIELDS_BOTTOM + _ROW_FIELDS_TOP:
        payload[name] = selection.evaluation.fields[name][row]
    np.savez_compressed(path, **payload)
    return path
