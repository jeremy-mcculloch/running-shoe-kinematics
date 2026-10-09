"""Per-frame passive toe-spring equilibrium over the stored contact intervals.

The measured pitch ``phi`` is the rearfoot (heel -> MTP) angle, so the chord
frame of the lookup rotates with the toe: ``varphi(alpha) = phi - atan2(dy_mtp +
alpha phi1_mtp, L - toe_length)``. For each interval record the prescribed fixed-frame force
``(Fx*, Fy*)`` is rotated into the chord frame at ``varphi(alpha)`` and the
anchor translations are eliminated analytically, including the curved-sole
closure (``toe_spring.RearfootToeModel``). The toe balance
``Q_alpha,shoe->foot(alpha) = dU/dalpha`` is solved exactly on the configured
angle bounds; no small-angle approximation is used.

Search (no FEM solve): the interval records near the previous interval are
screened first (local, then expanded radii), then all records. Screening finds
every toe root of every record at once (``toe_spring.screen_toe_roots``) and
applies the all-node unilateral checks (``force_control.reconstruct_rows``).
The admissible (record, root) pairs of the first stage that has any, or else the
least-violating pairs, are re-solved with the exact scalar toe solver and
returned as candidates for the sequence selector.

Units: the lookup is plane strain with unit thickness, so its generalized
force is per metre width. The total shoe-on-foot generalized force that
balances the physical toe spring is ``width_m * Q_lookup`` (N·m).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compliance_fem.contact.lookup import (
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MZ,
    ContactLookupResult,
    lookup_q_alpha_basis,
    lookup_rearfoot_geometry,
)
from compliance_fem.contact.topology import interval_distance
from compliance_fem.contact.corotation import contract_basis, rotate_vector_to_fixed
from compliance_fem.contact.force_control import (
    EXACT_ALL,
    SEARCH_EXPANDED,
    SEARCH_FALLBACK,
    SEARCH_GLOBAL,
    SEARCH_LOCAL,
    SelectionConfig,
    Tolerances,
    reconstruct_rows,
    refine_reconstruction,
    refine_top_k,
)
from compliance_fem.gait.candidate import (
    WrenchCandidate,
    _candidate_geometry,
    diagnostics_from_reconstruction,
)
from compliance_fem.contact.toe_spring import (
    ToeRoot,
    ToeSpringConfig,
    order_toe_roots,
    q_theta_from_q_alpha,
    rearfoot_toe_model,
    screen_toe_roots,
    solve_toe_equilibrium_model,
    spring_dU_dalpha,
    spring_generalized_force_alpha,
    spring_moment_theta,
)


@dataclass(frozen=True)
class ToeCandidateState:
    """Toe-equilibrium diagnostics attached to one (interval, root) candidate."""

    toe_model: str
    alpha: float
    theta_rad: float
    Q_alpha_shoe_on_foot: float
    Q_theta_shoe_on_foot: float
    toe_spring_moment_Nm: float
    toe_spring_generalized_force_alpha: float
    toe_equilibrium_residual_Nm: float
    toe_equilibrium_relative_residual: float
    toe_equilibrium_tangent: float
    toe_equilibrium_stable: bool
    toe_root_count: int
    toe_root_index: int
    toe_spring_energy_J: float
    toe_potential_J: float
    toe_solve_status: str
    converged: bool
    low_load: bool
    q0_Nm: float
    q1_Nm: float
    roots_theta_deg: tuple[float, ...]
    kd_cond: float
    varphi_rad: float = float("nan")
    Q_coeff: tuple[float, ...] = ()

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))


@dataclass
class _Pairs:
    """Screened (record, theta) pairs with their reconstruction."""

    rows: np.ndarray
    theta: np.ndarray
    has_root: np.ndarray
    stable: np.ndarray
    rec: dict


def _screen_pairs(
    lookup: ContactLookupResult,
    rows: np.ndarray,
    *,
    S_all: np.ndarray,
    q_all: np.ndarray,
    F_fixed: np.ndarray,
    phi_rad: float,
    geometry,
    config: ToeSpringConfig,
    width: float,
    tol: Tolerances,
    low_load: bool,
    cond_limit: float,
) -> _Pairs | None:
    if rows.size == 0:
        return None
    scr = screen_toe_roots(
        S_all[rows], q_all[rows], F_fixed, phi_rad, geometry, config, width,
        rows=rows, cond_limit=cond_limit,
    )
    p_idx = scr.pair_row
    p_theta = scr.pair_theta
    p_stable = scr.pair_tangent > 0.0
    if low_load and p_idx.size:
        th0 = config.toe_neutral_angle_rad
        keep = np.zeros(p_idx.size, dtype=bool)
        for r in np.unique(p_idx):
            sel = np.flatnonzero(p_idx == r)
            keep[sel[np.argmin(np.abs(p_theta[sel] - th0))]] = True
        p_idx, p_theta, p_stable = p_idx[keep], p_theta[keep], p_stable[keep]
    no_root = np.flatnonzero(scr.well_conditioned & ~scr.row_has_root & np.isfinite(scr.diag_theta))
    idx = np.concatenate([p_idx, no_root]).astype(int)
    theta = np.concatenate([p_theta, scr.diag_theta[no_root]])
    has_root = np.concatenate([np.ones(p_idx.size, bool), np.zeros(no_root.size, bool)])
    stable = np.concatenate([p_stable, np.zeros(no_root.size, bool)])
    if idx.size == 0:
        return None
    alpha = np.tan(theta)
    varphi = geometry.chord_rotation(phi_rad, alpha)
    c, s = np.cos(varphi), np.sin(varphi)
    z = np.stack([c * F_fixed[0] + s * F_fixed[1], -s * F_fixed[0] + c * F_fixed[1], c - 1.0, -s, alpha])
    d = np.einsum("pij,jp->pi", scr.D[idx], z) + scr.d_offset[idx]
    gamma = np.column_stack([alpha, d[:, 0], d[:, 1], c - 1.0, -s])
    penalty = float(tol.flag_penalty) * ((~has_root).astype(float) + (~stable).astype(float))
    rec = reconstruct_rows(
        lookup, rows[idx], gamma, varphi, F_fixed, tol,
        extra_penalty=penalty, kf_cond=scr.kd_cond[idx], full_fields=False,
    )
    return _Pairs(rows=rows[idx], theta=theta, has_root=has_root, stable=stable, rec=rec)


def _concat(a: _Pairs | None, b: _Pairs | None) -> _Pairs | None:
    if a is None:
        return b
    if b is None:
        return a
    rec = {}
    for k, v in a.rec.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == a.rows.size:
            rec[k] = np.concatenate([v, b.rec[k]])
        else:
            rec[k] = v
    return _Pairs(
        rows=np.concatenate([a.rows, b.rows]),
        theta=np.concatenate([a.theta, b.theta]),
        has_root=np.concatenate([a.has_root, b.has_root]),
        stable=np.concatenate([a.stable, b.stable]),
        rec=rec,
    )


def _stages(lookup: ContactLookupResult, previous, cfg: SelectionConfig) -> list[tuple[str, np.ndarray]]:
    valid = np.asarray(lookup.valid_rows, dtype=int)
    stages: list[tuple[str, np.ndarray]] = []
    if previous is not None and cfg.use_temporal_continuity:
        starts = np.asarray(lookup.contact_start_index)[valid]
        ends = np.asarray(lookup.contact_end_index)[valid]
        dist = interval_distance(starts, ends, previous)
        stages.append((SEARCH_LOCAL, valid[dist <= cfg.local_radius]))
        for r in cfg.expansion_radii:
            if r > cfg.local_radius:
                stages.append((SEARCH_EXPANDED, valid[dist <= r]))
    stages.append((SEARCH_GLOBAL, valid))
    return stages


def _shortlist(
    lookup: ContactLookupResult, pairs: _Pairs, pool: np.ndarray, previous, cfg: SelectionConfig,
    k: int, admissible: bool,
) -> np.ndarray:
    """Order-independent shortlist: admissible pairs by (stability, complementarity
    [+ topology penalty], interval distance, row, theta); otherwise by violation score."""
    rows = pairs.rows[pool]
    if admissible:
        dist = np.zeros(pool.size)
        if previous is not None and cfg.use_temporal_continuity:
            starts = np.asarray(lookup.contact_start_index, dtype=int)[rows]
            ends = np.asarray(lookup.contact_end_index, dtype=int)[rows]
            dist = interval_distance(starts, ends, previous).astype(float)
        comp = np.nan_to_num(pairs.rec["complementarity_score"][pool], nan=np.inf)
        primary = comp + cfg.topology_change_penalty * dist
        unstable = (~pairs.stable[pool]).astype(int)
        order = np.lexsort((pairs.theta[pool], rows, dist, primary, unstable))
    else:
        rec = pairs.rec
        ordered = refine_top_k(
            pool,
            lambda p: (pairs.theta[p], pairs.rows[p], rec["violation_score"][p]),
            lambda p: rec["fields_exact"][p],
            lambda p: refine_reconstruction(lookup, rec, p),
            k,
        )
        top = ordered[: max(1, int(k))]
        finite = np.isfinite(rec["violation_score"][top])
        return top[finite] if np.any(finite) else top
    return pool[order[: max(1, int(k))]]


def solve_passive_toe_candidates(
    lookup: ContactLookupResult,
    *,
    Fx_star: float,
    Fy_star: float,
    Mz_meas: float,
    phi_rad: float,
    config: ToeSpringConfig,
    width_m: float,
    tolerances: Tolerances | None = None,
    low_load: bool = False,
    cond_limit: float = 1.0e8,
    previous_interval: tuple[int, int] | None = None,
    selection_config: SelectionConfig | None = None,
    max_candidates: int = 16,
) -> list[WrenchCandidate]:
    """Passive toe equilibrium candidates for one frame.

    ``Fx_star``, ``Fy_star`` and ``Mz_meas`` are model units (per metre width,
    foot-on-shoe sign); ``phi_rad`` is the fixed-frame heel -> MTP angle.
    ``Mz_meas`` is not prescribed; it only feeds the moment residual diagnostic.
    With ``low_load`` each record keeps only the root closest to ``theta0``.
    Every returned candidate carries its ``search_method``. When admissible
    candidates exist, only admissible candidates are returned.
    """
    tol = tolerances or Tolerances()
    cfg = selection_config or SelectionConfig()
    width = float(width_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"width_m must be positive, got {width_m!r}.")
    S_all = np.asarray(lookup.scalar_lookup, dtype=float)
    q_all = lookup_q_alpha_basis(lookup)
    geometry = lookup_rearfoot_geometry(lookup)
    F_fixed = np.array([float(Fx_star), float(Fy_star)])

    screened = np.zeros(lookup.n_records, dtype=bool)
    pairs: _Pairs | None = None
    method = SEARCH_FALLBACK
    pool = np.zeros(0, dtype=int)
    for stage_method, stage_rows in _stages(lookup, previous_interval, cfg):
        new_rows = stage_rows[~screened[stage_rows]]
        screened[new_rows] = True
        pairs = _concat(
            pairs,
            _screen_pairs(
                lookup, new_rows, S_all=S_all, q_all=q_all, F_fixed=F_fixed, phi_rad=float(phi_rad),
                geometry=geometry, config=config, width=width, tol=tol, low_load=low_load,
                cond_limit=cond_limit,
            ),
        )
        if pairs is None:
            continue
        in_stage = np.isin(pairs.rows, stage_rows)
        adm = np.flatnonzero(in_stage & pairs.rec["admissible"] & pairs.has_root)
        if adm.size:
            method = stage_method
            pool = adm
            break
    if pairs is None:
        return []
    admissible_found = method != SEARCH_FALLBACK
    if not admissible_found:
        pool = np.flatnonzero(np.isfinite(pairs.rec["violation_score"]))
        if pool.size == 0:
            pool = np.arange(pairs.rows.size)
    chosen = _shortlist(lookup, pairs, pool, previous_interval, cfg, max_candidates, admissible_found)
    cands = _exact_candidates(
        lookup, pairs, chosen, S_all=S_all, q_all=q_all, F_fixed=F_fixed, phi_rad=float(phi_rad),
        geometry=geometry, config=config, width=width, tol=tol, low_load=low_load,
        cond_limit=cond_limit, Mz_meas=float(Mz_meas), method=method,
    )
    if admissible_found and any(c.admissible for c in cands):
        cands = [c for c in cands if c.admissible]
    return cands


def _exact_candidates(
    lookup, pairs: _Pairs, chosen: np.ndarray, *, S_all, q_all, F_fixed, phi_rad, geometry,
    config: ToeSpringConfig, width, tol, low_load, cond_limit, Mz_meas, method,
) -> list[WrenchCandidate]:
    k = config.toe_stiffness_Nm_per_rad
    th0 = config.toe_neutral_angle_rad
    wrench_cols = [SCALAR_FX, SCALAR_FY, SCALAR_MZ]
    prepared = []
    for p in chosen:
        row = int(pairs.rows[p])
        model = rearfoot_toe_model(S_all[row], q_all[row], F_fixed, phi_rad, geometry, width, cond_limit=cond_limit)
        if not model.well_conditioned:
            continue
        sol = solve_toe_equilibrium_model(model, config)
        roots: list[ToeRoot] = list(sol.roots)
        status = sol.status + ("+low_load" if low_load else "")
        if roots:
            i_match = int(np.argmin([abs(r.theta_rad - pairs.theta[p]) for r in roots]))
            root = roots[i_match]
        elif sol.diagnostic is not None:
            root, i_match = sol.diagnostic, -1
        else:
            continue
        rank = -1
        if roots:
            order = order_toe_roots(roots)
            rank = int(order.index(i_match))
        gamma = model.gamma(root.alpha)
        prepared.append((row, model, sol, root, rank, status, gamma, float(model.varphi(root.alpha))))
    if not prepared:
        return []
    rows = np.array([x[0] for x in prepared], dtype=int)
    gam = np.stack([x[6] for x in prepared])
    vphi = np.array([x[7] for x in prepared])
    penalty = float(tol.flag_penalty) * np.array(
        [float(not x[3].converged) + float(not x[3].stable) for x in prepared]
    )
    rec = reconstruct_rows(
        lookup, rows, gam, vphi, F_fixed, tol, extra_penalty=penalty,
        kf_cond=np.array([x[1].kd_cond for x in prepared]), full_fields=False, exact=EXACT_ALL,
    )
    out: list[WrenchCandidate] = []
    for i, (row, model, sol, root, rank, status, gamma, varphi) in enumerate(prepared):
        S = S_all[row]
        w_pred = contract_basis(S[:, wrench_cols], gamma, mode_axis=0)
        fx_f, fy_f = rotate_vector_to_fixed(w_pred[0], w_pred[1], varphi)
        Q_alpha = float(model.generalized_force(root.alpha))
        state = ToeCandidateState(
            toe_model=config.toe_model,
            alpha=float(root.alpha),
            theta_rad=float(root.theta_rad),
            Q_alpha_shoe_on_foot=Q_alpha,
            Q_theta_shoe_on_foot=float(q_theta_from_q_alpha(Q_alpha, root.alpha)),
            toe_spring_moment_Nm=float(spring_moment_theta(root.alpha, k, th0)),
            toe_spring_generalized_force_alpha=float(spring_generalized_force_alpha(root.alpha, k, th0)),
            toe_equilibrium_residual_Nm=float(Q_alpha - spring_dU_dalpha(root.alpha, k, th0)),
            toe_equilibrium_relative_residual=float(root.relative_residual),
            toe_equilibrium_tangent=float(root.tangent),
            toe_equilibrium_stable=bool(root.stable),
            toe_root_count=int(sol.n_roots),
            toe_root_index=rank,
            toe_spring_energy_J=float(root.spring_energy_J),
            toe_potential_J=float(root.potential_J),
            toe_solve_status=status,
            converged=bool(root.converged),
            low_load=bool(low_load),
            q0_Nm=float(sol.q0),
            q1_Nm=float(sol.q1),
            roots_theta_deg=tuple(float(r.theta_deg) for r in sol.roots),
            kd_cond=float(model.kd_cond),
            varphi_rad=varphi,
            Q_coeff=tuple(float(c) for c in model.coeff),
        )
        force_ok = bool(rec["force_ok"][i])
        unilateral_ok = _unilateral_only(rec, i)
        admissible = bool(root.converged and rec["admissible"][i])
        cand_status = f"passive_spring:{status}"
        if not root.converged:
            cand_status += "+diagnostic_min_residual"
        if not root.stable:
            cand_status += "+unstable"
        if not unilateral_ok:
            cand_status += "+unilateral_violation"
        if not force_ok:
            cand_status += "+force_residual"
        if bool(rec["disconnected_contact_warning"][i]):
            cand_status += "+interior_tension"
        out.append(
            WrenchCandidate(
                row=row,
                **_candidate_geometry(lookup, row),
                d_ax=float(gamma[1]),
                d_ay=float(gamma[2]),
                alpha=float(root.alpha),
                theta_deg=float(root.theta_deg),
                Fx_pred=float(fx_f),
                Fy_pred=float(fy_f),
                Mz_pred=float(w_pred[2]),
                force_residual=float(rec["force_residual"][i]),
                moment_residual=float(abs(float(w_pred[2]) - Mz_meas)),
                cop_residual=float("nan"),
                cond=float(model.kd_cond),
                max_free_penetration=float(rec["max_free_penetration"][i]),
                min_contact_reaction=float(rec["min_contact_reaction"][i]),
                admissible=admissible,
                status=cand_status,
                gamma=gamma,
                toe=state,
                search_method=method,
                diagnostics=diagnostics_from_reconstruction(rec, i),
            )
        )
    return out


def _unilateral_only(rec: dict, i: int) -> bool:
    sc = rec["scales"]
    return bool(
        rec["finite"][i]
        and rec["min_free_gap"][i] >= -sc.tau_g_eff
        and rec["min_contact_reaction"][i] >= -sc.tau_R_eff
    )


def pick_instant_best_passive(
    cands: list[WrenchCandidate],
    previous_theta_deg: float | None = None,
) -> WrenchCandidate | None:
    """Single-frame preference: admissible, converged, stable, continuity, root rank."""
    if not cands:
        return None

    def key(c: WrenchCandidate) -> tuple:
        toe = c.toe
        converged = bool(toe is not None and toe.converged)
        stable = bool(toe is not None and toe.toe_equilibrium_stable)
        jump = (
            abs(c.theta_deg - previous_theta_deg)
            if previous_theta_deg is not None and np.isfinite(previous_theta_deg) and np.isfinite(c.theta_deg)
            else 0.0
        )
        return (
            0 if c.admissible else 1,
            0 if converged else 1,
            0 if stable else 1,
            jump,
            toe.toe_root_index if toe is not None else 1_000,
            c.max_free_penetration if np.isfinite(c.max_free_penetration) else 1e30,
            c.force_residual,
            c.row,
        )

    return min(cands, key=key)
