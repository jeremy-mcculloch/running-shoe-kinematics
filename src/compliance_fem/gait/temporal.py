"""Sequence-level (Viterbi) interval selection for gait replay.

Temporal continuity is a tie-breaker only: in every frame that has at least one
admissible candidate the inadmissible candidates are removed before the dynamic
program, so a penetrating or tensile interval can never be kept for continuity.
All topology-label transitions (heel / interior / toe / full) are allowed; an
interval change costs ``edge_jump * (|dx_start| + |dx_end|) / L`` plus
``topology_jump`` when the label changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compliance_fem.gait.wrench_control import WrenchCandidate


@dataclass(frozen=True)
class TemporalWeights:
    force_moment: float = 1.0
    penetration: float = 10.0
    tension: float = 10.0
    theta_jump: float = 0.5
    edge_jump: float = 1.0
    topology_jump: float = 2.0
    inadmissible: float = 50.0
    # Passive toe spring (ignored for candidates without toe state).
    toe_residual: float = 10.0
    toe_unstable: float = 20.0
    toe_root_rank: float = 0.05


def _emission_cost(c: WrenchCandidate, w: TemporalWeights, F_scale: float, L_scale: float) -> float:
    fs = max(F_scale, 1.0)
    ls = max(L_scale, 1e-3)
    pen = c.max_free_penetration if np.isfinite(c.max_free_penetration) else ls
    cost = w.force_moment * (c.force_residual / fs + c.moment_residual / (fs * ls))
    cost += w.penetration * (pen / ls)
    if np.isfinite(c.min_contact_reaction) and c.min_contact_reaction < 0.0:
        cost += w.tension * (abs(c.min_contact_reaction) / fs)
    if not c.admissible:
        cost += w.inadmissible
    if not np.isfinite(c.theta_deg):
        cost += w.inadmissible
    toe = getattr(c, "toe", None)
    if toe is not None:
        rel = float(toe.toe_equilibrium_relative_residual)
        cost += w.toe_residual * (min(rel, 1.0) if np.isfinite(rel) else 1.0)
        if not toe.converged:
            cost += w.inadmissible
        if not toe.toe_equilibrium_stable:
            cost += w.toe_unstable
        cost += w.toe_root_rank * max(int(toe.toe_root_index), 0)
    return float(cost) if np.isfinite(cost) else 1.0e12


def interval_jump(a: WrenchCandidate, b: WrenchCandidate, L_scale: float) -> float:
    """``(|x_i - x_i'| + |x_j - x_j'|) / L`` between two candidates' intervals."""
    if a.row < 0 or b.row < 0:
        return 0.0
    d = abs(b.contact_start_x - a.contact_start_x) + abs(b.contact_end_x - a.contact_end_x)
    return float(d) / max(L_scale, 1e-3) if np.isfinite(d) else 0.0


def _transition_cost(a: WrenchCandidate, b: WrenchCandidate, w: TemporalWeights, L_scale: float) -> float:
    cost = 0.0
    if a.contact_type != b.contact_type:
        cost += w.topology_jump
    if np.isfinite(a.theta_deg) and np.isfinite(b.theta_deg):
        cost += w.theta_jump * abs(b.theta_deg - a.theta_deg)
    cost += w.edge_jump * interval_jump(a, b, L_scale)
    return float(cost)


def viterbi_select(
    frames: list[list[WrenchCandidate]],
    *,
    weights: TemporalWeights | None = None,
    F_scale: float = 1.0,
    L_scale: float = 1.0,
    top_k: int = 8,
) -> list[WrenchCandidate]:
    """Select one candidate per frame with temporal regularization.

    Inadmissible candidates are dropped from frames that have an admissible one;
    then the ``top_k`` lowest-emission candidates per frame are kept (ties broken
    by record row so the result does not depend on candidate order).
    """
    w = weights or TemporalWeights()
    if not frames:
        return []

    pruned: list[list[WrenchCandidate]] = []
    for cands in frames:
        if not cands:
            raise ValueError("empty candidate list in Viterbi frame")
        adm = [c for c in cands if c.admissible]
        pool = adm if adm else list(cands)
        scored = sorted(pool, key=lambda c: (_emission_cost(c, w, F_scale, L_scale), c.row, c.theta_deg))
        pruned.append(scored[: max(1, top_k)])

    T = len(pruned)
    cost = [[float("inf")] * len(pruned[t]) for t in range(T)]
    back: list[list[int]] = [[-1] * len(pruned[t]) for t in range(T)]
    for j, c in enumerate(pruned[0]):
        cost[0][j] = _emission_cost(c, w, F_scale, L_scale)
    for t in range(1, T):
        for j, cj in enumerate(pruned[t]):
            em = _emission_cost(cj, w, F_scale, L_scale)
            best_i = 0
            best = float("inf")
            for i, ci in enumerate(pruned[t - 1]):
                val = cost[t - 1][i] + _transition_cost(ci, cj, w, L_scale) + em
                if val < best:
                    best = val
                    best_i = i
            cost[t][j] = best
            back[t][j] = best_i

    j = int(np.argmin(cost[-1]))
    path_idx = [j]
    for t in range(T - 1, 0, -1):
        j = back[t][j]
        path_idx.append(j)
    path_idx.reverse()
    return [pruned[t][path_idx[t]] for t in range(T)]
