"""Continuous node-to-ground contact for one pose: one complementarity solve, no intervals.

Signs (fixed frame ``F``: x heel -> toe, y up, ground ``y = 0``)
-----------------------------------------------------------------
* ``varphi`` is the CCW rotation of the top-attached frame ``T`` relative to
  the reference mesh; ``Q(varphi) = [[c, -s], [s, c]]``.
* Body point ``X`` (reference, local) with local displacement ``d`` sits at
  ``r^F = T + Q (X + d)``. ``T = (T_x, T_y)`` is a **global** frame translation
  shared by every active set: ``T_y`` is solved, ``T_x`` is a pure gauge
  (proportional shear transmits no horizontal constraint) fixed so that the
  frame point ``(x_r, y_r)`` keeps its reference abscissa.
* Gap ``g_k = y^F_k = T_y + s (x_k + u_k) + c (y_k + v_k)``; positive = open.
* Ground force on node ``k`` in ``F``: ``(tau_k, lambda_k) = (mu_eff, 1) lambda_k``,
  ``lambda_k >= 0`` compressive. Local components ``Q^T (mu_eff, 1) lambda_k``.
* Prescribed top load ``F*`` is foot-on-shoe (``F*_y < 0`` in stance). Global
  equilibrium of the free body gives ``sum(lambda) = -F*_y`` and
  ``mu_eff = F*_x / F*_y``, so both force components are met exactly.

The normal-gap response is affine in the reactions,
``g = q + T_y 1 + H lambda`` with ``H = N G_bb B`` (``N`` extracts the ground
normal ``(s, c)`` of each local displacement, ``B`` maps ``lambda`` to local
nodal forces). ``H`` is symmetric positive definite for ``mu_eff = 0``.

Contact sets are node based; fields are continuous across one-node active-set
changes because entering/leaving nodes pass through ``g = lambda = 0``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from compliance_fem.contact.active_set import (
    STATUS_NO_LOAD,
    MixedLCPResult,
    solve_mixed_lcp,
)
from compliance_fem.contact.contact_operator import ContactOperator
from compliance_fem.contact.toe_spring import compute_toe_moment


@dataclass(frozen=True)
class ContactTolerances:
    """Scale-aware tolerances (absolute values are resolved per solve).

    ``gap_tolerance = gap_rel_tol * h``, ``reaction_tolerance = reaction_rel_tol * F_s``
    with ``h`` the median bottom spacing and ``F_s = max(|F*|, force_floor)``.
    """

    gap_rel_tol: float = 1.0e-9
    reaction_rel_tol: float = 1.0e-9
    complementarity_rel_tol: float = 1.0e-9
    force_balance_rel_tol: float = 1.0e-8
    moment_balance_rel_tol: float = 1.0e-8
    toe_equilibrium_abs_tol_Nm: float = 1.0e-6
    toe_equilibrium_rel_tol: float = 1.0e-8
    maximum_active_set_iterations: int = 200
    maximum_toe_iterations: int = 80
    condition_number_warning: float = 1.0e10
    force_floor: float = 1.0
    mu_eff_warning: float = 2.0

    def resolve(self, h: float, F_scale: float, L: float) -> dict[str, float]:
        Fs = max(float(F_scale), float(self.force_floor))
        gap = float(self.gap_rel_tol) * float(h)
        reac = float(self.reaction_rel_tol) * Fs
        return {
            "gap_tolerance": gap,
            "reaction_tolerance": reac,
            "complementarity_tolerance": float(self.complementarity_rel_tol) * float(h) * Fs,
            "force_balance_tolerance": float(self.force_balance_rel_tol) * Fs,
            "moment_balance_tolerance": float(self.moment_balance_rel_tol) * Fs * float(L),
            "force_scale": Fs,
        }


@dataclass(frozen=True)
class ContactPatch:
    """One connected run of active bottom nodes (heel -> toe order)."""

    start: int
    end: int
    start_x: float
    end_x: float
    n_nodes: int
    # Derived display estimate of the patch edges (see ``contact_edge_estimate``); never a constraint.
    edge_estimate_start_x: float = float("nan")
    edge_estimate_end_x: float = float("nan")


def contact_patches(active: np.ndarray, x_bottom: np.ndarray) -> list[ContactPatch]:
    """Connected components of the active mask in bottom-boundary order."""
    a = np.asarray(active, dtype=bool)
    idx = np.flatnonzero(a)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    x = np.asarray(x_bottom, dtype=float)
    return [ContactPatch(int(s), int(e), float(x[s]), float(x[e]), int(e - s + 1)) for s, e in zip(starts, ends)]


def contact_edge_estimate(patch: ContactPatch, lam: np.ndarray, x_bottom: np.ndarray) -> ContactPatch:
    """Sub-element edge coordinates for **display only**.

    Linearly extrapolates the nodal reactions of the two outermost active nodes
    to zero and clamps the result to the first free boundary edge. This is a
    reporting quantity; it never imposes a zero-gap condition inside a linear
    edge (which would force the neighbouring vertex to zero too).
    """
    x = np.asarray(x_bottom, dtype=float)
    n = x.size

    def _edge(k: int, inner: int, outer: int) -> float:
        if outer < 0 or outer >= n:
            return float(x[k])
        lk = float(lam[k])
        li = float(lam[inner]) if 0 <= inner < n and inner != k else lk
        if li > lk >= 0.0:
            xe = x[k] + (x[k] - x[inner]) * lk / (li - lk)
        else:
            xe = 0.5 * (x[k] + x[outer])
        lo, hi = sorted((float(x[k]), float(x[outer])))
        return float(np.clip(xe, lo, hi))

    s, e = patch.start, patch.end
    return ContactPatch(
        patch.start, patch.end, patch.start_x, patch.end_x, patch.n_nodes,
        edge_estimate_start_x=_edge(s, s + 1 if e > s else s, s - 1),
        edge_estimate_end_x=_edge(e, e - 1 if e > s else e, e + 1),
    )


@dataclass
class ContactSolution:
    """Contact state of one pose; local vectors are component-major ``[x...; y...]``."""

    varphi: float
    alpha: float
    F_star: np.ndarray  # fixed-frame foot-on-shoe load (per metre width)
    mu_eff: float
    normal_total: float
    lam: np.ndarray  # (n_b,) compressive normal reactions, fixed frame
    tau: np.ndarray  # (n_b,) tangential reactions, fixed frame
    gap: np.ndarray  # (n_b,) fixed-frame gaps from the reconstructed positions
    active: np.ndarray  # (n_b,) bool
    T: np.ndarray  # (2,) frame translation (T_x gauge, T_y solved)
    f_b: np.ndarray  # (2 n_b,) local bottom nodal forces
    d_b: np.ndarray  # (2 n_b,) local bottom displacement
    F_t: np.ndarray  # (2 n_t,) local top nodal forces (foot on shoe)
    d_t: np.ndarray  # (2 n_t,) local top displacement
    rigid_amplitudes: np.ndarray  # (3,) compliance-gauge rigid amplitudes about (x_r, y_r)
    lcp: MixedLCPResult
    tolerances: dict
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_bottom(self) -> int:
        return int(self.lam.size)

    @property
    def converged(self) -> bool:
        return bool(self.lcp.converged)

    def local_split(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = v.size // 2
        return v[:n], v[n:]

    @property
    def boundary_loads(self) -> np.ndarray:
        """``[F_t; f_b]`` in the order the plate / render influence matrices use."""
        return np.concatenate([self.F_t, self.f_b])


def _normal_operator(op: ContactOperator, c: float, s: float, bx: float, by: float) -> np.ndarray:
    Gxx, Gxy, Gyx, Gyy = op.G_blocks()
    return s * (bx * Gxx + by * Gxy) + c * (bx * Gyx + by * Gyy)


def solve_contact_state(
    op: ContactOperator,
    varphi: float,
    alpha: float,
    F_star: np.ndarray,
    *,
    warm_active: np.ndarray | None = None,
    tolerances: ContactTolerances | None = None,
) -> ContactSolution:
    """Solve the unilateral contact of one pose ``(varphi, alpha)`` under the fixed-frame load ``F_star``."""
    tol = tolerances or ContactTolerances()
    t0 = time.perf_counter()
    F = np.asarray(F_star, dtype=float).reshape(2)
    c, s = float(np.cos(varphi)), float(np.sin(varphi))
    n = op.n_bottom
    Lam = -float(F[1])
    F_scale = float(np.hypot(F[0], F[1]))
    tolv = tol.resolve(op.h_mesh, F_scale, op.L)
    loaded = Lam > tolv["reaction_tolerance"]
    mu = float(F[0] / F[1]) if loaded else 0.0
    # Local components of the fixed-frame unit ground force (mu, 1).
    bx, by = c * mu + s, -s * mu + c
    H = _normal_operator(op, c, s, bx, by)
    x, y = op.x_bottom, op.y_bottom
    dax, day = op.d_alpha[:n], op.d_alpha[n:]
    q = s * (x + alpha * dax) + c * (y + alpha * day)
    t1 = time.perf_counter()
    lcp = solve_mixed_lcp(
        H, q, Lam if loaded else 0.0, active0=warm_active,
        gap_tol=tolv["gap_tolerance"], reaction_tol=tolv["reaction_tolerance"],
        max_iter=tol.maximum_active_set_iterations,
    )
    t2 = time.perf_counter()
    lam = lcp.lam
    f_b = np.concatenate([bx * lam, by * lam])
    d_b = alpha * op.d_alpha + op.G_bb @ f_b
    F_t = alpha * op.f_alpha + op.P_tb @ f_b
    a = alpha * op.a_alpha + op.A_rb @ f_b
    d_t = alpha * op.w_alpha
    u, v = d_b[:n], d_b[n:]
    T_y = float(lcp.t)
    T_x = op.x_r - (c * op.x_r - s * op.y_r)
    gap = T_y + s * (x + u) + c * (y + v)
    tau = mu * lam
    sol = ContactSolution(
        varphi=float(varphi), alpha=float(alpha), F_star=F, mu_eff=mu, normal_total=Lam,
        lam=lam, tau=tau, gap=gap, active=lcp.active.copy(), T=np.array([T_x, T_y]),
        f_b=f_b, d_b=d_b, F_t=F_t, d_t=d_t, rigid_amplitudes=a, lcp=lcp, tolerances=tolv,
    )
    t3 = time.perf_counter()
    sol.diagnostics = contact_diagnostics(op, sol)
    sol.timings = {
        "free_response_s": t1 - t0,
        "contact_solve_s": t2 - t1,
        "reconstruction_s": t3 - t2,
        "diagnostics_s": time.perf_counter() - t3,
    }
    return sol


def fixed_frame_positions(op: ContactOperator, sol: ContactSolution, X: np.ndarray, Y: np.ndarray,
                          u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``r^F = T + Q (X + d)`` for reference points ``(X, Y)`` with local displacement ``(u, v)``."""
    c, s = np.cos(sol.varphi), np.sin(sol.varphi)
    px, py = np.asarray(X) + np.asarray(u), np.asarray(Y) + np.asarray(v)
    return sol.T[0] + c * px - s * py, sol.T[1] + s * px + c * py


def contact_diagnostics(op: ContactOperator, sol: ContactSolution) -> dict:
    """Scalars, equilibrium, complementarity and patch reporting for one solution."""
    n, nt = op.n_bottom, op.n_top
    c, s = np.cos(sol.varphi), np.sin(sol.varphi)
    f_tx, f_ty = sol.F_t[:nt], sol.F_t[nt:]
    Fx_loc, Fy_loc = float(f_tx.sum()), float(f_ty.sum())
    Fx_fix, Fy_fix = c * Fx_loc - s * Fy_loc, s * Fx_loc + c * Fy_loc
    xF, yF = fixed_frame_positions(op, sol, op.x_bottom, op.y_bottom, sol.d_b[:n], sol.d_b[n:])
    tolv = sol.tolerances
    lam, g, act = sol.lam, sol.gap, sol.active
    # Free-body equilibrium in the local frame (forces and moment about (x_r, y_r)).
    fbx, fby = sol.f_b[:n], sol.f_b[n:]
    eq_fx = Fx_loc + float(fbx.sum())
    eq_fy = Fy_loc + float(fby.sum())
    eq_m = float(
        np.sum((op.x_top - op.x_r) * f_ty - (op.y_top - op.y_r) * f_tx)
        + np.sum((op.x_bottom - op.x_r) * fby - (op.y_bottom - op.y_r) * fbx)
    )
    Lam = float(lam.sum())
    T_toe, _, _ = compute_toe_moment(f_tx, f_ty, op.x_top, op.y_top, op.L - op.toe_length)
    patches = [contact_edge_estimate(p, lam, op.x_bottom) for p in contact_patches(act, op.x_bottom)]
    cop_x = float(np.sum(lam * xF) / Lam) if Lam > tolv["reaction_tolerance"] else float("nan")
    inactive = ~act
    out = {
        "Fx_fixed": float(Fx_fix),
        "Fy_fixed": float(Fy_fix),
        "Fx_local": Fx_loc,
        "Fy_local": Fy_loc,
        "force_residual": float(np.hypot(Fx_fix - sol.F_star[0], Fy_fix - sol.F_star[1])),
        "ground_force_residual": float(np.hypot(sol.tau.sum() + sol.F_star[0], Lam + sol.F_star[1])),
        "equilibrium_force_residual": float(np.hypot(eq_fx, eq_fy)),
        "equilibrium_moment_residual": abs(eq_m),
        # Top moment sum(x f_y - y f_x) in the shoe frame about (0, 0) (heel-bottom reference, gait Mz convention).
        "Mz_shoe": float(op.x_top @ f_ty - op.y_top @ f_tx),
        "Mv_shoe": float(op.x_top @ f_ty),
        "T_toe": float(T_toe),
        "Q_alpha_shoe_on_foot_per_m": float(-op.psi_top @ f_ty),
        # Ground-reaction moment about the fixed origin and centre of pressure (fixed frame).
        "grf_moment_fixed": float(np.sum(xF * lam - yF * sol.tau)),
        "cop_fixed_x": cop_x,
        "cop_heel_relative_x": cop_x - float(xF[0]) if np.isfinite(cop_x) else float("nan"),
        "min_gap": float(np.min(g)),
        "min_gap_node": int(np.argmin(g)),
        "max_active_gap": float(np.max(np.abs(g[act]))) if act.any() else 0.0,
        "min_inactive_gap": float(np.min(g[inactive])) if inactive.any() else float("inf"),
        "min_reaction": float(np.min(lam[act])) if act.any() else 0.0,
        "max_complementarity": float(np.max(np.abs(g * lam))),
        "max_penetration": float(max(0.0, -np.min(g))),
        "max_tension": float(max(0.0, -np.min(lam))),
        "n_active": int(act.sum()),
        "n_patches": len(patches),
        "patches": patches,
        "first_active": int(np.flatnonzero(act)[0]) if act.any() else -1,
        "last_active": int(np.flatnonzero(act)[-1]) if act.any() else -1,
        "disconnected_contact": len(patches) > 1,
        "active_set_iterations": int(sol.lcp.iterations),
        "active_set_rule": sol.lcp.pivot_rule,
        "factorizations": int(sol.lcp.factorizations),
        "active_block_rcond": float(sol.lcp.rcond),
        "lcp_status": sol.lcp.status,
        "mu_eff": float(sol.mu_eff),
        "loaded": sol.lcp.status != STATUS_NO_LOAD,
    }
    out["admissible"] = bool(
        sol.converged
        and out["max_penetration"] <= tolv["gap_tolerance"] + 1e3 * np.finfo(float).eps * op.L
        and out["max_tension"] <= tolv["reaction_tolerance"]
    )
    return out
