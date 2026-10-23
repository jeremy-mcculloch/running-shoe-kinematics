"""Exact passive toe-spring equilibrium coupled to the continuous contact solve.

The measured pitch ``phi`` is the rearfoot (heel -> MTP) angle, so the frame
rotation depends on the toe shape amplitude, ``varphi(alpha) = phi -
atan2(y_mtp_0 + alpha y_mtp_shape_mode, L - toe_length)`` (``toe_spring.RearfootGeometry``).
For each trial ``theta`` (``alpha = tan(theta)``, never ``alpha = theta``) the
contact complementarity problem is solved to convergence at ``varphi(alpha)``
and the toe balance

    g(theta) = Q_alpha,shoe->foot(alpha) - k (arctan(alpha) - theta0) / (1 + alpha^2) = 0,
    Q_alpha,shoe->foot = width * (-psi^T f_{t,y})

is solved for ``theta`` by Brent's method (nested scalar solve). The mapping
``d theta / d alpha = 1 / (1 + alpha^2)`` is the virtual-work factor in the
spring term; ``Q_theta = Q_alpha (1 + alpha^2)``.

Root choice: with a previous ``theta`` (gait sample or GUI) the bracket is grown
around it and the root on the connected branch is taken (continuation). Cold
starts scan the angle bounds, refine every sign change and order the roots by
the existing rule (lowest total potential, ``toe_spring.order_toe_roots``).
Every residual evaluation warm-starts the active set from the closest
already-evaluated ``theta``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from compliance_fem.contact.contact_operator import ContactOperator
from compliance_fem.contact.continuous_contact import ContactSolution, ContactTolerances, solve_contact_state
from compliance_fem.contact.toe_spring import (
    ToeSpringConfig,
    q_theta_from_q_alpha,
    spring_d2U_dalpha2,
    spring_dU_dalpha,
    spring_energy,
    spring_moment_theta,
)

DEFAULT_COLD_SCAN_POINTS = 61
WARM_STEPS_DEG = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 150.0)


@dataclass
class WarmStart:
    """State carried between consecutive evaluations (gait samples)."""

    theta_rad: float | None = None
    active: np.ndarray | None = None


@dataclass
class ToeFrameSolution:
    """Coupled toe + contact solution of one load sample."""

    contact: ContactSolution
    theta_rad: float
    alpha: float
    varphi: float
    Q_alpha_shoe_on_foot: float
    Q_theta_shoe_on_foot: float
    toe_spring_moment_Nm: float
    toe_spring_generalized_force_alpha: float
    toe_equilibrium_residual_Nm: float
    toe_equilibrium_relative_residual: float
    toe_equilibrium_tangent: float
    toe_equilibrium_stable: bool
    toe_spring_energy_J: float
    converged: bool
    status: str
    roots_theta_deg: tuple[float, ...]
    toe_iterations: int
    active_set_iterations: int
    low_load: bool
    timings: dict = field(default_factory=dict)

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))

    def warm_start(self) -> WarmStart:
        return WarmStart(theta_rad=self.theta_rad, active=self.contact.active.copy())


class _Residual:
    """Toe residual with an evaluation cache used for active-set warm starts."""

    def __init__(self, op, F_star, phi_rearfoot, cfg, width, tol, active0):
        self.op, self.F, self.phi, self.cfg, self.width, self.tol = op, F_star, phi_rearfoot, cfg, width, tol
        self.geo = op.rearfoot
        self.cache: dict[float, tuple[float, ContactSolution]] = {}
        self.seed = active0
        self.lcp_iterations = 0
        self.contact_time = 0.0

    def state(self, theta: float) -> tuple[float, ContactSolution]:
        theta = float(theta)
        if theta in self.cache:
            return self.cache[theta]
        if self.cache:
            near = min(self.cache, key=lambda t: abs(t - theta))
            warm = self.cache[near][1].active
        else:
            warm = self.seed
        alpha = float(np.tan(theta))
        varphi = float(self.geo.chord_rotation(self.phi, alpha))
        t0 = time.perf_counter()
        sol = solve_contact_state(self.op, varphi, alpha, self.F, warm_active=warm, tolerances=self.tol)
        self.contact_time += time.perf_counter() - t0
        self.lcp_iterations += int(sol.lcp.iterations)
        Q = self.width * float(sol.diagnostics["Q_alpha_shoe_on_foot_per_m"])
        g = Q - float(spring_dU_dalpha(alpha, self.cfg.toe_stiffness_Nm_per_rad, self.cfg.toe_neutral_angle_rad))
        self.cache[theta] = (g, sol)
        return g, sol

    def __call__(self, theta: float) -> float:
        return self.state(theta)[0]


def _potential(res: _Residual, theta: float, n: int = 24) -> float:
    """Total potential U(alpha) - int_0^alpha Q d alpha' (trapezoid over alpha)."""
    k, th0 = res.cfg.toe_stiffness_Nm_per_rad, res.cfg.toe_neutral_angle_rad
    alpha = float(np.tan(theta))
    a = np.linspace(0.0, alpha, n)
    Q = np.array([res(np.arctan(ai)) + float(spring_dU_dalpha(ai, k, th0)) for ai in a])
    return float(spring_energy(alpha, k, th0) - np.trapezoid(Q, a))


def solve_toe_contact(
    op: ContactOperator,
    F_star: np.ndarray,
    phi_rearfoot: float,
    cfg: ToeSpringConfig,
    width_m: float,
    *,
    warm: WarmStart | None = None,
    low_load: bool = False,
    tolerances: ContactTolerances | None = None,
    cold_scan_points: int = DEFAULT_COLD_SCAN_POINTS,
) -> ToeFrameSolution:
    """Solve contact and the exact toe-spring equilibrium for one sample (per-metre loads, total N·m spring)."""
    tol = tolerances or ContactTolerances(
        toe_equilibrium_abs_tol_Nm=cfg.toe_equilibrium_abs_tol_Nm, toe_equilibrium_rel_tol=cfg.toe_equilibrium_rel_tol
    )
    t_start = time.perf_counter()
    lo, hi = cfg.toe_angle_min_rad, cfg.toe_angle_max_rad
    k, th0 = cfg.toe_stiffness_Nm_per_rad, cfg.toe_neutral_angle_rad
    res = _Residual(op, np.asarray(F_star, dtype=float), float(phi_rearfoot), cfg, float(width_m), tol,
                    None if warm is None else warm.active)
    xtol = 1e-13
    roots: list[float] = []
    status = "ok"
    start = None if warm is None else warm.theta_rad
    if low_load:
        start = th0  # branch connected to the neutral angle
    root = None
    if start is not None and np.isfinite(start):
        start = float(np.clip(start, lo, hi))
        g0 = res(start)
        if g0 == 0.0:
            root = start
        else:
            for step in np.deg2rad(WARM_STEPS_DEG):
                found = []
                for side in (+1.0, -1.0):
                    t1 = float(np.clip(start + side * step, lo, hi))
                    if t1 != start and np.sign(res(t1)) != np.sign(g0):
                        found.append(t1)
                if found:
                    t1 = min(found, key=lambda t: abs(t - start))
                    # Steps grow geometrically, so the first bracket holds the nearest root.
                    a_, b_ = sorted((start, t1))
                    root = optimize.brentq(res, a_, b_, xtol=xtol, rtol=4 * np.finfo(float).eps,
                                           maxiter=tol.maximum_toe_iterations)
                    break
        if root is not None:
            roots = [root]
            status = "continuation"
    if root is None:
        grid = np.linspace(lo, hi, int(cold_scan_points))
        gv = np.array([res(t) for t in grid])
        for i in np.flatnonzero(gv[:-1] * gv[1:] < 0.0):
            roots.append(optimize.brentq(res, grid[i], grid[i + 1], xtol=xtol, rtol=4 * np.finfo(float).eps,
                                         maxiter=tol.maximum_toe_iterations))
        roots += [float(grid[i]) for i in np.flatnonzero(gv == 0.0)]
        if roots:
            if low_load:
                root = min(roots, key=lambda t: abs(t - th0))
            else:
                root = min(roots, key=lambda t: _potential(res, t))
            status = "ok" if len(roots) == 1 else "multiple_roots"
        else:
            root = float(grid[int(np.argmin(np.abs(gv)))])
            status = "no_root_in_bounds"
    g, sol = res.state(root)
    alpha = float(np.tan(root))
    # Tangent of the total potential, d2U/da2 - dQ/da, by central differences in alpha.
    da = 1e-6 * max(1.0, abs(alpha))
    gp = res(np.arctan(alpha + da))
    gm = res(np.arctan(alpha - da))
    dg_da = (gp - gm) / (2 * da)
    tangent = -float(dg_da)
    Q = g + float(spring_dU_dalpha(alpha, k, th0))
    scale = max(abs(Q), abs(float(spring_dU_dalpha(alpha, k, th0))), cfg.toe_equilibrium_abs_tol_Nm)
    tol_abs = cfg.toe_equilibrium_abs_tol_Nm + cfg.toe_equilibrium_rel_tol * scale
    converged = status != "no_root_in_bounds" and abs(g) <= tol_abs and sol.converged
    t_end = time.perf_counter()
    return ToeFrameSolution(
        contact=sol,
        theta_rad=float(root),
        alpha=alpha,
        varphi=float(sol.varphi),
        Q_alpha_shoe_on_foot=float(Q),
        Q_theta_shoe_on_foot=float(q_theta_from_q_alpha(Q, alpha)),
        toe_spring_moment_Nm=float(spring_moment_theta(alpha, k, th0)),
        toe_spring_generalized_force_alpha=float(-spring_dU_dalpha(alpha, k, th0)),
        toe_equilibrium_residual_Nm=float(g),
        toe_equilibrium_relative_residual=float(abs(g) / scale),
        toe_equilibrium_tangent=tangent,
        toe_equilibrium_stable=bool(tangent > 0.0 or (spring_d2U_dalpha2(alpha, k, th0) > 0 and tangent == 0.0)),
        toe_spring_energy_J=float(spring_energy(alpha, k, th0)),
        converged=bool(converged),
        status=status + ("+low_load" if low_load else ""),
        roots_theta_deg=tuple(float(np.rad2deg(r)) for r in sorted(roots)),
        toe_iterations=len(res.cache),
        active_set_iterations=res.lcp_iterations,
        low_load=bool(low_load),
        timings={"contact_solves_s": res.contact_time, "toe_outer_s": t_end - t_start - res.contact_time,
                 "total_s": t_end - t_start},
    )
