"""Validation study of the continuous unilateral contact solve.

Reproducible from a measured-sole config (default ``configs/setup.json``)::

    venv/bin/python scripts/validate_continuous_contact.py [config.json] [--out DIR]

For every case the shared-operator solution is checked against an independent
direct sparse FEM solve of the same state (top clamped to ``alpha w``, bottom
nodal ground forces from the solution, plate constraints ``B_p``): the
reconstructed bottom displacement, top forces and plate fields are compared and
the direct-FEM gaps are checked for complementarity. Sweeps over a continuously
varied input are plotted with the active-set changes marked:

* heel / midfoot / toe dominant and nearly full contact (pose sweep);
* a node entering (load sweep) and leaving (rotation sweep) contact;
* passive toe-spring deformation (toe solve along a load ramp);
* a representative gait replay (Wang data when present, else a synthetic stance).

Writes PNG figures and ``summary.json`` to ``--out`` (default
``outputs/validation_continuous_contact``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

from compliance_fem.contact.contact_model import build_contact_model, plate_fields  # noqa: E402
from compliance_fem.contact.continuous_contact import solve_contact_state  # noqa: E402
from compliance_fem.contact.continuous_toe import solve_toe_contact  # noqa: E402
from compliance_fem.contact.model_setup import load_model_setup  # noqa: E402
from compliance_fem.fem.compliance import ComplianceResult, compute_compliance  # noqa: E402
from compliance_fem.gait.replay import replay_stance  # noqa: E402


class DirectFEM:
    """Sparse FEM with the top selector clamped; factorized once, then one solve per state."""

    def __init__(self, fem: ComplianceResult) -> None:
        K = fem.K.tocsc()
        self.n_q = int(fem.n_primal)
        self.u_top, self.v_top = fem.selector_dofs("top")
        self.u_bot, self.v_bot = fem.selector_dofs("bottom")
        self.plate_dofs = np.concatenate([fem.plate_u_dof_ids, fem.plate_v_dof_ids, fem.plate_rotation_dof_ids])
        B = fem.plate_inextensibility_constraint.tocsc()
        n_l = B.shape[0]
        A = sparse.csc_matrix(sparse.bmat([[K, B.T], [B, None]], format="csc"))
        n = A.shape[0]
        self.D = np.unique(np.concatenate([self.u_top, self.v_top]))
        self.I = np.setdiff1d(np.arange(n), self.D)
        self.A = A
        self.lu = splu(A[self.I][:, self.I].tocsc())
        self.n_l = n_l

    def solve(self, w_top: np.ndarray, f_b: np.ndarray) -> dict:
        n_t, n_b = self.u_top.size, self.u_bot.size
        x = np.zeros(self.A.shape[0])
        x[self.u_top] = w_top[:n_t]
        x[self.v_top] = w_top[n_t:]
        f = np.zeros(self.A.shape[0])
        np.add.at(f, self.u_bot, f_b[:n_b])
        np.add.at(f, self.v_bot, f_b[n_b:])
        rhs = f[self.I] - self.A[self.I][:, self.D] @ x[self.D]
        x[self.I] = self.lu.solve(rhs)
        r = self.A @ x - f
        u = x[: self.n_q]
        return {
            "u": u,
            "lam": x[self.n_q :],
            "top_force": np.concatenate([r[self.u_top], r[self.v_top]]),
            "bottom": np.concatenate([u[self.u_bot], u[self.v_bot]]),
        }


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def direct_check(direct: DirectFEM, model, sol) -> dict:
    ref = direct.solve(sol.d_t, sol.f_b)
    op = model.contact_operator
    n = op.n_bottom
    c, s = np.cos(sol.varphi), np.sin(sol.varphi)
    u, v = ref["bottom"][:n], ref["bottom"][n:]
    gap = sol.T[1] + s * (op.x_bottom + u) + c * (op.y_bottom + v)
    pf = plate_fields(model, sol)
    lam_scale = max(float(sol.lam.max()), 1.0)
    return {
        "bottom_displacement_rel_error": rel(sol.d_b, ref["bottom"]),
        "top_force_rel_error": rel(sol.F_t, ref["top_force"]),
        "plate_rel_error": rel(
            np.concatenate([pf["u"], pf["v"], pf["theta"]]),
            ref["u"][direct.plate_dofs],
        ),
        "direct_min_gap_over_h": float(gap.min() / op.h_mesh),
        "direct_max_active_gap_over_h": float(np.max(np.abs(gap[sol.active])) / op.h_mesh) if sol.active.any() else 0.0,
        "direct_max_complementarity_rel": float(np.max(np.abs(gap * sol.lam)) / (op.h_mesh * lam_scale)),
    }


def sweep(model, direct, params, solve_fn: Callable[[float, Any], Any], label: str):
    """Solve along ``params``; return per-step scalars, transitions and the direct-FEM check summary."""
    op = model.contact_operator
    rows, checks, states = [], [], []
    prev = None
    t_solve = 0.0
    for p in params:
        t0 = time.perf_counter()
        out = solve_fn(p, prev)
        sol = getattr(out, "contact", out)
        prev = out
        t_solve += time.perf_counter() - t0
        states.append(out)
        d = sol.diagnostics
        pf = plate_fields(model, sol)
        rows.append({
            "p": float(p),
            "Fx": d["Fx_fixed"], "Fy": d["Fy_fixed"], "Mz": d["Mz_shoe"], "cop": d["cop_fixed_x"],
            "theta": float(np.rad2deg(getattr(out, "theta_rad", np.arctan(sol.alpha)))),
            "n_active": d["n_active"], "first": d["first_active"], "last": d["last_active"],
            "n_patches": d["n_patches"],
            "min_gap": d["min_gap"], "min_reaction": d["min_reaction"],
            "plate_v_mid": float(pf["v"][pf["v"].size // 2]),
            "max_complementarity": d["max_complementarity"], "active": sol.active.copy(),
            "iterations": d["active_set_iterations"],
        })
        checks.append(direct_check(direct, model, sol))
    act = np.array([r["active"] for r in rows])
    transitions = np.flatnonzero(np.any(act[1:] != act[:-1], axis=1))
    # Continuity: bisect every active-set change to its transition point and compare
    # the left / right limits of the physical outputs (relative to each output's range).
    keys = ("Fx", "Fy", "Mz", "cop", "theta", "plate_v_mid")
    def _scale(key: str) -> float:
        y = np.array([r[key] for r in rows], float)
        y = y[np.isfinite(y)]
        return max(float(np.ptp(y)) if y.size else 0.0, float(np.max(np.abs(y))) if y.size else 0.0, 1e-300)

    ranges = {k: _scale(k) for k in keys}
    # Prescribed force components may be identically zero: scale them by the load magnitude.
    F_mag = max(float(np.max([np.hypot(r["Fx"], r["Fy"]) for r in rows])), 1e-300)
    ranges["Fx"] = ranges["Fy"] = F_mag
    limit_gap = {k: 0.0 for k in (*keys, "bottom_displacement", "normal_reaction")}
    for k in transitions:
        lo, hi = float(params[k]), float(params[k + 1])
        warm = states[k]  # the sweep state at the left grid point (same branch)
        a0 = getattr(solve_fn(lo, warm), "contact", solve_fn(lo, warm)).active
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if np.array_equal(getattr(solve_fn(mid, warm), "contact", solve_fn(mid, warm)).active, a0):
                lo = mid
            else:
                hi = mid
        L, R = solve_fn(lo, warm), solve_fn(hi, warm)
        sl, sr = getattr(L, "contact", L), getattr(R, "contact", R)
        for key, (vl, vr) in _outputs(model, L, R).items():
            if key in ranges:
                limit_gap[key] = max(limit_gap[key], abs(vl - vr) / ranges[key])
        limit_gap["bottom_displacement"] = max(
            limit_gap["bottom_displacement"], float(np.max(np.abs(sl.d_b - sr.d_b)) / max(np.max(np.abs(sl.d_b)), 1e-300))
        )
        limit_gap["normal_reaction"] = max(
            limit_gap["normal_reaction"], float(np.max(np.abs(sl.lam - sr.lam)) / max(float(sl.lam.max()), 1e-300))
        )
    summary = {
        "label": label,
        "n_steps": len(rows),
        "n_transitions": int(transitions.size),
        "max_patches": int(max(r["n_patches"] for r in rows)),
        "max_iterations": int(max(r["iterations"] for r in rows)),
        "solve_time_per_step_ms": 1e3 * t_solve / max(len(rows), 1),
        "max_relative_limit_difference_at_transitions": limit_gap,
        **{f"max_{k}": float(max(c[k] for c in checks)) for k in checks[0] if k != "direct_min_gap_over_h"},
        "min_direct_min_gap_over_h": float(min(c["direct_min_gap_over_h"] for c in checks)),
    }
    return rows, transitions, summary


def _outputs(model, L, R) -> dict:
    out = {}
    for name, obj in (("L", L), ("R", R)):
        sol = getattr(obj, "contact", obj)
        d = sol.diagnostics
        pf = plate_fields(model, sol)
        out[name] = {"Fx": d["Fx_fixed"], "Fy": d["Fy_fixed"], "Mz": d["Mz_shoe"], "cop": d["cop_fixed_x"],
                     "theta": float(np.rad2deg(getattr(obj, "theta_rad", np.arctan(sol.alpha)))),
                     "plate_v_mid": float(pf["v"][pf["v"].size // 2])}
    return {k: (out["L"][k], out["R"][k]) for k in out["L"]}


def plot_sweep(rows, transitions, title, xlabel, path: Path) -> None:
    p = np.array([r["p"] for r in rows])
    panels = [
        ("Fy", "fixed-frame Fy (N/m)"), ("Mz", "top Mz (N·m/m)"), ("cop", "COP x (m)"),
        ("theta", "toe angle θ (°)"), ("n_active", "active nodes"), ("min_gap", "min gap (m)"),
        ("min_reaction", "min active λ (N/m)"), ("plate_v_mid", "plate v, mid node (m)"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    for ax, (key, lab) in zip(axes.ravel(), panels):
        y = np.array([r[key] for r in rows], float)
        ax.plot(p, y, ".-", ms=3, lw=1)
        if key == "n_active":
            ax.plot(p, [r["first"] for r in rows], "--", lw=0.8, label="first active")
            ax.plot(p, [r["last"] for r in rows], ":", lw=0.8, label="last active")
            ax.legend(fontsize=7)
        for k in transitions:
            ax.axvline(0.5 * (p[k] + p[k + 1]), color="0.8", lw=0.6, zorder=0)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"{title} (grey lines: active-set changes)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="configs/setup.json")
    parser.add_argument("--out", default="outputs/validation_continuous_contact")
    parser.add_argument("--trc", default=None, help="Wang TRC (default: first preferred trial under data/wang)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    setup = load_model_setup(args.config)
    t0 = time.perf_counter()
    fem = compute_compliance(setup.sole)
    model = build_contact_model(fem, toe_length=setup.toe_length_m, kappa=setup.softplus_kappa,
                                reciprocity_tol=setup.lookup.reciprocity_tol)
    direct = DirectFEM(fem)
    op = model.contact_operator
    width = float(setup.runtime.shoe_width_m)
    toe_cfg = setup.toe_spring
    summary: dict = {"config": str(Path(args.config).resolve()), "n_bottom_nodes": op.n_bottom,
                     "build_s": time.perf_counter() - t0, "cases": {}}

    def static(F, alpha):
        return lambda varphi, prev: solve_contact_state(
            op, float(varphi), alpha, np.asarray(F, float),
            warm_active=None if prev is None else getattr(prev, "contact", prev).active,
        )

    Fbody = np.array([0.0, -12000.0])  # N per metre width
    cases: dict[str, tuple[Any, Any, str, str]] = {
        "pose_heel_to_toe": (np.deg2rad(np.linspace(12.0, -12.0, 241)), static(Fbody, 0.0),
                             "pose sweep: heel-dominant -> midfoot -> toe-dominant", "chord rotation varphi (rad)"),
        "node_leaving": (np.deg2rad(np.linspace(0.0, 4.0, 161)), static(Fbody, 0.0),
                         "rotation sweep: nodes leaving contact", "chord rotation varphi (rad)"),
    }

    def load_sweep(Fy, prev):
        return solve_contact_state(op, 0.0, 0.0, np.array([0.1 * abs(Fy), Fy]),
                                   warm_active=None if prev is None else prev.active)

    cases["node_entering_to_near_full"] = (np.linspace(-500.0, -60000.0, 241), load_sweep,
                                           "load sweep: nodes entering, up to nearly full contact",
                                           "Fy* (N/m, foot on shoe)")

    def toe_ramp(Fy, prev):
        warm = None if prev is None else prev.warm_start()
        return solve_toe_contact(op, np.array([0.12 * abs(Fy), Fy]), np.deg2rad(-8.0), toe_cfg, width, warm=warm)

    cases["passive_toe_spring"] = (np.linspace(-500.0, -60000.0, 161), toe_ramp,
                                   "passive toe spring along a forefoot load ramp", "Fy* (N/m, foot on shoe)")

    for name, (params, fn, title, xlabel) in cases.items():
        rows, tr, summ = sweep(model, direct, params, fn, title)
        plot_sweep(rows, tr, title, xlabel, out / f"{name}.png")
        summary["cases"][name] = summ

    # Representative gait replay.
    trc = mot = None
    try:
        from compliance_fem.gait.wang_io import discover_trials

        trials = [t for t in discover_trials(Path("data/wang")) if t.trc_path and t.mot_path]
        pick = next((t for t in trials if t.is_preferred_r1), trials[0] if trials else None)
        if pick is not None:
            trc, mot = pick.trc_path, pick.mot_path
    except Exception:  # noqa: BLE001
        trc = mot = None
    t = time.perf_counter()
    if trc is not None:
        r = replay_stance(model, trc_path=trc, mot_path=mot, toe_config=toe_cfg, shoe_width_m=width)
        source = str(trc)
    else:
        tt = np.linspace(0.0, 0.25, 301)
        Fy = -1800.0 * np.sin(np.pi * tt / 0.25) ** 2
        r = replay_stance(model, synthetic={"times": tt, "Fx": 0.2 * Fy * np.cos(np.pi * tt / 0.25), "Fy": Fy,
                                            "Mz": np.zeros_like(tt), "phi": np.deg2rad(10.0) * (0.5 - tt / 0.25)},
                          toe_config=toe_cfg, shoe_width_m=width)
        source = "synthetic stance"
    t_replay = time.perf_counter() - t
    rows = []
    for i, fr in enumerate(r.frames):
        d = fr.contact.diagnostics
        rows.append({"p": float(r.times[i]), "Fx": d["Fx_fixed"], "Fy": d["Fy_fixed"], "Mz": d["Mz_shoe"],
                     "cop": d["cop_fixed_x"], "theta": fr.theta_deg, "n_active": d["n_active"],
                     "first": d["first_active"], "last": d["last_active"], "n_patches": d["n_patches"],
                     "min_gap": d["min_gap"], "min_reaction": d["min_reaction"],
                     "plate_v_mid": float(plate_fields(model, fr.contact)["v"][model.plate.n_nodes // 2]),
                     "active": fr.contact.active})
    act = np.array([q["active"] for q in rows])
    tr = np.flatnonzero(np.any(act[1:] != act[:-1], axis=1))
    plot_sweep(rows, tr, f"gait replay ({source})", "time (s)", out / "gait_replay.png")
    # Time cannot be bisected (sampled data): compare the step change across each
    # active-set change with the neighbouring steps instead.
    step_ratio = {}
    for key in ("Mz", "theta", "plate_v_mid"):
        y = np.array([q[key] for q in rows], float)
        dy = np.abs(np.diff(y))
        ratios = [dy[k] / max(np.max(np.delete(dy[max(k - 3, 0): k + 4], k - max(k - 3, 0))), 1e-300) for k in tr]
        step_ratio[key] = float(max(ratios)) if ratios else 0.0
    checks = [direct_check(direct, model, fr.contact) for fr in r.frames[:: max(1, len(r.frames) // 40)]]
    summary["cases"]["gait_replay"] = {
        "source": source,
        "n_frames": int(r.times.size),
        "n_transitions": int(tr.size),
        "all_frames_valid": bool(r.frame_valid.all()),
        "max_patches": int(max(q["n_patches"] for q in rows)),
        "max_step_ratio_at_active_set_changes": step_ratio,
        "time_per_sample_ms": 1e3 * t_replay / max(r.times.size, 1),
        "median_sample_ms": 1e3 * float(np.median(r.timings["total_s"])),
        "max_toe_residual_Nm": float(np.nanmax(np.abs(r.toe_equilibrium_residual_Nm))),
        "max_penetration_m": float(np.nanmax(r.contact_scalars["max_penetration"])),
        "max_complementarity": float(np.nanmax(r.contact_scalars["max_complementarity"])),
        **{f"max_{k}": float(max(c[k] for c in checks)) for k in checks[0] if k != "direct_min_gap_over_h"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
