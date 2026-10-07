"""Validate the measured carbon-plated sole end to end at a given shoe length.

Builds the measured geometry and mesh, the compliance blocks and the full
interval lookup, then compares stored lookup records against direct sparse FEM
solves for heel, interior, toe and full contact intervals (all six affine
columns). Writes ``validation.json`` and prints a summary.

    python examples/measured_sole_validation.py --shoe-length-mm 270
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import MeasuredSoleConfig
from compliance_fem.contact_direct_fem import solve_direct_fem_contact, standard_validation_intervals
from compliance_fem.contact_lookup import (
    AFFINE_COLUMN_NAMES,
    N_AFFINE_COLUMNS,
    from_compliance_result,
    generate_contact_lookup,
)
from compliance_fem.contact_topology import interval_row

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "geometry" / "sole_geometry_normalized.csv"


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / denom)


def _balance(x, y, fx, fy) -> tuple[float, float, float]:
    return float(np.sum(fx)), float(np.sum(fy)), float(np.sum(x * fy - y * fx))


def compare_record(result, lookup, spec, a, kappa) -> dict:
    """Lookup record vs direct FEM for every affine column of one interval."""
    row = interval_row(spec.start, spec.end, spec.n_bottom)
    if not lookup.valid_mask[row]:
        raise RuntimeError(f"Interval {spec.start}..{spec.end} is not a valid lookup record.")
    free, contact = spec.sets()
    xt, yt = np.asarray(result.x_top), np.asarray(result.y_top)
    xb, yb = np.asarray(result.x_bottom), np.asarray(result.y_bottom)
    cols = []
    f = {name: value[0] for name, value in lookup.record_fields([row]).items()}
    for k in range(N_AFFINE_COLUMNS):
        fem = solve_direct_fem_contact(result, spec, k, a, kappa)
        lk_ft_x = f["top_force_x"][k]
        lk_ft_y = f["top_force_y"][k]
        lk_rc_x = f["reaction_x"][k, contact]
        lk_rc_y = f["reaction_y"][k, contact]
        errs = {
            "top_force": _rel(np.r_[lk_ft_x, lk_ft_y], fem["f_t"]),
            "contact_reaction": _rel(np.r_[lk_rc_x, lk_rc_y], fem["r_c"]),
        }
        if free.size:
            lk_g = np.r_[f["bottom_u"][k, free], f["bottom_v"][k, free]]
            errs["free_bottom_displacement"] = _rel(lk_g, fem["g_f"])
        # Equilibrium of all external nodal forces (top + contact) about the origin.
        Fx, Fy, Mz = _balance(
            np.r_[xt, xb[contact]], np.r_[yt, yb[contact]], np.r_[lk_ft_x, lk_rc_x], np.r_[lk_ft_y, lk_rc_y]
        )
        scale = max(float(np.sum(np.abs(np.r_[lk_ft_x, lk_ft_y]))), 1e-30)
        L = float(result.config.L)
        cols.append(
            {
                "column": AFFINE_COLUMN_NAMES[k],
                **errs,
                "max_rel_error": max(errs.values()),
                "force_balance_rel": float(np.hypot(Fx, Fy) / scale),
                "moment_balance_rel": float(abs(Mz) / (scale * L)),
                "fem_free_traction_rel": float(
                    np.linalg.norm(fem["r_free"]) / max(np.linalg.norm(fem["f_t"]), 1e-30)
                ),
            }
        )
    return {
        "start": int(spec.start),
        "end": int(spec.end),
        "x_start_m": float(xb[spec.start]),
        "x_end_m": float(xb[spec.end]),
        "n_contact": int(contact.size),
        "max_rel_error": max(c["max_rel_error"] for c in cols),
        "max_force_balance_rel": max(c["force_balance_rel"] for c in cols),
        "max_moment_balance_rel": max(c["moment_balance_rel"] for c in cols),
        "columns": cols,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--shoe-length-mm", type=float, default=270.0)
    ap.add_argument("--mesh-size-mm", type=float, default=3.0)
    ap.add_argument("--softplus-a-frac", type=float, default=0.78, help="a as a fraction of L")
    ap.add_argument("--kappa", type=float, default=160.0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "measured_sole_validation"))
    args = ap.parse_args()

    cfg = MeasuredSoleConfig(
        shoe_length_mm=args.shoe_length_mm,
        geometry_csv=args.geometry_csv,
        mesh_size=args.mesh_size_mm * 1e-3,
    )
    t0 = time.perf_counter()
    result = compute_compliance(cfg)
    t_fem = time.perf_counter() - t0
    meta = result.geometry_metadata
    a = args.softplus_a_frac * cfg.L

    t0 = time.perf_counter()
    lookup = generate_contact_lookup(from_compliance_result(result), a=a, kappa=args.kappa, fem_result=result)
    t_lookup = time.perf_counter() - t0

    specs = standard_validation_intervals(result.x_bottom, cfg.L)
    comparisons = {name: compare_record(result, lookup, spec, a, args.kappa) for name, spec in specs.items()}

    valid = lookup.valid_mask
    report = {
        "shoe_length_mm": cfg.shoe_length_mm,
        "geometry_csv": str(args.geometry_csv),
        "physical_landmarks_m": meta["physical_landmarks"],
        "region_areas_m2": meta["region_areas_m2"],
        "total_area_m2": meta["total_area_m2"],
        "plate_arc_length_m": meta["plate_arc_length_m"],
        "projection_distances_m": meta["projection_distances_m"],
        "corner_interior_angles_deg": meta["corner_interior_angles_deg"],
        "mesh_quality": meta["mesh_quality"],
        "n_top_selector_nodes": int(len(result.x_top)),
        "n_bottom_nodes": int(len(result.x_bottom)),
        "n_plate_nodes": int(len(result.plate_node_ids)),
        "shared_toe_node_id": int(result.shared_toe_node_id),
        "compliance": {
            "reciprocity_error": float(result.reciprocity_error),
            "rigid_mode_error": float(result.rigid_mode_error),
            "inextensibility_residuals": {k: float(v) for k, v in result.inextensibility_residuals.items()},
            "constraint_rank": int(result.constraint_rank),
            "solve_residuals": {k: float(v) for k, v in result.solve_residuals.items()},
            "fem_seconds": t_fem,
        },
        "lookup": {
            "softplus_a_m": a,
            "softplus_kappa": args.kappa,
            "n_records": int(lookup.n_records),
            "n_valid": int(np.count_nonzero(valid)),
            "reciprocity_error": float(lookup.reciprocity_error),
            "max_force_equilibrium_residual": float(np.nanmax(lookup.force_equilibrium_residuals[valid])),
            "max_moment_equilibrium_residual": float(np.nanmax(lookup.moment_equilibrium_residuals[valid])),
            "max_plate_bp_residual": float(np.nanmax(lookup.plate_bp_residual[valid]))
            if lookup.plate_bp_residual is not None
            else None,
            "seconds": t_lookup,
        },
        "direct_fem": comparisons,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(report, indent=2, default=float))

    q = report["mesh_quality"]
    print(f"shoe length {cfg.shoe_length_mm:g} mm; FEM {t_fem:.1f} s; lookup {t_lookup:.1f} s")
    print(f"mesh: {q.get('n_nodes')} nodes, {q.get('n_elements')} triangles, min angle {q.get('min_angle_deg'):.1f} deg")
    print(
        f"selectors: top {report['n_top_selector_nodes']}, bottom {report['n_bottom_nodes']}, "
        f"plate {report['n_plate_nodes']}; valid {report['lookup']['n_valid']}/{report['lookup']['n_records']}"
    )
    c = report["compliance"]
    print(
        f"reciprocity {c['reciprocity_error']:.2e}, rigid {c['rigid_mode_error']:.2e}, "
        f"inextensibility {max(c['inextensibility_residuals'].values()):.2e}, rank {c['constraint_rank']}"
    )
    for name, cmp in comparisons.items():
        print(
            f"  {name:26s} {cmp['start']:3d}..{cmp['end']:3d}  lookup-vs-FEM {cmp['max_rel_error']:.2e}  "
            f"force bal {cmp['max_force_balance_rel']:.1e}  moment bal {cmp['max_moment_balance_rel']:.1e}"
        )
    print(f"wrote {out / 'validation.json'}")


if __name__ == "__main__":
    main()
