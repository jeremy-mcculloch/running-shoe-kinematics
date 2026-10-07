"""CLI: chronological gait force replay with COP-inferred toe bending."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compliance_fem.contact_lookup import load_contact_lookup
from compliance_fem.gait.download import write_data_readme
from compliance_fem.gait.replay import replay_stance, save_replay_result
from compliance_fem.gait.wang_io import discover_trials, parse_wang_trial_name, read_opensim_mot, read_opensim_trc
from compliance_fem.toe_spring import TOE_MODELS, ToeSpringConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compliance-fem-gait-replay",
        description=(
            "Replay measured Fx, Fy and marker foot pitch through the contact "
            "lookup. Default toe model: passive toe spring (k=25 N·m/rad) solved "
            "exactly from Q_alpha,shoe->foot = dU/dalpha per contact record."
        ),
    )
    p.add_argument("--lookup", type=Path, help="Contact lookup directory or NPZ")
    p.add_argument("--trc", type=Path, default=None)
    p.add_argument("--mot", type=Path, default=None)
    p.add_argument("--markers-csv", type=Path, default=None)
    p.add_argument("--forces-csv", type=Path, default=None)
    p.add_argument("--c3d", type=Path, default=None, help="Reserved (not yet implemented)")
    p.add_argument("--dataset-root", type=Path, default=None, help="Discover TRC/MOT pairs")
    p.add_argument("--foot", choices=("left", "right"), default="right")
    p.add_argument("--force-plate", default="auto")
    p.add_argument("--stance-index", type=int, default=0)
    p.add_argument("--vertical-force-threshold", type=float, default=20.0)
    defaults = ToeSpringConfig()
    p.add_argument(
        "--toe-model",
        choices=TOE_MODELS,
        default=None,
        help="passive_spring (default): exact passive toe-spring equilibrium; "
        "passive_spring_elastic_equivalent: same against the visco elastic-equivalent "
        "wrench (approximation); prescribed_legacy / fit_cop_legacy: old θ rules.",
    )
    p.add_argument(
        "--theta-mode",
        choices=("fit-cop", "force-phi"),
        default=None,
        help="Deprecated legacy selector: fit-cop → fit_cop_legacy, force-phi → prescribed_legacy.",
    )
    p.add_argument("--toe-stiffness", type=float, default=defaults.toe_stiffness_Nm_per_rad,
                   help="Toe spring stiffness k (N·m/rad, >= 0).")
    p.add_argument("--toe-neutral-angle-rad", type=float, default=defaults.toe_neutral_angle_rad)
    p.add_argument("--toe-damping", type=float, default=defaults.toe_damping_Nms_per_rad,
                   help="Recorded only; does not enter the quasistatic equation.")
    p.add_argument("--toe-angle-min-deg", type=float, default=defaults.toe_angle_min_deg)
    p.add_argument("--toe-angle-max-deg", type=float, default=defaults.toe_angle_max_deg)
    p.add_argument("--toe-abs-tol", type=float, default=defaults.toe_equilibrium_abs_tol_Nm,
                   help="Absolute toe-equilibrium tolerance (N·m).")
    p.add_argument("--toe-rel-tol", type=float, default=defaults.toe_equilibrium_rel_tol)
    p.add_argument("--toe-scan-points", type=int, default=defaults.toe_root_scan_points)
    p.add_argument("--toe-low-force-threshold", type=float, default=defaults.toe_low_force_threshold_N,
                   help="Experimental |F| (N) below which the frame is flagged low-load.")
    p.add_argument("--theta-min-deg", type=float, default=0.0, help="Legacy models only.")
    p.add_argument("--theta-max-deg", type=float, default=45.0, help="Legacy models only.")
    p.add_argument("--cop-fit-min-force", type=float, default=50.0)
    p.add_argument(
        "--shoe-width-m",
        type=float,
        default=0.10,
        help="Effective shoe width (m); experimental forces/moments divided by this "
        "for plane-strain unit-thickness lookup (default 0.10).",
    )
    p.add_argument(
        "--position-units",
        choices=("auto", "m", "mm"),
        default="auto",
        help="Marker/COP position units (auto from metadata/magnitude).",
    )
    p.add_argument(
        "--moment-units",
        choices=("auto", "N-m", "N-mm"),
        default="auto",
        help="Moment units (auto from metadata/magnitude).",
    )
    p.add_argument(
        "--cop-fy-min",
        type=float,
        default=20.0,
        help="Lab |Fy| below which measured COP is marked invalid (N).",
    )
    p.add_argument("--cond-warn", type=float, default=1.0e8)
    p.add_argument(
        "--visco-model",
        choices=("elastic", "sls", "fractional", "fung"),
        default="elastic",
    )
    p.add_argument("--visco-config", type=Path, default=None, help="JSON parameter file")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gait_replay"))
    p.add_argument("--init-data-dir", type=Path, default=None, help="Write Wang download README")
    # Listing / discovery
    p.add_argument("--list-trials", action="store_true")
    p.add_argument("--list-markers", action="store_true")
    p.add_argument("--list-force-channels", action="store_true")
    p.add_argument("--list-stances", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.init_data_dir is not None:
        path = write_data_readme(args.init_data_dir)
        print(f"Wrote dataset instructions to {path}")
        return 0

    if args.dataset_root is not None and args.list_trials:
        trials = discover_trials(args.dataset_root)
        for t in trials:
            print(
                f"{t.stem}\tsubject={t.subject}\tcode={t.code}\t"
                f"pr1={t.is_preferred_r1}\ttrc={t.trc_path}\tmot={t.mot_path}"
            )
        return 0

    if args.list_markers:
        if args.trc is None:
            raise SystemExit("--list-markers requires --trc")
        trc = read_opensim_trc(args.trc)
        print("rate_hz", trc["rate_hz"], "units_original", trc["units_original"])
        print("markers:", ", ".join(trc["labels"]))
        return 0

    if args.list_force_channels:
        if args.mot is None:
            raise SystemExit("--list-force-channels requires --mot")
        mot = read_opensim_mot(args.mot)
        print("rate_hz", mot["rate_hz"])
        for name in mot["column_names"]:
            print(name)
        return 0

    if args.list_stances:
        if args.mot is None:
            raise SystemExit("--list-stances requires --mot")
        from compliance_fem.gait.stance import detect_stances
        from compliance_fem.gait.wang_io import extract_plate_wrench

        mot = read_opensim_mot(args.mot)
        plate = extract_plate_wrench(mot, args.force_plate)
        fy = plate["fy"]
        if float(np_max := __import__("numpy").max(fy)) < float(__import__("numpy").max(-fy)):
            fy = -fy
        for s in detect_stances(plate["times"], fy, threshold=args.vertical_force_threshold):
            print(
                f"stance[{s.index}] t=[{s.t0:.4f},{s.t1:.4f}] "
                f"peak_Fy={s.peak_fy:.3f} samples={s.i0}:{s.i1}"
            )
        return 0

    if args.c3d is not None:
        raise SystemExit("C3D input is reserved but not implemented yet; use --trc/--mot")

    if args.lookup is None or args.trc is None or args.mot is None:
        raise SystemExit("replay requires --lookup --trc --mot (or use listing flags)")

    visco_cfg = {}
    if args.visco_config is not None:
        visco_cfg = json.loads(Path(args.visco_config).read_text(encoding="utf-8"))

    lookup = load_contact_lookup(args.lookup)
    toe_cfg = ToeSpringConfig(
        toe_stiffness_Nm_per_rad=args.toe_stiffness,
        toe_neutral_angle_rad=args.toe_neutral_angle_rad,
        toe_damping_Nms_per_rad=args.toe_damping,
        toe_angle_min_deg=args.toe_angle_min_deg,
        toe_angle_max_deg=args.toe_angle_max_deg,
        toe_equilibrium_abs_tol_Nm=args.toe_abs_tol,
        toe_equilibrium_rel_tol=args.toe_rel_tol,
        toe_root_scan_points=args.toe_scan_points,
        toe_low_force_threshold_N=args.toe_low_force_threshold,
    )
    result = replay_stance(
        lookup,
        trc_path=args.trc,
        mot_path=args.mot,
        foot=args.foot,
        force_plate=str(args.force_plate),
        stance_index=args.stance_index,
        vertical_force_threshold=args.vertical_force_threshold,
        theta_mode=args.theta_mode,
        toe_model=args.toe_model,
        toe_config=toe_cfg,
        theta_min_deg=args.theta_min_deg,
        theta_max_deg=args.theta_max_deg,
        cop_fit_min_force=args.cop_fit_min_force,
        shoe_width_m=args.shoe_width_m,
        position_units=args.position_units,
        moment_units=args.moment_units,
        cop_fy_min=args.cop_fy_min,
        cond_warn=args.cond_warn,
        visco_model=args.visco_model,
        visco_config=visco_cfg,
    )
    paths = save_replay_result(result, args.output_dir)
    print("gait-replay complete")
    print(f"  frames: {result.times.size}")
    print(f"  theta range: [{float(np_nanmin(result.theta_deg)):.3g}, {float(np_nanmax(result.theta_deg)):.3g}] deg")
    print(f"  mean force residual: {float(__import__('numpy').nanmean(result.force_residual)):.3g}")
    print(f"  mean moment residual: {float(__import__('numpy').nanmean(result.moment_residual)):.3g}")
    print(f"  toe model: {result.toe_model}")
    summary = result.provenance.get("toe_summary")
    if summary:
        print(
            "  toe equilibrium: max |g| = {max_abs_residual_Nm:.3g} N·m, max roots/frame = "
            "{max_root_count}, no-root frames = {n_no_root_frames}, unstable = {n_unstable_frames}, "
            "low-load = {n_low_load_frames}, max spring energy = {max_spring_energy_J:.3g} J, "
            "unilateral-violation frames = {n_unilateral_violation_frames}".format(
                **{k: (float("nan") if v is None else v) for k, v in summary.items()}
            )
        )
    for w in result.warnings:
        print(f"  warning: {w}")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    # Also dump trial stem if parseable
    try:
        print("  trial:", parse_wang_trial_name(args.trc.stem))
    except ValueError:
        pass
    return 0


def np_nanmin(a):
    import numpy as np

    return np.nanmin(a)


def np_nanmax(a):
    import numpy as np

    return np.nanmax(a)


if __name__ == "__main__":
    sys.exit(main())
