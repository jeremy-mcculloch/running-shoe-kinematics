"""CLI for generating single-contiguous-interval contact lookup tables."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from compliance_fem.compliance import (
    compute_compliance,
    is_layered_compliance_npz,
    is_measured_compliance_npz,
    layered_config_from_compliance_npz,
    measured_config_from_compliance_npz,
)
from compliance_fem.cli import add_measured_arguments, measured_setup_from_args
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_lookup import (
    from_compliance_result,
    generate_contact_lookup,
    load_force_compliance,
    save_contact_lookup,
)
from compliance_fem.contact_plotting import save_lookup_plots
from compliance_fem.contact_topology import ContactType
from compliance_fem.measured_config_file import (
    DEFAULT_KAPPA,
    DEFAULT_RECIPROCITY_TOL,
    LookupSettings,
    MeasuredSoleSetup,
    RuntimeSettings,
    attach_setup_metadata,
    save_measured_sole_config,
    with_lookup_overrides,
)
from compliance_fem.toe_spring import ToeSpringConfig

DEFAULT_A = 0.2262
DEFAULT_OUTPUT = Path("outputs/contact_lookup")


def default_setup_for(cfg) -> MeasuredSoleSetup:
    """Setup wrapping a measured config recovered from a compliance NPZ."""
    return MeasuredSoleSetup(sole=cfg, lookup=LookupSettings(), toe_spring=ToeSpringConfig(), runtime=RuntimeSettings())


def _progress(done: int, total: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    eta = elapsed / max(done, 1) * (total - done)
    sys.stdout.write(f"\r  intervals {done}/{total}  elapsed {elapsed:7.1f} s  eta {eta:7.1f} s")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a single-contiguous-interval contact lookup table.")
    parser.add_argument(
        "--compliance",
        type=Path,
        default=None,
        help="Path to compliance_results.npz (nodal-force blocks).",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=None,
        help=f"Softplus transition location in m (default {DEFAULT_A}; measured sole: from --config, else 0.78 L).",
    )
    parser.add_argument(
        "--kappa", type=float, default=None, help=f"Softplus sharpness (default {DEFAULT_KAPPA:g} or from --config)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output directory (default {DEFAULT_OUTPUT}, or lookup.output_dir from --config).",
    )
    parser.add_argument(
        "--include-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignored: every interval 0 <= i <= j < N_b is always enumerated. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--reciprocity-tol",
        type=float,
        default=None,
        help=(
            f"Maximum relative C symmetry/reciprocity error accepted before symmetrization "
            f"(default {DEFAULT_RECIPROCITY_TOL:g} or from --config)."
        ),
    )
    parser.add_argument(
        "--store-fields",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Also store the per-node bases (bottom displacements, reactions, top forces, "
            "plate fields) for every interval. Default: only per-record scalars are "
            "stored and nodal fields are recomputed on demand (or lookup.store_nodal_fields "
            "from --config)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress reporting.")
    parser.add_argument(
        "--geometry",
        choices=["rectangle", "measured-sole"],
        default="rectangle",
        help="In-process geometry when --compliance is not given.",
    )
    # Optional in-process regeneration when no NPZ is provided.
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=0.1)
    parser.add_argument("--E", type=float, default=1.0e6)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--order", type=int, default=1, choices=[1, 2])
    parser.add_argument("--sole-rocker-height", type=float, default=0.0)
    parser.add_argument("--sole-rocker-apex", type=float, default=0.5)
    add_measured_arguments(parser)
    args = parser.parse_args(argv)
    if args.config is not None:
        if args.compliance is not None:
            parser.error("--config and --compliance are mutually exclusive.")
        args.geometry = "measured-sole"

    fem_result = None
    setup = None
    if args.compliance is not None:
        if is_measured_compliance_npz(args.compliance):
            cfg = measured_config_from_compliance_npz(args.compliance)
            fem_result = compute_compliance(cfg)
            blocks = from_compliance_result(fem_result)
            setup = default_setup_for(cfg)
        elif is_layered_compliance_npz(args.compliance):
            # Recompute FEM in-process so the factorization is available for plate recovery.
            cfg = layered_config_from_compliance_npz(args.compliance)
            fem_result = compute_compliance(cfg)
            blocks = from_compliance_result(fem_result)
        else:
            blocks = load_force_compliance(args.compliance)
    elif args.geometry == "measured-sole":
        setup = measured_setup_from_args(args)
        fem_result = compute_compliance(setup.sole)
        blocks = from_compliance_result(fem_result)
    else:
        config = ProblemConfig(
            L=args.L,
            H=args.H,
            E=args.E,
            nu=args.nu,
            nx=args.nx,
            ny=args.ny,
            order=args.order,
            sole_rocker_height=args.sole_rocker_height,
            sole_rocker_apex=args.sole_rocker_apex,
        )
        blocks = from_compliance_result(compute_compliance(config))

    if setup is not None:
        setup = with_lookup_overrides(
            setup, a_m=args.a, kappa=args.kappa, reciprocity_tol=args.reciprocity_tol, output_dir=args.output,
            store_nodal_fields=args.store_fields,
        )
        a, kappa, rtol = setup.a, setup.kappa, setup.lookup.reciprocity_tol
        output = setup.output_dir or DEFAULT_OUTPUT
        store_fields = setup.lookup.store_nodal_fields
    else:
        a = DEFAULT_A if args.a is None else args.a
        kappa = DEFAULT_KAPPA if args.kappa is None else args.kappa
        rtol = DEFAULT_RECIPROCITY_TOL if args.reciprocity_tol is None else args.reciprocity_tol
        output = args.output or DEFAULT_OUTPUT
        store_fields = bool(args.store_fields)
    args.output = Path(output)

    t0 = time.perf_counter()
    lookup = generate_contact_lookup(
        blocks,
        a=a,
        kappa=kappa,
        include_endpoints=args.include_endpoints,
        reciprocity_tol=rtol,
        fem_result=fem_result,
        progress=None if args.quiet else (lambda d, n: _progress(d, n, t0)),
        store_fields=store_fields,
    )
    if setup is not None:
        attach_setup_metadata(lookup, setup)
    npz_path = save_contact_lookup(lookup, args.output)
    save_lookup_plots(lookup, args.output)
    if setup is not None:
        save_measured_sole_config(setup, args.output / "measured_sole_config.json")

    n_plate_nodes = (
        int(lookup.plate_node_ids.size)
        if lookup.has_plate_response and lookup.plate_node_ids is not None
        else 0
    )
    valid = lookup.valid_mask
    label_counts = {}
    for kind in (ContactType.HEEL, ContactType.INTERIOR, ContactType.TOE, ContactType.FULL):
        rows = lookup.rows_for(kind)
        label_counts[kind.value] = {"theoretical": int(rows.size), "valid": int(np.count_nonzero(valid[rows]))}
    n_b = int(lookup.n_bottom_nodes)
    summary = {
        "contact_set_model": "single_contiguous_interval",
        "n_bottom_nodes": n_b,
        "n_records_theoretical": n_b * (n_b + 1) // 2,
        "n_records": int(lookup.n_records),
        "n_valid_records": int(np.count_nonzero(valid)),
        "label_counts": label_counts,
        "rejection_counts": lookup.rejection_summary(),
        "build_time_s": float(lookup.build_time_s),
        "file_size_bytes": int(npz_path.stat().st_size),
        "softplus_a": lookup.softplus_a,
        "softplus_kappa": lookup.softplus_kappa,
        "reciprocity_error": lookup.reciprocity_error,
        "max_solve_residual": float(np_max_safe(lookup.solve_residuals[valid])),
        "has_plate_response": bool(lookup.has_plate_response),
        "nodal_fields_stored": bool(lookup.nodal_fields_stored),
        "n_plate_nodes": n_plate_nodes,
        "EI_plate": float(lookup.EI_plate) if lookup.has_plate_response else None,
        "schema_version": int(lookup.schema_version),
        "geometry_type": str(lookup.geometry_type),
        "geometry": lookup.geometry_metadata or None,
        "measured_config_source": None if setup is None else setup.source_path,
        "measured_setup_fingerprint": None if setup is None else setup.fingerprint(),
        "output": str(npz_path),
    }
    with (args.output / "generation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"Contact lookup written to {args.output}")
    counts = ", ".join(f"{k}={v['valid']}/{v['theoretical']}" for k, v in label_counts.items())
    print(
        f"N_b={n_b}  records valid/theoretical={summary['n_valid_records']}/{summary['n_records_theoretical']} "
        f"({counts})  build={summary['build_time_s']:.1f} s  size={summary['file_size_bytes'] / 1e6:.1f} MB  "
        f"max_solve_residual={summary['max_solve_residual']:.3e}  plate_response={summary['has_plate_response']}  "
        f"nodal_fields={'stored' if lookup.nodal_fields_stored else 'on demand'}"
        + (f"  n_plate={n_plate_nodes}" if lookup.has_plate_response else "")
    )
    if summary["rejection_counts"]:
        print(f"rejections: {summary['rejection_counts']}")


def np_max_safe(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


if __name__ == "__main__":
    main()
