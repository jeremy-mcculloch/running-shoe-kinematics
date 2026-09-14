"""CLI for generating contact-edge lookup tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compliance_fem.compliance import (
    compute_compliance,
    is_layered_compliance_npz,
    layered_config_from_compliance_npz,
)
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_lookup import (
    from_compliance_result,
    generate_contact_lookup,
    load_force_compliance,
    save_contact_lookup,
)
from compliance_fem.contact_plotting import save_lookup_plots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a contact-edge lookup table.")
    parser.add_argument(
        "--compliance",
        type=Path,
        default=None,
        help="Path to compliance_results.npz (nodal-force blocks).",
    )
    parser.add_argument("--a", type=float, default=0.2262, help="Softplus transition location.")
    parser.add_argument("--kappa", type=float, default=160, help="Softplus sharpness.")
    parser.add_argument("--output", type=Path, default=Path("outputs/contact_lookup"))
    parser.add_argument(
        "--include-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated/ignored for schema v6: all heel/toe/full topologies are always "
            "enumerated. Kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--reciprocity-tol",
        type=float,
        default=1e-6,
        help="Maximum relative C symmetry/reciprocity error accepted before symmetrization.",
    )
    # Optional in-process regeneration when no NPZ is provided.
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=0.1)
    parser.add_argument("--E", type=float, default=1.0e6)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--order", type=int, default=1, choices=[1, 2])
    args = parser.parse_args(argv)

    fem_result = None
    if args.compliance is not None:
        if is_layered_compliance_npz(args.compliance):
            # Recompute FEM in-process so the factorization is available for plate recovery.
            cfg = layered_config_from_compliance_npz(args.compliance)
            fem_result = compute_compliance(cfg)
            blocks = from_compliance_result(fem_result)
        else:
            blocks = load_force_compliance(args.compliance)
    else:
        config = ProblemConfig(
            L=args.L,
            H=args.H,
            E=args.E,
            nu=args.nu,
            nx=args.nx,
            ny=args.ny,
            order=args.order,
        )
        result = compute_compliance(config)
        blocks = from_compliance_result(result)

    lookup = generate_contact_lookup(
        blocks,
        a=args.a,
        kappa=args.kappa,
        include_endpoints=args.include_endpoints,
        reciprocity_tol=args.reciprocity_tol,
        fem_result=fem_result,
    )
    npz_path = save_contact_lookup(lookup, args.output)
    save_lookup_plots(lookup, args.output)

    from compliance_fem.contact_topology import ContactType

    n_plate_nodes = (
        int(lookup.plate_node_ids.size)
        if lookup.has_plate_response and lookup.plate_node_ids is not None
        else 0
    )
    n_heel = int(lookup.rows_for(ContactType.HEEL).size)
    n_toe = int(lookup.rows_for(ContactType.TOE).size)
    n_full = int(lookup.rows_for(ContactType.FULL).size)
    summary = {
        "n_records": int(lookup.n_records),
        "n_heel": n_heel,
        "n_toe": n_toe,
        "n_full": n_full,
        "n_candidates": int(lookup.n_records),
        "softplus_a": lookup.softplus_a,
        "softplus_kappa": lookup.softplus_kappa,
        "reciprocity_error": lookup.reciprocity_error,
        "max_solve_residual": float(np_max_safe(lookup.solve_residuals)),
        "has_plate_response": bool(lookup.has_plate_response),
        "n_plate_nodes": n_plate_nodes,
        "EI_plate": float(lookup.EI_plate) if lookup.has_plate_response else None,
        "schema_version": int(lookup.schema_version),
        "output": str(npz_path),
    }
    with (args.output / "generation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Contact lookup written to {args.output}")
    print(
        f"records={summary['n_records']} "
        f"(heel={n_heel}, toe={n_toe}, full={n_full})  "
        f"max_solve_residual={summary['max_solve_residual']:.3e}  "
        f"plate_response={summary['has_plate_response']}"
        + (f"  n_plate={n_plate_nodes}" if lookup.has_plate_response else "")
    )


def np_max_safe(values) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


if __name__ == "__main__":
    main()
