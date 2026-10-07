"""CLI for querying an interval contact lookup table with a raw coefficient vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compliance_fem.contact_lookup import load_contact_lookup
from compliance_fem.contact_plotting import plot_query_diagnostics
from compliance_fem.contact_query import select_candidate
from compliance_fem.contact_topology import ContactType


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query an interval contact lookup with a raw runtime coefficient vector. "
            "This is a lookup-level debug path; the force-controlled runtime "
            "(phi, theta, Fx, Fy) lives in the GUI and force_control."
        )
    )
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs=5,
        required=True,
        metavar=("ALPHA", "D_AX", "D_AY", "R_X", "R_Y"),
        help=(
            "gamma in saved basis order (top_shape_alpha, contact_translation_x, "
            "contact_translation_y, contact_rotation_x, contact_rotation_y), i.e. "
            "[tan(theta), d_ax, d_ay, cos(varphi)-1, -sin(varphi)]. The curved-sole "
            "closure column always has coefficient 1."
        ),
    )
    parser.add_argument(
        "--contact-type",
        choices=[t.value for t in ContactType],
        default=None,
        help="Report the best record within one topology label only (default: all records).",
    )
    parser.add_argument("--tau-g", type=float, default=0.0)
    parser.add_argument("--tau-R", type=float, default=0.0)
    parser.add_argument("--fy-tol", type=float, default=1e-14)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    lookup = load_contact_lookup(args.lookup)
    selected = select_candidate(lookup, args.coefficients, tau_g=args.tau_g, tau_R=args.tau_R, fy_tol=args.fy_tol)

    output_dir = args.output or Path(args.lookup).parent / "query"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_query_diagnostics(lookup, selected.profile, selected, output_dir)

    family = None
    if args.contact_type is not None:
        rows = np.intersect1d(lookup.rows_for(ContactType(args.contact_type)), lookup.valid_rows)
        best = int(rows[int(np.argmin(selected.profile.violation[rows]))])
        family = {
            "contact_type": args.contact_type,
            "n_records": int(rows.size),
            "best_row": best,
            "best_interval": [int(lookup.contact_start_index[best]), int(lookup.contact_end_index[best])],
            "best_anchor_x": float(lookup.contact_anchor_reference_x[best]),
            "best_violation": float(selected.profile.violation[best]),
            "best_admissible": bool(selected.profile.admissible[best]),
        }

    summary = {
        "coefficients": list(map(float, args.coefficients)),
        "n_records": int(lookup.n_records),
        "selected_row": selected.candidate_row,
        "selected_contact_type": selected.contact_type,
        "selected_interval": [selected.contact_start_index, selected.contact_end_index],
        "selected_anchor_x": selected.anchor_x,
        "exactly_admissible": selected.exactly_admissible,
        "n_admissible": int(selected.admissible_rows.size),
        "Fy": selected.Fy,
        "M": selected.M,
        "x_ce": selected.x_ce,
        "violation": selected.violation,
        "family": family,
    }
    with (output_dir / "query_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
