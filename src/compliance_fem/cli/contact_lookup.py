"""CLI: build the FEM and contact lookup described by a model setup JSON config."""

from __future__ import annotations

import argparse
from pathlib import Path

from compliance_fem.contact.model_setup import ensure_lookup_exists


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the FEM compliance and the single-contiguous-interval contact lookup "
            "described by a model setup JSON config. Every setting comes from the config."
        )
    )
    parser.add_argument("config", type=Path, help="Model setup JSON config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the lookup in lookup.output_dir was already built from this config.",
    )
    parser.add_argument(
        "--save-fem",
        action="store_true",
        help=(
            "Also write the FEM results to <output>/fem: compliance_results.npz, mesh.msh, "
            "mesh.vtk, compliance heatmaps and fem_summary.json."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not write the lookup diagnostic plots or the compliance heatmaps.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress reporting.")
    args = parser.parse_args(argv)
    ensure_lookup_exists(
        args.config, force=args.force, save_fem=args.save_fem, skip_plots=args.skip_plots, quiet=args.quiet
    )
