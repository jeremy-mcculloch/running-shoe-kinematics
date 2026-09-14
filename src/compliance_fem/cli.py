"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compliance_fem.compliance import compute_compliance, save_compliance_npz
from compliance_fem.config import LayeredPlateConfig, ProblemConfig
from compliance_fem.geometry import generate_rectangular_mesh
from compliance_fem.layered_geometry import generate_layered_mesh
from compliance_fem.plotting import plot_compliance_heatmap, save_all_plots
from compliance_fem.validation import run_standard_validations

LAYERED_REQUIRED = (
    "h1_heel",
    "h1_toe",
    "h2_heel",
    "h2_toe",
    "E1",
    "nu1",
    "E_heel",
    "E_toe",
    "nu2",
    "EI_plate",
    "ny1",
    "ny2",
)


def _save_mesh_vtk(mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))


def _add_layered_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h1-heel", dest="h1_heel", type=float, default=None)
    parser.add_argument("--h1-toe", dest="h1_toe", type=float, default=None)
    parser.add_argument("--h2-heel", dest="h2_heel", type=float, default=None)
    parser.add_argument("--h2-toe", dest="h2_toe", type=float, default=None)
    parser.add_argument("--E1", dest="E1", type=float, default=None)
    parser.add_argument("--nu1", dest="nu1", type=float, default=None)
    parser.add_argument("--E-heel", dest="E_heel", type=float, default=None)
    parser.add_argument("--E-toe", dest="E_toe", type=float, default=None)
    parser.add_argument("--nu2", dest="nu2", type=float, default=None)
    parser.add_argument("--EI-plate", dest="EI_plate", type=float, default=None)
    parser.add_argument("--ny1", type=int, default=None)
    parser.add_argument("--ny2", type=int, default=None)
    parser.add_argument("--element-order", dest="element_order", type=int, choices=[1, 2], default=None)


def _layered_config(args: argparse.Namespace) -> LayeredPlateConfig:
    missing = [name for name in LAYERED_REQUIRED if getattr(args, name) is None]
    if args.element_order is None:
        missing.append("element_order")
    if missing:
        raise SystemExit(
            "layered-plate geometry requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    return LayeredPlateConfig(
        L=args.L,
        h1_heel=args.h1_heel,
        h1_toe=args.h1_toe,
        h2_heel=args.h2_heel,
        h2_toe=args.h2_toe,
        E1=args.E1,
        nu1=args.nu1,
        E_heel=args.E_heel,
        E_toe=args.E_toe,
        nu2=args.nu2,
        EI_plate=args.EI_plate,
        nx=args.nx,
        ny1=args.ny1,
        ny2=args.ny2,
        element_order=args.element_order,
        element_type=args.element_type,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute plane-strain compliance matrices.")
    parser.add_argument(
        "--geometry",
        choices=["rectangle", "layered-plate"],
        default="rectangle",
    )
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=0.5)
    parser.add_argument("--E", type=float, default=1.0e6)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--order", type=int, default=1, choices=[1, 2])
    parser.add_argument("--element-type", choices=["quad", "tri"], default="quad")
    parser.add_argument("--output", type=Path, default=None)
    _add_layered_arguments(parser)
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = Path(
            "outputs/layered_plate" if args.geometry == "layered-plate" else "outputs/rectangle"
        )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.geometry == "layered-plate":
        config: ProblemConfig | LayeredPlateConfig = _layered_config(args)
        mesh_data = generate_layered_mesh(config, output_msh=output_dir / "mesh.msh")
        result = compute_compliance(config, mesh_data=mesh_data)
        reports = []
        save_compliance_npz(result, output_dir / "compliance_results.npz")
        summary = {
            "geometry_type": config.geometry_type,
            "rigid_mode_error": result.rigid_mode_error,
            "rigid_constraint_residual": result.rigid_constraint_residual,
            "reciprocity_error": result.reciprocity_error,
            "symmetry_errors": result.symmetry_errors,
            "solve_residuals": result.solve_residuals,
            "inextensibility_residuals": result.inextensibility_residuals,
            "gauge_residuals": result.gauge_residuals,
            "number_of_plate_nodes": result.n_plate_nodes,
            "number_of_plate_constraints": result.n_lambda,
            "constraint_rank": result.constraint_rank,
        }
        plot_compliance_heatmap(
            result.Cbt_force, r"$C_{bt}^F$", output_dir / "Cbt_force_heatmap.png"
        )
        plot_compliance_heatmap(
            result.Cbb_force, r"$C_{bb}^F$", output_dir / "Cbb_force_heatmap.png"
        )
        _save_mesh_vtk(mesh_data.mesh, output_dir / "mesh.vtk")
    else:
        config = ProblemConfig(
            L=args.L,
            H=args.H,
            E=args.E,
            nu=args.nu,
            nx=args.nx,
            ny=args.ny,
            order=args.order,
            element_type=args.element_type,
        )
        mesh_data = generate_rectangular_mesh(config, output_msh=output_dir / "mesh.msh")
        result = compute_compliance(config, mesh_data=mesh_data)
        reports = run_standard_validations(result)
        save_compliance_npz(result, output_dir / "compliance_results.npz")
        summary = {
            "geometry_type": config.geometry_type,
            "rigid_mode_error": result.rigid_mode_error,
            "reciprocity_error": result.reciprocity_error,
            "symmetry_errors": result.symmetry_errors,
            "solve_residuals": result.solve_residuals,
            "validation": [
                {
                    "case": r.case_name,
                    "displacement_rel_error": r.displacement_rel_error,
                    "force_balance_error": r.force_balance_error,
                    "moment_balance_error": r.moment_balance_error,
                }
                for r in reports
            ],
        }
        _save_mesh_vtk(mesh_data.mesh, output_dir / "mesh.vtk")
        save_all_plots(result, reports, output_dir)

    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Results written to {output_dir}")
    np.set_printoptions(precision=8, suppress=True, linewidth=120)
    print(f"Top nodal x ({len(result.x_top)} nodes):\n{result.x_top}")
    print(f"Bottom nodal x ({len(result.x_bottom)} nodes):\n{result.x_bottom}")


if __name__ == "__main__":
    main()
