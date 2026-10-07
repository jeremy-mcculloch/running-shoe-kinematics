"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compliance_fem.compliance import compute_compliance, save_compliance_npz
from compliance_fem.config import (
    FOAM_MATERIALS,
    LayeredPlateConfig,
    MeasuredSoleConfig,
    ProblemConfig,
)
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


def _add_sole_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sole-rocker-height",
        dest="sole_rocker_height",
        type=float,
        default=0.0,
        help="Parabolic rocker height of the bottom surface (0 = flat sole).",
    )
    parser.add_argument(
        "--sole-rocker-apex",
        dest="sole_rocker_apex",
        type=float,
        default=0.5,
        help="Rocker apex location as a fraction of L.",
    )


DEFAULT_MEASURED_CSV = Path("data/geometry/sole_geometry_normalized.csv")


def add_measured_arguments(parser: argparse.ArgumentParser) -> None:
    """Measured-sole options (``--geometry measured-sole``)."""
    group = parser.add_argument_group("measured-sole geometry")
    group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Measured-sole JSON config (implies --geometry measured-sole); explicit flags override it.",
    )
    group.add_argument(
        "--geometry-csv",
        type=Path,
        default=None,
        help=f"Normalized sole CSV (default {DEFAULT_MEASURED_CSV}).",
    )
    group.add_argument(
        "--shoe-length-mm",
        type=float,
        default=None,
        help="Projected heel-bottom-to-toe-tip length in mm (required; not the arc length).",
    )
    group.add_argument("--upper-foam", choices=FOAM_MATERIALS, default=None)
    group.add_argument("--lower-foam", choices=FOAM_MATERIALS, default=None)
    group.add_argument("--mesh-size-mm", type=float, default=None, help="Target element size (mm).")
    group.add_argument("--toe-refinement", type=float, default=None)
    group.add_argument("--heel-corner-refinement", type=float, default=None)
    group.add_argument("--interface-refinement", type=float, default=None)
    group.add_argument("--plate-end-refinement", type=float, default=None)
    if not any(a.dest == "EI_plate" for a in parser._actions):
        group.add_argument("--EI-plate", dest="EI_plate", type=float, default=None)


def measured_setup_from_args(args: argparse.Namespace):
    """Measured-sole setup from ``--config`` (if given) with explicit flags taking precedence."""
    from compliance_fem.measured_config_file import (
        default_setup,
        load_measured_sole_config,
        with_sole_overrides,
    )

    if getattr(args, "config", None) is not None:
        setup = load_measured_sole_config(args.config)
    else:
        if args.shoe_length_mm is None:
            raise SystemExit(
                "--geometry measured-sole requires --shoe-length-mm (projected heel-to-toe length in mm) "
                "or --config."
            )
        setup = default_setup(args.geometry_csv or DEFAULT_MEASURED_CSV, args.shoe_length_mm)
    return with_sole_overrides(
        setup,
        shoe_length_mm=args.shoe_length_mm,
        geometry_csv=None if args.geometry_csv is None else str(args.geometry_csv),
        upper_foam_material=args.upper_foam,
        lower_foam_material=args.lower_foam,
        EI_plate=getattr(args, "EI_plate", None),
        mesh_size=None if args.mesh_size_mm is None else args.mesh_size_mm / 1000.0,
        toe_refinement=args.toe_refinement,
        heel_corner_refinement=args.heel_corner_refinement,
        interface_refinement=args.interface_refinement,
        plate_end_refinement=args.plate_end_refinement,
    )


def measured_config_from_args(args: argparse.Namespace) -> MeasuredSoleConfig:
    return measured_setup_from_args(args).sole


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
        sole_rocker_height=args.sole_rocker_height,
        sole_rocker_apex=args.sole_rocker_apex,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute plane-strain compliance matrices.")
    parser.add_argument(
        "--geometry",
        choices=["rectangle", "layered-plate", "measured-sole"],
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
    _add_sole_arguments(parser)
    add_measured_arguments(parser)
    args = parser.parse_args(argv)
    if args.config is not None:
        args.geometry = "measured-sole"

    if args.output is None:
        args.output = Path(
            {
                "layered-plate": "outputs/layered_plate",
                "measured-sole": "outputs/measured_sole",
            }.get(args.geometry, "outputs/rectangle")
        )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.geometry == "measured-sole":
        from compliance_fem.measured_mesh import generate_measured_mesh

        from compliance_fem.measured_config_file import save_measured_sole_config

        setup = measured_setup_from_args(args)
        config = setup.sole
        mesh_data = generate_measured_mesh(config, output_msh=output_dir / "mesh.msh")
        result = compute_compliance(config, mesh_data=mesh_data)
        reports = []
        save_compliance_npz(result, output_dir / "compliance_results.npz")
        save_measured_sole_config(setup, output_dir / "measured_sole_config.json")
        summary = {
            "geometry_type": config.geometry_type,
            "config_source": setup.source_path,
            "geometry": result.geometry_metadata,
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
        plot_compliance_heatmap(result.Cbt_force, r"$C_{bt}^F$", output_dir / "Cbt_force_heatmap.png")
        plot_compliance_heatmap(result.Cbb_force, r"$C_{bb}^F$", output_dir / "Cbb_force_heatmap.png")
        _save_mesh_vtk(mesh_data.mesh, output_dir / "mesh.vtk")
    elif args.geometry == "layered-plate":
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
            sole_rocker_height=args.sole_rocker_height,
            sole_rocker_apex=args.sole_rocker_apex,
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
        json.dump(summary, fh, indent=2, default=str)

    print(f"Results written to {output_dir}")
    np.set_printoptions(precision=8, suppress=True, linewidth=120)
    print(f"Top nodal x ({len(result.x_top)} nodes):\n{result.x_top}")
    print(f"Bottom nodal x ({len(result.x_bottom)} nodes):\n{result.x_bottom}")


if __name__ == "__main__":
    main()
