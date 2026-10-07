"""Standalone CLI: viscoelastic force history → elastic force history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compliance_fem.viscoelasticity.factory import (
    ElasticConfig,
    FractionalConfig,
    FungConfig,
    SLSConfig,
    create_material,
)
from compliance_fem.viscoelasticity.io import read_force_history, write_force_history


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="visco-force-map",
        description=(
            "Map a viscoelastic generalized-force history F_VE(t) to the "
            "equivalent elastic force F_e(t) = J * dF_VE/dt using a stateful "
            "online material backend (batch CLI calls update() per sample)."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        choices=("elastic", "sls", "fractional", "fung"),
        help="Constitutive model",
    )
    p.add_argument("--input", "-i", required=True, type=Path, help="Input CSV: time,Fx,Fy")
    p.add_argument("--output", "-o", required=True, type=Path, help="Output CSV path")
    p.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Force component count (default: inferred from input CSV)",
    )

    # SLS
    p.add_argument("--g-inf", type=float, default=None, help="SLS long-term ratio g_∞")
    p.add_argument("--tau-r", type=float, default=None, help="SLS relaxation time τ_r")

    # Fractional
    p.add_argument("--E0", type=float, default=None, help="Fractional reference modulus E0")
    p.add_argument("--T", type=float, default=None, help="Fractional characteristic time T")
    p.add_argument("--alpha", type=float, default=None, help="Fractional order α ∈ (0,1)")
    p.add_argument("--num-modes", type=int, default=32, help="Prony / SoE mode count")
    p.add_argument("--tau-min", type=float, default=None, help="Approximation τ_min")
    p.add_argument("--tau-max", type=float, default=None, help="Approximation τ_max")

    # Fung
    p.add_argument("--C", type=float, default=None, help="Fung spectral intensity C")
    p.add_argument("--tau1", type=float, default=None, help="Fung spectrum lower bound τ1")
    p.add_argument("--tau2", type=float, default=None, help="Fung spectrum upper bound τ2")
    return p


def _config_from_args(args: argparse.Namespace, n_components: int):
    if args.model == "elastic":
        return ElasticConfig(n_components=n_components)
    if args.model == "sls":
        if args.g_inf is None or args.tau_r is None:
            raise SystemExit("SLS requires --g-inf and --tau-r")
        return SLSConfig(g_inf=args.g_inf, tau_r=args.tau_r, n_components=n_components)
    if args.model == "fractional":
        missing = [n for n, v in (("E0", args.E0), ("T", args.T), ("alpha", args.alpha)) if v is None]
        if missing:
            raise SystemExit("fractional requires " + ", ".join(f"--{m}" for m in missing))
        return FractionalConfig(
            E0=args.E0,
            T=args.T,
            alpha=args.alpha,
            num_modes=args.num_modes,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            n_components=n_components,
        )
    if args.model == "fung":
        missing = [n for n, v in (("C", args.C), ("tau1", args.tau1), ("tau2", args.tau2)) if v is None]
        if missing:
            raise SystemExit("fung requires " + ", ".join(f"--{m}" for m in missing))
        return FungConfig(
            C=args.C,
            tau1=args.tau1,
            tau2=args.tau2,
            num_modes=args.num_modes,
            n_components=n_components,
        )
    raise SystemExit(f"unknown model {args.model!r}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    times, FVE = read_force_history(args.input)
    n_comp = int(args.n_components) if args.n_components is not None else int(FVE.shape[1])
    if n_comp != FVE.shape[1]:
        raise SystemExit(
            f"--n-components={n_comp} does not match input force columns ({FVE.shape[1]})"
        )

    config = _config_from_args(args, n_comp)
    material = create_material(config)
    summary = material.parameter_summary()

    print("visco-force-map")
    print(f"  model:           {summary.get('model')}")
    print(f"  input:           {args.input}")
    print(f"  output:          {args.output}")
    print(f"  n_samples:       {times.size}")
    print(f"  n_components:    {n_comp}")
    if times.size:
        print(f"  time_range:      [{times[0]:.6g}, {times[-1]:.6g}]")
    print(f"  parameters:      {json.dumps(summary, sort_keys=True)}")
    if "num_modes" in summary:
        print(f"  internal_modes:  {summary['num_modes']}")

    FE = material.evaluate_history(times, FVE, reset=True)
    write_force_history(args.output, times, FVE, FE)
    print(f"  wrote:           {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
