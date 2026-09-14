"""Rebuild layered FEM compliance + contact lookup in a fresh process.

Streamlit's script runner shares an address space with Tornado and often runs
off the main thread. Calling Gmsh / SuperLU there has caused native segfaults
on macOS. This worker is invoked via ``subprocess`` so meshing and factorization
never touch the Streamlit process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _run(params: dict, output_dir: Path) -> Path:
    from compliance_fem.compliance import compute_compliance
    from compliance_fem.contact_lookup import (
        from_compliance_result,
        generate_contact_lookup,
        save_contact_lookup,
    )
    from compliance_fem.gui_params import build_layered_config_from_gui

    cfg = build_layered_config_from_gui(
        L_m=float(params["L"]),
        h1_heel_m=float(params["h1_heel"]),
        h1_toe_m=float(params["h1_toe"]),
        h2_heel_m=float(params["h2_heel"]),
        h2_toe_m=float(params["h2_toe"]),
        E1=float(params["E1"]),
        nu1=float(params["nu1"]),
        E_heel=float(params["E_heel"]),
        E_toe=float(params["E_toe"]),
        nu2=float(params["nu2"]),
        EI_plate=float(params["EI_plate"]),
        nx=int(params["nx"]),
        ny1=int(params["ny1"]),
        ny2=int(params["ny2"]),
        element_order=int(params["element_order"]),
    )
    a = float(np.clip(float(params["softplus_a"]), 0.0, cfg.L))
    fem = compute_compliance(cfg)
    lookup = generate_contact_lookup(
        from_compliance_result(fem),
        a=a,
        kappa=float(params["softplus_kappa"]),
        reciprocity_tol=float(params["reciprocity_tol"]),
        fem_result=fem,
    )
    return save_contact_lookup(lookup, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    params = json.loads(args.params_json.read_text(encoding="utf-8"))
    npz = _run(params, args.output_dir)
    print(str(npz), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
