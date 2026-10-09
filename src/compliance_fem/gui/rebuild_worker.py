"""Rebuild the measured-sole FEM compliance + contact lookup in a fresh process.

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


def _run(params: dict, output_dir: Path) -> Path:
    from compliance_fem.gui.params import setup_from_gui
    from compliance_fem.contact.model_setup import ensure_lookup_exists

    return ensure_lookup_exists(setup_from_gui(params), output_dir=output_dir, force=True, skip_plots=True, quiet=True)


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
