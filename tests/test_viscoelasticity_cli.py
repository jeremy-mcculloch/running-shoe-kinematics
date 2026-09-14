"""CLI smoke tests for visco-force-map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compliance_fem.viscoelasticity.cli import main
from compliance_fem.viscoelasticity.io import read_force_history, write_force_history


def _write_input(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 21)
    FVE = np.column_stack([np.sin(2 * np.pi * t), 0.5 * t])
    # Temporary write via helper then strip elastic columns by rewriting.
    write_force_history(path, t, FVE, np.zeros_like(FVE))
    # Rewrite as input-only CSV
    rows = np.column_stack([t, FVE])
    np.savetxt(path, rows, delimiter=",", header="time,Fx,Fy", comments="", fmt="%.16g")


def test_cli_sls(tmp_path: Path):
    inp = tmp_path / "in.csv"
    out = tmp_path / "out_sls.csv"
    _write_input(inp)
    rc = main(
        [
            "--model",
            "sls",
            "--g-inf",
            "0.5",
            "--tau-r",
            "0.2",
            "-i",
            str(inp),
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    times, data = read_force_history(out)
    assert times.size == 21
    # Output has Fx_ve, Fy_ve, Fx_e, Fy_e — read_force_history keeps all force cols.
    assert data.shape[1] >= 4


def test_cli_fractional_and_fung(tmp_path: Path):
    inp = tmp_path / "in.csv"
    _write_input(inp)
    out_f = tmp_path / "out_frac.csv"
    out_g = tmp_path / "out_fung.csv"
    assert (
        main(
            [
                "--model",
                "fractional",
                "--E0",
                "1",
                "--T",
                "1",
                "--alpha",
                "0.5",
                "--num-modes",
                "16",
                "-i",
                str(inp),
                "-o",
                str(out_f),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--model",
                "fung",
                "--C",
                "0.1",
                "--tau1",
                "1e-3",
                "--tau2",
                "10",
                "--num-modes",
                "16",
                "-i",
                str(inp),
                "-o",
                str(out_g),
            ]
        )
        == 0
    )
    assert out_f.exists() and out_g.exists()
