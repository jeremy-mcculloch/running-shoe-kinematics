"""Example script for rectangular compliance analysis."""

from __future__ import annotations

from pathlib import Path

from compliance_fem.cli import main


if __name__ == "__main__":
    main(
        [
            "--L",
            "1.0",
            "--H",
            "0.5",
            "--E",
            "1.0e6",
            "--nu",
            "0.3",
            "--nx",
            "40",
            "--ny",
            "20",
            "--order",
            "1",
            "--output",
            str(Path("outputs/rectangle")),
        ]
    )
