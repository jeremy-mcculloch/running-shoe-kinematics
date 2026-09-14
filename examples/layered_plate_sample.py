"""Example script for layered-plate compliance analysis."""

from __future__ import annotations

from pathlib import Path

from compliance_fem.cli import main


if __name__ == "__main__":
    main(
        [
            "--geometry",
            "layered-plate",
            "--L",
            "0.29",
            "--h1-heel",
            "0.01",
            "--h1-toe",
            "0.01",
            "--h2-heel",
            "0.029",
            "--h2-toe",
            "0.025",
            "--E1",
            "2.6e5",
            "--nu1",
            "0.113",
            "--E-heel",
            "3.54e5",
            "--E-toe",
            "2.07e5",
            "--nu2",
            "0.113",
            "--EI-plate",
            "2.0",
            "--nx",
            "100",
            "--ny1",
            "12",
            "--ny2",
            "12",
            "--element-order",
            "1",
            "--output",
            str(Path("outputs/layered_plate")),
        ]
    )
