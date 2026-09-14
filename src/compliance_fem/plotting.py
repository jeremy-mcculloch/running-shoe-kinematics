"""Diagnostic plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compliance_fem.compliance import ComplianceResult
from compliance_fem.validation import ValidationReport


def plot_compliance_heatmap(matrix: np.ndarray, title: str, output_path: Path) -> None:
    """Save a heatmap of a compliance matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Force index")
    ax.set_ylabel("Displacement index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_validation_case(report: ValidationReport, x_top: np.ndarray, output_path: Path) -> None:
    """Compare analytical and computed top displacement."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_top, report.v_top_analytical, "-", label="Analytical", linewidth=2)
    ax.plot(x_top, report.v_top_computed, "o", label="Computed", markersize=3)
    ax.set_xlabel("x")
    ax.set_ylabel("Top vertical displacement")
    ax.set_title(f"Validation: {report.case_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(result: ComplianceResult, reports: list[ValidationReport], output_dir: Path) -> None:
    """Write compliance heatmaps and validation displacement plots."""
    plot_compliance_heatmap(
        result.Cbt_force,
        r"$\mathbf{C}_{bt}^{F}$ (bottom disp. / top force)",
        output_dir / "Cbt_force_heatmap.png",
    )
    plot_compliance_heatmap(
        result.Cbb_force,
        r"$\mathbf{C}_{bb}^{F}$ (bottom disp. / bottom force)",
        output_dir / "Cbb_force_heatmap.png",
    )
    for report in reports:
        plot_validation_case(report, result.x_top, output_dir / f"validation_{report.case_name}.png")
