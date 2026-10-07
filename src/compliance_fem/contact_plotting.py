"""Diagnostic plots for interval contact lookup tables.

Interval records are indexed by ``(i, j)``; per-record quantities are drawn as
heatmaps with the start index on the horizontal axis and the end index on the
vertical axis (only ``j >= i`` is populated).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compliance_fem.contact_basis import AFFINE_COLUMN_NAMES
from compliance_fem.contact_lookup import (
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    ContactLookupResult,
)
from compliance_fem.contact_query import SelectedCandidate, SuperposedProfile

COLUMN_LABELS = (r"closure $z_0$", r"$\alpha$", r"$B_x$", r"$B_y$", r"$B_{rx}$", r"$B_{ry}$")


def interval_grid(lookup: ContactLookupResult, values: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Scatter per-record values into an ``(N_b, N_b)`` grid indexed ``[j, i]``."""
    n_b = int(lookup.n_bottom_nodes)
    grid = np.full((n_b, n_b), np.nan)
    starts = np.asarray(lookup.contact_start_index, dtype=int)
    ends = np.asarray(lookup.contact_end_index, dtype=int)
    mask = np.zeros(lookup.n_records, dtype=bool)
    mask[lookup.valid_rows if rows is None else rows] = True
    v = np.asarray(values, dtype=float)
    mask &= np.isfinite(v)
    grid[ends[mask], starts[mask]] = v[mask]
    return grid


def _heatmap(ax, grid: np.ndarray, title: str, marker: tuple[int, int] | None = None) -> None:
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
    if marker is not None:
        ax.plot([marker[0]], [marker[1]], "rx", ms=9)
    ax.set_xlabel("start index i")
    ax.set_ylabel("end index j")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)


def plot_scalar_lookup(result: ContactLookupResult, output_dir: Path) -> None:
    """Heatmaps of every affine column of the scalar resultants over (i, j)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, title, stem in (
        (SCALAR_FX, r"local $F_x$", "Fx"),
        (SCALAR_FY, r"local $F_y$", "Fy"),
        (SCALAR_MV, r"$M_v = x_t^T f_{t,y}$", "Mv"),
        (SCALAR_TOE, r"$T_{\mathrm{toe}}$", "toe_moment"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for k, ax in enumerate(axes.flat):
            grid = interval_grid(result, result.scalar_lookup[:, k, idx])
            _heatmap(ax, grid, f"{title}: {COLUMN_LABELS[k]} ({AFFINE_COLUMN_NAMES[k]})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}_interval_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_interval_validity(result: ContactLookupResult, output_dir: Path) -> None:
    """Valid / rejected intervals and the boundary-matrix condition estimate."""
    output_dir = Path(output_dir)
    all_rows = np.arange(result.n_records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _heatmap(axes[0], interval_grid(result, result.valid_mask.astype(float), all_rows), "valid interval (1) / rejected (0)")
    cond = np.asarray(result.condition_estimates, dtype=float)
    _heatmap(axes[1], interval_grid(result, np.log10(cond)), "log10 cond(A) estimate (1 / rcond)")
    fig.tight_layout()
    fig.savefig(output_dir / "interval_validity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_solver_residuals(result: ContactLookupResult, output_dir: Path) -> None:
    """Boundary-system residuals per interval record (rows ordered by start, then end)."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = np.arange(result.n_records)
    floor = 1e-300
    for arr, label in (
        (result.solve_residuals, "solve"),
        (result.top_displacement_residuals, "top disp."),
        (result.contact_displacement_residuals, "contact disp."),
        (result.force_equilibrium_residuals, "force eq."),
        (result.moment_equilibrium_residuals, "moment eq."),
    ):
        ax.semilogy(rows, np.maximum(np.asarray(arr, dtype=float), floor), label=label, lw=0.8)
    ax.set_xlabel("record row (start index, then end index)")
    ax.set_ylabel("residual")
    ax.set_title("Boundary-system residuals per interval record")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "residuals_vs_record.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_query_diagnostics(
    lookup: ContactLookupResult,
    profile: SuperposedProfile,
    selected: SelectedCandidate,
    output_dir: Path,
) -> None:
    """Weighted top profile, violation heatmap, and the selected gap / reaction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c = profile.coefficients
    n_t = len(lookup.x_top)
    W = np.asarray(lookup.basis_top_displacements, dtype=float)
    w = W[n_t:, :] @ np.concatenate([[1.0], c])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lookup.x_top, w, "-o", markersize=3)
    ax.set_xlabel("x")
    ax.set_ylabel("local top vertical displacement")
    coeff_txt = ", ".join(f"{ci:.3g}" for ci in c)
    ax.set_title(rf"Weighted top profile $\gamma=({coeff_txt})$", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "weighted_top_displacement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    marker = (selected.contact_start_index, selected.contact_end_index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    V = np.where(np.isfinite(profile.violation), profile.violation, np.nan)
    _heatmap(axes[0], interval_grid(lookup, np.log10(np.maximum(V, 1e-30))), "log10 violation", marker)
    _heatmap(axes[1], interval_grid(lookup, profile.admissible.astype(float)), "admissible (1)", marker)
    fig.tight_layout()
    fig.savefig(output_dir / "violation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    x_i = float(lookup.x_bottom[selected.contact_start_index])
    x_j = float(lookup.x_bottom[selected.contact_end_index])
    for values, ylabel, title, filename in (
        (selected.gap, "fixed-frame normal gap", "Bottom normal gap", "selected_gap.png"),
        (selected.reaction, "normal nodal reaction (not pressure)", "Bottom normal reaction", "selected_reaction.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lookup.x_bottom, values, "-o", markersize=3)
        ax.axvline(x_i, color="C3", linestyle="--", label="contact edges x_i, x_j")
        ax.axvline(x_j, color="C3", linestyle="--")
        ax.axvline(selected.anchor_x, color="0.5", linestyle=":", label="numerical anchor")
        ax.set_xlabel("x (material)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} for the selected {selected.contact_type} interval {marker}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_lookup_plots(result: ContactLookupResult, output_dir: Path) -> None:
    """Write all generation-time diagnostic plots."""
    plot_scalar_lookup(result, output_dir)
    plot_interval_validity(result, output_dir)
    plot_solver_residuals(result, output_dir)
