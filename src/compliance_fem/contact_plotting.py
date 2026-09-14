"""Diagnostic plots for contact-edge lookup tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from compliance_fem.contact_basis import BASIS_MODE_NAMES
from compliance_fem.contact_lookup import (
    CORNER_NAMES,
    SCALAR_EDGE_GAP_V,
    SCALAR_EDGE_RY,
    SCALAR_FX,
    SCALAR_FY,
    SCALAR_MV,
    SCALAR_TOE,
    ContactLookupResult,
)
from compliance_fem.contact_query import SelectedCandidate, SuperposedProfile
from compliance_fem.contact_topology import ContactType

MODE_LABELS = (
    r"$\phi_1$",
    r"$B_x$",
    r"$B_y$",
    r"$B_{rx}$",
    r"$B_{ry}$",
)

PARTIAL_FAMILIES = (ContactType.HEEL, ContactType.TOE)


def plot_scalar_lookup(result: ContactLookupResult, output_dir: Path) -> None:
    """Plot the local basis scalars versus contact-edge location, per family.

    Heel and toe records both index on ``l`` but describe opposite contact
    intervals, so they get separate figures. The single full-contact record has
    no ``l`` and is reported in the corner-reaction figure instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantities = [
        (SCALAR_FX, r"local $F_x^{(k)}(l)$", "Fx"),
        (SCALAR_FY, r"local $F_y^{(k)}(l)$ (total top nodal force)", "Fy"),
        (SCALAR_MV, r"$M_v^{(k)}(l)=x_t^T f_{t,y}$", "Mv"),
        (
            SCALAR_EDGE_GAP_V,
            r"edge-free local vertical disp. $v^{(k)}(l)$",
            "edge_free_gap",
        ),
        (
            SCALAR_EDGE_RY,
            r"edge-contact local vertical nodal reaction $R_y^{(k)}(l)$",
            "edge_contact_reaction",
        ),
        (SCALAR_TOE, r"$T_{\mathrm{toe}}^{(k)}(l)$", "toe_moment"),
    ]
    for family in PARTIAL_FAMILIES:
        rows = result.rows_for(family)
        if rows.size == 0:
            continue
        l = result.candidate_l[rows]
        for idx, title, stem in quantities:
            fig, ax = plt.subplots(figsize=(7, 4))
            for k, label in enumerate(MODE_LABELS):
                ax.plot(l, result.scalar_lookup[rows, k, idx], "-o", markersize=3, label=label)
            ax.set_xlabel(rf"{family.value} contact edge $l$ (material)")
            ax.set_ylabel(title)
            ax.set_title(
                f"{family.value} contact: {title}  basis order: {', '.join(BASIS_MODE_NAMES)}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(output_dir / f"{stem}_vs_l_{family.value}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    plot_full_contact_corners(result, output_dir)


def plot_full_contact_corners(result: ContactLookupResult, output_dir: Path) -> None:
    """Plot the full-contact record's per-basis local corner reactions."""
    output_dir = Path(output_dir)
    row = result.full_contact_row()
    corners = result.corner_reactions_local[row]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    x = range(len(MODE_LABELS))
    for c, (ax, corner) in enumerate(zip(axes, CORNER_NAMES)):
        ax.bar([i - 0.18 for i in x], corners[:, c, 0], width=0.36, label=r"$R_x^{T}$")
        ax.bar([i + 0.18 for i in x], corners[:, c, 1], width=0.36, label=r"$R_y^{T}$")
        ax.axhline(0.0, color="0.3", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(MODE_LABELS)
        ax.set_title(f"full contact: {corner} corner local reaction")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("local nodal reaction force")
    fig.tight_layout()
    fig.savefig(output_dir / "full_contact_corner_reactions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_solver_residuals(result: ContactLookupResult, output_dir: Path) -> None:
    """Plot solver residual diagnostics per record, grouped by topology."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = range(result.n_records)
    ax.semilogy(rows, result.solve_residuals, label="solve")
    ax.semilogy(rows, result.top_displacement_residuals, label="top disp.")
    ax.semilogy(rows, result.contact_displacement_residuals, label="contact disp.")
    ax.semilogy(rows, result.force_equilibrium_residuals, label="force eq.")
    ax.semilogy(rows, result.moment_equilibrium_residuals, label="moment eq.")
    for family in (ContactType.HEEL, ContactType.TOE, ContactType.FULL):
        family_rows = result.rows_for(family)
        if family_rows.size:
            ax.axvline(float(family_rows[0]), color="0.5", ls=":", lw=1)
            ax.text(
                float(family_rows[0]),
                ax.get_ylim()[1],
                f" {family.value}",
                fontsize=8,
                va="top",
            )
    ax.set_xlabel("record index (heel, then toe, then full)")
    ax.set_ylabel("residual")
    ax.set_title("Boundary-system residuals per contact record")
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
    """Plot weighted profile, violation score, and selected gap/reaction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c = profile.coefficients
    n_t = len(lookup.x_top)
    W = lookup.basis_top_displacements
    w = W[n_t:, :] @ c

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lookup.x_top, w, "-o", markersize=3)
    ax.set_xlabel("x")
    ax.set_ylabel("local top vertical displacement")
    coeff_txt = ", ".join(f"{ci:.3g}" for ci in c)
    ax.set_title(rf"Weighted top profile $\gamma=({coeff_txt})$", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "weighted_top_displacement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    rows = range(lookup.n_records)
    ax.plot(rows, profile.violation, "-o", markersize=3, label="J(record)")
    if selected.admissible_rows.size:
        ax.plot(
            selected.admissible_rows,
            profile.violation[selected.admissible_rows],
            "o",
            label="admissible",
        )
    ax.axvline(selected.candidate_row, color="C3", linestyle="--", label="selected")
    ax.set_xlabel("record index (heel, then toe, then full)")
    ax.set_ylabel("violation score J")
    ax.set_title(f"Contact violation score (selected: {selected.contact_type})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "violation_vs_record.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    marker_x = float(lookup.anchor_reference_x[selected.candidate_row])
    marker_label = (
        "numerical anchor" if selected.contact_type == ContactType.FULL.value else "contact edge"
    )
    for values, ylabel, title, filename in (
        (
            selected.gap,
            "bottom local vertical displacement",
            "Free-bottom local vertical displacement",
            "selected_gap.png",
        ),
        (
            selected.reaction,
            "bottom nodal reaction force (not pressure)",
            "Bottom nodal reaction",
            "selected_reaction.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lookup.x_bottom, values, "-o", markersize=3)
        ax.axvline(marker_x, color="C3", linestyle="--", label=marker_label)
        ax.set_xlabel("x (material)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} for the selected {selected.contact_type} record")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_lookup_plots(result: ContactLookupResult, output_dir: Path) -> None:
    """Write all generation-time diagnostic plots."""
    plot_scalar_lookup(result, output_dir)
    plot_solver_residuals(result, output_dir)
