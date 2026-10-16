"""Nodal-force compliance blocks of the FEM boundary (top / bottom), validated and symmetrized.

The contact operator (:mod:`compliance_fem.contact.contact_operator`) is condensed
from these blocks; nothing here depends on a contact state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from compliance_fem.fem.boundaries import DOF_ORDERING_COMPONENT_MAJOR_UV

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult

REGENERATE_COMPLIANCE_MESSAGE = (
    "This compliance NPZ is not a vector (component-major u,v) file. "
    "Regenerate FEM with the current pipeline so Ctt/Ctb/Cbt/Cbb are 2n x 2n."
)


@dataclass
class ComplianceBlockMatrix:
    """Nodal-force compliance blocks and boundary coordinates."""

    Ctt: np.ndarray
    Ctb: np.ndarray
    Cbt: np.ndarray
    Cbb: np.ndarray
    x_top: np.ndarray
    x_bottom: np.ndarray
    L: float
    H: float
    reciprocity_error: float = 0.0
    y_top: np.ndarray | None = None
    y_bottom: np.ndarray | None = None
    x_plate: np.ndarray | None = None
    y_plate: np.ndarray | None = None
    dof_ordering: str = DOF_ORDERING_COMPONENT_MAJOR_UV
    n_top_nodes: int | None = None
    n_bottom_nodes: int | None = None


def boundary_y(blocks: ComplianceBlockMatrix) -> tuple[np.ndarray, np.ndarray]:
    n_t = len(blocks.x_top)
    n_b = len(blocks.x_bottom)
    y_t = (
        np.asarray(blocks.y_top, dtype=float)
        if blocks.y_top is not None
        else np.full(n_t, float(blocks.H))
    )
    y_b = (
        np.asarray(blocks.y_bottom, dtype=float)
        if blocks.y_bottom is not None
        else np.zeros(n_b, dtype=float)
    )
    return y_t, y_b


def _optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _require_vector_compliance_blocks(
    Ctt: np.ndarray,
    x_top: np.ndarray,
    x_bottom: np.ndarray,
    dof_ordering: str,
) -> None:
    n_t = int(len(x_top))
    n_b = int(len(x_bottom))
    if Ctt.ndim != 2 or Ctt.shape[0] % 2 != 0 or Ctt.shape[1] % 2 != 0:
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)
    if Ctt.shape != (2 * n_t, 2 * n_t):
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)
    if n_b < 1:
        raise ValueError("Compliance file has no bottom nodes.")
    if dof_ordering != DOF_ORDERING_COMPONENT_MAJOR_UV:
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)


def get_compliance_block_matrix(result: ComplianceResult) -> ComplianceBlockMatrix:
    """Extract nodal-force blocks from an in-memory ComplianceResult."""
    cfg = result.config
    dof_ordering = str(result.dof_ordering)
    Ctt = np.asarray(result.Ctt_force, dtype=float)
    x_top = np.asarray(result.x_top, dtype=float)
    x_bottom = np.asarray(result.x_bottom, dtype=float)
    _require_vector_compliance_blocks(Ctt, x_top, x_bottom, dof_ordering)
    return ComplianceBlockMatrix(
        Ctt=Ctt,
        Ctb=np.asarray(result.Ctb_force, dtype=float),
        Cbt=np.asarray(result.Cbt_force, dtype=float),
        Cbb=np.asarray(result.Cbb_force, dtype=float),
        x_top=x_top,
        x_bottom=x_bottom,
        L=float(cfg.shoe_length_m),
        H=float(cfg.shoe_height_m),
        reciprocity_error=float(result.reciprocity_error),
        y_top=_optional_array(result.y_top),
        y_bottom=_optional_array(result.y_bottom),
        x_plate=_optional_array(result.x_plate),
        y_plate=_optional_array(result.y_plate),
        dof_ordering=dof_ordering,
        n_top_nodes=int(result.n_top_nodes),
        n_bottom_nodes=int(result.n_bottom_nodes),
    )


def _relative_asymmetry(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))


def prepare_compliance_blocks(
    blocks: ComplianceBlockMatrix,
    reciprocity_tol: float = 1e-6,
) -> tuple[ComplianceBlockMatrix, float]:
    """Verify reciprocity and symmetrize roundoff-level asymmetry."""
    Ctt = np.array(blocks.Ctt, copy=True)
    Ctb = np.array(blocks.Ctb, copy=True)
    Cbt = np.array(blocks.Cbt, copy=True)
    Cbb = np.array(blocks.Cbb, copy=True)
    _require_vector_compliance_blocks(Ctt, blocks.x_top, blocks.x_bottom, blocks.dof_ordering)
    err_tt = _relative_asymmetry(Ctt)
    err_bb = _relative_asymmetry(Cbb)
    err_bt = float(np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30))
    reported = max(err_tt, err_bb, err_bt, blocks.reciprocity_error)
    if reported > reciprocity_tol:
        raise ValueError(
            f"Compliance reciprocity error {reported:.3e} exceeds tolerance {reciprocity_tol:.3e}. "
            "Re-run FEM or pass a larger --reciprocity-tol if the residual is still a solver artifact."
        )
    n_t = Ctt.shape[0]
    full = np.block([[Ctt, Ctb], [Cbt, Cbb]])
    full = 0.5 * (full + full.T)
    prepared = ComplianceBlockMatrix(
        Ctt=full[:n_t, :n_t],
        Ctb=full[:n_t, n_t:],
        Cbt=full[n_t:, :n_t],
        Cbb=full[n_t:, n_t:],
        x_top=blocks.x_top,
        x_bottom=blocks.x_bottom,
        L=blocks.L,
        H=blocks.H,
        reciprocity_error=reported,
        y_top=blocks.y_top,
        y_bottom=blocks.y_bottom,
        x_plate=blocks.x_plate,
        y_plate=blocks.y_plate,
        dof_ordering=blocks.dof_ordering,
        n_top_nodes=blocks.n_top_nodes,
        n_bottom_nodes=blocks.n_bottom_nodes,
    )
    return prepared, reported
