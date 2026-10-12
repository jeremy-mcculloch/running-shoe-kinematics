"""Per-frame gait replay candidates and their unilateral contact checks.

Every candidate is one stored contact interval ``I_ij`` evaluated at a solved
``[dx, dy, alpha]``; the unilateral checks use the same vectorized reconstruction
as :mod:`compliance_fem.contact.force_control`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.contact.lookup import ContactLookupResult, contact_type_from_code
from compliance_fem.contact.topology import ContactType

# Per-candidate diagnostics copied from ``reconstruct_rows`` (§9–§12, §15).
CANDIDATE_DIAGNOSTIC_FIELDS = (
    "min_free_gap",
    "min_free_gap_node",
    "max_free_penetration",
    "min_contact_reaction",
    "min_contact_reaction_node",
    "max_contact_tension",
    "heel_edge_reaction_local_x",
    "heel_edge_reaction_local_y",
    "toe_edge_reaction_local_x",
    "toe_edge_reaction_local_y",
    "heel_edge_normal_reaction",
    "toe_edge_normal_reaction",
    "heel_adjacent_free_gap",
    "toe_adjacent_free_gap",
    "complementarity_score",
    "gap_violation_term",
    "reaction_violation_term",
    "force_violation_term",
    "penalty_term",
    "violation_score",
    "disconnected_contact_warning",
)


@dataclass(frozen=True)
class WrenchCandidate:
    row: int
    contact_type: ContactType
    contact_start_index: int
    contact_end_index: int
    contact_start_x: float
    contact_end_x: float
    anchor_x: float
    d_ax: float
    d_ay: float
    alpha: float
    theta_deg: float
    Fx_pred: float
    Fy_pred: float
    Mz_pred: float
    force_residual: float
    moment_residual: float
    cop_residual: float
    cond: float
    max_free_penetration: float
    min_contact_reaction: float
    admissible: bool
    status: str
    gamma: NDArray[np.float64]
    toe: Any = None
    anchor_y: float = 0.0
    search_method: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def interval(self) -> tuple[int, int]:
        return int(self.contact_start_index), int(self.contact_end_index)

    @property
    def violation_score(self) -> float:
        return float(self.diagnostics.get("violation_score", np.inf))


def _candidate_geometry(lookup: ContactLookupResult, row: int) -> dict:
    return {
        "contact_type": contact_type_from_code(int(lookup.contact_type_codes[row])),
        "contact_start_index": int(lookup.contact_start_index[row]),
        "contact_end_index": int(lookup.contact_end_index[row]),
        "contact_start_x": float(lookup.contact_start_x[row]),
        "contact_end_x": float(lookup.contact_end_x[row]),
        "anchor_x": float(lookup.contact_anchor_reference_x[row]),
        "anchor_y": float(lookup.contact_anchor_reference_y[row]),
    }


def diagnostics_from_reconstruction(rec: dict, k: int) -> dict:
    out = {}
    for name in CANDIDATE_DIAGNOSTIC_FIELDS:
        v = rec[name][k]
        out[name] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    return out


