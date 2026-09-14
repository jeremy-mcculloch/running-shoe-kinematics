"""Explicit ground-contact topologies: heel, full, and toe.

Contact topology is stored explicitly and never inferred from the material
edge coordinate ``l``. Heel contact occupies ``[0, l]``, toe contact occupies
``[l, L]``, and full contact occupies ``[0, L]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ContactType(str, Enum):
    """Contiguous ground-contact configuration."""

    HEEL = "heel"
    FULL = "full"
    TOE = "toe"


class ContactMode(str, Enum):
    """Runtime selection mode for the GUI / evaluator."""

    AUTO = "auto"
    HEEL = "heel"
    FULL = "full"
    TOE = "toe"


# Stable integer codes stored in NPZ / CSV (do not reorder).
CONTACT_TYPE_CODES: dict[ContactType, int] = {
    ContactType.HEEL: 0,
    ContactType.TOE: 1,
    ContactType.FULL: 2,
}
CONTACT_TYPE_FROM_CODE: dict[int, ContactType] = {
    code: ctype for ctype, code in CONTACT_TYPE_CODES.items()
}


@dataclass(frozen=True)
class ContactRecordSpec:
    """One lookup-table record: topology plus optional transition edge node."""

    contact_type: ContactType
    edge_node_id: int | None = None

    def __post_init__(self) -> None:
        if self.contact_type is ContactType.FULL:
            if self.edge_node_id is not None:
                raise ValueError("Full contact has no physical edge node.")
        elif self.edge_node_id is None:
            raise ValueError(f"{self.contact_type.value} contact requires an edge_node_id.")

    def sets(self, n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(free, contact)`` node-index arrays for this record."""
        contact, free = contact_free_sets(self, n_bottom)
        return free, contact


def heel_sets(i: int, n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """Contact ``{0..i}`` and free ``{i+1..N-1}`` for heel candidate ``i``."""
    if i < 0 or i >= n_bottom - 1:
        raise IndexError(
            f"Heel candidate index {i} out of range for n_bottom={n_bottom} "
            "(valid i = 0..N_b-2)."
        )
    contact = np.arange(0, i + 1, dtype=int)
    free = np.arange(i + 1, n_bottom, dtype=int)
    return contact, free


def toe_sets(i: int, n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """Contact ``{i..N-1}`` and free ``{0..i-1}`` for toe candidate ``i``."""
    if i < 1 or i >= n_bottom:
        raise IndexError(
            f"Toe candidate index {i} out of range for n_bottom={n_bottom} "
            "(valid i = 1..N_b-1)."
        )
    contact = np.arange(i, n_bottom, dtype=int)
    free = np.arange(0, i, dtype=int)
    return contact, free


def full_sets(n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """Contact ``{0..N-1}`` and empty free set."""
    if n_bottom < 1:
        raise ValueError("Need at least one bottom node for full contact.")
    contact = np.arange(n_bottom, dtype=int)
    free = np.zeros(0, dtype=int)
    return contact, free


def contact_free_sets(
    spec: ContactRecordSpec,
    n_bottom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(contact, free)`` node-index arrays for a record spec."""
    if spec.contact_type is ContactType.HEEL:
        return heel_sets(int(spec.edge_node_id), n_bottom)
    if spec.contact_type is ContactType.TOE:
        return toe_sets(int(spec.edge_node_id), n_bottom)
    return full_sets(n_bottom)


def free_contact_sets(i: int, n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """Toe-topology alias returning ``(free, contact)`` for historical callers.

    Matches the pre-topology pipeline: free ``[0, i)``, contact ``[i, N)``,
    including the endpoint cases ``i=0`` (full contact) and ``i=N-1``.
    Prefer :func:`contact_free_sets` for new code.
    """
    if i < 0 or i >= n_bottom:
        raise IndexError(f"Candidate index {i} out of range for n_bottom={n_bottom}.")
    free = np.arange(0, i, dtype=int)
    contact = np.arange(i, n_bottom, dtype=int)
    return free, contact


def enumerate_records(n_bottom: int) -> list[ContactRecordSpec]:
    """Ordered lookup records: heel ``0..N-2``, toe ``1..N-1``, then full once."""
    if n_bottom < 2:
        raise ValueError(f"Need at least two bottom nodes, got n_bottom={n_bottom}.")
    records: list[ContactRecordSpec] = []
    for i in range(0, n_bottom - 1):
        records.append(ContactRecordSpec(ContactType.HEEL, i))
    for i in range(1, n_bottom):
        records.append(ContactRecordSpec(ContactType.TOE, i))
    records.append(ContactRecordSpec(ContactType.FULL, None))
    return records


def n_records(n_bottom: int) -> int:
    """Expected number of lookup records: ``2 N_b - 1``."""
    return 2 * int(n_bottom) - 1


def anchor_reference_x(
    spec: ContactRecordSpec,
    x_bottom: np.ndarray,
    L: float,
) -> float:
    """Material/reference anchor: ``l_i`` for partial contact, ``L/2`` for full."""
    if spec.contact_type is ContactType.FULL:
        return 0.5 * float(L)
    return float(np.asarray(x_bottom, dtype=float)[int(spec.edge_node_id)])


def edge_node_pair(
    spec: ContactRecordSpec,
    n_bottom: int,
) -> tuple[int | None, int | None]:
    """Return ``(edge_contact_node_id, edge_free_node_id)``.

    Heel: contact ``i``, first free ``i+1``.
    Toe: contact ``i``, last free ``i-1``.
    Full: ``(None, None)``.
    """
    if spec.contact_type is ContactType.FULL:
        return None, None
    i = int(spec.edge_node_id)
    if spec.contact_type is ContactType.HEEL:
        if i + 1 >= n_bottom:
            raise IndexError(f"Heel edge i={i} has no free neighbour.")
        return i, i + 1
    if i - 1 < 0:
        raise IndexError(f"Toe edge i={i} has no free neighbour.")
    return i, i - 1


def contact_span(
    contact_type: ContactType,
    l: float | None,
    L: float,
) -> tuple[float, float]:
    """Ground interval shaded for a selected topology."""
    if contact_type is ContactType.HEEL:
        return 0.0, float(l)
    if contact_type is ContactType.TOE:
        return float(l), float(L)
    return 0.0, float(L)


def code_of(contact_type: ContactType) -> int:
    return CONTACT_TYPE_CODES[contact_type]


def type_from_code(code: int) -> ContactType:
    try:
        return CONTACT_TYPE_FROM_CODE[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown contact_type code {code}.") from exc
