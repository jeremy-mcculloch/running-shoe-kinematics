"""Single-contiguous-interval ground contact.

Every lookup record is one closed, contiguous interval of bottom-surface nodes

    I_ij = {i, i+1, ..., j},   0 <= i <= j < N_b,

with bottom nodes ordered heel -> toe by reference ``x``. Nodes ``0..i-1`` form
the heel-side free region and ``j+1..N_b-1`` the toe-side free region. The
historical heel / toe / full topologies are derived labels of the same
interval (``i == 0`` heel-attached, ``j == N_b - 1`` toe-attached, both full);
an interval with free nodes on both sides is ``interior``. There is one code
path for all four labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

CONTACT_SET_MODEL = "single_contiguous_interval"
# Absent node id (no adjacent free node beyond a domain endpoint).
ABSENT_NODE_ID = -1


class ContactType(str, Enum):
    """Derived topology label of a contact interval."""

    HEEL = "heel"
    FULL = "full"
    TOE = "toe"
    INTERIOR = "interior"


class ContactMode(str, Enum):
    """Runtime interval filter for the GUI / evaluator."""

    AUTO = "auto"
    HEEL = "heel"
    INTERIOR = "interior"
    TOE = "toe"
    FULL = "full"
    SPECIFIC = "specific"


CONTACT_MODE_LABELS: dict[ContactMode, str] = {
    ContactMode.AUTO: "Auto interval",
    ContactMode.HEEL: "Heel-attached intervals",
    ContactMode.INTERIOR: "Interior intervals",
    ContactMode.TOE: "Toe-attached intervals",
    ContactMode.FULL: "Full contact",
    ContactMode.SPECIFIC: "Specific interval",
}

# Stable integer codes stored in NPZ / CSV (do not reorder).
CONTACT_TYPE_CODES: dict[ContactType, int] = {
    ContactType.HEEL: 0,
    ContactType.TOE: 1,
    ContactType.FULL: 2,
    ContactType.INTERIOR: 3,
}
CONTACT_TYPE_FROM_CODE: dict[int, ContactType] = {
    code: ctype for ctype, code in CONTACT_TYPE_CODES.items()
}


def parse_contact_mode(mode: ContactMode | str) -> ContactMode:
    """Accept enum values (case-insensitive) and the GUI labels."""
    if isinstance(mode, ContactMode):
        return mode
    text = str(mode)
    for key, label in CONTACT_MODE_LABELS.items():
        if text == label:
            return key
    return ContactMode(text.lower())


def interval_label(start: int, end: int, n_bottom: int) -> ContactType:
    """Topology label derived from the interval endpoints."""
    i, j, n = int(start), int(end), int(n_bottom)
    if i == 0 and j == n - 1:
        return ContactType.FULL
    if i == 0:
        return ContactType.HEEL
    if j == n - 1:
        return ContactType.TOE
    return ContactType.INTERIOR


def validate_interval(start: int, end: int, n_bottom: int) -> None:
    """Raise unless ``0 <= start <= end < n_bottom``."""
    i, j, n = int(start), int(end), int(n_bottom)
    if n < 1:
        raise ValueError(f"Need at least one bottom node, got n_bottom={n}.")
    if not (0 <= i <= j < n):
        raise ValueError(
            f"Invalid contact interval ({i}, {j}); require 0 <= i <= j < N_b = {n}."
        )


@dataclass(frozen=True)
class ContactInterval:
    """One contiguous contact interval ``{start..end}`` of ``n_bottom`` bottom nodes."""

    start: int
    end: int
    n_bottom: int

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end, self.n_bottom)

    @property
    def contact_node_ids(self) -> np.ndarray:
        return np.arange(self.start, self.end + 1, dtype=int)

    @property
    def heel_free_node_ids(self) -> np.ndarray:
        return np.arange(0, self.start, dtype=int)

    @property
    def toe_free_node_ids(self) -> np.ndarray:
        return np.arange(self.end + 1, self.n_bottom, dtype=int)

    @property
    def free_node_ids(self) -> np.ndarray:
        return np.concatenate([self.heel_free_node_ids, self.toe_free_node_ids])

    @property
    def heel_contact_edge_node_id(self) -> int:
        return int(self.start)

    @property
    def toe_contact_edge_node_id(self) -> int:
        return int(self.end)

    @property
    def heel_adjacent_free_node_id(self) -> int:
        return int(self.start - 1) if self.start > 0 else ABSENT_NODE_ID

    @property
    def toe_adjacent_free_node_id(self) -> int:
        return int(self.end + 1) if self.end < self.n_bottom - 1 else ABSENT_NODE_ID

    @property
    def n_contact(self) -> int:
        return int(self.end - self.start + 1)

    @property
    def n_free(self) -> int:
        return int(self.n_bottom - self.n_contact)

    @property
    def is_single_node(self) -> bool:
        return self.start == self.end

    @property
    def topology_label(self) -> ContactType:
        return interval_label(self.start, self.end, self.n_bottom)

    def contact_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_bottom, dtype=bool)
        mask[self.start : self.end + 1] = True
        return mask

    def sets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(free, contact)`` node-index arrays."""
        return self.free_node_ids, self.contact_node_ids


def n_intervals(n_bottom: int) -> int:
    """Number of contiguous intervals ``N_b (N_b + 1) / 2``."""
    n = int(n_bottom)
    return n * (n + 1) // 2


def enumerate_intervals(n_bottom: int) -> list[ContactInterval]:
    """Every interval once, ordered by ``start`` then ``end``."""
    n = int(n_bottom)
    if n < 1:
        raise ValueError(f"Need at least one bottom node, got n_bottom={n}.")
    return [ContactInterval(i, j, n) for i in range(n) for j in range(i, n)]


def interval_row(start: int, end: int, n_bottom: int) -> int:
    """Row of interval ``(start, end)`` in :func:`enumerate_intervals` order."""
    validate_interval(start, end, n_bottom)
    i, j, n = int(start), int(end), int(n_bottom)
    return i * n - (i * (i - 1)) // 2 + (j - i)


def interval_arrays(n_bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of every interval in enumeration order."""
    n = int(n_bottom)
    starts = np.concatenate([np.full(n - i, i, dtype=int) for i in range(n)])
    ends = np.concatenate([np.arange(i, n, dtype=int) for i in range(n)])
    return starts, ends


def labels_from_arrays(starts: np.ndarray, ends: np.ndarray, n_bottom: int) -> np.ndarray:
    """Vectorized topology codes for interval endpoint arrays."""
    s = np.asarray(starts, dtype=int)
    e = np.asarray(ends, dtype=int)
    last = int(n_bottom) - 1
    codes = np.full(s.shape, CONTACT_TYPE_CODES[ContactType.INTERIOR], dtype=int)
    codes[(s == 0) & (e < last)] = CONTACT_TYPE_CODES[ContactType.HEEL]
    codes[(s > 0) & (e == last)] = CONTACT_TYPE_CODES[ContactType.TOE]
    codes[(s == 0) & (e == last)] = CONTACT_TYPE_CODES[ContactType.FULL]
    return codes


def contact_mask_from_arrays(starts: np.ndarray, ends: np.ndarray, n_bottom: int) -> np.ndarray:
    """Dense ``(n_rows, n_bottom)`` contact membership from endpoint arrays."""
    nodes = np.arange(int(n_bottom), dtype=int)[None, :]
    s = np.asarray(starts, dtype=int)[:, None]
    e = np.asarray(ends, dtype=int)[:, None]
    return (nodes >= s) & (nodes <= e)


def mode_mask(
    starts: np.ndarray,
    ends: np.ndarray,
    n_bottom: int,
    mode: ContactMode | str,
    specific: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rows permitted by a contact-mode filter."""
    mode = parse_contact_mode(mode)
    s = np.asarray(starts, dtype=int)
    e = np.asarray(ends, dtype=int)
    last = int(n_bottom) - 1
    if mode is ContactMode.AUTO:
        return np.ones(s.shape, dtype=bool)
    if mode is ContactMode.HEEL:
        return s == 0
    if mode is ContactMode.TOE:
        return e == last
    if mode is ContactMode.INTERIOR:
        return (s > 0) & (e < last)
    if mode is ContactMode.FULL:
        return (s == 0) & (e == last)
    if specific is None:
        raise ValueError("Specific interval mode requires (start, end).")
    i, j = int(specific[0]), int(specific[1])
    validate_interval(i, j, n_bottom)
    return (s == i) & (e == j)


def interval_distance(
    starts: np.ndarray,
    ends: np.ndarray,
    previous: tuple[int, int] | None,
) -> np.ndarray:
    """Topology-change distance ``|i - i_prev| + |j - j_prev|`` (0 without a previous)."""
    s = np.asarray(starts, dtype=int)
    e = np.asarray(ends, dtype=int)
    if previous is None:
        return np.zeros(s.shape, dtype=int)
    return np.abs(s - int(previous[0])) + np.abs(e - int(previous[1]))


def interval_anchor_x(x_bottom: np.ndarray, start: int, end: int) -> float:
    """Numerical anchor ``x_a = (x_i + x_j) / 2`` (not a physical contact edge)."""
    x = np.asarray(x_bottom, dtype=float)
    return 0.5 * (float(x[int(start)]) + float(x[int(end)]))


def interval_anchor_y(x_bottom: np.ndarray, y_bottom: np.ndarray | None, x_anchor: float) -> float:
    """Reference bottom height at the anchor, linearly interpolated (0 for a flat sole)."""
    if y_bottom is None:
        return 0.0
    return float(np.interp(float(x_anchor), np.asarray(x_bottom, dtype=float), np.asarray(y_bottom, dtype=float)))


def contact_span(start_x: float, end_x: float) -> tuple[float, float]:
    """Reference material interval ``[x_i, x_j]`` shaded for a selected record."""
    return float(start_x), float(end_x)


def code_of(contact_type: ContactType | str) -> int:
    return CONTACT_TYPE_CODES[ContactType(contact_type)]


def type_from_code(code: int) -> ContactType:
    try:
        return CONTACT_TYPE_FROM_CODE[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown contact_type code {code}.") from exc
