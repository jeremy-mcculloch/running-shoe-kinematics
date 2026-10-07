"""Unit detection and one-shot SI conversion for Wang force/marker files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

PositionUnit = Literal["auto", "m", "mm"]
MomentUnit = Literal["auto", "N-m", "N-mm"]


@dataclass(frozen=True)
class UnitResolution:
    source_position_units: str
    source_moment_units: str
    position_scale: float  # multiply source → metres
    moment_scale: float  # multiply source → N·m
    detection_notes: tuple[str, ...] = ()
    ambiguous: bool = False


def _finite_abs_max(arr: NDArray[np.float64]) -> float:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0 or not np.any(np.isfinite(a)):
        return float("nan")
    return float(np.nanmax(np.abs(a)))


def resolve_position_units(
    *,
    declared: str | None,
    sample_positions: NDArray[np.float64] | None,
    mode: PositionUnit = "auto",
) -> tuple[str, float, list[str]]:
    """Return (unit_label, scale_to_m, notes)."""
    notes: list[str] = []
    if mode == "m":
        return "m", 1.0, notes
    if mode == "mm":
        return "mm", 1.0e-3, notes

    decl = (declared or "").strip().lower()
    if decl in {"m", "meter", "metre", "meters", "metres"}:
        return "m", 1.0, ["declared metadata: metres"]
    if decl in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}:
        return "mm", 1.0e-3, ["declared metadata: millimetres"]

    peak = _finite_abs_max(sample_positions) if sample_positions is not None else float("nan")
    if not np.isfinite(peak):
        raise ValueError("auto position units: no finite samples and no declared unit")
    # Lab positions in m are typically O(0.1–10); mm exports are O(100–10000).
    if peak > 50.0:
        notes.append(f"auto: peak |position|={peak:.4g} → treat as mm")
        return "mm", 1.0e-3, notes
    if peak < 20.0:
        notes.append(f"auto: peak |position|={peak:.4g} → treat as m")
        return "m", 1.0, notes
    raise ValueError(
        f"auto position units ambiguous (peak |position|={peak:.4g}); "
        "pass position_units='m' or 'mm'"
    )


def resolve_moment_units(
    *,
    declared: str | None,
    sample_moments: NDArray[np.float64] | None,
    mode: MomentUnit = "auto",
) -> tuple[str, float, list[str]]:
    """Return (unit_label, scale_to_N_m, notes)."""
    notes: list[str] = []
    if mode == "N-m":
        return "N-m", 1.0, notes
    if mode == "N-mm":
        return "N-mm", 1.0e-3, notes

    decl = (declared or "").strip().lower().replace("·", "-").replace(" ", "")
    if decl in {"n-m", "nm", "n.m", "newton-metre", "newton-meter"}:
        return "N-m", 1.0, ["declared metadata: N·m"]
    if decl in {"n-mm", "nmm", "mm.n", "n.mm", "newton-millimetre", "newton-millimeter"}:
        return "N-mm", 1.0e-3, ["declared metadata: N·mm"]

    peak = _finite_abs_max(sample_moments) if sample_moments is not None else float("nan")
    if not np.isfinite(peak):
        # No moment channel — treat as already N·m (zeros).
        notes.append("auto: no moment samples; assuming N·m")
        return "N-m", 1.0, notes
    # Free moments are usually <~50 N·m; N·mm exports often exceed hundreds.
    if peak > 500.0:
        notes.append(f"auto: peak |moment|={peak:.4g} → treat as N·mm")
        return "N-mm", 1.0e-3, notes
    if peak <= 500.0:
        notes.append(f"auto: peak |moment|={peak:.4g} → treat as N·m")
        return "N-m", 1.0, notes
    raise ValueError(
        f"auto moment units ambiguous (peak |moment|={peak:.4g}); "
        "pass moment_units='N-m' or 'N-mm'"
    )


def resolve_units(
    *,
    position_mode: PositionUnit = "auto",
    moment_mode: MomentUnit = "auto",
    declared_position: str | None = None,
    declared_moment: str | None = None,
    sample_positions: NDArray[np.float64] | None = None,
    sample_moments: NDArray[np.float64] | None = None,
) -> UnitResolution:
    pos_u, pos_s, pos_notes = resolve_position_units(
        declared=declared_position, sample_positions=sample_positions, mode=position_mode
    )
    mom_u, mom_s, mom_notes = resolve_moment_units(
        declared=declared_moment, sample_moments=sample_moments, mode=moment_mode
    )
    return UnitResolution(
        source_position_units=pos_u,
        source_moment_units=mom_u,
        position_scale=pos_s,
        moment_scale=mom_s,
        detection_notes=tuple(pos_notes + mom_notes),
        ambiguous=False,
    )
