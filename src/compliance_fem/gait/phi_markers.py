"""Marker-derived heel-to-forefoot pitch angle φ(t)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# Prefer mid-forefoot when medial/lateral metatarsal markers exist.
# Includes Wang et al. labels (RHeel/RM1/RM5/RMid) as well as OpenSim-style names.
_HEEL_CANDIDATES = (
    "RHEEL",
    "LHEEL",
    "RHEE",
    "LHEE",
    "HEEL",
    "R_HEEL",
    "L_HEEL",
)
_FORE_MED = (
    "RM1",
    "LM1",
    "RMT1",
    "LMT1",
    "R_MT1",
    "L_MT1",
    "RTOE_MED",
    "LTOE_MED",
)
_FORE_LAT = (
    "RM5",
    "LM5",
    "RMT5",
    "LMT5",
    "R_MT5",
    "L_MT5",
    "RTOE_LAT",
    "LTOE_LAT",
)
_FORE_MID = (
    "RMID",
    "LMID",
    "RTOE",
    "LTOE",
    "RMT2",
    "LMT2",
    "R_TOE",
    "L_TOE",
    "TOE",
)


@dataclass(frozen=True)
class PhiSeries:
    times: NDArray[np.float64]
    phi_rad: NDArray[np.float64]
    heel_label: str
    fore_label: str
    missing_mask: NDArray[np.bool_]


def _pick_label(available: set[str], candidates: tuple[str, ...], foot: str) -> str | None:
    foot = foot.lower()
    pref = "R" if foot.startswith("r") else "L" if foot.startswith("l") else ""
    ordered = list(candidates)
    if pref:
        ordered = [c for c in candidates if c.upper().startswith(pref)] + [
            c for c in candidates if not c.upper().startswith(pref)
        ]
    upper_map = {a.upper(): a for a in available}
    for c in ordered:
        if c.upper() in upper_map:
            return upper_map[c.upper()]
    return None


def select_chord_markers(
    labels: list[str],
    *,
    foot: str = "right",
    heel_override: str | None = None,
    fore_override: str | None = None,
) -> tuple[str, str]:
    available = set(labels)
    if heel_override:
        if heel_override not in available:
            raise KeyError(f"heel marker {heel_override!r} not in {sorted(available)}")
        heel = heel_override
    else:
        heel = _pick_label(available, _HEEL_CANDIDATES, foot)
        if heel is None:
            raise KeyError(f"no heel marker found among {labels}")

    if fore_override:
        if fore_override not in available:
            raise KeyError(f"forefoot marker {fore_override!r} not in {sorted(available)}")
        return heel, fore_override

    med = _pick_label(available, _FORE_MED, foot)
    lat = _pick_label(available, _FORE_LAT, foot)
    if med is not None and lat is not None:
        return heel, f"MID({med}+{lat})"
    mid = _pick_label(available, _FORE_MID, foot)
    if mid is None:
        raise KeyError(f"no forefoot marker found among {labels}")
    return heel, mid


def _fore_points(
    markers: dict[str, NDArray[np.float64]],
    fore_label: str,
) -> NDArray[np.float64]:
    if fore_label.startswith("MID(") and "+" in fore_label:
        inner = fore_label[4:-1]
        a, b = inner.split("+", 1)
        return 0.5 * (markers[a] + markers[b])
    return markers[fore_label]


def estimate_phi(
    times: NDArray[np.float64],
    markers: dict[str, NDArray[np.float64]],
    *,
    foot: str = "right",
    heel_override: str | None = None,
    fore_override: str | None = None,
) -> PhiSeries:
    """φ = atan2(y_f - y_h, x_f - x_h), unwrapped; missing samples flagged."""
    labels = list(markers.keys())
    heel_l, fore_l = select_chord_markers(
        labels, foot=foot, heel_override=heel_override, fore_override=fore_override
    )
    heel = markers[heel_l if heel_l in markers else heel_l]
    # heel_l is always a real key from select
    heel = markers[heel_l]
    fore = _fore_points(markers, fore_l)
    dx = fore[:, 0] - heel[:, 0]
    dy = fore[:, 1] - heel[:, 1]
    missing = ~(np.isfinite(dx) & np.isfinite(dy) & (np.hypot(dx, dy) > 1e-9))
    phi = np.arctan2(dy, dx)
    phi = np.unwrap(phi)
    phi[missing] = np.nan
    return PhiSeries(
        times=np.asarray(times, dtype=np.float64),
        phi_rad=phi,
        heel_label=heel_l,
        fore_label=fore_l,
        missing_mask=missing,
    )


def interpolate_angle_to_grid(
    t_src: NDArray[np.float64],
    phi_src: NDArray[np.float64],
    t_dst: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate unwrapped angle onto ``t_dst`` (NaNs skipped via finite mask)."""
    mask = np.isfinite(phi_src) & np.isfinite(t_src)
    if np.count_nonzero(mask) < 2:
        return np.full(t_dst.shape, np.nan)
    return np.interp(t_dst, t_src[mask], phi_src[mask])
