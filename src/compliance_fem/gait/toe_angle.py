"""Prescribed toe-bend angle from foot pitch and vertical load."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compressive_fy_load(fy: ArrayLike) -> NDArray[np.float64]:
    """Positive compressive top-load magnitude: ``max(0, -Fy)``."""
    return np.maximum(0.0, -np.asarray(fy, dtype=np.float64))


def prescribed_toe_angle_deg(
    phi_rad: ArrayLike,
    fy: ArrayLike,
    fy_max: float,
    *,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
) -> NDArray[np.float64]:
    """Toe angle from foot pitch and normalized vertical load.

    Uses

        θ = relu(−φ_deg) · (1 − (1 − F_y^c / F_y^max)^4)

    where ``φ_deg`` is the marker heel-to-toe pitch in degrees (negative when
    the toe is below the heel), ``F_y^c = max(0, −F_y)`` is compressive top
    force, and ``F_y^max`` is the peak compressive load over the replay window.
    Result is clipped to ``[theta_min_deg, theta_max_deg]``.
    """
    phi_deg = np.rad2deg(np.asarray(phi_rad, dtype=np.float64))
    fy_c = compressive_fy_load(fy)
    denom = float(fy_max) if float(fy_max) > 0.0 else 1.0
    ratio = np.clip(fy_c / denom, 0.0, 1.0)
    load = 1.0 - np.power(1.0 - ratio, 4.0)
    theta = np.maximum(0.0, -phi_deg) * load
    return np.clip(theta, float(theta_min_deg), float(theta_max_deg))
