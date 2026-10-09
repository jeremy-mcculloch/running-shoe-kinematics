"""Laboratory ↔ heel-attached foot-frame transforms for COP and forces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class LabSagittalPoint:
    """Point in the laboratory/OpenSim ground sagittal plane (+x anterior, +y up)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]


@dataclass(frozen=True)
class FootFramePoint:
    """Point in the heel-attached foot frame (+x heel→toe along φ, +y foot-normal)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]


def rotation_Q(phi: float | NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(cos φ, sin φ)`` for the planar rotation Q(φ)."""
    phi = np.asarray(phi, dtype=np.float64)
    return np.cos(phi), np.sin(phi)


def lab_vector_to_foot(
    vx_lab: NDArray[np.float64] | float,
    vy_lab: NDArray[np.float64] | float,
    phi: NDArray[np.float64] | float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate a lab sagittal vector into the foot frame: ``v_foot = Q(φ)^T v_lab``."""
    c, s = rotation_Q(phi)
    vx = np.asarray(vx_lab, dtype=np.float64)
    vy = np.asarray(vy_lab, dtype=np.float64)
    return c * vx + s * vy, -s * vx + c * vy


def foot_vector_to_lab(
    vx_foot: NDArray[np.float64] | float,
    vy_foot: NDArray[np.float64] | float,
    phi: NDArray[np.float64] | float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate a foot-frame vector into the lab frame: ``v_lab = Q(φ) v_foot``."""
    c, s = rotation_Q(phi)
    vx = np.asarray(vx_foot, dtype=np.float64)
    vy = np.asarray(vy_foot, dtype=np.float64)
    return c * vx - s * vy, s * vx + c * vy


def cop_lab_to_foot(
    cop_lab_x: NDArray[np.float64] | float,
    cop_lab_y: NDArray[np.float64] | float,
    heel_lab_x: NDArray[np.float64] | float,
    heel_lab_y: NDArray[np.float64] | float,
    phi: NDArray[np.float64] | float,
) -> FootFramePoint:
    """Global COP → heel-attached foot frame.

    ``p_foot = Q(φ)^T (p_lab − r_heel_lab)``.
    """
    dx = np.asarray(cop_lab_x, dtype=np.float64) - np.asarray(heel_lab_x, dtype=np.float64)
    dy = np.asarray(cop_lab_y, dtype=np.float64) - np.asarray(heel_lab_y, dtype=np.float64)
    x_f, y_f = lab_vector_to_foot(dx, dy, phi)
    return FootFramePoint(x=np.asarray(x_f, dtype=np.float64), y=np.asarray(y_f, dtype=np.float64))


def cop_foot_to_lab(
    cop_foot_x: NDArray[np.float64] | float,
    cop_foot_y: NDArray[np.float64] | float,
    heel_lab_x: NDArray[np.float64] | float,
    heel_lab_y: NDArray[np.float64] | float,
    phi: NDArray[np.float64] | float,
) -> LabSagittalPoint:
    """Inverse of :func:`cop_lab_to_foot`."""
    dx, dy = foot_vector_to_lab(cop_foot_x, cop_foot_y, phi)
    return LabSagittalPoint(
        x=np.asarray(heel_lab_x, dtype=np.float64) + dx,
        y=np.asarray(heel_lab_y, dtype=np.float64) + dy,
    )


def force_induced_moment_about_point(
    fx: NDArray[np.float64] | float,
    fy: NDArray[np.float64] | float,
    point_x: NDArray[np.float64] | float,
    point_y: NDArray[np.float64] | float,
    origin_x: float | NDArray[np.float64] = 0.0,
    origin_y: float | NDArray[np.float64] = 0.0,
) -> NDArray[np.float64]:
    """Sagittal moment of force through ``point`` about ``origin``: (r×F)_z."""
    rx = np.asarray(point_x, dtype=np.float64) - np.asarray(origin_x, dtype=np.float64)
    ry = np.asarray(point_y, dtype=np.float64) - np.asarray(origin_y, dtype=np.float64)
    return rx * np.asarray(fy, dtype=np.float64) - ry * np.asarray(fx, dtype=np.float64)


def translate_moment_between_points(
    mz_o: NDArray[np.float64] | float,
    fx: NDArray[np.float64] | float,
    fy: NDArray[np.float64] | float,
    *,
    from_xy: tuple[float, float] | tuple[NDArray[np.float64], NDArray[np.float64]],
    to_xy: tuple[float, float] | tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """``M_A = M_O - [(r_A - r_O) × F]_z``."""
    ox, oy = from_xy
    ax, ay = to_xy
    dx = np.asarray(ax, dtype=np.float64) - np.asarray(ox, dtype=np.float64)
    dy = np.asarray(ay, dtype=np.float64) - np.asarray(oy, dtype=np.float64)
    return (
        np.asarray(mz_o, dtype=np.float64)
        - dx * np.asarray(fy, dtype=np.float64)
        + dy * np.asarray(fx, dtype=np.float64)
    )


def model_cop_from_wrench(
    mz_about_origin: NDArray[np.float64] | float,
    fy: NDArray[np.float64] | float,
    *,
    fy_min: float = 20.0,
) -> NDArray[np.float64]:
    """Longitudinal COP about the model origin from ``x ≈ M_z / F_y`` (ground y≈0)."""
    fy = np.asarray(fy, dtype=np.float64)
    mz = np.asarray(mz_about_origin, dtype=np.float64)
    out = np.full(np.broadcast(mz, fy).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(fy) & np.isfinite(mz) & (np.abs(fy) >= float(fy_min))
    out[ok] = mz[ok] / fy[ok]
    return out


def cop_validity_mask(
    *,
    fy_lab_vertical: NDArray[np.float64],
    cop_foot_x: NDArray[np.float64],
    shoe_length: float,
    fy_min: float = 20.0,
    x_tol: float = 0.05,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return ``(cop_valid, out_of_shoe_flag)``.

    ``cop_valid`` requires |F_y| ≥ fy_min (lab vertical GRF, typically upward +)
    and finite heel-relative COP. Out-of-shoe is flagged but not clipped.
    """
    fy = np.asarray(fy_lab_vertical, dtype=np.float64)
    x = np.asarray(cop_foot_x, dtype=np.float64)
    force_ok = np.isfinite(fy) & (np.abs(fy) >= float(fy_min))
    finite = np.isfinite(x)
    valid = force_ok & finite
    L = float(shoe_length)
    out_of_shoe = valid & ((x < -float(x_tol)) | (x > L + float(x_tol)))
    return valid, out_of_shoe


def interpolate_marker_xy(
    t_src: NDArray[np.float64],
    xy: NDArray[np.float64],
    t_dst: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear interpolate marker ``(n,2|3)`` sagittal xy onto ``t_dst``."""
    t_src = np.asarray(t_src, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    out = np.full((t_dst.size, 2), np.nan, dtype=np.float64)
    if t_src.size < 2:
        return out
    for k in range(2):
        col = xy[:, k]
        mask = np.isfinite(t_src) & np.isfinite(col)
        if np.count_nonzero(mask) < 2:
            continue
        out[:, k] = np.interp(t_dst, t_src[mask], col[mask])
    return out
