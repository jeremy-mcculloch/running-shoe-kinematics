"""Sagittal projection, force-sign convention, and wrench algebra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SagittalWrench:
    """Model-fixed sagittal wrench at reference point O.

    Sign convention (documented + tested)
    ------------------------------------
    OpenSim / Wang lab: ground-on-body force, +x anterior, +y superior, +z right.

    Shoe model fixed frame: +x heel→toe (anterior), +y up. Force-control top
    resultants are the force applied **to the shoe top** (foot→shoe). Action–
    reaction with a massless shoe contact patch implies

        F_top ≈ −F_GRF_ground_on_body

    so an upward measured GRF (+Fy_lab) becomes a downward (compressive) top
    load (Fy_model < 0). Moments follow the same sign flip of the force couple
    when mapping ground-on-body → top-of-shoe wrench about the same material
    origin after translating COP into the shoe frame.
    """

    Fx: NDArray[np.float64]
    Fy: NDArray[np.float64]
    Mz: NDArray[np.float64]
    cop_x: NDArray[np.float64]
    cop_y: NDArray[np.float64]
    Fz_omitted: NDArray[np.float64]
    times: NDArray[np.float64]


def moment_about_origin(
    fx: float | NDArray[np.float64],
    fy: float | NDArray[np.float64],
    cop_x: float | NDArray[np.float64],
    cop_y: float | NDArray[np.float64],
    *,
    x_o: float = 0.0,
    y_o: float = 0.0,
    mz_free: float | NDArray[np.float64] = 0.0,
) -> NDArray[np.float64]:
    """Sagittal moment about O: ``(x_cop-x_o) Fy - (y_cop-y_o) Fx + M_free``."""
    return (
        (np.asarray(cop_x, dtype=np.float64) - x_o) * np.asarray(fy, dtype=np.float64)
        - (np.asarray(cop_y, dtype=np.float64) - y_o) * np.asarray(fx, dtype=np.float64)
        + np.asarray(mz_free, dtype=np.float64)
    )


def translate_moment(
    mz_a: float | NDArray[np.float64],
    fx: float | NDArray[np.float64],
    fy: float | NDArray[np.float64],
    *,
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
) -> NDArray[np.float64]:
    """Translate planar moment from point A to point B: ``M_B = M_A + r_{A→B} × F``."""
    dx = from_xy[0] - to_xy[0]
    dy = from_xy[1] - to_xy[1]
    # r_A_from_B × F = dx*Fy - dy*Fx when going A→ expressed at B...
    # M_B = M_A + (x_A - x_B) Fy - (y_A - y_B) Fx
    return (
        np.asarray(mz_a, dtype=np.float64)
        + dx * np.asarray(fy, dtype=np.float64)
        - dy * np.asarray(fx, dtype=np.float64)
    )


def lab_grf_to_model_top_wrench(
    *,
    times: NDArray[np.float64],
    fx_lab: NDArray[np.float64],
    fy_lab: NDArray[np.float64],
    fz_lab: NDArray[np.float64],
    cop_x_lab: NDArray[np.float64],
    cop_y_lab: NDArray[np.float64],
    mz_lab: NDArray[np.float64] | None = None,
    ml_warn_fraction: float = 0.25,
    shoe_origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[SagittalWrench, list[str]]:
    """Project lab GRF into model sagittal top wrench with documented sign flip."""
    warnings: list[str] = []
    fx_lab = np.asarray(fx_lab, dtype=np.float64)
    fy_lab = np.asarray(fy_lab, dtype=np.float64)
    fz_lab = np.asarray(fz_lab, dtype=np.float64)
    # Wang/OpenSim: +x anterior, +y superior → model Fx, Fy same axes before sign flip.
    f_horiz = fx_lab
    f_vert = fy_lab
    f_ml = fz_lab
    # If vertical channel was stored as z-up (rare for this dataset), detect.
    if np.nanmax(np.abs(fy_lab)) < 0.1 * max(np.nanmax(np.abs(fz_lab)), 1.0):
        f_vert = fz_lab
        f_ml = fy_lab
        warnings.append("vertical GRF taken from lab Z (Y looked empty)")

    # Top-of-shoe load ≈ − ground-on-body GRF.
    Fx = -f_horiz
    Fy = -f_vert
    cop_x = np.asarray(cop_x_lab, dtype=np.float64)
    cop_y = np.asarray(cop_y_lab, dtype=np.float64)
    mz_free = np.zeros_like(Fx) if mz_lab is None else -np.asarray(mz_lab, dtype=np.float64)
    Mz = moment_about_origin(
        Fx, Fy, cop_x, cop_y, x_o=shoe_origin_xy[0], y_o=shoe_origin_xy[1], mz_free=mz_free
    )

    f_mag = np.hypot(f_horiz, f_vert)
    ml_frac = np.divide(
        np.abs(f_ml),
        np.maximum(f_mag, 1e-9),
        out=np.zeros_like(f_ml),
        where=f_mag > 1e-9,
    )
    if float(np.nanmax(ml_frac)) > ml_warn_fraction:
        warnings.append(
            f"mediolateral force exceeds {ml_warn_fraction:.0%} of sagittal force "
            f"(peak fraction {float(np.nanmax(ml_frac)):.2f}); sagittal projection omits it"
        )

    wrench = SagittalWrench(
        Fx=Fx,
        Fy=Fy,
        Mz=Mz,
        cop_x=cop_x - shoe_origin_xy[0],
        cop_y=cop_y - shoe_origin_xy[1],
        Fz_omitted=f_ml,
        times=np.asarray(times, dtype=np.float64),
    )
    return wrench, warnings
