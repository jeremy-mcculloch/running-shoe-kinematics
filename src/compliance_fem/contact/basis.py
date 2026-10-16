"""Top shape mode of the toe joint (the only prescribed top displacement).

The top surface is glued to the foot. In the top-attached frame its
displacement is ``alpha * (0, phi1(x))`` with ``alpha = tan(theta)`` and the
endpoint-fixed softplus ramp

    phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L),

so ``theta`` bends the toe segment without moving the heel or toe tip. No
co-rotation angle appears anywhere in this module.
"""

from __future__ import annotations

import numpy as np

from compliance_fem.contact.corotation import SHAPE_MODE_SIGN

def softplus_w2(x: np.ndarray, L: float, toe_length: float, kappa: float) -> np.ndarray:
    """Unnormalized softplus ramp s(x) approaching max(0, x - (L - toe_length)) for large kappa.

    ``L`` and ``toe_length`` are in metres. Evaluated stably as
    (L / kappa) * logaddexp(0, kappa * (x - (L - toe_length)) / L).
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    x = np.asarray(x, dtype=float)
    return (L / kappa) * np.logaddexp(0.0, kappa * (x - (L - toe_length)) / L)


def kappa_from_min_bend_radius(L: float, min_bend_radius_mm: float) -> float:
    """Softplus sharpness whose tightest bend at a 45° toe angle has this radius.

    At θ = 45°, α = tan(θ) = 1, so the chord-relative top displacement is φ₁.
    φ₁'' equals the softplus second derivative and peaks at the joint x = L - toe_length
    with value κ / (4L). Taking that peak as the maximum curvature,

        R_min = 4L / κ,    κ = 4L / R_min.

    ``L`` and the radius must use consistent length units after the millimetre
    conversion below.
    """
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    radius_m = float(min_bend_radius_mm) / 1000.0
    if radius_m <= 0.0:
        raise ValueError(f"min_bend_radius_mm must be positive, got {min_bend_radius_mm}.")
    return 4.0 * float(L) / radius_m


def min_bend_radius_mm_from_kappa(L: float, kappa: float) -> float:
    """Inverse of :func:`kappa_from_min_bend_radius`."""
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    return 4.0 * float(L) * 1000.0 / float(kappa)


def shape_mode_phi1(x: np.ndarray, L: float, toe_length: float, kappa: float) -> np.ndarray:
    """Endpoint-fixed top shape mode phi1(x), zero at x=0 and x=L.

    phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L)

    No peak, norm, or slope normalization is applied. The overall sign lives in
    ``corotation.SHAPE_MODE_SIGN``; positive theta tilts the toe segment up
    relative to the heel-to-toe chord.
    """
    x = np.asarray(x, dtype=float)
    s = softplus_w2(x, L=L, toe_length=toe_length, kappa=kappa)
    s0 = float(softplus_w2(np.array([0.0]), L=L, toe_length=toe_length, kappa=kappa)[0])
    sL = float(softplus_w2(np.array([float(L)]), L=L, toe_length=toe_length, kappa=kappa)[0])
    t = x / float(L)
    return SHAPE_MODE_SIGN * (s - (1.0 - t) * s0 - t * sL)


def top_shape_displacement(x_top: np.ndarray, L: float, toe_length: float, kappa: float) -> np.ndarray:
    """Top displacement per unit ``alpha``, component-major ``[u; v] = [0; phi1(x)]`` (2 n_t,)."""
    x_top = np.asarray(x_top, dtype=float)
    return np.concatenate([np.zeros(x_top.size), shape_mode_phi1(x_top, L=L, toe_length=toe_length, kappa=kappa)])
