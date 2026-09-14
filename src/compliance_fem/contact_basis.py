"""Five-mode boundary displacement basis for the co-rotating contact lookup.

There is exactly one top shape mode (``n_s = 1``) and four contact-motion
modes, giving five phi-independent basis responses per candidate:

    0: top_shape_phi1          top u=0, v=phi1(x);  contact u=v=0
    1: contact_translation_x   top u=v=0;           contact u=1,      v=0
    2: contact_translation_y   top u=v=0;           contact u=0,      v=1
    3: contact_rotation_x      top u=v=0;           contact u=x-x_a,  v=0
    4: contact_rotation_y      top u=v=0;           contact u=0,      v=x-x_a

The rotation-mode anchor ``x_a`` is the contact edge ``l_i`` for heel and toe
contact and the numerical anchor ``L/2`` for full contact. The same five modes,
in this order, are stored for every topology.

No co-rotation angle appears anywhere in this module.
"""

from __future__ import annotations

import numpy as np

from compliance_fem.corotation import SHAPE_MODE_SIGN

N_BASIS_MODES = 5
BASIS_MODE_NAMES = (
    "top_shape_phi1",
    "contact_translation_x",
    "contact_translation_y",
    "contact_rotation_x",
    "contact_rotation_y",
)

MODE_PHI1 = 0
MODE_BX = 1
MODE_BY = 2
MODE_BRX = 3
MODE_BRY = 4

SHAPE_MODE_DEFINITION = "phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L), s = softplus ramp"
SHAPE_MODE_NORMALIZATION = "none"


def softplus_w2(x: np.ndarray, L: float, a: float, kappa: float) -> np.ndarray:
    """Unnormalized softplus ramp s(x) approaching max(0, x - a) for large kappa.

    Evaluated stably as (L / kappa) * logaddexp(0, kappa * (x - a) / L).
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}.")
    if L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")
    x = np.asarray(x, dtype=float)
    return (L / kappa) * np.logaddexp(0.0, kappa * (x - a) / L)


def shape_mode_phi1(x: np.ndarray, L: float, a: float, kappa: float) -> np.ndarray:
    """Endpoint-fixed top shape mode phi1(x), zero at x=0 and x=L.

    phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L)

    No peak, norm, or slope normalization is applied. The overall sign lives in
    ``corotation.SHAPE_MODE_SIGN``; positive theta tilts the toe segment up
    relative to the heel-to-toe chord.
    """
    x = np.asarray(x, dtype=float)
    s = softplus_w2(x, L=L, a=a, kappa=kappa)
    s0 = float(softplus_w2(np.array([0.0]), L=L, a=a, kappa=kappa)[0])
    sL = float(softplus_w2(np.array([float(L)]), L=L, a=a, kappa=kappa)[0])
    t = x / float(L)
    return SHAPE_MODE_SIGN * (s - (1.0 - t) * s0 - t * sL)


def build_top_displacement_matrix(
    x_top: np.ndarray,
    L: float,
    a: float,
    kappa: float,
) -> np.ndarray:
    """Return W_t with shape (2 n_t, 5), component-major [u; v].

    Only mode 0 prescribes nonzero top displacement (u=0, v=phi1). The four
    contact-motion modes clamp the whole top boundary in the rotating frame.
    """
    x_top = np.asarray(x_top, dtype=float)
    n = int(x_top.size)
    W = np.zeros((2 * n, N_BASIS_MODES), dtype=float)
    W[n:, MODE_PHI1] = shape_mode_phi1(x_top, L=L, a=a, kappa=kappa)
    return W


def build_contact_displacement_matrix(x_contact: np.ndarray, x_anchor: float) -> np.ndarray:
    """Return W_c with shape (2 n_c, 5), component-major [u; v].

    Columns 1-4 hold B_x, B_y, B_rx(.; x_anchor), B_ry(.; x_anchor); mode 0 is
    zero. ``x_anchor`` is the contact edge ``l_i`` for heel or toe contact and
    the numerical anchor ``L/2`` for full contact.

    The single expression ``ds = x - x_anchor`` covers every topology with no
    sign branching: ``ds <= 0`` over a heel contact set, ``ds >= 0`` over a toe
    contact set, and ``ds = 0`` exactly at the anchor.
    """
    x_c = np.asarray(x_contact, dtype=float)
    n = int(x_c.size)
    W = np.zeros((2 * n, N_BASIS_MODES), dtype=float)
    if n == 0:
        return W
    ds = x_c - float(x_anchor)
    W[:n, MODE_BX] = 1.0
    W[n:, MODE_BY] = 1.0
    W[:n, MODE_BRX] = ds
    W[n:, MODE_BRY] = ds
    return W
