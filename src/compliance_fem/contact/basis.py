"""Affine boundary displacement basis for the co-rotating interval contact lookup.

There is exactly one top shape mode (``n_s = 1``) and four contact-motion
modes, giving five phi-independent linear basis responses per interval:

    0: top_shape_alpha         top u=0, v=phi1(x);  contact u=v=0
    1: contact_translation_x   top u=v=0;           contact u=1,       v=0
    2: contact_translation_y   top u=v=0;           contact u=0,       v=1
    3: contact_rotation_x      top u=v=0;           contact u=x-x_a,   v=0
    4: contact_rotation_y      top u=v=0;           contact u=0,       v=x-x_a

plus one affine constant column, the curved-sole closure:

    C: curved_sole_closure     top u=v=0;           contact u=0,       v=-(y_b(x)-y_a)

Every stored response tensor uses six affine columns ``[C, 0, 1, 2, 3, 4]``;
column 0 always has runtime coefficient exactly 1, so the closure is included
once and only once by :func:`compliance_fem.contact.corotation.contract_basis`.

The rotation-mode anchor ``(x_a, y_a)`` is the interval midpoint
``x_a = (x_i + x_j)/2`` with ``y_a`` interpolated from the reference bottom
profile. It is a numerical anchor, not a physical contact edge.

Why the closure is constant: the contact law places every active node on the
ground at its reference ``x`` (sticking). In the rotating frame this requires

    d_k = d_a + (Q^T - I)(x_k - x_a, 0) + (0, -(y_k - y_a)),

and ``r_k^F = r_a^F + Q[(X_k - X_a) + d_k - d_a] = (x_k, 0)`` exactly. The rotation
modes act on the flattened offset ``(x_k - x_a, 0)``; the vertical flattening
``-(y_k - y_a)`` is independent of ``varphi`` and is never rotated. For a flat
sole the closure column is identically zero.

No co-rotation angle appears anywhere in this module.
"""

from __future__ import annotations

import numpy as np

from compliance_fem.contact.corotation import SHAPE_MODE_SIGN

N_BASIS_MODES = 5
BASIS_MODE_NAMES = (
    "top_shape_alpha",
    "contact_translation_x",
    "contact_translation_y",
    "contact_rotation_x",
    "contact_rotation_y",
)
BASIS_ORDER = list(BASIS_MODE_NAMES)

MODE_ALPHA = 0
MODE_PHI1 = MODE_ALPHA
MODE_BX = 1
MODE_BY = 2
MODE_BRX = 3
MODE_BRY = 4

# Affine storage: column 0 is the curved-sole closure (coefficient fixed at 1).
N_AFFINE_COLUMNS = N_BASIS_MODES + 1
AFFINE_CONSTANT_NAME = "curved_sole_closure"
AFFINE_COLUMN_NAMES = (AFFINE_CONSTANT_NAME, *BASIS_MODE_NAMES)
COL_CONST = 0
COL_ALPHA = MODE_ALPHA + 1
COL_BX = MODE_BX + 1
COL_BY = MODE_BY + 1
COL_BRX = MODE_BRX + 1
COL_BRY = MODE_BRY + 1
CURVED_SOLE_CLOSURE_CONVENTION = (
    "affine column 0 of every response tensor; coefficient fixed at 1; "
    "prescribed rotating-frame contact displacement u=0, v=-(y_b(x_k)-y_a) with the top clamped; "
    "never multiplied by Q(varphi)"
)

SHAPE_MODE_DEFINITION = "phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L), s = softplus ramp"
SHAPE_MODE_NORMALIZATION = "none"


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


def build_top_displacement_matrix(
    x_top: np.ndarray,
    L: float,
    toe_length: float,
    kappa: float,
) -> np.ndarray:
    """Return W_t with shape (2 n_t, 5), component-major [u; v].

    Only mode 0 prescribes nonzero top displacement (u=0, v=phi1). The four
    contact-motion modes clamp the whole top boundary in the rotating frame.
    """
    x_top = np.asarray(x_top, dtype=float)
    n = int(x_top.size)
    W = np.zeros((2 * n, N_BASIS_MODES), dtype=float)
    W[n:, MODE_ALPHA] = shape_mode_phi1(x_top, L=L, toe_length=toe_length, kappa=kappa)
    return W


def build_top_affine_matrix(x_top: np.ndarray, L: float, toe_length: float, kappa: float) -> np.ndarray:
    """Affine top displacement matrix (2 n_t, 6); the closure column is zero on the top."""
    W5 = build_top_displacement_matrix(x_top, L=L, toe_length=toe_length, kappa=kappa)
    return np.concatenate([np.zeros((W5.shape[0], 1)), W5], axis=1)


def build_contact_displacement_matrix(x_contact: np.ndarray, x_anchor: float) -> np.ndarray:
    """Return W_c with shape (2 n_c, 5), component-major [u; v].

    Columns 1-4 hold B_x, B_y, B_rx(.; x_anchor), B_ry(.; x_anchor) acting on the
    flattened offset ``ds = x - x_anchor``; mode 0 is zero. ``ds < 0`` heel-side
    of the anchor, ``ds > 0`` toe-side, ``ds = 0`` exactly at the anchor.
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


def curved_sole_closure(y_contact: np.ndarray | None, y_anchor: float, n_contact: int) -> np.ndarray:
    """Closure displacement (2 n_c,) component-major: u = 0, v = -(y_k - y_a)."""
    out = np.zeros(2 * int(n_contact), dtype=float)
    if y_contact is None or int(n_contact) == 0:
        return out
    out[int(n_contact) :] = -(np.asarray(y_contact, dtype=float) - float(y_anchor))
    return out


def build_contact_affine_matrix(
    x_contact: np.ndarray,
    x_anchor: float,
    y_contact: np.ndarray | None = None,
    y_anchor: float = 0.0,
) -> np.ndarray:
    """Affine contact displacement matrix (2 n_c, 6): ``[closure, W_c]``."""
    W5 = build_contact_displacement_matrix(x_contact, x_anchor)
    n = int(np.asarray(x_contact).size)
    closure = curved_sole_closure(y_contact, y_anchor, n)
    return np.concatenate([closure[:, None], W5], axis=1)
