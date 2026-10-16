"""Top-attached co-rotating frame: angles, force rotation, and basis contraction.

Frames
------
``F``
    Fixed ground frame. Positive x is heel-to-toe, positive y is up.
``T``
    Top-attached rotating frame, i.e. the reference mesh frame rigidly rotated
    by ``varphi``.

The GUI supplies ``phi``, the absolute fixed-frame angle of the current
heel-to-toe chord. The rigid rotation relative to the reference mesh is

    varphi = phi - phi_ref,
    phi_ref = atan2(y_top(L) - y_top(0), L).

Only ``varphi`` may appear in transformations involving ``Q^T - I``.

Deliberate approximation
------------------------
Boundary positions are transformed with the exact finite rotation ``Q(varphi)``
(true ``sin``/``cos``), while foam strain is still evaluated with the linear
strain-displacement operator. That is intentional: the formulation stays
infinitesimal-strain and linear-elastic so the boundary operators can be
precomputed once, independently of ``phi``. The consequence is that the model
is fast but not objective for locally large rotations.
"""

from __future__ import annotations

import numpy as np

# Sign of the single top shape mode, defined in exactly one place.
#
# s(x) is the convex, increasing softplus ramp, so the endpoint-fixed mode
# phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L) is non-positive in the interior
# with its minimum near x = L - toe_length. Its mean slope on [L - toe_length, L]
# is (L - toe_length)/L > 0, so a positive amplitude alpha = tan(theta) already
# tilts the toe segment upward relative to the heel-to-toe chord. No sign flip is required; flip this
# constant (and nothing else) if the softplus convention ever changes.
SHAPE_MODE_SIGN = 1.0

def reference_chord_angle(x_top: np.ndarray, y_top: np.ndarray | None) -> float:
    """Reference heel-to-toe chord angle phi_ref = atan2(y_top(L) - y_top(0), L).

    Returns 0.0 for a horizontal (or missing) top profile.
    """
    if y_top is None:
        return 0.0
    x = np.asarray(x_top, dtype=float).reshape(-1)
    y = np.asarray(y_top, dtype=float).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return 0.0
    return float(np.arctan2(float(y[-1] - y[0]), span))


def rotate_vector_to_fixed(vx_local: np.ndarray, vy_local: np.ndarray, varphi: float):
    """Return (vx^F, vy^F) = Q(varphi) (vx^T, vy^T), elementwise over arrays."""
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    vx = np.asarray(vx_local, dtype=float)
    vy = np.asarray(vy_local, dtype=float)
    return c * vx - s * vy, s * vx + c * vy


def transform_to_fixed_frame(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    u_local: np.ndarray,
    v_local: np.ndarray,
    x_anchor: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    anchor: tuple[float, float] | None = None,
    elastic_scale: float = 1.0,
    y_anchor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map rotating-frame material points to the fixed frame.

        r^F = r_a^F + Q(varphi) [ (X - X_a) + s_d (d^T - d_a^T) ]

    With ``elastic_scale = 1`` this is the exact transformation
    ``r^F = r_a^F + Q(varphi)[(X + d^T) - (X_a + d_a^T)]``. Larger values
    exaggerate only the elastic part; the rigid rotation is never scaled.

    ``X_a = (x_anchor, y_anchor)`` is a material reference point (the display
    pivot, ``contact_model.display_pivot``) with local displacement ``d_a``;
    ``anchor`` is its true fixed-frame position ``r_a^F`` (default ``(x_anchor, 0)``).
    """
    if anchor is None:
        anchor = (float(x_anchor), 0.0)
    dx = np.asarray(x_ref, dtype=float) - float(x_anchor)
    dy = np.asarray(y_ref, dtype=float) - float(y_anchor)
    du = np.asarray(u_local, dtype=float) - float(d_ax)
    dv = np.asarray(v_local, dtype=float) - float(d_ay)
    s = float(elastic_scale)
    px = dx + s * du
    py = dy + s * dv
    rx, ry = rotate_vector_to_fixed(px, py, varphi)
    return float(anchor[0]) + rx, float(anchor[1]) + ry
