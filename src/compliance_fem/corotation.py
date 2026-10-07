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
infinitesimal-strain and linear-elastic so the five basis responses can be
precomputed once, independently of ``phi``, and combined by scalars at runtime.
The consequence is that the model is fast but not objective for locally large
rotations.
"""

from __future__ import annotations

import numpy as np

# Sign of the single top shape mode, defined in exactly one place.
#
# s(x) is the convex, increasing softplus ramp, so the endpoint-fixed mode
# phi1(x) = s(x) - (1 - x/L) s(0) - (x/L) s(L) is non-positive in the interior
# with its minimum near x = a. Its mean slope on [a, L] is a/L > 0, so a
# positive amplitude alpha = tan(theta) already tilts the toe segment upward
# relative to the heel-to-toe chord. No sign flip is required; flip this
# constant (and nothing else) if the softplus convention ever changes.
SHAPE_MODE_SIGN = 1.0

N_BASIS_COEFFICIENTS = 5

# Runtime coefficient names in saved basis order. The two translation entries
# are the interval-anchor translation solved from the force-controlled 2x2 system.
BASIS_COEFFICIENT_NAMES = (
    "alpha_tan_theta",
    "d_anchor_x",
    "d_anchor_y",
    "cos_varphi_minus_1",
    "minus_sin_varphi",
)


def rotation_matrix(varphi: float) -> np.ndarray:
    """Return Q(varphi) = [[cos, -sin], [sin, cos]] (counterclockwise positive)."""
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    return np.array([[c, -s], [s, c]], dtype=float)


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


def corotation_angle(phi: float, phi_ref: float = 0.0) -> float:
    """Rigid rotation of the mesh relative to the reference configuration."""
    return float(phi) - float(phi_ref)


def rotate_force_to_local(Fx_fixed: float, Fy_fixed: float, varphi: float) -> np.ndarray:
    """Return F^T = Q(varphi)^T F^F for a prescribed fixed-frame force."""
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    return np.array(
        [c * float(Fx_fixed) + s * float(Fy_fixed), -s * float(Fx_fixed) + c * float(Fy_fixed)],
        dtype=float,
    )


def rotate_vector_to_fixed(vx_local: np.ndarray, vy_local: np.ndarray, varphi: float):
    """Return (vx^F, vy^F) = Q(varphi) (vx^T, vy^T), elementwise over arrays."""
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    vx = np.asarray(vx_local, dtype=float)
    vy = np.asarray(vy_local, dtype=float)
    return c * vx - s * vy, s * vx + c * vy


def fixed_frame_normal_component(dx_local: np.ndarray, dy_local: np.ndarray, varphi: float) -> np.ndarray:
    """Ground-normal (fixed-frame y) component of a local vector.

    Used for both the signed nonpenetration gap and the compression-only
    normal reaction: ``sin(varphi) dx^T + cos(varphi) dy^T``.
    """
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    return s * np.asarray(dx_local, dtype=float) + c * np.asarray(dy_local, dtype=float)


def fixed_frame_tangential_component(dx_local: np.ndarray, dy_local: np.ndarray, varphi: float) -> np.ndarray:
    """Ground-tangential (fixed-frame x) component: cos dx^T - sin dy^T."""
    c = float(np.cos(varphi))
    s = float(np.sin(varphi))
    return c * np.asarray(dx_local, dtype=float) - s * np.asarray(dy_local, dtype=float)


def rotation_coefficients(varphi: float) -> tuple[float, float]:
    """Return (r_x, r_y) = (cos(varphi) - 1, -sin(varphi))."""
    return float(np.cos(varphi) - 1.0), float(-np.sin(varphi))


def shape_amplitude(theta: float) -> float:
    """Return alpha = tan(theta), rejecting angles at +-90 degrees."""
    theta = float(theta)
    if abs(abs(theta) - 0.5 * np.pi) < 1e-12:
        raise ValueError(f"theta={theta} rad is at +-90 degrees; tan(theta) is undefined.")
    return float(np.tan(theta))


def basis_coefficients(
    alpha: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
) -> np.ndarray:
    """Return gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)].

    The ordering matches the saved basis ordering
    ``(top_shape_alpha, contact_translation_x, contact_translation_y,
    contact_rotation_x, contact_rotation_y)``.

    ``(d_ax, d_ay)`` is the interval-anchor translation from the
    force-controlled 2x2 solve.
    """
    r_x, r_y = rotation_coefficients(varphi)
    return np.array([float(alpha), float(d_ax), float(d_ay), r_x, r_y], dtype=float)


def known_coefficients(alpha: float, varphi: float) -> np.ndarray:
    """gamma with the two unknown translation entries zeroed.

    Contracting a basis with this vector yields the ``known`` contribution used
    by the runtime 2x2 change of basis.
    """
    return basis_coefficients(alpha, 0.0, 0.0, varphi)


N_AFFINE_COEFFICIENTS = N_BASIS_COEFFICIENTS + 1


def affine_coefficients(gamma: np.ndarray) -> np.ndarray:
    """Prepend the fixed closure coefficient: ``[1, gamma_0..gamma_4]``."""
    g = np.asarray(gamma, dtype=float).reshape(-1)
    if g.size == N_AFFINE_COEFFICIENTS:
        if g[0] != 1.0:
            raise ValueError("Affine coefficient vectors must have column-0 coefficient exactly 1.")
        return g
    if g.size != N_BASIS_COEFFICIENTS:
        raise ValueError(
            f"gamma must have {N_BASIS_COEFFICIENTS} entries in saved basis order, got {g.size}."
        )
    return np.concatenate([[1.0], g])


def contract_basis(basis: np.ndarray, gamma: np.ndarray, mode_axis: int = 0) -> np.ndarray:
    """Contract the mode axis of ``basis`` with ``gamma``.

    Single reusable contraction so the coefficient ordering cannot diverge
    between scalar outputs, nodal vectors, and plotting fields.

    Parameters
    ----------
    basis
        Array whose ``mode_axis`` has length 6 (affine storage: column 0 is the
        curved-sole closure with coefficient 1, columns 1-5 the linear modes) or
        length 5 (linear modes only).
    gamma
        Length-5 runtime coefficient vector in saved basis order. For an affine
        basis the closure coefficient 1 is prepended here, exactly once.
    mode_axis
        Axis of ``basis`` holding the basis modes. Defaults to the leading axis.
    """
    g = np.asarray(gamma, dtype=float).reshape(-1)
    arr = np.asarray(basis, dtype=float)
    axis = mode_axis if mode_axis >= 0 else arr.ndim + mode_axis
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"mode_axis={mode_axis} is out of range for array with ndim={arr.ndim}.")
    n_axis = arr.shape[axis]
    if n_axis == N_AFFINE_COEFFICIENTS:
        g = affine_coefficients(g)
    elif n_axis == N_BASIS_COEFFICIENTS:
        if g.size != N_BASIS_COEFFICIENTS:
            raise ValueError(
                f"gamma must have {N_BASIS_COEFFICIENTS} entries in saved basis order, got {g.size}."
            )
    else:
        raise ValueError(
            f"basis axis {axis} has length {n_axis}; expected "
            f"{N_AFFINE_COEFFICIENTS} (affine) or {N_BASIS_COEFFICIENTS} (linear)."
        )
    return np.tensordot(g, arr, axes=([0], [axis]))


def contract_affine_rows(basis_rows: np.ndarray, gamma_rows: np.ndarray) -> np.ndarray:
    """Row-wise affine contraction ``out[r, ...] = basis[r, 0, ...] + sum_k gamma[r, k] basis[r, k+1, ...]``.

    ``basis_rows`` has shape ``(n_rows, 6, ...)`` and ``gamma_rows`` ``(n_rows, 5)``.
    """
    B = np.asarray(basis_rows, dtype=float)
    G = np.asarray(gamma_rows, dtype=float)
    if B.shape[1] != N_AFFINE_COEFFICIENTS or G.shape[-1] != N_BASIS_COEFFICIENTS:
        raise ValueError("contract_affine_rows expects (n, 6, ...) basis and (n, 5) gamma.")
    tail = B.shape[2:]
    flat = B.reshape(B.shape[0], N_AFFINE_COEFFICIENTS, -1)
    out = flat[:, 0, :] + np.einsum("rk,rkm->rm", G, flat[:, 1:, :])
    return out.reshape((B.shape[0], *tail))


def contact_displacement(
    x_contact: np.ndarray,
    x_anchor: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    y_contact: np.ndarray | None = None,
    y_anchor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact prescribed contact displacement in the rotating frame.

    ``d_c^T(x) = d_a^T + (Q(varphi)^T - I)(x - x_a, 0) + (0, -(y - y_a))``:

        u_c = d_ax + (cos(varphi) - 1)(x - x_anchor)
        v_c = d_ay - sin(varphi)(x - x_anchor) - (y - y_anchor)

    The last term is the curved-sole closure (zero for a flat sole); it is
    independent of ``varphi`` and is not rotated. The expression is identical
    for every interval, with no sign branching on the topology.

    The apparent infinitesimal axial strain in the ``u_c`` term is an accepted
    approximation of this project; it is not replaced by a small-angle form.
    """
    x = np.asarray(x_contact, dtype=float)
    ds = x - float(x_anchor)
    r_x, r_y = rotation_coefficients(varphi)
    u = float(d_ax) + r_x * ds
    v = float(d_ay) + r_y * ds
    if y_contact is not None:
        v = v - (np.asarray(y_contact, dtype=float) - float(y_anchor))
    return u, v


def transform_to_fixed_frame(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    u_local: np.ndarray,
    v_local: np.ndarray,
    x_anchor: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    anchor: tuple[float, float] = None,
    elastic_scale: float = 1.0,
    y_anchor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map rotating-frame material points to the fixed frame.

        r^F = r_a^F + Q(varphi) [ (X - X_a) + s_d (d^T - d_a^T) ]

    With ``elastic_scale = 1`` this is the exact transformation
    ``r^F = r_a^F + Q(varphi)[(X + d^T) - (X_a + d_a^T)]``. Larger values
    exaggerate only the elastic part; the rigid rotation is never scaled.

    ``X_a = (x_anchor, y_anchor)`` is the interval-midpoint anchor on the
    reference bottom profile. ``anchor`` is the visualization and reporting
    gauge ``r_a^F``; it defaults to the ground point ``(x_anchor, 0)``.
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
