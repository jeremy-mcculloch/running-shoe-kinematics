"""Shared boundary-condensed contact operator (replaces per-interval records).

Frames and signs
----------------
``T`` is the top-attached (co-rotated) frame, ``F`` the fixed ground frame
(x heel -> toe, y up, ground at ``y = 0``). The top surface is glued to the
foot: its local displacement is prescribed, ``d_t = alpha shape_mode_toe_bend`` with
``shape_mode_toe_bend = (0, phi1(x))`` and ``alpha = tan(theta)``. Every bottom node may
receive a ground force; nothing else is loaded.

Given the local bottom nodal forces ``f_b`` (component-major ``[f_x...; f_y...]``)
the top-clamped body response is unique (the clamp removes the rigid modes):

    d_b = alpha db_toe + db_fb f_b          (bottom displacement, local)
    F_t = alpha ft_toe + ft_fb f_b          (top nodal force = foot on shoe, local)
    a   = alpha r_toe + r_fb f_b            (rigid amplitudes of the compliance gauge)

with ``R_t^T F_t + R_b^T f_b = 0`` (free-body equilibrium) built in. ``db_fb`` is
the top-clamped bottom compliance: symmetric positive definite. Everything is
assembled once from the (reciprocity-checked, symmetrized) compliance blocks
``C_tt, C_tb, C_bt, C_bb`` of a :class:`~compliance_fem.fem.compliance.ComplianceResult`
by factoring the small ``(2 n_t + 3)`` matrix ``[[C_tt, R_t], [R_t^T, 0]]``; no FEM
solve and no explicit inverse is used at runtime.

``a`` are the rigid amplitudes about ``(x_r, y_r)`` that the plate and render
influence matrices expect, so full fields are recovered from ``[F_t; f_b]``
and ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg

from compliance_fem.contact.basis import toe_bend_displacement
from compliance_fem.contact.corotation import reference_chord_angle
from compliance_fem.contact.toe_spring import (
    RearfootGeometry,
    rearfoot_geometry,
    rearfoot_toe_mode,
    warn_if_mtp_not_on_top_grid,
)
from compliance_fem.fem.boundaries import vector_rigid_mode_matrix

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult

# Rigid modes of the compliance gauge are about (L/2, SOLVER_YR); the plate and render
# influence matrices use the same reference point.
SOLVER_YR = 0.0
CONTACT_OPERATOR_PREFIX = "contact_operator_"
CONTACT_OPERATOR_ARRAYS = (
    "x_top",
    "y_top",
    "x_bottom",
    "y_bottom",
    "shape_mode_toe_bend",
    "db_fb",
    "db_toe",
    "ft_fb",
    "ft_toe",
    "r_fb",
    "r_toe",
    "toe_shape_mode_flat_rearfoot",
)
CONTACT_OPERATOR_SCALARS = (
    "L",
    "H",
    "toe_length",
    "kappa",
    "phi_ref",
    "x_r",
    "y_r",
    "h_mesh",
    "rearfoot_y_mtp_shape_mode",
    "rearfoot_y_mtp_0",
)
CONTACT_FORMULATION = (
    "node-to-ground unilateral contact 0 <= lambda perp g >= 0 on every bottom node, "
    "g = T_y + sin(varphi)(x+u) + cos(varphi)(y+v); proportional shear (tau, lambda) = "
    "(mu_eff, 1) lambda with mu_eff = Fx*/Fy*; sum(lambda) = -Fy* with T_y its multiplier"
)
TANGENTIAL_CONTACT_LAW = "proportional_shear_full_slip"


@dataclass
class ContactOperator:
    """Top-clamped bottom-boundary operators of one geometry / material (local frame).

    Array shapes: ``n_t`` top and ``n_b`` bottom nodes; vectors are component-major.
    """

    x_top: np.ndarray
    y_top: np.ndarray
    x_bottom: np.ndarray
    y_bottom: np.ndarray
    shape_mode_toe_bend: np.ndarray  # (2 n_t,) top displacement per unit alpha
    db_fb: np.ndarray  # (2 n_b, 2 n_b) bottom displacement per unit bottom force
    db_toe: np.ndarray  # (2 n_b,)
    ft_fb: np.ndarray  # (2 n_t, 2 n_b) top force per unit bottom force
    ft_toe: np.ndarray  # (2 n_t,)
    r_fb: np.ndarray  # (3, 2 n_b) rigid amplitudes per unit bottom force
    r_toe: np.ndarray  # (3,)
    toe_shape_mode_flat_rearfoot: np.ndarray  # (n_t,) rearfoot-fixed toe mode, Q_alpha = -psi^T f_ty
    L: float
    H: float
    toe_length: float
    kappa: float
    phi_ref: float
    x_r: float
    y_r: float
    h_mesh: float
    rearfoot_y_mtp_shape_mode: float
    rearfoot_y_mtp_0: float

    @property
    def n_top(self) -> int:
        return int(self.x_top.size)

    @property
    def n_bottom(self) -> int:
        return int(self.x_bottom.size)

    @property
    def rearfoot(self) -> RearfootGeometry:
        return RearfootGeometry(
            L=float(self.L),
            toe_length=float(self.toe_length),
            y_mtp_shape_mode=float(self.rearfoot_y_mtp_shape_mode),
            y_mtp_0=float(self.rearfoot_y_mtp_0),
        )

    def G_blocks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(G_xx, G_xy, G_yx, G_yy)`` component blocks of ``db_fb``."""
        n = self.n_bottom
        G = self.db_fb
        return G[:n, :n], G[:n, n:], G[n:, :n], G[n:, n:]

    def to_payload(self) -> dict[str, np.ndarray]:
        out = {CONTACT_OPERATOR_PREFIX + k: np.asarray(getattr(self, k), dtype=float) for k in CONTACT_OPERATOR_ARRAYS}
        for k in CONTACT_OPERATOR_SCALARS:
            out[CONTACT_OPERATOR_PREFIX + k] = np.asarray(float(getattr(self, k)))
        return out

    @classmethod
    def from_payload(cls, data) -> ContactOperator:
        files = set(getattr(data, "files", data.keys()))
        missing = [k for k in (*CONTACT_OPERATOR_ARRAYS, *CONTACT_OPERATOR_SCALARS) if CONTACT_OPERATOR_PREFIX + k not in files]
        if missing:
            raise KeyError(f"contact operator arrays missing: {missing}")
        kw: dict[str, Any] = {k: np.asarray(data[CONTACT_OPERATOR_PREFIX + k], dtype=float) for k in CONTACT_OPERATOR_ARRAYS}
        kw.update({k: float(np.asarray(data[CONTACT_OPERATOR_PREFIX + k]).reshape(-1)[0]) for k in CONTACT_OPERATOR_SCALARS})
        return cls(**kw)


def median_spacing(x: np.ndarray) -> float:
    d = np.diff(np.sort(np.asarray(x, dtype=float)))
    d = d[d > 0.0]
    return float(np.median(d)) if d.size else 1.0


def _relative_asymmetry(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))


def symmetrized_compliance_blocks(
    compliance: ComplianceResult, reciprocity_tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """``(C_tt, C_tb, C_bt, C_bb, reciprocity_error)`` with roundoff-level asymmetry removed.

    ``reciprocity_error`` is the largest relative asymmetry (``C_tt``, ``C_bb``, ``C_bt``
    vs ``C_tb^T``); above ``reciprocity_tol`` the blocks are rejected, never symmetrized.
    """
    Ctt, Ctb, Cbt, Cbb = (
        np.asarray(m, dtype=float)
        for m in (compliance.Ctt_force, compliance.Ctb_force, compliance.Cbt_force, compliance.Cbb_force)
    )
    err_bt = float(np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30))
    reported = max(_relative_asymmetry(Ctt), _relative_asymmetry(Cbb), err_bt)
    if reported > reciprocity_tol:
        raise ValueError(
            f"Compliance reciprocity error {reported:.3e} exceeds tolerance {reciprocity_tol:.3e}. "
            "Re-run FEM or pass a larger --reciprocity-tol if the residual is still a solver artifact."
        )
    m = Ctt.shape[0]
    full = np.block([[Ctt, Ctb], [Cbt, Cbb]])
    full = 0.5 * (full + full.T)
    return full[:m, :m], full[:m, m:], full[m:, :m], full[m:, m:], reported


def build_contact_operator(
    compliance: ComplianceResult,
    *,
    toe_length: float,
    kappa: float,
    reciprocity_tol: float = 1e-6,
    x_reference: float | None = None,
    y_reference: float = SOLVER_YR,
) -> tuple[ContactOperator, dict[str, float]]:
    """
    The compliance blocks currently map top and bottom displacements to top and bottom forces. However, in our model 
    the top displacements are prescribed to be a rigid body motion plus a single shape mode. If we rotate our reference
    frame to be aligned with the top surface, then the top displacement is a scalar times a known shape mode. 

    Therefore, we can find the matrices that map bottom forces and the top shape mode to bottom displacements 
    and top forces, as well as the rigid body amplitudes.

    Returns the operator and its build diagnostics (``reciprocity_error``,
    ``G_symmetry_error``, ``G_min_eigenvalue``, ``G_max_eigenvalue``, ``clamp_condition``),
    which the solver never reads.
    """
    # Get coordinates of top / bottom surfaces
    x_top, y_top = compliance.x_top, compliance.y_top
    x_bottom, y_bottom = compliance.x_bottom, compliance.y_bottom
    n_t, n_b = x_top.size, x_bottom.size
    # Define reference point for rigid body modes
    shoe_length = float(compliance.config.shoe_length_m)
    x_reference = 0.5 * shoe_length if x_reference is None else float(x_reference)
    y_reference = float(y_reference)

    # Construct rigid body mode matrices for top and bottom surfaces
    R_t = vector_rigid_mode_matrix(x_top, y_top, x_reference, y_reference)
    R_b = vector_rigid_mode_matrix(x_bottom, y_bottom, x_reference, y_reference)
    Ctt, Ctb, Cbt, Cbb, reciprocity_error = symmetrized_compliance_blocks(compliance, reciprocity_tol)
    warn_if_mtp_not_on_top_grid(x_top, shoe_length, toe_length)

    # Build toe bend shape mode for the top surface
    shape_mode_toe_bend = toe_bend_displacement(x_top, shoe_length, toe_length, kappa)

    # Let fb = bottom nodal forces, ft = top nodal forces, dt = top nodal displacements, db = bottom nodal displacements, r = rigid body modes, toe = toe bend coefficient.
    # We know that [Ctt, Ctb, Rt; Cbt, Cbb, Rb; Rt^T, Rb^T, 0] * [ft; fb; lambda] = [dt; db; 0]
    # We can rearrange this to be [Ctt, Rt; Rt^T, 0] * [ft; lambda] = [-Ctb; -Rb^T] * fb + [dt; 0]
    # We know top_displacement = toe * shape_mode_toe_bend, so we can write
    # [Ctt, Rt; Rt^T, 0] * [ft; lambda] = [-Ctb, shape_mode_toe_bend; -Rb^T, 0] * [fb; toe]
    # or A * [ft; lambda] = B * [fb; toe]
    m = 2 * n_t
    
    A = np.zeros((m + 3, m + 3))
    A[:m, :m] = Ctt
    A[:m, m:] = R_t
    A[m:, :m] = R_t.T
    B = np.zeros((m + 3, 2 * n_b + 1))
    B[:m, : 2 * n_b] = -Ctb
    B[m:, : 2 * n_b] = -R_b.T
    B[:m, -1] = shape_mode_toe_bend

    # Solving this system gives
    # [ft; lambda] = A^-1 * B * [fb; toe]
    # [ft; lambda] = [ft_fb, ft_toe; r_fb, r_toe] * [fb; toe]
    lu = linalg.lu_factor(A)
    rcond, _ = linalg.lapack.dgecon(lu[0], float(np.linalg.norm(A, 1)), norm="1")
    X = linalg.lu_solve(lu, B)
    ft_fb, ft_toe = X[:m, : 2 * n_b], X[:m, -1]
    r_fb, r_toe = X[m:, : 2 * n_b], X[m:, -1]

    # From the original system we also have 
    # db = Cbt * ft + Cbb * fb + Rb * lambda 
    # = Cbb * fb + [Cbt, Rb] * [ft; lambda] 
    # = Cbb * fb + [Cbt, Rb] * [ft_fb, ft_toe; r_fb, r_toe] * [fb; toe]
    # = (Cbb + [Cbt, Rb] * [ft_fb; r_fb]) * fb + [Cbt, Rb] * [ft_toe; r_toe] * toe
    # = db_fb * fb + db_toe * toe
    db_fb = Cbt @ ft_fb + Cbb + R_b @ r_fb
    db_toe = Cbt @ ft_toe + R_b @ r_toe
    sym_err = float(np.linalg.norm(db_fb - db_fb.T) / max(np.linalg.norm(db_fb), 1e-300))
    db_fb = 0.5 * (db_fb + db_fb.T)
    eig = np.linalg.eigvalsh(db_fb)

    # Find coefficients such that y_mtp - y_heel = y_mtp_shape_mode * toe_bend_coefficient + y_mtp_0
    geo = rearfoot_geometry(x_top, y_top, shoe_length, toe_length, kappa)
    shape_mode_toe_bend_top = shape_mode_toe_bend[n_t:]

    # Compute the toe shape mode keeping the rearfoot fixed (y_mtp = y_mtp_0)
    toe_shape_mode_flat_rearfoot = rearfoot_toe_mode(shape_mode_toe_bend_top, x_top, shoe_length, toe_length, geo.y_mtp_shape_mode)

    operator = ContactOperator(
        x_top=x_top, y_top=y_top, x_bottom=x_bottom, y_bottom=y_bottom,
        shape_mode_toe_bend=shape_mode_toe_bend, db_fb=db_fb, db_toe=db_toe, ft_fb=ft_fb, ft_toe=ft_toe, r_fb=r_fb, r_toe=r_toe,
        toe_shape_mode_flat_rearfoot=toe_shape_mode_flat_rearfoot, L=shoe_length, H=float(compliance.config.shoe_height_m),
        toe_length=float(toe_length), kappa=float(kappa),
        phi_ref=float(reference_chord_angle(x_top, y_top)), x_r=x_reference, y_r=y_reference, h_mesh=median_spacing(x_bottom),
        rearfoot_y_mtp_shape_mode=float(geo.y_mtp_shape_mode), rearfoot_y_mtp_0=float(geo.y_mtp_0),
    )
    diagnostics = {
        "reciprocity_error": float(reciprocity_error),
        "G_symmetry_error": sym_err,
        "G_min_eigenvalue": float(eig[0]),
        "G_max_eigenvalue": float(eig[-1]),
        "clamp_condition": float(1.0 / rcond) if rcond > 0 else float("inf"),
    }
    return operator, diagnostics
