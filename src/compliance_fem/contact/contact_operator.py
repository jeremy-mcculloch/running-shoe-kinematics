"""Shared boundary-condensed contact operator (replaces per-interval records).

Frames and signs
----------------
``T`` is the top-attached (co-rotated) frame, ``F`` the fixed ground frame
(x heel -> toe, y up, ground at ``y = 0``). The top surface is glued to the
foot: its local displacement is prescribed, ``d_t = alpha w_alpha`` with
``w_alpha = (0, phi1(x))`` and ``alpha = tan(theta)``. Every bottom node may
receive a ground force; nothing else is loaded.

Given the local bottom nodal forces ``f_b`` (component-major ``[f_x...; f_y...]``)
the top-clamped body response is unique (the clamp removes the rigid modes):

    d_b = alpha d_alpha + G_bb f_b          (bottom displacement, local)
    F_t = alpha f_alpha + P_tb f_b          (top nodal force = foot on shoe, local)
    a   = alpha a_alpha + A_rb f_b          (rigid amplitudes of the compliance gauge)

with ``R_t^T F_t + R_b^T f_b = 0`` (free-body equilibrium) built in. ``G_bb`` is
the top-clamped bottom compliance: symmetric positive definite. Everything is
assembled once from the stored compliance blocks ``C_tt, C_tb, C_bt, C_bb`` by
factoring the small ``(2 n_t + 3)`` matrix ``[[C_tt, R_t], [R_t^T, 0]]``; no FEM
solve and no explicit inverse is used at runtime.

``a`` are the rigid amplitudes about ``(x_r, y_r)`` that the plate and render
influence matrices expect, so full fields are recovered from ``[F_t; f_b]``
and ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg

from compliance_fem.contact.basis import top_shape_displacement
from compliance_fem.contact.compliance_blocks import boundary_y
from compliance_fem.contact.corotation import reference_chord_angle
from compliance_fem.contact.toe_spring import (
    RearfootGeometry,
    rearfoot_geometry,
    rearfoot_toe_mode,
    warn_if_mtp_not_on_top_grid,
)
from compliance_fem.fem.boundaries import vector_rigid_mode_matrix

# Rigid modes of the compliance gauge are about (L/2, SOLVER_YR); the plate and render
# influence matrices use the same reference point.
SOLVER_YR = 0.0
CONTACT_OPERATOR_PREFIX = "contact_operator_"
CONTACT_OPERATOR_ARRAYS = (
    "x_top",
    "y_top",
    "x_bottom",
    "y_bottom",
    "w_alpha",
    "G_bb",
    "d_alpha",
    "P_tb",
    "f_alpha",
    "A_rb",
    "a_alpha",
    "psi_top",
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
    "rearfoot_phi1_mtp",
    "rearfoot_dy_mtp",
    "G_symmetry_error",
    "G_min_eigenvalue",
    "G_max_eigenvalue",
    "clamp_condition",
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
    w_alpha: np.ndarray  # (2 n_t,) top displacement per unit alpha
    G_bb: np.ndarray  # (2 n_b, 2 n_b) bottom displacement per unit bottom force
    d_alpha: np.ndarray  # (2 n_b,)
    P_tb: np.ndarray  # (2 n_t, 2 n_b) top force per unit bottom force
    f_alpha: np.ndarray  # (2 n_t,)
    A_rb: np.ndarray  # (3, 2 n_b) rigid amplitudes per unit bottom force
    a_alpha: np.ndarray  # (3,)
    psi_top: np.ndarray  # (n_t,) rearfoot-fixed toe mode, Q_alpha = -psi^T f_ty
    L: float
    H: float
    toe_length: float
    kappa: float
    phi_ref: float
    x_r: float
    y_r: float
    h_mesh: float
    rearfoot_phi1_mtp: float
    rearfoot_dy_mtp: float
    G_symmetry_error: float = 0.0
    G_min_eigenvalue: float = float("nan")
    G_max_eigenvalue: float = float("nan")
    clamp_condition: float = float("nan")
    metadata: dict = field(default_factory=dict)

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
            phi1_mtp=float(self.rearfoot_phi1_mtp),
            dy_mtp=float(self.rearfoot_dy_mtp),
        )

    def G_blocks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(G_xx, G_xy, G_yx, G_yy)`` component blocks of ``G_bb``."""
        n = self.n_bottom
        G = self.G_bb
        return G[:n, :n], G[:n, n:], G[n:, :n], G[n:, n:]

    def to_payload(self) -> dict[str, np.ndarray]:
        out = {CONTACT_OPERATOR_PREFIX + k: np.asarray(getattr(self, k), dtype=float) for k in CONTACT_OPERATOR_ARRAYS}
        for k in CONTACT_OPERATOR_SCALARS:
            out[CONTACT_OPERATOR_PREFIX + k] = np.asarray(float(getattr(self, k)))
        return out

    @classmethod
    def from_payload(cls, data, metadata: dict | None = None) -> ContactOperator:
        files = set(getattr(data, "files", data.keys()))
        missing = [k for k in (*CONTACT_OPERATOR_ARRAYS, *CONTACT_OPERATOR_SCALARS) if CONTACT_OPERATOR_PREFIX + k not in files]
        if missing:
            raise KeyError(f"contact operator arrays missing: {missing}")
        kw: dict[str, Any] = {k: np.asarray(data[CONTACT_OPERATOR_PREFIX + k], dtype=float) for k in CONTACT_OPERATOR_ARRAYS}
        kw.update({k: float(np.asarray(data[CONTACT_OPERATOR_PREFIX + k]).reshape(-1)[0]) for k in CONTACT_OPERATOR_SCALARS})
        return cls(**kw, metadata=dict(metadata or {}))


def median_spacing(x: np.ndarray) -> float:
    d = np.diff(np.sort(np.asarray(x, dtype=float)))
    d = d[d > 0.0]
    return float(np.median(d)) if d.size else 1.0


def build_contact_operator(
    blocks,
    *,
    toe_length: float,
    kappa: float,
    x_r: float | None = None,
    y_r: float = SOLVER_YR,
) -> ContactOperator:
    """Condense the (symmetrized) compliance blocks to the top-clamped bottom operators.

    ``blocks`` is a ``ComplianceBlockMatrix`` (``prepare_compliance_blocks`` output).
    """
    x_t = np.asarray(blocks.x_top, dtype=float)
    x_b = np.asarray(blocks.x_bottom, dtype=float)
    y_t, y_b = boundary_y(blocks)
    n_t, n_b = x_t.size, x_b.size
    L = float(blocks.L)
    xr = 0.5 * L if x_r is None else float(x_r)
    R_t = vector_rigid_mode_matrix(x_t, y_t, xr, y_r)
    R_b = vector_rigid_mode_matrix(x_b, y_b, xr, y_r)
    Ctt, Ctb, Cbt, Cbb = (np.asarray(m, dtype=float) for m in (blocks.Ctt, blocks.Ctb, blocks.Cbt, blocks.Cbb))
    warn_if_mtp_not_on_top_grid(x_t, L, toe_length)
    w_alpha = top_shape_displacement(x_t, L, toe_length, kappa)

    m = 2 * n_t
    M = np.zeros((m + 3, m + 3))
    M[:m, :m] = Ctt
    M[:m, m:] = R_t
    M[m:, :m] = R_t.T
    rhs = np.zeros((m + 3, 2 * n_b + 1))
    rhs[:m, : 2 * n_b] = -Ctb
    rhs[m:, : 2 * n_b] = -R_b.T
    rhs[:m, -1] = w_alpha
    lu = linalg.lu_factor(M)
    rcond, _ = linalg.lapack.dgecon(lu[0], float(np.linalg.norm(M, 1)), norm="1")
    X = linalg.lu_solve(lu, rhs)
    P_tb, f_alpha = X[:m, : 2 * n_b], X[:m, -1]
    A_rb, a_alpha = X[m:, : 2 * n_b], X[m:, -1]
    G = Cbt @ P_tb + Cbb + R_b @ A_rb
    d_alpha = Cbt @ f_alpha + R_b @ a_alpha
    sym_err = float(np.linalg.norm(G - G.T) / max(np.linalg.norm(G), 1e-300))
    G = 0.5 * (G + G.T)
    eig = np.linalg.eigvalsh(G)

    geo = rearfoot_geometry(x_t, y_t, L, toe_length, kappa)
    phi1_top = w_alpha[n_t:]
    psi = rearfoot_toe_mode(phi1_top, x_t, L, toe_length, geo.phi1_mtp)
    return ContactOperator(
        x_top=x_t, y_top=np.asarray(y_t, dtype=float), x_bottom=x_b, y_bottom=np.asarray(y_b, dtype=float),
        w_alpha=w_alpha, G_bb=G, d_alpha=d_alpha, P_tb=P_tb, f_alpha=f_alpha, A_rb=A_rb, a_alpha=a_alpha,
        psi_top=psi, L=L, H=float(blocks.H), toe_length=float(toe_length), kappa=float(kappa),
        phi_ref=float(reference_chord_angle(x_t, y_t)), x_r=xr, y_r=float(y_r), h_mesh=median_spacing(x_b),
        rearfoot_phi1_mtp=float(geo.phi1_mtp), rearfoot_dy_mtp=float(geo.dy_mtp),
        G_symmetry_error=sym_err, G_min_eigenvalue=float(eig[0]), G_max_eigenvalue=float(eig[-1]),
        clamp_condition=float(1.0 / rcond) if rcond > 0 else float("inf"),
        metadata={"contact_formulation": CONTACT_FORMULATION, "tangential_contact_law": TANGENTIAL_CONTACT_LAW},
    )
