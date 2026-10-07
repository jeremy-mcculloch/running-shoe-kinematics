"""Passive toe-joint spring: exact quasistatic equilibrium of the shape coordinate.

Generalized coordinate
----------------------
The single top shape mode prescribes the rotating-frame top displacement

    u_t(alpha) = alpha [0; phi1(x)],        alpha = tan(theta),

with ``phi1`` in metres (``contact_basis.shape_mode_phi1``) and ``alpha``
dimensionless. ``phi1`` vanishes at the heel (x=0) and the toe tip (x=L), so
the lookup frame ``T`` is the heel-to-toe *chord* frame. ``theta = arctan(alpha)``
is kept exact; nothing here uses a small-angle approximation.

What is held fixed: the rearfoot
--------------------------------
The measured foot pitch ``phi`` is the heel -> metatarsal-head line, i.e. the
rearfoot segment [0, a], not the heel-to-toe chord. Bending the toe at a fixed
rearfoot rotates the chord, so the chord-frame rotation depends on alpha
exactly as

    varphi(alpha) = phi - atan2(y_top(a) - y_top(0) + alpha phi1(a), a),

and the foot's virtual displacement for the toe coordinate is the sole motion
relative to the rearfoot line,

    psi(x) = phi1(x) - (x / a) phi1(a),        psi(0) = psi(a) = 0,

which is ~ max(0, x - a): a rotation of the toe segment about the MTP point.

Generalized force (virtual work)
--------------------------------
The lookup solves ``u = C F + R a`` with ``R^T F = 0``: ``F_t`` is the external
nodal load applied *to* the shoe top, i.e. the foot-on-shoe force (the load
term of ``K u = f`` in the FEM weak form). The virtual work of that load on a
toe variation at fixed rearfoot is ``(psi^T f_{t,y}) delta alpha``, so by
action-reaction the shoe acts on the foot's toe coordinate with

    Q_alpha,shoe->foot = -psi^T f_{t,y}        (N·m per metre width).

Because ``psi ~ max(0, x - a)``, ``Q_alpha,shoe->foot ~ -M_toe``: a negative
foot-on-shoe toe moment about the MTP point (load under the toes) drives a
positive (dorsiflexing) theta. The chord-fixed quantity ``-phi1^T f_{t,y}``
instead holds the toe tip fixed and rotates the rearfoot; it is negative for
any compressive load regardless of where the load acts, and is not used.

Spring
------
    U(alpha) = 1/2 k (arctan(alpha) - theta0)^2
    dU/dalpha = k (arctan(alpha) - theta0) / (1 + alpha^2)
    d2U/dalpha2 = k (1 - 2 alpha (arctan(alpha) - theta0)) / (1 + alpha^2)^2

Toe equilibrium (foot toe segment, shoe force + spring):

    g(alpha) = Q_alpha,shoe->foot(alpha) - dU/dalpha = 0.

Per contact record the translations are eliminated with the force balance, so
``Q(alpha) = c . [F_x^T, F_y^T, r_x, r_y, alpha]`` with the local force and the
rotation coefficients evaluated at ``varphi(alpha)`` (``RearfootToeModel``).
With ``Pi(alpha) = U(alpha) - int_0^alpha Q`` we have ``Pi' = -g``; a root is
stable when ``Pi'' = d2U/dalpha2 - dQ/dalpha > 0``. The affine special case
``Q = q0 + q1 alpha`` (fixed frame) is kept for synthetic checks. Damping
``c_toe`` is recorded but never enters this quasistatic equation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import linalg, optimize

from compliance_fem.contact_basis import (
    COL_ALPHA,
    COL_BRX,
    COL_BRY,
    COL_BX,
    COL_BY,
    COL_CONST,
    N_AFFINE_COLUMNS,
    N_BASIS_MODES,
)


def _affine_tables(scalars_row: np.ndarray, q_basis_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return six-row affine ``(S, q)``; a five-row linear table gets a zero closure row."""
    S = np.asarray(scalars_row, dtype=float)
    q = np.asarray(q_basis_row, dtype=float).reshape(-1)
    if S.shape[0] == N_BASIS_MODES:
        S = np.vstack([np.zeros((1, S.shape[1])), S])
    if q.size == N_BASIS_MODES:
        q = np.concatenate([[0.0], q])
    if S.shape[0] != N_AFFINE_COLUMNS or q.size != N_AFFINE_COLUMNS:
        raise ValueError("Scalar and Q tables must have 6 affine (or 5 linear) rows.")
    return S, q

TOE_MODEL_PASSIVE_SPRING = "passive_spring"
TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT = "passive_spring_elastic_equivalent"
TOE_MODEL_PRESCRIBED_LEGACY = "prescribed_legacy"
TOE_MODEL_FIT_COP_LEGACY = "fit_cop_legacy"
TOE_MODELS = (
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    TOE_MODEL_PRESCRIBED_LEGACY,
    TOE_MODEL_FIT_COP_LEGACY,
)
PASSIVE_TOE_MODELS = (TOE_MODEL_PASSIVE_SPRING, TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT)
DEFAULT_TOE_MODEL = TOE_MODEL_PASSIVE_SPRING

TOE_GENERALIZED_FORCE_DEFINITION = (
    "Q_alpha_shoe_on_foot = -psi^T f_{t,y}, psi(x) = phi1(x) - (x/a) phi1(a) (rearfoot "
    "heel->MTP line held fixed); f_t is the foot-on-shoe nodal load (load term of "
    "K u = f), phi1 the top vertical shape mode in metres"
)
TOE_EQUATION = (
    "g(alpha) = Q_alpha_shoe_on_foot(alpha) - k*(arctan(alpha) - theta0)/(1 + alpha^2) = 0, "
    "theta = arctan(alpha), chord rotation varphi(alpha) = phi_rearfoot - "
    "atan2(y_top(a) - y_top(0) + alpha*phi1(a), a)"
)
VISCO_REJECTION_MESSAGE = (
    "toe_model='passive_spring' is validated only for visco_model='elastic'. The "
    "viscoelastic backends map the resultant wrench componentwise and do not provide "
    "the viscoelastic generalized force Q_alpha,VE needed for an exact toe balance. "
    "Use visco_model='elastic', or choose "
    "toe_model='passive_spring_elastic_equivalent' to solve the spring against the "
    "elastic-equivalent wrench (an approximation)."
)


@dataclass(frozen=True)
class ToeSpringConfig:
    """Toe-model configuration (angles of the bounds in degrees, all else SI)."""

    toe_model: str = DEFAULT_TOE_MODEL
    toe_stiffness_Nm_per_rad: float = 25.0
    toe_neutral_angle_rad: float = 0.0
    toe_damping_Nms_per_rad: float = 0.0
    toe_angle_min_deg: float = -75.0
    toe_angle_max_deg: float = 75.0
    toe_equilibrium_abs_tol_Nm: float = 1.0e-6
    toe_equilibrium_rel_tol: float = 1.0e-8
    toe_root_scan_points: int = 721
    toe_low_force_threshold_N: float = 50.0

    def __post_init__(self) -> None:
        if self.toe_model not in TOE_MODELS:
            raise ValueError(f"toe_model must be one of {TOE_MODELS}, got {self.toe_model!r}.")
        k = float(self.toe_stiffness_Nm_per_rad)
        if not np.isfinite(k) or k < 0.0:
            raise ValueError(f"toe_stiffness_Nm_per_rad must be finite and >= 0, got {k}.")
        c = float(self.toe_damping_Nms_per_rad)
        if not np.isfinite(c) or c < 0.0:
            raise ValueError(f"toe_damping_Nms_per_rad must be finite and >= 0, got {c}.")
        th0 = float(self.toe_neutral_angle_rad)
        if not np.isfinite(th0) or abs(th0) >= 0.5 * np.pi:
            raise ValueError(f"toe_neutral_angle_rad must lie strictly inside +-pi/2, got {th0}.")
        lo = float(self.toe_angle_min_deg)
        hi = float(self.toe_angle_max_deg)
        if not (np.isfinite(lo) and np.isfinite(hi)) or not (-90.0 < lo < hi < 90.0):
            raise ValueError(
                "toe angle bounds must satisfy -90 < toe_angle_min_deg < toe_angle_max_deg < 90, "
                f"got [{lo}, {hi}]."
            )
        if not self.toe_equilibrium_abs_tol_Nm > 0.0:
            raise ValueError("toe_equilibrium_abs_tol_Nm must be positive.")
        if not self.toe_equilibrium_rel_tol >= 0.0:
            raise ValueError("toe_equilibrium_rel_tol must be >= 0.")
        if int(self.toe_root_scan_points) < 3:
            raise ValueError("toe_root_scan_points must be >= 3.")
        if not self.toe_low_force_threshold_N >= 0.0:
            raise ValueError("toe_low_force_threshold_N must be >= 0.")

    @property
    def is_passive(self) -> bool:
        return self.toe_model in PASSIVE_TOE_MODELS

    @property
    def theta_min_rad(self) -> float:
        return float(np.deg2rad(self.toe_angle_min_deg))

    @property
    def theta_max_rad(self) -> float:
        return float(np.deg2rad(self.toe_angle_max_deg))

    def to_provenance(self) -> dict:
        out = asdict(self)
        out["toe_equation"] = TOE_EQUATION
        out["toe_generalized_force_definition"] = TOE_GENERALIZED_FORCE_DEFINITION
        out["damping_enters_equation"] = False
        return out


# --------------------------------------------------------------------------------------
# Spring (exact in theta = arctan(alpha))
# --------------------------------------------------------------------------------------


def spring_energy(alpha, k: float, theta0: float = 0.0):
    """U = 1/2 k (arctan(alpha) - theta0)^2 in J."""
    th = np.arctan(np.asarray(alpha, dtype=float))
    return 0.5 * float(k) * (th - float(theta0)) ** 2


def spring_dU_dalpha(alpha, k: float, theta0: float = 0.0):
    """dU/dalpha = k (arctan(alpha) - theta0) / (1 + alpha^2), in N·m."""
    a = np.asarray(alpha, dtype=float)
    return float(k) * (np.arctan(a) - float(theta0)) / (1.0 + a * a)


def spring_d2U_dalpha2(alpha, k: float, theta0: float = 0.0):
    """d2U/dalpha2 = k (1 - 2 alpha (arctan(alpha) - theta0)) / (1 + alpha^2)^2."""
    a = np.asarray(alpha, dtype=float)
    return float(k) * (1.0 - 2.0 * a * (np.arctan(a) - float(theta0))) / (1.0 + a * a) ** 2


def spring_moment_theta(alpha, k: float, theta0: float = 0.0):
    """Spring generalized force on theta, M_spring,theta = -k (theta - theta0), N·m."""
    return -float(k) * (np.arctan(np.asarray(alpha, dtype=float)) - float(theta0))


def spring_generalized_force_alpha(alpha, k: float, theta0: float = 0.0):
    """Spring generalized force on alpha, -dU/dalpha, N·m."""
    return -spring_dU_dalpha(alpha, k, theta0)


def q_theta_from_q_alpha(q_alpha, alpha):
    """Q_theta = Q_alpha d(alpha)/d(theta) = Q_alpha (1 + alpha^2)."""
    a = np.asarray(alpha, dtype=float)
    return np.asarray(q_alpha, dtype=float) * (1.0 + a * a)


# --------------------------------------------------------------------------------------
# Generalized force from stored top forces (single shared utility)
# --------------------------------------------------------------------------------------


def top_shape_mode_vertical(basis_top_displacements: np.ndarray, n_top: int) -> np.ndarray:
    """phi1 at the top nodes (m), read from the stored component-major W_t."""
    W = np.asarray(basis_top_displacements, dtype=float)
    if W.shape[0] != 2 * int(n_top):
        raise ValueError(f"W_t has {W.shape[0]} rows; expected 2*n_top = {2 * int(n_top)}.")
    col = COL_ALPHA if W.shape[1] == N_AFFINE_COLUMNS else COL_ALPHA - 1
    if np.any(W[: int(n_top), col] != 0.0):
        raise ValueError("Top shape mode has a horizontal component; Q_alpha would need f_{t,x}.")
    return W[int(n_top) :, col].copy()


@dataclass(frozen=True)
class RearfootGeometry:
    """Rearfoot line (heel x=0 -> MTP x=a) in the reference chord frame.

    ``phi1_a`` is the shape mode at the MTP point (m, < 0) and ``dy_a`` the
    reference top rise ``y_top(a) - y_top(0)`` (m).
    """

    a: float
    phi1_a: float
    dy_a: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.a) and self.a > 0.0):
            raise ValueError(f"MTP location a must be positive, got {self.a}.")
        if not (np.isfinite(self.phi1_a) and np.isfinite(self.dy_a)):
            raise ValueError("Rearfoot geometry must be finite.")

    @property
    def reference_angle(self) -> float:
        """Reference rearfoot-line angle in the mesh frame, atan2(dy_a, a)."""
        return float(np.arctan2(self.dy_a, self.a))

    def chord_rotation(self, phi_rearfoot: float, alpha):
        """varphi(alpha) = phi_rearfoot - atan2(dy_a + alpha phi1(a), a)."""
        a = np.asarray(alpha, dtype=float)
        return float(phi_rearfoot) - np.arctan2(self.dy_a + a * self.phi1_a, self.a)

    def chord_rotation_derivative(self, alpha):
        """d varphi / d alpha = -a phi1(a) / (a^2 + (dy_a + alpha phi1(a))^2)."""
        a = np.asarray(alpha, dtype=float)
        h = self.dy_a + a * self.phi1_a
        return -self.a * self.phi1_a / (self.a * self.a + h * h)

    def to_dict(self) -> dict:
        return {"a": float(self.a), "phi1_a": float(self.phi1_a), "dy_a": float(self.dy_a)}


def rearfoot_toe_mode(phi1_top: np.ndarray, x_top: np.ndarray, a: float, phi1_a: float) -> np.ndarray:
    """psi(x) = phi1(x) - (x/a) phi1(a): sole motion relative to the heel->MTP line."""
    x = np.asarray(x_top, dtype=float)
    return np.asarray(phi1_top, dtype=float) - (x / float(a)) * float(phi1_a)


def q_alpha_foot_on_shoe(top_force_y, mode_top: np.ndarray):
    """Q_alpha,foot->shoe = mode^T f_{t,y}; contracts the last (top-node) axis."""
    return np.tensordot(np.asarray(top_force_y, dtype=float), np.asarray(mode_top, dtype=float), axes=([-1], [0]))


def q_alpha_shoe_on_foot(top_force_y, mode_top: np.ndarray):
    """Q_alpha,shoe->foot = -mode^T f_{t,y} (action-reaction of the stored foot-on-shoe load)."""
    return -q_alpha_foot_on_shoe(top_force_y, mode_top)


def q_alpha_shoe_on_foot_basis(
    top_force_y_basis: np.ndarray,
    basis_top_displacements: np.ndarray,
    n_top: int,
    x_top: np.ndarray,
    geometry: RearfootGeometry,
) -> np.ndarray:
    """Per-record, per-mode ``Q_alpha_shoe_on_foot_basis[record, mode]`` (N·m/m), -psi^T f_{t,y}."""
    phi1 = top_shape_mode_vertical(basis_top_displacements, n_top)
    psi = rearfoot_toe_mode(phi1, x_top, geometry.a, geometry.phi1_a)
    return q_alpha_shoe_on_foot(top_force_y_basis, psi)


# --------------------------------------------------------------------------------------
# Affine reduction d(alpha), Q(alpha)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToeAffineModel:
    """Force-controlled reduction for one record (per unit width).

    Local resultant ``F = K_d d + F_alpha alpha + F_r``; prescribing ``F = F*`` gives
    ``d(alpha) = d0 + d1 alpha`` and ``Q(alpha) = q0 + q1 alpha``.
    """

    K_d: np.ndarray
    F_alpha: np.ndarray
    F_r: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    q0: float
    q1: float
    kd_cond: float
    kd_smin: float
    well_conditioned: bool

    def displacement(self, alpha: float) -> np.ndarray:
        return self.d0 + self.d1 * float(alpha)

    def generalized_force(self, alpha):
        return self.q0 + self.q1 * np.asarray(alpha, dtype=float)


def toe_affine_model(
    scalars_row: np.ndarray,
    q_basis_row: np.ndarray,
    F_local_star: np.ndarray,
    r_x: float,
    r_y: float,
    *,
    fx_index: int = 0,
    fy_index: int = 1,
    cond_limit: float = 1.0e8,
    smin_threshold: float = 0.0,
) -> ToeAffineModel:
    """Build ``d0, d1, q0, q1`` analytically from the stored affine tables.

    The curved-sole closure (affine column 0) enters ``F_r`` and ``q0`` with
    coefficient 1. ``K_d`` is LU-factored once and both right-hand sides
    ``F* - F_r`` and ``-F_alpha`` are solved against it; no inverse is formed.
    """
    S, q = _affine_tables(scalars_row, q_basis_row)
    K_d = np.array(
        [
            [S[COL_BX, fx_index], S[COL_BY, fx_index]],
            [S[COL_BX, fy_index], S[COL_BY, fy_index]],
        ],
        dtype=float,
    )
    F_alpha = np.array([S[COL_ALPHA, fx_index], S[COL_ALPHA, fy_index]], dtype=float)
    F_r = (
        np.array([S[COL_CONST, fx_index], S[COL_CONST, fy_index]])
        + float(r_x) * np.array([S[COL_BRX, fx_index], S[COL_BRX, fy_index]])
        + float(r_y) * np.array([S[COL_BRY, fx_index], S[COL_BRY, fy_index]])
    )
    nan2 = np.full(2, np.nan)
    finite = (
        np.all(np.isfinite(K_d))
        and np.all(np.isfinite(F_alpha))
        and np.all(np.isfinite(F_r))
        and np.all(np.isfinite(q))
    )
    if not finite:
        return ToeAffineModel(K_d, F_alpha, F_r, nan2, nan2, np.nan, np.nan, np.inf, np.nan, False)
    svals = np.linalg.svd(K_d, compute_uv=False)
    smin = float(svals[-1])
    cond = float(svals[0] / smin) if smin > 0.0 else float("inf")
    if smin <= float(smin_threshold) or not np.isfinite(cond) or cond > float(cond_limit):
        return ToeAffineModel(K_d, F_alpha, F_r, nan2, nan2, np.nan, np.nan, cond, smin, False)
    lu_piv = linalg.lu_factor(K_d, check_finite=True)
    rhs = np.column_stack([np.asarray(F_local_star, dtype=float).reshape(2) - F_r, -F_alpha])
    D = linalg.lu_solve(lu_piv, rhs, check_finite=True)
    d0 = D[:, 0].copy()
    d1 = D[:, 1].copy()
    q_d = np.array([q[COL_BX], q[COL_BY]], dtype=float)
    q_r = q[COL_CONST] + float(r_x) * q[COL_BRX] + float(r_y) * q[COL_BRY]
    q0 = float(q_r + q_d @ d0)
    q1 = float(q[COL_ALPHA] + q_d @ d1)
    return ToeAffineModel(K_d, F_alpha, F_r, d0, d1, q0, q1, cond, smin, True)


def gamma_from_alpha(model: ToeAffineModel, alpha: float, r_x: float, r_y: float) -> np.ndarray:
    """gamma = [alpha, d_ax, d_ay, r_x, r_y] in saved basis order."""
    d = model.displacement(alpha)
    return np.array([float(alpha), float(d[0]), float(d[1]), float(r_x), float(r_y)], dtype=float)


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


@dataclass(frozen=True)
class RearfootToeModel:
    """Force-controlled reduction for one record with the rearfoot angle prescribed.

    With ``z(alpha) = [F_x^T, F_y^T, r_x, r_y, alpha]`` evaluated at the chord
    rotation ``varphi(alpha)``, the force balance gives ``d = D z + d_offset``
    and the total shoe-on-foot generalized force is ``Q = coeff . z + q_offset``
    (N·m). The offsets carry the curved-sole closure (affine column 0, zero for a
    flat sole). ``D`` and ``d_offset`` come from one LU factorization of ``K_d``.
    """

    coeff: np.ndarray
    D: np.ndarray
    F_fixed: np.ndarray
    phi_rearfoot: float
    geometry: RearfootGeometry
    kd_cond: float
    kd_smin: float
    well_conditioned: bool
    d_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    q_offset: float = 0.0

    def varphi(self, alpha):
        return self.geometry.chord_rotation(self.phi_rearfoot, alpha)

    def _z(self, alpha) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(alpha, dtype=float)
        v = self.varphi(a)
        c, s = np.cos(v), np.sin(v)
        Fx, Fy = float(self.F_fixed[0]), float(self.F_fixed[1])
        z = np.stack([c * Fx + s * Fy, -s * Fx + c * Fy, c - 1.0, -s, a])
        return z, v

    def generalized_force(self, alpha):
        z, _ = self._z(alpha)
        return np.tensordot(self.coeff, z, axes=([0], [0])) + self.q_offset

    def generalized_force_derivative(self, alpha):
        a = np.asarray(alpha, dtype=float)
        z, v = self._z(a)
        dz_dv = np.stack([z[1], -z[0], -np.sin(v), -np.cos(v), np.zeros_like(a)])
        dv = self.geometry.chord_rotation_derivative(a)
        return np.tensordot(self.coeff[:4], dz_dv[:4], axes=([0], [0])) * dv + self.coeff[4]

    def work(self, alpha):
        """int_0^alpha Q(s) ds (Gauss-Legendre; Q is smooth in alpha)."""
        a = np.asarray(alpha, dtype=float)
        s = 0.5 * a[..., None] * (_GL_NODES + 1.0)
        return 0.5 * a * np.sum(_GL_WEIGHTS * self.generalized_force(s), axis=-1)

    @property
    def q0(self) -> float:
        return float(self.generalized_force(0.0))

    @property
    def q1(self) -> float:
        return float(self.generalized_force_derivative(0.0))

    def gamma(self, alpha: float) -> np.ndarray:
        """gamma = [alpha, d_ax, d_ay, r_x, r_y] at varphi(alpha)."""
        z, _ = self._z(float(alpha))
        d = self.D @ z + self.d_offset
        return np.array([float(alpha), float(d[0]), float(d[1]), float(z[2]), float(z[3])], dtype=float)


def rearfoot_toe_model(
    scalars_row: np.ndarray,
    q_basis_row: np.ndarray,
    F_fixed_star: np.ndarray,
    phi_rearfoot: float,
    geometry: RearfootGeometry,
    width_m: float = 1.0,
    *,
    fx_index: int = 0,
    fy_index: int = 1,
    cond_limit: float = 1.0e8,
    smin_threshold: float = 0.0,
) -> RearfootToeModel:
    """Build the exact rearfoot-prescribed reduction for one record.

    ``phi_rearfoot`` is the absolute fixed-frame angle of the heel->MTP line;
    the mesh rotation is ``varphi(alpha) = phi_rearfoot - atan2(dy_a + alpha
    phi1(a), a)``. ``q_basis_row`` must be the rearfoot-fixed basis
    (-psi^T f_{t,y}); ``coeff`` is scaled by ``width_m`` to total N·m. The
    curved-sole closure (affine column 0) is included exactly once, as the
    constant terms ``d_offset = -K_d^{-1} F_closure`` and ``q_offset``.
    """
    S, q = _affine_tables(scalars_row, q_basis_row)
    F_fixed = np.asarray(F_fixed_star, dtype=float).reshape(2)
    K_d = np.array(
        [
            [S[COL_BX, fx_index], S[COL_BY, fx_index]],
            [S[COL_BX, fy_index], S[COL_BY, fy_index]],
        ],
        dtype=float,
    )
    F_const = np.array([S[COL_CONST, fx_index], S[COL_CONST, fy_index]], dtype=float)
    F_alpha = np.array([S[COL_ALPHA, fx_index], S[COL_ALPHA, fy_index]], dtype=float)
    F_brx = np.array([S[COL_BRX, fx_index], S[COL_BRX, fy_index]], dtype=float)
    F_bry = np.array([S[COL_BRY, fx_index], S[COL_BRY, fy_index]], dtype=float)
    nan5 = np.full(5, np.nan)
    nanD = np.full((2, 5), np.nan)
    nan2 = np.full(2, np.nan)
    phi = float(phi_rearfoot)
    finite = all(
        np.all(np.isfinite(x)) for x in (K_d, F_const, F_alpha, F_brx, F_bry, q, F_fixed)
    ) and np.isfinite(phi)
    if not finite:
        return RearfootToeModel(nan5, nanD, F_fixed, phi, geometry, np.inf, np.nan, False, nan2, np.nan)
    svals = np.linalg.svd(K_d, compute_uv=False)
    smin = float(svals[-1])
    cond = float(svals[0] / smin) if smin > 0.0 else float("inf")
    if smin <= float(smin_threshold) or not np.isfinite(cond) or cond > float(cond_limit):
        return RearfootToeModel(nan5, nanD, F_fixed, phi, geometry, cond, smin, False, nan2, np.nan)
    lu_piv = linalg.lu_factor(K_d, check_finite=True)
    rhs = np.column_stack([[1.0, 0.0], [0.0, 1.0], -F_brx, -F_bry, -F_alpha, -F_const])
    sol = linalg.lu_solve(lu_piv, rhs, check_finite=True)
    D = sol[:, :5]
    d_offset = sol[:, 5].copy()
    q_d = np.array([q[COL_BX], q[COL_BY]], dtype=float)
    coeff = q_d @ D + np.array([0.0, 0.0, q[COL_BRX], q[COL_BRY], q[COL_ALPHA]])
    q_offset = float(q[COL_CONST] + q_d @ d_offset)
    w = float(width_m)
    return RearfootToeModel(w * coeff, D, F_fixed, phi, geometry, cond, smin, True, d_offset, w * q_offset)


@dataclass(frozen=True)
class ToeRootScreen:
    """Vectorized toe roots for many records (same equation as the scalar solver).

    ``pair_row`` / ``pair_theta`` list every bracketed root in the angle bounds
    (theta refined by bisection to ~1e-15 rad). ``row_has_root`` marks rows with
    at least one root; ``diag_theta`` is the minimum-|g| scan angle for rows without one.
    """

    rows: np.ndarray
    coeff: np.ndarray  # (n, 5)
    q_offset: np.ndarray  # (n,)
    D: np.ndarray  # (n, 2, 5)
    d_offset: np.ndarray  # (n, 2)
    kd_cond: np.ndarray
    well_conditioned: np.ndarray
    pair_row: np.ndarray  # index into ``rows``
    pair_theta: np.ndarray
    pair_tangent: np.ndarray
    pair_residual: np.ndarray
    row_has_root: np.ndarray
    diag_theta: np.ndarray
    diag_residual: np.ndarray


def _z_of_theta(theta: np.ndarray, F_fixed: np.ndarray, phi_rearfoot: float, geometry: RearfootGeometry):
    alpha = np.tan(theta)
    v = geometry.chord_rotation(phi_rearfoot, alpha)
    c, s = np.cos(v), np.sin(v)
    Fx, Fy = float(F_fixed[0]), float(F_fixed[1])
    z = np.stack([c * Fx + s * Fy, -s * Fx + c * Fy, c - 1.0, -s, alpha])
    dz = np.stack([z[1], -z[0], -s, -c, np.zeros_like(alpha)])
    dv = geometry.chord_rotation_derivative(alpha)
    return alpha, z, dz, dv


def screen_toe_roots(
    scalars_rows: np.ndarray,
    q_rows: np.ndarray,
    F_fixed_star: np.ndarray,
    phi_rearfoot: float,
    geometry: RearfootGeometry,
    cfg: ToeSpringConfig,
    width_m: float = 1.0,
    *,
    rows: np.ndarray | None = None,
    cond_limit: float = 1.0e8,
    bisection_iterations: int = 64,
) -> ToeRootScreen:
    """Find all toe roots for many records at once.

    Each record's ``Q(alpha) = coeff . z(alpha) + q_offset`` is built from the
    same affine reduction as :func:`rearfoot_toe_model` (batched 2x2 solves, no
    inverse). Because ``z`` depends only on ``alpha`` and the frame inputs, the
    residual on the theta scan grid for every record is one matrix product. Sign
    changes are bracketed and refined by vectorized bisection in theta.
    """
    S = np.asarray(scalars_rows, dtype=float)
    q = np.asarray(q_rows, dtype=float)
    n = S.shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=int)
    F_fixed = np.asarray(F_fixed_star, dtype=float).reshape(2)
    K = np.stack(
        [
            np.stack([S[:, COL_BX, 0], S[:, COL_BY, 0]], axis=-1),
            np.stack([S[:, COL_BX, 1], S[:, COL_BY, 1]], axis=-1),
        ],
        axis=1,
    )
    finite = np.all(np.isfinite(K), axis=(1, 2)) & np.all(np.isfinite(S[:, :, :2]), axis=(1, 2))
    finite &= np.all(np.isfinite(q), axis=1)
    svals = np.full((n, 2), np.nan)
    if np.any(finite):
        svals[finite] = np.linalg.svd(K[finite], compute_uv=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(svals[:, 1] > 0.0, svals[:, 0] / svals[:, 1], np.inf)
    well = finite & np.isfinite(cond) & (cond <= float(cond_limit))
    rhs = np.zeros((n, 2, 6))
    rhs[:, 0, 0] = 1.0
    rhs[:, 1, 1] = 1.0
    rhs[:, :, 2] = -S[:, COL_BRX, :2]
    rhs[:, :, 3] = -S[:, COL_BRY, :2]
    rhs[:, :, 4] = -S[:, COL_ALPHA, :2]
    rhs[:, :, 5] = -S[:, COL_CONST, :2]
    sol = np.full((n, 2, 6), np.nan)
    if np.any(well):
        sol[well] = np.linalg.solve(K[well], rhs[well])
    D = sol[:, :, :5]
    d_offset = sol[:, :, 5]
    q_d = q[:, [COL_BX, COL_BY]]
    coeff = np.einsum("nk,nkm->nm", q_d, D)
    coeff[:, 2] += q[:, COL_BRX]
    coeff[:, 3] += q[:, COL_BRY]
    coeff[:, 4] += q[:, COL_ALPHA]
    q_offset = q[:, COL_CONST] + np.einsum("nk,nk->n", q_d, d_offset)
    w = float(width_m)
    coeff *= w
    q_offset = q_offset * w

    k = cfg.toe_stiffness_Nm_per_rad
    th0 = cfg.toe_neutral_angle_rad
    thetas = np.linspace(cfg.theta_min_rad, cfg.theta_max_rad, int(cfg.toe_root_scan_points))
    alpha, z, _, _ = _z_of_theta(thetas, F_fixed, phi_rearfoot, geometry)
    G = coeff @ z + q_offset[:, None] - spring_dU_dalpha(alpha, k, th0)[None, :]
    G[~well] = np.nan
    sign_change = (G[:, :-1] * G[:, 1:] < 0.0) | (G[:, :-1] == 0.0)
    pr, pc = np.nonzero(sign_change)
    lo = thetas[pc].copy()
    hi = thetas[pc + 1].copy()
    g_lo = G[pr, pc].copy()
    exact = g_lo == 0.0
    for _ in range(int(bisection_iterations)):
        mid = 0.5 * (lo + hi)
        a_m, z_m, _, _ = _z_of_theta(mid, F_fixed, phi_rearfoot, geometry)
        g_m = np.einsum("pk,kp->p", coeff[pr], z_m) + q_offset[pr] - spring_dU_dalpha(a_m, k, th0)
        left = (g_lo * g_m <= 0.0) & ~exact
        hi = np.where(left, mid, hi)
        lo = np.where(left | exact, lo, mid)
        g_lo = np.where(left | exact, g_lo, g_m)
    root = np.where(exact, lo, 0.5 * (lo + hi))
    a_r, z_r, dz_r, dv_r = _z_of_theta(root, F_fixed, phi_rearfoot, geometry)
    Q_r = np.einsum("pk,kp->p", coeff[pr], z_r) + q_offset[pr]
    dQ = np.einsum("pk,kp->p", coeff[pr, :4], dz_r[:4]) * dv_r + coeff[pr, 4]
    tangent = spring_d2U_dalpha2(a_r, k, th0) - dQ
    residual = Q_r - spring_dU_dalpha(a_r, k, th0)
    has_root = np.zeros(n, dtype=bool)
    has_root[pr] = True
    absG = np.where(np.isfinite(G), np.abs(G), np.inf)
    i_min = np.argmin(absG, axis=1)
    diag_theta = thetas[i_min]
    diag_res = G[np.arange(n), i_min]
    return ToeRootScreen(
        rows=rows,
        coeff=coeff,
        q_offset=q_offset,
        D=D,
        d_offset=d_offset,
        kd_cond=cond,
        well_conditioned=well,
        pair_row=pr,
        pair_theta=root,
        pair_tangent=tangent,
        pair_residual=residual,
        row_has_root=has_root,
        diag_theta=diag_theta,
        diag_residual=diag_res,
    )


@dataclass(frozen=True)
class AffineGeneralizedForce:
    """Q(alpha) = q0 + q1 alpha (fixed chord frame; synthetic checks)."""

    q0: float
    q1: float

    def generalized_force(self, alpha):
        return self.q0 + self.q1 * np.asarray(alpha, dtype=float)

    def generalized_force_derivative(self, alpha):
        return np.full_like(np.asarray(alpha, dtype=float), self.q1)

    def work(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return self.q0 * a + 0.5 * self.q1 * a * a


# --------------------------------------------------------------------------------------
# Exact equilibrium solve
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToeRoot:
    alpha: float
    theta_rad: float
    residual_Nm: float
    relative_residual: float
    tangent: float
    stable: bool
    spring_energy_J: float
    potential_J: float
    converged: bool
    at_bound: bool

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))


@dataclass(frozen=True)
class ToeEquilibriumSolution:
    """All equilibrium roots in the angle bounds, sorted by theta.

    ``diagnostic`` holds the minimum-|g| point when no root was found; it is
    never labelled converged unless it meets the tolerance.
    """

    roots: tuple[ToeRoot, ...]
    status: str
    q0: float
    q1: float
    diagnostic: ToeRoot | None = None
    scan_theta_rad: np.ndarray | None = field(default=None, repr=False)
    scan_residual: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_roots(self) -> int:
        return len(self.roots)


def toe_residual(alpha, q0: float, q1: float, k: float, theta0: float = 0.0):
    """g(alpha) = q0 + q1 alpha - k (arctan(alpha) - theta0)/(1 + alpha^2), in N·m."""
    a = np.asarray(alpha, dtype=float)
    return float(q0) + float(q1) * a - spring_dU_dalpha(a, k, theta0)


def toe_residual_tolerance(alpha, q0: float, q1: float, cfg: ToeSpringConfig):
    """abs_tol + rel_tol * max(|q0|, |q1 alpha|, |dU/dalpha|)."""
    a = np.asarray(alpha, dtype=float)
    scale = np.maximum.reduce(
        [
            np.full_like(a, abs(float(q0))),
            np.abs(float(q1) * a),
            np.abs(spring_dU_dalpha(a, cfg.toe_stiffness_Nm_per_rad, cfg.toe_neutral_angle_rad)),
        ]
    )
    return cfg.toe_equilibrium_abs_tol_Nm + cfg.toe_equilibrium_rel_tol * scale


def total_potential(alpha, q0: float, q1: float, k: float, theta0: float = 0.0):
    """Reduced potential with Pi'(alpha) = -g(alpha): U - (q0 alpha + q1 alpha^2 / 2)."""
    a = np.asarray(alpha, dtype=float)
    return spring_energy(a, k, theta0) - (float(q0) * a + 0.5 * float(q1) * a * a)


def potential_curvature(alpha, q1: float, k: float, theta0: float = 0.0):
    """Pi''(alpha) = d2U/dalpha2 - q1; stable when > 0."""
    return spring_d2U_dalpha2(alpha, k, theta0) - float(q1)


def model_residual(model, alpha, cfg: ToeSpringConfig):
    """g(alpha) = Q(alpha) - dU/dalpha for any generalized-force model (N·m)."""
    a = np.asarray(alpha, dtype=float)
    return model.generalized_force(a) - spring_dU_dalpha(
        a, cfg.toe_stiffness_Nm_per_rad, cfg.toe_neutral_angle_rad
    )


def model_residual_tolerance(model, alpha, cfg: ToeSpringConfig):
    """abs_tol + rel_tol * max(|Q(0)|, |Q(alpha) - Q(0)|, |dU/dalpha|)."""
    a = np.asarray(alpha, dtype=float)
    q0 = float(model.generalized_force(0.0))
    scale = np.maximum.reduce(
        [
            np.full_like(a, abs(q0)),
            np.abs(model.generalized_force(a) - q0),
            np.abs(spring_dU_dalpha(a, cfg.toe_stiffness_Nm_per_rad, cfg.toe_neutral_angle_rad)),
        ]
    )
    return cfg.toe_equilibrium_abs_tol_Nm + cfg.toe_equilibrium_rel_tol * scale


def _make_root(theta: float, model, cfg: ToeSpringConfig, lo: float, hi: float) -> ToeRoot:
    k = cfg.toe_stiffness_Nm_per_rad
    th0 = cfg.toe_neutral_angle_rad
    alpha = float(np.tan(theta))
    Q = float(model.generalized_force(alpha))
    g = Q - float(spring_dU_dalpha(alpha, k, th0))
    tol = float(model_residual_tolerance(model, alpha, cfg))
    scale = max(abs(Q), abs(float(spring_dU_dalpha(alpha, k, th0))), cfg.toe_equilibrium_abs_tol_Nm)
    tangent = float(spring_d2U_dalpha2(alpha, k, th0) - float(model.generalized_force_derivative(alpha)))
    return ToeRoot(
        alpha=alpha,
        theta_rad=float(theta),
        residual_Nm=g,
        relative_residual=abs(g) / scale,
        tangent=tangent,
        stable=bool(np.isfinite(tangent) and tangent > 0.0),
        spring_energy_J=float(spring_energy(alpha, k, th0)),
        potential_J=float(spring_energy(alpha, k, th0) - float(model.work(alpha))),
        converged=bool(np.isfinite(g) and abs(g) <= tol),
        at_bound=bool(abs(theta - lo) <= 1e-12 or abs(theta - hi) <= 1e-12),
    )


def solve_toe_equilibrium(
    q0: float,
    q1: float,
    cfg: ToeSpringConfig,
    *,
    keep_scan: bool = False,
) -> ToeEquilibriumSolution:
    """Roots of the affine case ``g = q0 + q1 alpha - dU/dalpha`` (total N·m)."""
    return solve_toe_equilibrium_model(AffineGeneralizedForce(float(q0), float(q1)), cfg, keep_scan=keep_scan)


def solve_toe_equilibrium_model(
    model,
    cfg: ToeSpringConfig,
    *,
    keep_scan: bool = False,
) -> ToeEquilibriumSolution:
    """Find every root of ``g = Q(alpha) - dU/dalpha`` with ``theta`` in the bounds.

    ``model`` supplies the *total* shoe-on-foot generalized force (N·m) through
    ``generalized_force``, ``generalized_force_derivative`` and ``work``. The
    bounds are scanned on a uniform theta grid; every sign change is refined
    with Brent's method in theta (alpha = tan(theta)), exact grid zeros and
    endpoint roots are kept, and touching (even-multiplicity) roots are
    captured by refining local minima of |g| and accepting them only within
    tolerance.
    """
    lo = cfg.theta_min_rad
    hi = cfg.theta_max_rad
    q0 = float(model.generalized_force(0.0))
    q1 = float(model.generalized_force_derivative(0.0))
    if not (np.isfinite(q0) and np.isfinite(q1)):
        return ToeEquilibriumSolution((), "nonfinite_coefficients", q0, q1)

    def h(theta: float) -> float:
        return float(model_residual(model, np.tan(theta), cfg))

    thetas = np.linspace(lo, hi, int(cfg.toe_root_scan_points))
    g = model_residual(model, np.tan(thetas), cfg)
    finite = np.isfinite(g)
    found: list[float] = []

    def _add(theta: float) -> None:
        if np.isfinite(theta) and lo - 1e-15 <= theta <= hi + 1e-15:
            theta = float(np.clip(theta, lo, hi))
            if all(abs(theta - t) > 1e-10 for t in found):
                found.append(theta)

    for i in np.flatnonzero(finite & (g == 0.0)):
        _add(float(thetas[i]))
    pair_ok = finite[:-1] & finite[1:]
    for i in np.flatnonzero(pair_ok & (g[:-1] * g[1:] < 0.0)):
        root = optimize.brentq(
            h, thetas[i], thetas[i + 1], xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=200
        )
        _add(float(root))
    for idx in (0, thetas.size - 1):
        if finite[idx]:
            tol = float(model_residual_tolerance(model, np.tan(thetas[idx]), cfg))
            if abs(float(g[idx])) <= tol:
                _add(float(thetas[idx]))
    absg = np.where(finite, np.abs(g), np.inf)
    mid = slice(1, thetas.size - 1)
    touching = (
        np.isfinite(absg[mid])
        & (absg[mid] <= absg[:-2])
        & (absg[mid] <= absg[2:])
        & (g[:-2] * g[2:] > 0.0)
        & (g[mid] * g[:-2] > 0.0)
    )
    for j in np.flatnonzero(touching):
        i = j + 1
        res = optimize.minimize_scalar(
            lambda t: abs(h(t)),
            bounds=(thetas[i - 1], thetas[i + 1]),
            method="bounded",
            options={"xatol": 1e-14},
        )
        t_star = float(res.x)
        tol = float(model_residual_tolerance(model, np.tan(t_star), cfg))
        if abs(h(t_star)) <= tol:
            _add(t_star)

    roots = sorted((_make_root(t, model, cfg, lo, hi) for t in found), key=lambda r: r.theta_rad)
    roots = tuple(r for r in roots if r.converged and np.isfinite(r.alpha))
    diagnostic = None
    if roots:
        status = "ok" if len(roots) == 1 else "multiple_roots"
    else:
        status = "no_root_in_bounds"
        if np.any(finite):
            i = int(np.argmin(absg))
            a_lo = thetas[max(i - 1, 0)]
            a_hi = thetas[min(i + 1, thetas.size - 1)]
            res = optimize.minimize_scalar(
                lambda t: abs(h(t)), bounds=(a_lo, a_hi), method="bounded", options={"xatol": 1e-14}
            )
            t_star = float(res.x) if abs(h(float(res.x))) <= absg[i] else float(thetas[i])
            diagnostic = _make_root(t_star, model, cfg, lo, hi)
    return ToeEquilibriumSolution(
        roots=roots,
        status=status,
        q0=q0,
        q1=q1,
        diagnostic=diagnostic,
        scan_theta_rad=thetas if keep_scan else None,
        scan_residual=g if keep_scan else None,
    )


def order_toe_roots(
    roots: tuple[ToeRoot, ...] | list[ToeRoot],
    *,
    previous_theta_rad: float | None = None,
    contact_violation: list[float] | None = None,
    force_residual: list[float] | None = None,
    continuity_tol_rad: float = 1.0e-9,
) -> list[int]:
    """Preference order: continuity, lowest potential, contact violation, force fit."""
    n = len(roots)
    cv = list(contact_violation) if contact_violation is not None else [0.0] * n
    fr = list(force_residual) if force_residual is not None else [0.0] * n

    def key(i: int) -> tuple:
        r = roots[i]
        if previous_theta_rad is not None and np.isfinite(previous_theta_rad):
            jump = abs(r.theta_rad - float(previous_theta_rad))
            cont = np.floor(jump / max(continuity_tol_rad, 1e-300))
        else:
            cont = 0.0
        return (
            cont,
            r.potential_J,
            cv[i] if np.isfinite(cv[i]) else np.inf,
            fr[i] if np.isfinite(fr[i]) else np.inf,
        )

    return sorted(range(n), key=key)


def relaxed_root_index(roots: tuple[ToeRoot, ...] | list[ToeRoot], theta0: float = 0.0) -> int | None:
    """Root on the branch connected to the unloaded neutral state (closest to theta0)."""
    if not roots:
        return None
    return int(np.argmin([abs(r.theta_rad - float(theta0)) for r in roots]))
