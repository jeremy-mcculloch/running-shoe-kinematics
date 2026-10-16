"""Passive toe-joint spring: exact quasistatic equilibrium of the shape coordinate.

Generalized coordinate
----------------------
The single top shape mode prescribes the rotating-frame top displacement

    u_t(alpha) = alpha [0; phi1(x)],        alpha = tan(theta),

with ``phi1`` in metres (``contact_basis.shape_mode_phi1``) and ``alpha``
dimensionless. ``phi1`` vanishes at the heel (x=0) and the toe tip (x=L), so
the top-attached frame ``T`` is the heel-to-toe *chord* frame. ``theta = arctan(alpha)``
is kept exact; nothing here uses a small-angle approximation.

What is held fixed: the rearfoot
--------------------------------
The measured foot pitch ``phi`` is the heel -> metatarsal-head line, i.e. the
rearfoot segment [0, L - toe_length], not the heel-to-toe chord. With
``x_mtp = L - toe_length`` (the MTP point), bending the toe at a fixed rearfoot
rotates the chord, so the chord-frame rotation depends on alpha exactly as

    varphi(alpha) = phi - atan2(y_top(x_mtp) - y_top(0) + alpha phi1(x_mtp), x_mtp),

and the foot's virtual displacement for the toe coordinate is the sole motion
relative to the rearfoot line,

    psi(x) = phi1(x) - (x / x_mtp) phi1(x_mtp),        psi(0) = psi(x_mtp) = 0,

which is ~ max(0, x - x_mtp): a rotation of the toe segment about the MTP point.

Generalized force (virtual work)
--------------------------------
The contact operator uses ``u = C F + R a`` with ``R^T F = 0``: ``F_t`` is the external
nodal load applied *to* the shoe top, i.e. the foot-on-shoe force (the load
term of ``K u = f`` in the FEM weak form). The virtual work of that load on a
toe variation at fixed rearfoot is ``(psi^T f_{t,y}) delta alpha``, so by
action-reaction the shoe acts on the foot's toe coordinate with

    Q_alpha,shoe->foot = -psi^T f_{t,y}        (N·m per metre width).

Because ``psi ~ max(0, x - x_mtp)``, ``Q_alpha,shoe->foot ~ -M_toe``: a negative
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

``Q`` comes from the converged continuous contact solve at ``varphi(alpha)``
(``contact.continuous_toe``). With ``Pi(alpha) = U(alpha) - int_0^alpha Q`` we
have ``Pi' = -g``; a root is stable when ``Pi'' = d2U/dalpha2 - dQ/dalpha > 0``.
Damping ``c_toe`` is recorded but never enters this quasistatic equation.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from compliance_fem.contact.basis import shape_mode_phi1

TOE_MODEL_PASSIVE_SPRING = "passive_spring"
TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT = "passive_spring_elastic_equivalent"
TOE_MODELS = (TOE_MODEL_PASSIVE_SPRING, TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT)
DEFAULT_TOE_MODEL = TOE_MODEL_PASSIVE_SPRING

TOE_GENERALIZED_FORCE_DEFINITION = (
    "Q_alpha_shoe_on_foot = -psi^T f_{t,y}, psi(x) = phi1(x) - (x/x_mtp) phi1(x_mtp), "
    "x_mtp = L - toe_length (rearfoot heel->MTP line held fixed); f_t is the "
    "foot-on-shoe nodal load (load term of K u = f), phi1 the top vertical shape mode in metres"
)
TOE_EQUATION = (
    "g(alpha) = Q_alpha_shoe_on_foot(alpha) - k*(arctan(alpha) - theta0)/(1 + alpha^2) = 0, "
    "theta = arctan(alpha), chord rotation varphi(alpha) = phi_rearfoot - "
    "atan2(y_top(x_mtp) - y_top(0) + alpha*phi1(x_mtp), x_mtp), x_mtp = L - toe_length"
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
        if not np.isfinite(self.toe_stiffness_Nm_per_rad) or self.toe_stiffness_Nm_per_rad < 0.0:
            raise ValueError(f"toe_stiffness_Nm_per_rad must be finite and >= 0, got {self.toe_stiffness_Nm_per_rad}.")
        if not np.isfinite(self.toe_damping_Nms_per_rad) or self.toe_damping_Nms_per_rad < 0.0:
            raise ValueError(f"toe_damping_Nms_per_rad must be finite and >= 0, got {self.toe_damping_Nms_per_rad}.")
        if not np.isfinite(self.toe_neutral_angle_rad) or abs(self.toe_neutral_angle_rad) >= 0.5 * np.pi:
            raise ValueError(f"toe_neutral_angle_rad must lie strictly inside +-pi/2, got {self.toe_neutral_angle_rad}.")
        if not (np.isfinite(self.toe_angle_min_deg) and np.isfinite(self.toe_angle_max_deg)) or not (-90.0 < self.toe_angle_min_deg < self.toe_angle_max_deg < 90.0):
            raise ValueError(f"toe angle bounds must satisfy -90 < toe_angle_min_deg < toe_angle_max_deg < 90, got [{self.toe_angle_min_deg}, {self.toe_angle_max_deg}].")
        if not self.toe_equilibrium_abs_tol_Nm > 0.0:
            raise ValueError("toe_equilibrium_abs_tol_Nm must be positive.")
        if not self.toe_equilibrium_rel_tol >= 0.0:
            raise ValueError("toe_equilibrium_rel_tol must be >= 0.")
        if int(self.toe_root_scan_points) < 3:
            raise ValueError("toe_root_scan_points must be >= 3.")
        if not self.toe_low_force_threshold_N >= 0.0:
            raise ValueError("toe_low_force_threshold_N must be >= 0.")

    @property
    def toe_angle_min_rad(self) -> float:
        return float(np.deg2rad(self.toe_angle_min_deg))

    @property
    def toe_angle_max_rad(self) -> float:
        return float(np.deg2rad(self.toe_angle_max_deg))

    def to_dict(self) -> dict:
        return asdict(self)


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


def q_theta_from_q_alpha(q_alpha, alpha):
    """Q_theta = Q_alpha d(alpha)/d(theta) = Q_alpha (1 + alpha^2)."""
    a = np.asarray(alpha, dtype=float)
    return np.asarray(q_alpha, dtype=float) * (1.0 + a * a)


@dataclass(frozen=True)
class RearfootGeometry:
    """Rearfoot line (heel x=0 -> MTP x=L - toe_length) in the reference chord frame.

    ``L`` and ``toe_length`` are in metres. ``phi1_mtp`` is the shape mode at the
    MTP point (m, < 0) and ``dy_mtp`` the reference top rise
    ``y_top(L - toe_length) - y_top(0)`` (m).
    """

    L: float
    toe_length: float
    phi1_mtp: float
    dy_mtp: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.L) and np.isfinite(self.toe_length) and 0.0 < self.toe_length < self.L):
            raise ValueError(f"toe_length must lie inside (0, L={self.L}), got {self.toe_length}.")
        if not (np.isfinite(self.phi1_mtp) and np.isfinite(self.dy_mtp)):
            raise ValueError("Rearfoot geometry must be finite.")

    @property
    def reference_angle(self) -> float:
        """Reference rearfoot-line angle in the mesh frame, atan2(dy_mtp, L - toe_length)."""
        return float(np.arctan2(self.dy_mtp, self.L - self.toe_length))

    def chord_rotation(self, phi_rearfoot: float, alpha):
        """varphi(alpha) = phi_rearfoot - atan2(dy_mtp + alpha phi1_mtp, L - toe_length)."""
        a = np.asarray(alpha, dtype=float)
        return float(phi_rearfoot) - np.arctan2(self.dy_mtp + a * self.phi1_mtp, self.L - self.toe_length)

    def chord_rotation_derivative(self, alpha):
        """d varphi / d alpha = -x_mtp phi1_mtp / (x_mtp^2 + (dy_mtp + alpha phi1_mtp)^2), x_mtp = L - toe_length."""
        a = np.asarray(alpha, dtype=float)
        x_mtp = self.L - self.toe_length
        h = self.dy_mtp + a * self.phi1_mtp
        return -x_mtp * self.phi1_mtp / (x_mtp * x_mtp + h * h)

    def to_dict(self) -> dict:
        return {
            "L": float(self.L),
            "toe_length": float(self.toe_length),
            "phi1_mtp": float(self.phi1_mtp),
            "dy_mtp": float(self.dy_mtp),
        }


def rearfoot_toe_mode(
    phi1_top: np.ndarray, x_top: np.ndarray, L: float, toe_length: float, phi1_mtp: float
) -> np.ndarray:
    """psi(x) = phi1(x) - (x/(L - toe_length)) phi1_mtp: sole motion relative to the heel->MTP line."""
    x = np.asarray(x_top, dtype=float)
    return np.asarray(phi1_top, dtype=float) - (x / (float(L) - float(toe_length))) * float(phi1_mtp)


def ramp_virtual_displacement(x_top: np.ndarray, x_mtp: float) -> np.ndarray:
    """Return rho[j] = max(0, x_top[j] - x_mtp), with x_mtp = L - toe_length."""
    return np.maximum(0.0, np.asarray(x_top, dtype=float) - float(x_mtp))


def interpolate_top_height(x_top: np.ndarray, y_top: np.ndarray, x: float) -> float:
    """Height of the top surface at ``x``, interpolated from nodal y_top."""
    return float(np.interp(float(x), np.asarray(x_top, dtype=float), np.asarray(y_top, dtype=float)))


def ramp_horizontal_lever(x_top: np.ndarray, y_top: np.ndarray, x_mtp: float, H_mtp: float) -> np.ndarray:
    """eta[j] = (y_j - H_mtp) for x_j >= x_mtp, else 0. Flat top => eta = 0."""
    x_top = np.asarray(x_top, dtype=float)
    y_top = np.asarray(y_top, dtype=float)
    eta = np.zeros_like(x_top, dtype=float)
    mask = x_top >= float(x_mtp)
    eta[mask] = y_top[mask] - float(H_mtp)
    return eta


def compute_toe_moment(
    f_tx: np.ndarray,
    f_ty: np.ndarray,
    x_top: np.ndarray,
    y_top: np.ndarray,
    x_mtp: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Toe moment about P_toe = (x_mtp, H_mtp): T = rho^T f_ty - eta^T f_tx.

    ``x_mtp = L - toe_length`` is the MTP point. Returns
    ``(T_toe, T_toe_vertical, H_mtp)``. Columns of ``f_tx`` / ``f_ty`` are RHS.
    """
    f_tx = np.asarray(f_tx, dtype=float)
    f_ty = np.asarray(f_ty, dtype=float)
    squeeze = f_tx.ndim == 1
    if squeeze:
        f_tx = f_tx[:, None]
        f_ty = f_ty[:, None]
    H_mtp = interpolate_top_height(x_top, y_top, x_mtp)
    rho = ramp_virtual_displacement(x_top, x_mtp)
    eta = ramp_horizontal_lever(x_top, y_top, x_mtp, H_mtp)
    T = rho @ f_ty - eta @ f_tx
    T_vert = rho @ f_ty
    if squeeze:
        return np.asarray(T).reshape(-1)[0], np.asarray(T_vert).reshape(-1)[0], H_mtp
    return np.asarray(T, dtype=float), np.asarray(T_vert, dtype=float), H_mtp


def warn_if_mtp_not_on_top_grid(x_top: np.ndarray, L: float, toe_length: float) -> None:
    """Warn when L - toe_length is not a top-node coordinate (nodal ramp is used; no remesh)."""
    x_top = np.asarray(x_top, dtype=float)
    if not np.any(np.isclose(x_top, float(L) - float(toe_length), rtol=0.0, atol=1e-12)):
        warnings.warn(
            f"Softplus joint L - toe_length = {float(L) - float(toe_length)} m is not a top-node coordinate; "
            "toe moment uses the existing nodal ramp without remeshing.",
            UserWarning,
            stacklevel=2,
        )


def rearfoot_geometry(
    x_top: np.ndarray,
    y_top: np.ndarray | None,
    L: float,
    toe_length: float,
    kappa: float,
) -> RearfootGeometry:
    """Heel->MTP line of the reference top, with the MTP point at ``x = L - toe_length``."""
    x_mtp = float(L) - float(toe_length)
    phi1_mtp = float(shape_mode_phi1(np.array([x_mtp]), float(L), float(toe_length), float(kappa))[0])
    if y_top is None:
        dy_mtp = 0.0
    else:
        dy_mtp = interpolate_top_height(x_top, y_top, x_mtp) - interpolate_top_height(x_top, y_top, 0.0)
    return RearfootGeometry(L=float(L), toe_length=float(toe_length), phi1_mtp=phi1_mtp, dy_mtp=float(dy_mtp))
