"""Problem configuration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

GEOMETRY_TYPE_RECTANGLE = "rectangle"
GEOMETRY_TYPE_LAYERED = "layered_trapezoids_inextensible_plate"
PLATE_AXIAL_MODEL = "exact_inextensible"
NU_LOCKING_WARN_THRESHOLD = 0.45


def _validate_poisson_ratio(nu: float, name: str) -> None:
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio {name} must satisfy -1 < nu < 0.5 for plane strain.")
    if nu >= NU_LOCKING_WARN_THRESHOLD:
        warnings.warn(
            f"{name}={nu} is close to 0.5; a displacement-only continuum formulation may lock.",
            UserWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class RuntimeAngleConfig:
    """Runtime angle inputs and conditioning thresholds for the co-rotating frame.

    Angles are entered in degrees in the GUI and converted to radians before
    reaching the solver. ``phi`` is the absolute fixed-frame heel-to-toe chord
    angle (counterclockwise positive); ``theta`` is the toe-bending shape angle
    whose amplitude is ``tan(theta)``.
    """

    phi_default_deg: float = 0.0
    phi_min_deg: float = -30.0
    phi_max_deg: float = 30.0
    phi_step_deg: float = 0.1
    theta_default_deg: float = 5.0
    theta_min_deg: float = 0.0
    theta_max_deg: float = 45.0
    theta_step_deg: float = 0.1
    angle_display_units: Literal["deg"] = "deg"
    kf_cond_warn: float = 1e8
    lookup_schema_version: int = 6
    # Topology-selection default for the GUI contact-mode control. "auto" routes
    # on the two full-contact corner normal reactions.
    contact_mode_default: Literal["auto", "heel", "full", "toe"] = "auto"
    full_contact_anchor_fraction: float = 0.5
    tau_g_default: float = 0.0
    tau_R_default: float = 0.0

    def __post_init__(self) -> None:
        if not self.phi_min_deg <= self.phi_default_deg <= self.phi_max_deg:
            raise ValueError("phi_default_deg must lie inside [phi_min_deg, phi_max_deg].")
        if not self.theta_min_deg <= self.theta_default_deg <= self.theta_max_deg:
            raise ValueError("theta_default_deg must lie inside [theta_min_deg, theta_max_deg].")
        if self.phi_step_deg <= 0.0 or self.theta_step_deg <= 0.0:
            raise ValueError("Angle slider steps must be positive.")
        if abs(self.theta_min_deg) >= 90.0 or abs(self.theta_max_deg) >= 90.0:
            raise ValueError("theta range must stay strictly inside +-90 degrees (tan(theta)).")
        if self.kf_cond_warn <= 1.0:
            raise ValueError("kf_cond_warn must exceed 1.")
        if not 0.0 < self.full_contact_anchor_fraction < 1.0:
            raise ValueError("full_contact_anchor_fraction must lie strictly inside (0, 1).")
        if self.tau_g_default < 0.0 or self.tau_R_default < 0.0:
            raise ValueError("Admissibility tolerances must be non-negative.")


@dataclass(frozen=True)
class ProblemConfig:
    """Geometry, material, and discretization parameters."""

    L: float
    H: float
    E: float
    nu: float
    nx: int
    ny: int
    order: int = 1
    element_type: Literal["quad", "tri"] = "quad"

    def __post_init__(self) -> None:
        if self.L <= 0 or self.H <= 0:
            raise ValueError("L and H must be positive.")
        if self.E <= 0:
            raise ValueError("Young's modulus E must be positive.")
        _validate_poisson_ratio(self.nu, "nu")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be at least 1.")
        if self.order not in (1, 2):
            raise ValueError("Only element orders 1 and 2 are supported.")

    @property
    def geometry_type(self) -> str:
        return GEOMETRY_TYPE_RECTANGLE


@dataclass(frozen=True)
class LayeredPlateConfig:
    """Layered trapezoidal foams separated by an inextensible bending plate."""

    L: float
    h1_heel: float
    h1_toe: float
    h2_heel: float
    h2_toe: float
    E1: float
    nu1: float
    E_heel: float
    E_toe: float
    nu2: float
    EI_plate: float
    nx: int
    ny1: int
    ny2: int
    element_order: int = 1
    element_type: Literal["quad", "tri"] = "quad"

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be positive.")
        for name, value in (
            ("h1_heel", self.h1_heel),
            ("h1_toe", self.h1_toe),
            ("h2_heel", self.h2_heel),
            ("h2_toe", self.h2_toe),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, value in (("E1", self.E1), ("E_heel", self.E_heel), ("E_toe", self.E_toe)):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.EI_plate < 0:
            raise ValueError("EI_plate must be nonnegative.")
        _validate_poisson_ratio(self.nu1, "nu1")
        _validate_poisson_ratio(self.nu2, "nu2")
        if self.nx < 1 or self.ny1 < 1 or self.ny2 < 1:
            raise ValueError("nx, ny1, and ny2 must be at least 1.")
        if self.element_order not in (1, 2):
            raise ValueError("Only element orders 1 and 2 are supported.")

    @property
    def geometry_type(self) -> str:
        return GEOMETRY_TYPE_LAYERED

    @property
    def H(self) -> float:
        """Reference outer height for lookup/GUI scale compatibility."""
        return max(self.h1_heel + self.h2_heel, self.h1_toe + self.h2_toe)

    @property
    def E(self) -> float:
        """Representative modulus (upper foam) for legacy NPZ keys."""
        return self.E1

    @property
    def nu(self) -> float:
        """Representative Poisson ratio (upper foam) for legacy NPZ keys."""
        return self.nu1

    @property
    def order(self) -> int:
        return self.element_order

    @property
    def ny(self) -> int:
        return self.ny1 + self.ny2

    def h1(self, x: float | np.ndarray) -> np.ndarray:
        """Upper-foam thickness h1(x)."""
        x_arr = np.asarray(x, dtype=float)
        return self.h1_heel + (self.h1_toe - self.h1_heel) * (x_arr / self.L)

    def h2(self, x: float | np.ndarray) -> np.ndarray:
        """Lower-foam thickness h2(x)."""
        x_arr = np.asarray(x, dtype=float)
        return self.h2_heel + (self.h2_toe - self.h2_heel) * (x_arr / self.L)

    def y_plate(self, x: float | np.ndarray) -> np.ndarray:
        """Plate/interface centerline y_p(x) = h2(x)."""
        return self.h2(x)

    def y_top(self, x: float | np.ndarray) -> np.ndarray:
        """Outer top surface y_t(x) = h2(x) + h1(x)."""
        return self.h2(x) + self.h1(x)

    def y_bottom(self, x: float | np.ndarray) -> np.ndarray:
        """Ground-facing bottom surface y_b(x) = 0."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def E2(self, x: float | np.ndarray) -> np.ndarray:
        """Lower-foam Young's modulus E2(x)."""
        x_arr = np.asarray(x, dtype=float)
        return self.E_heel + (self.E_toe - self.E_heel) * (x_arr / self.L)

    @property
    def plate_slope(self) -> float:
        """Constant interface slope m_p."""
        return (self.h2_toe - self.h2_heel) / self.L

    @property
    def plate_gamma(self) -> float:
        """γ_p = sqrt(1 + m_p^2)."""
        return float(np.sqrt(1.0 + self.plate_slope**2))

    @property
    def plate_length(self) -> float:
        """Plate arc length L_p = γ_p L."""
        return self.plate_gamma * self.L

    @property
    def plate_tangent(self) -> np.ndarray:
        """Unit tangent t_p = (1, m_p) / γ_p, heel to toe."""
        gamma = self.plate_gamma
        return np.array([1.0 / gamma, self.plate_slope / gamma], dtype=float)

    @property
    def plate_normal(self) -> np.ndarray:
        """Unit normal n_p = (-m_p, 1) / γ_p."""
        gamma = self.plate_gamma
        return np.array([-self.plate_slope / gamma, 1.0 / gamma], dtype=float)

    @property
    def lower_vertices(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.0, 0.0),
            (self.L, 0.0),
            (self.L, self.h2_toe),
            (0.0, self.h2_heel),
        )

    @property
    def upper_vertices(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.0, self.h2_heel),
            (self.L, self.h2_toe),
            (self.L, self.h2_toe + self.h1_toe),
            (0.0, self.h2_heel + self.h1_heel),
        )
