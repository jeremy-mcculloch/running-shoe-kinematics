"""Problem configuration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

GEOMETRY_TYPE_RECTANGLE = "rectangle"
GEOMETRY_TYPE_LAYERED = "layered_trapezoids_inextensible_plate"
GEOMETRY_TYPE_MEASURED = "measured_sole"
PLATE_AXIAL_MODEL = "exact_inextensible"
NU_LOCKING_WARN_THRESHOLD = 0.45
CONTINUUM_MODEL = "plane_strain"

# Foam library. The layered model's established arrangement is FFTurbo above the
# plate (E1, nu1) and FFLeap below it with a heel-to-toe modulus gradient.
FOAM_FFTURBO = "FFTurbo"
FOAM_FFLEAP = "FFLeap"
FOAM_MATERIALS = (FOAM_FFTURBO, FOAM_FFLEAP)
DEFAULT_UPPER_FOAM = FOAM_FFTURBO
DEFAULT_LOWER_FOAM = FOAM_FFLEAP


def rocker_bottom_profile(
    x: float | np.ndarray,
    L: float,
    height: float,
    apex_fraction: float,
) -> np.ndarray:
    """Reference bottom profile ``y_b(x) = h ((x - x_c) / max(x_c, L - x_c))^2``.

    A symmetric-parabola rocker sole: lowest point ``y_b = 0`` at the apex
    ``x_c = apex_fraction * L`` and height ``h`` at the farther end. ``h = 0``
    is the flat sole ``y_b = 0``.
    """
    x_arr = np.asarray(x, dtype=float)
    if float(height) == 0.0:
        return np.zeros_like(x_arr)
    x_c = float(apex_fraction) * float(L)
    half = max(x_c, float(L) - x_c)
    return float(height) * ((x_arr - x_c) / half) ** 2


def _validate_rocker(height: float, apex_fraction: float, max_height: float) -> None:
    if height < 0.0:
        raise ValueError("sole_rocker_height must be non-negative.")
    if height >= max_height:
        raise ValueError(
            f"sole_rocker_height={height} must be smaller than the mapped foam thickness {max_height}."
        )
    if not 0.0 <= apex_fraction <= 1.0:
        raise ValueError("sole_rocker_apex must lie in [0, 1].")


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
    lookup_schema_version: int = 10
    # Interval-selection default for the GUI contact-mode control.
    contact_mode_default: Literal["auto", "heel", "interior", "toe", "full", "specific"] = "auto"
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
    # Optional curved (rocker) sole; 0 keeps the flat bottom y = 0.
    sole_rocker_height: float = 0.0
    sole_rocker_apex: float = 0.5

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
        _validate_rocker(self.sole_rocker_height, self.sole_rocker_apex, self.H)

    @property
    def geometry_type(self) -> str:
        return GEOMETRY_TYPE_RECTANGLE

    def y_bottom(self, x: float | np.ndarray) -> np.ndarray:
        """Ground-facing reference bottom surface ``y_b(x)``."""
        return rocker_bottom_profile(x, self.L, self.sole_rocker_height, self.sole_rocker_apex)

    def y_top(self, x: float | np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.H)


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
    # Optional curved (rocker) sole applied to the lower foam; 0 keeps y_b = 0.
    sole_rocker_height: float = 0.0
    sole_rocker_apex: float = 0.5

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
        _validate_rocker(
            self.sole_rocker_height, self.sole_rocker_apex, min(self.h2_heel, self.h2_toe)
        )

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
        """Ground-facing reference bottom surface y_b(x) (0 for a flat sole)."""
        return rocker_bottom_profile(x, self.L, self.sole_rocker_height, self.sole_rocker_apex)

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


@dataclass(frozen=True)
class FoamMaterial:
    """Isotropic foam with an optional linear heel-to-toe modulus gradient."""

    name: str
    E_heel: float
    E_toe: float
    nu: float

    def E(self, x: float | np.ndarray, L: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return self.E_heel + (self.E_toe - self.E_heel) * (x_arr / float(L))

    @property
    def is_graded(self) -> bool:
        return self.E_heel != self.E_toe


def validate_shoe_length_mm(shoe_length_mm) -> float:
    """Return a validated projected heel-to-toe shoe length in millimetres."""
    if shoe_length_mm is None:
        raise ValueError(
            "shoe_length_mm is required for the measured-sole geometry: the projected "
            "heel-bottom-to-toe-tip length in millimetres (not the outsole arc length)."
        )
    try:
        value = float(shoe_length_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"shoe_length_mm must be a number, got {shoe_length_mm!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"shoe_length_mm must be finite, got {shoe_length_mm!r}.")
    if value <= 0.0:
        raise ValueError(f"shoe_length_mm must be positive, got {value}.")
    return value


@dataclass(frozen=True)
class MeasuredSoleConfig:
    """Image-derived carbon-plated sole: two conforming foams and a partial curved plate.

    The geometry comes from a normalized CSV whose coordinates are divided by the
    projected heel-bottom-to-toe-tip length; ``shoe_length_mm`` restores SI
    coordinates with one common scale for both axes.
    """

    shoe_length_mm: float | None
    geometry_csv: str | None = None
    normalized_geometry: object | None = field(default=None, compare=False, repr=False)
    upper_foam_material: str = DEFAULT_UPPER_FOAM
    lower_foam_material: str = DEFAULT_LOWER_FOAM
    ffturbo_E: float = 2.6e5
    ffturbo_nu: float = 0.113
    ffleap_E_heel: float = 3.54e5
    ffleap_E_toe: float = 2.07e5
    ffleap_nu: float = 0.113
    EI_plate: float = 2.0
    # Mesh sizes in metres; refinement entries are multipliers of mesh_size.
    mesh_size: float = 0.003
    toe_refinement: float = 0.4
    heel_corner_refinement: float = 0.5
    interface_refinement: float = 0.6
    plate_end_refinement: float = 0.4
    curvature_max_turn_deg: float = 12.0
    min_angle_deg: float = 12.0
    # Landmark-on-curve tolerance as a fraction of the shoe length.
    landmark_tolerance: float = 2.0e-3
    element_order: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "shoe_length_mm", validate_shoe_length_mm(self.shoe_length_mm))
        if self.normalized_geometry is None:
            if self.geometry_csv is None:
                raise ValueError("Measured-sole geometry requires geometry_csv (normalized CSV path).")
            from compliance_fem.measured_geometry import load_normalized_sole_csv

            object.__setattr__(self, "normalized_geometry", load_normalized_sole_csv(self.geometry_csv))
        for name in (self.upper_foam_material, self.lower_foam_material):
            if name not in FOAM_MATERIALS:
                raise ValueError(f"Unknown foam material {name!r}; expected one of {FOAM_MATERIALS}.")
        for name, value in (
            ("ffturbo_E", self.ffturbo_E),
            ("ffleap_E_heel", self.ffleap_E_heel),
            ("ffleap_E_toe", self.ffleap_E_toe),
        ):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive.")
        _validate_poisson_ratio(self.ffturbo_nu, "ffturbo_nu")
        _validate_poisson_ratio(self.ffleap_nu, "ffleap_nu")
        if self.EI_plate < 0.0:
            raise ValueError("EI_plate must be nonnegative.")
        if not self.mesh_size > 0.0:
            raise ValueError("mesh_size must be positive.")
        for name in ("toe_refinement", "heel_corner_refinement", "interface_refinement", "plate_end_refinement"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1] (multiplier of mesh_size).")
        if not 0.0 < self.curvature_max_turn_deg <= 90.0:
            raise ValueError("curvature_max_turn_deg must lie in (0, 90].")
        if not 0.0 < self.landmark_tolerance < 0.05:
            raise ValueError("landmark_tolerance must lie in (0, 0.05) of the shoe length.")
        if self.element_order != 1:
            raise ValueError(
                "The measured-sole geometry supports element_order=1 (linear triangles) only."
            )

    @property
    def geometry_type(self) -> str:
        return GEOMETRY_TYPE_MEASURED

    @property
    def shoe_length_m(self) -> float:
        return float(self.shoe_length_mm) / 1000.0

    @property
    def L(self) -> float:
        """Projected heel-to-toe length in metres (the lookup length scale)."""
        return self.shoe_length_m

    @property
    def H(self) -> float:
        """Vertical extent of the scaled sole (display and lookup scale)."""
        geom = self.normalized_geometry
        ys = np.concatenate([np.asarray(c)[:, 1] for c in geom.curves.values()])
        return float(self.shoe_length_m * (ys.max() - ys.min()))

    @property
    def E(self) -> float:
        return float(self.material(self.upper_foam_material).E_heel)

    @property
    def nu(self) -> float:
        return float(self.material(self.upper_foam_material).nu)

    @property
    def order(self) -> int:
        return self.element_order

    @property
    def nx(self) -> int:
        return max(1, int(round(self.shoe_length_m / self.mesh_size)))

    @property
    def ny(self) -> int:
        return 1

    @property
    def source_geometry_filename(self) -> str:
        geom = self.normalized_geometry
        name = getattr(geom, "source_filename", None) or self.geometry_csv or ""
        return str(name)

    def material(self, name: str) -> FoamMaterial:
        if name == FOAM_FFTURBO:
            return FoamMaterial(FOAM_FFTURBO, self.ffturbo_E, self.ffturbo_E, self.ffturbo_nu)
        if name == FOAM_FFLEAP:
            return FoamMaterial(FOAM_FFLEAP, self.ffleap_E_heel, self.ffleap_E_toe, self.ffleap_nu)
        raise ValueError(f"Unknown foam material {name!r}.")

    def region_materials(self) -> dict[str, FoamMaterial]:
        return {
            "upper_foam": self.material(self.upper_foam_material),
            "lower_foam": self.material(self.lower_foam_material),
        }
