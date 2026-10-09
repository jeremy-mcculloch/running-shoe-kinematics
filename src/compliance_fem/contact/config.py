"""Problem configuration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

PLATE_AXIAL_MODEL = "exact_inextensible"
NU_LOCKING_WARN_THRESHOLD = 0.45
# Foam library: FFTurbo above the plate and FFLeap below it with a heel-to-toe
# modulus gradient by default.
FOAM_FFTURBO = "FFTurbo"
FOAM_FFLEAP = "FFLeap"
FOAM_MATERIALS = (FOAM_FFTURBO, FOAM_FFLEAP)
DEFAULT_UPPER_FOAM = FOAM_FFTURBO
DEFAULT_LOWER_FOAM = FOAM_FFLEAP


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
class FoamMaterial:
    """Isotropic foam with an optional linear heel-to-toe modulus gradient."""

    name: str
    E_heel: float
    E_toe: float
    nu: float

    def E(self, x: float | np.ndarray, L: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return self.E_heel + (self.E_toe - self.E_heel) * (x_arr / float(L))

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
class SoleConfig:
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
    ffturbo_E_Pa: float = 2.6e5
    ffturbo_nu: float = 0.113
    ffleap_E_heel_Pa: float = 3.54e5
    ffleap_E_toe_Pa: float = 2.07e5
    ffleap_nu: float = 0.113
    EI_plate_Nm2_per_m: float = 2.0
    # Refinement entries are multipliers of mesh_size_m.
    mesh_size_m: float = 0.003
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
            from compliance_fem.geometry.profile import load_normalized_sole_csv

            object.__setattr__(self, "normalized_geometry", load_normalized_sole_csv(self.geometry_csv))
        for name in (self.upper_foam_material, self.lower_foam_material):
            if name not in FOAM_MATERIALS:
                raise ValueError(f"Unknown foam material {name!r}; expected one of {FOAM_MATERIALS}.")
        for name, value in (
            ("ffturbo_E_Pa", self.ffturbo_E_Pa),
            ("ffleap_E_heel_Pa", self.ffleap_E_heel_Pa),
            ("ffleap_E_toe_Pa", self.ffleap_E_toe_Pa),
        ):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive.")
        _validate_poisson_ratio(self.ffturbo_nu, "ffturbo_nu")
        _validate_poisson_ratio(self.ffleap_nu, "ffleap_nu")
        if self.EI_plate_Nm2_per_m < 0.0:
            raise ValueError("EI_plate_Nm2_per_m must be nonnegative.")
        if not self.mesh_size_m > 0.0:
            raise ValueError("mesh_size_m must be positive.")
        for name in ("toe_refinement", "heel_corner_refinement", "interface_refinement", "plate_end_refinement"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1] (multiplier of mesh_size_m).")
        if not 0.0 < self.curvature_max_turn_deg <= 90.0:
            raise ValueError("curvature_max_turn_deg must lie in (0, 90].")
        if not 0.0 < self.landmark_tolerance < 0.05:
            raise ValueError("landmark_tolerance must lie in (0, 0.05) of the shoe length.")
        if self.element_order != 1:
            raise ValueError(
                "The measured-sole geometry supports element_order=1 (linear triangles) only."
            )

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

    def material(self, name: str) -> FoamMaterial:
        if name == FOAM_FFTURBO:
            return FoamMaterial(FOAM_FFTURBO, self.ffturbo_E_Pa, self.ffturbo_E_Pa, self.ffturbo_nu)
        if name == FOAM_FFLEAP:
            return FoamMaterial(FOAM_FFLEAP, self.ffleap_E_heel_Pa, self.ffleap_E_toe_Pa, self.ffleap_nu)
        raise ValueError(f"Unknown foam material {name!r}.")

    def region_materials(self) -> dict[str, FoamMaterial]:
        return {
            "upper_foam": self.material(self.upper_foam_material),
            "lower_foam": self.material(self.lower_foam_material),
        }
