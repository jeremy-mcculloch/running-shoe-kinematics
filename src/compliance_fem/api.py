"""Stateful Python interface for driving the sole model from an external codebase.

Each :meth:`SoleModel.step` takes the fixed-frame load on the shoe top and the
rearfoot pitch, solves the passive toe-spring equilibrium over the stored
contact intervals (the toe angle is an output), and returns nodal positions,
displacements, forces and moments. No FEM solve happens at runtime; the
precomputed contact lookup is only contracted.

Conventions (all inputs and outputs are per shoe, SI):

- Fixed frame: x forward (heel -> toe), y up, ground at y = 0.
- ``Fx_N``, ``Fy_N``: force applied **by the foot on the shoe top**
  (``F_top = -GRF``), so normal stance loading has ``Fy_N < 0``.
- ``phi_deg``: fixed-frame heel -> MTP (rearfoot) angle, counterclockwise
  positive, in degrees.
- The lookup is 2D plane strain per metre width; loads are divided by
  ``shoe_width_m`` on input and forces / moments multiplied by it on output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from compliance_fem.contact_lookup import SCALAR_MZ, SCALAR_TOE, ContactLookupResult, load_contact_lookup
from compliance_fem.corotation import contract_basis
from compliance_fem.force_control import EXACT_ALL, SelectionConfig, Tolerances, reconstruct_rows
from compliance_fem.gait.passive_toe import pick_instant_best_passive, solve_passive_toe_candidates
from compliance_fem.toe_spring import ToeSpringConfig


def _nan2(n: int) -> np.ndarray:
    return np.full((2, n), np.nan)


@dataclass(frozen=True)
class SoleState:
    """Result of one :meth:`SoleModel.step`.

    Nodal arrays are ``(2, n)`` with rows ``(x, y)``; top nodes run heel -> toe
    along the top surface, bottom nodes heel -> toe along the sole.
    """

    valid: bool
    admissible: bool
    status: str
    theta_deg: float
    phi_deg: float
    chord_rotation_rad: float
    contact_interval: tuple[int, int] | None
    contact_start_x_m: float
    contact_end_x_m: float
    # Deformed positions in the fixed (ground) frame.
    top_xy_m: np.ndarray
    bottom_xy_m: np.ndarray
    # Displacements in the rotating shoe frame (reference configuration axes).
    top_displacement_m: np.ndarray
    bottom_displacement_m: np.ndarray
    # Fixed-frame nodal forces: foot-on-shoe at the top, ground-on-shoe at the bottom.
    top_force_N: np.ndarray
    bottom_reaction_N: np.ndarray
    contact_mask: np.ndarray
    Fx_N: float
    Fy_N: float
    # sum(x f_y - y f_x) of the top forces, shoe-frame components and reference
    # coordinates, about the heel-bottom reference point (lookup / gait convention).
    Mz_shoe_Nm: float
    # Same moment from fixed-frame forces and deformed positions, about the ground origin.
    Mz_fixed_Nm: float
    toe_moment_Nm: float
    toe_spring_moment_Nm: float
    force_residual_N: float
    moment_residual_Nm: float
    toe_root_count: int
    toe_stable: bool
    low_load: bool
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly copy (arrays become nested lists)."""
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
        return out


class SoleModel:
    """Load a contact lookup once and evaluate it step by step.

    The model keeps the previous contact interval and toe angle so consecutive
    steps stay on a continuous branch; call :meth:`reset` between independent
    sequences.
    """

    def __init__(
        self,
        lookup: ContactLookupResult | str | Path,
        *,
        shoe_width_m: float = 0.10,
        toe_config: ToeSpringConfig | None = None,
        tolerances: Tolerances | None = None,
        selection_config: SelectionConfig | None = None,
        cond_limit: float = 1.0e8,
    ) -> None:
        self.lookup = lookup if isinstance(lookup, ContactLookupResult) else load_contact_lookup(lookup)
        width = float(shoe_width_m)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"shoe_width_m must be positive, got {shoe_width_m!r}.")
        self.shoe_width_m = width
        self.toe_config = toe_config or ToeSpringConfig()
        if not self.toe_config.is_passive:
            raise ValueError("SoleModel solves the passive toe spring; use a passive toe_model.")
        self.tolerances = tolerances or Tolerances()
        self.selection_config = selection_config or SelectionConfig()
        self.cond_limit = float(cond_limit)
        self.setup = None
        lk = self.lookup
        self.x_top = np.asarray(lk.x_top, dtype=float)
        self.y_top = np.asarray(lk.y_top if lk.y_top is not None else np.full(self.x_top.size, lk.H), dtype=float)
        self.x_bottom = np.asarray(lk.x_bottom, dtype=float)
        self.y_bottom = np.asarray(
            lk.y_bottom if lk.y_bottom is not None else np.zeros(self.x_bottom.size), dtype=float
        )
        self.reset()

    @classmethod
    def from_config(
        cls,
        config: str | Path,
        *,
        lookup: ContactLookupResult | str | Path | None = None,
        build_if_missing: bool = False,
        require_matching_lookup: bool = True,
        tolerances: Tolerances | None = None,
        selection_config: SelectionConfig | None = None,
    ) -> SoleModel:
        """Build from a measured-sole JSON config.

        Shoe width and toe spring come from the config; the lookup is read from
        ``lookup.output_dir`` unless ``lookup`` is given. With
        ``build_if_missing`` a missing or stale lookup is (re)generated. With
        ``require_matching_lookup`` a lookup whose recorded build fingerprint
        differs from the config raises; lookups without a fingerprint only warn.
        """
        import warnings

        from compliance_fem.measured_config_file import (
            ConfigFileError,
            ensure_measured_lookup,
            load_measured_sole_config,
            lookup_matches_setup,
        )

        setup = load_measured_sole_config(config)
        if lookup is None:
            if setup.output_dir is None:
                raise ConfigFileError(f"{config}: set lookup.output_dir or pass lookup=... to SoleModel.from_config.")
            lookup = ensure_measured_lookup(setup) if build_if_missing else setup.output_dir
        lk = lookup if isinstance(lookup, ContactLookupResult) else load_contact_lookup(lookup)
        match = lookup_matches_setup(lk, setup)
        if match is False and require_matching_lookup:
            raise ConfigFileError(
                f"The lookup was built from different parameters than {config}. Regenerate it with "
                f"`python -m compliance_fem.measured_config_file build {config}` "
                "or pass require_matching_lookup=False."
            )
        if match is None:
            warnings.warn(
                "The lookup records no measured-setup fingerprint; cannot confirm it matches the config.",
                UserWarning,
                stacklevel=2,
            )
        model = cls(
            lk,
            shoe_width_m=setup.runtime.shoe_width_m,
            toe_config=setup.toe_spring,
            tolerances=tolerances,
            selection_config=selection_config,
        )
        model.setup = setup
        return model

    @property
    def n_top(self) -> int:
        return int(self.x_top.size)

    @property
    def n_bottom(self) -> int:
        return int(self.x_bottom.size)

    def reset(self) -> None:
        self._previous_interval: tuple[int, int] | None = None
        self._previous_theta_deg: float | None = None

    def step(self, Fx_N: float, Fy_N: float, phi_deg: float, *, Mz_Nm: float = float("nan")) -> SoleState:
        """Solve one load state. ``Mz_Nm`` (optional, measured top moment about
        the heel-bottom reference point) is not prescribed; it only sets
        ``moment_residual_Nm``."""
        w = self.shoe_width_m
        F_model = np.array([float(Fx_N), float(Fy_N)]) / w
        low_load = float(np.hypot(Fx_N, Fy_N)) < float(self.toe_config.toe_low_force_threshold_N)
        cands = solve_passive_toe_candidates(
            self.lookup,
            Fx_star=float(F_model[0]),
            Fy_star=float(F_model[1]),
            Mz_meas=float(Mz_Nm) / w,
            phi_rad=float(np.deg2rad(phi_deg)),
            config=self.toe_config,
            width_m=w,
            tolerances=self.tolerances,
            low_load=low_load,
            cond_limit=self.cond_limit,
            previous_interval=self._previous_interval,
            selection_config=self.selection_config,
        )
        best = pick_instant_best_passive(cands, self._previous_theta_deg)
        if best is None or best.row < 0 or not np.all(np.isfinite(best.gamma)):
            return self._invalid_state(phi_deg, low_load, "no_candidate")

        varphi = float(best.toe.varphi_rad)
        rec = reconstruct_rows(
            self.lookup,
            np.array([best.row]),
            np.asarray(best.gamma, dtype=float)[None, :],
            varphi,
            F_model,
            self.tolerances,
            full_fields=True,
            exact=EXACT_ALL,
        )
        c, s = np.cos(varphi), np.sin(varphi)
        Q = np.array([[c, -s], [s, c]])
        x_a, y_a = float(best.anchor_x), float(best.anchor_y)
        d_a = np.array([[best.d_ax], [best.d_ay]])
        r_a = np.array([[x_a], [0.0]])

        d_top = np.vstack([rec["top_u"][0], rec["top_v"][0]])
        d_bot = np.vstack([rec["full_bottom_u"][0], rec["full_bottom_v"][0]])
        X_top = np.vstack([self.x_top - x_a, self.y_top - y_a])
        X_bot = np.vstack([self.x_bottom - x_a, self.y_bottom - y_a])
        top_xy = r_a + Q @ (X_top + d_top - d_a)
        bottom_xy = r_a + Q @ (X_bot + d_bot - d_a)
        f_top = w * (Q @ np.vstack([rec["top_force_x"][0], rec["top_force_y"][0]]))
        r_bot = w * (Q @ np.vstack([rec["full_bottom_reaction_x"][0], rec["full_bottom_reaction_y"][0]]))

        S = contract_basis(np.asarray(self.lookup.scalar_lookup[best.row], dtype=float), best.gamma, mode_axis=0)
        Mz_fixed = float(np.sum(top_xy[0] * f_top[1] - top_xy[1] * f_top[0]))
        start, end = best.interval
        contact = np.zeros(self.n_bottom, dtype=bool)
        contact[start : end + 1] = True

        self._previous_interval = best.interval
        if np.isfinite(best.theta_deg):
            self._previous_theta_deg = float(best.theta_deg)
        toe = best.toe
        return SoleState(
            valid=True,
            admissible=bool(best.admissible),
            status=str(best.status),
            theta_deg=float(best.theta_deg),
            phi_deg=float(phi_deg),
            chord_rotation_rad=varphi,
            contact_interval=(int(start), int(end)),
            contact_start_x_m=float(best.contact_start_x),
            contact_end_x_m=float(best.contact_end_x),
            top_xy_m=top_xy,
            bottom_xy_m=bottom_xy,
            top_displacement_m=d_top,
            bottom_displacement_m=d_bot,
            top_force_N=f_top,
            bottom_reaction_N=r_bot,
            contact_mask=contact,
            Fx_N=float(best.Fx_pred) * w,
            Fy_N=float(best.Fy_pred) * w,
            Mz_shoe_Nm=float(S[SCALAR_MZ]) * w,
            Mz_fixed_Nm=Mz_fixed,
            toe_moment_Nm=float(S[SCALAR_TOE]) * w,
            toe_spring_moment_Nm=float(toe.toe_spring_moment_Nm),
            force_residual_N=float(best.force_residual) * w,
            moment_residual_Nm=float(best.moment_residual) * w,
            toe_root_count=int(toe.toe_root_count),
            toe_stable=bool(toe.toe_equilibrium_stable),
            low_load=bool(low_load),
            diagnostics={
                "search_method": best.search_method,
                "max_free_penetration_m": float(best.max_free_penetration),
                "min_contact_reaction_N_per_m": float(best.min_contact_reaction),
                "toe_equilibrium_residual_Nm": float(toe.toe_equilibrium_residual_Nm),
                "roots_theta_deg": list(toe.roots_theta_deg),
                "condition_number": float(best.cond),
            },
        )

    def _invalid_state(self, phi_deg: float, low_load: bool, status: str) -> SoleState:
        nt, nb = self.n_top, self.n_bottom
        nan = float("nan")
        return SoleState(
            valid=False, admissible=False, status=status, theta_deg=nan, phi_deg=float(phi_deg),
            chord_rotation_rad=nan, contact_interval=None, contact_start_x_m=nan, contact_end_x_m=nan,
            top_xy_m=_nan2(nt), bottom_xy_m=_nan2(nb), top_displacement_m=_nan2(nt),
            bottom_displacement_m=_nan2(nb), top_force_N=_nan2(nt), bottom_reaction_N=_nan2(nb),
            contact_mask=np.zeros(nb, dtype=bool), Fx_N=nan, Fy_N=nan, Mz_shoe_Nm=nan, Mz_fixed_Nm=nan,
            toe_moment_Nm=nan, toe_spring_moment_Nm=nan, force_residual_N=nan, moment_residual_Nm=nan,
            toe_root_count=0, toe_stable=False, low_load=bool(low_load),
        )
