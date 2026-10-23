"""Stateful Python interface for driving the sole model from an external codebase.

Each :meth:`SoleModel.step` takes the fixed-frame load on the shoe top and the
rearfoot pitch, solves the continuous unilateral contact together with the
exact passive toe-spring equilibrium (the toe angle and the contact set are
outputs), and returns nodal positions, displacements, forces and moments. No
FEM solve happens at runtime; only the shared boundary operators are used.

Conventions (all inputs and outputs are per shoe, SI):

- Fixed frame: x forward (heel -> toe), y up, ground at y = 0.
- ``Fx_N``, ``Fy_N``: force applied **by the foot on the shoe top**
  (``F_top = -GRF``), so normal stance loading has ``Fy_N < 0``.
- ``phi_deg``: fixed-frame heel -> MTP (rearfoot) angle, counterclockwise
  positive, in degrees.
- Ground contact is normal unilateral (positive gap = open, positive normal
  reaction = compressive) with proportional shear: every active node carries
  the tangential force ``(Fx_N / Fy_N)`` times its normal reaction.
- The model is 2D plane strain per metre width; loads are divided by
  ``shoe_width_m`` on input and forces / moments multiplied by it on output.
- Every converged step also returns auxiliary 2.5D out-of-plane tangent
  stiffnesses (``SoleState.out_of_plane_stiffness``: lateral k_z in N/m, roll and
  yaw in N·m/rad; see docs/out_of_plane_stiffness.md). They are computed from
  the converged state and never feed back into it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from compliance_fem.contact.contact_model import ContactModel, load_contact_model
from compliance_fem.contact.continuous_contact import ContactTolerances, fixed_frame_positions
from compliance_fem.contact.continuous_toe import WarmStart, solve_toe_contact
from compliance_fem.contact.out_of_plane import (
    OutOfPlaneSettings,
    OutOfPlaneStiffness,
    compute_out_of_plane_stiffness,
)
from compliance_fem.contact.toe_spring import ToeSpringConfig


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
    # First and last active bottom node (None without contact); see contact_patches
    # for disconnected contact.
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
    # coordinates, about the heel-bottom reference point (gait convention).
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
    contact_patches: tuple[tuple[int, int], ...] = ()
    bottom_gap_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centre_of_pressure_x_m: float = float("nan")
    # Auxiliary 2.5D out-of-plane tangent stiffnesses (None when disabled or invalid step).
    out_of_plane_stiffness: OutOfPlaneStiffness | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly copy (arrays become nested lists)."""
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
        return out


class SoleModel:
    """Load a contact model once and evaluate it step by step.

    The previous toe angle and active contact set warm-start the next step, so
    consecutive steps stay on a continuous branch; call :meth:`reset` between
    independent sequences.
    """

    def __init__(
        self,
        model: ContactModel | str | Path,
        *,
        shoe_width_m: float = 0.10,
        toe_config: ToeSpringConfig | None = None,
        tolerances: ContactTolerances | None = None,
        out_of_plane: OutOfPlaneSettings | None = None,
    ) -> None:
        self.model = model if isinstance(model, ContactModel) else load_contact_model(model)
        width = float(shoe_width_m)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"shoe_width_m must be positive, got {shoe_width_m!r}.")
        self.shoe_width_m = width
        self.toe_config = toe_config or ToeSpringConfig()
        self.tolerances = tolerances
        # Constant 2.5D strip width defaults to the sagittal shoe width (same physical width).
        self.out_of_plane = out_of_plane or OutOfPlaneSettings(shoe_width_m=width)
        self.setup = None
        op = self.model.contact_operator
        self.x_top, self.y_top = op.x_top, op.y_top
        self.x_bottom, self.y_bottom = op.x_bottom, op.y_bottom
        self.reset()

    @classmethod
    def from_config(
        cls,
        config: str | Path,
        *,
        model: ContactModel | str | Path | None = None,
        build_if_missing: bool = False,
        require_matching_lookup: bool = True,
        tolerances: ContactTolerances | None = None,
        out_of_plane: OutOfPlaneSettings | None = None,
    ) -> SoleModel:
        """Build from a measured-sole JSON config.

        Shoe width and toe spring come from the config; the contact model is
        read from ``lookup.output_dir`` unless ``model`` is given. With
        ``build_if_missing`` a missing or stale file is (re)generated. With
        ``require_matching_lookup`` a model whose recorded build fingerprint
        differs from the config raises; models without a fingerprint only warn.
        """
        import warnings

        from compliance_fem.contact.model_setup import (
            ConfigFileError,
            ensure_lookup_exists,
            load_model_setup,
            lookup_matches_setup,
        )

        setup = load_model_setup(config)
        if model is None:
            if setup.output_dir is None:
                raise ConfigFileError(f"{config}: set lookup.output_dir or pass model=... to SoleModel.from_config.")
            model = ensure_lookup_exists(setup) if build_if_missing else setup.output_dir
        cm = model if isinstance(model, ContactModel) else load_contact_model(model)
        match = lookup_matches_setup(cm, setup)
        if match is False and require_matching_lookup:
            raise ConfigFileError(
                f"The contact model was built from different parameters than {config}. Regenerate it with "
                f"`compliance-fem-contact-lookup {config} --force` "
                "or pass require_matching_lookup=False."
            )
        if match is None:
            warnings.warn(
                "The contact model records no measured-setup fingerprint; cannot confirm it matches the config.",
                UserWarning,
                stacklevel=2,
            )
        out = cls(cm, shoe_width_m=setup.runtime.shoe_width_m, toe_config=setup.toe_spring, tolerances=tolerances,
                  out_of_plane=out_of_plane)
        out.setup = setup
        return out

    @property
    def n_top(self) -> int:
        return int(self.x_top.size)

    @property
    def n_bottom(self) -> int:
        return int(self.x_bottom.size)

    def reset(self) -> None:
        self._warm: WarmStart | None = None

    def step(self, Fx_N: float, Fy_N: float, phi_deg: float, *, Mz_Nm: float = float("nan")) -> SoleState:
        """Solve one load state. ``Mz_Nm`` (optional, measured top moment about
        the heel-bottom reference point) is not prescribed; it only sets
        ``moment_residual_Nm``."""
        w = self.shoe_width_m
        F_model = np.array([float(Fx_N), float(Fy_N)]) / w
        low_load = float(np.hypot(Fx_N, Fy_N)) < float(self.toe_config.toe_low_force_threshold_N)
        op = self.model.contact_operator
        t_step = time.perf_counter()
        fr = solve_toe_contact(
            op, F_model, float(np.deg2rad(phi_deg)), self.toe_config, w, warm=self._warm,
            low_load=low_load, tolerances=self.tolerances,
            cold_scan_points=int(self.toe_config.toe_root_scan_points),
        )
        t_sagittal = time.perf_counter() - t_step
        sol = fr.contact
        d = sol.diagnostics
        if not np.isfinite(fr.theta_rad):
            return self._invalid_state(phi_deg, low_load, fr.status)
        self._warm = fr.warm_start()
        nt, nb = self.n_top, self.n_bottom
        varphi = float(sol.varphi)
        c, s = np.cos(varphi), np.sin(varphi)
        Q = np.array([[c, -s], [s, c]])
        d_top = np.vstack([sol.d_t[:nt], sol.d_t[nt:]])
        d_bot = np.vstack([sol.d_b[:nb], sol.d_b[nb:]])
        top_xy = np.vstack(fixed_frame_positions(op, sol, self.x_top, self.y_top, d_top[0], d_top[1]))
        bottom_xy = np.vstack(fixed_frame_positions(op, sol, self.x_bottom, self.y_bottom, d_bot[0], d_bot[1]))
        f_top = w * (Q @ np.vstack([sol.F_t[:nt], sol.F_t[nt:]]))
        r_bot = w * np.vstack([sol.tau, sol.lam])
        Mz_fixed = float(np.sum(top_xy[0] * f_top[1] - top_xy[1] * f_top[0]))
        first, last = int(d["first_active"]), int(d["last_active"])
        Mz_shoe = float(d["Mz_shoe"]) * w
        # 2.5D tangent: read-only on the converged state, after contact and toe have converged.
        t_oop = time.perf_counter()
        oop = (
            compute_out_of_plane_stiffness(
                op, sol, self.out_of_plane, poisson_ratio=self.model.contact_foam_poisson_ratio
            )
            if fr.converged
            else None
        )
        t_oop = time.perf_counter() - t_oop
        return SoleState(
            valid=bool(fr.converged),
            admissible=bool(d["admissible"]),
            status=f"continuous:{fr.status}",
            theta_deg=float(fr.theta_deg),
            phi_deg=float(phi_deg),
            chord_rotation_rad=varphi,
            contact_interval=(first, last) if first >= 0 else None,
            contact_start_x_m=float(self.x_bottom[first]) if first >= 0 else float("nan"),
            contact_end_x_m=float(self.x_bottom[last]) if last >= 0 else float("nan"),
            top_xy_m=top_xy,
            bottom_xy_m=bottom_xy,
            top_displacement_m=d_top,
            bottom_displacement_m=d_bot,
            top_force_N=f_top,
            bottom_reaction_N=r_bot,
            contact_mask=sol.active.copy(),
            Fx_N=float(d["Fx_fixed"]) * w,
            Fy_N=float(d["Fy_fixed"]) * w,
            Mz_shoe_Nm=Mz_shoe,
            Mz_fixed_Nm=Mz_fixed,
            toe_moment_Nm=float(d["T_toe"]) * w,
            toe_spring_moment_Nm=float(fr.toe_spring_moment_Nm),
            force_residual_N=float(d["force_residual"]) * w,
            moment_residual_Nm=abs(Mz_shoe - float(Mz_Nm)) if np.isfinite(Mz_Nm) else float("nan"),
            toe_root_count=len(fr.roots_theta_deg),
            toe_stable=bool(fr.toe_equilibrium_stable),
            low_load=bool(low_load),
            contact_patches=tuple((p.start, p.end) for p in d["patches"]),
            bottom_gap_m=sol.gap.copy(),
            centre_of_pressure_x_m=float(d["cop_fixed_x"]),
            out_of_plane_stiffness=oop,
            diagnostics={
                "min_gap_m": float(d["min_gap"]),
                "min_normal_reaction_N_per_m": float(d["min_reaction"]),
                "max_complementarity": float(d["max_complementarity"]),
                "toe_equilibrium_residual_Nm": float(fr.toe_equilibrium_residual_Nm),
                "roots_theta_deg": list(fr.roots_theta_deg),
                "active_set_iterations": int(fr.active_set_iterations),
                "toe_iterations": int(fr.toe_iterations),
                "active_block_rcond": float(d["active_block_rcond"]),
                "mu_eff": float(d["mu_eff"]),
                "time_s": float(fr.timings["total_s"]),
                "sagittal_solve_time_s": float(t_sagittal),
                "out_of_plane_stiffness_time_s": float(t_oop),
                "total_step_time_s": float(time.perf_counter() - t_step),
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
