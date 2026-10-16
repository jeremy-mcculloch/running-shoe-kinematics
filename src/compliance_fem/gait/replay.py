"""Chronological gait stance replay: viscoelastic map + continuous contact + passive toe spring.

Each sample prescribes the fixed-frame top load ``(Fx*, Fy*)`` (foot on shoe,
``F_top = -GRF``) and the measured rearfoot pitch ``phi``. The contact set is
not chosen from a table: one warm-started complementarity solve per toe-root
evaluation (``contact.continuous_toe``) returns the active bottom nodes, the
reactions, and the exact passive toe angle. The previous sample's toe angle
and active set warm-start the next one, so the replay follows one continuous
branch. The measured COP / Mz is not prescribed; it is reported as a residual.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem import __version__
from compliance_fem.contact.contact_model import ContactModel, contact_label
from compliance_fem.contact.contact_operator import CONTACT_FORMULATION, TANGENTIAL_CONTACT_LAW
from compliance_fem.contact.continuous_contact import ContactTolerances
from compliance_fem.contact.continuous_toe import ToeFrameSolution, WarmStart, solve_toe_contact
from compliance_fem.contact.out_of_plane import (
    OUT_OF_PLANE_UNITS,
    OutOfPlaneSettings,
    OutOfPlaneStiffness,
    compute_out_of_plane_stiffness,
)
from compliance_fem.contact.toe_spring import (
    TOE_EQUATION,
    TOE_GENERALIZED_FORCE_DEFINITION,
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    VISCO_REJECTION_MESSAGE,
    ToeSpringConfig,
)
from compliance_fem.gait.cop_frames import (
    cop_lab_to_foot,
    cop_validity_mask,
    force_induced_moment_about_point,
    interpolate_marker_xy,
    model_cop_from_wrench,
)
from compliance_fem.gait.phi_markers import estimate_phi, interpolate_angle_to_grid
from compliance_fem.gait.sagittal import SagittalWrench, lab_grf_to_model_top_wrench
from compliance_fem.gait.stance import StanceInterval, detect_stances, pad_stance_window
from compliance_fem.gait.wang_io import extract_plate_wrench, read_opensim_mot, read_opensim_trc
from compliance_fem.viscoelasticity import create_material


def resolve_toe_config(
    toe_model: str | None = None,
    toe_config: ToeSpringConfig | None = None,
) -> ToeSpringConfig:
    """Resolve the toe model; the default is ``passive_spring``."""
    from dataclasses import replace

    cfg = toe_config or ToeSpringConfig()
    if toe_model is not None and toe_model != cfg.toe_model:
        cfg = replace(cfg, toe_model=toe_model)
    return cfg


# Per-frame scalar contact diagnostics copied from ``ContactSolution.diagnostics``.
CONTACT_SCALAR_FIELDS = (
    "min_gap",
    "min_inactive_gap",
    "max_active_gap",
    "min_reaction",
    "max_complementarity",
    "max_penetration",
    "max_tension",
    "n_active",
    "n_patches",
    "first_active",
    "last_active",
    "mu_eff",
    "cop_fixed_x",
    "cop_heel_relative_x",
    "grf_moment_fixed",
    "equilibrium_force_residual",
    "equilibrium_moment_residual",
    "ground_force_residual",
    "active_set_iterations",
    "active_block_rcond",
)


@dataclass
class GaitReplayResult:
    times: NDArray[np.float64]
    stance_percent: NDArray[np.float64]
    measured_Fx_ve: NDArray[np.float64]
    measured_Fy_ve: NDArray[np.float64]
    measured_Mz_ve: NDArray[np.float64]
    elastic_Fx: NDArray[np.float64]
    elastic_Fy: NDArray[np.float64]
    elastic_Mz: NDArray[np.float64]
    phi: NDArray[np.float64]
    varphi: NDArray[np.float64]
    alpha: NDArray[np.float64]
    theta_deg: NDArray[np.float64]
    contact_type: list[str]
    frame_valid: NDArray[np.bool_]
    predicted_Fx: NDArray[np.float64]
    predicted_Fy: NDArray[np.float64]
    predicted_Mz: NDArray[np.float64]
    force_residual: NDArray[np.float64]
    moment_residual: NDArray[np.float64]
    cop_residual: NDArray[np.float64]
    status: list[str]
    # Nodal contact state per frame (n_frames, N_b): fixed-frame gap, compressive
    # normal reaction, tangential reaction and active mask.
    gap: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    normal_reaction: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    tangential_reaction: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    active: NDArray[np.bool_] = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    contact_scalars: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # Active span (first / last active node) and its derived display-only edge estimates.
    contact_start_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contact_end_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contact_start_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_end_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_edge_estimate_start_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_edge_estimate_end_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    patches: list[list[tuple[int, int]]] = field(default_factory=list)
    timings: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    # Frame-aware COP / marker diagnostics (SI metres, rad)
    cop_lab_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cop_lab_y: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cop_foot_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cop_foot_y: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cop_model_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    heel_lab_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    heel_lab_y: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    fore_lab_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    fore_lab_y: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    force_lab_fx: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    force_lab_fy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    measured_moment_about_model_anchor: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    predicted_moment_about_model_anchor: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    predicted_toe_moment: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cop_valid: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    cop_out_of_shoe: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    model_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Passive toe spring.
    toe_model: str = TOE_MODEL_PASSIVE_SPRING
    toe_stiffness_Nm_per_rad: float = float("nan")
    toe_neutral_angle_rad: float = float("nan")
    toe_damping_Nms_per_rad: float = float("nan")
    Q_alpha_shoe_on_foot: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    Q_theta_shoe_on_foot: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_spring_moment_Nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_spring_generalized_force_alpha: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_equilibrium_residual_Nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_equilibrium_relative_residual: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_equilibrium_tangent: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_equilibrium_stable: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    toe_root_count: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    toe_iterations: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    toe_spring_energy_J: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_solve_status: list[str] = field(default_factory=list)
    toe_low_load: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    toe_roots_deg: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    # Auxiliary 2.5D out-of-plane tangent stiffnesses per frame (SI; NaN for invalid frames).
    lateral_translation_stiffness: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # N/m
    roll_stiffness: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # N*m/rad
    yaw_stiffness: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # N*m/rad
    oop_stiffness_status: list[str] = field(default_factory=list)
    oop_active_contact_node_count: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    oop_yaw_reference_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # m, fixed frame
    oop_condition_number: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    oop_near_active_set_change: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Solved frames (not serialized); feed the GUI renderers.
    frames: list[ToeFrameSolution] = field(default_factory=list, repr=False)

    @property
    def theta_rad(self) -> NDArray[np.float64]:
        return np.deg2rad(np.asarray(self.theta_deg, dtype=np.float64))

    def contact_column_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contact_start_index": self.contact_start_index,
            "contact_end_index": self.contact_end_index,
            "contact_start_x": self.contact_start_x,
            "contact_end_x": self.contact_end_x,
            "contact_edge_estimate_start_x": self.contact_edge_estimate_start_x,
            "contact_edge_estimate_end_x": self.contact_edge_estimate_end_x,
            "contact_patches": np.asarray([";".join(f"{a}-{b}" for a, b in p) for p in self.patches], dtype=object),
        }
        for name, arr in self.contact_scalars.items():
            out[name] = arr
        for name, arr in self.timings.items():
            out[f"time_{name}"] = arr
        return out

    def out_of_plane_column_dict(self) -> dict[str, Any]:
        """2.5D stiffness columns; units are part of the column names."""
        n = int(self.times.size)

        def col(arr, fill=np.nan, dtype: type = np.float64) -> np.ndarray:
            a = np.asarray(arr)
            return a if a.size == n else np.full(n, fill, dtype=dtype)

        return {
            "gait_cycle_percent": self.stance_percent,
            "lateral_translation_stiffness_N_per_m": col(self.lateral_translation_stiffness),
            "roll_stiffness_Nm_per_rad": col(self.roll_stiffness),
            "yaw_stiffness_Nm_per_rad": col(self.yaw_stiffness),
            "oop_active_contact_node_count": col(self.oop_active_contact_node_count, 0, np.int64),
            "oop_yaw_reference_x_m": col(self.oop_yaw_reference_x),
            "oop_condition_number": col(self.oop_condition_number),
            "oop_near_active_set_change": col(self.oop_near_active_set_change, 0, np.int8).astype(np.int8),
            "oop_stiffness_status": np.asarray(
                self.oop_stiffness_status if len(self.oop_stiffness_status) == n else ["not_computed"] * n,
                dtype=object,
            ),
        }

    def toe_column_dict(self) -> dict[str, Any]:
        n = int(self.times.size)
        return {
            "toe_model": np.asarray([self.toe_model] * n, dtype=object),
            "toe_stiffness_Nm_per_rad": np.full(n, float(self.toe_stiffness_Nm_per_rad)),
            "toe_neutral_angle_rad": np.full(n, float(self.toe_neutral_angle_rad)),
            "toe_damping_Nms_per_rad": np.full(n, float(self.toe_damping_Nms_per_rad)),
            "theta_rad": self.theta_rad,
            "theta_deg": np.asarray(self.theta_deg, dtype=np.float64),
            "toe_alpha": np.asarray(self.alpha, dtype=np.float64),
            "Q_alpha_shoe_on_foot": self.Q_alpha_shoe_on_foot,
            "Q_theta_shoe_on_foot": self.Q_theta_shoe_on_foot,
            "toe_spring_moment_Nm": self.toe_spring_moment_Nm,
            "toe_spring_generalized_force_alpha": self.toe_spring_generalized_force_alpha,
            "toe_equilibrium_residual_Nm": self.toe_equilibrium_residual_Nm,
            "toe_equilibrium_relative_residual": self.toe_equilibrium_relative_residual,
            "toe_equilibrium_tangent": self.toe_equilibrium_tangent,
            "toe_equilibrium_stable": self.toe_equilibrium_stable.astype(np.int8),
            "toe_root_count": self.toe_root_count,
            "toe_iterations": self.toe_iterations,
            "toe_spring_energy_J": self.toe_spring_energy_J,
            "toe_solve_status": np.asarray(self.toe_solve_status, dtype=object),
            "toe_low_load": self.toe_low_load.astype(np.int8),
        }

    def to_column_dict(self) -> dict[str, Any]:
        return {
            "time": self.times,
            "stance_percent": self.stance_percent,
            "measured_Fx_ve": self.measured_Fx_ve,
            "measured_Fy_ve": self.measured_Fy_ve,
            "measured_moment": self.measured_Mz_ve,
            "cop_lab_x": self.cop_lab_x,
            "cop_lab_y": self.cop_lab_y,
            "cop_foot_x": self.cop_foot_x,
            "cop_foot_y": self.cop_foot_y,
            "cop_model_x": self.cop_model_x,
            "heel_lab_x": self.heel_lab_x,
            "heel_lab_y": self.heel_lab_y,
            "fore_lab_x": self.fore_lab_x,
            "fore_lab_y": self.fore_lab_y,
            "force_lab_fx": self.force_lab_fx,
            "force_lab_fy": self.force_lab_fy,
            "elastic_equivalent_Fx": self.elastic_Fx,
            "elastic_equivalent_Fy": self.elastic_Fy,
            "elastic_equivalent_moment": self.elastic_Mz,
            "phi": self.phi,
            "varphi": self.varphi,
            "alpha": self.alpha,
            "theta": self.theta_deg,
            "contact_type": np.asarray(self.contact_type, dtype=object),
            "frame_valid": self.frame_valid.astype(np.int8),
            "predicted_Fx": self.predicted_Fx,
            "predicted_Fy": self.predicted_Fy,
            "predicted_moment": self.predicted_Mz,
            "force_residual": self.force_residual,
            "moment_residual": self.moment_residual,
            "COP_residual": self.cop_residual,
            "measured_moment_about_model_anchor": self.measured_moment_about_model_anchor,
            "predicted_moment_about_model_anchor": self.predicted_moment_about_model_anchor,
            "predicted_toe_moment": self.predicted_toe_moment,
            "status": np.asarray(self.status, dtype=object),
            "cop_valid": self.cop_valid.astype(np.int8),
            "cop_out_of_shoe": self.cop_out_of_shoe.astype(np.int8),
            **self.contact_column_dict(),
            **self.toe_column_dict(),
            **self.out_of_plane_column_dict(),
        }


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay_stance(
    contact_model: ContactModel,
    *,
    trc_path: str | Path | None = None,
    mot_path: str | Path | None = None,
    foot: str = "right",
    force_plate: str = "auto",
    stance_index: int = 0,
    vertical_force_threshold: float = 20.0,
    toe_model: str | None = None,
    toe_config: ToeSpringConfig | None = None,
    shoe_width_m: float = 0.10,
    visco_model: str = "elastic",
    visco_config: dict[str, Any] | None = None,
    tolerances: ContactTolerances | None = None,
    pre_pad_s: float = 0.05,
    post_pad_s: float = 0.05,
    heel_marker: str | None = None,
    fore_marker: str | None = None,
    position_units: str = "auto",
    moment_units: str = "auto",
    cop_fy_min: float = 20.0,
    synthetic: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    out_of_plane: OutOfPlaneSettings | None = None,
) -> GaitReplayResult:
    """Replay one stance (no per-frame FEM, no interval tables).

    Default ``toe_model='passive_spring'``: prescribe experimental Fx, Fy and
    marker φ; θ and the contact set follow from the exact passive toe-spring
    equilibrium coupled to the continuous contact solve. The measured COP/Mz is
    not prescribed and is reported as a validation residual. Requires
    ``visco_model='elastic'`` unless ``toe_model='passive_spring_elastic_equivalent'``
    is chosen explicitly. ``progress(done, total)``, if given, is called after
    each frame. ``out_of_plane`` (default: constant width ``shoe_width_m``,
    Poisson-derived lateral ratio) sets the auxiliary 2.5D stiffnesses computed
    from every converged frame; they never affect the sagittal solution.
    """
    toe_cfg = resolve_toe_config(toe_model, toe_config)
    model = toe_cfg.toe_model
    if model == TOE_MODEL_PASSIVE_SPRING and str(visco_model).lower() != "elastic":
        raise ValueError(VISCO_REJECTION_MESSAGE)

    warnings: list[str] = []
    if toe_cfg.toe_damping_Nms_per_rad != 0.0:
        warnings.append(
            f"toe_damping_Nms_per_rad={toe_cfg.toe_damping_Nms_per_rad:g} is recorded but does not "
            "enter the quasistatic toe equilibrium"
        )
    if model == TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT and str(visco_model).lower() != "elastic":
        warnings.append(
            "passive_spring_elastic_equivalent: toe spring solved against the elastic-equivalent "
            "wrench from the viscoelastic map; this is an approximation (no Q_alpha,VE)"
        )
    model_info: dict[str, Any] = {
        "software_version": __version__,
        "visco_model": visco_model,
        "viscoelastic_assumption": (
            "componentwise scalar kernel on generalized wrench; "
            "exact only for uniform relaxation spectrum"
        ),
        "force_sign": "F_top = -F_GRF_ground_on_body",
        "toe_config": toe_cfg.to_dict(),
        "toe_angle_method": "passive-toe-spring-exact",
        "toe_angle_method_version": "1",
        "shoe_width_m": float(shoe_width_m),
        "force_width_scaling": (
            "Experimental Fx,Fy,Mz (N, N·m) are divided by shoe_width_m to match "
            "plane-strain unit-thickness model units (N/m, N·m/m)."
        ),
        "laboratory_frame_definition": "+x anterior, +y superior, +z right (OpenSim)",
        "foot_frame_definition": (
            "heel-attached; +x along heel→forefoot chord at angle φ; "
            "p_foot = Q(φ)^T (p_lab - r_heel_lab)"
        ),
        "cop_reference_frame": {
            "cop_lab_*": "global laboratory/OpenSim ground",
            "cop_foot_*": "heel-attached foot frame",
            "cop_model_x": "model frame about the heel-bottom reference (0,0); x≈Mz/Fy",
        },
    }

    # Diagnostics filled after stance window is known.
    cop_lab_x = cop_lab_y = np.zeros(0)
    heel_lab_x = heel_lab_y = np.zeros(0)
    fore_lab_x = fore_lab_y = np.zeros(0)
    force_lab_fx = force_lab_fy = np.zeros(0)
    fy_lab_vertical = np.zeros(0)
    free_tz_lab = np.zeros(0)
    heel_label = fore_label = None

    if synthetic is not None:
        times = np.asarray(synthetic["times"], dtype=np.float64)
        wrench = SagittalWrench(
            Fx=np.asarray(synthetic["Fx"], dtype=np.float64),
            Fy=np.asarray(synthetic["Fy"], dtype=np.float64),
            Mz=np.asarray(synthetic["Mz"], dtype=np.float64),
            cop_x=np.asarray(synthetic.get("cop_x", np.zeros_like(times)), dtype=np.float64),
            cop_y=np.asarray(synthetic.get("cop_y", np.zeros_like(times)), dtype=np.float64),
            Fz_omitted=np.zeros_like(times),
            times=times,
        )
        phi = np.asarray(synthetic.get("phi", np.zeros_like(times)), dtype=np.float64)
        stance = StanceInterval(
            0, 0, times.size - 1, float(times[0]), float(times[-1]), float(np.max(-wrench.Fy))
        )
        i0, i1 = 0, times.size - 1
        model_info["source"] = "synthetic"
        model_info["source_format"] = "synthetic"
        visco_init = "relaxed-at-window-start"
        cop_lab_x = np.asarray(synthetic.get("cop_x", np.zeros_like(times)), dtype=np.float64)
        cop_lab_y = np.asarray(synthetic.get("cop_y", np.zeros_like(times)), dtype=np.float64)
        heel_lab_x = np.asarray(synthetic.get("heel_x", np.zeros_like(times)), dtype=np.float64)
        heel_lab_y = np.asarray(synthetic.get("heel_y", np.zeros_like(times)), dtype=np.float64)
        fore_lab_x = np.asarray(
            synthetic.get("fore_x", heel_lab_x + float(contact_model.L)),
            dtype=np.float64,
        )
        fore_lab_y = np.asarray(synthetic.get("fore_y", heel_lab_y), dtype=np.float64)
        force_lab_fx = -np.asarray(synthetic["Fx"], dtype=np.float64)  # undo top sign for lab
        force_lab_fy = -np.asarray(synthetic["Fy"], dtype=np.float64)
        fy_lab_vertical = force_lab_fy.copy()
        free_tz_lab = np.zeros_like(times)
    else:
        if trc_path is None or mot_path is None:
            raise ValueError("trc_path and mot_path are required unless synthetic=...")
        trc_path = Path(trc_path)
        mot_path = Path(mot_path)
        trc = read_opensim_trc(trc_path, position_units=position_units)
        mot = read_opensim_mot(
            mot_path, position_units=position_units, moment_units=moment_units
        )
        plate = extract_plate_wrench(mot, force_plate)
        fy_lab = np.asarray(plate["fy"], dtype=np.float64)
        if float(np.nanmax(fy_lab)) < float(np.nanmax(-fy_lab)):
            fy_for_stance = -fy_lab
        else:
            fy_for_stance = fy_lab
        stances = detect_stances(
            plate["times"], fy_for_stance, threshold=vertical_force_threshold
        )
        if not stances:
            raise RuntimeError("no stance intervals detected")
        if stance_index < 0 or stance_index >= len(stances):
            raise IndexError(f"stance_index {stance_index} out of range 0..{len(stances)-1}")
        stance = stances[stance_index]
        i0, i1 = pad_stance_window(plate["times"], stance, pre_s=pre_pad_s, post_s=post_pad_s)
        sl = slice(i0, i1 + 1)
        times = plate["times"][sl]
        # Lab GRF → model-top Fx/Fy; Mz is recomputed about the heel below.
        wrench, wwarn = lab_grf_to_model_top_wrench(
            times=times,
            fx_lab=plate["fx"][sl],
            fy_lab=plate["fy"][sl],
            fz_lab=plate["fz"][sl],
            cop_x_lab=plate["px"][sl],
            cop_y_lab=plate["py"][sl],
            mz_lab=plate["mz"][sl],
        )
        warnings.extend(wwarn)
        phi_series = estimate_phi(
            trc["times"],
            trc["markers"],
            foot=foot,
            heel_override=heel_marker,
            fore_override=fore_marker,
        )
        phi = interpolate_angle_to_grid(phi_series.times, phi_series.phi_rad, times)
        heel_label = phi_series.heel_label
        fore_label = phi_series.fore_label
        from compliance_fem.gait.phi_markers import _fore_points

        heel_xy = interpolate_marker_xy(
            trc["times"], trc["markers"][heel_label], times
        )
        fore_pts = _fore_points(trc["markers"], fore_label)
        fore_xy = interpolate_marker_xy(trc["times"], fore_pts, times)
        heel_lab_x, heel_lab_y = heel_xy[:, 0], heel_xy[:, 1]
        fore_lab_x, fore_lab_y = fore_xy[:, 0], fore_xy[:, 1]
        cop_lab_x = np.asarray(plate["px"][sl], dtype=np.float64)
        cop_lab_y = np.asarray(plate["py"][sl], dtype=np.float64)
        force_lab_fx = np.asarray(plate["fx"][sl], dtype=np.float64)
        force_lab_fy = np.asarray(plate["fy"][sl], dtype=np.float64)
        fy_lab_vertical = force_lab_fy.copy()
        free_tz_lab = np.asarray(plate["mz"][sl], dtype=np.float64)
        model_info.update(
            {
                "source": "wang_trc_mot",
                "source_format": mot.get("source_format", "opensim_mot"),
                "trc": str(trc_path),
                "mot": str(mot_path),
                "trc_sha256": _file_sha256(trc_path),
                "mot_sha256": _file_sha256(mot_path),
                "trc_rate_hz": trc["rate_hz"],
                "mot_rate_hz": mot["rate_hz"],
                "force_plate": plate["plate_id"],
                "channel_map": plate["channel_map"],
                "heel_marker_labels": heel_label,
                "forefoot_marker_labels": fore_label,
                "stance_index": stance_index,
                "n_stances_detected": len(stances),
                "vertical_force_threshold": vertical_force_threshold,
                "source_position_units": mot.get("source_position_units"),
                "source_moment_units": mot.get("source_moment_units"),
                "internal_position_units": "m",
                "internal_moment_units": "N·m",
                "source_axis_mapping": mot.get("axis_mapping"),
                "trc_position_units": trc.get("source_position_units"),
                "unit_notes": list(mot.get("unit_notes", [])) + list(trc.get("unit_notes", [])),
                "free_torque_interpretation": mot.get("free_torque_interpretation"),
                "force_sign_conversion": "F_top = -F_GRF (lab sagittal components)",
                "moment_reference_point": {
                    "wrench_Mz_star": (
                        "top-of-shoe moment about lab heel: "
                        "-(r_COP/heel × F_lab + T_z)/w  (matches the model Mz about the heel-bottom reference)"
                    ),
                    "measured_moment_about_model_anchor": "same as wrench_Mz_star",
                    "predicted_moment_about_model_anchor": "model top Mz about the heel-bottom reference (0,0)",
                },
            }
        )
        if i0 == 0 or float(np.max(np.abs(wrench.Fy[: max(1, int(0.01 * wrench.Fy.size))]))) > 1.0:
            visco_init = "relaxed-at-window-start"
            warnings.append(
                "insufficient zero-load prehistory; visco states start relaxed at window start"
            )
        else:
            visco_init = "zero-load-prestance"
    model_info["visco_initialization"] = visco_init

    # Heel-relative COP (always). For TRC+MOT, replace lab-origin Mz with the
    # top-of-shoe moment about the heel so it matches the model Mz convention.
    # Synthetic trials already pass model-top Mz directly — keep it.
    foot_cop_early = cop_lab_to_foot(cop_lab_x, cop_lab_y, heel_lab_x, heel_lab_y, phi)
    if synthetic is not None:
        Mz_top = np.asarray(wrench.Mz, dtype=np.float64)
    else:
        M_ground_about_heel = force_induced_moment_about_point(
            force_lab_fx,
            force_lab_fy,
            cop_lab_x,
            cop_lab_y,
            origin_x=heel_lab_x,
            origin_y=heel_lab_y,
        ) + np.asarray(free_tz_lab, dtype=np.float64)
        # Action–reaction: top-of-shoe moment matching the model Mz convention.
        Mz_top = -M_ground_about_heel

    # The plane-strain model assumes unit out-of-plane thickness: convert total
    # experimental loads to per-metre line loads.
    width = float(shoe_width_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"shoe_width_m must be positive, got {shoe_width_m!r}")
    inv_w = 1.0 / width
    wrench = SagittalWrench(
        Fx=np.asarray(wrench.Fx, dtype=np.float64) * inv_w,
        Fy=np.asarray(wrench.Fy, dtype=np.float64) * inv_w,
        Mz=np.asarray(Mz_top, dtype=np.float64) * inv_w,
        cop_x=np.asarray(foot_cop_early.x, dtype=np.float64),
        cop_y=np.asarray(foot_cop_early.y, dtype=np.float64),
        Fz_omitted=np.asarray(wrench.Fz_omitted, dtype=np.float64) * inv_w,
        times=np.asarray(wrench.times, dtype=np.float64),
    )
    n = times.size
    # Viscoelastic map on (Fx, Fy) only; Mz is not prescribed and stays a diagnostic.
    cfg = dict(visco_config or {})
    cfg["model"] = visco_model
    cfg["n_components"] = 2
    material = create_material(cfg)
    material.reset()
    Fe = np.zeros((n, 3), dtype=np.float64)
    Fve = np.column_stack([wrench.Fx, wrench.Fy, wrench.Mz])
    for i in range(n):
        Fe[i, :2] = material.update(Fve[i, :2], float(times[i]))
        Fe[i, 2] = Fve[i, 2]

    op = contact_model.operator
    n_b = op.n_bottom
    toe_low_force_model = float(toe_cfg.toe_low_force_threshold_N) * inv_w
    model_info["contact_formulation"] = CONTACT_FORMULATION
    model_info["tangential_contact_law"] = TANGENTIAL_CONTACT_LAW
    model_info["toe_equation"] = TOE_EQUATION
    model_info["toe_generalized_force"] = TOE_GENERALIZED_FORCE_DEFINITION
    model_info["toe_generalized_force_width_scaling"] = (
        "Q_total (N*m) = shoe_width_m * Q_model (N*m per metre width)"
    )
    model_info["toe_rearfoot_kinematics"] = {
        "phi_is": "fixed-frame heel -> MTP (rearfoot) marker angle",
        "chord_rotation": "varphi(alpha) = phi - atan2(dy_mtp + alpha*phi1_mtp, L - toe_length)",
        "varphi_column": "chord-frame rotation at the solved toe angle",
        **op.rearfoot.to_dict(),
    }

    frames: list[ToeFrameSolution] = []
    oop_settings = out_of_plane or OutOfPlaneSettings(shoe_width_m=float(shoe_width_m))
    oop_list: list[OutOfPlaneStiffness | None] = []
    model_info["out_of_plane"] = oop_settings.metadata()
    warm: WarmStart | None = None
    phi_last = 0.0
    t_loop = time.perf_counter()
    for i in range(n):
        if np.isfinite(phi[i]):
            phi_last = float(phi[i])
        low_load = float(np.hypot(Fe[i, 0], Fe[i, 1])) < toe_low_force_model
        fr = solve_toe_contact(
            op, Fe[i, :2], phi_last, toe_cfg, width, warm=warm, low_load=low_load,
            tolerances=tolerances, cold_scan_points=int(toe_cfg.toe_root_scan_points),
        )
        frames.append(fr)
        oop_list.append(
            compute_out_of_plane_stiffness(op, fr.contact, oop_settings, poisson_ratio=contact_model.contact_foam_poisson_ratio)
            if fr.converged
            else None
        )
        warm = fr.warm_start()
        if progress is not None:
            progress(i + 1, n)
    runtime_s = time.perf_counter() - t_loop
    model_info["runtime_total_s"] = float(runtime_s)
    model_info["runtime_per_sample_s"] = float(runtime_s / max(n, 1))

    st0 = stance.t0
    st1 = max(stance.t1, st0 + 1e-12)
    diags = [fr.contact.diagnostics for fr in frames]

    def col(name: str) -> NDArray[np.float64]:
        return np.array([float(d[name]) for d in diags], dtype=np.float64)

    foot_cop = foot_cop_early
    predicted_Fy = col("Fy_fixed")
    predicted_M_anchor = col("Mz_shoe")
    predicted_toe = col("T_toe")
    cop_model_x = model_cop_from_wrench(
        predicted_M_anchor,
        predicted_Fy,
        fy_min=float(cop_fy_min) / max(float(shoe_width_m), 1e-12),
    )
    measured_M_anchor_model = np.asarray(Fe[:, 2], dtype=np.float64)
    model_info["moment_sign_convention"] = (
        "measured_moment_about_model_anchor = Fe_Mz = -(r_COP/heel × F_lab + T_z)/w "
        "(top-of-shoe convention, matches the model Mz about the heel-bottom reference). "
        "Mz is not prescribed; the moment/COP residual validates the spring-predicted θ "
        "against measured COP."
    )

    cop_valid, cop_oos = cop_validity_mask(
        fy_lab_vertical=fy_lab_vertical,
        cop_foot_x=foot_cop.x,
        shoe_length=float(contact_model.L),
        fy_min=float(cop_fy_min),
    )
    cop_res = np.full(times.size, np.nan, dtype=np.float64)
    ok = cop_valid & np.isfinite(cop_model_x)
    cop_res[ok] = np.abs(foot_cop.x[ok] - cop_model_x[ok])

    n_valid = int(np.count_nonzero(cop_valid))
    if n_valid:
        in_shoe = cop_valid & ~cop_oos
        model_info["cop_validation"] = {
            "n_valid_frames": n_valid,
            "fraction_in_shoe_interval": float(np.count_nonzero(in_shoe) / n_valid),
            "cop_foot_x_min": float(np.nanmin(foot_cop.x[cop_valid])),
            "cop_foot_x_max": float(np.nanmax(foot_cop.x[cop_valid])),
            "mean_abs_cop_residual_m": float(np.nanmean(cop_res[ok])) if np.any(ok) else None,
            "cop_fy_min_N": float(cop_fy_min),
        }
        mom_ok = cop_valid & np.isfinite(measured_M_anchor_model) & np.isfinite(predicted_M_anchor)
        if np.any(mom_ok):
            model_info["moment_validation"] = {
                "corr_measured_top_vs_predicted_Mz": float(
                    np.corrcoef(
                        measured_M_anchor_model[mom_ok], predicted_M_anchor[mom_ok]
                    )[0, 1]
                ),
                "mean_abs_moment_residual": float(
                    np.mean(np.abs(measured_M_anchor_model[mom_ok] - predicted_M_anchor[mom_ok]))
                ),
            }
    if int(np.count_nonzero(cop_oos)):
        warnings.append(
            f"{int(np.count_nonzero(cop_oos))} frames have heel-relative COP outside "
            f"[0, L]±tol (diagnostic only; values not clipped)"
        )

    model_info["toe_angle_note"] = (
        "θ from the exact passive toe-spring equilibrium with a converged contact solve per "
        "residual evaluation; Brent roots on [toe_angle_min_deg, toe_angle_max_deg], continued "
        "from the previous sample. toe_damping_Nms_per_rad is recorded only."
    )

    labels = [contact_label(int(d["first_active"]), int(d["last_active"]), n_b) for d in diags]
    x_b = op.x_bottom

    def _span_x(k: int) -> float:
        return float(x_b[k]) if k >= 0 else float("nan")

    first_patch = [d["patches"][0] if d["patches"] else None for d in diags]
    last_patch = [d["patches"][-1] if d["patches"] else None for d in diags]
    statuses = []
    for fr, d in zip(frames, diags):
        s = f"continuous:{fr.status}"
        if not fr.contact.converged:
            s += f"+{d['lcp_status']}"
        if not d["admissible"]:
            s += "+contact_tolerance_exceeded"
        if d["disconnected_contact"]:
            s += "+multiple_contact_patches"
        if abs(float(d["mu_eff"])) > (tolerances or ContactTolerances()).mu_eff_warning:
            s += "+large_shear_ratio"
        statuses.append(s)
    timing_keys = ("contact_solves_s", "toe_outer_s", "total_s")

    result = GaitReplayResult(
        times=times,
        stance_percent=100.0 * np.clip((times - st0) / (st1 - st0), 0.0, 1.0),
        measured_Fx_ve=Fve[:, 0],
        measured_Fy_ve=Fve[:, 1],
        measured_Mz_ve=Fve[:, 2],
        elastic_Fx=Fe[:, 0],
        elastic_Fy=Fe[:, 1],
        elastic_Mz=Fe[:, 2],
        phi=phi,
        varphi=np.array([fr.varphi for fr in frames], dtype=np.float64),
        alpha=np.array([fr.alpha for fr in frames], dtype=np.float64),
        theta_deg=np.array([fr.theta_deg for fr in frames], dtype=np.float64),
        contact_type=labels,
        frame_valid=np.array([bool(fr.converged) for fr in frames], dtype=bool),
        predicted_Fx=col("Fx_fixed"),
        predicted_Fy=predicted_Fy,
        predicted_Mz=predicted_M_anchor.copy(),
        force_residual=col("force_residual"),
        moment_residual=np.abs(predicted_M_anchor - measured_M_anchor_model),
        cop_residual=cop_res,
        status=statuses,
        gap=np.stack([fr.contact.gap for fr in frames]) if frames else np.zeros((0, n_b)),
        normal_reaction=np.stack([fr.contact.lam for fr in frames]) if frames else np.zeros((0, n_b)),
        tangential_reaction=np.stack([fr.contact.tau for fr in frames]) if frames else np.zeros((0, n_b)),
        active=np.stack([fr.contact.active for fr in frames]) if frames else np.zeros((0, n_b), dtype=bool),
        contact_scalars={name: col(name) for name in CONTACT_SCALAR_FIELDS},
        contact_start_index=np.array([int(d["first_active"]) for d in diags], dtype=np.int64),
        contact_end_index=np.array([int(d["last_active"]) for d in diags], dtype=np.int64),
        contact_start_x=np.array([_span_x(int(d["first_active"])) for d in diags]),
        contact_end_x=np.array([_span_x(int(d["last_active"])) for d in diags]),
        contact_edge_estimate_start_x=np.array([p.edge_estimate_start_x if p else np.nan for p in first_patch]),
        contact_edge_estimate_end_x=np.array([p.edge_estimate_end_x if p else np.nan for p in last_patch]),
        patches=[[(p.start, p.end) for p in d["patches"]] for d in diags],
        timings={k: np.array([fr.timings.get(k, np.nan) for fr in frames]) for k in timing_keys},
        cop_lab_x=np.asarray(cop_lab_x, dtype=np.float64),
        cop_lab_y=np.asarray(cop_lab_y, dtype=np.float64),
        cop_foot_x=foot_cop.x,
        cop_foot_y=foot_cop.y,
        cop_model_x=cop_model_x,
        heel_lab_x=np.asarray(heel_lab_x, dtype=np.float64),
        heel_lab_y=np.asarray(heel_lab_y, dtype=np.float64),
        fore_lab_x=np.asarray(fore_lab_x, dtype=np.float64),
        fore_lab_y=np.asarray(fore_lab_y, dtype=np.float64),
        force_lab_fx=np.asarray(force_lab_fx, dtype=np.float64),
        force_lab_fy=np.asarray(force_lab_fy, dtype=np.float64),
        measured_moment_about_model_anchor=measured_M_anchor_model,
        predicted_moment_about_model_anchor=predicted_M_anchor,
        predicted_toe_moment=predicted_toe,
        cop_valid=cop_valid,
        cop_out_of_shoe=cop_oos,
        model_info=model_info,
        warnings=warnings,
        toe_model=model,
        toe_stiffness_Nm_per_rad=float(toe_cfg.toe_stiffness_Nm_per_rad),
        toe_neutral_angle_rad=float(toe_cfg.toe_neutral_angle_rad),
        toe_damping_Nms_per_rad=float(toe_cfg.toe_damping_Nms_per_rad),
        frames=frames,
    )
    _attach_toe_outputs(result, frames)
    _attach_out_of_plane_outputs(result, oop_list)
    n_multi = int(sum(len(p) > 1 for p in result.patches))
    if n_multi:
        warnings.append(
            f"{n_multi} frames have more than one disconnected contact patch (valid geometry, "
            "mesh-scale oscillation, or a sign / conditioning problem; patches are kept separate)"
        )
    model_info["contact_summary"] = _contact_summary(result, contact_model)
    model_info["toe_summary"] = _toe_summary(result)
    return result


def _attach_toe_outputs(result: GaitReplayResult, frames: list[ToeFrameSolution]) -> None:
    """Copy per-frame toe diagnostics onto the result."""
    result.Q_alpha_shoe_on_foot = np.array([f.Q_alpha_shoe_on_foot for f in frames])
    result.Q_theta_shoe_on_foot = np.array([f.Q_theta_shoe_on_foot for f in frames])
    result.toe_spring_moment_Nm = np.array([f.toe_spring_moment_Nm for f in frames])
    result.toe_spring_generalized_force_alpha = np.array([f.toe_spring_generalized_force_alpha for f in frames])
    result.toe_equilibrium_residual_Nm = np.array([f.toe_equilibrium_residual_Nm for f in frames])
    result.toe_equilibrium_relative_residual = np.array([f.toe_equilibrium_relative_residual for f in frames])
    result.toe_equilibrium_tangent = np.array([f.toe_equilibrium_tangent for f in frames])
    result.toe_equilibrium_stable = np.array([f.toe_equilibrium_stable for f in frames], dtype=bool)
    result.toe_root_count = np.array([len(f.roots_theta_deg) for f in frames], dtype=np.int64)
    result.toe_iterations = np.array([f.toe_iterations for f in frames], dtype=np.int64)
    result.toe_spring_energy_J = np.array([f.toe_spring_energy_J for f in frames])
    result.toe_solve_status = [f.status for f in frames]
    result.toe_low_load = np.array([f.low_load for f in frames], dtype=bool)
    width = max((len(f.roots_theta_deg) for f in frames), default=0)
    roots = np.full((len(frames), width), np.nan)
    for i, f in enumerate(frames):
        roots[i, : len(f.roots_theta_deg)] = f.roots_theta_deg
    result.toe_roots_deg = roots


def _attach_out_of_plane_outputs(result: GaitReplayResult, oop: list[OutOfPlaneStiffness | None]) -> None:
    """Store the per-frame 2.5D stiffnesses produced alongside each solved frame (NaN when invalid)."""
    nan = float("nan")
    result.lateral_translation_stiffness = np.array([o.lateral_translation_stiffness if o else nan for o in oop])
    result.roll_stiffness = np.array([o.roll_stiffness if o else nan for o in oop])
    result.yaw_stiffness = np.array([o.yaw_stiffness if o else nan for o in oop])
    result.oop_stiffness_status = [o.status if o else "invalid_frame" for o in oop]
    result.oop_active_contact_node_count = np.array([o.active_contact_node_count if o else 0 for o in oop], dtype=np.int64)
    result.oop_yaw_reference_x = np.array([o.yaw_reference_x if o else nan for o in oop])
    result.oop_condition_number = np.array([o.condition_number if o else nan for o in oop])
    result.oop_near_active_set_change = np.array([bool(o.near_active_set_change) if o else False for o in oop])
    result.timings["out_of_plane_s"] = np.array([o.compute_time_s if o else nan for o in oop])
    result.model_info["out_of_plane"]["units"] = dict(OUT_OF_PLANE_UNITS)


def load_replay_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read a saved ``gait_replay.npz`` (directory or file).

    Files written before the 2.5D stiffnesses existed stay usable: the missing
    stiffness columns are returned as NaN with status ``"not_available"``; they
    are never reconstructed.
    """
    p = Path(path)
    npz = p / "gait_replay.npz" if p.is_dir() else p
    with np.load(npz, allow_pickle=True) as data:
        out = {k: np.asarray(data[k]) for k in data.files}
    n = int(np.asarray(out["time"]).size)
    for key in ("lateral_translation_stiffness_N_per_m", "roll_stiffness_Nm_per_rad", "yaw_stiffness_Nm_per_rad",
                "oop_yaw_reference_x_m", "oop_condition_number"):
        out.setdefault(key, np.full(n, np.nan))
    out.setdefault("oop_stiffness_status", np.asarray(["not_available"] * n, dtype=object))
    out.setdefault("oop_active_contact_node_count", np.zeros(n, dtype=np.int64))
    return out


def _finite_max(a) -> float | None:
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else None


def _contact_summary(result: GaitReplayResult, contact_model: ContactModel) -> dict[str, Any]:
    cs = result.contact_scalars
    labels = ("none", "heel", "interior", "toe", "full")
    tot = result.timings.get("total_s", np.zeros(0))
    return {
        "contact_set_model": "continuous_node_to_ground_complementarity",
        "n_bottom_nodes": int(contact_model.n_bottom_nodes),
        "label_counts": {k: int(sum(c == k for c in result.contact_type)) for k in labels},
        "n_invalid_frames": int(np.count_nonzero(~result.frame_valid)),
        "n_multiple_patch_frames": int(sum(len(p) > 1 for p in result.patches)),
        "max_penetration_m": _finite_max(cs.get("max_penetration", [])),
        "max_tension": _finite_max(cs.get("max_tension", [])),
        "max_complementarity": _finite_max(cs.get("max_complementarity", [])),
        "max_force_residual": _finite_max(result.force_residual),
        "max_active_set_iterations": _finite_max(cs.get("active_set_iterations", [])),
        "mean_time_per_sample_s": float(np.mean(tot)) if np.size(tot) else None,
        "max_time_per_sample_s": float(np.max(tot)) if np.size(tot) else None,
    }


def _toe_summary(result: GaitReplayResult) -> dict[str, Any]:
    th = np.asarray(result.theta_deg, dtype=float)
    res = np.abs(np.asarray(result.toe_equilibrium_residual_Nm, dtype=float))
    rel = np.asarray(result.toe_equilibrium_relative_residual, dtype=float)
    status = result.toe_solve_status
    return {
        "theta_deg_min": float(np.nanmin(th)) if np.any(np.isfinite(th)) else None,
        "theta_deg_max": float(np.nanmax(th)) if np.any(np.isfinite(th)) else None,
        "max_abs_residual_Nm": _finite_max(res),
        "max_relative_residual": _finite_max(rel),
        "n_frames": int(th.size),
        "n_multiple_root_frames": int(np.count_nonzero(result.toe_root_count > 1)),
        "n_no_root_frames": int(sum("no_root" in s for s in status)),
        "n_unstable_frames": int(np.count_nonzero(~result.toe_equilibrium_stable)),
        "n_low_load_frames": int(np.count_nonzero(result.toe_low_load)),
        "mean_toe_iterations": float(np.mean(result.toe_iterations)) if result.toe_iterations.size else None,
        "max_spring_energy_J": _finite_max(result.toe_spring_energy_J),
    }


def save_replay_result(result: GaitReplayResult, output_dir: str | Path) -> dict[str, Path]:
    """Write CSV (per-frame scalars) + NPZ (scalars and nodal contact arrays) + model-info JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = result.to_column_dict()
    numeric_keys = [k for k, v in cols.items() if isinstance(v, np.ndarray) and v.dtype != object]
    text_keys = [k for k, v in cols.items() if isinstance(v, np.ndarray) and v.dtype == object]
    header = ",".join(numeric_keys + text_keys)
    rows = np.column_stack([np.asarray(cols[k], dtype=float) for k in numeric_keys])
    csv_path = output_dir / "gait_replay.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(result.times.size):
            nums = ",".join(f"{rows[i, j]:.16g}" for j in range(rows.shape[1]))
            texts = ",".join(str(cols[k][i]).replace(",", ";") for k in text_keys)
            f.write(f"{nums},{texts}\n")
    npz_path = output_dir / "gait_replay.npz"
    np.savez_compressed(
        npz_path,
        toe_roots_deg=result.toe_roots_deg,
        gap=result.gap,
        normal_reaction=result.normal_reaction,
        tangential_reaction=result.tangential_reaction,
        active=result.active,
        **{k: cols[k] for k in cols},
    )
    meta_path = output_dir / "model_info.json"
    meta_path.write_text(
        json.dumps({"model_info": result.model_info, "warnings": result.warnings}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return {"csv": csv_path, "npz": npz_path, "model_info": meta_path}
