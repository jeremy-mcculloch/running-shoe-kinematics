"""Chronological gait stance replay through viscoelasticity + contact lookup."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from compliance_fem import __version__
from compliance_fem.contact_lookup import (
    SCALAR_MZ,
    SCALAR_TOE,
    ContactLookupResult,
    lookup_rearfoot_geometry,
)
from compliance_fem.contact_topology import ContactType, interval_distance
from compliance_fem.corotation import basis_coefficients, contract_basis
from compliance_fem.force_control import (
    SEARCH_FALLBACK,
    SEARCH_GLOBAL,
    SelectionConfig,
    Tolerances,
    evaluate_candidates,
    refine_top_k,
    select_contact_candidate,
)
from compliance_fem.gait.phi_markers import estimate_phi, interpolate_angle_to_grid
from compliance_fem.gait.sagittal import SagittalWrench, lab_grf_to_model_top_wrench
from compliance_fem.gait.stance import StanceInterval, detect_stances, pad_stance_window
from compliance_fem.gait.temporal import TemporalWeights, viterbi_select
from compliance_fem.gait.toe_angle import compressive_fy_load, prescribed_toe_angle_deg
from compliance_fem.gait.cop_frames import (
    TOE_ANGLE_METHOD,
    TOE_ANGLE_METHOD_VERSION,
    cop_lab_to_foot,
    cop_validity_mask,
    force_induced_moment_about_point,
    interpolate_marker_xy,
    model_cop_from_wrench,
)
from compliance_fem.gait.wang_io import (
    extract_plate_wrench,
    read_opensim_mot,
    read_opensim_trc,
)
from compliance_fem.gait.wrench_control import (
    CANDIDATE_DIAGNOSTIC_FIELDS,
    WrenchCandidate,
    pick_instant_best,
    solve_wrench_control,
)
from compliance_fem.gait.passive_toe import (
    pick_instant_best_passive,
    solve_passive_toe_candidates,
)
from compliance_fem.toe_spring import (
    TOE_EQUATION,
    TOE_GENERALIZED_FORCE_DEFINITION,
    TOE_MODEL_FIT_COP_LEGACY,
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT,
    TOE_MODEL_PRESCRIBED_LEGACY,
    VISCO_REJECTION_MESSAGE,
    ToeSpringConfig,
)
from compliance_fem.viscoelasticity import create_material

ThetaMode = Literal["fit-cop", "force-phi"]
_THETA_MODE_TO_TOE_MODEL = {
    "fit-cop": TOE_MODEL_FIT_COP_LEGACY,
    "force-phi": TOE_MODEL_PRESCRIBED_LEGACY,
}


def resolve_toe_config(
    toe_model: str | None = None,
    toe_config: ToeSpringConfig | None = None,
    theta_mode: str | None = None,
) -> ToeSpringConfig:
    """Resolve the toe model; the default is ``passive_spring``.

    ``theta_mode`` is the deprecated selector for the two legacy models
    (``fit-cop`` → ``fit_cop_legacy``, ``force-phi`` → ``prescribed_legacy``).
    """
    from dataclasses import replace

    cfg = toe_config or ToeSpringConfig()
    requested = toe_model
    if theta_mode is not None:
        if theta_mode not in _THETA_MODE_TO_TOE_MODEL:
            raise ValueError(f"theta_mode must be one of {tuple(_THETA_MODE_TO_TOE_MODEL)}, got {theta_mode!r}.")
        mapped = _THETA_MODE_TO_TOE_MODEL[theta_mode]
        if requested is not None and requested != mapped:
            raise ValueError(
                f"theta_mode={theta_mode!r} selects toe_model={mapped!r}, which conflicts with "
                f"toe_model={requested!r}."
            )
        requested = mapped
    if requested is not None and requested != cfg.toe_model:
        cfg = replace(cfg, toe_model=requested)
    return cfg


@dataclass
class GaitReplayResult:
    times: NDArray[np.float64]
    stance_percent: NDArray[np.float64]
    measured_Fx_ve: NDArray[np.float64]
    measured_Fy_ve: NDArray[np.float64]
    measured_Mz_ve: NDArray[np.float64]
    measured_COP_x: NDArray[np.float64]  # backward compat: global lab COP x (m)
    elastic_Fx: NDArray[np.float64]
    elastic_Fy: NDArray[np.float64]
    elastic_Mz: NDArray[np.float64]
    phi: NDArray[np.float64]
    varphi: NDArray[np.float64]
    alpha: NDArray[np.float64]
    theta_deg: NDArray[np.float64]
    contact_type: list[str]
    record_index: NDArray[np.int64]
    d_ax: NDArray[np.float64]
    d_ay: NDArray[np.float64]
    predicted_Fx: NDArray[np.float64]
    predicted_Fy: NDArray[np.float64]
    predicted_Mz: NDArray[np.float64]
    force_residual: NDArray[np.float64]
    moment_residual: NDArray[np.float64]
    cop_residual: NDArray[np.float64]
    cond: NDArray[np.float64]
    max_free_penetration: NDArray[np.float64]
    min_contact_reaction: NDArray[np.float64]
    selection_status: list[str]
    instant_contact_type: list[str]
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
    measured_moment_about_model_anchor: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    predicted_moment_about_model_anchor: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    predicted_toe_moment: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0)
    )
    cop_valid: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    cop_out_of_shoe: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Toe model (passive spring fields are NaN for legacy models).
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
    toe_root_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    toe_spring_energy_J: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_solve_status: list[str] = field(default_factory=list)
    toe_low_load: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    toe_q0_Nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_q1_Nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    toe_roots_deg: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    toe_Q_coeff: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 5)))
    # Selected contact interval I_ij (indices are -1 / coordinates NaN without a selection).
    contact_start_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contact_end_index: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contact_start_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_end_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_anchor_x: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    contact_anchor_y: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    candidate_search_method: list[str] = field(default_factory=list)
    # Edge / unilateral / violation diagnostics of the selected candidate, keyed by
    # ``gait.wrench_control.CANDIDATE_DIAGNOSTIC_FIELDS``.
    contact_diagnostics: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def theta_rad(self) -> NDArray[np.float64]:
        return np.deg2rad(np.asarray(self.theta_deg, dtype=np.float64))

    @property
    def contact_anchor_translation_x(self) -> NDArray[np.float64]:
        return self.d_ax

    @property
    def contact_anchor_translation_y(self) -> NDArray[np.float64]:
        return self.d_ay

    def interval_column_dict(self) -> dict[str, Any]:
        n = int(self.times.size)

        def col(arr, fill=np.nan, dtype=np.float64):
            a = np.asarray(arr)
            return a if a.size == n else np.full(n, fill, dtype=dtype)

        out = {
            "contact_start_index": col(self.contact_start_index, -1, np.int64).astype(np.int64),
            "contact_end_index": col(self.contact_end_index, -1, np.int64).astype(np.int64),
            "contact_start_x": col(self.contact_start_x),
            "contact_end_x": col(self.contact_end_x),
            "contact_anchor_x": col(self.contact_anchor_x),
            "contact_anchor_y": col(self.contact_anchor_y),
            "contact_anchor_translation_x": self.d_ax,
            "contact_anchor_translation_y": self.d_ay,
            "force_reconstruction_error": self.force_residual,
            "force_control_condition_number": self.cond,
            "candidate_search_method": np.asarray(
                self.candidate_search_method if len(self.candidate_search_method) == n else [""] * n,
                dtype=object,
            ),
        }
        for name, arr in self.contact_diagnostics.items():
            key = "candidate_violation_score" if name == "violation_score" else name
            out[key] = col(arr)
        return out

    def toe_column_dict(self) -> dict[str, Any]:
        n = int(self.times.size)

        def col(arr, fill=np.nan, dtype=np.float64):
            a = np.asarray(arr)
            return a if a.size == n else np.full(n, fill, dtype=dtype)

        return {
            "toe_model": np.asarray([self.toe_model] * n, dtype=object),
            "toe_stiffness_Nm_per_rad": np.full(n, float(self.toe_stiffness_Nm_per_rad)),
            "toe_neutral_angle_rad": np.full(n, float(self.toe_neutral_angle_rad)),
            "toe_damping_Nms_per_rad": np.full(n, float(self.toe_damping_Nms_per_rad)),
            "theta_rad": self.theta_rad,
            "theta_deg": np.asarray(self.theta_deg, dtype=np.float64),
            "toe_alpha": np.asarray(self.alpha, dtype=np.float64),
            "toe_angle_rad": self.theta_rad,
            "toe_equilibrium_residual": col(self.toe_equilibrium_residual_Nm),
            "Q_alpha_shoe_on_foot": col(self.Q_alpha_shoe_on_foot),
            "Q_theta_shoe_on_foot": col(self.Q_theta_shoe_on_foot),
            "toe_spring_moment_Nm": col(self.toe_spring_moment_Nm),
            "toe_spring_generalized_force_alpha": col(self.toe_spring_generalized_force_alpha),
            "toe_equilibrium_residual_Nm": col(self.toe_equilibrium_residual_Nm),
            "toe_equilibrium_relative_residual": col(self.toe_equilibrium_relative_residual),
            "toe_equilibrium_tangent": col(self.toe_equilibrium_tangent),
            "toe_equilibrium_stable": col(self.toe_equilibrium_stable, 0, np.int8).astype(np.int8),
            "toe_root_count": col(self.toe_root_count, 0, np.int64).astype(np.int64),
            "toe_root_index": col(self.toe_root_index, -1, np.int64).astype(np.int64),
            "toe_spring_energy_J": col(self.toe_spring_energy_J),
            "toe_solve_status": np.asarray(
                self.toe_solve_status if len(self.toe_solve_status) == n else [""] * n, dtype=object
            ),
            "toe_low_load": col(self.toe_low_load, 0, np.int8).astype(np.int8),
            "toe_q0_Nm": col(self.toe_q0_Nm),
            "toe_q1_Nm": col(self.toe_q1_Nm),
        }

    def to_column_dict(self) -> dict[str, Any]:
        return {
            "time": self.times,
            "stance_percent": self.stance_percent,
            "measured_Fx_ve": self.measured_Fx_ve,
            "measured_Fy_ve": self.measured_Fy_ve,
            "measured_moment": self.measured_Mz_ve,
            "measured_COP_x": self.measured_COP_x,
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
            "selected_contact_type": np.asarray(self.contact_type, dtype=object),
            "selected_record_index": self.record_index,
            "contact_translation_x": self.d_ax,
            "contact_translation_y": self.d_ay,
            "predicted_Fx": self.predicted_Fx,
            "predicted_Fy": self.predicted_Fy,
            "predicted_moment": self.predicted_Mz,
            "force_residual": self.force_residual,
            "moment_residual": self.moment_residual,
            "COP_residual": self.cop_residual,
            "measured_moment_about_model_anchor": self.measured_moment_about_model_anchor,
            "predicted_moment_about_model_anchor": self.predicted_moment_about_model_anchor,
            "predicted_toe_moment": self.predicted_toe_moment,
            "wrench_matrix_condition_number": self.cond,
            "max_free_penetration": self.max_free_penetration,
            "min_contact_normal_reaction": self.min_contact_reaction,
            "selection_status": np.asarray(self.selection_status, dtype=object),
            "instant_best_contact_type": np.asarray(self.instant_contact_type, dtype=object),
            "cop_valid": self.cop_valid.astype(np.int8),
            "cop_out_of_shoe": self.cop_out_of_shoe.astype(np.int8),
            **self.interval_column_dict(),
            **self.toe_column_dict(),
        }


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _empty_candidate(row: int = -1) -> WrenchCandidate:
    return WrenchCandidate(
        row=row,
        contact_type=ContactType.FULL,
        contact_start_index=-1,
        contact_end_index=-1,
        contact_start_x=float("nan"),
        contact_end_x=float("nan"),
        anchor_x=float("nan"),
        anchor_y=float("nan"),
        d_ax=float("nan"),
        d_ay=float("nan"),
        alpha=float("nan"),
        theta_deg=float("nan"),
        Fx_pred=float("nan"),
        Fy_pred=float("nan"),
        Mz_pred=float("nan"),
        force_residual=float("inf"),
        moment_residual=float("inf"),
        cop_residual=float("nan"),
        cond=float("inf"),
        max_free_penetration=float("nan"),
        min_contact_reaction=float("nan"),
        admissible=False,
        status="no_selection",
        gamma=np.full(5, np.nan),
        search_method=SEARCH_FALLBACK,
    )


def _candidates_force_phi(
    lookup: ContactLookupResult,
    *,
    Fx: float,
    Fy: float,
    phi_rad: float,
    theta_deg: float,
    Mz_meas: float,
    tolerances: Tolerances | None,
    previous_interval: tuple[int, int] | None = None,
    max_candidates: int = 16,
) -> list[WrenchCandidate]:
    """Interval candidates for prescribed Fx, Fy, φ, θ (2×2 translation solve).

    Uses the staged interval search; returns the admissible records of the
    first stage that has any (or the least-violating records otherwise).
    """
    phi_deg = float(np.rad2deg(phi_rad))
    ev = evaluate_candidates(lookup, Fx, Fy, phi_deg, float(theta_deg), tolerances=tolerances)
    sel = select_contact_candidate(ev, previous_interval=previous_interval)
    method = sel.candidate_search_method
    if sel.selected_row is None:
        return []
    if method == SEARCH_FALLBACK:
        pool = np.flatnonzero(ev.evaluated & np.isfinite(ev.violation_score))
        V = ev.violation_score
        ranked = refine_top_k(pool, lambda p: (p, V[p]), lambda p: ev.fields_exact[p], ev.ensure_exact,
                              max(1, int(max_candidates)))
        ranked = ranked[np.isfinite(V[ranked])]
        order = np.searchsorted(pool, ranked)
    else:
        pool = sel.admissible_rows
        comp = np.nan_to_num(ev.complementarity_score[pool], nan=np.inf)
        dist = interval_distance(ev.contact_start_index[pool], ev.contact_end_index[pool], previous_interval)
        order = np.lexsort((pool, comp, dist))
    rows = list(pool[order[: max(1, int(max_candidates))]])
    if sel.selected_row not in rows:
        rows.insert(0, sel.selected_row)
    ev.ensure_exact(np.asarray(rows, dtype=int))
    out: list[WrenchCandidate] = []
    alpha = float(ev.alpha)
    varphi = float(ev.varphi)
    for row in rows:
        row = int(row)
        d_ax = float(ev.d_ax[row])
        d_ay = float(ev.d_ay[row])
        gamma = basis_coefficients(alpha, d_ax, d_ay, varphi)
        # Full planar moment about (0,0): Mz = x·fy − y·fx (not Mv = x·fy).
        mz_p = float(
            contract_basis(np.asarray(lookup.scalar_lookup[row], dtype=float), gamma, mode_axis=0)[SCALAR_MZ]
        )
        diag = {
            name: (bool(ev.fields[name][row]) if name == "disconnected_contact_warning" else float(ev.fields[name][row]))
            for name in CANDIDATE_DIAGNOSTIC_FIELDS
        }
        out.append(
            WrenchCandidate(
                row=row,
                contact_type=ev.contact_type(row),
                contact_start_index=int(ev.contact_start_index[row]),
                contact_end_index=int(ev.contact_end_index[row]),
                contact_start_x=float(ev.contact_start_x[row]),
                contact_end_x=float(ev.contact_end_x[row]),
                anchor_x=float(ev.anchor_x[row]),
                anchor_y=float(ev.anchor_y[row]),
                d_ax=d_ax,
                d_ay=d_ay,
                alpha=alpha,
                theta_deg=float(theta_deg),
                Fx_pred=float(ev.Fx[row]),
                Fy_pred=float(ev.Fy[row]),
                Mz_pred=mz_p,
                force_residual=float(ev.force_residual[row]),
                # Moment is not prescribed in this mode; keep diagnostic only.
                moment_residual=float(abs(mz_p - Mz_meas)) if np.isfinite(mz_p) else float("inf"),
                cop_residual=float("nan"),
                cond=float(ev.kf_cond[row]) if np.isfinite(ev.kf_cond[row]) else float("inf"),
                max_free_penetration=float(ev.max_free_penetration[row]),
                min_contact_reaction=float(ev.min_contact_reaction[row]),
                admissible=bool(ev.admissible[row]),
                status="force_phi",
                gamma=gamma,
                search_method=method,
                diagnostics=diag,
            )
        )
    return out


def replay_stance(
    lookup: ContactLookupResult,
    *,
    trc_path: str | Path | None = None,
    mot_path: str | Path | None = None,
    foot: str = "right",
    force_plate: str = "auto",
    stance_index: int = 0,
    vertical_force_threshold: float = 20.0,
    theta_mode: ThetaMode | None = None,
    toe_model: str | None = None,
    toe_config: ToeSpringConfig | None = None,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
    cop_fit_min_force: float = 50.0,
    shoe_width_m: float = 0.10,
    cond_warn: float = 1.0e8,
    visco_model: str = "elastic",
    visco_config: dict[str, Any] | None = None,
    temporal_weights: TemporalWeights | None = None,
    tolerances: Tolerances | None = None,
    pre_pad_s: float = 0.05,
    post_pad_s: float = 0.05,
    heel_marker: str | None = None,
    fore_marker: str | None = None,
    position_units: str = "auto",
    moment_units: str = "auto",
    cop_fy_min: float = 20.0,
    synthetic: dict[str, Any] | None = None,
    selection_config: SelectionConfig | None = None,
) -> GaitReplayResult:
    """Replay one stance (no per-frame FEM).

    Default ``toe_model='passive_spring'``: prescribe experimental Fx, Fy and
    marker φ; θ follows from the exact passive toe-spring equilibrium
    ``Q_alpha,shoe->foot(alpha) = dU/dalpha`` on every contact record
    (``toe_spring``). The measured COP/Mz is not prescribed and is reported as
    a validation residual. Requires ``visco_model='elastic'`` unless
    ``toe_model='passive_spring_elastic_equivalent'`` is chosen explicitly.

    Legacy models (never selected automatically): ``fit_cop_legacy``
    (``theta_mode='fit-cop'``, 3×3 COP/Mz wrench fit seeded by the force-phi
    formula) and ``prescribed_legacy`` (``theta_mode='force-phi'``,
    θ = ReLU(−φ)(1−(1−Fy/Fymax)^4)).
    """
    toe_cfg = resolve_toe_config(toe_model, toe_config, theta_mode)
    model = toe_cfg.toe_model
    if model == TOE_MODEL_FIT_COP_LEGACY:
        theta_mode = "fit-cop"
    elif model == TOE_MODEL_PRESCRIBED_LEGACY:
        theta_mode = "force-phi"
    else:
        theta_mode = None
    passive = toe_cfg.is_passive
    if model == TOE_MODEL_PASSIVE_SPRING and str(visco_model).lower() != "elastic":
        raise ValueError(VISCO_REJECTION_MESSAGE)

    warnings: list[str] = []
    if passive and toe_cfg.toe_damping_Nms_per_rad != 0.0:
        warnings.append(
            f"toe_damping_Nms_per_rad={toe_cfg.toe_damping_Nms_per_rad:g} is recorded but does not "
            "enter the quasistatic toe equilibrium"
        )
    if model == TOE_MODEL_PASSIVE_SPRING_ELASTIC_EQUIVALENT and str(visco_model).lower() != "elastic":
        warnings.append(
            "passive_spring_elastic_equivalent: toe spring solved against the elastic-equivalent "
            "wrench from the viscoelastic map; this is an approximation (no Q_alpha,VE)"
        )
    if passive:
        method, method_version = "passive-toe-spring-exact", "1"
    elif theta_mode == "fit-cop":
        method, method_version = "fit-cop-3x3-wrench", "2-heel-relative-Mz"
    else:
        method, method_version = TOE_ANGLE_METHOD, TOE_ANGLE_METHOD_VERSION
    provenance: dict[str, Any] = {
        "software_version": __version__,
        "visco_model": visco_model,
        "viscoelastic_assumption": (
            "componentwise scalar kernel on generalized wrench; "
            "exact only for uniform relaxation spectrum"
        ),
        "force_sign": "F_top = -F_GRF_ground_on_body",
        "toe_model": model,
        "toe_config": toe_cfg.to_provenance(),
        "theta_mode": theta_mode,
        "toe_angle_method": method,
        "toe_angle_method_version": method_version,
        "shoe_width_m": float(shoe_width_m),
        "force_width_scaling": (
            "Experimental Fx,Fy,Mz (N, N·m) are divided by shoe_width_m to match "
            "plane-strain unit-thickness lookup units (N/m, N·m/m)."
        ),
        "laboratory_frame_definition": "+x anterior, +y superior, +z right (OpenSim)",
        "foot_frame_definition": (
            "heel-attached; +x along heel→forefoot chord at angle φ; "
            "p_foot = Q(φ)^T (p_lab - r_heel_lab)"
        ),
        "cop_reference_frame": {
            "cop_lab_*": "global laboratory/OpenSim ground",
            "cop_foot_*": "heel-attached foot frame",
            "cop_model_x": "model frame about lookup moment origin (0,0); x≈Mz/Fy",
            "measured_COP_x": "deprecated alias of cop_lab_x (NOT heel-relative)",
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
        provenance["source"] = "synthetic"
        provenance["source_format"] = "synthetic"
        visco_init = "relaxed-at-window-start"
        cop_lab_x = np.asarray(synthetic.get("cop_x", np.zeros_like(times)), dtype=np.float64)
        cop_lab_y = np.asarray(synthetic.get("cop_y", np.zeros_like(times)), dtype=np.float64)
        heel_lab_x = np.asarray(synthetic.get("heel_x", np.zeros_like(times)), dtype=np.float64)
        heel_lab_y = np.asarray(synthetic.get("heel_y", np.zeros_like(times)), dtype=np.float64)
        fore_lab_x = np.asarray(
            synthetic.get("fore_x", heel_lab_x + float(getattr(lookup, "L", 0.3))),
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
        provenance.update(
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
                        "-(r_COP/heel × F_lab + T_z)/w  (matches lookup Mz about 0)"
                    ),
                    "measured_moment_about_model_anchor": "same as wrench_Mz_star",
                    "predicted_moment_about_model_anchor": "lookup Mz about (0,0)",
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
    provenance["visco_initialization"] = visco_init

    # Heel-relative COP (always). For TRC+MOT, replace lab-origin Mz with the
    # top-of-shoe moment about the heel so fit-cop matches COP / Mz. Synthetic
    # trials already pass model-top Mz directly — keep it.
    foot_cop_early = cop_lab_to_foot(cop_lab_x, cop_lab_y, heel_lab_x, heel_lab_y, phi)
    if synthetic is not None:
        Mz_for_fit = np.asarray(wrench.Mz, dtype=np.float64)
    else:
        M_ground_about_heel = force_induced_moment_about_point(
            force_lab_fx,
            force_lab_fy,
            cop_lab_x,
            cop_lab_y,
            origin_x=heel_lab_x,
            origin_y=heel_lab_y,
        ) + np.asarray(free_tz_lab, dtype=np.float64)
        # Action–reaction: top-of-shoe moment matching lookup Mz convention.
        Mz_for_fit = -M_ground_about_heel

    # Plane-strain lookup assumes unit out-of-plane thickness: convert total
    # experimental loads to per-metre line loads.
    width = float(shoe_width_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError(f"shoe_width_m must be positive, got {shoe_width_m!r}")
    inv_w = 1.0 / width
    wrench = SagittalWrench(
        Fx=np.asarray(wrench.Fx, dtype=np.float64) * inv_w,
        Fy=np.asarray(wrench.Fy, dtype=np.float64) * inv_w,
        Mz=np.asarray(Mz_for_fit, dtype=np.float64) * inv_w,
        cop_x=np.asarray(foot_cop_early.x, dtype=np.float64),
        cop_y=np.asarray(foot_cop_early.y, dtype=np.float64),
        Fz_omitted=np.asarray(wrench.Fz_omitted, dtype=np.float64) * inv_w,
        times=np.asarray(wrench.times, dtype=np.float64),
    )
    # Thresholds entered in experimental Newtons → model N/m.
    cop_fit_min_force_model = float(cop_fit_min_force) * inv_w

    n = times.size
    # Viscoelastic map on the wrench used by the selected θ mode.
    cfg = dict(visco_config or {})
    cfg["model"] = visco_model
    # Passive and prescribed models only prescribe (Fx, Fy); Mz stays a diagnostic.
    n_comp = 3 if theta_mode == "fit-cop" else 2
    cfg["n_components"] = n_comp
    material = create_material(cfg)
    material.reset()
    Fe = np.zeros((n, 3), dtype=np.float64)
    Fve = np.column_stack([wrench.Fx, wrench.Fy, wrench.Mz])
    for i in range(n):
        if n_comp == 2:
            Fe[i, :2] = material.update(Fve[i, :2], float(times[i]))
            Fe[i, 2] = Fve[i, 2]
        else:
            Fe[i] = material.update(Fve[i], float(times[i]))

    phi_ref = float(getattr(lookup, "phi_ref", 0.0))
    varphi = phi - phi_ref

    fy_c = compressive_fy_load(Fe[:, 1])
    fy_max = float(np.max(fy_c)) if fy_c.size else 0.0
    provenance["fy_max_compressive"] = fy_max
    # Legacy force-phi formula: prescribed θ in prescribed_legacy, seed for
    # fit_cop_legacy. It is never evaluated for the passive spring.
    if fy_max <= 0.0 and theta_mode == "force-phi":
        warnings.append("Fy_max compressive load is zero; toe angle load factor is undefined")
    if passive:
        theta_force_phi = np.full(n, np.nan)
    else:
        theta_force_phi = prescribed_toe_angle_deg(
            phi,
            Fe[:, 1],
            fy_max,
            theta_min_deg=theta_min_deg,
            theta_max_deg=theta_max_deg,
        )
    toe_low_force_model = float(toe_cfg.toe_low_force_threshold_N) * inv_w
    if passive:
        provenance["toe_angle_formula"] = TOE_EQUATION
        provenance["toe_generalized_force"] = TOE_GENERALIZED_FORCE_DEFINITION
        provenance["toe_generalized_force_width_scaling"] = (
            "Q_total (N*m) = shoe_width_m * Q_lookup (N*m per metre width)"
        )
        provenance["toe_generalized_force_source"] = getattr(
            lookup, "toe_generalized_force_source", "unknown"
        )
        provenance["toe_low_load_rule"] = (
            "|F*| below toe_low_force_threshold_N: keep the root on the branch connected "
            "to theta0 per record; frame flagged low_load; no formula fallback"
        )
        provenance["toe_rearfoot_kinematics"] = {
            "phi_is": "fixed-frame heel -> MTP (rearfoot) marker angle",
            "chord_rotation": "varphi(alpha) = phi - atan2(dy_a + alpha*phi1(a), a)",
            "varphi_column": "chord-frame rotation at the selected root",
            **lookup_rearfoot_geometry(lookup).to_dict(),
        }
        theta_prescribed = np.full(n, np.nan)
    elif theta_mode == "force-phi":
        provenance["toe_angle_formula"] = (
            "theta_deg = relu(-phi_deg) * (1 - (1 - Fy_c/Fy_max)^4); "
            "Fy_c = max(0, -Fy_top)"
        )
        theta_prescribed = theta_force_phi
    else:
        provenance["toe_angle_formula"] = (
            "3×3 wrench fit of (Fx,Fy,Mz_heel) → (dx,dy,α=tanθ); "
            "force-phi formula seeds θ when free α is out of bounds or Kw is "
            "ill-conditioned; θ=0 when |Fy| < cop_fit_min_force"
        )
        provenance["theta_prior"] = "force-phi-relu-neg-phi"
        theta_prescribed = np.full(n, np.nan)

    frames: list[list[WrenchCandidate]] = []
    instant: list[WrenchCandidate] = []
    prev_interval: tuple[int, int] | None = None
    sel_cfg = selection_config or SelectionConfig()

    prev_theta_deg: float | None = None
    t_loop = time.perf_counter()
    for i in range(n):
        phi_i = float(phi[i]) if np.isfinite(phi[i]) else phi_ref
        if passive:
            low_load = float(np.hypot(Fe[i, 0], Fe[i, 1])) < toe_low_force_model
            cands = solve_passive_toe_candidates(
                lookup,
                Fx_star=float(Fe[i, 0]),
                Fy_star=float(Fe[i, 1]),
                Mz_meas=float(Fe[i, 2]),
                phi_rad=phi_i,
                config=toe_cfg,
                width_m=width,
                tolerances=tolerances,
                low_load=low_load,
                cond_limit=cond_warn,
                previous_interval=prev_interval,
                selection_config=sel_cfg,
            )
            if not cands:
                cands = [_empty_candidate()]
            frames.append(cands)
            best = pick_instant_best_passive(cands, prev_theta_deg)
            assert best is not None
            instant.append(best)
            if np.isfinite(best.theta_deg):
                prev_theta_deg = float(best.theta_deg)
            if best.row >= 0:
                prev_interval = best.interval
            continue
        if theta_mode == "force-phi":
            theta_i = float(theta_prescribed[i])
            cands = _candidates_force_phi(
                lookup,
                Fx=float(Fe[i, 0]),
                Fy=float(Fe[i, 1]),
                phi_rad=phi_i,
                theta_deg=theta_i,
                Mz_meas=float(Fe[i, 2]),
                tolerances=tolerances,
                previous_interval=prev_interval,
            )
        else:
            # Fit COP / Mz: free 3×3, seeded by force-phi when α is unusable;
            # freeze θ = 0 at negligible vertical load.
            low = abs(float(Fe[i, 1])) < cop_fit_min_force_model
            theta_guess = 0.0 if low else float(theta_force_phi[i])
            cands = solve_wrench_control(
                lookup,
                Fx_star=float(Fe[i, 0]),
                Fy_star=float(Fe[i, 1]),
                Mz_star=float(Fe[i, 2]),
                phi_rad=phi_i,
                theta_min_deg=theta_min_deg,
                theta_max_deg=theta_max_deg,
                cond_warn=cond_warn,
                tolerances=tolerances,
                low_force_fallback=low,
                theta_prior_deg=theta_guess,
            )

        if not cands:
            cands = [_empty_candidate()]
        frames.append(cands)
        best = pick_instant_best(cands)
        assert best is not None
        instant.append(best)
        if best.row >= 0:
            prev_interval = best.interval
    runtime_s = time.perf_counter() - t_loop
    provenance["runtime_total_s"] = float(runtime_s)
    provenance["runtime_per_sample_s"] = float(runtime_s / max(n, 1))

    # Moment is not prescribed for force-phi or the passive spring: drop it from
    # the Viterbi emission so topology tracks Fx/Fy/admissibility (and toe balance).
    if theta_mode == "force-phi" or passive:
        from dataclasses import replace as _replace

        frames_for_viterbi = [[_replace(c, moment_residual=0.0) for c in cands] for cands in frames]
    else:
        frames_for_viterbi = frames

    L_scale = float(getattr(lookup, "L", 1.0))
    F_scale = float(max(np.max(np.abs(Fe[:, :2])), 1.0))
    selected = viterbi_select(
        frames_for_viterbi,
        weights=temporal_weights,
        F_scale=F_scale,
        L_scale=L_scale,
        top_k=16 if passive else 8,
    )

    st0 = stance.t0
    st1 = max(stance.t1, st0 + 1e-12)

    # --- Frame-aware COP / moment diagnostics ---
    foot_cop = foot_cop_early
    predicted_Fy = np.array([c.Fy_pred for c in selected], dtype=np.float64)
    predicted_M_anchor = np.array(
        [
            float(
                contract_basis(
                    np.asarray(lookup.scalar_lookup[int(c.row)], dtype=float),
                    np.asarray(c.gamma, dtype=float),
                    mode_axis=0,
                )[SCALAR_MZ]
            )
            if c.row >= 0 and np.all(np.isfinite(c.gamma))
            else float("nan")
            for c in selected
        ],
        dtype=np.float64,
    )
    predicted_toe = np.array(
        [
            float(
                contract_basis(
                    np.asarray(lookup.scalar_lookup[int(c.row)], dtype=float),
                    np.asarray(c.gamma, dtype=float),
                    mode_axis=0,
                )[SCALAR_TOE]
            )
            if c.row >= 0 and np.all(np.isfinite(c.gamma))
            else float("nan")
            for c in selected
        ],
        dtype=np.float64,
    )
    predicted_Mz = predicted_M_anchor.copy()
    cop_model_x = model_cop_from_wrench(
        predicted_M_anchor,
        predicted_Fy,
        fy_min=float(cop_fy_min) / max(float(shoe_width_m), 1e-12),
    )
    # Same top-of-shoe heel moment used as Mz_star (width-scaled), after visco for fit-cop.
    measured_M_anchor_model = np.asarray(Fe[:, 2], dtype=np.float64)
    provenance["moment_sign_convention"] = (
        "measured_moment_about_model_anchor = Fe_Mz = -(r_COP/heel × F_lab + T_z)/w "
        "(top-of-shoe convention, matches lookup Mz about 0). "
        + (
            "Passive spring: Mz is not prescribed; the moment/COP residual validates the "
            "spring-predicted θ against measured COP."
            if passive
            else "θ from 3×3 fit when theta_mode=fit-cop; θ=0 at low |Fy|."
        )
    )

    cop_valid, cop_oos = cop_validity_mask(
        fy_lab_vertical=fy_lab_vertical,
        cop_foot_x=foot_cop.x,
        shoe_length=float(getattr(lookup, "L", 1.0)),
        fy_min=float(cop_fy_min),
    )
    cop_res = np.full(times.size, np.nan, dtype=np.float64)
    ok = cop_valid & np.isfinite(cop_model_x)
    cop_res[ok] = np.abs(foot_cop.x[ok] - cop_model_x[ok])

    n_valid = int(np.count_nonzero(cop_valid))
    if n_valid:
        in_shoe = cop_valid & ~cop_oos
        provenance["cop_validation"] = {
            "n_valid_frames": n_valid,
            "fraction_in_shoe_interval": float(np.count_nonzero(in_shoe) / n_valid),
            "cop_foot_x_min": float(np.nanmin(foot_cop.x[cop_valid])),
            "cop_foot_x_max": float(np.nanmax(foot_cop.x[cop_valid])),
            "mean_abs_cop_residual_m": float(np.nanmean(cop_res[ok])) if np.any(ok) else None,
            "cop_fy_min_N": float(cop_fy_min),
        }
        mom_ok = cop_valid & np.isfinite(measured_M_anchor_model) & np.isfinite(predicted_M_anchor)
        if np.any(mom_ok):
            provenance["moment_validation"] = {
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

    # Observation on θ:
    if passive:
        provenance["toe_angle_note"] = (
            "θ from the exact passive toe-spring equilibrium per contact record "
            f"(k={toe_cfg.toe_stiffness_Nm_per_rad:g} N·m/rad, θ0={toe_cfg.toe_neutral_angle_rad:g} rad); "
            "Brent roots on [toe_angle_min_deg, toe_angle_max_deg]; no formula fallback."
        )
    elif theta_mode == "fit-cop":
        provenance["toe_angle_note"] = (
            "θ from 3×3 COP/Mz wrench fit using heel-relative top-of-shoe moment; "
            "force-phi formula is the θ prior when free α is OOB/ill-conditioned; "
            "θ fixed at 0 when |Fy| < cop_fit_min_force."
        )
    else:
        provenance["toe_angle_note"] = (
            "θ = relu(-φ_deg)*(1-(1-Fy_c/Fy_max)^4) (legacy force-phi mode)."
        )

    if passive:
        varphi = np.array(
            [
                float(c.toe.varphi_rad)
                if c.toe is not None and np.isfinite(c.toe.varphi_rad)
                else float(varphi[i])
                for i, c in enumerate(selected)
            ],
            dtype=np.float64,
        )

    result = GaitReplayResult(
        times=times,
        stance_percent=100.0 * np.clip((times - st0) / (st1 - st0), 0.0, 1.0),
        measured_Fx_ve=Fve[:, 0],
        measured_Fy_ve=Fve[:, 1],
        measured_Mz_ve=Fve[:, 2],
        measured_COP_x=np.asarray(cop_lab_x, dtype=np.float64),
        elastic_Fx=Fe[:, 0],
        elastic_Fy=Fe[:, 1],
        elastic_Mz=Fe[:, 2],
        phi=phi,
        varphi=varphi,
        alpha=np.array([c.alpha for c in selected], dtype=np.float64),
        theta_deg=np.array([c.theta_deg for c in selected], dtype=np.float64),
        contact_type=[c.contact_type.value for c in selected],
        record_index=np.array([c.row for c in selected], dtype=np.int64),
        d_ax=np.array([c.d_ax for c in selected], dtype=np.float64),
        d_ay=np.array([c.d_ay for c in selected], dtype=np.float64),
        predicted_Fx=np.array([c.Fx_pred for c in selected], dtype=np.float64),
        predicted_Fy=predicted_Fy,
        predicted_Mz=predicted_Mz,
        force_residual=np.array([c.force_residual for c in selected], dtype=np.float64),
        moment_residual=np.array(
            [
                float(abs(predicted_M_anchor[i] - measured_M_anchor_model[i]))
                if np.isfinite(predicted_M_anchor[i]) and np.isfinite(measured_M_anchor_model[i])
                else (
                    float(abs(selected[i].Mz_pred - Fe[i, 2]))
                    if theta_mode == "force-phi" and np.isfinite(selected[i].Mz_pred)
                    else selected[i].moment_residual
                )
                for i in range(len(selected))
            ],
            dtype=np.float64,
        ),
        cop_residual=cop_res,
        cond=np.array([c.cond for c in selected], dtype=np.float64),
        max_free_penetration=np.array([c.max_free_penetration for c in selected], dtype=np.float64),
        min_contact_reaction=np.array([c.min_contact_reaction for c in selected], dtype=np.float64),
        selection_status=[c.status for c in selected],
        instant_contact_type=[c.contact_type.value for c in instant],
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
        provenance=provenance,
        warnings=warnings,
        toe_model=model,
        toe_stiffness_Nm_per_rad=float(toe_cfg.toe_stiffness_Nm_per_rad) if passive else float("nan"),
        toe_neutral_angle_rad=float(toe_cfg.toe_neutral_angle_rad) if passive else float("nan"),
        toe_damping_Nms_per_rad=float(toe_cfg.toe_damping_Nms_per_rad) if passive else float("nan"),
    )
    _attach_interval_outputs(result, selected)
    provenance["interval_summary"] = _interval_summary(result, lookup)
    if passive:
        _attach_toe_outputs(result, selected)
        provenance["toe_summary"] = _toe_summary(result)
    return result


def _attach_interval_outputs(result: GaitReplayResult, selected: list[WrenchCandidate]) -> None:
    """Copy the selected interval, edge, unilateral and selection diagnostics onto the result."""
    result.contact_start_index = np.array([c.contact_start_index for c in selected], dtype=np.int64)
    result.contact_end_index = np.array([c.contact_end_index for c in selected], dtype=np.int64)
    result.contact_start_x = np.array([c.contact_start_x for c in selected], dtype=np.float64)
    result.contact_end_x = np.array([c.contact_end_x for c in selected], dtype=np.float64)
    result.contact_anchor_x = np.array([c.anchor_x for c in selected], dtype=np.float64)
    result.contact_anchor_y = np.array([c.anchor_y for c in selected], dtype=np.float64)
    methods = []
    for c in selected:
        if c.search_method:
            methods.append(c.search_method)
        else:
            methods.append(SEARCH_GLOBAL if c.admissible else SEARCH_FALLBACK)
    result.candidate_search_method = methods
    diag: dict[str, NDArray[np.float64]] = {}
    for name in CANDIDATE_DIAGNOSTIC_FIELDS:
        diag[name] = np.array(
            [float(c.diagnostics.get(name, np.nan)) for c in selected], dtype=np.float64
        )
    result.contact_diagnostics = diag


def _interval_summary(result: GaitReplayResult, lookup: ContactLookupResult) -> dict[str, Any]:
    labels = ("heel", "interior", "toe", "full")
    return {
        "contact_set_model": "single_contiguous_interval",
        "n_bottom_nodes": int(lookup.n_bottom_nodes),
        "n_interval_records": int(lookup.n_records),
        "n_valid_interval_records": int(np.count_nonzero(lookup.valid_mask)),
        "label_counts": {k: int(sum(c == k for c in result.contact_type)) for k in labels},
        "search_method_counts": {
            m: int(sum(s == m for s in result.candidate_search_method))
            for m in sorted(set(result.candidate_search_method))
        },
        "n_disconnected_contact_warning_frames": int(
            np.nansum(result.contact_diagnostics.get("disconnected_contact_warning", np.zeros(0)))
        ),
        "n_inadmissible_frames": int(
            sum(m == SEARCH_FALLBACK for m in result.candidate_search_method)
        ),
    }


def _attach_toe_outputs(result: GaitReplayResult, selected: list[WrenchCandidate]) -> None:
    """Copy per-frame toe diagnostics of the selected candidates onto the result."""
    nan = float("nan")

    def get(c: WrenchCandidate, name: str, default):
        toe = getattr(c, "toe", None)
        return getattr(toe, name) if toe is not None else default

    result.Q_alpha_shoe_on_foot = np.array([get(c, "Q_alpha_shoe_on_foot", nan) for c in selected])
    result.Q_theta_shoe_on_foot = np.array([get(c, "Q_theta_shoe_on_foot", nan) for c in selected])
    result.toe_spring_moment_Nm = np.array([get(c, "toe_spring_moment_Nm", nan) for c in selected])
    result.toe_spring_generalized_force_alpha = np.array(
        [get(c, "toe_spring_generalized_force_alpha", nan) for c in selected]
    )
    result.toe_equilibrium_residual_Nm = np.array(
        [get(c, "toe_equilibrium_residual_Nm", nan) for c in selected]
    )
    result.toe_equilibrium_relative_residual = np.array(
        [get(c, "toe_equilibrium_relative_residual", nan) for c in selected]
    )
    result.toe_equilibrium_tangent = np.array([get(c, "toe_equilibrium_tangent", nan) for c in selected])
    result.toe_equilibrium_stable = np.array(
        [bool(get(c, "toe_equilibrium_stable", False)) for c in selected], dtype=bool
    )
    result.toe_root_count = np.array([int(get(c, "toe_root_count", 0)) for c in selected], dtype=np.int64)
    result.toe_root_index = np.array([int(get(c, "toe_root_index", -1)) for c in selected], dtype=np.int64)
    result.toe_spring_energy_J = np.array([get(c, "toe_spring_energy_J", nan) for c in selected])
    result.toe_solve_status = [str(get(c, "toe_solve_status", "no_selection")) for c in selected]
    result.toe_low_load = np.array([bool(get(c, "low_load", False)) for c in selected], dtype=bool)
    result.toe_q0_Nm = np.array([get(c, "q0_Nm", nan) for c in selected])
    result.toe_q1_Nm = np.array([get(c, "q1_Nm", nan) for c in selected])
    roots = [tuple(get(c, "roots_theta_deg", ())) for c in selected]
    width = max((len(r) for r in roots), default=0)
    padded = np.full((len(roots), width), np.nan)
    for i, r in enumerate(roots):
        padded[i, : len(r)] = r
    result.toe_roots_deg = padded
    coeff = np.full((len(selected), 5), np.nan)
    for i, c in enumerate(selected):
        q = tuple(get(c, "Q_coeff", ()))
        if len(q) == 5:
            coeff[i] = q
    result.toe_Q_coeff = coeff


def _toe_summary(result: GaitReplayResult) -> dict[str, Any]:
    th = np.asarray(result.theta_deg, dtype=float)
    res = np.abs(np.asarray(result.toe_equilibrium_residual_Nm, dtype=float))
    rel = np.asarray(result.toe_equilibrium_relative_residual, dtype=float)
    status = result.toe_solve_status
    return {
        "theta_deg_min": float(np.nanmin(th)) if np.any(np.isfinite(th)) else None,
        "theta_deg_max": float(np.nanmax(th)) if np.any(np.isfinite(th)) else None,
        "max_abs_residual_Nm": float(np.nanmax(res)) if np.any(np.isfinite(res)) else None,
        "max_relative_residual": float(np.nanmax(rel)) if np.any(np.isfinite(rel)) else None,
        "n_frames": int(th.size),
        "n_multiple_root_frames": int(np.count_nonzero(result.toe_root_count > 1)),
        "max_root_count": int(np.max(result.toe_root_count)) if result.toe_root_count.size else 0,
        "n_no_root_frames": int(sum("no_root" in s for s in status)),
        "n_unstable_frames": int(np.count_nonzero(~result.toe_equilibrium_stable)),
        "n_low_load_frames": int(np.count_nonzero(result.toe_low_load)),
        "n_unilateral_violation_frames": int(
            sum("unilateral_violation" in s for s in result.selection_status)
        ),
        "min_contact_reaction": (
            float(np.nanmin(result.min_contact_reaction))
            if np.any(np.isfinite(result.min_contact_reaction))
            else None
        ),
        "max_spring_energy_J": (
            float(np.nanmax(result.toe_spring_energy_J))
            if np.any(np.isfinite(result.toe_spring_energy_J))
            else None
        ),
        "contact_type_counts": {
            k: int(sum(c == k for c in result.contact_type)) for k in ("heel", "interior", "toe", "full")
        },
    }


def save_replay_result(result: GaitReplayResult, output_dir: str | Path) -> dict[str, Path]:
    """Write CSV + NPZ + provenance JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = result.to_column_dict()
    numeric_keys = [
        k
        for k, v in cols.items()
        if isinstance(v, np.ndarray) and v.dtype != object
    ]
    text_keys = [
        k
        for k, v in cols.items()
        if isinstance(v, np.ndarray) and v.dtype == object
    ]
    header = ",".join(numeric_keys + text_keys)
    rows = np.column_stack([cols[k] for k in numeric_keys])
    csv_path = output_dir / "gait_replay.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(result.times.size):
            nums = ",".join(f"{rows[i, j]:.16g}" for j in range(rows.shape[1]))
            texts = ",".join(str(cols[k][i]).replace(",", ";") for k in text_keys)
            f.write(f"{nums},{texts}\n")
    npz_path = output_dir / "gait_replay.npz"
    np.savez_compressed(
        npz_path, toe_roots_deg=result.toe_roots_deg, toe_Q_coeff=result.toe_Q_coeff, **{k: cols[k] for k in cols}
    )
    meta_path = output_dir / "provenance.json"
    meta_path.write_text(
        json.dumps(
            {"provenance": result.provenance, "warnings": result.warnings},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "npz": npz_path, "provenance": meta_path}
