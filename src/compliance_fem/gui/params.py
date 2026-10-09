"""Streamlit GUI parameter defaults and dual slider/number widgets.

Display units are chosen for human editing (mm, MPa, N·m², kN). Values
are converted to SI before they reach ``SoleConfig`` / the solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import streamlit as st

from compliance_fem.contact.config import SoleConfig


@dataclass(frozen=True)
class GuiDefaults:
    """Sidebar defaults (SI unless the name says otherwise)."""

    shoe_length_mm: float = 270.0
    mesh_size_mm: float = 3.0
    # Softplus (matches the measured-sole config defaults)
    softplus_toe_length: float = 0.0594
    softplus_kappa: float = 160.0
    # Materials
    ffturbo_E_Pa: float = 2.6e5
    ffturbo_nu: float = 0.113
    ffleap_E_heel_Pa: float = 3.54e5
    ffleap_E_toe_Pa: float = 2.07e5
    ffleap_nu: float = 0.113
    EI_plate_Nm2_per_m: float = 2.0
    # Forces (newtons)
    Fx: float = 0.0
    Fy: float = -1.0e4


DEFAULTS = GuiDefaults()


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


ANGLES = RuntimeAngleConfig()

# Display ranges (converted to SI when applied).
L_MM_MIN = 200.0
L_MM_MAX = 400.0
TOE_LENGTH_MM_MIN = 1.0
TOE_LENGTH_MM_MAX = 200.0
NU_MIN = 0.0
NU_MAX = 0.49  # plane-strain bulk lock at 0.5
# Moduli edited in MPa (SI Pa = MPa × 1e6).
E_MPA_MIN = 0.01  # 10 kPa
E_MPA_MAX = 10.0  # 10 MPa
# Plate bending stiffness edited directly in N·m² (SI).
EI_NM2_MIN = 0.1
EI_NM2_MAX = 100.0
KAPPA_MIN = 1.0
KAPPA_MAX = 1000.0
FX_N_MIN = -1.0e3
FX_N_MAX = 1.0e3
FY_N_MIN = -2.0e4  # GUI vertical GRF max = 20 kN upward
FY_N_MAX = 0.0


def mm_to_m(mm: float) -> float:
    return float(mm) * 1.0e-3


def m_to_mm(m: float) -> float:
    return float(m) * 1.0e3


def pa_to_mpa(pa: float) -> float:
    return float(pa) * 1.0e-6


def mpa_to_pa(mpa: float) -> float:
    return float(mpa) * 1.0e6


def kn_to_n(kn: float) -> float:
    return float(kn) * 1.0e3


def n_to_kn(n: float) -> float:
    return float(n) * 1.0e-3


def _ensure_key(key: str, value: float) -> None:
    if key not in st.session_state:
        st.session_state[key] = float(value)


def _printf(fmt: str, value: float) -> str:
    try:
        return fmt % float(value)
    except (TypeError, ValueError):
        return f"{float(value):.4g}"


def dual_linear(
    label: str,
    *,
    key: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float,
    fmt: str = "%.4g",
    help: str | None = None,
) -> float:
    """Slider + typed number input sharing one display value.

    The value returned is always the live widget state (not a parallel canonical
    copy). Overwriting widget keys from a stale copy before ``st.slider`` made
    Apply/rerun keep the visible slider position while the plot used defaults.
    """
    default = float(np.clip(default, min_value, max_value))
    slider_key = f"{key}__slider"
    num_key = f"{key}__num"
    _ensure_key(key, default)
    _ensure_key(slider_key, float(st.session_state[key]))
    _ensure_key(num_key, float(st.session_state[key]))

    # Bounds can shrink (e.g. toe length vs L); clip existing widget state in place.
    st.session_state[slider_key] = float(
        np.clip(st.session_state[slider_key], min_value, max_value)
    )
    st.session_state[num_key] = float(
        np.clip(st.session_state[num_key], min_value, max_value)
    )

    def _from_slider() -> None:
        st.session_state[key] = float(st.session_state[slider_key])
        st.session_state[num_key] = float(st.session_state[key])

    def _from_num() -> None:
        val = float(np.clip(st.session_state[num_key], min_value, max_value))
        st.session_state[key] = val
        st.session_state[slider_key] = val
        st.session_state[num_key] = val

    c1, c2 = st.columns([3, 2])
    with c1:
        slider_val = st.slider(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=slider_key,
            on_change=_from_slider,
            help=help,
        )
    with c2:
        st.number_input(
            "type",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            format=fmt,
            key=num_key,
            on_change=_from_num,
            label_visibility="collapsed",
            help=help,
        )
    val = float(slider_val)
    st.session_state[key] = val
    return val


def dual_log(
    label: str,
    *,
    key: str,
    min_value: float,
    max_value: float,
    default: float,
    fmt: str = "%.4g",
    help: str | None = None,
    log_step: float = 0.01,
) -> float:
    """Log-scaled slider + typed number input.

    The slider moves in log10-space but labels each stop with the true
    (linear) value via ``format_func``, so dragging shows decimals such as
    ``0.26`` rather than ``10^-0.585`` or raw log exponents.
    """
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("Log dual input requires positive bounds.")
    if log_step <= 0.0:
        raise ValueError("log_step must be positive.")
    default = float(np.clip(default, min_value, max_value))
    log_min = float(np.log10(min_value))
    log_max = float(np.log10(max_value))
    # Fresh key so a prior st.slider float under ``__log_slider`` cannot clash
    # with select_slider's discrete option list.
    log_key = f"{key}__log_select"
    num_key = f"{key}__num"

    log_options = [
        round(float(x), 2)
        for x in np.arange(log_min, log_max + 0.5 * log_step, log_step)
    ]
    if not log_options or log_options[-1] < log_max - 1e-9:
        log_options.append(round(log_max, 2))
    # Deduplicate while preserving order (arange + explicit max can repeat).
    deduped: list[float] = []
    for lg in log_options:
        if not deduped or abs(deduped[-1] - lg) > 1e-12:
            deduped.append(lg)
    log_options = deduped

    def _nearest_log(lg: float) -> float:
        return float(min(log_options, key=lambda opt: abs(opt - lg)))

    _ensure_key(key, default)
    st.session_state[key] = float(np.clip(st.session_state[key], min_value, max_value))
    _ensure_key(log_key, _nearest_log(float(np.log10(st.session_state[key]))))
    _ensure_key(num_key, float(st.session_state[key]))

    st.session_state[log_key] = _nearest_log(
        float(np.clip(float(st.session_state[log_key]), log_min, log_max))
    )
    st.session_state[num_key] = float(
        np.clip(st.session_state[num_key], min_value, max_value)
    )

    def _from_slider() -> None:
        st.session_state[key] = float(10.0 ** float(st.session_state[log_key]))
        st.session_state[num_key] = float(st.session_state[key])

    def _from_num() -> None:
        val = float(np.clip(st.session_state[num_key], min_value, max_value))
        st.session_state[key] = val
        st.session_state[num_key] = val
        st.session_state[log_key] = _nearest_log(float(np.log10(val)))

    def _format_log(lg: float) -> str:
        return _printf(fmt, 10.0 ** float(lg))

    c1, c2 = st.columns([3, 2])
    with c1:
        log_val = st.select_slider(
            label,
            options=log_options,
            format_func=_format_log,
            key=log_key,
            on_change=_from_slider,
            help=help,
        )
    with c2:
        st.number_input(
            "type",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(min_value),
            format=fmt,
            key=num_key,
            on_change=_from_num,
            label_visibility="collapsed",
            help=help,
        )
    val = float(10.0 ** float(log_val))
    st.session_state[key] = val
    return val


def sole_config_from_gui(params: dict) -> SoleConfig:
    """Measured-sole config from canonical GUI params (SI moduli, mm lengths)."""
    return SoleConfig(
        shoe_length_mm=float(params["shoe_length_mm"]),
        geometry_csv=str(params["geometry_csv"]),
        upper_foam_material=str(params["upper_foam_material"]),
        lower_foam_material=str(params["lower_foam_material"]),
        ffturbo_E_Pa=float(params["ffturbo_E_Pa"]),
        ffturbo_nu=float(params["ffturbo_nu"]),
        ffleap_E_heel_Pa=float(params["ffleap_E_heel_Pa"]),
        ffleap_E_toe_Pa=float(params["ffleap_E_toe_Pa"]),
        ffleap_nu=float(params["ffleap_nu"]),
        EI_plate_Nm2_per_m=float(params["EI_plate_Nm2_per_m"]),
        mesh_size_m=mm_to_m(float(params["mesh_size_mm"])),
        toe_refinement=float(params["toe_refinement"]),
        heel_corner_refinement=float(params["heel_corner_refinement"]),
        interface_refinement=float(params["interface_refinement"]),
        plate_end_refinement=float(params["plate_end_refinement"]),
        **{
            k: float(params[k])
            for k in ("landmark_tolerance", "curvature_max_turn_deg", "min_angle_deg")
            if k in params
        },
    )


def setup_from_gui(params: dict, toe_config=None, shoe_width_m: float | None = None):
    """Full measured-sole setup (JSON-config equivalent) from canonical GUI params."""
    from compliance_fem.contact.model_setup import (
        LookupConfig,
        ModelSetup,
        RuntimeConfig,
    )
    from compliance_fem.contact.toe_spring import ToeSpringConfig

    runtime = RuntimeConfig() if shoe_width_m is None else RuntimeConfig(shoe_width_m=float(shoe_width_m))
    return ModelSetup(
        sole=sole_config_from_gui(params),
        lookup=LookupConfig(
            toe_length_mm=float(params["softplus_toe_length"]) * 1000.0,
            min_bend_radius_mm=4.0 * float(params["shoe_length_mm"]) / float(params["softplus_kappa"]),
            reciprocity_tol=float(params["reciprocity_tol"]),
        ),
        toe_spring=toe_config or ToeSpringConfig(),
        runtime=runtime,
    )


def gui_params_from_setup(setup) -> dict:
    """Canonical GUI measured params (SI moduli, mm lengths) from a config setup."""
    s = setup.sole
    return {
        "geometry_csv": str(s.geometry_csv),
        "shoe_length_mm": float(s.shoe_length_mm),
        "upper_foam_material": s.upper_foam_material,
        "lower_foam_material": s.lower_foam_material,
        "ffturbo_E_Pa": float(s.ffturbo_E_Pa),
        "ffturbo_nu": float(s.ffturbo_nu),
        "ffleap_E_heel_Pa": float(s.ffleap_E_heel_Pa),
        "ffleap_E_toe_Pa": float(s.ffleap_E_toe_Pa),
        "ffleap_nu": float(s.ffleap_nu),
        "EI_plate_Nm2_per_m": float(s.EI_plate_Nm2_per_m),
        "mesh_size_mm": float(s.mesh_size_m) * 1000.0,
        "toe_refinement": float(s.toe_refinement),
        "heel_corner_refinement": float(s.heel_corner_refinement),
        "interface_refinement": float(s.interface_refinement),
        "plate_end_refinement": float(s.plate_end_refinement),
        "landmark_tolerance": float(s.landmark_tolerance),
        "curvature_max_turn_deg": float(s.curvature_max_turn_deg),
        "min_angle_deg": float(s.min_angle_deg),
        "softplus_toe_length": float(setup.toe_length_m),
        "softplus_kappa": float(setup.softplus_kappa),
        "reciprocity_tol": float(setup.lookup.reciprocity_tol),
    }
