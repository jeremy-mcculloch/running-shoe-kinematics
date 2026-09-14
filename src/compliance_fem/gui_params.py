"""Streamlit GUI parameter defaults and dual slider/number widgets.

Display units are chosen for human editing (mm, MPa, N·m², kN). Values
are converted to SI before they reach ``LayeredPlateConfig`` / the solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from compliance_fem.config import LayeredPlateConfig


@dataclass(frozen=True)
class GuiLayeredDefaults:
    """Defaults from ``examples/layered_plate_sample.py`` and contact_lookup CLI."""

    # Geometry (SI metres)
    L: float = 0.29
    h1_heel: float = 0.01
    h1_toe: float = 0.01
    h2_heel: float = 0.029
    h2_toe: float = 0.025
    # Softplus (CLI defaults)
    softplus_a: float = 0.2262
    softplus_kappa: float = 160.0
    # Materials (SI)
    E1: float = 2.6e5
    nu1: float = 0.113
    E_heel: float = 3.54e5
    E_toe: float = 2.07e5
    nu2: float = 0.113
    EI_plate: float = 2.0
    # Mesh (sample)
    nx: int = 100
    ny1: int = 12
    ny2: int = 12
    element_order: int = 1
    # Forces (SI newtons)
    Fx: float = 0.0
    Fy: float = -1.0e4

    @property
    def L_minus_a(self) -> float:
        return float(self.L - self.softplus_a)


DEFAULTS = GuiLayeredDefaults()

# Display ranges (converted to SI when applied).
H_MM_MIN = 0.5
H_MM_MAX = 50.0
L_MM_MIN = 200.0
L_MM_MAX = 400.0
LA_MM_MIN = 0.0
LA_MM_MAX = 200.0
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
FY_N_MIN = -1.0e4
FY_N_MAX = 0.0


def mm_to_m(mm: float) -> float:
    return float(mm) * 1.0e-3


def m_to_mm(m: float) -> float:
    return float(m) * 1.0e3


def pa_to_mpa(pa: float) -> float:
    return float(pa) * 1.0e-6


def mpa_to_pa(mpa: float) -> float:
    return float(mpa) * 1.0e6


def nmm2_to_si(ei_nmm2: float) -> float:
    """N·mm² → N·m²."""
    return float(ei_nmm2) * 1.0e-6


def si_to_nmm2(ei_si: float) -> float:
    return float(ei_si) * 1.0e6


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

    # Bounds can shrink (e.g. L−a vs L); clip existing widget state in place.
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


def build_layered_config_from_gui(
    *,
    L_m: float,
    h1_heel_m: float,
    h1_toe_m: float,
    h2_heel_m: float,
    h2_toe_m: float,
    E1: float,
    nu1: float,
    E_heel: float,
    E_toe: float,
    nu2: float,
    EI_plate: float,
    nx: int,
    ny1: int,
    ny2: int,
    element_order: int,
) -> LayeredPlateConfig:
    """Construct a validated layered config from GUI SI values."""
    return LayeredPlateConfig(
        L=float(L_m),
        h1_heel=float(h1_heel_m),
        h1_toe=float(h1_toe_m),
        h2_heel=float(h2_heel_m),
        h2_toe=float(h2_toe_m),
        E1=float(E1),
        nu1=float(nu1),
        E_heel=float(E_heel),
        E_toe=float(E_toe),
        nu2=float(nu2),
        EI_plate=float(EI_plate),
        nx=int(nx),
        ny1=int(ny1),
        ny2=int(ny2),
        element_order=int(element_order),
    )
