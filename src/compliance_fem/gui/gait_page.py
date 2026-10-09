"""Streamlit page: Gait force replay (passive toe-spring θ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from compliance_fem.contact.lookup import load_contact_lookup
from compliance_fem.gait.download import write_data_readme
from compliance_fem.gait.replay import replay_stance, save_replay_result
from compliance_fem.gui.gait_plots import plotly_foot_animation_figure
from compliance_fem.gait.wang_io import discover_trials
from compliance_fem.contact.toe_spring import (
    TOE_MODEL_PASSIVE_SPRING,
    TOE_MODELS,
    RearfootGeometry,
    RearfootToeModel,
    ToeSpringConfig,
    spring_dU_dalpha,
)


def _frame_q_alpha_curve(result, i: int, alpha: np.ndarray) -> np.ndarray:
    """Exact Q_alpha,shoe->foot(alpha) of the selected record (chord frame follows alpha)."""
    coeff = np.asarray(result.toe_Q_coeff, dtype=float)
    kin = result.model_info.get("toe_rearfoot_kinematics")
    if coeff.shape[0] <= i or kin is None or not np.all(np.isfinite(coeff[i])):
        return float(result.toe_q0_Nm[i]) + float(result.toe_q1_Nm[i]) * alpha
    model = RearfootToeModel(
        coeff=coeff[i],
        D=np.full((2, 5), np.nan),
        F_fixed=np.array([float(result.elastic_Fx[i]), float(result.elastic_Fy[i])]),
        phi_rearfoot=float(result.phi[i]),
        geometry=RearfootGeometry(
            L=kin["L"], toe_length=kin["toe_length"], phi1_mtp=kin["phi1_mtp"], dy_mtp=kin["dy_mtp"]
        ),
        kd_cond=float("nan"),
        kd_smin=float("nan"),
        well_conditioned=True,
    )
    return np.asarray(model.generalized_force(alpha), dtype=float)


def toe_equilibrium_figure(result, i: int, cfg: ToeSpringConfig):
    """Shoe-on-foot Q_alpha(θ) and spring dU/dα(θ) for the selected record at frame i."""
    import plotly.graph_objects as go

    k = float(result.toe_stiffness_Nm_per_rad)
    th0 = float(result.toe_neutral_angle_rad)
    th = np.linspace(cfg.theta_min_rad, cfg.theta_max_rad, 601)
    a = np.tan(th)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.rad2deg(th), y=_frame_q_alpha_curve(result, i, a), name="Q_α shoe→foot(θ)"))
    fig.add_trace(go.Scatter(x=np.rad2deg(th), y=spring_dU_dalpha(a, k, th0), name="spring dU/dα"))
    roots = result.toe_roots_deg[i] if result.toe_roots_deg.size else np.array([])
    roots = roots[np.isfinite(roots)]
    if roots.size:
        ar = np.tan(np.deg2rad(roots))
        fig.add_trace(
            go.Scatter(
                x=roots, y=spring_dU_dalpha(ar, k, th0), mode="markers",
                marker=dict(size=10, symbol="circle-open"), name="all roots",
            )
        )
    fig.add_vline(x=float(result.theta_deg[i]), line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"θ={float(result.theta_deg[i]):.2f}°")
    fig.update_layout(
        title="Toe equilibrium at this frame (selected record): roots where the curves cross",
        xaxis_title="θ (deg)", yaxis_title="N·m (total, per α)", height=380,
    )
    return fig


def toe_stance_figure(result):
    """θ, generalized forces, moments, residual, energy, and contact state over stance."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    t = result.times
    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True,
        subplot_titles=(
            "θ (deg)", "Q_α shoe→foot and spring −dU/dα (N·m)",
            "Q_θ shoe→foot and spring moment (N·m)", "|toe residual| (N·m)",
            "Spring energy (J)", "Contact state / low load",
        ),
    )
    fig.add_trace(go.Scatter(x=t, y=result.theta_deg, name="θ"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=result.Q_alpha_shoe_on_foot, name="Q_α"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=t, y=result.toe_spring_generalized_force_alpha, name="spring −dU/dα"), row=1, col=2
    )
    fig.add_trace(go.Scatter(x=t, y=result.Q_theta_shoe_on_foot, name="Q_θ"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=result.toe_spring_moment_Nm, name="M_spring,θ"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=np.abs(result.toe_equilibrium_residual_Nm) + 1e-300, name="|g|"), row=2, col=2
    )
    fig.update_yaxes(type="log", row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=result.toe_spring_energy_J, name="U"), row=3, col=1)
    type_map = {"heel": 0, "full": 1, "toe": 2}
    fig.add_trace(
        go.Scatter(x=t, y=[type_map.get(c, -1) for c in result.contact_type], mode="lines+markers",
                   name="contact (0 heel, 1 full, 2 toe)"),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=result.toe_low_load.astype(float) * 0.5, name="low load (0.5)",
                   line=dict(dash="dot")),
        row=3, col=2,
    )
    fig.update_layout(height=820, title="Passive toe spring over stance")
    return fig


def render_gait_replay_page() -> None:
    st.title("Gait replay (Wang dataset)")
    st.caption(
        "Default toe model: passive toe spring (k = 25 N·m/rad, θ0 = 0). θ solves "
        "Q_α,shoe→foot(α) = k(arctan α − θ0)/(1+α²) exactly for every contact record "
        "(no small-angle approximation, no per-frame FEM); measured COP is not "
        "prescribed and serves as validation. Legacy θ rules are opt-in. Forces are "
        "divided by shoe width (default 10 cm) for plane-strain per-metre lookup units; "
        "the toe generalized force is multiplied back by the width. F_top = −F_GRF."
    )

    data_root = Path(
        st.text_input("Dataset root", value="data/wang", key="gait_data_root")
    )
    if st.button("Write download README into dataset root"):
        path = write_data_readme(data_root)
        st.success(f"Wrote {path}")

    trials = discover_trials(data_root) if data_root.exists() else []
    if trials:
        stems = [t for t in trials if t.trc_path and t.mot_path]
        if not stems:
            st.warning(
                "Trials found but none have both TRC and MOT. "
                "Check that MOT files use a matching stem (…_force.mot is OK)."
            )
            stems = []
        labels = [f"{t.stem} (pr1={t.is_preferred_r1})" for t in stems]
        # Prefer a pr1 trial in the default index when available.
        default_idx = next((i for i, t in enumerate(stems) if t.is_preferred_r1), 0)
        idx = st.selectbox(
            "Trial",
            options=list(range(len(stems))) if stems else [0],
            format_func=lambda i: labels[i] if stems else "(none)",
            index=min(default_idx, max(len(stems) - 1, 0)) if stems else 0,
            key="gait_trial",
            disabled=not stems,
        )
        if stems:
            trial = stems[idx]
            trc_path = trial.trc_path
            mot_path = trial.mot_path
        else:
            trc_path = None
            mot_path = None
    else:
        st.info("No matched TRC/MOT pairs found. Enter paths manually.")
        trc_path = Path(st.text_input("TRC path", key="gait_trc"))
        mot_path = Path(st.text_input("MOT path", key="gait_mot"))

    lookup_path = Path(
        st.text_input(
            "Contact lookup path",
            value="outputs/contact_lookup",
            key="gait_lookup",
            help="Directory containing contact_lookup.npz, or the .npz file itself.",
        )
    )
    foot = st.selectbox("Foot", ["right", "left"], key="gait_foot")
    plate = st.text_input("Force plate", value="auto", key="gait_plate")
    stance_index = st.number_input("Stance index", min_value=0, value=0, step=1)
    visco = st.selectbox(
        "Visco model", ["elastic", "sls", "fractional", "fung"], key="gait_visco"
    )
    toe_model = st.selectbox(
        "Toe model",
        list(TOE_MODELS),
        index=list(TOE_MODELS).index(TOE_MODEL_PASSIVE_SPRING),
        key="gait_toe_model",
        help=(
            "passive_spring (default): exact passive toe-spring equilibrium, elastic only. "
            "passive_spring_elastic_equivalent: same against the visco elastic-equivalent "
            "wrench (approximation)."
        ),
    )
    defaults = ToeSpringConfig()
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        toe_k = st.number_input("k (N·m/rad)", value=defaults.toe_stiffness_Nm_per_rad, min_value=0.0)
    with s2:
        toe_th0 = st.number_input("θ0 (rad)", value=defaults.toe_neutral_angle_rad, format="%.4f")
    with s3:
        toe_min = st.number_input("θ bound min (deg)", value=defaults.toe_angle_min_deg)
    with s4:
        toe_max = st.number_input("θ bound max (deg)", value=defaults.toe_angle_max_deg)
    with s5:
        toe_low = st.number_input("Low-load |F| (N)", value=defaults.toe_low_force_threshold_N, min_value=0.0)
    with s6:
        shoe_width = st.number_input(
            "Shoe width (m)",
            value=0.10,
            min_value=0.01,
            step=0.01,
            format="%.3f",
            help="Experimental Fx,Fy,Mz are divided by this width for the unit-thickness plane-strain lookup.",
        )

    if st.button("Run replay", type="primary"):
        if trc_path is None or mot_path is None or not Path(trc_path).exists():
            st.error("TRC/MOT paths required")
            return
        if not Path(mot_path).exists():
            st.error(f"MOT not found: {mot_path}")
            return
        try:
            lookup = load_contact_lookup(lookup_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        try:
            toe_cfg = ToeSpringConfig(
                toe_model=toe_model,
                toe_stiffness_Nm_per_rad=float(toe_k),
                toe_neutral_angle_rad=float(toe_th0),
                toe_angle_min_deg=float(toe_min),
                toe_angle_max_deg=float(toe_max),
                toe_low_force_threshold_N=float(toe_low),
            )
            with st.spinner("Replaying stance…"):
                result = replay_stance(
                    lookup,
                    trc_path=trc_path,
                    mot_path=mot_path,
                    foot=foot,
                    force_plate=str(plate),
                    stance_index=int(stance_index),
                    toe_config=toe_cfg,
                    shoe_width_m=float(shoe_width),
                    visco_model=visco,
                )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["gait_replay_result"] = result
        st.session_state["gait_toe_cfg"] = toe_cfg
        st.session_state["gait_replay_lookup"] = lookup
        st.session_state.pop("gait_anim_fig", None)
        st.session_state.pop("gait_anim_key", None)
        out = Path("outputs/gait_replay_gui")
        save_replay_result(result, out)
        st.success(f"Saved results under {out}")

    result = st.session_state.get("gait_replay_result")
    if result is None:
        return

    for w in result.warnings:
        st.warning(w)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = result.times
    n_frames = int(t.size)

    st.subheader("Foot pose over stance (true scale)")
    scale_mode = "true"
    manual_scale = 1.0

    play_cols = st.columns([1.2, 1.2, 1.5])
    with play_cols[0]:
        play_fps = float(
            st.number_input(
                "Play rate (fps)",
                min_value=0.5,
                max_value=60.0,
                value=10.0,
                step=0.5,
                key="gait_play_fps",
                help="Client-side animation rate (Plotly; no page redraw between frames).",
            )
        )
    with play_cols[1]:
        loop_play = st.checkbox("Loop", value=True, key="gait_play_loop")
    with play_cols[2]:
        if n_frames >= 2:
            dt_med = float(np.median(np.diff(t)))
            realtime_fps = (1.0 / dt_med) if dt_med > 1e-12 else 10.0
            if st.button(
                f"Realtime (~{realtime_fps:.0f} fps)",
                key="gait_play_realtime",
                use_container_width=True,
                help="Set play rate from the median sample interval of this stance.",
            ):
                st.session_state["gait_play_fps"] = float(
                    min(60.0, max(0.5, realtime_fps))
                )
                st.rerun()

    lookup = st.session_state.get("gait_replay_lookup")
    if lookup is None:
        try:
            lookup = load_contact_lookup(lookup_path)
            st.session_state["gait_replay_lookup"] = lookup
        except FileNotFoundError:
            st.error(
                "Lookup not available for shape replay. Re-run with a valid contact lookup path."
            )
            lookup = None

    if lookup is not None:
        anim_key = (
            id(result),
            scale_mode,
            float(manual_scale),
            float(play_fps),
            bool(loop_play),
            str(lookup_path),
        )
        if (
            st.session_state.get("gait_anim_key") != anim_key
            or "gait_anim_fig" not in st.session_state
        ):
            with st.spinner("Building smooth shoe animation (one-time)…"):
                try:
                    st.session_state["gait_anim_fig"] = plotly_foot_animation_figure(
                        lookup,
                        result,
                        scale_mode=scale_mode,
                        manual_scale=manual_scale,
                        fps=play_fps,
                        loop=loop_play,
                    )
                    st.session_state["gait_anim_key"] = anim_key
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("gait_anim_fig", None)
                    st.session_state.pop("gait_anim_key", None)
                    st.error(f"Could not build foot animation: {exc}")
        fig_anim = st.session_state.get("gait_anim_fig")
        if fig_anim is not None:
            st.plotly_chart(
                fig_anim,
                use_container_width=True,
                key="gait_foot_animation",
            )
            st.caption(
                "Use the figure’s Play / Pause and slider for flicker-free playback "
                "(animation runs in the browser; traces update in place so the prior "
                "outline stays until the next frame is ready)."
            )

    @st.fragment
    def _inspect_and_diagnostics() -> None:
        if "gait_frame" not in st.session_state:
            st.session_state["gait_frame"] = 0
        st.session_state["gait_frame"] = int(
            np.clip(st.session_state["gait_frame"], 0, max(0, n_frames - 1))
        )
        i = st.slider(
            "Inspect frame (diagnostics below)",
            0,
            max(0, n_frames - 1),
            key="gait_frame",
            help=(
                "Chooses which frame’s metrics / Mz–θ plot to show. "
                "Runs in a fragment so it does not remount the animation above."
            ),
        )
        st.caption(
            f"t = {float(t[i]):.4f} s · stance {float(result.stance_percent[i]):.1f}% · "
            f"contact={result.contact_type[i]} · θ={float(result.theta_deg[i]):.2f}° · "
            f"φ={float(np.rad2deg(result.phi[i])):.2f}°"
        )

        st.write(
            {
                "t": float(t[i]),
                "stance_percent": float(result.stance_percent[i]),
                "phi_deg": float(np.rad2deg(result.phi[i])),
                "theta_deg": float(result.theta_deg[i]),
                "contact": result.contact_type[i],
                "Global COP x (m)": float(result.cop_lab_x[i]) if result.cop_lab_x.size else None,
                "Heel-relative COP x (m)": float(result.cop_foot_x[i]) if result.cop_foot_x.size else None,
                "Model-predicted COP x (m)": float(result.cop_model_x[i]) if result.cop_model_x.size else None,
                "cop_valid": bool(result.cop_valid[i]) if result.cop_valid.size else None,
                "force_residual": float(result.force_residual[i]),
                "status": result.selection_status[i],
            }
        )

        if result.Q_alpha_shoe_on_foot.size == n_frames:
            st.subheader("Passive toe spring (this frame)")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("θ (deg)", f"{float(result.theta_deg[i]):.3f}")
            m2.metric("α = tan θ", f"{float(result.alpha[i]):.4f}")
            m3.metric("Q_α shoe→foot (N·m)", f"{float(result.Q_alpha_shoe_on_foot[i]):.4g}")
            m4.metric("Q_θ shoe→foot (N·m)", f"{float(result.Q_theta_shoe_on_foot[i]):.4g}")
            m5.metric("Spring moment (N·m)", f"{float(result.toe_spring_moment_Nm[i]):.4g}")
            m6.metric("Spring energy (J)", f"{float(result.toe_spring_energy_J[i]):.4g}")
            st.write(
                {
                    "toe_model": result.toe_model,
                    "k (N·m/rad)": float(result.toe_stiffness_Nm_per_rad),
                    "neutral angle (rad)": float(result.toe_neutral_angle_rad),
                    "toe_equilibrium_residual_Nm": float(result.toe_equilibrium_residual_Nm[i]),
                    "toe_equilibrium_relative_residual": float(result.toe_equilibrium_relative_residual[i]),
                    "toe_root_count": int(result.toe_root_count[i]),
                    "toe_root_index": int(result.toe_root_index[i]),
                    "all roots θ (deg)": [
                        round(float(v), 4) for v in result.toe_roots_deg[i] if np.isfinite(v)
                    ]
                    if result.toe_roots_deg.size
                    else [],
                    "toe_equilibrium_tangent (Π'')": float(result.toe_equilibrium_tangent[i]),
                    "toe_equilibrium_stable": bool(result.toe_equilibrium_stable[i]),
                    "topology": result.contact_type[i],
                    "low_load": bool(result.toe_low_load[i]),
                    "toe_solve_status": result.toe_solve_status[i],
                }
            )
            cfg_view = st.session_state.get("gait_toe_cfg") or ToeSpringConfig()
            st.plotly_chart(toe_equilibrium_figure(result, i, cfg_view), use_container_width=True)
            st.plotly_chart(toe_stance_figure(result), use_container_width=True)

        if result.cop_lab_x.size == result.times.size:
            st.subheader("COP frames")
            fig_lab = go.Figure()
            fig_lab.add_trace(
                go.Scatter(
                    x=result.cop_lab_x,
                    y=result.cop_lab_y,
                    mode="lines",
                    name="Global COP",
                )
            )
            fig_lab.add_trace(
                go.Scatter(
                    x=result.heel_lab_x,
                    y=result.heel_lab_y,
                    mode="lines",
                    name="Heel (lab)",
                )
            )
            fig_lab.add_trace(
                go.Scatter(
                    x=result.fore_lab_x,
                    y=result.fore_lab_y,
                    mode="lines",
                    name="Forefoot (lab)",
                )
            )
            fig_lab.add_trace(
                go.Scatter(
                    x=[result.heel_lab_x[i], result.fore_lab_x[i]],
                    y=[result.heel_lab_y[i], result.fore_lab_y[i]],
                    mode="lines+markers",
                    name="Foot chord (frame)",
                    line=dict(color="#9467bd", width=3),
                )
            )
            fig_lab.update_layout(
                title="Laboratory frame: Global COP + markers",
                xaxis_title="lab x anterior (m)",
                yaxis_title="lab y superior (m)",
                yaxis_scaleanchor="x",
                height=420,
            )
            st.plotly_chart(fig_lab, use_container_width=True)

            fig_cop = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    "Heel-relative vs model-predicted COP x",
                    "Global COP x and heel lab x",
                ),
            )
            fig_cop.add_trace(
                go.Scatter(x=t, y=result.cop_foot_x, name="Heel-relative COP x"),
                row=1,
                col=1,
            )
            fig_cop.add_trace(
                go.Scatter(x=t, y=result.cop_model_x, name="Model-predicted COP x"),
                row=1,
                col=1,
            )
            if result.cop_valid.size:
                invalid = ~result.cop_valid
                if np.any(invalid):
                    fig_cop.add_trace(
                        go.Scatter(
                            x=t[invalid],
                            y=result.cop_foot_x[invalid],
                            mode="markers",
                            name="COP invalid (low Fy)",
                            marker=dict(color="#aaaaaa", size=5),
                        ),
                        row=1,
                        col=1,
                    )
            fig_cop.add_trace(
                go.Scatter(x=t, y=result.cop_lab_x, name="Global COP x"),
                row=2,
                col=1,
            )
            fig_cop.add_trace(
                go.Scatter(x=t, y=result.heel_lab_x, name="Heel global x"),
                row=2,
                col=1,
            )
            fig_cop.add_vline(x=float(t[i]), line=dict(color="#444", dash="dot"), row=1, col=1)
            fig_cop.add_vline(x=float(t[i]), line=dict(color="#444", dash="dot"), row=2, col=1)
            fig_cop.update_layout(height=520, title="COP time histories (explicit frames)")
            st.plotly_chart(fig_cop, use_container_width=True)

        st.subheader("Replay diagnostics")
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            subplot_titles=(
                "Vertical force (model top, N/m)",
                "Heel-relative COP x (m)",
                "Moments (heel about 0 / toe about L − toe length)",
                "φ and θ",
                "Residuals",
                "Contact type",
            ),
        )
        fig.add_trace(go.Scatter(x=t, y=result.elastic_Fy, name="Fy_e"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=result.predicted_Fy, name="Fy_pred"), row=1, col=1)
        if result.cop_foot_x.size:
            fig.add_trace(
                go.Scatter(x=t, y=result.cop_foot_x, name="Heel-relative COP x"),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(x=t, y=result.cop_model_x, name="Model-predicted COP x"),
                row=1,
                col=2,
            )
        if result.measured_moment_about_model_anchor.size:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=result.measured_moment_about_model_anchor,
                    name="M meas (top≡−GRF about heel)/w",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=result.predicted_moment_about_model_anchor,
                    name="M pred (lookup Mz about 0)",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
        else:
            fig.add_trace(
                go.Scatter(x=t, y=result.elastic_Mz, name="Mz_e"),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=t, y=result.predicted_Mz, name="Mz_pred"),
                row=2,
                col=1,
                secondary_y=False,
            )
        if result.predicted_toe_moment.size == t.size:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=result.predicted_toe_moment,
                    name="T_toe pred (about L − toe length, H_mtp)",
                    line=dict(color="#2ca02c", dash="dash"),
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        fig.update_yaxes(title_text="Mz (N·m/m)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="T_toe (N·m/m)", row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=t, y=np.rad2deg(result.phi), name="φ (deg)"), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=result.theta_deg, name="θ (deg)"), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=result.force_residual, name="|F| res"), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=result.cop_residual, name="|COP| res"), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=result.moment_residual, name="|M| res"), row=3, col=1)
        type_map = {"heel": 0, "full": 1, "toe": 2}
        fig.add_trace(
            go.Scatter(
                x=t,
                y=[type_map.get(c, -1) for c in result.contact_type],
                name="contact",
                mode="lines+markers",
            ),
            row=3,
            col=2,
        )
        ti = float(t[i])
        for r in (1, 2, 3):
            for c in (1, 2):
                fig.add_vline(
                    x=ti,
                    line=dict(color="#444444", width=1, dash="dot"),
                    row=r,
                    col=c,
                )
        fig.update_layout(height=800, title="Gait replay diagnostics (SI)")
        st.plotly_chart(fig, use_container_width=True)

    _inspect_and_diagnostics()
