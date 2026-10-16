# Gait force replay (Wang et al. dataset)

Chronological open-loop replay of measured sagittal ground-reaction forces
through the viscoelastic mapper and the continuous unilateral contact model
([continuous_contact.md](continuous_contact.md)). Each sample prescribes the
fixed-frame top load \((F_x^\star,F_y^\star)\) and the measured rearfoot pitch
\(\phi\); the contact set (which bottom nodes touch the ground) and the
reactions are outputs of one warm-started complementarity solve, not a choice
from a table. By default the **toe angle \(\theta\) is solved from a passive
toe spring** (\(k=25\) N·m/rad, \(\theta_0=0\)) in exact quasistatic
equilibrium with the shoe's generalized force; see
[passive_toe_spring.md](passive_toe_spring.md). No FEM solve or
refactorization is performed per gait frame.

Implementation: `src/compliance_fem/gait/replay.py` (`replay_stance`,
`save_replay_result`), with the GUI page in `src/compliance_fem/gui/gait_page.py`.

## Citation

Yuan Wang et al., *Dataset of walking and running biomechanics with different
step widths across different speeds*, Scientific Data 12, 802 (2025).

- Article: https://doi.org/10.1038/s41597-025-05113-6
- TRC: https://doi.org/10.17608/k6.auckland.27015298
- MOT: https://doi.org/10.17608/k6.auckland.27015517

Replays run from the **Gait replay** page of the GUI
(`streamlit run src/compliance_fem/gui/app.py`). Set the dataset root (default
`data/wang`; **Write download README into dataset root** lists the files to fetch),
download matched TRC/MOT from Figshare (browser may be required), then pick the
trial, foot, force plate, stance index, viscoelastic model, toe spring
(\(k\), \(\theta_0\), angle bounds, low-load threshold) and shoe width. Each run
writes `gait_replay.csv`, `gait_replay.npz` and `model_info.json` to
`outputs/gait_replay_gui`.

Check the stance foot: for `P4 pr1 01` the first stance on the plate is the
**left** foot; the right foot yields an implausible
\(\phi\in[-110^\circ,-8^\circ]\).

Preferred-speed 3.0 m/s running trials use codes containing `pr1` (confirm exact
filenames on disk; do not hard-code one subject).

## Coordinate and force signs

Wang/OpenSim TRC/MOT after their transform: **+x anterior, +y superior, +z right**.
Markers/COP are mm (converted to m); forces N; moments often N·mm (→ N·m).

Shoe model fixed frame: **+x heel→toe, +y up**. Force-control top resultants are
the load applied **to the shoe top** (foot → shoe). Mapping:

\[
\mathbf F_{\mathrm{top}} \approx -\mathbf F_{\mathrm{GRF}}^{\text{ground-on-body}}.
\]

An upward measured GRF therefore produces compressive (negative \(F_y\)) top
loading. The measured moment follows the same action–reaction sign: it is the
top-of-shoe moment about the lab heel marker,
\(-(\mathbf r_{\mathrm{COP}/\mathrm{heel}}\times\mathbf F_{\mathrm{lab}}+T_z)\),
which matches the model moment about the heel-bottom reference point.

**Width scaling:** the contact model is plane-strain with
**unit out-of-plane thickness**, so stored responses are per metre width
(N/m, N·m/m). Experimental totals are converted by dividing by an effective
shoe width (default \(w=0.10\,\mathrm{m}\)):

\[
(F_x,F_y,M)_ {\mathrm{model}} = (F_x,F_y,M)_{\mathrm{exp}} / w.
\]

COP is unchanged (length). Override with the `shoe_width_m` argument of
`replay_stance` or the GUI control.

## Center of pressure (frames and units)

The Wang **MOT** COP (`*_force_px/py/pz`) is a **global laboratory / OpenSim
ground-frame** point (+x anterior, +y superior, +z right). It is **not**
heel-relative or shoe-relative.

Processed OpenSim-ready TRC/MOT files in this workspace declare **metres** and
**N·m**; the paper’s raw CSV uses **mm** / **N·mm** and a different CoP axis
order (`Cx` ML, `Cy` AP, `Cz` vertical). The importers (and `replay_stance`)
take `position_units = auto|m|mm` and `moment_units = auto|N-m|N-mm` and
record the resolved units in `model_info.json`. Conversion to SI is applied
**once**.

Heel-relative COP used for validation:

\[
\mathbf p_{\mathrm{COP}}^{\mathrm{foot}}
=
Q(\phi)^\mathsf T
\bigl(\mathbf p_{\mathrm{COP}}^{\mathrm{lab}}-\mathbf r_{\mathrm{heel}}^{\mathrm{lab}}\bigr).
\]

Measured COP is transformed into the heel-attached foot frame for diagnostics
only; it does not enter \(\theta\) or the contact solve. The model-predicted
COP `cop_model_x` \(\approx M_z/F_y\) about the heel-bottom reference is compared
with it (`COP_residual`, `model_info.cop_validation`) on frames with enough
vertical force (`cop_fy_min`, default 20 N). The contact solve also reports
the reaction-weighted ground COP in the fixed frame (`cop_fixed_x`,
`cop_heel_relative_x`).

## How each sample is solved

For every sample, in time order:

1. The viscoelastic map turns the measured \((F_x,F_y)\) into the
   elastic-equivalent load (identity for `elastic`).
2. `contact.continuous_toe.solve_toe_contact` solves the passive toe spring
   and the contact problem together. The marker pitch \(\phi\) is the rearfoot
   (heel → MTP) angle, so the chord frame rotates with the toe,
   \(\varphi(\alpha)=\phi-\operatorname{atan2}(\Delta y_{\mathrm{mtp}}+\alpha\varphi_1(x_{\mathrm{mtp}}),L-\ell_{\mathrm{toe}})\),
   with \(x_{\mathrm{mtp}}=L-\ell_{\mathrm{toe}}\) and \(\ell_{\mathrm{toe}}\) the toe length.
   Each evaluation of the toe residual
   \(g=Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}(\alpha)-k(\arctan\alpha-\theta_0)/(1+\alpha^2)\)
   runs one converged contact solve at \((\varphi(\alpha),\alpha)\), and Brent's
   method finds \(\theta=\arctan\alpha\) within the angle bounds. The
   rearfoot-fixed generalized force
   \(Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}=-w\,\psi^\mathsf T\mathbf f_{t,y}\)
   (\(\psi\approx\max(0,x-x_{\mathrm{mtp}})\), so \(Q_\alpha\approx-M_{\mathrm{toe}}\)) makes a
   negative foot-on-shoe toe moment give a positive (dorsiflexing) \(\theta\).
3. The contact solve is a mixed linear complementarity problem for the nodal
   normal reactions, \(0\le\lambda\perp g\ge0\) with \(\sum\lambda=-F_y^\star\),
   solved by a warm-started primal-dual active-set method. Tangential contact
   is **proportional shear**: each active node's ground force is
   \((\mu_{\mathrm{eff}},1)\,\lambda_k\) with \(\mu_{\mathrm{eff}}=F_x^\star/F_y^\star\), so
   \(F_x\) and \(F_y\) are reproduced exactly. Nodes enter and leave contact
   continuously through \(g=\lambda=0\).

The previous sample's \(\theta\) and active set warm-start the next sample
(continuation), so the replay follows one continuous branch; there is no
candidate list, no interval search and no sequence selector. Below
`toe_low_force_threshold_N` the toe search restarts from \(\theta_0\). When a
marker angle is missing the last finite \(\phi\) is held. The measured
COP / moment is not prescribed; it is a validation residual. The default
`passive_spring` toe model requires `visco_model = "elastic"` (or the explicit
`passive_spring_elastic_equivalent` approximation). Full derivations:
[passive_toe_spring.md](passive_toe_spring.md),
[continuous_contact.md](continuous_contact.md).

Anatomical MTP angle is **not** identical to the model's distributed softplus
coordinate.

## Outputs

`save_replay_result` writes:

- `gait_replay.csv`: one row per sample. Loads (`measured_*_ve`,
  `elastic_equivalent_*`, `predicted_Fx/Fy/moment`, residuals), kinematics
  (`phi`, `varphi` = chord rotation at the solved root, `alpha`, `theta`),
  COP frames, `frame_valid`, `status`, `contact_type`, the contact span
  (`contact_start_index` / `contact_end_index` = first / last active bottom
  node, -1 without contact; `contact_start_x` / `contact_end_x`), the
  display-only sub-element edge estimates `contact_edge_estimate_start_x` /
  `_end_x`, `contact_patches` (`"a-b;c-d"` node ranges), the per-frame contact
  scalars (`min_gap`, `min_reaction`, `max_complementarity`,
  `max_penetration`, `max_tension`, `n_active`, `n_patches`, `mu_eff`,
  `cop_fixed_x`, `cop_heel_relative_x`, equilibrium residuals,
  `active_set_iterations`, `active_block_rcond`, …), timings
  (`time_contact_solves_s`, `time_toe_outer_s`, `time_total_s`) and the toe
  columns listed in [passive_toe_spring.md](passive_toe_spring.md#outputs).
- `gait_replay.npz`: the same columns plus the nodal arrays `gap`,
  `normal_reaction`, `tangential_reaction` (fixed frame, ground on shoe, per
  metre width) and `active`, each `(n_frames, N_b)`, and `toe_roots_deg`.
- `model_info.json`: sources and file hashes, units, frame and sign
  definitions, toe configuration and equations, `contact_formulation`,
  `tangential_contact_law`, runtime per sample, `cop_validation` /
  `moment_validation`, `contact_summary` (label counts, invalid and
  multiple-patch frames, maximum penetration / tension / complementarity /
  force residual, active-set iterations, time per sample) and `toe_summary`;
  plus the warnings.

`contact_type` is a reporting label of the active span: `none`, `heel` (first
active node is the heel), `toe` (last is the toe tip), `full` (both) or
`interior`. Disconnected contact is reported, not merged: `n_patches > 1` and
the status carries `+multiple_contact_patches`.

`status` is `continuous:<toe status>` (`continuation`, `ok`, `multiple_roots`,
`no_root_in_bounds`, optionally `+low_load`), with `+<lcp status>` when the
contact solve did not converge, `+contact_tolerance_exceeded` when the
penetration or tension tolerances are exceeded, `+multiple_contact_patches`,
and `+large_shear_ratio` when \(|\mu_{\mathrm{eff}}|>2\). `frame_valid` is true when
the toe root and the contact solve both converged.

**Performance.** At the 3 mm measured mesh (\(N_b=126\)) a sample takes about
2 ms (a warm toe solve needs about 6–8 contact solves), and a 300-frame stance
replays in about 0.6 s.

### 2.5D out-of-plane stiffness

Each converged frame also stores the auxiliary 2.5D tangent stiffnesses
computed from that frame's state (`replay_stance(..., out_of_plane=OutOfPlaneSettings(...))`;
default: constant width `shoe_width_m`, thin-layer `η_t` from the contact-foam
ν): `lateral_translation_stiffness` (N/m), `roll_stiffness` and
`yaw_stiffness` (N·m/rad), `oop_stiffness_status`, `oop_active_contact_node_count`,
`oop_yaw_reference_x`, `oop_condition_number`, `oop_near_active_set_change`
(NaN / `invalid_frame` for unconverged frames). CSV and NPZ columns carry the
units in their names (`lateral_translation_stiffness_N_per_m`, ...);
`gait_cycle_percent` is the stance-normalized percent. The gait page plots the
three histories (time or stance %, NaN gaps) and exports them as CSV.
`load_replay_arrays` reads older replay files with N/A stiffnesses. See
[out_of_plane_stiffness.md](out_of_plane_stiffness.md).

## Viscoelasticity

The stateful backends map the measured force componentwise:

\[
[F_x,F_y]_{VE}(t)\;\longrightarrow\;[F_x,F_y]_e(t).
\]

The moment is not mapped; it is not prescribed and remains a diagnostic. This
matches the correspondence principle for a **scalar, spatially uniform**
relaxation kernel. It is only an effective approximation for heterogeneous
layers, changing contact domains, or nonlinear QLV. Use `visco_model =
"elastic"` for an identity map. The default passive toe spring rejects
non-elastic models (no automatic fallback); see the viscoelastic-extension
notes in [passive_toe_spring.md](passive_toe_spring.md). Usage of the
backends: [viscoelasticity.md](viscoelasticity.md).

## Limitations

- Open-loop: GRF is prescribed; the model cannot predict a changed external GRF.
- Experimental shoe ≠ modelled geometry/materials.
- Mediolateral force is omitted (warning if large).
- Tangential contact is proportional shear (full slip at the measured shear
  ratio), not a stick/slip Coulomb law; see
  [continuous_contact.md](continuous_contact.md).
- The passive toe coordinate is the distributed softplus shape amplitude, not a
  rigid anatomical MTP rotation; see [passive_toe_spring.md](passive_toe_spring.md).
- C3D import is not implemented (use TRC+MOT).
