# 2.5D out-of-plane tangent stiffness

The sole model is a **2D sagittal** plane-strain FEM. For use in 3D
musculoskeletal models, every converged evaluation also reports three
auxiliary, linearized **out-of-plane** stiffnesses around the current state.
They are a 2.5D transverse-strip approximation, **not** a 3D FEM response, and
they never feed back into anything: the sagittal contact solve, active set,
passive toe angle, deformation, plotted sole and replay solution are identical
with the calculation on or off.

Code: `contact/out_of_plane.py` (`OutOfPlaneSettings`,
`compute_out_of_plane_stiffness`, `OutOfPlaneStiffness`); called by
`api.SoleModel.step`, `gait.replay.replay_stance` and the GUI on the same
converged state.

## Degrees of freedom, loads, signs, units

Axes: x heel → toe, y up, z mediolateral (right-handed).

| dof | conjugate | stiffness | units |
|---|---|---|---|
| lateral translation `δz` | `δF_z` | `k_z = δF_z/δz` | N/m |
| roll `δφ_x` (about x; inversion/eversion) | `δM_x` | `k_roll = δM_x/δφ_x` | N·m/rad |
| yaw `δφ_y` (about y; internal/external) | `δM_y` | `k_yaw = δM_y/δφ_y` | N·m/rad |

A roll lifts a point at `z` by `δu_y = z δφ_x`; a yaw moves a point at
horizontal lever `r = x − x_yaw` by `δu_z = −r δφ_y`. Only the diagonal is
reported. `OutOfPlaneStiffness.as_matrix()` returns the 3×3 tangent in the
order `(z, φ_x, φ_y)` with NaN off-diagonal terms: the couplings are not derived
yet.

## Assumptions

* **Transverse strips.** Each active bottom node carries a strip of width
  `b_i`; every out-of-plane slice behaves like the 2D section and slices do not
  interact.
* **Frozen active set.** Infinitesimal out-of-plane perturbations do not
  change the converged sagittal active set `A`. The values are therefore
  **local, one-sided tangent stiffnesses**: they are piecewise smooth and can
  jump when the sagittal active set changes (a node entering or leaving
  contact). On the 3 mm measured mesh the contact edge crosses a node every
  2–3 gait samples, so the histories are inherently piecewise.
* **Normal tangent.** `K_n = H_AA⁻¹`, with `H_AA` the **pure normal** gap
  compliance of the active nodes (`μ_eff = 0`, symmetric positive definite,
  per metre width): `δλ_A = K_n δd_n,A`, positive indentation → positive
  compressive reaction. `H_AA` is Cholesky-factored once per evaluation; no
  inverse is formed.
* **Why unaugmented `H_AA` is the consistent tangent.** The sagittal solve is
  force-controlled (`Σλ = −F*_y`, with `T_y` its multiplier) and coupled to the
  passive toe angle. The roll indentation `z δφ_x` is antisymmetric across
  every strip, so it adds no net vertical force (the force constraint does not
  bind) and, to first order, no toe generalized force (`Q_α` comes from the
  in-plane top forces). Lateral translation and yaw act on lateral dofs that
  the sagittal model does not have. Neither `T_y` nor `θ` couples in.
* **Lateral shear (`out_of_plane_shear_model = "normal_stiffness_ratio"`).**
  The sagittal FEM has no lateral shear compliance, so `K_t = η_t K_n` with
  `η_t = lateral_to_normal_stiffness_ratio ≥ 0` an **elastic** stiffness ratio,
  not a Coulomb friction coefficient (it says nothing about slip). Default when
  not set: the thin-layer ratio of shear to constrained-normal modulus of the
  contact (lower) foam, `η_t = G/M = (1 − 2ν)/(2(1 − ν))` (≈ 0.436 for
  `ν = 0.113`), reported as an assumption (`lateral_ratio_source`).

## Width and the plane-strain convention

The FEM is plane strain **per metre of out-of-plane width** (reference
extrusion width 1 m; `fem_width_convention` in the metadata). The physical
width enters **exactly once**, here. `OutOfPlaneSettings.shoe_width_m` is a
constant width (default: the run's `shoe_width_m`, the same width that scales
the sagittal loads); `shoe_width_x` / `shoe_width_values` (metres, strictly
increasing x) give a heel → toe profile interpolated at the active nodes'
reference x. The sagittal solve itself always uses the constant
`shoe_width_m`; a profile only weights the 2.5D values.

With node widths `b_i` the quadratic forms are

    k_z    = η_t · (√b)ᵀ        H_AA⁻¹ (√b)
    k_yaw  = η_t · (√b ∘ r)ᵀ    H_AA⁻¹ (√b ∘ r)
    k_roll =       (√(b³/12))ᵀ  H_AA⁻¹ (√(b³/12))

i.e. the physical normal tangent `K_phys = diag(√b) K_n diag(√b)` and the roll
weight `W_roll = diag(b²/12)`. For a constant width this is exact under the
strip assumption: `k_z = η_t b 1ᵀK_n 1`, `k_yaw = η_t b rᵀK_n r` and
`k_roll = (b²/12)·k_vertical` with `k_vertical = b 1ᵀK_n 1` (strip integrals
`∫ dz = b`, `∫ z² dz = b³/12`). For a varying width it is the geometric-mean
strip approximation. Doubling a constant width doubles `k_z` and `k_yaw` and
multiplies `k_roll` by 8 (4 at a fixed physical vertical tangent).

## Yaw reference

`yaw_reference = "center_of_pressure"` (default) uses the sagittal centre of
pressure `x_cop = Σλ x^F / Σλ` when the normal load exceeds the reaction
tolerance; `"shoe_reference"` uses the material point `(fixed_yaw_reference_x, 0)`
(default `L/2`) carried rigidly by the frame. Levers `r_i = x^F_i − x_yaw` are
fixed-frame horizontal distances, so translating everything together leaves
`k_yaw` unchanged. Without load the COP mode falls back to the shoe reference
(with a warning). The yaw axis used (`yaw_reference_x`, fixed frame;
`yaw_reference_material_x`) and its source are stored with every evaluation.
`k_yaw` is quadratic in the reference and smallest at the stiffness centroid
`1ᵀK x / 1ᵀK 1`.

## Status and diagnostics

| status | values |
|---|---|
| `ok` | finite |
| `no_contact` | 0, 0, 0 (no active node) |
| `ill_conditioned` | finite; condition estimate of `H_AA` above `condition_number_warning` (1e10) |
| `tangent_singular` | NaN (condition estimate ≥ 1e14) |
| `tangent_not_positive_definite` | NaN (Cholesky failed) |
| `lateral_ratio_unavailable` | `k_z`, `k_yaw` NaN (no `η_t` and no foam ν) |
| `disabled` | NaN |

Also returned: condition number (`dpocon` on the Cholesky factor), symmetry
error and minimum eigenvalue of `H_AA`, `tangent_kind`, `augmented = False`,
`regularized = False` (nothing is regularized), the width summary (mode,
min / max / mean), and `near_active_set_change` (smallest active reaction
≤ 0.5 % of the peak, or smallest inactive gap ≤ 0.5 % of the node spacing).
Histories are never smoothed.

## Where the values appear

* `SoleModel.step(...)` → `SoleState.out_of_plane_stiffness` (None for an
  unconverged step), plus `diagnostics["sagittal_solve_time_s"]`,
  `["out_of_plane_stiffness_time_s"]`, `["total_step_time_s"]`.
* Main GUI: sidebar "2.5D out-of-plane stiffness" settings (enable, width
  source / profile in mm, `η_t` mode, yaw reference, fixed x, condition
  threshold) and a "2.5D out-of-plane tangent stiffness" results section
  (`k_z` in kN/m, roll and yaw in N·m/rad, active nodes, width, `η_t`, yaw
  axis, status). The plotted sole is unchanged; no out-of-plane displacement or
  3D view is drawn. Changing these settings never rebuilds the FEM.
* Gait replay: per-frame arrays `lateral_translation_stiffness`,
  `roll_stiffness`, `yaw_stiffness`, `oop_stiffness_status`,
  `oop_active_contact_node_count`, `oop_yaw_reference_x`,
  `oop_condition_number`, `oop_near_active_set_change` (NaN /
  `invalid_frame` for unconverged frames); CSV / NPZ columns
  `lateral_translation_stiffness_N_per_m`, `roll_stiffness_Nm_per_rad`,
  `yaw_stiffness_Nm_per_rad`, `oop_*`, `gait_cycle_percent` (stance-normalized
  percent); settings in `model_info["out_of_plane"]`. The gait page plots the
  three histories against time or stance percent (NaN gaps, invalid and
  near-change samples marked) and exports them as CSV; the plots read the
  stored values and never recompute them. `gait.replay.load_replay_arrays`
  reads older replay files with `N/A` (NaN, `not_available`) stiffnesses.

## Cost

One Cholesky factorization of the `a × a` active block (`O(a³)`, `a` ≈ 50–100)
plus three triangular solves (`O(a²)`); no FEM factorization, no contact
re-solve, no interval enumeration. Measured on the 3 mm sole: median 0.28 ms
per gait sample against 2.3 ms for the sagittal contact and toe solve.

## Limitations

* 2.5D strips: no transverse coupling between slices, no transverse variation
  of the sagittal contact topology, small roll and yaw angles.
* Lateral shear is a stiffness ratio of the normal tangent, not a modelled
  shear field; contact slip is not represented.
* Diagonal only; coupling terms (e.g. `k_z`–`k_yaw`) are not derived.
* A varying width profile weights the strips but does not change the 2D
  sagittal solve, which uses one constant width.
