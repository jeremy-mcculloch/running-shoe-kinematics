# Driving the sole model from another codebase

`compliance_fem.api.SoleModel` wraps a precomputed contact model behind a
single `step()` call. The caller supplies the load on the shoe and the rearfoot
pitch; the model solves the continuous unilateral ground contact together with
the passive toe-spring equilibrium (the toe angle and the set of bottom nodes
in contact are both **outputs**), and returns nodal positions, displacements,
forces and moments. No FEM is solved at runtime; only the shared boundary
operators stored in `contact_lookup.npz` are used
([continuous_contact.md](continuous_contact.md)).

```python
from compliance_fem.api import SoleModel

# lookup.output_dir written by `compliance-fem-contact-lookup configs/setup.json`
model = SoleModel("outputs/measured_sole_lookup", shoe_width_m=0.10)  # load once

for Fx, Fy, phi in external_frames:            # N, N, degrees
    st = model.step(Fx, Fy, phi)
    if not (st.valid and st.admissible):
        log_warning(st.status)                 # unconverged or outside the contact tolerances
    send(st.top_xy_m, st.bottom_xy_m, st.Mz_fixed_Nm, st.theta_deg)

model.reset()                                  # before an unrelated sequence
```

`SoleModel` also accepts an in-memory `ContactModel`
(`compliance_fem.contact.contact_model`), a custom `ToeSpringConfig`
(stiffness, neutral angle, angle bounds, root-scan points, low-load
threshold), and the `ContactTolerances` used by gait replay.

For the measured carbon-plated sole, prefer building from the JSON config.
It supplies the model location, shoe width and toe spring, and it checks that
the stored model was generated from the same parameters (see
[measured_sole.md](measured_sole.md#configuration-file)):

```python
model = SoleModel.from_config("configs/setup.json", build_if_missing=True)
model.setup.toe_length_m, model.setup.sole.EI_plate_Nm2_per_m   # the parsed configuration
```

## Inputs

| argument | meaning |
|----------|---------|
| `Fx_N`, `Fy_N` | fixed-frame force **applied by the foot on the shoe top** (`F_top = -GRF`), whole shoe, N. Stance loading has `Fy_N < 0`. |
| `phi_deg` | fixed-frame heel -> MTP (rearfoot) angle, counterclockwise positive (toe up), degrees |
| `Mz_Nm` (optional) | measured top moment about the heel-bottom reference point; not prescribed, only fills `moment_residual_Nm` |

Fixed frame: x forward (heel -> toe), y up, flat ground at y = 0. The model is
2D plane strain per metre width, so loads are divided by `shoe_width_m` and
forces / moments multiplied back; outputs are whole-shoe values.

Ground contact is normal unilateral (positive gap = open, positive normal
reaction = compressive) with **proportional shear**: every active node carries
the tangential force `mu_eff = Fx_N / Fy_N` times its normal reaction, so both
force components are reproduced exactly. The horizontal frame translation is
a gauge fixed so the frame point `(L/2, 0)` keeps its reference abscissa.

## Outputs (`SoleState`)

| field | meaning |
|-------|---------|
| `valid` | the toe root and the contact solve both converged |
| `admissible` | the contact solve converged and penetration / tension are within the scale-aware tolerances |
| `status` | `continuous:<toe status>` (`continuation`, `ok`, `multiple_roots`, `no_root_in_bounds`, optionally `+low_load`) |
| `theta_deg` | toe angle from the passive spring equilibrium |
| `chord_rotation_rad` | rotation of the shoe frame relative to the reference shoe |
| `contact_interval` | `(first, last)` active bottom node, or `None` without contact. With disconnected contact this span includes free nodes; use `contact_patches`. |
| `contact_patches` | connected runs of active nodes, `((start, end), ...)` heel -> toe |
| `contact_start_x_m`, `contact_end_x_m` | reference x of the first / last active node |
| `contact_mask` | active bottom nodes, `(n_bottom,)` bool |
| `bottom_gap_m` | fixed-frame ground gap of every bottom node, `(n_bottom,)` |
| `centre_of_pressure_x_m` | fixed-frame x of the reaction-weighted ground centre of pressure (NaN without load) |
| `top_xy_m`, `bottom_xy_m` | deformed nodal positions, fixed frame, `(2, n)` |
| `top_displacement_m`, `bottom_displacement_m` | nodal displacements in the rotating shoe frame, `(2, n)` |
| `top_force_N` | foot-on-shoe nodal forces, fixed frame |
| `bottom_reaction_N` | ground-on-shoe nodal reactions, fixed frame, rows (tangential, normal); zero on free nodes |
| `Fx_N`, `Fy_N` | reproduced resultant (equals the input when `force_residual_N` ~ 0) |
| `Mz_shoe_Nm` | sum(x f_y - y f_x) of top forces, shoe-frame components and reference coordinates, about the heel-bottom reference point (gait replay convention) |
| `Mz_fixed_Nm` | the same moment from fixed-frame forces and deformed positions about the ground origin |
| `toe_moment_Nm` | toe-hinge moment of the top forces about the MTP point |
| `toe_spring_moment_Nm` | spring moment -k(θ - θ0) at the solution |
| `force_residual_N`, `moment_residual_Nm` | resultant mismatch; mismatch with the optional measured `Mz_Nm` |
| `toe_root_count`, `toe_stable`, `low_load` | toe-equilibrium diagnostics |
| `diagnostics` | minimum gap, minimum active normal reaction (N/m), maximum complementarity, toe residual, toe roots, active-set and toe iterations, active-block reciprocal condition, `mu_eff`, solve time, and `sagittal_solve_time_s` / `out_of_plane_stiffness_time_s` / `total_step_time_s` |
| `out_of_plane_stiffness` | auxiliary 2.5D `OutOfPlaneStiffness` (lateral `k_z` N/m, roll and yaw N·m/rad, status, yaw axis, width, `η_t`, condition); `None` for an unconverged step; never affects the other fields ([out_of_plane_stiffness.md](out_of_plane_stiffness.md)). Configure with `SoleModel(..., out_of_plane=OutOfPlaneSettings(...))` |

Node order: top nodes heel -> toe along the top surface (`model.x_top`,
`model.y_top` give reference coordinates), bottom nodes heel -> toe along the
sole (`model.x_bottom`, `model.y_bottom`). `SoleState.to_dict()` returns a
JSON-serializable copy. If the toe solve returns no finite angle, every array
of the state is NaN and `valid` is false.

## State and performance

The model remembers the previous toe angle and active contact set, so
consecutive steps stay on one continuous branch (root continuation in the toe
solve, warm-started active set in the contact solve). Call `reset()` between
independent sequences.

At the 3 mm measured mesh (`N_b = 126`) the model file is about 19.4 MB. A
warm `step()` takes about 2 ms (about 6–8 contact solves); a cold step (the
first after `reset()`, which scans the toe-angle bounds) takes about 40 ms.

## Non-Python callers

Keep one long-lived Python process that owns the `SoleModel` and exchange
`SoleState.to_dict()` over HTTP, ZeroMQ or gRPC. Do not reload the model per
request.

## Limitations

Elastic only: the viscoelastic mapper used by gait replay is not applied
inside `step()`; pass elastic-equivalent loads if needed. Flat rigid ground,
node-to-ground unilateral contact, and proportional shear (full slip at the
prescribed shear ratio, not a stick/slip Coulomb law). See
[measured_sole.md](measured_sole.md), [continuous_contact.md](continuous_contact.md)
and [passive_toe_spring.md](passive_toe_spring.md).
