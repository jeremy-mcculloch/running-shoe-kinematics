# Driving the sole model from another codebase

`compliance_fem.api.SoleModel` wraps a precomputed contact lookup behind a
single `step()` call. The caller supplies the load on the shoe and the rearfoot
pitch; the model solves the passive toe-spring equilibrium (the toe angle is an
**output**), picks the admissible contact interval, and returns nodal
positions, displacements, forces and moments. No FEM is solved at runtime.

```python
from compliance_fem.api import SoleModel

model = SoleModel("outputs/measured_sole_lookup", shoe_width_m=0.10)  # load once

for Fx, Fy, phi in external_frames:            # N, N, degrees
    st = model.step(Fx, Fy, phi)
    if not st.admissible:
        log_warning(st.status)                 # approximate (least-violating) state
    send(st.top_xy_m, st.bottom_xy_m, st.Mz_fixed_Nm, st.theta_deg)

model.reset()                                  # before an unrelated sequence
```

`SoleModel` also accepts an in-memory `ContactLookupResult`, a custom
`ToeSpringConfig` (stiffness, neutral angle, angle bounds, low-load threshold),
and the `Tolerances` / `SelectionConfig` used by gait replay.

For the measured carbon-plated sole, prefer building from the JSON config.
It supplies the lookup location, shoe width and toe spring, and it checks that
the lookup was generated from the same parameters (see
[measured_sole.md](measured_sole.md#configuration-file)):

```python
model = SoleModel.from_config("configs/measured_sole_270mm.json", build_if_missing=True)
model.setup.a, model.setup.sole.EI_plate        # the parsed configuration
```

## Inputs

| argument | meaning |
|----------|---------|
| `Fx_N`, `Fy_N` | fixed-frame force **applied by the foot on the shoe top** (`F_top = -GRF`), whole shoe, N. Stance loading has `Fy_N < 0`. |
| `phi_deg` | fixed-frame heel -> MTP (rearfoot) angle, counterclockwise positive (toe up), degrees |
| `Mz_Nm` (optional) | measured top moment about the heel-bottom reference point; not prescribed, only fills `moment_residual_Nm` |

Fixed frame: x forward (heel -> toe), y up, flat ground at y = 0. The lookup is
2D plane strain per metre width, so loads are divided by `shoe_width_m` and
forces / moments multiplied back; outputs are whole-shoe values.

## Outputs (`SoleState`)

| field | meaning |
|-------|---------|
| `valid`, `admissible`, `status` | a candidate exists / satisfies all gap, reaction and force checks / solver status string |
| `theta_deg` | toe angle from the passive spring equilibrium |
| `chord_rotation_rad` | rotation of the shoe frame relative to the reference shoe |
| `contact_interval`, `contact_start_x_m`, `contact_end_x_m`, `contact_mask` | bottom nodes in ground contact |
| `top_xy_m`, `bottom_xy_m` | deformed nodal positions, fixed frame, `(2, n)` |
| `top_displacement_m`, `bottom_displacement_m` | nodal displacements in the rotating shoe frame, `(2, n)` |
| `top_force_N` | foot-on-shoe nodal forces, fixed frame |
| `bottom_reaction_N` | ground-on-shoe nodal reactions, fixed frame (zero on free nodes) |
| `Fx_N`, `Fy_N` | reproduced resultant (equals the input when `force_residual_N` ~ 0) |
| `Mz_shoe_Nm` | sum(x f_y - y f_x) of top forces, shoe-frame components and reference coordinates, about the heel-bottom reference point (gait replay convention) |
| `Mz_fixed_Nm` | the same moment from fixed-frame forces and deformed positions about the ground origin |
| `toe_moment_Nm` | toe-hinge moment of the top forces |
| `toe_spring_moment_Nm` | spring moment k(θ - θ0) at the solution |
| `toe_root_count`, `toe_stable`, `low_load` | toe-equilibrium diagnostics |
| `diagnostics` | search method, penetration, minimum contact reaction, all toe roots, condition number |

Node order: top nodes heel -> toe along the top surface (`model.x_top`,
`model.y_top` give reference coordinates), bottom nodes heel -> toe along the
sole (`model.x_bottom`, `model.y_bottom`). `SoleState.to_dict()` returns a
JSON-serializable copy.

## State and performance

The model remembers the previous contact interval and toe angle so consecutive
steps stay on one continuous branch (local interval search first, root
continuity in the toe solve). Call `reset()` between independent sequences.

The measured-sole lookup (about 20 MB, nodal fields recomputed on demand)
loads in well under a second; each `step()` takes about 0.25–0.35 s. Results
are bitwise identical to a lookup built with stored nodal fields.

## Non-Python callers

Keep one long-lived Python process that owns the `SoleModel` and exchange
`SoleState.to_dict()` over HTTP, ZeroMQ or gRPC. Do not reload the lookup per
request.

## Limitations

Elastic only: the viscoelastic mapper used by gait replay is not applied
inside `step()`; pass elastic-equivalent loads if needed. Single contiguous
contact interval with perfect sticking; non-admissible states are returned as
the least-violating interval and flagged. See
[measured_sole.md](measured_sole.md), [interval_contact.md](interval_contact.md)
and [passive_toe_spring.md](passive_toe_spring.md).
