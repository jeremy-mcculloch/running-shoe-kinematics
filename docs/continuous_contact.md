# Continuous unilateral ground contact

The sole meets a flat rigid ground through one complementarity problem over
**every** bottom node. There is no table of contact intervals and no
winner-take-all selection: the active set is an output of the solve, and nodes
enter and leave contact through the state `g = λ = 0`, so forces, displacements
and deformation are continuous across active-set changes (their derivatives
may jump there).

Code: `contact/contact_operator.py` (shared operators), `contact/active_set.py`
(solver), `contact/continuous_contact.py` (one pose), `contact/continuous_toe.py`
(toe spring coupling), `contact/contact_model.py` (storage, full-field recovery).

## Frames and signs

| quantity | convention |
|---|---|
| fixed frame `F` | x heel → toe, y up, ground `y = 0` |
| `φ`, `φ_ref`, `varphi` | CCW positive; `Q(varphi) = [[c, −s], [s, c]]` |
| body point | `r^F = T + Q(varphi)(X + d)`, `X` reference, `d` local (top-attached frame `T`) displacement |
| gap `g_k` | `g_k = y^F_k = T_y + s (x_k + u_k) + c (y_k + v_k)`; **positive = open** |
| normal reaction `λ_k` | fixed-frame vertical ground force on node `k`; **positive = compressive** |
| tangential reaction `τ_k` | fixed-frame horizontal ground force, `τ_k = μ_eff λ_k` |
| top load `F*` | foot on shoe (`F_top = −GRF`), fixed frame; stance has `F*_y < 0` |
| top moment `Mz` | `Σ (x f_y − y f_x)` of the top nodal forces, local components, reference coordinates, about the heel-bottom point (0, 0) |

The ground normal is the fixed `e_y`, never the sole surface normal: the gap is
measured from the actual fixed-frame node positions of the rotated, deformed
sole.

## Model

* **Top**: glued to the foot. In `T` its displacement is prescribed,
  `d_t = α w_α`, `w_α = (0, φ₁(x))`, `α = tan θ` (the existing softplus toe mode).
* **Bottom**: every node may carry a ground force; nothing else is loaded.
* **Pose**: `varphi` is prescribed (marker pitch), `T_y` is unknown,
  `T_x` is a gauge (below).

### Tangential contact: proportional shear

Each active node's ground force has the fixed direction `(μ_eff, 1)`:

    (τ_k, λ_k) = (μ_eff, 1) λ_k,      μ_eff = F*_x / F*_y.

This is Coulomb full slip at the measured shear ratio. Free-body equilibrium
then gives `Σλ = −F*_y` and `Στ = −F*_x`, so **both prescribed force components
are met exactly**, and an entering node's tangential force starts at zero
together with its normal force. Because no tangential displacement is
constrained, horizontal translation along the ground is a pure gauge: `T_x` is
fixed so the frame point `(x_r, y_r) = (L/2, 0)` keeps its reference abscissa.
The previous model stuck every active node at its reference abscissa; that law
jumps whenever a node enters contact (the node's pre-contact horizontal
position generally differs from the stuck one), so it cannot give continuous
fields. Proportional shear is a modelling change: there is no sticking and the
shear traction is distributed like the pressure.

## Shared operators (`ContactOperator`)

Given local bottom nodal forces `f_b` (component-major `[f_x…; f_y…]`), the
top-clamped body response is unique (the clamp removes the rigid modes):

    d_b = α db_toe + db_fb f_b,     F_t = α ft_toe + ft_fb f_b,     a = α r_toe + r_fb f_b,

with `R_t^T F_t + R_b^T f_b = 0` built in. They are condensed once from the
stored nodal-force compliance blocks by LU-factoring the small
`(2 n_t + 3)` matrix `[[C_tt, R_t], [R_t^T, 0]]` (no FEM solve, no explicit
inverse). `db_fb` is the top-clamped bottom compliance; it is symmetric
(roundoff `G_symmetry_error ≈ 1e-12`, then symmetrized) and positive definite
(`G_min_eigenvalue > 0`, both recorded in the model metadata, not on the operator). `a` are the
rigid amplitudes about `(x_r, y_r)` that the plate and render influence
matrices use.

With `B = [(c μ + s) I; (c − s μ) I]` (local components of the unit ground
force) and `N = [s I, c I]` (ground-normal component of a local displacement),

    g = q + T_y 1 + H λ,     q = s(x + α db_toe,x) + c(y + α db_toe,y),     H = N db_fb B.

`H` is symmetric positive definite for `μ_eff = 0` (`H = N db_fb Nᵀ`); for
`μ_eff ≠ 0` it is nonsymmetric with a positive definite symmetric part, i.e. a
P-matrix for moderate shear ratios. `H` costs `O(N_b²)` to form per evaluation
from the four component blocks of `db_fb`.

## Complementarity problem

    0 ≤ λ ⊥ g ≥ 0,      g = q + T_y 1 + H λ,      1ᵀλ = Λ = −F*_y.

`T_y` is free; it is exactly the Lagrange multiplier of the force constraint,
so for `μ_eff = 0` these are the KKT conditions of

    min_{λ ≥ 0, 1ᵀλ = Λ}  ½ λᵀ H λ + qᵀ λ,

and global force equilibrium is part of the problem rather than regularized
away. With `Λ ≤ 0` (unloaded) the solution is `λ = 0` and `T_y = −min q`
(the sole just touches).

### Active-set algorithm (`solve_mixed_lcp`)

For an active set `A` the equality block

    [H_AA  1] [λ_A]   [−q_A]
    [1ᵀ    0] [T_y] = [ Λ  ]

is LU-factored and `g_I = q_I + T_y + H_IA λ_A`. Updates:

1. **PDAS block step**: drop active nodes with `λ < −τ_λ`, add inactive nodes
   with `g < −τ_g` (deterministic, all violators at once).
2. **Cycling protection**: if an active set repeats, switch to **Murty's
   least-index rule** (flip only the smallest-index violator), which
   terminates finitely for P-matrices. `pivot_rule` reports which ran.
3. Warm start from the previous sample's (or toe iterate's) active set; a cold
   start begins at the node with the smallest free gap.

`ActiveBlockSolver` refactors the block per active-set change (`O(a³)`); its
interface allows rank-one Cholesky/LU up- and downdates (`O(a²)`) later without
changing callers.

## Passive toe spring (`solve_toe_contact`)

Unchanged physics: `k_toe = 25 N·m/rad`, `θ0 = 0`, `α = tan θ` exactly,

    g(θ) = Q_α,shoe→foot(α) − k (arctan α − θ0)/(1 + α²) = 0,
    Q_α,shoe→foot = width · (−ψᵀ f_t,y),
    varphi(α) = φ_rearfoot − atan2(y_mtp_0 + α y_mtp_shape_mode, L − ℓ_toe).

Nested exact scalar solve: every residual evaluation solves the contact
problem to convergence at `varphi(α)` (warm-started from the nearest evaluated
`θ`), then Brent's method brackets and refines the root in `θ`.

* **Warm** (previous sample / GUI state): the bracket grows geometrically
  around the previous `θ` and the root on the connected branch is taken
  (`status = continuation`); typically 6–8 contact solves per sample.
* **Cold**: the bounds are scanned with `toe_root_scan_points` evaluations,
  every sign change is refined, and the root with the lowest total potential
  `U − ∫Q dα` is chosen (`multiple_roots` if more than one).
* **Low load** (`|F| <` `toe_low_force_threshold_N`): start from `θ0`.

Stability is `Π'' = d²U/dα² − dQ/dα` (central difference). `Q` is continuous in
`α` because the contact solution is, so the root is continuous along a gait.

## Recovery of outputs

From `[F_t; f_b]` and `a` (all linear, no solves): bottom/top displacements
(`d_b`, `d_t`), every bottom gap and normal/tangential reaction, totals and the
reaction moment, top `Mz`, toe moment `T_toe`, `Q_α`, the foam field at every
mesh vertex and the strain energy density (shared render influence), the plate
`u, v, θ`, curvature, bending moment, shear and axial constraint force (shared
plate influence, Hermite postprocess). Centre of pressure
`x_cop = Σ λ_k x^F_k / Σ λ_k` in the fixed frame (NaN when `Σλ` is below the
reaction tolerance), plus its heel-relative value.

Diagnostics per solve: `min g`, `min λ` over the active set, `max |g λ|`,
penetration, tension, force / equilibrium force / equilibrium moment
residuals, active-block reciprocal condition, iterations, rule, timings.

## Reporting contact sets

Active nodes are sorted heel → toe and split into connected **patches**
(first/last node, material coordinates, node count). A single patch is
reported as the contact interval; several patches are kept separate (never
filled in) and raise a warning (`+multiple_contact_patches` in replay status):
it may be valid geometry, mesh-scale oscillation, or a sign/conditioning
problem. Contiguity is never imposed by the solver.

**Contact-edge estimate (display only)**: for each patch end, the reactions of
the two outermost active nodes are extrapolated linearly to zero and clamped
to the first free edge. It is a reporting quantity; no zero-gap condition is
imposed inside a linear edge (that would force the neighbouring vertex to zero
as well).

Display: `r = r_p + Q[(X − X_p) + s_d (d − d_p)]` about the reaction-weighted
contact centroid `X_p` (continuous in the solution), so `s_d` scales only the
elastic part and never `varphi`. The GUI distinguishes active, inactive and
transition nodes (active with `λ ≤ 2 %` of the peak, or inactive with
`g ≤ 2 % h`), shades each patch separately and plots the gap and reaction over
all bottom nodes.

## Tolerances (`ContactTolerances`)

Scale-aware, resolved per solve with `h` the median bottom node spacing and
`F_s = max(|F*|, force_floor)`:

| name | value |
|---|---|
| `gap_tolerance` | `gap_rel_tol · h` (1e-9) |
| `reaction_tolerance` | `reaction_rel_tol · F_s` (1e-9) |
| `complementarity_tolerance` | `complementarity_rel_tol · h · F_s` |
| `force_balance_tolerance` / `moment_balance_tolerance` | `1e-8 · F_s` / `1e-8 · F_s · L` |
| `toe_equilibrium_*` | from `ToeSpringConfig` (abs 1e-6 N·m, rel 1e-8) |
| `maximum_active_set_iterations` / `maximum_toe_iterations` | 200 / 80 |
| `condition_number_warning` | 1e10 (active block) |
| `mu_eff_warning` | 2.0 (flags large shear ratios) |

Values within tolerance may be reported as numerical zero; significant
penetration or tension is never clipped (it fails `admissible`).

## Storage (`contact_lookup.npz`, `schema_version = 17`)

Three sections, each written and read by its own class, plus two JSON blobs:

- `contact_operator_*` (`ContactOperator`): arrays `x_top`, `y_top`, `x_bottom`,
  `y_bottom`, `shape_mode_toe_bend`, `db_fb`, `db_toe`, `ft_fb`, `ft_toe`, `r_fb`,
  `r_toe`, `toe_shape_mode_flat_rearfoot` and scalars `L`, `H`, toe length, `kappa`,
  `phi_ref`, `x_r`, `y_r`, `h_mesh`, `rearfoot_y_mtp_shape_mode`, `rearfoot_y_mtp_0`.
- `render_*` (`RenderSection`): mesh points / triangles / element regions,
  per-element `E` and `ν`, the boundary and region-loop node ids, and the
  displacement influence of **every** mesh vertex (`influence_loads`,
  `influence_rigid`; rows `[u...; v...]` in vertex order).
- `plate_*` (`PlateResponse`): plate `node_ids`, `EI_plate_Nm2_per_m`, rotation
  `theta`, element axial force `plate_tension` (positive = tension) and the
  inextensibility residual `inextensibility_residual`, each per unit boundary load,
  with `rigid_*` counterparts per unit rigid amplitude. The plate reference
  coordinates are the render mesh points at `node_ids`, the element frames come
  from `fem.plate.plate_element_frames`, and the plate nodal `u, v` are the
  plate-node rows of the render influence, so none of them is stored twice.
- `metadata_json` (formulation, tangential law, the operator build diagnostics
  `reciprocity_error`, `G_symmetry_error`, `G_min_eigenvalue`, `G_max_eigenvalue`,
  `clamp_condition`, the bottom foam Poisson ratio, the setup and its fingerprint)
  and `geometry_metadata_json`.
Files with any other schema (e.g. the schema-12 interval tables) are rejected
with a regeneration message. At the 3 mm measured mesh (`N_b = 126`) the build
takes ~1.5 s after the FEM solve and the file is 19.4 MB (mostly the render
influence).

## Runtime complexity

Per active-set iteration: `O(a³)` block factorization + `O(N a)` gap update,
`O(N²)` to form `H`; worst case `O(N³)`. With rank-one updates the iteration
target is `O(a² + N a)`. No contiguous-interval enumeration (`N(N+1)/2`
candidates, `O(N³)` checks per sample) happens anywhere. Measured (Apple
silicon, `N_b = 126`): one warm contact solve ≈ 0.3 ms (1 iteration), cold
3–7 iterations; one gait sample with the toe spring ≈ 2 ms (median 2.2 ms,
6–8 contact solves); a 300-frame stance replays in 0.6 s.

## Validation

`scripts/validate_continuous_contact.py [config]` rebuilds the model from the
config, solves pose / load / rotation / toe sweeps and the Wang gait replay,
checks every state against an **independent direct sparse FEM solve** (top
clamped, the solution's bottom forces applied, plate constraints `B_p`), and
bisects every active-set change to its transition point to compare left and
right limits. Results for `configs/setup.json` (3 mm mesh):

| case | transitions | direct FEM (rel. error) | max left/right difference at transitions (rel.) |
|---|---|---|---|
| pose: heel → midfoot → toe | 101 | displacement 2e-9, top force 5e-9, plate 3e-7 | reactions 6e-8, displacement 1e-8, Mz/COP 2e-10 |
| rotation: nodes leaving | 20 | 1e-9 / 4e-9 / 2e-7 | 4e-8 / 8e-9 / 5e-10 |
| load: entering → nearly full | 52 | 1e-9 / 4e-9 / 4e-7 | 9e-9 / 2e-9 / 1e-10 |
| passive toe ramp | 66 | 1e-9 / 3e-9 / 8e-7 | 4e-8 / 7e-9 / θ 2e-10 |
| Wang P1 pr1 01 replay (300 frames) | 125 | 2e-9 / 7e-9 / 7e-7 | (time not bisectable; step change at a transition ≤ 4× its neighbours') |

The direct-FEM gaps satisfy complementarity to ~1e-9 h. The plate error
(≤ 8e-7) is the roundoff of the plate influence recovery on this mesh, not a
model difference (on the synthetic test soles all fields agree to ~1e-12).
Left/right differences are bounded by the bisection width, i.e. the fields are
continuous to numerical tolerance.

## Tests (`tests/test_continuous_contact.py`, `tests/test_api.py`)

Operator dimensions, symmetry and definiteness; `H` against direct FEM
perturbation solves; admissibility (zero reaction on separated nodes, zero gap
on active ones, no penetration, no tension, complementarity) across soles and
poses; global force and moment equilibrium; full-field and plate recovery
against direct FEM; nodes entering / leaving with zero reaction; continuity of
force, moment, displacement, plate and COP across bisected one-node
transitions; permitted slope change; warm / cold / random-start agreement;
cycling fallback on a PDAS-cycling P-matrix; patch reporting (two-patch and
single-patch states); display-only edge estimate; exact toe spring
(`θ = arctan α`, residual, coupling to the contact solve, warm continuation);
a smooth synthetic gait sweep without jumps; online cost (one small block per
iteration); serialization and old-schema rejection; plotting with zero, one
and two patches; `SoleModel.step`.

## Limitations

* Proportional shear is a full-slip assumption with a uniform traction ratio;
  sticking or finite-friction contact would need a history-dependent or
  frictional complementarity model. Large `|Fx/Fy|` (near zero vertical load)
  is flagged; `H` can then lose its P-matrix property.
* Contact is node-to-ground on the linear boundary: the active set changes
  one node at a time and the edge estimate is display only.
* Infinitesimal strain with exact boundary rotation (not objective for large
  local rotations), as for the rest of the model.
* Solutions are unique when `H` is a P-matrix; multiple toe roots are possible
  and are resolved by continuation (warm) or lowest potential (cold).
* `ActiveBlockSolver` refactors per iteration; rank-one updates are not yet
  implemented.
