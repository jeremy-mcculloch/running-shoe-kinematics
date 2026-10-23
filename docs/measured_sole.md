# Measured carbon-plated sole

The measured-sole model is the only geometry: an image-derived shoe profile
with two conforming foam regions separated by a
measured foam interface, and an inextensible Euler–Bernoulli carbon plate that
covers only part of that interface. It reuses the existing solver stack
unchanged in structure: the same plane-strain foam assembly, Hermite plate,
exact inextensibility rows \(B_p\), saddle-point compliance solve, shared
contact operator with the continuous unilateral contact solve, passive toe
spring and gait replay. There is no separate solver.

## Geometry file

Use the normalized CSV (`data/geometry/sole_geometry.csv`), never
the original pixel-coordinate file. Columns:

| column | meaning |
|--------|---------|
| `record_type` | `curve` (one curve sample) or `plate` (one plate end) |
| `name` | curve name, or `heel_end` / `toe_end` for plate rows |
| `index` | sample index for curves (empty for plate rows) |
| `x_over_length` | \(x/L\) |
| `y_over_length` | \(y/L\) (empty for plate rows) |

Records are parsed **by name**, never by row position. The file holds three
curves, each ordered heel to toe, and the plate extent:

```text
record_type,name,index,x_over_length,y_over_length
plate,heel_end,,0.0499860886,
plate,toe_end,,0.9156159499,
curve,top_surface,0,0.008099327,0.099702947
...
```

There are no separate landmark rows; every named point comes from the curves:

| point | definition |
|-------|------------|
| `heel_bottom` | first `bottom_surface` sample, must be the origin |
| `toe_tip` | last `bottom_surface` sample, must have \(x/L=1\) and equal the last `top_surface` sample |
| `heel_top` | first `top_surface` sample |
| `foam_interface_heel_endpoint` / `_toe_endpoint` | first / last `foam_interface` sample |
| `plate_heel_endpoint` / `_toe_endpoint` | the foam interface at the `plate` rows' \(x\) |

Validation rejects (with a `GeometryFileError` naming the file, line and record):
missing columns, curves or plate rows; unknown record types or names; landmark
`point` rows (the old format); duplicate curve indices or plate rows;
non-consecutive indices (must be `0..n-1`); non-numeric or non-finite
coordinates; curves that are not strictly increasing in \(x\) (duplicate or
reversed samples); the bottom not starting at the origin; top and bottom not
ending at one toe tip with \(x/L=1\); plate ends outside the interface or out
of order. Geometrically, the top must lie above the bottom, the interface
strictly between them, and all three exterior corners (heel bottom, toe tip,
heel top) must be convex.

## `shoe_length_mm` and SI conversion

`shoe_length_mm` is the **projected** heel-bottom-to-toe-tip length along \(x\)
(`normalization = "projected_heel_bottom_to_toe_x_span"`), not the arc length of
the bottom or top curve. It is required, must be finite and positive, and is
the only scale:

\[
x = L\,\hat x,\qquad y = L\,\hat y,\qquad L=\texttt{shoe\_length\_mm}/1000\ \mathrm{m}.
\]

Both axes use the same factor. Coordinates are never flipped, reversed,
re-centred or independently rescaled; the CSV is already in the
heel-to-toe \(x\) / up \(y\) convention (`coordinate_system =
"heel_to_toe_x_up_y"`). Material moduli are in Pa, \(EI\) in N·m² per unit width,
and all solver quantities in SI.

## Exterior topology

The exterior loop, counterclockwise, is

```text
heel_bottom --bottom_surface--> toe_tip --top_surface (reversed)--> heel_top --heel edge--> heel_bottom
```

- `HEEL_EDGE` is the straight segment `heel_top -> heel_bottom`.
- `TOE_TIP` is a single sharp vertex where top and bottom meet; it is **not**
  rounded or filleted.
- Positive signed area is checked; with every curve strictly increasing in
  \(x\), "top above bottom" rules out self-intersection.
- The two foam-interface endpoints are projected onto the exterior (heel edge
  and bottom curve respectively) and inserted as vertices, so the supplied
  exterior curves and the straight heel edge are kept exactly. Each endpoint
  must lie within `landmark_tolerance` of the exterior (it moves about 3 µm and
  60 µm at 270 mm).

Boundary tags: `TOP_SURFACE`, `BOTTOM_SURFACE`, `HEEL_EDGE`, `TOE_TIP`,
`FOAM_INTERFACE`, `PLATE_SEGMENT`; region tags `upper_foam`, `lower_foam`.

## Two foams

The interface splits the sole into two conforming regions sharing interface
nodes (no duplicated nodes, no contact between foams):

| region | default material | modulus |
|--------|------------------|---------|
| `upper_foam` (above the interface) | FFTurbo | \(E=2.6\times10^5\) Pa, \(\nu=0.113\) |
| `lower_foam` (below, contacts the ground) | FFLeap | \(E(x)=E_h+(E_t-E_h)\,x/L\), \(E_h=3.54\times10^5\), \(E_t=2.07\times10^5\) Pa, \(\nu=0.113\) |

Either region may be assigned either foam (`sole.upper_foam_material`,
`sole.lower_foam_material` in the config, or the GUI selectboxes). Each element belongs to exactly one region; the foam
stiffness is the sum of per-region plane-strain assemblies.

## Partial curved plate

The plate lies on the foam interface between `plate_heel_endpoint` and
`plate_toe_endpoint`, the interface points at the `plate` rows' \(x\). Each
endpoint is inserted as a mesh vertex (an existing sample is reused only if it
coincides within \(10^{-9}L\)). Plate nodes
are the interface nodes between them, so the plate is an ordered polyline of
\(n\) nodes and \(n-1\) elements (99 / 98 at the default 3 mm mesh).

Each plate element uses its own frame:

\[
L_e=\lVert \mathbf x_{e,2}-\mathbf x_{e,1}\rVert,\qquad
\mathbf t_e=(\mathbf x_{e,2}-\mathbf x_{e,1})/L_e,\qquad
\mathbf n_e=(-t_{e,y},\,t_{e,x}).
\]

Bending uses the Hermite element in each element's local normal direction, and
inextensibility rows are \(\mathbf t_e\cdot(\mathbf u_{e,2}-\mathbf u_{e,1})=0\)
per element. Both plate ends are free (natural rotation conditions); interface
nodes outside the plate are ordinary shared foam nodes. On a straight plate all
element frames coincide.

## Selectors and the shared toe

- Top selector: `TOP_SURFACE` nodes from `heel_top` towards the toe, **excluding
  the toe vertex**.
- Bottom (contact) selector: `BOTTOM_SURFACE` nodes from `heel_bottom` to
  `toe_tip`, **including the toe vertex**.

`shared_toe_policy = "bottom_contact_owns_toe_vertex"`: the toe vertex belongs
only to the bottom selector, so the top and bottom selectors are disjoint and
the compliance blocks stay well defined. Heel-edge nodes other than its two
corners belong to neither selector (traction free). Both selectors are sorted
heel-to-toe; their node ids and DOFs are stored on the compliance result and in
the NPZ rather than recomputed from geometric predicates.

## Contact

The ground is flat, \(y=0\), with fixed-frame upward normal. Every bottom
node (126 at the default 3 mm mesh, 270 mm) is a potential contact node; the
contact set is an output of one complementarity solve per pose, not a choice
among precomputed intervals. It may be empty, one patch, or several
disconnected patches, which are reported separately. The formulation is in
[continuous_contact.md](continuous_contact.md); in short:

- The top is glued to the foot (prescribed shape \(\alpha\,\varphi_1\) in the
  top-attached frame), and the top-clamped bottom compliance \(G_{bb}\) and
  the related operators are condensed once from the compliance blocks. The
  measured bottom curvature enters through the reference coordinates
  \((x_k,y_k)\) of the bottom nodes.
- Gaps use the fixed-frame ground normal,
  \[
  g_k=T_y+\sin\varphi\,(x_k+u_k)+\cos\varphi\,(y_k+v_k),
  \]
  and the normal reactions solve
  \(0\le\lambda\perp g\ge0\), \(\sum\lambda=-F_y^\star\).
- Tangential contact is proportional shear: each active node's ground force is
  \((\mu_{\mathrm{eff}},1)\,\lambda_k\) with \(\mu_{\mathrm{eff}}=F_x^\star/F_y^\star\).

A state is admissible when the solve converged and penetration and tension
are within mesh- and load-scaled tolerances.

## Passive toe spring and moments

The passive toe spring is unchanged: \(k_{\mathrm{toe}}=25\) N·m/rad (default),
generalized coordinate \(\alpha=\tan\theta\) solved exactly, with a
converged contact solve per residual evaluation
([passive_toe_spring.md](passive_toe_spring.md)). On the curved top the moment
about the origin uses both force components,

\[
M_z=\sum_k \bigl(x_k f_{y,k}-y_k f_{x,k}\bigr),
\]

because the top selector is no longer at constant height.

## Lookup schema and compatibility

The contact model file `contact_lookup.npz` uses `schema_version = 17` and the
compliance NPZ uses `compliance_schema_version = 7`. The contact model stores
the shared contact operator (`contact_operator_*`: top-clamped bottom
compliance `db_fb` and the related top-force, rigid-amplitude and toe-mode
operators, with rigid amplitudes about \(x_r=L/2,\ y_r=0\)), the render section
(`render_*`: mesh points, triangles, element regions and moduli, curve / loop
node ids, render influence matrices), the plate response (`plate_*`) and
`geometry_metadata_json` (layout in `docs/continuous_contact.md`). Files with any
other schema version (including the older per-interval lookups) are rejected
with a regenerate message, and so is a file missing part of any section.

The render influence covers **every mesh vertex** (not only the region
loops), and the section also stores per-element plane-strain moduli
`mesh_element_E` / `mesh_element_nu` (graded foams evaluated at the element
centroid). The GUI uses them to draw the deformed mesh and colour each
triangle by its strain energy density
\(W=\tfrac{\lambda}{2}(\operatorname{tr}\varepsilon)^2+\mu\,\varepsilon:\varepsilon\)
(J/m³, from the true displacement, P1 strain per triangle;
`RenderSection.strain_energy_density`). The plate is coloured by
its bending energy per unit length \(\tfrac12 EI\kappa^2\) (plate colour
quantity `strain_energy_density`; the plate is inextensible, so there is no
membrane term). Both colour scales are separate and cap at the 99th
percentile; the gait animation fixes them over the stance. Compliance
NPZs store the normalized geometry and every scalar `SoleConfig` setting under
its field name (`EI_plate_Nm2_per_m`, `mesh_size_m`, ...) for reference; contact
models are always regenerated from the config.

## Configuration file

Everything that is not in the geometry CSV lives in one JSON file (example:
[`configs/setup.json`](../configs/setup.json)),
handled by `compliance_fem.contact.model_setup`. The file has one section per
settings dataclass (`sole` = `SoleConfig`, `lookup` = `LookupConfig`,
`toe_spring` = `ToeSpringConfig`, `runtime` = `RuntimeConfig`), keyed by the
dataclass field names, plus the optional top-level `schema_version` (must be
2) and `description`. Unknown keys, booleans in numeric fields, non-finite
numbers and missing required keys are rejected with a `ConfigFileError`.
Relative paths resolve against the config file's directory.

| section | key | unit | default |
|---------|-----|------|---------|
| sole | `geometry_csv` (required) | path | — |
| sole | `shoe_length_mm` (required) | mm | — |
| sole | `landmark_tolerance` | fraction of L | 0.002 |
| sole | `upper_foam_material`, `lower_foam_material` | `FFTurbo` / `FFLeap` | FFTurbo / FFLeap |
| sole | `ffturbo_E_Pa`, `ffturbo_nu` | Pa, — | 2.60e5, 0.113 |
| sole | `ffleap_E_heel_Pa`, `ffleap_E_toe_Pa`, `ffleap_nu` | Pa, Pa, — | 3.54e5, 2.07e5, 0.113 |
| sole | `EI_plate_Nm2_per_m` | N·m²/m | 2.0 |
| sole | `mesh_size_m` | m | 0.003 |
| sole | `toe_refinement`, `heel_corner_refinement`, `interface_refinement`, `plate_end_refinement` | multiplier of `mesh_size_m` | 0.4, 0.5, 0.6, 0.4 |
| sole | `curvature_max_turn_deg`, `min_angle_deg` | deg | 12, 12 |
| sole | `element_order` | — | 1 (only linear triangles are supported) |
| lookup | `toe_length_mm` | mm | 59.4 |
| lookup | `min_bend_radius_mm` | mm | 6.75 |
| lookup | `reciprocity_tol` | — | 1e-6 |
| lookup | `output_dir` | path | none (`outputs/contact_lookup` when building) |
| toe_spring | `toe_model`, `toe_stiffness_Nm_per_rad`, `toe_neutral_angle_rad`, `toe_damping_Nms_per_rad` | —, N·m/rad, rad, N·m·s/rad | passive_spring, 25, 0, 0 |
| toe_spring | `toe_angle_min_deg`, `toe_angle_max_deg` | deg | −75, 75 |
| toe_spring | `toe_low_force_threshold_N`, `toe_equilibrium_abs_tol_Nm`, `toe_equilibrium_rel_tol`, `toe_root_scan_points` | N, N·m, —, — | 50, 1e-6, 1e-8, 721 |
| runtime | `shoe_width_m` | m | 0.10 |

`toe_length_mm` is the distance from the softplus joint to the toe tip, so the
joint sits at `L - toe_length` from the heel. `min_bend_radius_mm`
is the tightest radius of that shape at a 45° toe angle (α = tan 45° = 1). The
dimensionless softplus sharpness used by the solver is `κ = 4L / R`.

**Fingerprint.** A SHA-256 over the normalized geometry values (not the CSV
path), shoe length, landmark tolerance, materials, plate EI, mesh settings,
toe length, `kappa` and `reciprocity_tol`. `toe_spring` and `runtime` are
excluded because they do not change the model. Building a contact model from a
config stores the full setup (`model_setup`) and the fingerprint
(`model_setup_fingerprint`) in its metadata and writes a copy of the config as
`measured_sole_config.json` and a `generation_summary.json` (operator
diagnostics, build time, file size) next to `contact_lookup.npz`.

```bash
compliance-fem-contact-lookup configs/setup.json [--force] [--save-fem] [--skip-plots] [--quiet]
```

`compliance-fem-contact-lookup` skips the build if `lookup.output_dir` already
holds a schema-17 file with the same fingerprint, unless `--force` is given. It
uses the config file as is: there are no per-parameter override flags, so edit
(or copy) the config to change a value.

```python
from compliance_fem.api import SoleModel
model = SoleModel.from_config("configs/setup.json")  # build_if_missing=True to build
state = model.step(Fx_N=-100.0, Fy_N=-1500.0, phi_deg=5.0)
```

`from_config` loads the contact model from `lookup.output_dir`, raises if its
fingerprint differs from the config (pass `require_matching_lookup=False` to
override), warns if the file has no fingerprint, and takes the toe spring
and shoe width from the config.

In the GUI, set **Measured-sole parameters from → Config file (JSON)**. The
config then supplies every model parameter. If its `output_dir` holds a
contact model with a matching fingerprint, that model is loaded instead of
being rebuilt. **Export measured-sole config (JSON)** downloads the current
sidebar values as a config file.

## Usage

```bash
# mesh -> FEM -> contact model, written to lookup.output_dir;
# --save-fem also writes mesh, compliance NPZ, heatmaps and fem_summary.json to <output_dir>/fem/
compliance-fem-contact-lookup configs/setup.json --save-fem
```

The Streamlit sidebar exposes the CSV path, **Overall shoe length (mm)** (default
270), upper/lower foam, mesh size and the four refinement factors, and shows a
geometry summary. The model is cached by a key that includes the CSV SHA-256.
The plot is drawn from the stored mesh coordinates: both foam regions filled by
material, the exterior, the foam interface, and only the partial plate attached
to the interface. The deformed shape of every mesh vertex comes from the
solved boundary loads and rigid amplitudes through the stored influence
matrices, \(\mathbf d=G_{\mathrm{loads}}\,[\mathbf F_t;\mathbf f_b]+G_{\mathrm{rigid}}\,\mathbf a\)
(`render_influence_loads`, `render_influence_rigid`), and the plate fields
likewise from the plate response, so no FEM solve is needed at
runtime. The display scale \(s_d\) exaggerates only the elastic part, about
the reaction-weighted contact centroid, so the drawn shape does not jump when
the contact set changes.

## Validation at 270 mm (3 mm mesh)

Recorded on the full-resolution model. `tests/test_measured_sole.py` runs the
geometry, mesh, plate and compliance checks and the contact / toe checks
(fixed-frame gap and reaction, exact passive toe equilibrium, no runtime FEM
solve, serialization) on a coarse mesh of the same geometry; the contact
solution is compared with an independent direct FEM solve in
`tests/test_continuous_contact.py` (see
[continuous_contact.md](continuous_contact.md#validation) for the measured-sole
validation script).

- Landmarks: toe tip (270.0, 52.5) mm, heel top (2.19, 26.9) mm. Areas: upper
  foam 2110 mm², lower foam 5561 mm². Plate arc length 236.4 mm.
- Mesh: 1961 nodes, 3660 triangles (1157 upper, 2503 lower), minimum angle
  25.6°, boundary deviation below 1e-16 m. 118 top-selector nodes, 126 bottom nodes,
  99 plate nodes.
- Compliance: reciprocity 1.2e-9, rigid-mode error 1.3e-18, inextensibility
  1e-19, constraint rank 98 (full).
- Contact model: built in about 1.5 s from the compliance result; the file is
  about 19.4 MB, mostly the render influence.

## Limitations

- 2D plane strain (per unit width), small-strain linear elasticity: no
  large-rotation foam kinematics, no hyperelastic or viscoelastic foam in the
  FEM (the viscoelastic mapper remains a separate elastic-equivalent step).
- The plate is inextensible Euler–Bernoulli, perfectly bonded to the interface,
  with free ends; no delamination or plate shear.
- Flat rigid ground, node-to-ground unilateral contact, proportional shear
  (full slip at the prescribed shear ratio, no stick/slip Coulomb law).
- The geometry is only as accurate as the image digitization; the sharp toe
  vertex is a geometric singularity, so stresses there are mesh dependent
  (nodal forces and compliance remain well defined).
- The projected toe-side plate endpoint creates a sub-millimetre plate element,
  which inflates the stiffness matrix norm. Compliance solves are accepted on a
  normwise backward error of at most 1e-13 when the relative residual sits
  near 1e-8.
