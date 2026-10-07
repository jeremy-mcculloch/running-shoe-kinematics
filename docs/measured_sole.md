# Measured carbon-plated sole (`geometry_type = "measured_sole"`)

The measured-sole model replaces the rectangle / layered-trapezoid outline with
an image-derived shoe profile: two conforming foam regions separated by a
measured foam interface, and an inextensible Euler–Bernoulli carbon plate that
covers only part of that interface. It reuses the existing solver stack
unchanged in structure: the same plane-strain foam assembly, Hermite plate,
exact inextensibility rows \(B_p\), saddle-point compliance solve, interval
contact lookup, passive toe spring, force control and gait replay. There is no
separate solver.

## Geometry file

Use the **normalized** CSV (`data/geometry/sole_geometry_normalized.csv`), never
the original pixel-coordinate file. Columns:

| column | meaning |
|--------|---------|
| `record_type` | `point` (landmark) or `curve` (one curve sample) |
| `name` | landmark or curve name |
| `index` | sample index for curves (empty for points) |
| `x_over_length` | \(x/L\) |
| `y_over_length` | \(y/L\) |

Records are parsed **by name**, never by row position. Required landmarks:
`toe_tip`, `heel_top`, `heel_bottom`, `foam_interface_toe_endpoint`,
`foam_interface_heel_endpoint`, `plate_toe_endpoint`, `plate_heel_endpoint`.
Required curves and their endpoints:

| curve | first sample | last sample |
|-------|--------------|-------------|
| `top_surface` | `heel_top` | `toe_tip` |
| `bottom_surface` | `heel_bottom` | `toe_tip` |
| `foam_interface` | `foam_interface_heel_endpoint` | `foam_interface_toe_endpoint` |

Validation rejects (with a `GeometryFileError` naming the file, line and record):
missing columns, landmarks or curves; unknown record types; duplicate landmarks;
duplicate curve indices; non-consecutive indices (must be `0..n-1`); non-numeric
or non-finite coordinates; curves that are not strictly increasing in \(x\)
(duplicate or reversed samples); curve endpoints that disagree with their
landmarks; `heel_bottom` not at the origin; `toe_tip` not at \(x/L=1\).
Unrecognized names are ignored and reported in the diagnostics. Names such as
`corner_2_interior` in the source are **not** interpreted: all three exterior
corners (heel bottom, toe tip, heel top) are required to be convex and are
checked geometrically.

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
- Positive signed area and the absence of self-intersections are checked.
- The two foam-interface endpoints are projected onto the exterior (heel edge
  and bottom curve respectively) and inserted as vertices, so the supplied
  exterior curves and the straight heel edge are kept exactly. The projection
  distances are stored in `geometry_metadata["projection_distances_m"]`
  (about 3 µm and 60 µm at 270 mm).

Boundary tags: `TOP_SURFACE`, `BOTTOM_SURFACE`, `HEEL_EDGE`, `TOE_TIP`,
`FOAM_INTERFACE`, `PLATE_SEGMENT`; region tags `upper_foam`, `lower_foam`.

## Two foams

The interface splits the sole into two conforming regions sharing interface
nodes (no duplicated nodes, no contact between foams):

| region | default material | modulus |
|--------|------------------|---------|
| `upper_foam` (above the interface) | FFTurbo | \(E=2.6\times10^5\) Pa, \(\nu=0.113\) |
| `lower_foam` (below, contacts the ground) | FFLeap | \(E(x)=E_h+(E_t-E_h)\,x/L\), \(E_h=3.54\times10^5\), \(E_t=2.07\times10^5\) Pa, \(\nu=0.113\) |

Either region may be assigned either foam (`--upper-foam`, `--lower-foam` or
the GUI selectboxes). Each element belongs to exactly one region; the foam
stiffness is the sum of per-region plane-strain assemblies.

## Partial curved plate

The plate lies on the foam interface between the projected
`plate_heel_endpoint` and `plate_toe_endpoint`. Each endpoint is projected onto
the interface polyline and inserted as a mesh vertex; it is never snapped to an
existing sample (projection distances are below 0.2 µm at 270 mm). Plate nodes
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
nodes outside the plate are ordinary shared foam nodes. On the straight
layered plate all frames coincide and the assembly reduces exactly to the
previous one.

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

The ground is flat, \(y=0\), with fixed-frame upward normal. The model assumes
**one contiguous contact interval** of bottom nodes, exactly as in
[interval_contact.md](interval_contact.md); every interval \(\mathcal I_{ij}\) is
precomputed (8001 records for 126 bottom nodes at 270 mm). Two separated
contact patches, or tension inside an active interval, need a multi-interval
model and are flagged rather than represented.

Active nodes stick perfectly: each one is prescribed the rotating-frame
displacement that places it on the ground at its reference abscissa. The
**affine curved-sole closure** term \((0,-(y_k-y_a))\) carries the measured
bottom curvature and is stored as affine column 0 with coefficient 1.
Gaps and normal reactions use the fixed-frame ground normal:

\[
g_k=\sin\varphi\,\Delta x_k+\cos\varphi\,\Delta y_k,\qquad
R_{n,k}=\sin\varphi\,R_{x,k}+\cos\varphi\,R_{y,k}.
\]

Admissibility requires \(g\ge0\) on every free node and \(R_n\ge0\) on every
contact node (with mesh-scaled tolerances). Tangential reactions may take either
sign; there is no Coulomb limit, so contact is "infinite friction" sticking.

## Passive toe spring and moments

The passive toe spring is unchanged: \(k_{\mathrm{toe}}=25\) N·m/rad (default),
generalized coordinate \(\alpha=\tan\theta\) solved exactly per interval
([passive_toe_spring.md](passive_toe_spring.md)). On the curved top the moment
about the origin uses both force components,

\[
M_z=\sum_k \bigl(x_k f_{y,k}-y_k f_{x,k}\bigr),
\]

because the top selector is no longer at constant height.

## Lookup schema and compatibility

| artefact | previous | new |
|----------|----------|-----|
| contact lookup `schema_version` | 9 | **10** |
| compliance schema (rectangle / layered) | 3 | 3 (unchanged) |
| compliance schema (measured sole) | — | **4** |

Schema v10 adds `rigid_alpha_basis` (per-record rigid amplitudes about
\(x_r=L/2,\ y_r=0\)) for every lookup, plus a measured geometry section
(mesh points, triangles, element regions, curve / loop node ids, render
influence matrices, `geometry_metadata_json`, `normalized_geometry_json`).
v9 rectangle and layered files are migrated on load
(`migrated_from_schema = 9`); v1–v8 files and any v9 file claiming
measured-sole geometry are rejected with a regenerate message. A v10 measured
file missing any geometry array is rejected. Measured compliance NPZs store
the normalized geometry, so a lookup can be regenerated without the original
CSV.

## Configuration file

Everything that is not in the geometry CSV lives in one JSON file (example:
[`configs/measured_sole_270mm.json`](../configs/measured_sole_270mm.json)),
handled by `compliance_fem.measured_config_file`. Unknown keys, booleans in
numeric fields, non-finite numbers and missing required keys are rejected with
a `ConfigFileError`. Relative paths resolve against the config file's directory.

| section | key | unit | default |
|---------|-----|------|---------|
| geometry | `csv` (required) | path | — |
| geometry | `shoe_length_mm` (required) | mm | — |
| geometry | `landmark_tolerance` | normalized | 0.002 |
| materials | `upper_foam`, `lower_foam` | `FFTurbo` / `FFLeap` | FFTurbo / FFLeap |
| materials | `FFTurbo.E_Pa`, `FFTurbo.nu` | Pa, — | 2.60e5, 0.113 |
| materials | `FFLeap.E_heel_Pa`, `FFLeap.E_toe_Pa`, `FFLeap.nu` | Pa, Pa, — | 3.54e5, 2.07e5, 0.113 |
| materials | `plate.EI_Nm2_per_m` | N·m²/m | 2.0 |
| mesh | `size_mm` | mm | 3.0 |
| mesh | `toe_refinement`, `heel_corner_refinement`, `interface_refinement`, `plate_end_refinement` | factor | 0.4, 0.5, 0.6, 0.4 |
| mesh | `curvature_max_turn_deg`, `min_angle_deg` | deg | 12, 12 |
| mesh | `element_order` | 1 or 2 | 1 |
| lookup | `a_over_length` **or** `a_m` | — / m | 0.78 × L |
| lookup | `kappa` | — | 160 |
| lookup | `reciprocity_tol` | — | 1e-6 |
| lookup | `output_dir` | path | none |
| lookup | `store_nodal_fields` | bool | false |
| toe_spring | `model`, `stiffness_Nm_per_rad`, `neutral_angle_rad`, `damping_Nms_per_rad` | | passive_spring, 25, 0, 0 |
| toe_spring | `angle_min_deg`, `angle_max_deg` | deg | −75, 75 |
| toe_spring | `low_force_threshold_N`, `equilibrium_abs_tol_Nm`, `equilibrium_rel_tol`, `root_scan_points` | | 50, 1e-6, 1e-8, 721 |
| runtime | `shoe_width_m` | m | 0.10 |

`a_over_length` keeps the softplus parameter proportional when the shoe length
changes; give `a_m` to pin it absolutely (exactly one of the two).

**Fingerprint.** A SHA-256 over the normalized geometry values (not the CSV
path), shoe length, landmark tolerance, materials, plate EI, mesh settings,
`a`, `kappa` and `reciprocity_tol`. `toe_spring`, `runtime` and
`store_nodal_fields` are excluded because they do not change the model. Building a lookup from a config
stores the full setup (`measured_setup`) and the fingerprint
(`measured_setup_fingerprint`) in the lookup metadata and writes a copy of the
config as `measured_sole_config.json` next to the lookup.

```bash
python -m compliance_fem.measured_config_file template my_sole.json --shoe-length-mm 260
python -m compliance_fem.measured_config_file check configs/measured_sole_270mm.json
python -m compliance_fem.measured_config_file build configs/measured_sole_270mm.json [--force] [--store-fields]
```

`store_nodal_fields` (or `--store-fields`) keeps the per-node bases in the NPZ.
By default they are left out and recomputed on demand, bitwise identically
([interval_contact.md](interval_contact.md#stored-scalars-on-demand-nodal-fields)).

`build` reuses `lookup.output_dir` if it already holds a lookup with the same
fingerprint, unless `--force` is given or the config asks for stored nodal
fields that the existing file lacks. Both CLIs accept `--config`; any
explicit flag (`--mesh-size-mm`, `--a`, `--kappa`, `--output`, ...) overrides the
file value. The written `measured_sole_config.json` records the values that
were actually used.

```python
from compliance_fem.api import SoleModel
model = SoleModel.from_config("configs/measured_sole_270mm.json")  # build_if_missing=True to build
state = model.step(Fx_N=-100.0, Fy_N=-1500.0, phi_deg=5.0)
```

`from_config` loads the lookup from `lookup.output_dir`, raises if its
fingerprint differs from the config (pass `require_matching_lookup=False` to
override), warns if the lookup has no fingerprint, and takes the toe spring
and shoe width from the file.

In the GUI, with the measured geometry selected, set **Measured-sole parameters
from → Config file (JSON)**. The config then supplies every model parameter.
If its `output_dir` holds a lookup with a matching fingerprint, that lookup is
loaded instead of being rebuilt. **Download measured-sole config** exports the
current sidebar values as a config file.

## Usage

```bash
# compliance
python -m compliance_fem.cli --geometry measured-sole \
  --geometry-csv data/geometry/sole_geometry_normalized.csv \
  --shoe-length-mm 270 --output outputs/measured_sole

# contact lookup (FEM recomputed in-process for plate / render recovery)
python -m compliance_fem.contact_lookup_cli --geometry measured-sole \
  --geometry-csv data/geometry/sole_geometry_normalized.csv \
  --shoe-length-mm 270 --a 0.2106 --kappa 160 --output outputs/measured_sole_lookup

# end-to-end validation report (writes validation.json)
python examples/measured_sole_validation.py --shoe-length-mm 270
```

Optional flags: `--upper-foam`, `--lower-foam`, `--mesh-size-mm`,
`--toe-refinement`, `--heel-corner-refinement`, `--interface-refinement`,
`--plate-end-refinement`, `--EI-plate`.

In the Streamlit app choose **Geometry model → Measured carbon-plated sole**.
The sidebar then exposes the CSV path, **Overall shoe length (mm)** (default
270), upper/lower foam, mesh size and the four refinement factors, and shows a
geometry summary. The model is cached by a key that includes the CSV SHA-256.
The plot is drawn from the stored mesh coordinates: both foam regions filled by
material, the exterior, the foam interface, and only the partial plate attached
to the interface. The deformed shape comes from
\(\mathbf d=G\,[\mathbf F_t;\mathbf F_b]+R\,\boldsymbol\alpha\) using stored
influence matrices, so no FEM solve is needed at runtime.

## Validation at 270 mm (3 mm mesh)

From `examples/measured_sole_validation.py`:

- Landmarks: toe tip (270.0, 52.5) mm, heel top (2.19, 26.9) mm. Areas: upper
  foam 2110 mm², lower foam 5561 mm². Plate arc length 236.4 mm.
- Mesh: 1961 nodes, 3660 triangles (1157 upper, 2503 lower), minimum angle
  25.6°, boundary deviation below 1e-16 m. 118 top-selector nodes, 126 bottom nodes,
  99 plate nodes.
- Compliance: reciprocity 1.2e-9, rigid-mode error 1.3e-18, inextensibility
  1e-19, constraint rank 98 (full).
- Lookup: 8001 / 8001 valid intervals; force / moment equilibrium residuals
  up to 3.3e-9 / 3.8e-10.
- Lookup records vs direct sparse FEM (all six affine columns, maximum relative
  error): heel 4.2e-8, toe 5.0e-9, full 4.5e-9, symmetric interior 7.7e-9,
  asymmetric interior 1.7e-8, one free node each side 3.1e-8, single node
  4.1e-8. External force / moment balance of each record is at round-off level
  (around 1e-16).

## Limitations

- 2D plane strain (per unit width), small-strain linear elasticity: no
  large-rotation foam kinematics, no hyperelastic or viscoelastic foam in the
  FEM (the viscoelastic mapper remains a separate elastic-equivalent step).
- The plate is inextensible Euler–Bernoulli, perfectly bonded to the interface,
  with free ends; no delamination or plate shear.
- One contiguous contact interval, perfect sticking, flat rigid ground.
- The geometry is only as accurate as the image digitization; the sharp toe
  vertex is a geometric singularity, so stresses there are mesh dependent
  (nodal forces and compliance remain well defined).
- The projected toe-side plate endpoint creates a sub-millimetre plate element,
  which inflates the stiffness matrix norm. Compliance solves are accepted on a
  normwise backward error of at most 1e-13 when the relative residual sits
  near 1e-8.
- The stored per-record plate elongation residual \(\lVert B_p q\rVert\) reaches
  about 5e-6 m per unit translation coefficient, the same order as the layered
  lookup. For physical translations of millimetres this is negligible.
- File size: about 20 MB for the 270 mm lookup with on-demand nodal fields (the
  default; loads in about 0.1–0.3 s), or 377 MB with `store_nodal_fields`. The
  first GUI build takes about 1–2 minutes; cached reruns take under a second.
