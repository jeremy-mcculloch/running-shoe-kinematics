# Compliance FEM

Plane-strain finite-element computation of **vector** (component-major \(u,v\))
compliance matrices for a homogeneous rectangular elastic body or a layered
trapezoidal foam pair with an inextensible internal plate, plus a schema-v9
single-contiguous-interval ground-contact lookup built from those nodal-force
compliance blocks (layered tables may include an optional `plate_response`
section). See [docs/interval_contact.md](docs/interval_contact.md).

## Physical problem

The body occupies

\[
\Omega = \{(x,y): 0 \le x \le L,\ 0 \le y \le H\}
\]

and is made of an isotropic linear-elastic material with Young's modulus \(E\),
Poisson's ratio \(\nu\), unit out-of-plane thickness, and **plane-strain**
kinematics.

The displacement \(\mathbf u = (u,v)\) satisfies the standard linear-elasticity
weak form assembled with scikit-fem. Natural (zero-traction) conditions are
imposed on the unloaded boundaries while the global stiffness matrix \(\mathbf K\)
is assembled **without** displacement boundary conditions.

## Nodal-force vs nodal-traction compliance

Two related operators are computed:

| Symbol | Meaning |
|--------|---------|
| \(\mathbf C^{F}_{ab}\) | \(2n_a\times 2n_b\) nodal-force operator, component-major \([u;v]\) / \([f_x;f_y]\) |
| \(\mathbf C^{q,yy}\) | Vertical traction map from the \(yy\) block of \(\mathbf C^F\) times the 1D boundary mass |

Boundary vectors are ordered **component-major**, nodes sorted by increasing \(x\):

\[
U_\Gamma=\begin{bmatrix}u_\Gamma\\ v_\Gamma\end{bmatrix},\quad
F_\Gamma=\begin{bmatrix}f_{\Gamma,x}\\ f_{\Gamma,y}\end{bmatrix},\qquad
\texttt{dof\_ordering}=\texttt{component\_major\_uv}.
\]

**The contact lookup uses only the stacked nodal-force matrices** \(\mathbf C^{F}_{tt}\),
\(\mathbf C^{F}_{tb}\), \(\mathbf C^{F}_{bt}\), \(\mathbf C^{F}_{bb}\) (each \(2n\times 2n\)).
Old vertical-only \(n\times n\) compliance files must be regenerated; there is no silent fallback.

Here \(\mathbf M_t\) and \(\mathbf M_b\) are consistent 1D boundary mass matrices
on the top and bottom edges (vertical DOFs). If \(\mathbf q_t\) holds nodal
vertical traction samples, then \(\mathbf F_{t,y} = \mathbf M_t \mathbf q_t\).

A callable traction \(f(x)\) can also be integrated directly:

\[
\mathbf F_{t,y} = \int_{\Gamma_t} \mathbf N^T f(x)\,ds.
\]

## Rigid-body gauge

Because the body is unconstrained, \(\mathbf K\) has three rigid-body modes
(horizontal translation, vertical translation, rotation about the mesh
centroid). Compliance columns are computed by solving the symmetric
saddle-point system

\[
\begin{pmatrix}\mathbf K & \mathbf R\\ \mathbf R^T & 0\end{pmatrix}
\begin{pmatrix}\mathbf d\\ \boldsymbol\lambda\end{pmatrix}
=
\begin{pmatrix}\mathbf f\\ 0\end{pmatrix},
\]

which enforces the gauge \(\mathbf R^T\mathbf d = 0\).

**Important:** an isolated unit nodal force is not self-equilibrated, so a single
compliance column is gauge-dependent. Physically equilibrated load combinations
are gauge-independent up to a rigid motion.

## Same-boundary singularity

The continuum kernel \(C_{bb}(x,\xi)\) is logarithmically singular at \(x=\xi\).
Therefore diagonal entries of the discrete nodal-force matrix \(\mathbf C^{F}_{bb}\)
depend on mesh resolution and must not be interpreted as exact pointwise continuum
values. Responses to finite-width or smooth traction distributions converge under
mesh refinement.

## Interval contact lookup

The implementation supports one contiguous contact interval. If the normal reactions become tensile inside an otherwise active interval, or if two separated sole regions simultaneously contact the ground with a free region between them, a multi-interval contact model is required.

With \(N_b\) bottom nodes the lookup stores **every** contiguous interval
\(\mathcal I_{ij}=\{i,\dots,j\}\), \(0\le i\le j<N_b\): \(N_b(N_b+1)/2\) records
(\(N_b-1\) heel-attached, \(N_b-1\) toe-attached, one full, and
\((N_b-1)(N_b-2)/2\) interior intervals; one-node intervals included). The
heel/toe/full/interior label is derived from \((i,j)\) and is only a filter.
Each record has its own free/contact partition, a **numerical** anchor
\(x_a=(x_i+x_j)/2\), \(y_a=y_b(x_a)\) (`contact_anchor_definition="interval_midpoint"`;
the physical contact edges are \(l_h=x_i\), \(l_t=x_j\)), adjacent free node ids
(\(-1\) beyond an end of the sole), a status and a structured rejection reason.

Active nodes stick at their reference abscissa on the ground. The prescribed
rotating-frame contact displacement

\[
\mathbf d_k^{\mathcal T}=\mathbf d_a^{\mathcal T}+\bigl(Q(\varphi)^T-I\bigr)(x_k-x_a,0)^T+\bigl(0,-(y_k-y_a)\bigr)^T
\]

is exact in \(\varphi\). Its last term, the **curved-sole closure**, is stored as
affine column 0 of every response with coefficient fixed at 1 (zero on a flat
sole), so every response tensor has six columns:

| column | name | top \((u_t,v_t)\) | contact \((u_c,v_c)\) |
|---|------|-------------------|------------------------|
| 0 | `curved_sole_closure` | \((0,0)\) | \((0,\ -(y_k-y_a))\) |
| 1 | `top_shape_alpha` | \((0,\ \varphi_1(x))\) | \((0,0)\) |
| 2 | `contact_translation_x` | \((0,0)\) | \((1,0)\) |
| 3 | `contact_translation_y` | \((0,0)\) | \((0,1)\) |
| 4 | `contact_rotation_x` | \((0,0)\) | \((x-x_a,\ 0)\) |
| 5 | `contact_rotation_y` | \((0,0)\) | \((0,\ x-x_a)\) |

The single top shape mode is the endpoint-fixed softplus ramp

\[
\varphi_1(x)=s(x)-\Bigl(1-\tfrac{x}{L}\Bigr)s(0)-\tfrac{x}{L}\,s(L),
\qquad \varphi_1(0)=\varphi_1(L)=0,
\]

with no normalization. Its sign lives in one place, `corotation.SHAPE_MODE_SIGN`.

For each interval the dense boundary saddle system (prescribed top and contact
displacements, rigid modes \(\boldsymbol\alpha\) about \(x_r=L/2\), \(y_r=0\),
equilibrium rows) is LU-factored **once** and all six right-hand sides are solved
against that factorization; the global compliance blocks are reused and the
global stiffness matrix is never refactorized. Numerically rank-deficient,
failed, non-finite or high-residual records are rejected with a reason and
counted in `metadata["rejection_counts"]`.

Stored per record: bottom displacement and contact reaction bases
`(n_rec, 6, N_b)`, top force bases `(n_rec, 6, n_t)`, scalars
`scalar_lookup` `(n_rec, 6, 6)` = `[Fx, Fy, Mv, Mz, M_toe, M_toe_vertical]`,
heel/toe edge reactions and adjacent-free displacements (`edge_responses`, NaN
when absent), `kf_matrix`, residual and rank diagnostics, and
`Q_alpha_shoe_on_foot_basis` `(n_rec, 6)`: the generalized force of the shoe on
the foot along the top-shape coordinate with the rearfoot (heel → MTP line)
held fixed, \(Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}=-\psi^\mathsf T\mathbf f_{t,y}\),
\(\psi(x)=\varphi_1(x)-(x/a)\varphi_1(a)\), used by the passive toe spring
([docs/passive_toe_spring.md](docs/passive_toe_spring.md)). Toe moment about
\(P_{\mathrm{toe}}=(a,H_a)\) is

\[
T_{\mathrm{toe}}=\boldsymbol{\rho}_a^T\mathbf f_{t,y}-\boldsymbol{\eta}_a^T\mathbf f_{t,x},
\qquad
[\boldsymbol{\rho}_a]_j=\max(0,x_{t,j}-a).
\]

`schema_version` is **9**. Files with schema v8 or older (heel/toe/full tables
with \(2N_b-1\) records, contact-edge or \(L/2\) anchors and no curved-sole
closure) are **rejected** with a regenerate message; they cannot be migrated to
arbitrary interval support.

## Internal plate response

For layered lookups, plate fields are recovered from the **same** coupled
foam–plate FEM factorization used while generating the contact table—not a
separate plate solver. Each interval record stores six-column affine basis fields
\(u\), \(v\), \(\theta\), \(\lambda\), and axial force (same column order as the
boundary responses, closure first). Axial constraint rows \(B_p\) are unscaled
(\(t_p\cdot(u_{j+1}-u_j)=0\)), so the Lagrange multiplier already has force
units and equals the element axial constraint force.

At runtime the selected interval's stored plate response is contracted with the
same \([1,\boldsymbol\gamma]\) as the foam scalars. The centerline is rebuilt with cubic Hermite interpolation;
curvature \(\kappa=w''\), bending moment \(M=EI\,\kappa\), and shear
\(V=EI\,w'''\). Fixed-frame placement uses the shared foam anchor transform
\(r^{\mathcal F}=r_a^{\mathcal F}+Q(\varphi)[(\mathbf X-\mathbf X_a)+s_d(\mathbf d^{\mathcal T}-\mathbf d_a^{\mathcal T})]\);
the elastic display scale \(s_d\) never multiplies \(\varphi\).

The Streamlit GUI exposes plate toggles (show plate, undeformed plate, nodes,
rotations), a color quantity, samples/element, and a plate-results summary
panel. Rectangle lookups are also schema 9 but omit `plate_response`; the GUI
reports plate fields as **N/A**.

## Co-rotating top frame

Two frames are used. \(\mathcal F\) is the fixed ground frame; \(\mathcal T\) is
the top-attached frame, i.e. the reference mesh rigidly rotated by

\[
\varphi=\phi-\phi_{\mathrm{ref}},
\qquad
\phi_{\mathrm{ref}}=\operatorname{atan2}\bigl(y_{\mathrm{top}}(L)-y_{\mathrm{top}}(0),\,L\bigr),
\qquad
Q(\varphi)=\begin{bmatrix}\cos\varphi&-\sin\varphi\\ \sin\varphi&\cos\varphi\end{bmatrix}.
\]

\(\phi\) is the absolute fixed-frame heel-to-toe chord angle supplied by the GUI;
\(\phi_{\mathrm{ref}}\) is stored in the lookup file, so a non-horizontal
reference top surface is handled correctly. Only \(\varphi\) appears in
\(Q^T-I\) terms.

Contact nodes move rigidly with the frame relative to the interval anchor
\(x_a=(x_i+x_j)/2\) (plus the curved-sole closure on a curved bottom),

\[
\mathbf d_c^{\mathcal T}(x)=\mathbf d_a^{\mathcal T}+\bigl(Q(\varphi)^T-I\bigr)(\mathbf X-\mathbf X_a)
\;\Longrightarrow\;
\begin{cases}
u_c=d_{a,x}+(\cos\varphi-1)(x-x_a),\\
v_c=d_{a,y}-\sin\varphi\,(x-x_a),
\end{cases}
\]

which the four contact-motion modes span exactly, with true \(\sin\varphi\) and
\(\cos\varphi\) rather than a small-angle expansion.

**Deliberate approximation.** Boundary positions use the exact finite rotation,
while foam strain is still evaluated with the linear strain–displacement
operator. This is intentional: it keeps the formulation infinitesimal-strain and
linear-elastic, so the five basis responses can be precomputed once, independent
of \(\phi\), and combined by scalars at runtime. The price is that the model is
**not objective for locally large rotations**. It is not a geometrically
nonlinear FEM.

## Force-controlled inputs \(F_x^\star,F_y^\star,\phi,\theta\)

The GUI supplies fixed-frame top resultants \(F_x^\star,F_y^\star\)
(rightward/upward positive; compression typically negative \(F_y\)), plus \(\phi\)
and \(\theta\) in degrees. The runtime coefficients are

\[
\alpha=\tan\theta,\qquad r_x=\cos\varphi-1,\qquad r_y=-\sin\varphi,
\qquad
\boldsymbol\gamma=\begin{bmatrix}\alpha& d_{a,x}& d_{a,y}& r_x& r_y\end{bmatrix}^T .
\]

Positive \(\theta\) tilts the toe segment up relative to the chord. Because
\(\varphi_1\) vanishes at both ends, \(\theta\) never changes the heel or toe
height, only the bend between them.

The requested force is rotated into the local frame, \(F^{\mathcal T\star}=Q(\varphi)^T F^{\mathcal F\star}\),
and the unknown **anchor** translation follows from a \(2\times 2\) solve per
record (no inverse, no regularization)

\[
\underbrace{\begin{bmatrix}F_{x,B_x}&F_{x,B_y}\\F_{y,B_x}&F_{y,B_y}\end{bmatrix}}_{K_F}
\begin{bmatrix}d_{a,x}\\d_{a,y}\end{bmatrix}
=
F^{\mathcal T\star}
-F^{\mathcal T}_{\mathrm{closure}}-\bigl(\alpha F^{\mathcal T}_{\varphi_1}+r_x F^{\mathcal T}_{B_{rx}}+r_y F^{\mathcal T}_{B_{ry}}\bigr).
\]

Singular values, determinant, and condition number of \(K_F\) are reported;
records below the force threshold or above `kf_cond_warn` are **flagged and
skipped**, never regularized. Horizontal/vertical coupling means
\(F_x^\star=0\) does **not** imply \(d_{a,x}=0\). Reconstructed \(F_x,F_y\) are
verified against the targets in both frames. Every scalar and vector output is
then superposed with the same \(\boldsymbol\gamma\) through one shared
`contract_basis` contraction.

The lookup is indexed by the interval \((i,j)\); the shared visualization gauge
is \(r_a^{\mathcal F}=(x_a,0)\).

### Contact admissibility and interval selection

Gaps and reactions are tested in the **fixed** frame, since the ground normal is
fixed while the body rotates:

\[
g_k=\sin\varphi\,\Delta x_k+\cos\varphi\,\Delta y_k,
\qquad
R_{n,k}=\sin\varphi\,R_{x,k}^{\mathcal T}+\cos\varphi\,R_{y,k}^{\mathcal T},
\]

with \(\Delta x_k=(x_k-x_a)+u_k-d_{a,x}\) and \(\Delta y_k=(y_k-y_a)+v_k-d_{a,y}\).
An interval is admissible when it reproduces \(F^\star\), **every** free node has
\(g_k\ge-\tau_{g,\mathrm{eff}}\), and **every** contact node (both edges and all
interior nodes) has \(R_{n,k}\ge-\tau_{R,\mathrm{eff}}\), with mesh- and
load-scaled tolerances. For linear elements the nodal checks are exact (the gap
is piecewise linear); quadratic elements also check the per-segment minimum.
Interior tension sets `disconnected_contact_warning`. Selection does **not** use
the tangential reaction, \(M_v\), \(x_{\mathrm{cm}}\), or \(T_{\mathrm{toe}}\).

Contact modes: **Auto**, **Heel** (\(i=0\)), **Interior**, **Toe**
(\(j=N_b-1\)), **Full** (forced) and **Specific** (forced, validated \((i,j)\));
legacy names `Auto` / `Heel contact` / `Full contact` / `Toe contact` map onto
these. The search reports `candidate_search_method`:
`local_interval_search` / `expanded_interval_search` (near the previous
interval, only with temporal continuity enabled), `global_interval_search`,
`forced_interval`, or `least_violating_fallback` (no admissible interval;
minimum violation score, marked approximate). Among admissible intervals the
previous one is kept if still admissible, otherwise the smallest edge
complementarity score wins; continuity is a tie-breaker among physically
acceptable candidates only and never keeps a penetrating or tensile interval.
Details: [docs/interval_contact.md](docs/interval_contact.md).

The center of effort is computed in the fixed frame by rotating both the top
nodal reactions and the top nodal positions, and is `NaN` when \(|F_y^\star|\) is
below the zero tolerance. Its absolute value depends on the visualization gauge
\(r_a^{\mathcal F}=(x_a,0)\); only \(x_{\mathrm{cm}}-x_a\) is gauge independent.

### Sign conventions

- \(\phi\) and \(\varphi\) are counterclockwise positive.
- Positive fixed-frame normal gap = open gap above the ground.
- Positive fixed-frame \(R_n\) = upward compressive nodal force from the ground.
- Positive top \(F_y\) = upward; positive top \(F_x\) = rightward.
- \(R_t\) may take either sign: contact remains perfectly sticking.

### Generate lookup, query, and GUI

```bash
# regenerate lookup (schema v10: every contiguous contact interval, six-column affine basis;
# v9 rectangle/layered files are migrated on load, older files must be regenerated).
# Generation never sees phi: the table is built once and reused for every angle.
# Layered compliance NPZs recompute FEM in-process so plate_response can be recovered.
python -m compliance_fem.contact_lookup_cli \
  --compliance outputs/rectangle/compliance_results.npz \
  --a 0.6 --kappa 30 --reciprocity-tol 1e-6 --output outputs/contact_lookup
# add --store-fields to keep the per-node bases in the NPZ (default: recomputed on demand)

# raw gamma debug query: alpha d_ax d_ay r_x r_y
# optional --contact-type {heel,interior,toe,full} restricts the reported best interval
python -m compliance_fem.query_contact_lookup \
  --lookup outputs/contact_lookup/contact_lookup.npz \
  --coefficients 0.1 0.0 -1e-3 -0.002 0.07

# interactive force-controlled GUI — rebuilds layered FEM + lookup from sidebar
# geometry/material/softplus params (defaults = layered_plate_sample + CLI a/κ);
# Fx/Fy are kN sliders; optional export to outputs/gui_contact_lookup
streamlit run src/compliance_fem/app.py
```

Lookup outputs:

- `contact_lookup.npz` — interval arrays (`contact_start/end_index`, `contact_start/end_x`,
  anchors, adjacent free ids, `contact_mask`, status, rejection reasons), six-column
  `scalar_lookup` `(n_records,6,6)`, `edge_responses`, `kf_matrix`,
  `Q_alpha_shoe_on_foot_basis`, `phi_ref`, `rigid_alpha_basis`, `schema_version=10`,
  and the prepared compliance matrix `field_solver_compliance`. The per-node bases
  (bottom displacement / reaction, top force, plate) are recomputed on demand by
  `lookup.record_fields(rows)`, bitwise identical to stored ones; `--store-fields`
  (or `generate_contact_lookup(..., store_fields=True)`) also saves them. See
  [docs/interval_contact.md](docs/interval_contact.md#stored-scalars-on-demand-nodal-fields).
  Layered and measured tables also store `plate_response` data (mesh connectivity,
  plate influence matrices or `plate_*_basis`, `has_plate_response=True`);
  measured-sole tables add the stored mesh and render influence matrices
- `contact_lookup.csv` — one row per interval with scalars per column, status and residuals
- `contact_lookup_metadata.json` — contact law, anchor definition, closure convention,
  frame and sign conventions, label counts, rejection counts, build time
- `generation_summary.json` — \(N_b\), theoretical / valid record counts per label,
  rejection counts, build time and file size
- diagnostic plots of the scalar lookup, interval validity and solver residuals

### Interval contact performance

| Lookup | \(N_b\) | intervals (valid / theoretical) | rejected | file | build | runtime per sample |
|--------|---------|------------------|---------|------|-------|-------------------|
| `outputs/contact_lookup` (rectangle 1 m × 0.5 m, 80×40) | 81 | 3321 / 3321 | 0 | 45.1 MB | 639 s | 90–105 ms (Auto, all intervals); 42 ms per gait frame (force-phi) |
| `outputs/layered_contact_lookup` (layered plate, nx=100) | 101 | 5151 / 5151 | 0 | 185 MB | 746 s | 18–70 ms per gait frame (passive toe, local-first search) |
| measured sole, 270 mm, 3 mm mesh | 126 | 8001 / 8001 | 0 | 19.6 MB (377 MB with `--store-fields`) | 47–73 s | 0.24 s per `SoleModel.step` |

Build times were measured with both lookups generating concurrently on one
machine; the rectangle and layered sizes are for files with stored nodal
fields. Generation cost grows like \(N_b^2\) intervals times one dense
factorization. Runtime contracts stored scalars for every interval and re-solves
nodal fields only for the few intervals that pass the exact edge prefilter.

The Streamlit app no longer loads a prebuilt NPZ as its primary input. Sidebar
controls expose layered geometry (\(L\), \(h_1/h_2\) heel/toe), softplus
(\(L-a\), \(\kappa\)), materials (\(E_1\), \(E_{\mathrm{heel}}\), \(E_{\mathrm{toe}}\),
\(\nu\), \(EI\)), mesh, and fixed-frame loads \(F_x,F_y\) (kN). Changing those
values recomputes compliance and the contact lookup (cached with
`st.cache_resource`). Defaults match `examples/layered_plate_sample.py` and the
contact-lookup CLI (`a=0.2262`, \(\kappa=160\)). Display units are mm / Pa /
N·mm² / kN; solvers use SI. Heights use a 0.5 mm floor (configs require \(h>0\));
Poisson ratios cap at 0.49 to avoid plane-strain lock.

The app has a **contact-mode** control (Auto / Heel / Interior / Toe / Full /
Specific, with start/end node inputs for Specific). Absent edge quantities show
**N/A**. The fixed-frame plot draws the contact interval as a solid curve on the
ground, the free bottom dashed, the heel and toe contact edges at \(l_h\), \(l_t\),
the adjacent free nodes, and the interval anchor as a grey open cross labelled
numerical (not a contact boundary). Geometry uses

\[
r^{\mathcal F}=r_a^{\mathcal F}+Q(\varphi)\bigl[(\mathbf X-\mathbf X_a)+s_d(\mathbf d^{\mathcal T}-\mathbf d_a^{\mathcal T})\bigr],
\qquad r_a^{\mathcal F}=(x_a,0),
\]

so the rigid rotation is never exaggerated; only the elastic part is scaled by
\(s_d\), which is not a physical amplitude. Because the rotating-frame contact
displacement is dominated by the rigid term \((Q^T-I)(\mathbf X-\mathbf X_a)\),
which \(s_d\) would also magnify, the **auto** scale is capped so the contact
interval cannot visibly drift off the ground; under appreciable rotation it
falls back to true scale, where the rotation is already the visible effect.
The plot also shows the ground line, the heel-to-toe chord, and the fixed-frame
resultant-force direction. When `plate_response` is present, the internal plate
is drawn from the Hermite superposition above; otherwise plate controls show **N/A**.

On large layered saddles, stacked-\(C\) reciprocity may sit near \(10^{-7}\)
(already handled by symmetrization / `--reciprocity-tol`).

## scikit-fem API notes

This project uses scikit-fem v12 public APIs:

- `ElementTriP1` / `ElementTriP2` and `ElementQuad1` / `ElementQuad2` with
  `ElementVector` for displacement.
- `Basis.nodal_dofs` to address all nodal displacement components when building
  rigid modes (`get_dofs().nodal['u^1']` omits some nodes).
- `Basis.get_dofs(boundary).nodal['u^2']` for vertical boundary DOFs.
- `lame_parameters(E, nu)` from `skfem.models.elasticity` (plane-strain Lamé
  constants in 2D).
- Contact lookup uses \(x_r=L/2\) for the restored rigid modes (distinct from the
  mesh-centroid gauge used when assembling compliance).
- The ramp \(\boldsymbol{\rho}_a\) uses existing top DOF coordinates; if no node
  coincides with \(a\), the toe moment is the FE nodal representation of that ramp.

## Installation

```bash
pip install -e ".[dev]"
```

## Running the compliance example

```bash
python -m compliance_fem.cli \
  --L 1.0 \
  --H 0.5 \
  --E 1.0e6 \
  --nu 0.3 \
  --nx 80 \
  --ny 40 \
  --order 1 \
  --output outputs/rectangle
```

or

```bash
python examples/rectangular_sample.py
```

Layered trapezoids with an inextensible internal plate:

```bash
python -m compliance_fem.cli \
  --geometry layered-plate \
  --L 0.30 \
  --h1-heel 0.025 \
  --h1-toe 0.015 \
  --h2-heel 0.020 \
  --h2-toe 0.030 \
  --E1 2.0e6 \
  --nu1 0.30 \
  --E-heel 5.0e5 \
  --E-toe 1.5e6 \
  --nu2 0.30 \
  --EI-plate 10.0 \
  --nx 100 \
  --ny1 12 \
  --ny2 12 \
  --element-order 1 \
  --output outputs/layered_plate
```

or

```bash
python examples/layered_plate_sample.py
```

Image-derived carbon-plated sole (two foams, partial curved plate) from the
normalized geometry CSV; `--shoe-length-mm` is the projected heel-bottom-to-toe
length and is required:

```bash
python -m compliance_fem.cli \
  --geometry measured-sole \
  --geometry-csv data/geometry/sole_geometry_normalized.csv \
  --shoe-length-mm 270 \
  --output outputs/measured_sole

python examples/measured_sole_validation.py --shoe-length-mm 270
```

All non-CSV parameters (shoe length, materials, plate EI, mesh, softplus `a` /
`kappa`, toe spring, shoe width, lookup directory) can instead come from a JSON
config such as `configs/measured_sole_270mm.json`. Explicit CLI flags override
the file:

```bash
python -m compliance_fem.measured_config_file build configs/measured_sole_270mm.json
python -m compliance_fem.contact_lookup_cli --config configs/measured_sole_270mm.json --kappa 120 --output outputs/k120
```

```python
from compliance_fem.api import SoleModel
model = SoleModel.from_config("configs/measured_sole_270mm.json")
```

In the GUI choose **Geometry model → Measured carbon-plated sole** (CSV, overall
shoe length, upper/lower foam, mesh size, refinements). See
[docs/measured_sole.md](docs/measured_sole.md) for the file format, conventions,
validation and limitations.

The rectangle path remains the default (`--geometry rectangle`). Contact lookup
still consumes the same nodal-force blocks; changing only
\(F_x\), \(F_y\), \(\phi\), \(\theta\) or the contact interval requires neither regenerating FEM nor
regenerating the lookup table, since all angle dependence is applied at runtime.
Changing geometry or material parameters does. Old vertical-only compliance NPZs
and any lookup file older than `schema_version=9` must be regenerated (v9
rectangle/layered lookups are migrated to v10 on load).
When generating a layered lookup from a compliance NPZ, the CLI recomputes the
FEM in-process so the factorization is available for `plate_response` recovery.

Outputs include:

- `compliance_results.npz` — compliance blocks, mass matrices, coordinates, errors
- `mesh.msh`, `mesh.vtk` — mesh for ParaView
- `Cbt_force_heatmap.png`, `Cbb_force_heatmap.png`
- `validation_*.png`, `validation_summary.json`

The layered NPZ additionally stores `geometry_type`, plate metadata, and
`x_plate` / `y_plate` / `y_top` for plotting. The GUI uses those arrays when
present and keeps rectangle rendering for lookups without `plate_response`.

## Tests

```bash
pytest
```

## Project layout

```text
src/compliance_fem/
  geometry.py             Gmsh rectangle mesh generation
  layered_geometry.py     Two-trapezoid mesh with shared plate interface
  measured_geometry.py    Normalized sole CSV loader, validation, exterior/regions/plate
  measured_mesh.py        Gmsh two-foam mesh with tagged selectors and plate polyline
  measured_render.py      Stored-mesh render section and render influence matrices
  api.py                  SoleModel: stateful step() interface for external codebases
  measured_config_file.py Measured-sole JSON config: parse, fingerprint, build/reuse lookup
  assembly.py             Plane-strain stiffness (constant and E(x))
  plate.py                Hermite Euler–Bernoulli plate assembly
  plate_response.py       Plate basis recovery, Hermite postprocess, runtime state
  constraints.py          Exact plate inextensibility rows B_p
  rigid_modes.py          Rigid-body modes and verification
  boundaries.py           Boundary selectors and mass matrices
  compliance.py           Saddle-point solves and compliance blocks
  validation.py           Full-bottom-contact analytical checks
  corotation.py           Frame math: Q(varphi), phi_ref, gamma, basis contraction
  contact_topology.py     Contact intervals I_ij, labels, mode filters, validation
  contact_basis.py        Six-column affine basis (closure + softplus shape + contact motion)
  contact_lookup.py       Interval contact lookup table (schema v10)
  toe_spring.py           Passive toe spring: exact equilibrium, roots, stability
  contact_query.py        Raw gamma superposition (lookup-level debug path)
  force_control.py        Fx/Fy/φ/θ runtime: K_F solve, all-node checks, interval search
  shape_render.py         Fixed-frame deformed-outline plot data
  app.py                  Streamlit GUI (live layered rebuild + contact mode)
  gui_params.py           GUI defaults, dual slider/number widgets, SI conversion
  contact_direct_fem.py   Sparse FEM cross-checks
  contact_plotting.py     Lookup / query plots
  contact_lookup_cli.py   Lookup generation CLI
  query_contact_lookup.py Lookup query CLI
  plotting.py             Compliance diagnostic figures
  cli.py                  Compliance command-line driver
  viscoelasticity/        Viscoelastic F_VE(t) → elastic F_e(t) mapper + CLI
  gait/                   Wang gait force-replay (passive-toe-spring θ by default) + CLI/GUI
docs/viscoelasticity.md   Viscoelastic force-map mathematics and usage
docs/gait_replay.md       Gait replay conventions, citations, limitations
docs/passive_toe_spring.md Passive toe spring model, signs, solver, limitations
docs/interval_contact.md  Single-interval contact model, schema v10, selection, limits
docs/measured_sole.md     Measured carbon-plated sole: CSV, conventions, validation, limits
docs/external_api.md      SoleModel.step(): loads / pitch in, displacements / moments out
configs/measured_sole_270mm.json  Example measured-sole configuration (all non-CSV parameters)
src/compliance_fem/pages/1_Gait_replay.py    Streamlit multipage entry for gait replay
```

