# Compliance FEM

Plane-strain finite-element computation of **vector** (component-major \(u,v\))
compliance matrices for a measured carbon-plated running-shoe sole (two foams
with an inextensible internal plate), plus a
single-contiguous-interval ground-contact lookup built from those nodal-force
compliance blocks (with a `plate_response` section). See
[docs/measured_sole.md](docs/measured_sole.md) and
[docs/interval_contact.md](docs/interval_contact.md).

## Physical problem

The body \(\Omega\) is the measured sole outline of length \(L\) (heel at
\(x=0\), toe tip at \(x=L\)), split into an upper and a lower foam region. Each
region is isotropic linear-elastic (the lower foam may grade from heel to toe
modulus), with unit out-of-plane thickness and **plane-strain** kinematics.

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
\(\psi(x)=\varphi_1(x)-\frac{x}{L-\ell_{\mathrm{toe}}}\varphi_1(L-\ell_{\mathrm{toe}})\), used by the passive toe spring
([docs/passive_toe_spring.md](docs/passive_toe_spring.md)). Here \(\ell_{\mathrm{toe}}\) is
the toe length in metres (softplus joint to toe tip), so the joint sits at
\(x=L-\ell_{\mathrm{toe}}\). Toe moment about
\(P_{\mathrm{toe}}=(L-\ell_{\mathrm{toe}},H_{\mathrm{mtp}})\) is

\[
T_{\mathrm{toe}}=\boldsymbol{\rho}^T\mathbf f_{t,y}-\boldsymbol{\eta}^T\mathbf f_{t,x},
\qquad
[\boldsymbol{\rho}]_j=\max\bigl(0,x_{t,j}-(L-\ell_{\mathrm{toe}})\bigr).
\]

`schema_version` is **11**. Files with any other schema version are
**rejected** with a regenerate message.

## Internal plate response

Plate fields are recovered from the **same** coupled
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
panel.

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
load-scaled tolerances. The mesh uses linear elements, so the nodal checks are
exact (the gap is piecewise linear).
Interior tension sets `disconnected_contact_warning`. Selection does **not** use
the tangential reaction, \(M_v\), \(x_{\mathrm{cm}}\), or \(T_{\mathrm{toe}}\).

Contact modes: **Auto**, **Heel** (\(i=0\)), **Interior**, **Toe**
(\(j=N_b-1\)), **Full** (forced) and **Specific** (forced, validated \((i,j)\)).
The search reports `candidate_search_method`:
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

### Generate lookup and GUI

```bash
# regenerate lookup (every contiguous contact interval, six-column affine basis).
# Generation never sees phi: the table is built once and reused for every angle.
# Runs mesh -> FEM -> lookup; skipped if lookup.output_dir already matches the config.
compliance-fem-contact-lookup configs/setup.json [--force] [--save-fem] [--skip-plots] [--quiet]
# set lookup.store_nodal_fields in the config to keep the per-node bases in the NPZ

# interactive force-controlled GUI — rebuilds the measured-sole FEM + lookup from sidebar
# geometry/material/softplus params;
# Fx/Fy are kN sliders; optional export to outputs/gui_contact_lookup
streamlit run src/compliance_fem/gui/app.py
```

Lookup outputs:

- `contact_lookup.npz` — interval arrays (`contact_start/end_index`, `contact_start/end_x`,
  anchors, adjacent free ids, `contact_mask`, status, rejection reasons), six-column
  `scalar_lookup` `(n_records,6,6)`, `edge_responses`, `kf_matrix`,
  `Q_alpha_shoe_on_foot_basis`, `phi_ref`, `rigid_alpha_basis`, `schema_version=11`,
  and the prepared compliance matrix `field_solver_compliance`. The per-node bases
  (bottom displacement / reaction, top force, plate) are recomputed on demand by
  `lookup.record_fields(rows)`, bitwise identical to stored ones; `lookup.store_nodal_fields: true`
  in the config also saves them. See
  [docs/interval_contact.md](docs/interval_contact.md#stored-scalars-on-demand-nodal-fields).
  Tables also store `plate_response` data (mesh connectivity,
  plate influence matrices or `plate_*_basis`, `has_plate_response=True`),
  the stored mesh and render influence matrices
- `contact_lookup.csv` — one row per interval with scalars per column, status and residuals
- `contact_lookup_metadata.json` — contact law, anchor definition, closure convention,
  frame and sign conventions, label counts, rejection counts, build time
- `generation_summary.json` — \(N_b\), theoretical / valid record counts per label,
  rejection counts, build time and file size
- diagnostic plots of the scalar lookup, interval validity and solver residuals

### Interval contact performance

| Lookup | \(N_b\) | intervals (valid / theoretical) | rejected | file | build | runtime per sample |
|--------|---------|------------------|---------|------|-------|-------------------|
| measured sole, 270 mm, 3 mm mesh | 126 | 8001 / 8001 | 0 | 19.6 MB (377 MB with stored nodal fields) | 47–73 s | 0.24 s per `SoleModel.step` |

Generation cost grows like \(N_b^2\) intervals times one dense
factorization. Runtime contracts stored scalars for every interval and re-solves
nodal fields only for the few intervals that pass the exact edge prefilter.

The Streamlit app no longer loads a prebuilt NPZ as its primary input. Sidebar
controls expose the measured-sole geometry (CSV, shoe length, mesh size,
refinements), softplus (toe length \(\ell_{\mathrm{toe}}\), \(\kappa\)), materials
(\(E_1\), \(E_{\mathrm{heel}}\), \(E_{\mathrm{toe}}\), \(\nu\), \(EI\)), and
fixed-frame loads \(F_x,F_y\) (kN). Changing those values recomputes compliance
and the contact lookup (cached with `st.cache_resource`). Display units are
mm / Pa / N·mm² / kN; solvers use SI. Poisson ratios cap at 0.49 to avoid
plane-strain lock.

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

On large saddles, stacked-\(C\) reciprocity may sit near \(10^{-7}\)
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
- The ramp \(\boldsymbol{\rho}\) uses existing top DOF coordinates; if no node
  coincides with \(L-\ell_{\mathrm{toe}}\), the toe moment is the FE nodal representation of that ramp.

## Installation

```bash
pip install -e ".[dev]"
```

## Running the compliance example

Image-derived carbon-plated sole (two foams, partial curved plate) from the
normalized geometry CSV. Every measured-sole parameter (geometry CSV, shoe
length, materials, plate EI, mesh, softplus toe length / `kappa`, toe spring,
shoe width, lookup directory) comes from a JSON config such as
`configs/setup.json`, which the CLIs use as is:

```bash
compliance-fem-contact-lookup configs/setup.json --save-fem
```

To change a parameter, edit the config (or copy it); there are no
per-parameter override flags. The lookup is written to `lookup.output_dir`
(default `outputs/contact_lookup`). If that directory already holds a lookup
with the same fingerprint the build is skipped; `--force` rebuilds it.

```python
from compliance_fem.api import SoleModel
model = SoleModel.from_config("configs/setup.json")
```

See [docs/measured_sole.md](docs/measured_sole.md) for the file format,
conventions, validation and limitations.

Contact lookup consumes the same nodal-force blocks; changing only
\(F_x\), \(F_y\), \(\phi\), \(\theta\) or the contact interval requires neither regenerating FEM nor
regenerating the lookup table, since all angle dependence is applied at runtime.
Changing geometry or material parameters does. Compliance NPZs and lookups
written by an older version of the code (a different schema version) must be
regenerated.

`--save-fem` also writes the FEM artifacts to `<output_dir>/fem/`:

- `compliance_results.npz` — compliance blocks, mass matrices, coordinates, errors
- `mesh.msh`, `mesh.vtk` — mesh for ParaView
- `Cbt_force_heatmap.png`, `Cbb_force_heatmap.png`
- `fem_summary.json` — residual and reciprocity checks

The NPZ stores the `SoleConfig` settings under their field names, plate metadata, and
`x_plate` / `y_plate` / `y_top` for plotting. The GUI uses those arrays when
present.

## Public interface

Everything a user runs is listed here; all other modules (the rest of `contact`,
`fem`, `geometry`, `gait`, `plotting`, `gui`) are internal and may change without notice.

- `streamlit run src/compliance_fem/gui/app.py` — GUI, including the Gait replay page
- `compliance-fem-contact-lookup CONFIG [--force] [--save-fem] [--skip-plots] [--quiet]` — build mesh, FEM and lookup
- `visco-force-map ...` — viscoelastic → elastic force history CSV ([docs/viscoelasticity.md](docs/viscoelasticity.md))
- `compliance_fem.api` — `SoleModel` (`from_config`, `step`, `reset`) and `SoleState`
  ([docs/external_api.md](docs/external_api.md))
- `compliance_fem.contact.model_setup` — `load_model_setup`, `save_model_setup`, `ensure_lookup_exists`
  (the lookup command's Python equivalent), `lookup_matches_setup`, `matching_prebuilt_lookup`, `stored_lookup_fingerprint`
- `compliance_fem.viscoelasticity` — `create_material` and the material configs
- `compliance_fem` — `SoleConfig`, `compute_compliance`, `ComplianceResult`

## Tests

```bash
pytest
```

## Project layout

```text
src/compliance_fem/
  api.py                  SoleModel: stateful step() interface for external codebases
  geometry/
    profile.py            Normalized sole CSV loader, validation, exterior/regions/plate
    mesh.py               Gmsh two-foam mesh with tagged selectors and plate polyline
    render.py             Stored-mesh render section and render influence matrices
    gmsh_util.py          Gmsh session helpers
  fem/
    assembly.py           Plane-strain stiffness (constant and E(x))
    plate.py              Hermite Euler–Bernoulli plate assembly
    plate_response.py     Plate basis recovery, Hermite postprocess, runtime state
    constraints.py        Exact plate inextensibility rows B_p
    rigid_modes.py        Rigid-body modes and verification
    boundaries.py         Boundary selectors and mass matrices
    compliance.py         Saddle-point solves and compliance blocks
  contact/
    config.py             SoleConfig and foam materials
    model_setup.py        ModelSetup JSON config: parse, fingerprint, build/reuse lookup
    corotation.py         Frame math: Q(varphi), phi_ref, gamma, basis contraction
    topology.py           Contact intervals I_ij, labels, mode filters, validation
    basis.py              Six-column affine basis (closure + softplus shape + contact motion)
    lookup.py             Interval contact lookup table
    toe_spring.py         Passive toe spring: exact equilibrium, roots, stability
    force_control.py      Fx/Fy/φ/θ runtime: K_F solve, all-node checks, interval search
    direct_fem.py         Sparse FEM cross-checks
  plotting/
    compliance.py         Compliance diagnostic figures
    contact.py            Lookup diagnostic plots
    shape_render.py       Fixed-frame deformed-outline plot data
  cli/
    contact_lookup.py     Lookup generation CLI
    visco_force_map.py    Viscoelastic force-map CLI
  gui/
    app.py                Streamlit GUI (live measured-sole rebuild + contact mode)
    params.py             GUI defaults, angle-slider config, dual slider/number widgets, SI conversion
    rebuild_worker.py     Subprocess worker that rebuilds the lookup
    gait_page.py          Gait-replay page
    gait_plots.py         Gait-replay plotly figures
    pages/1_Gait_replay.py Streamlit multipage entry for gait replay
  viscoelasticity/        Viscoelastic F_VE(t) → elastic F_e(t) mapper
  gait/                   Wang gait force-replay backend for the GUI page
docs/viscoelasticity.md   Viscoelastic force-map mathematics and usage
docs/gait_replay.md       Gait replay conventions, citations, limitations
docs/passive_toe_spring.md Passive toe spring model, signs, solver, limitations
docs/interval_contact.md  Single-interval contact model, schema, selection, limits
docs/measured_sole.md     Measured carbon-plated sole: CSV, conventions, validation, limits
docs/external_api.md      SoleModel.step(): loads / pitch in, displacements / moments out
configs/setup.json             Example measured-sole configuration (all non-CSV parameters)
```

