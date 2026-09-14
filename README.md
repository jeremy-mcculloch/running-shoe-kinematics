# Compliance FEM

Plane-strain finite-element computation of **vector** (component-major \(u,v\))
compliance matrices for a homogeneous rectangular elastic body or a layered
trapezoidal foam pair with an inextensible internal plate, plus a schema-v6
heel/full/toe contact-topology lookup built from those nodal-force compliance
blocks (layered tables may include an optional `plate_response` section).

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

**Contact-edge lookup uses only the stacked nodal-force matrices** \(\mathbf C^{F}_{tt}\),
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

## Contact-edge lookup

Schema v6 stores three contiguous ground-contact topologies. Topology is stored
explicitly in `contact_type_codes` and is **never** inferred from the material
edge coordinate \(l\). Heel and toe families are solved independently (no
reflection of one into the other). With \(N_b\) bottom nodes the table has
\(2N_b-1\) records: \(N_b-1\) heel, \(N_b-1\) toe, and exactly one full.

| Topology | Contact interval | Contact nodes | Free nodes | Edge index \(i\) |
|----------|------------------|---------------|------------|------------------|
| **HEEL** | \([0,l]\) | \(\{0,\ldots,i\}\) | \(\{i+1,\ldots,N_b-1\}\) | \(i=0,\ldots,N_b-2\) |
| **TOE** | \([l,L]\) | \(\{i,\ldots,N_b-1\}\) | \(\{0,\ldots,i-1\}\) | \(i=1,\ldots,N_b-1\) |
| **FULL** | \([0,L]\) | \(\{0,\ldots,N_b-1\}\) | \(\emptyset\) | none (one record) |

The transition node \(i\) always belongs to the contact set. Full contact has no
lift-off edge; its basis modes use the numerical anchor \(x_a=L/2\) (a
decomposition point only, not a physical contact edge). `--include-endpoints` is
retained for CLI compatibility but ignored: all three topologies are always
enumerated.

For each record the code solves the dense boundary saddle system

\[
\mathbf A_i
\begin{pmatrix}
\mathbf F_t\\
\mathbf F_c\\
\boldsymbol\alpha
\end{pmatrix}
=
\begin{pmatrix}
\mathbf W_t\\
\mathbf W_c\\
\mathbf 0
\end{pmatrix},
\]

where \(\mathbf W_t\) and \(\mathbf W_c\) have five columns each (component-major
\([u;v]\)) describing prescribed **rotating-frame** boundary displacement.
The rotation-mode lever uses the record anchor \(x_a=l_i\) for heel/toe and
\(x_a=L/2\) for full; the same five modes apply to every topology:

| \(k\) | name | top \((u_t,v_t)\) | contact \((u_c,v_c)\) |
|---|------|-------------------|------------------------|
| 0 | `top_shape_phi1` | \((0,\ \varphi_1(x))\) | \((0,0)\) |
| 1 | `contact_translation_x` | \((0,0)\) | \((1,0)\) |
| 2 | `contact_translation_y` | \((0,0)\) | \((0,1)\) |
| 3 | `contact_rotation_x` | \((0,0)\) | \((x-x_a,\ 0)\) |
| 4 | `contact_rotation_y` | \((0,0)\) | \((0,\ x-x_a)\) |

The single top shape mode is the endpoint-fixed softplus ramp

\[
\varphi_1(x)=s(x)-\Bigl(1-\tfrac{x}{L}\Bigr)s(0)-\tfrac{x}{L}\,s(L),
\qquad \varphi_1(0)=\varphi_1(L)=0,
\]

with no normalization. Its sign lives in one place, `corotation.SHAPE_MODE_SIGN`.

\(\boldsymbol\alpha\) restores the three rigid modes (horizontal translation,
vertical translation, rotation \(u=-(y-y_r)\), \(v=(x-x_r)\)) about
\(x_r=L/2\), \(y_r=0\). Equilibrium \(R_t^T F_t + R_c^T F_c = 0\) occupies the
last three rows. \(\mathbf A_i\) is factorized once per record and all five
right-hand sides are solved together; the global stiffness matrix is **not**
refactorized per record.

Saved scalars per mode \(k\in\{0,\dots,4\}\) are topology-independent:

`[Fx, Fy, Mv, Mz, edge_free_gap_v, edge_free_gap_u, edge_contact_reaction_x,
edge_contact_reaction_y, M_toe, M_toe_vertical]`,

where \(M_v=x_t^T f_{t,y}\) and \(M_z\) is the top-force moment about \((0,0)\).
Edge free-gap / contact-reaction components are taken at the free/contact nodes
adjoining the transition (\(i+1\) / \(i\) for heel, \(i-1\) / \(i\) for toe) and
are `NaN` for full contact. Toe moment about \(P_{\mathrm{toe}}=(a,H_a)\) is

\[
T_{\mathrm{toe}}=\boldsymbol{\rho}_a^T\mathbf f_{t,y}-\boldsymbol{\eta}_a^T\mathbf f_{t,x},
\qquad
[\boldsymbol{\rho}_a]_j=\max(0,x_{t,j}-a).
\]

\(H_a\) is interpolated from \(y_{\mathrm{top}}\). On a flat top, \(\eta_a=0\).
If \(a\) is not a top node, the existing nodal ramp is used (no remesh). Because
\(\rho_a\) and \(\eta_a\) are **reference** levers about \((a,H_a)\), the toe
moment superposes as a scalar exactly like every other saved quantity.

Full contact additionally stores `corner_reactions_local` with shape
`[basis, corner={heel,toe}, component={x,y}]` (local rotating-frame nodal forces
at \(x=0\) and \(x=L\)). Those entries are finite only on the full-contact row;
heel/toe rows store `NaN`.

The lookup schema version is \(6\), with `scalar_lookup` shape `(n_records, 5, 10)`.
Files that are not `schema_version=6` **hard-fail** with a regenerate message.
Older tables cannot be migrated: v4 stored only the toe family (no full-contact
record or corner reactions); v5 has heel/full/toe but no `plate_response` fields.

## Internal plate response

For layered lookups, plate fields are recovered from the **same** coupled
foam–plate FEM factorization used while generating the contact table—not a
separate plate solver. Each record stores five-mode basis fields
\(u\), \(v\), \(\theta\), \(\lambda\), and axial force (same basis order as the
boundary modes). Axial constraint rows \(B_p\) are unscaled
(\(t_p\cdot(u_{j+1}-u_j)=0\)), so the Lagrange multiplier already has force
units and equals the element axial constraint force.

At runtime the stored modes are contracted with the same \(\boldsymbol\gamma\)
as the foam scalars. The centerline is rebuilt with cubic Hermite interpolation;
curvature \(\kappa=w''\), bending moment \(M=EI\,\kappa\), and shear
\(V=EI\,w'''\). Fixed-frame placement uses the shared foam anchor transform
\(r^{\mathcal F}=r_a^{\mathcal F}+Q(\varphi)[(\mathbf X-\mathbf X_a)+s_d(\mathbf d^{\mathcal T}-\mathbf d_a^{\mathcal T})]\);
the elastic display scale \(s_d\) never multiplies \(\varphi\).

The Streamlit GUI exposes plate toggles (show plate, undeformed plate, nodes,
rotations), a color quantity, samples/element, and a plate-results summary
panel. Rectangle lookups are still schema 6 but omit `plate_response`; the GUI
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

Contact nodes move rigidly with the frame relative to the record anchor
\(x_a\) (\(l_i\) for heel/toe, \(L/2\) for full),

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
-\bigl(\alpha F^{\mathcal T}_{\varphi_1}+r_x F^{\mathcal T}_{B_{rx}}+r_y F^{\mathcal T}_{B_{ry}}\bigr).
\]

Singular values, determinant, and condition number of \(K_F\) are reported;
records below the force threshold or above `kf_cond_warn` are **flagged and
skipped**, never regularized. Horizontal/vertical coupling means
\(F_x^\star=0\) does **not** imply \(d_{a,x}=0\). Reconstructed \(F_x,F_y\) are
verified against the targets in both frames. Every scalar and vector output is
then superposed with the same \(\boldsymbol\gamma\) through one shared
`contract_basis` contraction.

For heel/toe the rotated contact-edge coordinate is \(x_{l,\mathrm{rot}}=l_i+d_{a,x}\)
(NaN for full). The lookup itself is indexed by topology plus the **material**
coordinate \(l_i\) (NaN for full); the shared gauge is
\(r_a^{\mathcal F}=(x_a,0)\).

### Contact selection (fixed-frame normal gap and \(R_n\) only)

Gaps and reactions are tested in the **fixed** frame, since the ground normal is
fixed while the body rotates:

\[
g_j^{\mathcal F}=\sin\varphi\,\Delta x_j^{\mathcal T}+\cos\varphi\,\Delta y_j^{\mathcal T},
\qquad
R_n^{\mathcal F}=\sin\varphi\,R_{c,x}^{\mathcal T}+\cos\varphi\,R_{c,y}^{\mathcal T},
\]

with \(\Delta x_j=(x_j+u_j)-(x_a+d_{a,x})\) and \(\Delta y_j=v_j-d_{a,y}\).
Selection does **not** use the tangential reaction \(R_t\), \(M_v\),
\(x_{\mathrm{cm}}\), or \(T_{\mathrm{toe}}\).

**AUTO** routing starts from the full-contact corner normals \(R_n\) at \(x=0\)
(heel) and \(x=L\) (toe):

- both compressive \(\Rightarrow\) select full immediately
- heel tensile \(\Rightarrow\) search the toe family only
- toe tensile \(\Rightarrow\) search the heel family only
- both tensile (or full \(K_F\) ill-conditioned) \(\Rightarrow\) search both partial families

Full contact is **never** returned silently when either corner is tensile.
Forced modes restrict the pool to **Heel**, **Full**, or **Toe**. A partial
record is admissible when free \(g^{\mathcal F}\) and contact \(R_n^{\mathcal F}\)
satisfy the unilateral inequalities within \(\tau_g,\tau_R\); full validity uses
the two corner normals only (interior tension stays diagnostic). Among admissible
well-conditioned records in the routed pool, a deterministic representative
minimizes the edge score on the free/contact edge pair. If none are admissible,
the least-violating well-conditioned record is returned and marked approximate.

The dimensionless violation score uses the worst free-gap and contact-reaction
violations (plus force residual), with denominators \(\tau_g,\tau_R\) when
positive and otherwise data-driven gap/reaction scales.

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
# regenerate lookup (schema v6, heel/full/toe, five-mode co-rotating basis) — required after upgrading.
# Generation never sees phi: the table is built once and reused for every angle.
# Layered compliance NPZs recompute FEM in-process so plate_response can be recovered.
python -m compliance_fem.contact_lookup \
  --compliance outputs/rectangle/compliance_results.npz \
  --a 0.6 --kappa 30 --reciprocity-tol 1e-6 --output outputs/contact_lookup

# raw gamma debug query: alpha d_ax d_ay r_x r_y
# optional --contact-type {heel,toe,full} restricts the reported family best
python -m compliance_fem.query_contact_lookup \
  --lookup outputs/contact_lookup/contact_lookup.npz \
  --coefficients 0.1 0.0 -1e-3 -0.002 0.07

# interactive force-controlled GUI — rebuilds layered FEM + lookup from sidebar
# geometry/material/softplus params (defaults = layered_plate_sample + CLI a/κ);
# Fx/Fy are kN sliders; optional export to outputs/gui_contact_lookup
streamlit run src/compliance_fem/app.py
```

Lookup outputs:

- `contact_lookup.npz` — `scalar_lookup` `(n_records,5,10)`, `contact_type_codes`,
  `corner_reactions_local`, `top_force_x/y_basis`, `gap_u/v_basis`,
  `reaction_x/y_basis`, `kf_matrix` `(n_records,2,2)`, `kf_svals`, `phi_ref`,
  `schema_version=6`; layered tables may also store `plate_response` arrays
  (`plate_*_basis`, mesh connectivity, `has_plate_response=True`)
- `contact_lookup.csv` — scalars per mode plus \(K_F\) cond/det/\(\sigma_{\min}\) and residuals
- `contact_lookup_metadata.json` — topologies, frame and angle conventions, `phi_ref`,
  the five-mode basis order, `force_frame`, the shape-mode formula, plate-response
  flags when present, and the regenerate note
- diagnostic plots of \(F_x\), \(F_y\), \(M_v\), edge free gap / contact reaction,
  \(T_{\mathrm{toe}}\), and residuals vs \(l\) (per family)

The Streamlit app no longer loads a prebuilt NPZ as its primary input. Sidebar
controls expose layered geometry (\(L\), \(h_1/h_2\) heel/toe), softplus
(\(L-a\), \(\kappa\)), materials (\(E_1\), \(E_{\mathrm{heel}}\), \(E_{\mathrm{toe}}\),
\(\nu\), \(EI\)), mesh, and fixed-frame loads \(F_x,F_y\) (kN). Changing those
values recomputes compliance and the contact lookup (cached with
`st.cache_resource`). Defaults match `examples/layered_plate_sample.py` and the
contact-lookup CLI (`a=0.2262`, \(\kappa=160\)). Display units are mm / Pa /
N·mm² / kN; solvers use SI. Heights use a 0.5 mm floor (configs require \(h>0\));
Poisson ratios cap at 0.49 to avoid plane-strain lock.

The app also has a **contact-mode** control (Auto / Heel / Full / Toe). Free-edge
gap and contact-edge coordinate show **N/A** for full contact. The fixed-frame plot
shades the contact interval \([0,l]\), \([l,L]\), or \([0,L]\), marks the physical
contact edge for heel/toe, and uses a distinct open-square marker for the full-contact
numerical anchor \(x_a=L/2\). Geometry uses

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

The rectangle path remains the default (`--geometry rectangle`). Contact lookup
still consumes the same nodal-force blocks; changing only topology / \(l\),
\(F_x\), \(F_y\), \(\phi\), \(\theta\) requires neither regenerating FEM nor
regenerating the lookup table, since all angle dependence is applied at runtime.
Changing geometry or material parameters does. Old vertical-only compliance NPZs
and any lookup file that is not `schema_version=6` must be regenerated.
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
  assembly.py             Plane-strain stiffness (constant and E(x))
  plate.py                Hermite Euler–Bernoulli plate assembly
  plate_response.py       Plate basis recovery, Hermite postprocess, runtime state
  constraints.py          Exact plate inextensibility rows B_p
  rigid_modes.py          Rigid-body modes and verification
  boundaries.py           Boundary selectors and mass matrices
  compliance.py           Saddle-point solves and compliance blocks
  validation.py           Full-bottom-contact analytical checks
  corotation.py           Frame math: Q(varphi), phi_ref, gamma, basis contraction
  contact_topology.py     Heel / full / toe sets, record enumeration, anchors
  contact_basis.py        Five-mode boundary basis (softplus shape + contact motion)
  contact_lookup.py       Contact-topology lookup table (schema v6)
  contact_query.py        Raw gamma superposition (lookup-level debug path)
  force_control.py        Fx/Fy/φ/θ runtime: force rotation, 2x2 K_F solve, routing
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
docs/viscoelasticity.md   Viscoelastic force-map mathematics and usage
```

