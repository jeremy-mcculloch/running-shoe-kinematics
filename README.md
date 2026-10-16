# Compliance FEM

Plane-strain finite-element computation of **vector** (component-major \(u,v\))
compliance matrices for a measured carbon-plated running-shoe sole (two foams
with an inextensible internal plate), condensed once into shared boundary
operators for a **continuous unilateral ground-contact** solve: every bottom
node may touch the ground and the active set comes from one warm-started
complementarity solve per evaluation (no contact-interval table). See
[docs/continuous_contact.md](docs/continuous_contact.md) and
[docs/measured_sole.md](docs/measured_sole.md).

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

## Continuous ground contact

Every bottom node is a potential contact node on the flat ground `y = 0`.
With \(\lambda_k \ge 0\) the compressive normal nodal reaction and \(g_k\)
the fixed-frame normal gap (positive open), the contact state solves

\[
0\le\boldsymbol\lambda\perp\mathbf g\ge 0,\qquad
\mathbf g=\mathbf q+T_y\mathbf 1+H\boldsymbol\lambda,\qquad
\mathbf 1^T\boldsymbol\lambda=-F^\star_y,
\]

where \(T_y\) (the vertical frame translation) is the Lagrange multiplier of
the force constraint. \(H=N\,G_{bb}\,B\) comes from the **top-clamped bottom
compliance** \(G_{bb}\) condensed once from the nodal-force blocks
(symmetric positive definite), \(N\) takes the ground-normal component of a
local displacement and \(B\) maps \(\lambda\) to local nodal forces. A
warm-started primal-dual active-set method (Murty least-index fallback against
cycling) solves it; its cost depends on the active set, not on the number of
possible intervals. Nodes enter and leave contact through \(g=\lambda=0\), so
forces, displacements, plate fields and the centre of pressure are continuous
across active-set changes.

Tangential contact is **proportional shear**: every active node carries the
fixed-frame ground force \((\mu_{\mathrm{eff}},1)\lambda_k\) with
\(\mu_{\mathrm{eff}}=F^\star_x/F^\star_y\), so both prescribed force
components are met exactly and the horizontal translation is a gauge.
Disconnected contact patches are reported separately; a derived contact-edge
estimate is display only. Full formulation, tolerances, storage, complexity,
validation and limitations: [docs/continuous_contact.md](docs/continuous_contact.md).

The model file `contact_lookup.npz` has `schema_version` **13** (shared
operators, plate influence, render section, metadata). Files with any other
schema version are **rejected** with a regenerate message.

## Internal plate response

Plate fields are recovered from the **same** coupled foam–plate FEM
factorization used to build the compliance blocks: one multi-RHS solve gives
the plate nodal \(u\), \(v\), \(\theta\) and the axial multipliers per unit
boundary load and per unit rigid amplitude. At runtime the contact solution's
\([F_t; f_b]\) and rigid amplitudes give the plate state by two small matrix
products. Axial constraint rows \(B_p\) are unscaled
(\(t_p\cdot(u_{j+1}-u_j)=0\)), so the multiplier has force units and equals
the element axial constraint force. The centreline is rebuilt with cubic
Hermite interpolation; curvature \(\kappa=w''\), bending moment
\(M=EI\,\kappa\), shear \(V=EI\,w'''\).

## Co-rotating top frame

\(\mathcal F\) is the fixed ground frame; \(\mathcal T\) is the top-attached
frame, the reference mesh rotated by \(\varphi\) with
\(Q(\varphi)=\begin{bmatrix}\cos\varphi&-\sin\varphi\\ \sin\varphi&\cos\varphi\end{bmatrix}\).
A body point sits at \(r^{\mathcal F}=T+Q(\varphi)(\mathbf X+\mathbf d)\) with
\(\mathbf d\) the local displacement and one global translation \(T\) shared by
every active set. With a prescribed toe angle \(\varphi=\phi-\phi_{\mathrm{ref}}\),
\(\phi_{\mathrm{ref}}=\operatorname{atan2}(y_{\mathrm{top}}(L)-y_{\mathrm{top}}(0),L)\);
with the passive toe spring \(\phi\) is the rearfoot (heel → MTP) angle and
\(\varphi(\alpha)=\phi-\operatorname{atan2}(dy_{\mathrm{mtp}}+\alpha\varphi_1(x_{\mathrm{mtp}}),L-\ell_{\mathrm{toe}})\).

The top displacement is the single endpoint-fixed softplus mode
\(\mathbf d_t=\alpha\,(0,\varphi_1(x))\), \(\alpha=\tan\theta\),

\[
\varphi_1(x)=s(x)-\Bigl(1-\tfrac{x}{L}\Bigr)s(0)-\tfrac{x}{L}\,s(L),
\qquad \varphi_1(0)=\varphi_1(L)=0,
\]

with no normalization; its sign lives in `corotation.SHAPE_MODE_SIGN`.

**Deliberate approximation.** Boundary positions use the exact finite rotation,
while foam strain is evaluated with the linear strain–displacement operator.
This keeps the model linear-elastic so the boundary operators are precomputed
once, independent of \(\phi\). It is **not objective for locally large
rotations** and is not a geometrically nonlinear FEM.

## Force-controlled inputs and the passive toe spring

Inputs are the fixed-frame top resultants \(F^\star_x,F^\star_y\) (foot on
shoe, \(F_{\mathrm{top}}=-\mathrm{GRF}\), so stance has \(F^\star_y<0\)),
the pitch \(\phi\) and either a prescribed \(\theta\) or the passive toe
spring. The spring (\(k=25\) N·m/rad, \(\theta_0=0\)) solves the exact balance

\[
Q_{\alpha,\mathrm{shoe\to foot}}(\alpha)=k\,\frac{\arctan\alpha-\theta_0}{1+\alpha^2},
\qquad Q_{\alpha,\mathrm{shoe\to foot}}=w\,(-\boldsymbol\psi^T\mathbf f_{t,y}),
\]

by a nested Brent solve in \(\theta\) with a converged contact solve per
residual evaluation, continued from the previous sample
([docs/passive_toe_spring.md](docs/passive_toe_spring.md)). The toe moment about
\(P_{\mathrm{toe}}=(L-\ell_{\mathrm{toe}},H_{\mathrm{mtp}})\) is
\(T_{\mathrm{toe}}=\boldsymbol{\rho}^T\mathbf f_{t,y}-\boldsymbol{\eta}^T\mathbf f_{t,x}\),
\([\boldsymbol{\rho}]_j=\max(0,x_{t,j}-(L-\ell_{\mathrm{toe}}))\).

### 2.5D out-of-plane stiffness

Every converged evaluation also returns auxiliary linearized out-of-plane
stiffnesses for 3D use: lateral \(k_z\) (N/m), roll \(k_{\mathrm{roll}}\) and
yaw \(k_{\mathrm{yaw}}\) (N·m/rad), from the frozen active set's pure-normal
tangent \(H_{AA}^{-1}\), the shoe width (transverse strips) and an elastic
lateral-to-normal stiffness ratio \(\eta_t\). They never change the sagittal
solution. See [docs/out_of_plane_stiffness.md](docs/out_of_plane_stiffness.md).

### Sign conventions

- \(\phi\), \(\phi_{\mathrm{ref}}\) and \(\varphi\) are counterclockwise positive.
- Positive fixed-frame gap = open gap above the ground.
- Positive normal reaction \(\lambda\) = upward compressive nodal force from the ground.
- Positive top \(F_y\) = upward; positive top \(F_x\) = rightward (heel → toe).
- Tangential reactions follow the shear ratio: \(\tau_k=(F^\star_x/F^\star_y)\,\lambda_k\).

### Build the contact model and run the GUI

```bash
# mesh -> FEM -> shared contact operators; skipped if lookup.output_dir already matches the config
compliance-fem-contact-lookup configs/setup.json [--force] [--save-fem] [--skip-plots] [--quiet]

# interactive GUI: rebuilds the measured-sole FEM + contact model from the sidebar,
# continuous contact with the passive toe spring or a prescribed θ; gait replay page
streamlit run src/compliance_fem/gui/app.py

# validation study (direct FEM checks, continuity at every active-set change)
python scripts/validate_continuous_contact.py configs/setup.json
```

Outputs in `lookup.output_dir`: `contact_lookup.npz` (schema 13),
`measured_sole_config.json` and `generation_summary.json` (operator symmetry
and eigenvalue checks, build time, file size, fingerprint).

### Performance

| model | \(N_b\) | build (after FEM) | file | contact solve | gait sample (with toe spring) |
|---|---|---|---|---|---|
| measured sole, 270 mm, 3 mm mesh | 126 | 1.5 s | 19.4 MB | ~0.3 ms warm | ~2 ms (300-frame stance: 0.6 s) |

The GUI exposes the measured-sole geometry (CSV, shoe length, mesh size,
refinements), softplus toe length and \(\kappa\), materials and the
fixed-frame loads per metre width; changing model parameters rebuilds the FEM
and the contact model in a subprocess (cached). Display units are mm / Pa /
N·mm² / kN; solvers use SI. The deformed shape uses
\(r=r_p+Q[(\mathbf X-\mathbf X_p)+s_d(\mathbf d-\mathbf d_p)]\) about the
reaction-weighted contact centroid, so the display scale \(s_d\) multiplies
only the elastic part, never \(\varphi\).

On large saddles, stacked-\(C\) reciprocity may sit near \(10^{-7}\)
(already handled by symmetrization / `reciprocity_tol`).

## scikit-fem API notes

This project uses scikit-fem v12 public APIs:

- `ElementTriP1` / `ElementTriP2` and `ElementQuad1` / `ElementQuad2` with
  `ElementVector` for displacement.
- `Basis.nodal_dofs` to address all nodal displacement components when building
  rigid modes (`get_dofs().nodal['u^1']` omits some nodes).
- `Basis.get_dofs(boundary).nodal['u^2']` for vertical boundary DOFs.
- `lame_parameters(E, nu)` from `skfem.models.elasticity` (plane-strain Lamé
  constants in 2D).
- The contact operator, plate and render influences use \(x_r=L/2\) for the
  restored rigid modes (distinct from the mesh-centroid gauge used when
  assembling compliance).
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
per-parameter override flags. The contact model is written to `lookup.output_dir`
(default `outputs/contact_lookup`). If that directory already holds a schema-13
model with the same fingerprint the build is skipped; `--force` rebuilds it.

```python
from compliance_fem.api import SoleModel
model = SoleModel.from_config("configs/setup.json")
```

See [docs/measured_sole.md](docs/measured_sole.md) for the file format,
conventions, validation and limitations.

The contact model consumes the same nodal-force blocks; changing only
\(F_x\), \(F_y\), \(\phi\) or \(\theta\) requires neither regenerating FEM nor the
contact model, since all load and angle dependence is applied at runtime.
Changing geometry or material parameters does. Compliance NPZs and contact
models written by an older version of the code (a different schema version)
must be regenerated.

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
- `compliance-fem-contact-lookup CONFIG [--force] [--save-fem] [--skip-plots] [--quiet]` — build mesh, FEM and contact model
- `visco-force-map ...` — viscoelastic → elastic force history CSV ([docs/viscoelasticity.md](docs/viscoelasticity.md))
- `compliance_fem.api` — `SoleModel` (`from_config`, `step`, `reset`) and `SoleState`
  ([docs/external_api.md](docs/external_api.md))
- `compliance_fem.contact.model_setup` — `load_model_setup`, `save_model_setup`, `ensure_lookup_exists`
  (the build command's Python equivalent), `lookup_matches_setup`, `matching_prebuilt_lookup`, `stored_lookup_fingerprint`
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
    render.py             Render section: mesh, per-element moduli, render influence, strain energy
    gmsh_util.py          Gmsh session helpers
  fem/
    assembly.py           Plane-strain stiffness (constant and E(x))
    plate.py              Hermite Euler–Bernoulli plate assembly
    plate_response.py     Plate influence matrices, Hermite postprocess, runtime state
    constraints.py        Exact plate inextensibility rows B_p
    rigid_modes.py        Rigid-body modes and verification
    boundaries.py         Boundary selectors and mass matrices
    compliance.py         Saddle-point solves and compliance blocks
  contact/
    sole_config.py        SoleConfig and foam materials
    model_setup.py        ModelSetup JSON config: parse, fingerprint, build/reuse the contact model
    compliance_blocks.py  Nodal-force compliance blocks: extraction, reciprocity check, symmetrization
    contact_operator.py   Top-clamped bottom operators G_bb etc. (shared, condensed once)
    active_set.py         Warm-started primal-dual active-set solver for the contact LCP
    continuous_contact.py One-pose contact solve, signs, diagnostics, patches
    continuous_toe.py     Exact passive toe spring coupled to the contact solve
    out_of_plane.py       2.5D out-of-plane tangent stiffness (lateral, roll, yaw)
    contact_model.py      ContactModel: build, schema-13 storage, full-field / plate recovery
    corotation.py         Frame math: phi_ref, rotations, fixed-frame transform
    basis.py              Softplus top shape mode
    toe_spring.py         Spring law, rearfoot geometry and toe mode, toe moment
  plotting/
    compliance.py         Compliance diagnostic figures
    energy_plotly.py      Strain-energy colouring of foam and plate
    shape_render.py       Fixed-frame deformed-geometry plot data
  cli/
    contact_lookup.py     Contact-model build CLI
    visco_force_map.py    Viscoelastic force-map CLI
  gui/
    app.py                Streamlit GUI (live measured-sole rebuild, continuous contact)
    params.py             GUI defaults, angle-slider config, dual slider/number widgets, SI conversion
    rebuild_worker.py     Subprocess worker that rebuilds the contact model
    gait_page.py          Gait-replay page
    gait_plots.py         Gait-replay plotly figures
    pages/1_Gait_replay.py Streamlit multipage entry for gait replay
  viscoelasticity/        Viscoelastic F_VE(t) → elastic F_e(t) mapper
  gait/                   Wang gait force-replay backend for the GUI page
scripts/
  validate_continuous_contact.py  Validation study against direct FEM, continuity checks
tests/
  direct_fem.py           Direct sparse-FEM solves the shared operators are validated against
docs/continuous_contact.md Continuous unilateral contact: formulation, solver, storage, validation
docs/out_of_plane_stiffness.md 2.5D lateral / roll / yaw tangent stiffness: assumptions, formulas, outputs
docs/viscoelasticity.md   Viscoelastic force-map mathematics and usage
docs/gait_replay.md       Gait replay conventions, citations, limitations
docs/passive_toe_spring.md Passive toe spring model, signs, solver, limitations
docs/measured_sole.md     Measured carbon-plated sole: CSV, conventions, validation, limits
docs/external_api.md      SoleModel.step(): loads / pitch in, displacements / moments out
configs/setup.json             Example measured-sole configuration (all non-CSV parameters)
```
