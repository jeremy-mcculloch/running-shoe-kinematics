# Gait force replay (Wang et al. dataset)

Chronological open-loop replay of measured sagittal ground-reaction wrenches
through the existing viscoelastic mapper and single-interval contact lookup
([interval_contact.md](interval_contact.md)). By
default the **toe angle \(\theta\) is solved from a passive toe spring**
(\(k=25\) N·m/rad, \(\theta_0=0\)) in exact quasistatic equilibrium with the
shoe's generalized force; see [passive_toe_spring.md](passive_toe_spring.md).
No FEM refactorization is performed per gait frame.

## Citation

Yuan Wang et al., *Dataset of walking and running biomechanics with different
step widths across different speeds*, Scientific Data 12, 802 (2025).

- Article: https://doi.org/10.1038/s41597-025-05113-6
- TRC: https://doi.org/10.17608/k6.auckland.27015298
- MOT: https://doi.org/10.17608/k6.auckland.27015517

Initialize a local data folder:

```bash
compliance-fem-gait-replay --init-data-dir data/wang
```

Download matched TRC/MOT from Figshare (browser may be required), then:

```bash
compliance-fem-gait-replay --dataset-root data/wang --list-trials
compliance-fem-gait-replay --lookup outputs/contact_lookup \
  --trc data/wang/Pxx_pr1_TT.trc --mot data/wang/Pxx_pr1_TT.mot \
  --foot right --stance-index 0 --visco-model elastic \
  --output-dir outputs/gait_replay
```

Toe-model flags: `--toe-model {passive_spring,passive_spring_elastic_equivalent,prescribed_legacy,fit_cop_legacy}`
(default `passive_spring`), `--toe-stiffness`, `--toe-neutral-angle-rad`,
`--toe-damping` (recorded only), `--toe-angle-min-deg/--toe-angle-max-deg`
(±75°), `--toe-abs-tol`, `--toe-rel-tol`, `--toe-scan-points`,
`--toe-low-force-threshold`. `--theta-mode` is deprecated (maps to the legacy
models).

Check the stance foot: for `P4 pr1 01` the first stance on the plate is the
**left** foot (`--foot left`); `--foot right` yields an implausible
\(\phi\in[-110^\circ,-8^\circ]\).

Preferred-speed 3.0 m/s running trials use codes containing `pr1` (confirm exact
filenames on disk; do not hard-code one subject).

## Coordinate and force signs

Wang/OpenSim TRC/MOT after their transform: **+x anterior, +y superior, +z right**.
Markers/COP are mm (converted to m); forces N; moments often N·mm (→ N·m).

Shoe model fixed frame: **+x heel→toe, +y up**. Force-control top resultants are
the load applied **to the shoe top** (foot → shoe). Mapping:

\[
\mathbf F_{\mathrm{top}} \approx -\mathbf F_{\mathrm{GRF}}^{\text{ground-on-body}}.
\]

An upward measured GRF therefore produces compressive (negative \(F_y\)) top
loading. Moments follow the same action–reaction sign when forming the sagittal
wrench about the model origin.

**Width scaling (gait replay only):** the contact lookup is plane-strain with
**unit out-of-plane thickness**, so stored responses are per metre width
(N/m, N·m/m). Experimental totals are converted by dividing by an effective
shoe width (default \(w=0.10\,\mathrm{m}\)):

\[
(F_x,F_y,M)_ {\mathrm{model}} = (F_x,F_y,M)_{\mathrm{exp}} / w.
\]

COP is unchanged (length). Use `--shoe-width-m` / the GUI control to override.

## Center of pressure (frames and units)

The Wang **MOT** COP (`*_force_px/py/pz`) is a **global laboratory / OpenSim
ground-frame** point (+x anterior, +y superior, +z right). It is **not**
heel-relative, shoe-relative, or lookup-anchor-relative.

Processed OpenSim-ready TRC/MOT files in this workspace declare **metres** and
**N·m**; the paper’s raw CSV uses **mm** / **N·mm** and a different CoP axis
order (`Cx` ML, `Cy` AP, `Cz` vertical). Importers take
`--position-units auto|m|mm` and `--moment-units auto|N-m|N-mm` and record the
resolved units in provenance. Conversion to SI is applied **once**.

Heel-relative COP used for validation:

\[
\mathbf p_{\mathrm{COP}}^{\mathrm{foot}}
=
Q(\phi)^\mathsf T
\bigl(\mathbf p_{\mathrm{COP}}^{\mathrm{lab}}-\mathbf r_{\mathrm{heel}}^{\mathrm{lab}}\bigr).
\]

Measured COP is transformed into the heel-attached foot frame for diagnostics
and (only in ``fit_cop_legacy`` mode) enters \(\theta\) through the heel-relative sagittal
moment \(M_z \approx x_{\mathrm{COP}}^{\mathrm{foot}} F_y\) (top-of-shoe convention).

## How \(\theta\) is obtained

**Default (`passive_spring`):** the marker pitch \(\phi\) is the rearfoot
(heel → MTP) angle, so the lookup's chord frame rotates with the toe,
\(\varphi(\alpha)=\phi-\operatorname{atan2}(\Delta y_a+\alpha\varphi_1(a),a)\).
Per contact interval record the translations are eliminated in closed form and the
rearfoot-fixed generalized force
\(Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}(\alpha)=-\psi^\mathsf T\mathbf f_{t,y}\)
(\(\psi\approx\max(0,x-a)\), so \(Q_\alpha\approx-M_{\mathrm{toe}}\)) is balanced
against the spring: \(\theta=\arctan\alpha\) solves
\(Q_\alpha(\alpha)-k(\arctan\alpha-\theta_0)/(1+\alpha^2)=0\) by bracketed Brent
root finding within the angle bounds. A negative foot-on-shoe toe moment gives
a positive (dorsiflexing) \(\theta\). All roots, residuals, stability, and
spring energy are saved; the `varphi` column is the chord rotation at the
solved root. Measured COP/moment is a diagnostic only. Requires
`--visco-model elastic` (or the explicit `passive_spring_elastic_equivalent`
approximation). Full derivation: [passive_toe_spring.md](passive_toe_spring.md).

**Legacy (`fit_cop_legacy`, `--theta-mode fit-cop`):** Stored lookup responses are linear in
\(\gamma=[\alpha,d_x,d_y,r_x,r_y]^\mathsf T\) with \(\alpha=\tan\theta\) and
\((r_x,r_y)\) known from marker-derived \(\phi\). Prescribing the sagittal wrench
\((F_x,F_y,M_z)\) about the **heel / lookup origin** (top-of-shoe convention,
width-scaled) yields a **3×3** system for \((d_x,d_y,\alpha)\). The force-phi
formula below seeds \(\theta\) when the free \(\alpha\) is out of bounds or
\(K_W\) is ill-conditioned. When \(|F_y|\) is below ``--cop-fit-min-force``
(default 50 N experimental), \(\theta\) is fixed at \(0^\circ\) and only
translations are solved.

**Legacy (`prescribed_legacy`, `--theta-mode force-phi`):** \(\theta=\mathrm{ReLU}(-\phi)\,(1-(1-F_y^c/F_y^{\max})^4)\)
with compressive top load \(F_y^c=\max(0,-F_y)\). Contact translations
\((d_x,d_y)\) come from the usual 2×2 force solve at prescribed \((\phi,\theta)\).
Measured COP does **not** enter this θ formula.

Anatomical MTP angle is **not** identical to the model's distributed softplus
coordinate.

## Viscoelasticity

The stateful backends map the full generalized wrench componentwise:

\[
[F_x,F_y,M]_{VE}(t)\;\longrightarrow\;[F_x,F_y,M]_e(t).
\]

This matches the correspondence principle for a **scalar, spatially uniform**
relaxation kernel. It is only an effective approximation for heterogeneous
layers, changing contact domains, or nonlinear QLV. Use `--visco-model elastic`
for an identity map. The default passive toe spring rejects non-elastic models
(no automatic fallback); see the viscoelastic-extension notes in
[passive_toe_spring.md](passive_toe_spring.md).

## Limitations

- Open-loop: GRF is prescribed; the model cannot predict a changed external GRF.
- Experimental shoe ≠ modelled geometry/materials.
- Mediolateral force is omitted (warning if large).
- The passive toe coordinate is the distributed softplus shape amplitude, not a
  rigid anatomical MTP rotation; see [passive_toe_spring.md](passive_toe_spring.md).
- C3D import is reserved / not yet implemented (use TRC+MOT).
- Energy labels distinguish boundary work from unidentified material dissipation.
