# Passive toe spring (default toe model)

The toe coordinate is **solved, not prescribed**. Each gait frame's toe angle
\(\theta\) is the exact quasistatic equilibrium between the shoe's generalized
force on the distributed top-shape coordinate and a passive rotational spring.
No FEM solve is performed per gait frame: the stored contact lookup is linear in
the generalized coordinates, so every frame reduces to a scalar root find per
contact record.

Implementation: `src/compliance_fem/contact/toe_spring.py` (spring, affine model, root
solver), `src/compliance_fem/gait/passive_toe.py` (per-frame candidates), and
`src/compliance_fem/gait/replay.py` (runtime integration).

## Why \(\theta\) is an output

Earlier versions of the replay either prescribed \(\theta\) with an empirical formula
(\(\theta=\mathrm{ReLU}(-\phi)(1-(1-F_y^c/F_y^{\max})^4)\)) or
fitted \(\alpha\) to the measured sagittal moment/COP; both have been removed. Neither
is mechanics: the formula is arbitrary, and the COP fit makes the toe
coordinate absorb all moment-modelling error. With a passive joint the angle
is whatever balances the shoe's reaction against the joint stiffness, so it is
a model output that can be checked against measurements.

## Parameters

| Parameter | Default | Units |
|---|---|---|
| `toe_stiffness_Nm_per_rad` \(k\) | 25 | N·m/rad (total, whole foot width) |
| `toe_neutral_angle_rad` \(\theta_0\) | 0 | rad |
| `toe_damping_Nms_per_rad` \(c\) | 0 | N·m·s/rad (recorded only) |
| `toe_angle_min_deg`, `toe_angle_max_deg` | −75, 75 | deg |
| `toe_equilibrium_abs_tol_Nm`, `toe_equilibrium_rel_tol` | 1e-6, 1e-8 | N·m, – |
| `toe_root_scan_points` | 721 | – |
| `toe_low_force_threshold_N` | 50 | N (experimental total) |

**\(k=25\) N·m/rad** is the specified default passive toe-joint
stiffness. It is a lumped whole-foot value in N·m/rad (not per metre width):
\(Q_\alpha\) is converted to a total by multiplying the per-width lookup value
by the shoe width before it is balanced against the spring. It is a user
parameter (`--toe-stiffness`, GUI input); treat it as a modelling assumption to
be identified, not a measured property of these subjects.

**Neutral angle \(\theta_0=0\).** The unloaded shoe top (the softplus shape at
\(\alpha=0\), i.e. the stored reference geometry) is taken as the spring's
stress-free configuration. Zero load therefore gives exactly \(\theta=0\).

**No damping.** The equation is quasistatic: \(c\) is stored in `toe_config`, a
warning is issued if it is nonzero, and it never enters the solve.

## Coordinate: \(\alpha=\tan\theta\)

The lookup coordinates are
\(\gamma=[\alpha,d_x,d_y,r_x,r_y]^\mathsf T\) with the top-surface vertical
displacement \(v_{\mathrm{top}}=\alpha\,\varphi_1(x)\) from mode 0 only (modes
1–4 clamp the top). Here

\[
\varphi_1(x)=s(x)-\bigl(1-\tfrac{x}{L}\bigr)s(0)-\tfrac{x}{L}s(L),
\qquad s=\text{softplus (metres)},
\]

where \(s\) turns on at the softplus joint \(x_{\mathrm{mtp}}=L-\ell_{\mathrm{toe}}\)
and \(\ell_{\mathrm{toe}}\) is the toe length (joint to toe tip, metres). So
\(\varphi_1(0)=\varphi_1(L)=0\) and \(\varphi_1\le 0\) in the interior
(\(\varphi_1(x_{\mathrm{mtp}})\approx-(L-\ell_{\mathrm{toe}})\,\ell_{\mathrm{toe}}/L\)). The toe angle is
\(\theta=\arctan\alpha\) **exactly** — no small-angle substitution anywhere
(\(\arctan\alpha\neq\alpha\) in the spring energy and its derivatives).

## Spring energy and generalized forces

\[
U(\alpha)=\tfrac12 k\,(\arctan\alpha-\theta_0)^2,
\qquad
\frac{\partial U}{\partial\alpha}=\frac{k(\arctan\alpha-\theta_0)}{1+\alpha^2},
\qquad
\frac{\partial^2U}{\partial\alpha^2}=\frac{k\,[1-2\alpha(\arctan\alpha-\theta_0)]}{(1+\alpha^2)^2}.
\]

Spring moment in \(\theta\): \(M_{\mathrm{spring}}=-k(\theta-\theta_0)\).

## Generalized force from virtual work

The lookup solves \(\mathbf u=\mathbf C\mathbf F+\mathbf R\mathbf a\) with
\(\mathbf R^\mathsf T\mathbf F=0\); the stored top nodal forces are the
**load term of \(\mathbf K\mathbf u=\mathbf f\)** at the top DOFs, i.e. the
force the foot applies **to** the shoe (foot-on-shoe). This was checked
against direct FEM (`solve_direct_fem_contact(..., gamma=...)`, test 26):
\(\mathbf f_t=\mathbf K\mathbf u\) at the top DOFs matches the lookup
\(\mathbf f_{t,y}\).

### What is held fixed: the rearfoot, not the chord

\(\varphi_1\) vanishes at the heel **and the toe tip**, so the lookup frame is
the heel-to-toe chord frame, and a variation of \(\alpha\) at fixed chord
rotates the rearfoot nose-down while pinning the toe tip. That is not the
foot's toe degree of freedom. The measured pitch \(\phi\) comes from the heel
and metatarsal-head markers, i.e. it is the **rearfoot (heel → MTP) line**,
which is what the kinematics prescribe. Bending the toe at a fixed rearfoot
therefore:

1. rotates the chord frame, exactly (within the lookup's displacement
   kinematics):
   \[
   \varphi(\alpha)=\phi-\operatorname{atan2}\bigl(y_{\mathrm{top}}(x_{\mathrm{mtp}})-y_{\mathrm{top}}(0)+\alpha\,\varphi_1(x_{\mathrm{mtp}}),\;x_{\mathrm{mtp}}\bigr),
   \]
   so the model's deformed heel → MTP line always lies along the measured
   \(\phi\) (test `test_solved_configuration_keeps_measured_rearfoot_angle`);
2. moves the sole relative to the rearfoot line by
   \[
   \psi(x)=\varphi_1(x)-\frac{x}{x_{\mathrm{mtp}}}\varphi_1(x_{\mathrm{mtp}}),\qquad \psi(0)=\psi(x_{\mathrm{mtp}})=0,
   \]
   which is \(\approx\max(0,x-x_{\mathrm{mtp}})\): a rotation of the toe segment about the
   MTP point. \(\psi-\varphi_1\) is linear in \(x\) (a rigid rotation about the
   heel).

### Generalized force

The foot's virtual work on the shoe for a toe variation at fixed rearfoot is
\(\psi^\mathsf T\mathbf f_{t,y}\,\delta\alpha\). By action–reaction the shoe
pushes on the foot with \(-\mathbf f_{t,y}\):

\[
Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}
=-\,\psi^\mathsf T\mathbf f_{t,y}
=-\,\varphi_1^\mathsf T\mathbf f_{t,y}+\frac{\varphi_1(x_{\mathrm{mtp}})}{x_{\mathrm{mtp}}}\,M_v
\quad[\text{N·m per metre width}],
\qquad
Q_\alpha^{\mathrm{total}}=w\,Q_\alpha^{\mathrm{lookup}}\quad[\text{N·m}],
\]

with \(M_v=\mathbf x_t^\mathsf T\mathbf f_{t,y}\) and \(w\) = `shoe_width_m`
(default 0.10 m). The lookup stores the per-mode basis
`Q_alpha_shoe_on_foot_basis` (shape `(n_records, 6)`, column 0 the
curved-sole closure) so \(Q_\alpha=w\,\mathbf q_{\mathrm{basis}}\cdot[1,\gamma]\).

**Relation to the toe moment.** Because \(\psi\approx\max(0,x-x_{\mathrm{mtp}})\), the
stored ramp moment `M_toe_vertical` \(=\rho^\mathsf T\mathbf f_{t,y}\) with
\(\rho=\max(0,x-x_{\mathrm{mtp}})\) gives

\[
Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}\approx-M_{\mathrm{toe}},
\qquad |Q_\alpha+M_{\mathrm{toe,v}}|\le\max|\psi-\rho|\;\textstyle\sum|f_{t,y}|,
\]

exact in the sharp-kink limit (test `test_negative_toe_moment_gives_positive_theta`).
A **negative foot-on-shoe toe moment about the MTP point (load under the toes)
gives a positive (dorsiflexing) \(\theta\)**. The chord-fixed quantity
\(-\varphi_1^\mathsf T\mathbf f_{t,y}\) (used before this correction) equals
\(-\rho^\mathsf T\mathbf f+\tfrac{\ell_{\mathrm{toe}}}{L}M_v\approx F_y\,(L-\ell_{\mathrm{toe}})(1-x_c/L)\) and is
negative for any compressive load wherever it acts, which produced the wrong
(negative) toe angle.

**Stiffness sign (test 5).** At \(F^\star=0\) with the rearfoot at its
reference angle, the shoe's strain energy along the solved configuration path
is \(\tfrac12\mathbf u^\mathsf T\mathbf K\mathbf u=-\int_0^\alpha Q\approx\tfrac12(-Q'(0))\alpha^2>0\),
checked against direct FEM, so \(Q'(0)<0\): the shoe resists toe bending. A
plain sign flip of \(Q_\alpha\) would make this stiffness negative, which is
why the fix is the rearfoot-fixed virtual displacement, not a flipped sign.

**\(Q_\alpha\leftrightarrow Q_\theta\).** Since \(d\alpha/d\theta=1+\alpha^2\),

\[
Q_\theta=Q_\alpha\,(1+\alpha^2).
\]

The equilibrium \(Q_\alpha=\partial U/\partial\alpha\) is equivalent to
\(Q_\theta=k(\theta-\theta_0)\), i.e. \(Q_\theta+M_{\mathrm{spring}}=0\).

**Sign convention for \(\theta\).** \(\theta>0\) (\(\alpha>0\)) tilts the toe
segment **up** relative to the rearfoot line (dorsiflexion), since
\(\psi(L)\approx\ell_{\mathrm{toe}}>0\). Positive fixed-frame angles are counter-clockwise
with +x heel → toe and +y up. Moments in the lookup are foot-on-shoe.

## Reduction per interval record

Every contact interval \(\mathcal I_{ij}\) of the lookup
([interval_contact.md](interval_contact.md)) is reduced and solved on its own;
roots are never shared or averaged between intervals. For a record, \(\mathbf z(\alpha)=[F^T_x,F^T_y,r_x,r_y,\alpha]\) with the
local force \(\mathbf F^T=Q(\varphi)^\mathsf T\mathbf F^\star\) and
\((r_x,r_y)=(\cos\varphi-1,-\sin\varphi)\) evaluated at \(\varphi(\alpha)\).
The translations follow from the force balance
\(\mathbf K_d\mathbf d=\mathbf F^T-\mathbf F_{\mathrm{closure}}-\mathbf F_r-\mathbf F_\alpha\alpha\)
(the curved-sole closure enters with coefficient 1); \(\mathbf K_d\)
is LU-factored once (no explicit inverse), giving \(\mathbf d=\mathbf D\mathbf z\) and

\[
Q_\alpha(\alpha)=\mathbf c\cdot\mathbf z(\alpha),
\qquad
Q_\alpha'(\alpha)=\mathbf c_{1:4}\cdot\frac{\partial\mathbf z}{\partial\varphi}\,\varphi'(\alpha)+c_5,
\]

both analytic (`toe_spring.RearfootToeModel`). The reported `toe_q0_Nm`,
`toe_q1_Nm` are \(Q(0)\) and \(Q'(0)\). The model is checked against direct
superposition at the rotated chord angle and the analytic derivative against
finite differences. The fixed-frame affine form \(q_0+q_1\alpha\)
(`toe_affine_model`, `solve_toe_equilibrium`) is kept for synthetic checks.
Ill-conditioned \(\mathbf K_d\) (condition > `cond_limit`) marks the record failed.

## Exact equilibrium equation

\[
\boxed{\,g(\alpha)=Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}(\alpha)-\frac{k(\arctan\alpha-\theta_0)}{1+\alpha^2}=0\,}
\qquad(k=25,\ \theta_0=0).
\]

Equivalently the total potential
\(\Pi(\alpha)=U(\alpha)-\int_0^\alpha Q_\alpha\) is stationary (\(\Pi'=-g\));
the integral is evaluated by 32-point Gauss–Legendre quadrature.

## Root solver and bounds

`solve_toe_equilibrium_model` (vectorized over the scan; `solve_toe_equilibrium`
is its affine special case):

1. Uniform scan of \(\theta\) over \([\theta_{\min},\theta_{\max}]\) (default
   \(\pm75^\circ\), 721 points); evaluate \(g(\tan\theta)\); reject non-finite
   coefficients (`nonfinite_coefficients`).
2. Exact zeros on the grid are roots; every sign change is refined with
   **Brent** (`scipy.optimize.brentq`) in \(\theta\).
3. Endpoint roots are accepted when \(|g|\) is within tolerance.
4. Tangential (double) roots: local minima of \(|g|\) are refined with bounded
   minimization and accepted only if within tolerance.
5. De-duplicate (1e-10 rad), keep converged roots only, sort. Tolerance:
   \(|g|\le\) `abs_tol` + `rel_tol`·max(\(|Q(0)|,|Q(\alpha)-Q(0)|,|\partial U/\partial\alpha|\)).

If no root exists in bounds, the status is `no_root_in_bounds`; a diagnostic
\(\min|g|\) point is saved but is **never** labelled an equilibrium
(`converged=False`, status suffix `+diagnostic_min_residual`, candidate
inadmissible). Because \(\partial U/\partial\alpha\) is bounded
(\(\le k\max_\theta\theta\cos^2\theta\approx0.41k\approx10.3\) N·m at
\(\theta\approx37^\circ\)) while the shoe stiffens against bending
(\(Q'<0\)), a root normally exists in \((-90^\circ,90^\circ)\); ±75° covers
measured running loads. The saturation of
\(\partial U/\partial\alpha\) is the key difference from a linear
\(k\alpha\) spring.

## Stability and multiple roots

\(\Pi''(\alpha)=\partial^2U/\partial\alpha^2-Q_\alpha'(\alpha)\) (analytic). A
root is **stable** when \(\Pi''>0\); saved as `toe_equilibrium_tangent`
(\(\Pi''\)) and `toe_equilibrium_stable`. With \(Q'<0\), roots are stable
unless \(|\theta|\) is large enough that \(\partial^2U/\partial\alpha^2<Q'\).

All roots are stored (`toe_roots_deg`, `toe_root_count`). Each root becomes its
own candidate. Preference order (`order_toe_roots`): continuity with the
previous frame's \(\theta\), then lower potential \(\Pi\), then contact
violation, then force residual. The temporal (Viterbi) selector adds costs for
toe residual (`toe_residual` = 10·min(rel. residual, 1)), unstable roots
(`toe_unstable` = 20), root rank (`toe_root_rank` = 0.05), and inadmissibility.

## Low load

When \(\|F\|<\) `toe_low_force_threshold_N` (experimental total), only the
**relaxed root** (the root closest to \(\theta_0\)) is kept per record, status
`+low_load`. Zero load gives \(q_0=0\) and \(\theta=0\) exactly. There is no
formula fallback.

## Contact interval

For every root the all-node fixed-frame unilateral check (every free node's
ground gap, every contact node's normal reaction, edges and interior alike) is
re-evaluated **after** \(\alpha\) is solved for that interval. A candidate is
admissible only if the root converged, the unilateral check passes, and the
force residual is within tolerance. The search screens intervals near the
previous frame's interval first (local, then expanded radii), then all
intervals; the admissible candidates of the first stage that has any are
re-solved exactly and passed to the Viterbi sequence selector, which drops
inadmissible candidates whenever an admissible one exists. When no interval is
admissible at a frame (it happens for part of real stances; see below), the
least-violating candidates are kept, the selected one is labelled
`least_violating_fallback`, and its status carries `+unilateral_violation`
(plus `+interior_tension` when interior contact nodes are tensile, i.e. a
multi-interval contact state is indicated); the counts are in
`model_info.toe_summary`.

## Viscoelasticity: elastic only

The spring balances an **elastic** shoe response. `toe_model=passive_spring`
with any `visco_model` other than `elastic` raises an error. The explicit
opt-in `passive_spring_elastic_equivalent` solves the same equation against
the viscoelastic mapper's elastic-equivalent wrench and records a warning; it
is an approximation because the toe coordinate itself would have a
rate-dependent reaction. There is never an automatic fallback.

A proper viscoelastic extension needs: (1) the relaxation history applied to
\(Q_\alpha\) (not only to the resultant wrench), i.e. a stored
\(Q_\alpha\) response per internal state; (2) an evolution equation for
\(\alpha\) including the toe damping \(c\,\dot\theta\) and the viscous shoe
term, making the solve an ODE/DAE per frame; (3) consistent state updates when
the contact record changes.

## Lookup schema

The lookup (`LOOKUP_SCHEMA_VERSION = 10`) stores `Q_alpha_shoe_on_foot_basis`
(`(n_records, 6)`, N·m per metre width per unit coefficient, column 0 the
curved-sole closure) defined as the rearfoot-fixed
\(-\psi^\mathsf T\mathbf f_{t,y}\) for every contact interval, plus metadata
(`top_force_sign`, `toe_generalized_force`, units, source). The stored basis
must match recomputation from the stored top forces (rel. 1e-9). Files with
a different schema version are rejected and must be regenerated.

## Outputs

Per frame: `theta_rad`, `theta_deg`, `alpha`, `Q_alpha_shoe_on_foot`,
`Q_theta_shoe_on_foot`, `toe_spring_moment_Nm`,
`toe_spring_generalized_force_alpha`, `toe_equilibrium_residual_Nm`,
`toe_equilibrium_relative_residual`, `toe_equilibrium_tangent`,
`toe_equilibrium_stable`, `toe_root_count`, `toe_root_index`,
`toe_spring_energy_J`, `toe_solve_status`, `toe_low_load`, `toe_q0_Nm`,
`toe_q1_Nm`, `toe_roots_deg` (NPZ), plus constant columns `toe_model`,
`toe_stiffness_Nm_per_rad`, `toe_neutral_angle_rad`,
`toe_damping_Nms_per_rad`. The `varphi` column is the chord-frame rotation at
the selected root (it differs from the measured rearfoot `phi` by the bent-toe
chord offset). The NPZ also stores `toe_Q_coeff` (per-frame \(\mathbf c\)) so
the GUI can redraw the exact \(Q_\alpha(\theta)\) curve. `model_info.json` includes
`toe_config`, `toe_equation`, `toe_rearfoot_kinematics` (`L`, `toe_length`, `phi1_mtp` \(=\varphi_1(L-\ell_{\mathrm{toe}})\), `dy_mtp`)
and `toe_summary`.

## Example results (width 0.1 m, elastic)

| Trial | Frames | \(\theta\) range | peak at | max \(|g|\) | low-load | max \(U\) | contacts | unilateral-violation frames |
|---|---|---|---|---|---|---|---|---|
| P4 pr1 01, left, stance 0 | 281 | −0.9° … 31.4° | 76 % stance | 1.6e-14 N·m | 60 | 3.75 J | heel 15, full 12, toe 254 | 19 (3 with interior tension) |
| P1 pr1 01, right, stance 0 | 300 | 0° … 23.2° | 65 % stance | 1.8e-14 N·m | 65 | 2.05 J | full 72, toe 228 | 114 (46 with interior tension) |

Every frame had exactly one root and all were stable. \(\theta>0\) wherever
the foot-on-shoe toe moment is negative (100 % of loaded P1 frames, 99.5 % of
P4). The chord frame differs from the measured rearfoot angle by up to 8°.
In every
unilateral-violation frame no interval of the lookup admits a violation-free
solution at the measured wrench; the interior-tension frames indicate that a
multi-interval contact model would be needed there.

## Limitations

- **Distributed coordinate vs anatomical MTP.** \(\alpha\) scales a smooth
  softplus shape across the whole top surface; it is not a rigid rotation about
  an MTP axis, although with the rearfoot held fixed \(\psi\approx\max(0,x-x_{\mathrm{mtp}})\)
  makes it behave like one for a sharp softplus. The toe dorsiflexes with load
  under the toes and peaks late in stance (65–76 %), which is the same sense
  and timing as anatomical MTP dorsiflexion; magnitudes still depend on the
  soft \(k\) relative to the shoe's generalized stiffness and on the width
  scaling.
- For a soft softplus (small \(\kappa\)) \(\psi\) dips below the rearfoot line
  on \((0,L-\ell_{\mathrm{toe}})\), so loads under the rearfoot contribute slightly to
  \(Q_\alpha\) and \(Q_\alpha\approx-M_{\mathrm{toe}}\) is only approximate.
- Quasistatic and elastic only; damping is ignored.
- The bounded spring force means \(\theta\) grows quickly once
  \(|Q_\alpha|\) approaches ~10 N·m (P4 reaches 9.98 N·m).
- P4 pr1 01 must be run with `--foot left` (the right-foot default gives an
  implausible \(\phi\) range for that trial).
