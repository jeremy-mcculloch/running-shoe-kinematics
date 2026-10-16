# Passive toe spring (default toe model)

The toe coordinate is **solved, not prescribed**. Each gait frame's toe angle
\(\theta\) is the exact quasistatic equilibrium between the shoe's generalized
force on the distributed top-shape coordinate and a passive rotational spring.
No FEM solve is performed per gait frame: the shoe response comes from the
shared contact operator ([continuous_contact.md](continuous_contact.md)), and
every evaluation of the toe residual runs one converged contact solve at the
trial angle.

Implementation: `src/compliance_fem/contact/toe_spring.py` (spring law,
`RearfootGeometry`, `rearfoot_toe_mode`, `compute_toe_moment`),
`src/compliance_fem/contact/continuous_toe.py` (nested root solve
`solve_toe_contact`), and `src/compliance_fem/gait/replay.py`,
`src/compliance_fem/api.py`, `src/compliance_fem/gui/app.py` (callers).

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
| `toe_root_scan_points` | 721 | – (cold-start scan) |
| `toe_low_force_threshold_N` | 50 | N (experimental total) |

**\(k=25\) N·m/rad** is the specified default passive toe-joint
stiffness. It is a lumped whole-foot value in N·m/rad (not per metre width):
\(Q_\alpha\) is converted to a total by multiplying the per-width model value
by the shoe width before it is balanced against the spring. It is a user
parameter (`toe_spring.toe_stiffness_Nm_per_rad` in the JSON config, GUI
input); treat it as a modelling assumption to be identified, not a measured
property of these subjects.

**Neutral angle \(\theta_0=0\).** The unloaded shoe top (the softplus shape at
\(\alpha=0\), i.e. the stored reference geometry) is taken as the spring's
stress-free configuration. Zero load therefore gives exactly \(\theta=0\).

**No damping.** The equation is quasistatic: \(c\) is stored in `toe_config`, a
warning is issued if it is nonzero, and it never enters the solve.

## Coordinate: \(\alpha=\tan\theta\)

The top surface is glued to the foot: in the top-attached frame its
displacement is prescribed as \(u_{\mathrm{top}}=0\),
\(v_{\mathrm{top}}=\alpha\,\varphi_1(x)\) (`w_alpha` of the contact operator),
so \(\alpha\) is the only top degree of freedom. Here

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

The contact operator expresses the top-clamped response as
\(\mathbf F_t=\alpha\,\mathbf f_\alpha+\mathbf P_{tb}\,\mathbf f_b\)
([continuous_contact.md](continuous_contact.md)): \(\mathbf F_t\) is the
external nodal load at the top DOFs, i.e. the **load term of
\(\mathbf K\mathbf u=\mathbf f\)**, the force the foot applies **to** the shoe
(foot-on-shoe). `tests/test_continuous_contact.py` checks these top forces and
the bottom fields of solved states against an independent top-clamped direct
FEM solve (`tests/direct_fem.py`).

### What is held fixed: the rearfoot, not the chord

\(\varphi_1\) vanishes at the heel **and the toe tip**, so the top-attached
frame is the heel-to-toe chord frame, and a variation of \(\alpha\) at fixed
chord rotates the rearfoot nose-down while pinning the toe tip. That is not the
foot's toe degree of freedom. The measured pitch \(\phi\) comes from the heel
and metatarsal-head markers, i.e. it is the **rearfoot (heel → MTP) line**,
which is what the kinematics prescribe. Bending the toe at a fixed rearfoot
therefore:

1. rotates the chord frame, exactly (within the model's displacement
   kinematics; `RearfootGeometry.chord_rotation`):
   \[
   \varphi(\alpha)=\phi-\operatorname{atan2}\bigl(\Delta y_{\mathrm{mtp}}+\alpha\,\varphi_1(x_{\mathrm{mtp}}),\;L-\ell_{\mathrm{toe}}\bigr),
   \qquad \Delta y_{\mathrm{mtp}}=y_{\mathrm{top}}(x_{\mathrm{mtp}})-y_{\mathrm{top}}(0),
   \]
   so the model's deformed heel → MTP line always lies along the measured
   \(\phi\) (test `test_solved_configuration_keeps_measured_rearfoot_angle`);
2. moves the sole relative to the rearfoot line by
   \[
   \psi(x)=\varphi_1(x)-\frac{x}{x_{\mathrm{mtp}}}\varphi_1(x_{\mathrm{mtp}}),\qquad \psi(0)=\psi(x_{\mathrm{mtp}})=0,
   \]
   which is \(\approx\max(0,x-x_{\mathrm{mtp}})\): a rotation of the toe segment about the
   MTP point. \(\psi-\varphi_1\) is linear in \(x\) (a rigid rotation about the
   heel). \(\psi\) at the top nodes is stored as `psi_top` in the contact
   operator (`toe_spring.rearfoot_toe_mode`).

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
Q_\alpha^{\mathrm{total}}=w\,Q_\alpha^{\mathrm{model}}\quad[\text{N·m}],
\]

with \(M_v=\mathbf x_t^\mathsf T\mathbf f_{t,y}\) and \(w\) = `shoe_width_m`
(default 0.10 m). \(\mathbf f_{t,y}\) is the converged top force of the contact
solve at \((\varphi(\alpha),\alpha)\), so \(Q_\alpha\) includes the contact
response at that trial angle; the per-metre value is the diagnostic
`Q_alpha_shoe_on_foot_per_m`.

**Relation to the toe moment.** Because \(\psi\approx\max(0,x-x_{\mathrm{mtp}})\), the
vertical ramp moment \(M_{\mathrm{toe,v}}=\rho^\mathsf T\mathbf f_{t,y}\) with
\(\rho=\max(0,x-x_{\mathrm{mtp}})\) (`compute_toe_moment`) gives

\[
Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}\approx-M_{\mathrm{toe,v}},
\qquad |Q_\alpha+M_{\mathrm{toe,v}}|\le\max|\psi-\rho|\;\textstyle\sum|f_{t,y}|,
\]

exact in the sharp-kink limit. A **negative foot-on-shoe toe moment about the
MTP point (load under the toes) gives a positive (dorsiflexing) \(\theta\)**.
The chord-fixed quantity \(-\varphi_1^\mathsf T\mathbf f_{t,y}\) equals
\(-\rho^\mathsf T\mathbf f+\tfrac{\ell_{\mathrm{toe}}}{L}M_v\approx F_y\,(L-\ell_{\mathrm{toe}})(1-x_c/L)\) and is
negative for any compressive load wherever it acts, which would produce the
wrong (negative) toe angle; it is not used.

**Stiffness sign.** At \(F^\star=0\) with the rearfoot at its reference angle,
the shoe's strain energy along the solved configuration path is
\(\tfrac12\mathbf u^\mathsf T\mathbf K\mathbf u=-\int_0^\alpha Q\approx\tfrac12(-Q'(0))\alpha^2>0\),
so \(Q'(0)<0\): the shoe resists toe bending. A plain sign flip of
\(Q_\alpha\) would make this stiffness negative, which is why the rearfoot-fixed
virtual displacement is used rather than a flipped sign.

**\(Q_\alpha\leftrightarrow Q_\theta\).** Since \(d\alpha/d\theta=1+\alpha^2\),

\[
Q_\theta=Q_\alpha\,(1+\alpha^2).
\]

The equilibrium \(Q_\alpha=\partial U/\partial\alpha\) is equivalent to
\(Q_\theta=k(\theta-\theta_0)\), i.e. \(Q_\theta+M_{\mathrm{spring}}=0\).

**Sign convention for \(\theta\).** \(\theta>0\) (\(\alpha>0\)) tilts the toe
segment **up** relative to the rearfoot line (dorsiflexion), since
\(\psi(L)\approx\ell_{\mathrm{toe}}>0\). Positive fixed-frame angles are counter-clockwise
with +x heel → toe and +y up. Top forces and moments are foot-on-shoe.

## Exact equilibrium equation

\[
\boxed{\,g(\alpha)=Q_\alpha^{\mathrm{shoe}\to\mathrm{foot}}(\alpha)-\frac{k(\arctan\alpha-\theta_0)}{1+\alpha^2}=0\,}
\qquad(k=25,\ \theta_0=0).
\]

Equivalently the total potential
\(\Pi(\alpha)=U(\alpha)-\int_0^\alpha Q_\alpha\) is stationary (\(\Pi'=-g\)).
\(Q_\alpha(\alpha)\) is not an affine model: each evaluation of \(g\) at a trial
\(\theta\) sets \(\alpha=\tan\theta\) and \(\varphi(\alpha)\), solves the
unilateral contact problem to convergence at that pose (the active set
warm-started from the nearest already-evaluated \(\theta\)), and reads
\(Q_\alpha\) from the converged top forces. Because the contact solution is
continuous in the pose (nodes enter and leave contact through
\(g=\lambda=0\)), \(Q_\alpha\) and hence the root are continuous along a gait.

## Root solver and bounds

`continuous_toe.solve_toe_contact` solves \(g(\theta)=0\) in \(\theta\) on
\([\theta_{\min},\theta_{\max}]\) (default \(\pm75^\circ\)) with **Brent**
(`scipy.optimize.brentq`, `xtol` 1e-13):

1. **Continuation** (a previous \(\theta\) is available: the previous gait
   sample, the previous `SoleModel.step`, or the GUI state). The bracket is
   grown geometrically on both sides of the previous \(\theta\) (steps
   0.25°, 0.5°, …, 64°, 150°, clipped to the bounds) until \(g\) changes sign;
   the first bracket holds the nearest root, which is refined and taken
   (status `continuation`). A warm sample typically needs 6–8 residual
   evaluations (contact solves).
2. **Cold start** (no previous \(\theta\), or no sign change found by
   continuation). Uniform scan with `toe_root_scan_points` (default 721)
   evaluations; every sign change is refined with Brent and exact zeros on the
   grid are kept. With one root the status is `ok`; with several it is
   `multiple_roots` and the root with the **lowest total potential** \(\Pi\)
   is chosen (\(\Pi\) by trapezoidal integration over \(\alpha\)).

The tolerance is
\(|g|\le\) `toe_equilibrium_abs_tol_Nm` + `toe_equilibrium_rel_tol`·max(\(|Q_\alpha|,|\partial U/\partial\alpha|\), abs tol);
a frame is `converged` only when this holds and the contact solve at the root
converged.

If no root exists in bounds, the status is `no_root_in_bounds`; the grid point
of minimum \(|g|\) is returned as a diagnostic but is **never** labelled an
equilibrium (`converged=False`). Because \(\partial U/\partial\alpha\) is
bounded (\(\le k\max_\theta\theta\cos^2\theta\approx0.41k\approx10.3\) N·m at
\(\theta\approx37^\circ\)) while the shoe stiffens against bending
(\(Q'<0\)), a root normally exists in \((-90^\circ,90^\circ)\); ±75° covers
measured running loads. The saturation of
\(\partial U/\partial\alpha\) is the key difference from a linear
\(k\alpha\) spring.

## Stability and multiple roots

\(\Pi''(\alpha)=\partial^2U/\partial\alpha^2-Q_\alpha'(\alpha)\), evaluated by a
central difference of \(g\) in \(\alpha\) at the root. A root is **stable**
when \(\Pi''>0\); saved as `toe_equilibrium_tangent` (\(\Pi''\)) and
`toe_equilibrium_stable`. With \(Q'<0\), roots are stable unless \(|\theta|\)
is large enough that \(\partial^2U/\partial\alpha^2<Q'\).

The roots found are stored (`toe_roots_deg`, `toe_root_count`). Continuation
returns only the root on the connected branch, so `toe_root_count = 1` for
warm samples; all roots in bounds are only enumerated by a cold scan. There is
no candidate list and no sequence selector: the replay follows one continuous
branch, and the previous \(\theta\) and active set warm-start the next sample.

## Low load

When \(\|F\|<\) `toe_low_force_threshold_N` (experimental total), the root
search starts from \(\theta_0\) instead of the previous \(\theta\), i.e. it
follows the branch connected to the neutral angle (a cold scan, if needed,
takes the root closest to \(\theta_0\)); the status carries `+low_load`. With \(\theta_0=0\), zero
load gives \(g(0)=0\) and \(\theta=0\) exactly. There is no formula
fallback.

## Contact state

The contact state returned with the toe angle is the converged contact solve at
\((\varphi(\alpha^\star),\alpha^\star)\): every bottom node satisfies
\(0\le\lambda\perp g\ge0\) (within the scale-aware tolerances), and the
prescribed \(F_x,F_y\) are met exactly by proportional shear. The contact set
may be empty, one patch or several disconnected patches; nothing is chosen
from a table and there is no fallback state. See
[continuous_contact.md](continuous_contact.md) for the formulation,
diagnostics and patch reporting.

## Viscoelasticity: elastic only

The spring balances an **elastic** shoe response. `toe_model=passive_spring`
with any `visco_model` other than `elastic` raises an error. The explicit
opt-in `passive_spring_elastic_equivalent` solves the same equation against
the viscoelastic mapper's elastic-equivalent load and records a warning; it
is an approximation because the toe coordinate itself would have a
rate-dependent reaction. There is never an automatic fallback.

A proper viscoelastic extension needs: (1) the relaxation history applied to
\(Q_\alpha\) (not only to the resultant load), i.e. a \(Q_\alpha\) response
per internal state; (2) an evolution equation for \(\alpha\) including the toe
damping \(c\,\dot\theta\) and the viscous shoe term, making the solve an
ODE/DAE per frame; (3) consistent state updates when the contact set changes.

## Stored data

The contact model file (`contact_lookup.npz`, `schema_version = 13`) stores
no per-contact-set toe data. The toe solve uses the contact operator's
`w_alpha` (top shape mode), `f_alpha`, `P_tb`, `psi_top`, the softplus toe
length and `kappa`, and the rearfoot geometry scalars `rearfoot_phi1_mtp`,
`rearfoot_dy_mtp`. Files with a different schema version are rejected and
must be regenerated.

## Outputs

Per frame: `theta_rad`, `theta_deg`, `toe_alpha`, `Q_alpha_shoe_on_foot`,
`Q_theta_shoe_on_foot`, `toe_spring_moment_Nm`,
`toe_spring_generalized_force_alpha`, `toe_equilibrium_residual_Nm`,
`toe_equilibrium_relative_residual`, `toe_equilibrium_tangent`,
`toe_equilibrium_stable`, `toe_root_count`, `toe_iterations` (residual
evaluations, i.e. contact solves), `toe_spring_energy_J`, `toe_solve_status`,
`toe_low_load`, `toe_roots_deg` (NPZ), plus constant columns `toe_model`,
`toe_stiffness_Nm_per_rad`, `toe_neutral_angle_rad`,
`toe_damping_Nms_per_rad`. The `varphi` column is the chord-frame rotation at
the solved root (it differs from the measured rearfoot `phi` by the bent-toe
chord offset). The GUI redraws the \(Q_\alpha(\theta)\) and
\(\partial U/\partial\alpha\) curves of a frame with one exact contact solve per
plotted \(\theta\). `model_info.json` includes `toe_config`, `toe_equation`,
`toe_generalized_force`, `toe_rearfoot_kinematics` (`L`, `toe_length`,
`phi1_mtp` \(=\varphi_1(L-\ell_{\mathrm{toe}})\), `dy_mtp`) and `toe_summary`
(θ range, maximum residuals, multiple-root / no-root / unstable / low-load
frame counts, mean toe iterations, maximum spring energy).

## Limitations

- **Distributed coordinate vs anatomical MTP.** \(\alpha\) scales a smooth
  softplus shape across the whole top surface; it is not a rigid rotation about
  an MTP axis, although with the rearfoot held fixed \(\psi\approx\max(0,x-x_{\mathrm{mtp}})\)
  makes it behave like one for a sharp softplus. The toe dorsiflexes with load
  under the toes, the same sense as anatomical MTP dorsiflexion; magnitudes
  depend on the soft \(k\) relative to the shoe's generalized stiffness and on
  the width scaling.
- For a soft softplus (small \(\kappa\)) \(\psi\) dips below the rearfoot line
  on \((0,L-\ell_{\mathrm{toe}})\), so loads under the rearfoot contribute slightly to
  \(Q_\alpha\) and \(Q_\alpha\approx-M_{\mathrm{toe}}\) is only approximate.
- Quasistatic and elastic only; damping is ignored.
- The bounded spring force means \(\theta\) grows quickly once
  \(|Q_\alpha|\) approaches ~10 N·m.
- Continuation follows the branch connected to the previous sample; a
  different root that appears elsewhere in the bounds is not reported unless a
  cold scan runs.
- P4 pr1 01 must be run with foot `left` (the right-foot default gives an
  implausible \(\phi\) range for that trial).
