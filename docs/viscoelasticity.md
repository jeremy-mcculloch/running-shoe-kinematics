# Viscoelastic force → elastic force mapping

Standalone module: `compliance_fem.viscoelasticity`.

Maps a known viscoelastic generalized-force history \(F_{VE}(t)\) to the
equivalent elastic force \(F_e(t)\) that would produce the same response through
a normalized relaxation function \(G\):

\[
F_{VE}(t)=\int_0^t G(t-\tau)\,\dot F_e(\tau)\,d\tau,
\qquad
F_e(t)=\int_0^t J(t-\tau)\,\dot F_{VE}(\tau)\,d\tau.
\]

Initial conditions: \(F_{VE}(0)=F_e(0)=0\).

The constitutive map is **componentwise in time**. Any \(x/y\) coupling belongs
in the downstream elastic FEM operator, not here.

---

## Online API

```python
from compliance_fem.viscoelasticity import create_material, SLSConfig

material = create_material(SLSConfig(g_inf=0.5, tau_r=1.0))
material.reset()

for t_i, FVE_i in samples:
    FE_i = material.update(FVE_i, t_i)   # shape (n_components,)
```

Batch convenience (calls `update()` in a loop — no second algorithm):

```python
FE = material.evaluate_history(times, FVE_history)
```

---

## Models

### 1. SLS (exact recursive creep update)

\[
G(t)=g_\infty+(1-g_\infty)e^{-t/\tau_r},
\qquad
J(t)=1+\Bigl(\tfrac1{g_\infty}-1\Bigr)\bigl(1-e^{-t/\tau_c}\bigr),
\quad \tau_c=\tau_r/g_\infty.
\]

Internal state \(s=\int e^{-(t-\tau)/\tau_c}\,\dot F_{VE}\,d\tau\), updated with an
exact exponential integrator under piecewise-linear \(F_{VE}\).

**Parameters:** `g_inf`, `tau_r` (time units). Dimensionless `g_inf ∈ (0,1]`.

### 2. Fractional power-law creep (sum-of-exponentials approximation)

\[
J(t)=\frac1{E_0}\Bigl(\frac tT\Bigr)^\alpha,\qquad 0<\alpha<1,
\]

\[
F_e(t)=\frac{\alpha}{E_0 T^\alpha}\int_0^t (t-\tau)^{\alpha-1} F_{VE}(\tau)\,d\tau.
\]

**Exact constitutive model** vs **numerical backend:** a pure spring-pot has no
intrinsic \(\tau\) bounds. The online backend approximates

\[
t^{\alpha-1}\approx\sum_{k=1}^M a_k e^{-t/\tau_k}
\]

on a finite window `[tau_min, tau_max]` (default `[1e-6 T, 1e6 T]`) using
log-spaced poles with nonnegative least-squares kernel collocation (integral
quadrature fallback), then maintains \(M\) force-driven exponential states.

**Parameters:** `E0` (force/area), `T` (time), `alpha`, plus `num_modes`,
`tau_min`, `tau_max`. No silent unit conversion.

### 3. Fung continuous spectrum (log Prony + Prony inverse)

Spectrum \(S(\tau)=C/\tau\) on \([\tau_1,\tau_2]\):

\[
G(t)=\frac{1+C\bigl[E_1(t/\tau_2)-E_1(t/\tau_1)\bigr]}{1+C\ln(\tau_2/\tau_1)}.
\]

**Backend:** logarithmic quadrature → Prony series
\(G(t)\approx g_\infty+\sum h_k e^{-t/\tau_k}\), then the shared Prony
**inverse** recurrence for \(F_e\) given \(F_{VE}\) (same exponential-memory
engine as fractional, different weights / update form).

**Parameters:** `C`, `tau1`, `tau2`, `num_modes`.

---

## CLI

Entry point: `visco-force-map`.

Input CSV:

```text
time,Fx,Fy
0.0,0.0,0.0
0.01,1.2,0.4
...
```

Output CSV:

```text
time,Fx_ve,Fy_ve,Fx_e,Fy_e
...
```

Examples:

```bash
visco-force-map --model sls --g-inf 0.5 --tau-r 0.1 \
  -i fve.csv -o fe.csv

visco-force-map --model fractional --E0 1.0 --T 1.0 --alpha 0.5 \
  --num-modes 48 --tau-min 1e-4 --tau-max 1e4 \
  -i fve.csv -o fe.csv

visco-force-map --model fung --C 0.1 --tau1 1e-3 --tau2 1e2 --num-modes 40 \
  -i fve.csv -o fe.csv
```

The CLI initializes the material once and loops `update()` over every sample.

---

## Accuracy tradeoffs

| Model | Exact constitutive law | Online numerical method |
|---|---|---|
| SLS | Analytic \(G\), \(J\) | Exact exponential state update (piecewise-linear \(F_{VE}\)) |
| Fractional | Power-law \(J\) | Finite SoE window; refine `num_modes` / widen `[τ_min,τ_max]` |
| Fung | Continuous \(E_1\) spectrum | Log-spaced Prony of \(G\); refine `num_modes` |

Forward→inverse round-trips (synthesize \(F_e\), form \(F_{VE}=G*\dot F_e\), recover
\(F_e\)) are the preferred consistency checks for SLS and Fung.
