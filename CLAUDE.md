# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3.12 -m venv venv && venv/bin/pip install -e ".[dev]"   # needs Python >= 3.11; package compliance-fem, src layout
venv/bin/pyright                              # type check src + tests ([tool.pyright] in pyproject, same config Pylance uses)
pytest                                        # full suite (testpaths=tests, -q)
pytest tests/test_contact_lookup.py::test_name  # single test
pytest -k "rocker and basis"                  # subset by keyword

# Build mesh -> FEM -> contact lookup from a JSON config (skipped if output_dir fingerprint matches)
compliance-fem-contact-lookup configs/setup.json [--force] [--save-fem] [--skip-plots] [--quiet]

streamlit run src/compliance_fem/gui/app.py   # GUI (multipage; gait replay in gui/pages/)
visco-force-map ...                           # see docs/viscoelasticity.md
```

Tests build tiny synthetic soles in `tests/conftest.py` (`N_b = 13`, flat / rocker / asymmetric variants) instead of the measured geometry, so they run without `outputs/`. Some gait tests skip unless the Wang data and a built lookup exist (`data/wang/trc/...`, `outputs/contact_lookup/contact_lookup.npz`). `tests/direct_fem.py` and `tests/reference.py` are independent reference implementations that the lookup is checked against. They are test-only and must not be imported from `src/`.

## What this is

A 2D plane-strain FEM model of a carbon-plated running-shoe sole: two foam regions (the lower one may have a modulus that varies heel→toe) plus an inextensible Euler–Bernoulli plate. The heavy work runs **offline once per geometry/material**. Runtime only contracts precomputed bases, so it never solves FEM.

Pipeline (each stage feeds the next):

1. **geometry/**: normalized sole CSV (`data/geometry/sole_geometry.csv`) → validated profile → Gmsh two-foam mesh with tagged boundaries and a plate polyline.
2. **fem/**: assemble K **without** displacement BCs; the three rigid modes are removed with a saddle-point gauge (`R^T d = 0`). The output is component-major `[u; v]` nodal-force compliance blocks `C_tt, C_tb, C_bt, C_bb` (each `2n×2n`). Plate inextensibility rows `B_p` are exact constraints whose multipliers equal the axial force.
3. **contact/lookup.py**: for **every** contiguous bottom interval `(i, j)` (`N_b(N_b+1)/2` records), one dense boundary saddle system is LU-factored once and solved for a **six-column affine basis** (`basis.py`: column 0 = curved-sole closure with coefficient fixed at 1, then the top softplus shape mode, then four contact-motion modes). Nothing depends on the rotation angle φ; all φ dependence is applied at runtime. Per-node fields are recomputed on demand via `lookup.record_fields(rows)` unless `store_nodal_fields` is set.
4. **Runtime**: `contact/force_control.py` (fixed-frame Fx, Fy, φ, θ → 2×2 `K_F` solve for the anchor translation → `γ` → `corotation.contract_basis`, then all-node gap/reaction admissibility and interval search). `contact/toe_spring.py` + `gait/passive_toe.py` solve for θ as a passive-spring equilibrium. `api.SoleModel.step()` wraps all of this for external callers.
5. **gui/** and **gait/**: a Streamlit front end (rebuilds the lookup in a subprocess via `rebuild_worker.py`) and Wang et al. gait force replay.

`contact/model_setup.py` owns the JSON config (`configs/setup.json`; relative paths resolve against the config file's directory), fingerprints it, and decides whether an existing lookup can be reused.

## Invariants to preserve

- **Public interface** (README "Public interface"): the CLIs, `compliance_fem.api` (`SoleModel`, `SoleState`), the listed `contact.model_setup` functions, `compliance_fem.viscoelasticity.create_material`, and the top-level `SoleConfig` / `compute_compliance` / `ComplianceResult`. Everything else is internal and may be changed freely.
- Lookup `schema_version` is **12**. Files with any other version are rejected with a message to regenerate, and there is no fallback. Bump it when the stored layout changes.
- Units: solvers use SI. The lookup is per metre of width; `SoleModel` divides loads by `shoe_width_m` and multiplies forces and moments back. The GUI shows mm / kN / Pa / N·mm².
- Sign conventions: φ, φ_ref, and varphi are CCW-positive. Top `F` is foot-on-shoe (`F_top = -GRF`, so stance has `Fy < 0`). Positive gap means open. Positive `R_n` means compressive. The top-shape sign lives only in `corotation.SHAPE_MODE_SIGN`.
- Rotation is exact at the boundary (true sin/cos), but strain is linear. This is a deliberate infinitesimal-strain approximation, not geometric nonlinearity.
- Inadmissible or ill-conditioned records are **flagged and skipped, never regularized**. Rejections carry a structured reason and are counted in `metadata["rejection_counts"]`.
- The deformed-shape display scale `s_d` multiplies only the elastic part, never φ.
- scikit-fem v12 API quirks are documented in the README "scikit-fem API notes" (for example, use `Basis.nodal_dofs` for rigid modes).

## Docs

README.md has the full math (sign tables, basis columns, admissibility). Topic deep-dives are in `docs/`: `interval_contact.md`, `measured_sole.md`, `external_api.md`, `passive_toe_spring.md`, `gait_replay.md`, `viscoelasticity.md`. Update the relevant doc whenever you change a convention or the schema.
