# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3.12 -m venv venv && venv/bin/pip install -e ".[dev]"   # needs Python >= 3.11; package compliance-fem, src layout
venv/bin/pyright                              # type check src + tests + scripts ([tool.pyright] in pyproject, same config Pylance uses)
pytest                                        # full suite (testpaths=tests, -q)
pytest tests/test_continuous_contact.py::test_name  # single test
pytest -k "transition or toe"                 # subset by keyword

# Build mesh -> FEM -> shared contact operators from a JSON config (skipped if output_dir fingerprint matches)
compliance-fem-contact-lookup configs/setup.json [--force] [--save-fem] [--skip-plots] [--quiet]

streamlit run src/compliance_fem/gui/app.py   # GUI (multipage; gait replay in gui/pages/)
python scripts/validate_continuous_contact.py [config]   # validation study vs direct FEM, continuity at transitions
visco-force-map ...                           # see docs/viscoelasticity.md
```

Tests build tiny synthetic soles in `tests/conftest.py` (`N_b = 13`, flat / rocker / asymmetric variants; fixtures `*_case` = `(ComplianceResult, ContactModel)`, `*_model`) instead of the measured geometry, so they run without `outputs/`. A real-data COP test skips unless the Wang data and `outputs/contact_lookup/contact_lookup.npz` exist. `tests/direct_fem.py` is an independent direct sparse-FEM reference (top clamped, bottom nodal forces, plate constraints) that the condensed operators are checked against. It is test-only and must not be imported from `src/`.

## What this is

A 2D plane-strain FEM model of a carbon-plated running-shoe sole: two foam regions (the lower one may have a modulus that varies heel→toe) plus an inextensible Euler–Bernoulli plate, in unilateral contact with flat ground. The heavy work runs **offline once per geometry/material**; runtime only uses precomputed shared operators, so it never solves FEM.

Pipeline (each stage feeds the next):

1. **geometry/**: normalized sole CSV (`data/geometry/sole_geometry.csv`) → validated profile → Gmsh two-foam mesh with tagged boundaries and a plate polyline.
2. **fem/**: assemble K **without** displacement BCs; the three rigid modes are removed with a saddle-point gauge (`R^T d = 0`). The output is component-major `[u; v]` nodal-force compliance blocks `C_tt, C_tb, C_bt, C_bb` (each `2n×2n`). Plate inextensibility rows `B_p` are exact constraints whose multipliers equal the axial force. The same factorization gives the plate and render (all-vertex) influence matrices.
3. **contact/contact_operator.py**: the blocks are condensed once to top-clamped bottom operators (`G_bb` symmetric positive definite, plus top-force / rigid-amplitude / toe-mode maps). `contact/contact_model.py` bundles them with the plate influence and render section into the schema-13 `contact_lookup.npz`.
4. **Runtime**: `contact/continuous_contact.py` solves `0 ≤ λ ⊥ g = q + T_y 1 + Hλ ≥ 0`, `Σλ = −Fy*` over **every** bottom node with the warm-started active-set solver in `contact/active_set.py` (PDAS + Murty least-index fallback). `contact/continuous_toe.py` couples the exact passive toe spring (nested Brent in θ, one converged contact solve per residual, continuation between samples). `api.SoleModel.step()` and `gait/replay.py` wrap it, and `contact/out_of_plane.py` adds auxiliary 2.5D out-of-plane tangent stiffnesses from each converged state (read-only; they must never feed back into the sagittal solve). There are no contact-interval tables anywhere.
5. **gui/** and **gait/**: a Streamlit front end (rebuilds the model in a subprocess via `rebuild_worker.py`) and Wang et al. gait force replay.

`contact/model_setup.py` owns the JSON config (`configs/setup.json`; relative paths resolve against the config file's directory), fingerprints it, and decides whether an existing model file can be reused.

## Invariants to preserve

- **Public interface** (README "Public interface"): the CLIs, `compliance_fem.api` (`SoleModel`, `SoleState`), the listed `contact.model_setup` functions, `compliance_fem.viscoelasticity.create_material`, and the top-level `SoleConfig` / `compute_compliance` / `ComplianceResult`. Everything else is internal and may be changed freely.
- Model file `schema_version` is **13**. Files with any other version are rejected with a message to regenerate, and there is no fallback. Bump it when the stored layout changes.
- Units: solvers use SI. The model is per metre of width; `SoleModel` divides loads by `shoe_width_m` and multiplies forces and moments back; the toe spring is per shoe (`Q_total = width · Q`). The GUI shows mm / kN / Pa / N·mm².
- Sign conventions: φ, φ_ref, and varphi are CCW-positive. Top `F` is foot-on-shoe (`F_top = -GRF`, so stance has `Fy < 0`). Positive gap means open. Positive normal reaction `λ` means compressive. Tangential contact is proportional shear, `τ = (Fx*/Fy*) λ`. The top-shape sign lives only in `corotation.SHAPE_MODE_SIGN`. `θ = arctan α` exactly.
- The frame translation `T` is global (one for every active set; `T_x` is a gauge fixed at the frame point `(L/2, 0)`); never introduce contact-set-specific anchors or gauges, or fields will jump at active-set changes.
- Rotation is exact at the boundary (true sin/cos), but strain is linear. This is a deliberate infinitesimal-strain approximation, not geometric nonlinearity.
- Unconverged or inadmissible states are **flagged, never regularized or clipped**; tolerances are scale-aware (`ContactTolerances`). Disconnected contact patches are reported separately, never filled in. The contact-edge estimate is display only.
- The deformed-shape display scale `s_d` multiplies only the elastic part, never φ.
- scikit-fem v12 API quirks are documented in the README "scikit-fem API notes" (for example, use `Basis.nodal_dofs` for rigid modes).

## Docs

README.md has the overview math and sign conventions. Topic deep-dives are in `docs/`: `continuous_contact.md` (formulation, solver, storage, complexity, validation), `out_of_plane_stiffness.md` (2.5D lateral / roll / yaw tangent), `measured_sole.md`, `external_api.md`, `passive_toe_spring.md`, `gait_replay.md`, `viscoelasticity.md`. Update the relevant doc whenever you change a convention or the schema.
