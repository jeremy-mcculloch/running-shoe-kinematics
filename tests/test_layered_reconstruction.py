"""Direct FEM vs compliance reconstruction and mesh-refinement tests."""

from __future__ import annotations

import numpy as np

from compliance_fem.boundaries import (
    assemble_traction_load,
    build_vector_selector,
    split_uv,
    stacked_uv,
    vector_boundary_data,
    yy_block,
)
from compliance_fem.compliance import compute_compliance
from compliance_fem.config import LayeredPlateConfig


def _small_config(nx: int, ny: int) -> LayeredPlateConfig:
    return LayeredPlateConfig(
        L=0.30,
        h1_heel=0.025,
        h1_toe=0.015,
        h2_heel=0.020,
        h2_toe=0.030,
        E1=2.0e6,
        nu1=0.30,
        E_heel=5.0e5,
        E_toe=1.5e6,
        nu2=0.30,
        EI_plate=10.0,
        nx=nx,
        ny1=ny,
        ny2=ny,
        element_order=1,
    )


def _reconstruct_from_top_force(result, F_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_q = result.n_primal
    assert n_q is not None
    assert result.factorization is not None
    u_top, v_top, _, _ = vector_boundary_data(result.basis, "top")
    u_bot, v_bot, _, _ = vector_boundary_data(result.basis, "bottom")
    S_top = build_vector_selector(u_top, v_top, n_q)
    S_bottom = build_vector_selector(u_bot, v_bot, n_q)
    n_extra = result.n_lambda + 3
    rhs = np.concatenate([S_top.T @ F_t, np.zeros(n_extra)])
    sol = result.factorization.solve(rhs)
    U = sol[:n_q]
    return np.asarray(S_top @ U), np.asarray(S_bottom @ U)


def test_compliance_matches_augmented_solve() -> None:
    result = compute_compliance(_small_config(6, 2))
    traction = lambda x: np.sin(np.pi * x / result.config.L)
    F_ty = assemble_traction_load(result.basis, "top", traction, result.config.order)
    n_t = len(result.x_top)
    F_t = stacked_uv(np.zeros(n_t), F_ty)
    U_top, U_bottom = _reconstruct_from_top_force(result, F_t)
    np.testing.assert_allclose(U_top, result.Ctt_force @ F_t, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(U_bottom, result.Cbt_force @ F_t, rtol=1e-9, atol=1e-12)
    _, v_top = split_uv(U_top, n_t)
    if result.B_p is not None:
        n_q = result.n_primal
        u_dofs, v_dofs, _, _ = vector_boundary_data(result.basis, "top")
        S_top = build_vector_selector(u_dofs, v_dofs, n_q)
        n_extra = result.n_lambda + 3
        rhs = np.concatenate([S_top.T @ F_t, np.zeros(n_extra)])
        U = result.factorization.solve(rhs)[:n_q]
        assert float(np.linalg.norm(result.B_p @ U)) < 1e-8
        assert float(np.linalg.norm(v_top)) > 0.0


def _thickness_change(result) -> np.ndarray:
    pressure = lambda x: np.sin(np.pi * np.asarray(x) / result.config.L)
    F_top_y = assemble_traction_load(result.basis, "top", lambda x: -pressure(x), result.config.order)
    F_bottom_y = assemble_traction_load(
        result.basis, "bottom", lambda x: pressure(x), result.config.order
    )
    n_t = len(result.x_top)
    n_b = len(result.x_bottom)
    Ctt_yy = yy_block(result.Ctt_force, n_t, n_t)
    Ctb_yy = yy_block(result.Ctb_force, n_t, n_b)
    Cbt_yy = yy_block(result.Cbt_force, n_b, n_t)
    Cbb_yy = yy_block(result.Cbb_force, n_b, n_b)
    v_top = Ctt_yy @ F_top_y + Ctb_yy @ F_bottom_y
    v_bottom = Cbt_yy @ F_top_y + Cbb_yy @ F_bottom_y
    x_common = np.linspace(0.0, result.config.L, 40)
    return np.interp(x_common, result.x_top, v_top) - np.interp(
        x_common, result.x_bottom, v_bottom
    )


def test_smooth_load_mesh_refinement() -> None:
    coarse = compute_compliance(_small_config(4, 2))
    medium = compute_compliance(_small_config(8, 3))
    fine = compute_compliance(_small_config(12, 4))
    d_coarse = _thickness_change(coarse)
    d_medium = _thickness_change(medium)
    d_fine = _thickness_change(fine)
    err_cm = float(np.linalg.norm(d_coarse - d_medium))
    err_mf = float(np.linalg.norm(d_medium - d_fine))
    assert err_mf < err_cm
