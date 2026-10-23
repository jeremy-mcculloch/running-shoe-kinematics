"""Continuous node-to-ground contact: operators, complementarity solve, transitions, toe coupling.

Numbered comments refer to the validation list in docs/continuous_contact.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from compliance_fem.contact import active_set as active_set_module
from compliance_fem.contact.active_set import STATUS_CONVERGED, solve_mixed_lcp
from compliance_fem.contact.contact_model import (
    LOOKUP_SCHEMA_VERSION,
    load_contact_model,
    plate_fields,
    render_vertex_displacements,
    save_contact_model,
    selection_proxy,
)
from compliance_fem.contact.continuous_contact import (
    ContactTolerances,
    _normal_operator,
    contact_patches,
    solve_contact_state,
)
from compliance_fem.contact.continuous_toe import solve_toe_contact
from compliance_fem.contact.toe_spring import ToeSpringConfig, spring_dU_dalpha
from compliance_fem.gait.replay import replay_stance
from compliance_fem.plotting.shape_render import build_shape_plot_data

from direct_fem import relative_error, solve_top_clamped

WIDTH = 0.1


def _H(model, varphi: float, mu: float = 0.0) -> np.ndarray:
    c, s = np.cos(varphi), np.sin(varphi)
    return _normal_operator(model.contact_operator, c, s, c * mu + s, -s * mu + c)


def _assert_admissible(sol, rel: float = 1e-8) -> None:
    d = sol.diagnostics
    tol = sol.tolerances
    assert sol.converged, d["lcp_status"]
    lam, g, act = sol.lam, sol.gap, sol.active
    lam_scale = max(float(lam.max()), 1.0)
    assert np.all(lam[~act] == 0.0)  # 4. separated nodes carry no reaction
    assert np.max(np.abs(g[act])) <= 1e3 * tol["gap_tolerance"] + 1e-12  # 5. active nodes touch
    assert np.min(g) >= -(1e3 * tol["gap_tolerance"] + 1e-12)  # 6. no penetration
    assert np.min(lam) >= -rel * lam_scale  # 7. no tension
    assert np.max(np.abs(g * lam)) <= rel * lam_scale * 1e-6  # 8. complementarity


POSES = [
    (0.0, 0.0, (0.0, -2000.0)),
    (np.deg2rad(3.0), 0.1, (300.0, -3000.0)),
    (np.deg2rad(-4.0), -0.05, (-200.0, -1500.0)),
    (np.deg2rad(1.0), 0.3, (500.0, -6000.0)),
]


# 1, 3. Operator shape, symmetry and definiteness.
def test_gap_operator_dimensions_symmetry_and_definiteness(rocker_model) -> None:
    op = rocker_model.contact_operator
    n = op.n_bottom
    assert op.db_fb.shape == (2 * n, 2 * n)
    assert rocker_model.metadata["G_symmetry_error"] < 1e-10
    assert rocker_model.metadata["G_min_eigenvalue"] > 0.0
    for varphi in (0.0, 0.07, -0.1):
        H = _H(rocker_model, varphi)
        assert H.shape == (n, n)
        assert np.linalg.norm(H - H.T) <= 1e-12 * np.linalg.norm(H)
        assert np.linalg.eigvalsh(0.5 * (H + H.T))[0] > 0.0
    # Proportional shear makes H nonsymmetric but keeps a positive definite symmetric part.
    Hm = _H(rocker_model, 0.05, mu=0.3)
    assert np.linalg.norm(Hm - Hm.T) > 1e-6 * np.linalg.norm(Hm)
    assert np.linalg.eigvalsh(0.5 * (Hm + Hm.T))[0] > 0.0


# 2. H agrees with direct FEM perturbation solves.
@pytest.mark.parametrize("varphi", [0.0, np.deg2rad(5.0)])
def test_gap_operator_matches_direct_fem(rocker_case, varphi: float) -> None:
    fem, model = rocker_case
    op = model.contact_operator
    n = op.n_bottom
    c, s = np.cos(varphi), np.sin(varphi)
    H = _H(model, varphi)
    for j in (0, n // 3, n // 2, n - 2):
        f = np.zeros(2 * n)
        f[j], f[n + j] = s, c  # unit fixed-frame normal ground force at node j, local components
        ref = solve_top_clamped(fem, np.zeros(2 * op.n_top), f)
        dg = s * ref.bottom_displacement[:n] + c * ref.bottom_displacement[n:]
        assert relative_error(H[:, j], dg) < 1e-9


# 4-8. Complementarity and sign checks over soles and poses.
@pytest.mark.parametrize("sole", ["flat_model", "rocker_model", "asym_model"])
@pytest.mark.parametrize("pose", POSES)
def test_contact_solution_is_admissible(request, sole, pose) -> None:
    model = request.getfixturevalue(sole)
    varphi, alpha, F = pose
    sol = solve_contact_state(model.contact_operator, varphi, alpha, np.array(F))
    _assert_admissible(sol)
    assert sol.diagnostics["admissible"]


# 9-10. Global force and moment equilibrium.
@pytest.mark.parametrize("pose", POSES)
def test_global_force_and_moment_equilibrium(rocker_model, pose) -> None:
    varphi, alpha, F = pose
    sol = solve_contact_state(rocker_model.contact_operator, varphi, alpha, np.array(F))
    d = sol.diagnostics
    Fs = float(np.hypot(*F))
    assert d["force_residual"] < 1e-10 * Fs
    assert d["ground_force_residual"] < 1e-10 * Fs
    assert d["equilibrium_force_residual"] < 1e-10 * Fs
    assert d["equilibrium_moment_residual"] < 1e-10 * Fs * rocker_model.L
    assert sol.tau.sum() == pytest.approx(-F[0], rel=1e-10)
    assert sol.lam.sum() == pytest.approx(-F[1], rel=1e-10)


# 11-12. Reconstructed displacement and plate response agree with direct FEM.
def test_full_field_and_plate_match_direct_fem(rocker_case) -> None:
    fem, model = rocker_case
    sol = solve_contact_state(model.contact_operator, np.deg2rad(2.0), 0.15, np.array([200.0, -3000.0]))
    ref = solve_top_clamped(fem, sol.d_t, sol.f_b)
    assert relative_error(sol.d_b, ref.bottom_displacement) < 1e-9
    assert relative_error(sol.F_t, ref.top_force) < 1e-9
    u_dofs, v_dofs = fem.node_dofs(np.arange(model.render.n_vertices))
    u, v = render_vertex_displacements(model, sol)
    assert relative_error(np.concatenate([u, v]), np.concatenate([ref.u[u_dofs], ref.u[v_dofs]])) < 1e-9
    pf = plate_fields(model, sol)
    assert relative_error(pf["u"], ref.u[fem.plate_u_dof_ids]) < 1e-9
    assert relative_error(pf["v"], ref.u[fem.plate_v_dof_ids]) < 1e-9
    assert relative_error(pf["theta"], ref.u[fem.plate_rotation_dof_ids]) < 1e-9
    assert relative_error(pf["plate_tension"], ref.multipliers[: pf["plate_tension"].size]) < 1e-8


# --- one-node transitions -------------------------------------------------------------


def _bisect_transition(fn, p0: float, p1: float, iters: int = 60):
    """Narrow [p0, p1] around the single active-set change of ``fn(p).active``."""
    a0 = fn(p0).active
    for _ in range(iters):
        pm = 0.5 * (p0 + p1)
        if np.array_equal(fn(pm).active, a0):
            p0 = pm
        else:
            p1 = pm
    return fn(p0), fn(p1)


def _one_node_bracket(fn, grid):
    sols = [fn(p) for p in grid]
    for k in range(len(grid) - 1):
        diff = sols[k].active ^ sols[k + 1].active
        if diff.sum() == 1:
            return grid[k], grid[k + 1], int(np.flatnonzero(diff)[0])
    raise AssertionError("no one-node transition on the grid")


@pytest.fixture(scope="module")
def entering(rocker_model):
    op = rocker_model.contact_operator

    def fn(Fy):
        return solve_contact_state(op, 0.0, 0.0, np.array([0.1 * abs(Fy), Fy]))

    p0, p1, node = _one_node_bracket(fn, np.linspace(-800.0, -6000.0, 27))
    left, right = _bisect_transition(fn, p0, p1)
    return left, right, node


@pytest.fixture(scope="module")
def leaving(rocker_model):
    op = rocker_model.contact_operator

    def fn(varphi):
        return solve_contact_state(op, varphi, 0.05, np.array([150.0, -3000.0]))

    grid = np.deg2rad(np.linspace(0.0, 6.0, 31))
    p0, p1, node = _one_node_bracket(fn, grid)
    left, right = _bisect_transition(fn, p0, p1)
    assert left.active[node] != right.active[node]
    return left, right, node


# 13. A node enters contact with approximately zero reaction.
def test_node_enters_with_zero_reaction(rocker_model, entering) -> None:
    left, right, node = entering
    h = rocker_model.contact_operator.h_mesh
    assert not left.active[node] and right.active[node]
    assert right.lam[node] <= 1e-6 * right.lam.max()
    assert 0.0 <= left.gap[node] <= 1e-8 * h


# 14. A node leaves contact when its reaction reaches zero.
def test_node_leaves_with_zero_reaction(rocker_model, leaving) -> None:
    left, right, node = leaving
    h = rocker_model.contact_operator.h_mesh
    on, off = (left, right) if left.active[node] else (right, left)
    assert on.lam[node] <= 1e-6 * on.lam.max()
    assert 0.0 <= off.gap[node] <= 1e-8 * h


def _plate_vec(model, sol):
    pf = plate_fields(model, sol)
    return np.concatenate([pf["u"], pf["v"], pf["theta"], pf["plate_tension"]])


# 15-19. Force, moment, displacement, plate and COP are continuous at one-node transitions.
@pytest.mark.parametrize("which", ["entering", "leaving"])
def test_physical_outputs_continuous_across_transition(request, rocker_model, which) -> None:
    left, right, _ = request.getfixturevalue(which)
    dl, dr = left.diagnostics, right.diagnostics
    F = max(float(np.hypot(*left.F_star)), 1.0)
    assert abs(dl["Fx_fixed"] - dr["Fx_fixed"]) < 1e-6 * F
    assert abs(dl["Fy_fixed"] - dr["Fy_fixed"]) < 1e-6 * F
    assert abs(dl["Mz_shoe"] - dr["Mz_shoe"]) < 1e-6 * F * rocker_model.L
    assert np.max(np.abs(left.d_b - right.d_b)) < 1e-6 * np.max(np.abs(left.d_b))
    assert np.max(np.abs(left.lam - right.lam)) < 1e-6 * left.lam.max()
    pl, pr = _plate_vec(rocker_model, left), _plate_vec(rocker_model, right)
    assert np.max(np.abs(pl - pr)) < 1e-6 * np.max(np.abs(pl))
    assert abs(dl["cop_fixed_x"] - dr["cop_fixed_x"]) < 1e-6 * rocker_model.L


# 20. The derivative may change at a transition (the entering reaction grows from zero).
def test_derivative_change_is_permitted(rocker_model, entering) -> None:
    left, right, node = entering
    op = rocker_model.contact_operator
    h = 1e-3 * abs(right.F_star[1])

    def lam_node(Fy):
        return solve_contact_state(op, 0.0, 0.0, np.array([0.1 * abs(Fy), Fy])).lam[node]

    slope_left = (lam_node(left.F_star[1]) - lam_node(left.F_star[1] + h)) / h
    slope_right = (lam_node(right.F_star[1] - h) - lam_node(right.F_star[1])) / h
    assert slope_left == 0.0
    assert slope_right > 0.0


# 21-22. Warm, cold and arbitrary initial active sets give the same solution.
def test_warm_cold_and_random_starts_agree(rocker_model) -> None:
    op = rocker_model.contact_operator
    rng = np.random.default_rng(3)
    for varphi, alpha, F in POSES:
        cold = solve_contact_state(op, varphi, alpha, np.array(F))
        warm = solve_contact_state(op, varphi, alpha, np.array(F), warm_active=cold.active)
        assert warm.lcp.iterations == 1
        for start in [warm, *[None] * 4]:
            a0 = cold.active if start is not None else rng.random(op.n_bottom) < 0.5
            sol = solve_contact_state(op, varphi, alpha, np.array(F), warm_active=a0)
            assert np.array_equal(sol.active, cold.active)
            np.testing.assert_allclose(sol.lam, cold.lam, rtol=1e-9, atol=1e-9 * cold.lam.max())
            np.testing.assert_allclose(sol.d_b, cold.d_b, rtol=1e-9, atol=1e-15)


# 23. Cycling protection: a case where block PDAS revisits an active set.
def test_cycling_falls_back_to_least_index_and_terminates() -> None:
    H = np.array([
        [2.136447268281123, 3.1360739637225628, -4.091931863098667],
        [-1.709352074122451, 1.284651376394389, 1.1064403836474446],
        [2.8692245441418027, -1.8651232094609989, 1.1355572071580624],
    ])
    q = np.array([-1.296833982018917, -0.9062345180253583, -1.4190455494888305])
    r = solve_mixed_lcp(H, q, 1.0, active0=np.array([False, False, True]), gap_tol=1e-12, reaction_tol=1e-12)
    assert r.pivot_rule == "least_index"
    assert r.status == STATUS_CONVERGED
    assert r.lam.min() >= -1e-12 and r.gap.min() >= -1e-12
    assert np.max(np.abs(r.lam * r.gap)) < 1e-12
    assert r.lam.sum() == pytest.approx(1.0)


# 24-25. Disconnected patches are kept separate; one connected set is one interval.
def test_patch_reporting() -> None:
    x = np.linspace(0.0, 1.0, 10)
    mask = np.zeros(10, dtype=bool)
    mask[[1, 2, 3, 6, 7]] = True
    p = contact_patches(mask, x)
    assert [(q.start, q.end, q.n_nodes) for q in p] == [(1, 3, 3), (6, 7, 2)]
    assert contact_patches(np.zeros(10, dtype=bool), x) == []
    one = contact_patches(np.r_[np.zeros(4, bool), np.ones(3, bool), np.zeros(3, bool)], x)
    assert len(one) == 1 and (one[0].start, one[0].end) == (4, 6)


def test_two_low_regions_give_two_patches(rocker_model) -> None:
    op = rocker_model.contact_operator
    n = op.n_bottom
    H = _H(rocker_model, 0.0)
    x = op.x_bottom / op.L
    # Two dips in the free gap (a "W" sole) carry the load in two separate patches.
    q = 1e-3 * (np.cos(4 * np.pi * x) + 1.0) + 1e-4
    r = solve_mixed_lcp(H, q, 50.0, gap_tol=1e-15, reaction_tol=1e-9)
    patches = contact_patches(r.active, op.x_bottom)
    assert r.status == STATUS_CONVERGED and len(patches) == 2
    assert not r.active[n // 2]


def test_single_patch_state_reports_one_interval(rocker_model) -> None:
    sol = solve_contact_state(rocker_model.contact_operator, 0.0, 0.0, np.array([0.0, -2000.0]))
    d = sol.diagnostics
    assert d["n_patches"] == 1 and not d["disconnected_contact"]
    p = d["patches"][0]
    assert (p.start, p.end) == (d["first_active"], d["last_active"])
    assert p.n_nodes == d["n_active"]


# 26. The interpolated contact edge is a reporting quantity only.
def test_edge_estimate_is_reporting_only(rocker_model) -> None:
    op = rocker_model.contact_operator
    sol = solve_contact_state(op, 0.0, 0.0, np.array([0.0, -2000.0]))
    p = sol.diagnostics["patches"][0]
    x = op.x_bottom
    assert x[max(p.start - 1, 0)] <= p.edge_estimate_start_x <= x[p.start]
    assert x[p.end] <= p.edge_estimate_end_x <= x[min(p.end + 1, op.n_bottom - 1)]
    # No extra constraint: the gaps of the adjacent free nodes stay positive.
    if p.start > 0:
        assert sol.gap[p.start - 1] > 0.0
    if p.end < op.n_bottom - 1:
        assert sol.gap[p.end + 1] > 0.0


# 27-29. Exact passive toe spring coupled to contact.
@pytest.mark.parametrize("F,phi", [((150.0, -1500.0), 0.05), ((300.0, -6000.0), -0.05), ((0.0, -3000.0), 0.0)])
def test_passive_toe_spring_is_exact_and_coupled(rocker_model, F, phi) -> None:
    op = rocker_model.contact_operator
    cfg = ToeSpringConfig()
    F_m = np.asarray(F) / WIDTH
    fr = solve_toe_contact(op, F_m, phi, cfg, WIDTH)
    assert fr.converged
    tol = cfg.toe_equilibrium_abs_tol_Nm + cfg.toe_equilibrium_rel_tol * abs(fr.Q_alpha_shoe_on_foot)
    assert abs(fr.toe_equilibrium_residual_Nm) <= tol
    # theta = arctan(alpha) exactly, never alpha = theta.
    assert fr.alpha == pytest.approx(np.tan(fr.theta_rad), rel=1e-15)
    k, th0 = cfg.toe_stiffness_Nm_per_rad, cfg.toe_neutral_angle_rad
    exact = fr.Q_alpha_shoe_on_foot - float(spring_dU_dalpha(fr.alpha, k, th0))
    small_angle = fr.Q_alpha_shoe_on_foot - k * (fr.alpha - th0)
    assert exact == pytest.approx(fr.toe_equilibrium_residual_Nm, abs=1e-12)
    if abs(fr.theta_rad) > 0.05:
        assert abs(small_angle) > 10 * tol
    # The returned contact state is the converged solve at (varphi(alpha*), alpha*).
    varphi = float(op.rearfoot.chord_rotation(phi, fr.alpha))
    again = solve_contact_state(op, varphi, fr.alpha, F_m)
    np.testing.assert_allclose(again.lam, fr.contact.lam, rtol=1e-10, atol=1e-10 * again.lam.max())
    assert fr.varphi == pytest.approx(varphi, abs=1e-14)
    _assert_admissible(fr.contact)


def test_warm_toe_continuation_matches_cold(rocker_model) -> None:
    op = rocker_model.contact_operator
    cfg = ToeSpringConfig()
    F = np.array([300.0, -6000.0]) / WIDTH
    cold = solve_toe_contact(op, F, -0.05, cfg, WIDTH)
    warm = solve_toe_contact(op, F, -0.05, cfg, WIDTH, warm=cold.warm_start())
    assert warm.theta_rad == pytest.approx(cold.theta_rad, abs=1e-11)
    assert warm.toe_iterations < cold.toe_iterations


# 30. A smooth gait sweep shows no jumps at active-set changes.
def test_gait_sweep_has_no_jumps_at_active_set_changes(rocker_model) -> None:
    t = np.linspace(0.0, 0.2, 401)
    Fy = -4000.0 * np.sin(np.pi * t / 0.2) ** 2 - 50.0
    syn = {"times": t, "Fx": 0.15 * Fy * np.cos(np.pi * t / 0.2), "Fy": Fy, "Mz": np.zeros_like(t),
           "phi": np.deg2rad(8.0) * (t / 0.2 - 0.4)}
    r = replay_stance(rocker_model, synthetic=syn, shoe_width_m=WIDTH)
    assert r.frame_valid.all()
    change = np.flatnonzero(np.any(r.active[1:] != r.active[:-1], axis=1))
    assert change.size > 5
    series = {
        "theta": r.theta_deg,
        "Mz": r.predicted_Mz,
        "min_gap": r.contact_scalars["min_gap"],
        "max_disp": np.array([np.max(np.abs(f.contact.d_b)) for f in r.frames]),
        "max_lam": r.normal_reaction.max(axis=1),
    }
    for name, y in series.items():
        dy = np.abs(np.diff(y))
        for k in change:
            lo, hi = max(k - 3, 0), min(k + 4, dy.size)
            neighbours = np.delete(dy[lo:hi], k - lo)
            assert dy[k] <= 5.0 * np.max(neighbours) + 1e-12 * np.max(np.abs(y)), (name, k)


# 32. The online path solves one small block per active-set iteration (no interval enumeration).
def test_online_cost_scales_with_active_set_not_intervals(rocker_model, monkeypatch) -> None:
    calls: list[int] = []
    original = active_set_module.ActiveBlockSolver._factor

    def counting(self, idx):
        calls.append(int(idx.size))
        return original(self, idx)

    monkeypatch.setattr(active_set_module.ActiveBlockSolver, "_factor", counting)
    n = rocker_model.contact_operator.n_bottom
    sol = solve_contact_state(rocker_model.contact_operator, 0.0, 0.0, np.array([0.0, -3000.0]))
    assert len(calls) == sol.lcp.iterations <= 10 < n * (n + 1) // 2
    assert max(calls) <= n


# 33-34. Serialization round trip and rejection of old schemas.
def test_serialization_round_trip(rocker_model, tmp_path) -> None:
    path = save_contact_model(rocker_model, tmp_path)
    loaded = load_contact_model(tmp_path)
    for f in dataclasses.fields(rocker_model.contact_operator):
        a, b = getattr(rocker_model.contact_operator, f.name), getattr(loaded.contact_operator, f.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
    s1 = solve_contact_state(rocker_model.contact_operator, 0.03, 0.1, np.array([100.0, -2500.0]))
    s2 = solve_contact_state(loaded.contact_operator, 0.03, 0.1, np.array([100.0, -2500.0]))
    np.testing.assert_array_equal(s1.lam, s2.lam)
    # Stored arrays are bitwise identical; matrix products may differ in summation order only.
    np.testing.assert_allclose(plate_fields(rocker_model, s1)["v"], plate_fields(loaded, s2)["v"], rtol=1e-13)
    assert path.name == "contact_lookup.npz" and loaded.schema_version == LOOKUP_SCHEMA_VERSION


def test_old_schema_rejected_with_regeneration_message(rocker_model, tmp_path) -> None:
    save_contact_model(rocker_model, tmp_path)
    data = dict(np.load(tmp_path / "contact_lookup.npz", allow_pickle=True))
    data["schema_version"] = np.asarray(16)
    np.savez_compressed(tmp_path / "contact_lookup.npz", **data)
    with pytest.raises(ValueError, match=r"schema_version=17.*--force.*found schema_version=16"):
        load_contact_model(tmp_path)


# 35. Plotting handles no contact, one patch and several patches.
def test_plotting_handles_zero_one_and_two_patches(rocker_model) -> None:
    from compliance_fem.gui.app import _draw_shape, _profile_figure

    op = rocker_model.contact_operator
    none = solve_contact_state(op, 0.0, 0.0, np.array([0.0, 0.0]))
    one = solve_contact_state(op, 0.0, 0.0, np.array([0.0, -2000.0]))
    two = dataclasses.replace(one, active=one.active.copy(), lam=one.lam.copy())
    mid = (one.diagnostics["first_active"] + one.diagnostics["last_active"]) // 2
    two.active[mid] = False
    two.lam[mid] = 0.0
    from compliance_fem.contact.continuous_contact import contact_diagnostics

    two.diagnostics = contact_diagnostics(op, two)
    for sol, n_patches in ((none, 0), (one, 1), (two, 2)):
        shape = build_shape_plot_data(rocker_model, selection_proxy(rocker_model, sol), foam_energy=True,
                                      plate_color_quantity="strain_energy_density")
        assert shape.n_patches == n_patches == len(shape.contact_curves)
        fig = _draw_shape(shape)
        names = [str(t.name) for t in fig.data]  # pyright: ignore[reportAttributeAccessIssue]  (plotly leaves Figure.data untyped)
        assert sum(n.startswith("contact patch") for n in names) == n_patches
        _profile_figure(shape)


def test_no_load_state_touches_without_reaction(rocker_model) -> None:
    sol = solve_contact_state(rocker_model.contact_operator, 0.02, 0.1, np.array([0.0, 0.0]))
    assert not sol.active.any() and np.all(sol.lam == 0.0)
    assert sol.gap.min() == pytest.approx(0.0, abs=1e-15)


def test_tolerances_are_scale_aware() -> None:
    tol = ContactTolerances()
    r1 = tol.resolve(h=0.002, F_scale=1000.0, L=0.25)
    r2 = tol.resolve(h=0.004, F_scale=2000.0, L=0.25)
    assert r2["gap_tolerance"] == pytest.approx(2 * r1["gap_tolerance"])
    assert r2["reaction_tolerance"] == pytest.approx(2 * r1["reaction_tolerance"])
