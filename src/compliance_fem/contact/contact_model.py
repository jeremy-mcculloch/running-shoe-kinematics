"""Shared-operator contact model: build, serialize and recover full fields.

A :class:`ContactModel` bundles everything the contact runtime needs, none of
it tied to a particular contact set:

* :class:`~compliance_fem.contact.contact_operator.ContactOperator` (top-clamped
  bottom operators, toe mode, rearfoot geometry);
* the :class:`~compliance_fem.geometry.render.RenderSection` (reference mesh and the
  all-vertex displacement influence, for the deformed foam and its strain energy);
* the :class:`~compliance_fem.fem.plate_response.PlateResponse` (plate rotation and
  axial force influence; the plate ``u, v`` are rows of the render influence);
* geometry and build metadata.

It is stored in ``contact_lookup.npz`` (``schema_version`` 17). Files with any
other schema are rejected with a regeneration message.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from compliance_fem.contact.contact_operator import (
    CONTACT_FORMULATION,
    CONTACT_OPERATOR_PREFIX,
    TANGENTIAL_CONTACT_LAW,
    ContactOperator,
    build_contact_operator,
)
from compliance_fem.contact.continuous_contact import ContactSolution, fixed_frame_positions
from compliance_fem.fem.plate_response import PlateResponse
from compliance_fem.geometry.render import RenderSection

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult
    from compliance_fem.fem.plate_response import PlateRuntimeState

LOOKUP_SCHEMA_VERSION = 17
CONTACT_MODEL_FILE_KIND = "continuous_contact_model"
REGENERATE_LOOKUP_MESSAGE = (
    f"This contact-lookup file is not schema_version={LOOKUP_SCHEMA_VERSION} (shared continuous-contact "
    "operator). Regenerate it with `compliance-fem-contact-lookup <config.json> --force`."
)


@dataclass
class ContactModel:
    """Everything the continuous contact runtime needs for one geometry / material.

    ``metadata`` holds the formulation strings, the build diagnostics of the contact
    operator (``reciprocity_error``, ``G_symmetry_error``, ``G_min_eigenvalue``,
    ``G_max_eigenvalue``, ``clamp_condition``) and the bottom foam Poisson ratio.
    """

    contact_operator: ContactOperator
    render: RenderSection
    plate: PlateResponse
    geometry_metadata: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    build_time_s: float = float("nan")
    schema_version: int = LOOKUP_SCHEMA_VERSION

    @property
    def L(self) -> float:
        return float(self.contact_operator.L)

    @property
    def H(self) -> float:
        return float(self.contact_operator.H)

    @property
    def x_top(self) -> np.ndarray:
        return self.contact_operator.x_top

    @property
    def y_top(self) -> np.ndarray:
        return self.contact_operator.y_top

    @property
    def x_bottom(self) -> np.ndarray:
        return self.contact_operator.x_bottom

    @property
    def y_bottom(self) -> np.ndarray:
        return self.contact_operator.y_bottom

    @property
    def n_bottom_nodes(self) -> int:
        return self.contact_operator.n_bottom

    @property
    def n_top_nodes(self) -> int:
        return self.contact_operator.n_top

    @property
    def phi_ref(self) -> float:
        return float(self.contact_operator.phi_ref)

    @property
    def softplus_toe_length(self) -> float:
        return float(self.contact_operator.toe_length)

    @property
    def softplus_kappa(self) -> float:
        return float(self.contact_operator.kappa)

    @property
    def contact_foam_poisson_ratio(self) -> float | None:
        v = self.metadata.get("contact_foam_poisson_ratio")
        return None if v is None else float(v)

    @property
    def EI_plate_Nm2_per_m(self) -> float:
        return float(self.plate.EI_plate_Nm2_per_m)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_field_recovery(
    fem_result: ComplianceResult, *, x_r: float, y_r: float
) -> tuple[RenderSection, PlateResponse]:
    """Render section (every mesh vertex) and plate response from one unit-boundary-load solve."""
    from scipy import sparse

    from compliance_fem.fem.boundaries import build_vector_selector
    from compliance_fem.fem.plate_response import rigid_primal_from_alpha
    from compliance_fem.geometry.render import element_moduli

    md = fem_result.mesh_data
    # Free-body response to a unit force on each top then bottom DOF (component-major columns
    # [F_t; f_b]): one multi-RHS solve of the augmented system. sol = [q; mu; r], q in the FEM
    # gauge (R^T q = 0), mu the plate inextensibility multipliers.
    n_q = fem_result.n_primal
    n_lambda = fem_result.n_lambda
    S_top = build_vector_selector(*fem_result.selector_dofs("top"), n_q)
    S_bottom = build_vector_selector(*fem_result.selector_dofs("bottom"), n_q)
    loads = sparse.hstack([S_top.T, S_bottom.T]).toarray()
    rhs = np.zeros((n_q + n_lambda + 3, loads.shape[1]), dtype=float)
    rhs[:n_q] = loads
    sol = np.asarray(fem_result.factorization.solve(rhs), dtype=float)
    q = sol[:n_q]

    # Render section: every mesh vertex, rows [u...; v...], plus the rigid modes about (x_r, y_r).
    p = np.asarray(md.mesh.p, dtype=float)
    n_v = p.shape[1]
    u_r, v_r = fem_result.node_dofs(np.arange(n_v))
    R_render = np.zeros((2 * n_v, 3), dtype=float)
    R_render[:n_v, 0] = 1.0
    R_render[n_v:, 1] = 1.0
    R_render[:n_v, 2] = -(p[1] - float(y_r))
    R_render[n_v:, 2] = p[0] - float(x_r)
    element_E, element_nu = element_moduli(fem_result)
    render = RenderSection(
        mesh_points=p.copy(),
        mesh_triangles=np.asarray(md.mesh.t, dtype=int).copy(),
        mesh_element_region=np.asarray(md.triangle_id_to_region_id, dtype=int).copy(),
        mesh_element_E=element_E,
        mesh_element_nu=element_nu,
        top_curve_node_ids=np.asarray(md.top_curve_nodes, dtype=int),
        bottom_curve_node_ids=np.asarray(md.bottom_curve_nodes, dtype=int),
        heel_edge_node_ids=np.asarray(md.heel_edge_nodes, dtype=int),
        interface_node_ids=np.asarray(md.interface_nodes, dtype=int),
        upper_foam_loop=np.asarray(md.region_loops["upper_foam"], dtype=int),
        lower_foam_loop=np.asarray(md.region_loops["lower_foam"], dtype=int),
        influence_loads=np.vstack([q[u_r], q[v_r]]),
        influence_rigid=R_render,
    )

    # Plate response: rotation, axial force (the inextensibility multipliers, positive = tension)
    # and the inextensibility residual check; the plate u, v are render rows.
    B_p = fem_result.plate_inextensibility_constraint
    rot = fem_result.plate_rotation_dof_ids
    n_plate = fem_result.n_plate_nodes
    q_rigid = np.column_stack([rigid_primal_from_alpha(fem_result, e, x_r, y_r) for e in np.eye(3)])
    plate = PlateResponse(
        node_ids=fem_result.plate_node_ids,
        reference_x=fem_result.x_plate,
        reference_y=fem_result.y_plate,
        EI_plate_Nm2_per_m=float(fem_result.config.EI_plate_Nm2_per_m),
        theta=q[rot] if rot.size else np.zeros((n_plate, q.shape[1])),
        plate_tension=sol[n_q : n_q + n_lambda],
        inextensibility_residual=np.asarray(B_p @ q, dtype=float),
        rigid_theta=q_rigid[rot] if rot.size else np.zeros((n_plate, 3)),
        rigid_inextensibility_residual=np.asarray(B_p @ q_rigid, dtype=float),
    )
    return render, plate


def build_contact_model(
    fem_result: ComplianceResult,
    *,
    toe_length: float,
    kappa: float,
    reciprocity_tol: float = 1e-6,
) -> ContactModel:
    """Condense one FEM result to the shared contact model (operators, render section, plate)."""
    t0 = time.perf_counter()
    contact_operator, diagnostics = build_contact_operator(
        fem_result, toe_length=toe_length, kappa=kappa, reciprocity_tol=reciprocity_tol
    )
    render, plate = build_field_recovery(fem_result, x_r=contact_operator.x_r, y_r=contact_operator.y_r)
    meta = {
        "contact_formulation": CONTACT_FORMULATION,
        "tangential_contact_law": TANGENTIAL_CONTACT_LAW,
        **diagnostics,
        # Poisson ratio of the bottom (ground-contact) foam: default 2.5D lateral-to-normal ratio.
        "contact_foam_poisson_ratio": float(fem_result.config.region_materials()["lower_foam"].nu),
    }
    model = ContactModel(
        contact_operator=contact_operator, render=render, plate=plate,
        geometry_metadata=fem_result.geometry_metadata, metadata=meta,
    )
    model.build_time_s = time.perf_counter() - t0
    return model


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def jsonable(obj):
    """Recursively convert numpy values (and non-finite floats) for ``json.dumps``."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def _json_array(obj) -> np.ndarray:
    return np.asarray(json.dumps(jsonable(obj)))


def _read_json(data, key: str) -> dict:
    return json.loads(str(np.asarray(data[key]).reshape(-1)[0]))


def resolve_contact_lookup_path(path: Path | str) -> Path:
    """Resolve a lookup file or directory to ``contact_lookup.npz``."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "contact_lookup.npz"
        if not candidate.is_file():
            raise FileNotFoundError(f"No contact_lookup.npz in directory {path}")
        return candidate
    if path.is_file():
        return path
    sibling = path.parent / "contact_lookup.npz"
    if path.suffix.lower() != ".npz" and sibling.is_file():
        return sibling
    raise FileNotFoundError(f"Contact lookup not found: {path}")


def contact_model_payload(model: ContactModel) -> dict[str, Any]:
    return {
        "schema_version": int(LOOKUP_SCHEMA_VERSION),
        "file_kind": np.asarray(CONTACT_MODEL_FILE_KIND),
        "metadata_json": _json_array(model.metadata),
        "geometry_metadata_json": _json_array(model.geometry_metadata),
        "build_time_s": float(model.build_time_s),
        **model.contact_operator.to_payload(),
        **model.render.to_payload(),
        **model.plate.to_payload(),
    }


def save_contact_model(model: ContactModel, output_dir: Path | str) -> Path:
    """Write ``contact_lookup.npz`` (schema 17) into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "contact_lookup.npz"
    np.savez_compressed(npz, **contact_model_payload(model))
    return npz


def load_contact_model(path: Path | str) -> ContactModel:
    """Load the shared-operator model from ``contact_lookup.npz``; other schemas are rejected."""
    npz = resolve_contact_lookup_path(path)
    data = np.load(npz, allow_pickle=True)
    files = set(data.files)
    version = int(data["schema_version"]) if "schema_version" in files else 1
    if version != LOOKUP_SCHEMA_VERSION or CONTACT_OPERATOR_PREFIX + "db_fb" not in files:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} (found schema_version={version})")
    try:
        render = RenderSection.from_payload(data)
        model = ContactModel(
            contact_operator=ContactOperator.from_payload(data),
            render=render,
            plate=PlateResponse.from_payload(data, render.mesh_points),
            geometry_metadata=_read_json(data, "geometry_metadata_json"),
            metadata=_read_json(data, "metadata_json"),
            build_time_s=float(np.asarray(data["build_time_s"]).reshape(-1)[0]),
            schema_version=version,
        )
    except KeyError as exc:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} ({exc})") from exc
    return model


# ---------------------------------------------------------------------------
# Full-field recovery from one solved contact state
# ---------------------------------------------------------------------------


def render_vertex_displacements(model: ContactModel, sol: ContactSolution) -> tuple[np.ndarray, np.ndarray]:
    """Local ``(u, v)`` at every mesh vertex from the shared render influence."""
    return model.render.displacements(sol.boundary_loads, sol.rigid_amplitudes)


def plate_fields(model: ContactModel, sol: ContactSolution) -> dict[str, np.ndarray]:
    """Plate nodal ``u, v, theta`` and element ``plate_tension`` (axial force, positive = tension; local frame)."""
    plate, render = model.plate, model.render
    loads, a = sol.boundary_loads, sol.rigid_amplitudes
    rows = render.uv_rows(plate.node_ids)
    uv = render.influence_loads[rows] @ loads + render.influence_rigid[rows] @ a
    n_p = plate.n_nodes
    return {
        "u": uv[:n_p],
        "v": uv[n_p:],
        "theta": plate.theta @ loads + plate.rigid_theta @ a,
        "plate_tension": plate.plate_tension @ loads,
        "inextensibility_residual_norm": np.atleast_1d(
            np.linalg.norm(plate.inextensibility_residual @ loads + plate.rigid_inextensibility_residual @ a)
        ),
    }


def display_pivot(model: ContactModel, sol: ContactSolution) -> dict[str, float]:
    """Reaction-weighted contact centroid used as the ``s_d`` display pivot.

    Continuous in the solution (it is an average weighted by ``lambda``), so the
    exaggerated deformed shape never jumps when the active set changes; with no
    load it is the lowest bottom node. It only matters for ``s_d != 1``.
    """
    op = model.contact_operator
    n = op.n_bottom
    lam = sol.lam
    if lam.sum() > 0.0:
        w = lam / lam.sum()
    else:
        w = np.zeros(n)
        w[int(np.argmin(sol.gap))] = 1.0
    X, Y = float(w @ op.x_bottom), float(w @ op.y_bottom)
    du, dv = float(w @ sol.d_b[:n]), float(w @ sol.d_b[n:])
    rx, ry = fixed_frame_positions(op, sol, np.array([X]), np.array([Y]), np.array([du]), np.array([dv]))
    return {"x": X, "y": Y, "d_x": du, "d_y": dv, "fixed_x": float(rx[0]), "fixed_y": float(ry[0])}


def contact_label(first: int, last: int, n_bottom: int) -> str:
    """Reporting label of the active span: none / heel / toe / full / interior."""
    if first < 0:
        return "none"
    if first == 0 and last == n_bottom - 1:
        return "full"
    if first == 0:
        return "heel"
    if last == n_bottom - 1:
        return "toe"
    return "interior"


def plate_state_for_solution(
    model: ContactModel,
    sol: ContactSolution,
    *,
    elastic_scale: float = 1.0,
    samples_per_element: int = 16,
    color_quantity: str = "none",
) -> PlateRuntimeState:
    """Hermite plate state (fixed frame) for one contact solution."""
    from compliance_fem.fem.plate_response import build_plate_runtime_state

    f = plate_fields(model, sol)
    piv = display_pivot(model, sol)
    return build_plate_runtime_state(
        model.plate, f["u"], f["v"], f["theta"], f["plate_tension"],
        anchor_x=piv["x"], anchor_y=piv["y"], d_ax=piv["d_x"], d_ay=piv["d_y"],
        anchor=(piv["fixed_x"], piv["fixed_y"]), varphi=float(sol.varphi),
        elastic_scale=float(elastic_scale), samples_per_element=int(samples_per_element),
        color_quantity=str(color_quantity),
    )


def selection_proxy(model: ContactModel, frame, *, phi: float | None = None) -> SimpleNamespace:
    """State object for ``plotting.shape_render.build_shape_plot_data``.

    ``frame`` is a :class:`~compliance_fem.contact.continuous_toe.ToeFrameSolution`
    or a bare :class:`ContactSolution`.
    """
    sol: ContactSolution = getattr(frame, "contact", frame)
    op = model.contact_operator
    n = op.n_bottom
    piv = display_pivot(model, sol)
    d = sol.diagnostics
    first, last = int(d["first_active"]), int(d["last_active"])
    return SimpleNamespace(
        solution=sol,
        contact_label=contact_label(first, last, n),
        contact_start_index=first,
        contact_end_index=last,
        active=sol.active.copy(),
        patches=d["patches"],
        lam=sol.lam.copy(),
        gap=sol.gap.copy(),
        anchor_x=piv["x"],
        anchor_y=piv["y"],
        d_ax=piv["d_x"],
        d_ay=piv["d_y"],
        anchor_fixed=(piv["fixed_x"], piv["fixed_y"]),
        varphi=float(sol.varphi),
        alpha=float(sol.alpha),
        phi=float(phi) if phi is not None else float(sol.varphi + op.phi_ref),
        theta=float(getattr(frame, "theta_rad", np.arctan(sol.alpha))),
        Fx_star=float(sol.F_star[0]),
        Fy_star=float(sol.F_star[1]),
        full_bottom_u=sol.d_b[:n].copy(),
        full_bottom_v=sol.d_b[n:].copy(),
    )
