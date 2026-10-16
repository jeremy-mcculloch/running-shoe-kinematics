"""Shared-operator contact model: build, serialize and recover full fields.

A :class:`ContactModel` bundles everything the contact runtime needs, none of
it tied to a particular contact set:

* :class:`~compliance_fem.contact.contact_operator.ContactOperator` (top-clamped
  bottom operators, toe mode, rearfoot geometry);
* the plate influence matrices and plate mesh (plate fields from ``[F_t; f_b]``
  and the rigid amplitudes);
* the measured-sole render section (deformed foam mesh / strain energy);
* geometry and build metadata.

It is stored in ``contact_lookup.npz`` (``schema_version`` 13). Older files
(per-interval records) are rejected with a regeneration message.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

from compliance_fem.contact.contact_operator import (
    CONTACT_FORMULATION,
    CONTACT_OPERATOR_PREFIX,
    TANGENTIAL_CONTACT_LAW,
    ContactOperator,
    build_contact_operator,
)
from compliance_fem.contact.continuous_contact import ContactSolution, fixed_frame_positions

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult
    from compliance_fem.fem.plate_response import PlateRuntimeState

LOOKUP_SCHEMA_VERSION = 13
CONTACT_MODEL_FILE_KIND = "continuous_contact_model"
REGENERATE_LOOKUP_MESSAGE = (
    f"This contact-lookup file is not schema_version={LOOKUP_SCHEMA_VERSION} (shared continuous-contact "
    "operator). Regenerate it with `compliance-fem-contact-lookup <config.json> --force`."
)
PLATE_INFLUENCE_ARRAY_FIELDS = (
    "u", "v", "theta", "lam", "bp", "top", "bottom",
    "rigid_u", "rigid_v", "rigid_theta", "rigid_bp", "rigid_top", "rigid_bottom",
)
PLATE_INFLUENCE_KEY_PREFIX = "plate_influence_"
PLATE_MESH_KEYS = (
    "plate_node_ids", "plate_reference_x", "plate_reference_y", "plate_reference_arc_length",
    "plate_element_connectivity", "plate_element_lengths", "plate_element_tangents", "plate_element_normals",
    "plate_u_dof_ids", "plate_v_dof_ids", "plate_rotation_dof_ids", "plate_constraint_ids",
    "plate_tangent", "plate_normal",
)


@dataclass
class ContactModel:
    """Everything the continuous contact runtime needs for one geometry / material."""

    operator: ContactOperator
    plate_mesh: Any = None  # PlateMeshInfo | None
    plate_influence: Any = None  # PlateInfluence | None
    render_section: dict = field(default_factory=dict)
    geometry_metadata: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    plate_metadata: dict = field(default_factory=dict)
    x_plate: np.ndarray | None = None
    y_plate: np.ndarray | None = None
    build_time_s: float = float("nan")
    schema_version: int = LOOKUP_SCHEMA_VERSION

    @property
    def L(self) -> float:
        return float(self.operator.L)

    @property
    def H(self) -> float:
        return float(self.operator.H)

    @property
    def x_top(self) -> np.ndarray:
        return self.operator.x_top

    @property
    def y_top(self) -> np.ndarray:
        return self.operator.y_top

    @property
    def x_bottom(self) -> np.ndarray:
        return self.operator.x_bottom

    @property
    def y_bottom(self) -> np.ndarray:
        return self.operator.y_bottom

    @property
    def n_bottom_nodes(self) -> int:
        return self.operator.n_bottom

    @property
    def n_top_nodes(self) -> int:
        return self.operator.n_top

    @property
    def phi_ref(self) -> float:
        return float(self.operator.phi_ref)

    @property
    def softplus_toe_length(self) -> float:
        return float(self.operator.toe_length)

    @property
    def softplus_kappa(self) -> float:
        return float(self.operator.kappa)

    @property
    def has_plate_response(self) -> bool:
        return self.plate_mesh is not None and self.plate_influence is not None

    @property
    def contact_foam_poisson_ratio(self) -> float | None:
        v = self.metadata.get("contact_foam_poisson_ratio")
        return None if v is None else float(v)

    @property
    def EI_plate_Nm2_per_m(self) -> float:
        return float(self.plate_mesh.EI_plate_Nm2_per_m) if self.plate_mesh is not None else float("nan")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _supports_plate_recovery(fem_result: ComplianceResult | None) -> TypeGuard[ComplianceResult]:
    if fem_result is None:
        return False
    return (
        fem_result.B_p is not None
        and fem_result.factorization is not None
        and fem_result.x_plate is not None
        and fem_result.basis is not None
        and fem_result.n_primal is not None
    )


def _contact_foam_poisson_ratio(fem_result) -> float | None:
    try:
        return float(fem_result.config.region_materials()["lower_foam"].nu)
    except (AttributeError, KeyError, TypeError):
        return None


def build_contact_model(
    fem_result: ComplianceResult,
    *,
    toe_length: float,
    kappa: float,
    reciprocity_tol: float = 1e-6,
) -> ContactModel:
    """Condense one FEM result to the shared contact model (operators, plate, render)."""
    from compliance_fem.contact.compliance_blocks import get_compliance_block_matrix, prepare_compliance_blocks
    from compliance_fem.fem.plate_response import build_plate_influence, extract_plate_mesh_info
    from compliance_fem.geometry.render import build_render_section, has_render_inputs

    t0 = time.perf_counter()
    blocks = get_compliance_block_matrix(fem_result)
    prepared, reciprocity_error = prepare_compliance_blocks(blocks, reciprocity_tol=reciprocity_tol)
    op = build_contact_operator(prepared, toe_length=toe_length, kappa=kappa)
    plate_mesh = influence = None
    plate_meta: dict = {}
    if _supports_plate_recovery(fem_result):
        plate_mesh = extract_plate_mesh_info(fem_result)
        influence = build_plate_influence(fem_result, plate_mesh, x_r=op.x_r, y_r=op.y_r)
        plate_meta = {
            "n_foam": int(plate_mesh.n_foam),
            "n_primal": int(plate_mesh.n_primal),
            "EI_plate_Nm2_per_m": float(plate_mesh.EI_plate_Nm2_per_m),
        }
    render = build_render_section(fem_result, x_r=op.x_r, y_r=op.y_r) if has_render_inputs(fem_result) else {}
    meta = {
        "contact_formulation": CONTACT_FORMULATION,
        "tangential_contact_law": TANGENTIAL_CONTACT_LAW,
        "reciprocity_error": float(reciprocity_error),
        "G_symmetry_error": float(op.G_symmetry_error),
        "G_min_eigenvalue": float(op.G_min_eigenvalue),
        "G_max_eigenvalue": float(op.G_max_eigenvalue),
        "clamp_condition": float(op.clamp_condition),
        "n_bottom_nodes": int(op.n_bottom),
        "n_top_nodes": int(op.n_top),
        # Poisson ratio of the bottom (ground-contact) foam: default 2.5D lateral-to-normal ratio.
        "contact_foam_poisson_ratio": _contact_foam_poisson_ratio(fem_result),
    }
    model = ContactModel(
        operator=op, plate_mesh=plate_mesh, plate_influence=influence, render_section=render,
        geometry_metadata=dict(getattr(fem_result, "geometry_metadata", None) or {}), metadata=meta,
        plate_metadata=plate_meta,
        x_plate=None if prepared.x_plate is None else np.asarray(prepared.x_plate, dtype=float),
        y_plate=None if prepared.y_plate is None else np.asarray(prepared.y_plate, dtype=float),
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
    from compliance_fem.geometry.render import render_section_to_payload

    payload: dict[str, Any] = {
        "schema_version": int(LOOKUP_SCHEMA_VERSION),
        "file_kind": np.asarray(CONTACT_MODEL_FILE_KIND),
        "metadata_json": _json_array(model.metadata),
        "build_time_s": float(model.build_time_s),
        **model.operator.to_payload(),
    }
    if model.has_plate_response:
        pm = model.plate_mesh
        for key in PLATE_MESH_KEYS:
            attr = key if key in ("plate_tangent", "plate_normal") else key[len("plate_"):]
            payload[key] = np.asarray(getattr(pm, attr))
        payload["EI_plate_Nm2_per_m"] = float(pm.EI_plate_Nm2_per_m)
        payload["plate_metadata_json"] = _json_array(model.plate_metadata)
        for name in PLATE_INFLUENCE_ARRAY_FIELDS:
            payload[PLATE_INFLUENCE_KEY_PREFIX + name] = np.asarray(getattr(model.plate_influence, name))
    if model.x_plate is not None:
        payload["x_plate"] = model.x_plate
    if model.y_plate is not None:
        payload["y_plate"] = model.y_plate
    if model.render_section:
        payload.update(render_section_to_payload(model.render_section, model.geometry_metadata))
    else:
        payload["geometry_metadata_json"] = _json_array(model.geometry_metadata)
    return payload


def save_contact_model(model: ContactModel, output_dir: Path | str) -> Path:
    """Write ``contact_lookup.npz`` (schema 13) into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "contact_lookup.npz"
    np.savez_compressed(npz, **contact_model_payload(model))
    return npz


def load_contact_model(path: Path | str) -> ContactModel:
    """Load the shared-operator model from ``contact_lookup.npz``; other schemas are rejected."""
    from compliance_fem.fem.plate_response import PlateInfluence, plate_mesh_from_lookup
    from compliance_fem.geometry.render import RENDER_FLOAT_KEYS, RENDER_INT_KEYS, RENDER_KEYS

    npz = resolve_contact_lookup_path(path)
    data = np.load(npz, allow_pickle=True)
    files = set(data.files)
    version = int(data["schema_version"]) if "schema_version" in files else 1
    if version != LOOKUP_SCHEMA_VERSION or CONTACT_OPERATOR_PREFIX + "G_bb" not in files:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} (found schema_version={version})")
    meta = _read_json(data, "metadata_json")
    op = ContactOperator.from_payload(data, metadata=meta)
    plate_mesh = influence = None
    plate_meta: dict = {}
    if "plate_metadata_json" in files:
        plate_meta = _read_json(data, "plate_metadata_json")
        ns = SimpleNamespace(
            has_plate_response=True,
            EI_plate_Nm2_per_m=float(np.asarray(data["EI_plate_Nm2_per_m"]).reshape(-1)[0]),
            plate_metadata=plate_meta,
            **{k: np.asarray(data[k]) for k in PLATE_MESH_KEYS},
        )
        plate_mesh = plate_mesh_from_lookup(ns)
        influence = PlateInfluence(
            n_top=op.n_top, n_bottom=op.n_bottom,
            **{n: np.asarray(data[PLATE_INFLUENCE_KEY_PREFIX + n], dtype=float) for n in PLATE_INFLUENCE_ARRAY_FIELDS},
        )
    render: dict = {}
    if any(k in files for k in RENDER_KEYS):
        missing = [k for k in RENDER_KEYS if k not in files]
        if missing:
            raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Render section is missing {missing}.")
        render = {
            **{k: np.asarray(data[k], dtype=int) for k in RENDER_INT_KEYS},
            **{k: np.asarray(data[k], dtype=float) for k in RENDER_FLOAT_KEYS},
        }
    geometry = _read_json(data, "geometry_metadata_json") if "geometry_metadata_json" in files else {}
    return ContactModel(
        operator=op, plate_mesh=plate_mesh, plate_influence=influence, render_section=render,
        geometry_metadata=geometry, metadata=meta, plate_metadata=plate_meta,
        x_plate=np.asarray(data["x_plate"], dtype=float) if "x_plate" in files else None,
        y_plate=np.asarray(data["y_plate"], dtype=float) if "y_plate" in files else None,
        build_time_s=float(np.asarray(data["build_time_s"]).reshape(-1)[0]),
        schema_version=version,
    )


# ---------------------------------------------------------------------------
# Full-field recovery from one solved contact state
# ---------------------------------------------------------------------------


def render_vertex_displacements(model: ContactModel, sol: ContactSolution) -> tuple[np.ndarray, np.ndarray]:
    """Local ``(u, v)`` at every render node from the shared render influence."""
    sec = model.render_section
    if not sec:
        raise ValueError("Contact model has no render section.")
    d = (
        np.asarray(sec["render_influence_loads"]) @ sol.boundary_loads
        + np.asarray(sec["render_influence_rigid"]) @ sol.rigid_amplitudes
    )
    n = d.size // 2
    return d[:n], d[n:]


def plate_fields(model: ContactModel, sol: ContactSolution) -> dict[str, np.ndarray]:
    """Plate nodal ``u, v, theta`` and element multiplier / axial force (local frame)."""
    from compliance_fem.fem.plate_response import axial_force_from_multiplier

    inf = model.plate_influence
    if inf is None:
        raise ValueError("Contact model has no plate response.")
    loads, a = sol.boundary_loads, sol.rigid_amplitudes
    lam = inf.lam @ loads
    return {
        "u": inf.u @ loads + inf.rigid_u @ a,
        "v": inf.v @ loads + inf.rigid_v @ a,
        "theta": inf.theta @ loads + inf.rigid_theta @ a,
        "lam": lam,
        "axial_force": axial_force_from_multiplier(lam),
        "bp_residual": np.atleast_1d(np.linalg.norm(inf.bp @ loads + inf.rigid_bp @ a)),
    }


def display_pivot(model: ContactModel, sol: ContactSolution) -> dict[str, float]:
    """Reaction-weighted contact centroid used as the ``s_d`` display pivot.

    Continuous in the solution (it is an average weighted by ``lambda``), so the
    exaggerated deformed shape never jumps when the active set changes; with no
    load it is the lowest bottom node. It only matters for ``s_d != 1``.
    """
    op = model.operator
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
) -> PlateRuntimeState | None:
    """Hermite plate state (fixed frame) for one contact solution, or ``None`` without a plate."""
    from compliance_fem.fem.plate_response import build_plate_runtime_state

    if not model.has_plate_response:
        return None
    f = plate_fields(model, sol)
    piv = display_pivot(model, sol)
    return build_plate_runtime_state(
        model.plate_mesh, f["u"], f["v"], f["theta"], f["lam"], f["axial_force"],
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
    op = model.operator
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
