"""Internal plate response: influence matrices, Hermite postprocess, runtime fields.

The plate is the Euler–Bernoulli / Hermite interface between the two foams.
At build time the shared unit-boundary-load solve of the factorized augmented
free-body operator (``contact_model.build_field_recovery``, which also builds the
render section) gives the plate rotation and axial force per unit boundary load
and per unit rigid amplitude (:class:`PlateResponse`); the plate nodal ``u, v``
are rows of the render influence. At runtime one continuous contact solution supplies
``[F_t; f_b]`` and the rigid amplitudes, the plate fields are two small matrix
products (``contact_model.plate_fields``) and the centreline is rebuilt with
cubic Hermite interpolation.

Sign conventions (must match ``plate.py``)
------------------------------------------
* Plate tangent ``t_p`` points heel → toe.
* Plate normal ``n_p = (-t_y, t_x)``; for a flat plate ``n_p = (0, 1)`` (up).
* Rotation DOF ``θ = dw/ds`` with ``w = n_p · (u, v)``.
* Axial constraint rows are unscaled: ``t_p · (u_{j+1} - u_j) = 0``.
  The Lagrange multiplier therefore has force units and equals the element
  axial constraint force under this convention (positive tension when the
  multiplier resists relative extension in the ``+t_p`` sense of the row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from compliance_fem.contact.corotation import transform_to_fixed_frame
from compliance_fem.fem.plate import plate_element_frames

if TYPE_CHECKING:
    from compliance_fem.fem.compliance import ComplianceResult

DEFAULT_SAMPLES_PER_ELEMENT = 16
PLATE_KEY_PREFIX = "plate_"
# Stored arrays (the reference coordinates come from the render mesh points).
PLATE_ARRAY_FIELDS = (
    "node_ids",
    "theta",
    "plate_tension",
    "inextensibility_residual",
    "rigid_theta",
    "rigid_inextensibility_residual",
)

@dataclass
class PlateResponse:
    """Plate geometry and its fields per unit boundary load / unit rigid amplitude.

    Load columns follow the stacked boundary load ``[F_t (2 n_t); F_bottom (2 n_b)]``
    (component-major); rigid columns are the three ``vector_rigid_mode_matrix`` modes
    about ``(x_r, y_r)``. Plate recovery for one contact solution is a pair of small
    matrix products. The plate nodal ``u, v`` are not stored here: the plate nodes are
    mesh vertices, so they are rows of the render influence. Element frames come from
    :func:`~compliance_fem.fem.plate.plate_element_frames`, as in the FEM assembly.
    """

    node_ids: np.ndarray  # (n_plate,) mesh vertices, heel to toe
    reference_x: np.ndarray  # (n_plate,)
    reference_y: np.ndarray  # (n_plate,)
    EI_plate_Nm2_per_m: float
    theta: np.ndarray  # (n_plate, 2 n_t + 2 n_b) rotation dw/ds
    plate_tension: np.ndarray  # (n_el, 2 n_t + 2 n_b) element axial force, positive = tension
    inextensibility_residual: np.ndarray  # (n_el, 2 n_t + 2 n_b) B_p q, zero up to roundoff
    rigid_theta: np.ndarray  # (n_plate, 3)
    rigid_inextensibility_residual: np.ndarray  # (n_el, 3)

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_elements(self) -> int:
        return self.n_nodes - 1

    @cached_property
    def element_connectivity(self) -> np.ndarray:
        """(n_el, 2) consecutive node pairs."""
        return np.column_stack([np.arange(self.n_elements), np.arange(1, self.n_nodes)])

    @cached_property
    def _frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return plate_element_frames(np.vstack([self.reference_x, self.reference_y]))

    @property
    def element_lengths(self) -> np.ndarray:
        return self._frames[0]

    @property
    def element_tangents(self) -> np.ndarray:
        """(n_el, 2) unit tangents, heel -> toe."""
        return self._frames[1]

    @property
    def element_normals(self) -> np.ndarray:
        """(n_el, 2) unit normals ``(-t_y, t_x)``."""
        return self._frames[2]

    @cached_property
    def reference_arc_length(self) -> np.ndarray:
        s = np.zeros(self.n_nodes, dtype=float)
        s[1:] = np.cumsum(self.element_lengths)
        return s

    def to_payload(self) -> dict[str, np.ndarray]:
        out = {PLATE_KEY_PREFIX + k: np.asarray(getattr(self, k)) for k in PLATE_ARRAY_FIELDS}
        out[PLATE_KEY_PREFIX + "EI_plate_Nm2_per_m"] = np.asarray(float(self.EI_plate_Nm2_per_m))
        return out

    @classmethod
    def from_payload(cls, data, mesh_points: np.ndarray) -> PlateResponse:
        files = set(getattr(data, "files", data.keys()))
        missing = [k for k in (*PLATE_ARRAY_FIELDS, "EI_plate_Nm2_per_m") if PLATE_KEY_PREFIX + k not in files]
        if missing:
            raise KeyError(f"plate arrays missing: {missing}")
        node_ids = np.asarray(data[PLATE_KEY_PREFIX + "node_ids"], dtype=int)
        return cls(
            node_ids=node_ids,
            reference_x=np.asarray(mesh_points[0, node_ids], dtype=float),
            reference_y=np.asarray(mesh_points[1, node_ids], dtype=float),
            EI_plate_Nm2_per_m=float(np.asarray(data[PLATE_KEY_PREFIX + "EI_plate_Nm2_per_m"]).reshape(-1)[0]),
            **{
                k: np.asarray(data[PLATE_KEY_PREFIX + k], dtype=float)
                for k in PLATE_ARRAY_FIELDS
                if k != "node_ids"
            },
        )


@dataclass
class PlateRuntimeState:
    """Superposed plate response for one contact solution."""

    mesh: PlateResponse
    u_local: np.ndarray
    v_local: np.ndarray
    rotation_local: np.ndarray
    plate_tension: np.ndarray  # (n_el,) element axial force, positive = tension
    u_tangential: np.ndarray
    w_normal: np.ndarray
    maximum_plate_axial_displacement_difference: float
    plate_constraint_residual_norm: float
    sample_s: np.ndarray
    sample_x_local: np.ndarray
    sample_y_local: np.ndarray
    sample_x_fixed: np.ndarray
    sample_y_fixed: np.ndarray
    sample_rotation_local: np.ndarray
    sample_rotation_fixed: np.ndarray
    sample_curvature: np.ndarray
    sample_bending_moment: np.ndarray
    sample_shear_force: np.ndarray
    sample_plate_tension: np.ndarray
    sample_w: np.ndarray
    sample_u_s: np.ndarray
    sample_element_id: np.ndarray
    color_quantity: str = "none"
    color_values: np.ndarray | None = None
    color_units: str = ""
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hermite shape functions (local DOF order w_i, θ_i, w_j, θ_j)
# ---------------------------------------------------------------------------


def hermite_N(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return N1..N4 for cubic Hermite interpolation on [0, L]."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    N2 = L * (xi - 2.0 * xi**2 + xi**3)
    N3 = 3.0 * xi**2 - 2.0 * xi**3
    N4 = L * (-(xi**2) + xi**3)
    return N1, N2, N3, N4


def hermite_dN_ds(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dN_i/ds (θ = dw/ds)."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    dxi = 1.0 / L
    dN1 = (-6.0 * xi + 6.0 * xi**2) * dxi
    dN2 = 1.0 - 4.0 * xi + 3.0 * xi**2
    dN3 = (6.0 * xi - 6.0 * xi**2) * dxi
    dN4 = -2.0 * xi + 3.0 * xi**2
    return dN1, dN2, dN3, dN4


def hermite_d2N_ds2(xi: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return d²N_i/ds² (curvature shape functions)."""
    xi = np.asarray(xi, dtype=float)
    L = float(length)
    dxi2 = 1.0 / L**2
    d2N1 = (-6.0 + 12.0 * xi) * dxi2
    d2N2 = (-4.0 + 6.0 * xi) / L
    d2N3 = (6.0 - 12.0 * xi) * dxi2
    d2N4 = (-2.0 + 6.0 * xi) / L
    return d2N1, d2N2, d2N3, d2N4


def hermite_d3N_ds3(length: float) -> tuple[float, float, float, float]:
    """Return constant d³N_i/ds³ (shear shape functions)."""
    L = float(length)
    return 12.0 / L**3, 6.0 / L**2, -12.0 / L**3, 6.0 / L**2


def hermite_deflection(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    N1, N2, N3, N4 = hermite_N(xi, length)
    return N1 * w_i + N2 * theta_i + N3 * w_j + N4 * theta_j


def hermite_rotation(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    dN1, dN2, dN3, dN4 = hermite_dN_ds(xi, length)
    return dN1 * w_i + dN2 * theta_i + dN3 * w_j + dN4 * theta_j


def hermite_curvature(
    xi: np.ndarray,
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
) -> np.ndarray:
    d2N1, d2N2, d2N3, d2N4 = hermite_d2N_ds2(xi, length)
    return d2N1 * w_i + d2N2 * theta_i + d2N3 * w_j + d2N4 * theta_j


def hermite_shear(
    length: float,
    w_i: float,
    theta_i: float,
    w_j: float,
    theta_j: float,
    EI: float,
) -> float:
    d3N1, d3N2, d3N3, d3N4 = hermite_d3N_ds3(length)
    w_ppp = d3N1 * w_i + d3N2 * theta_i + d3N3 * w_j + d3N4 * theta_j
    return float(EI) * float(w_ppp)


# ---------------------------------------------------------------------------
# Build-time rigid recovery
# ---------------------------------------------------------------------------


def rigid_primal_from_alpha(result: ComplianceResult, alpha: np.ndarray, x_r: float, y_r: float) -> np.ndarray:
    """Continuum rigid displacement matching ``vector_rigid_mode_matrix`` about ``(x_r, y_r)``.

    Free-body force recovery returns the particular solution ``q = C F`` in the FEM
    gauge. Boundary solves then add rigid amplitudes ``Alpha`` so that
    ``C F + R Alpha = W``. This helper builds the matching primal field so plate
    recovery can reconstruct the full BC-satisfying response.
    """
    from compliance_fem.fem.rigid_modes import _component_dofs

    alpha = np.asarray(alpha, dtype=float).reshape(3)
    ax, ay, omega = float(alpha[0]), float(alpha[1]), float(alpha[2])
    q_r = np.zeros(int(result.n_primal), dtype=float)
    x_dofs = _component_dofs(result.basis, 0)
    y_dofs = _component_dofs(result.basis, 1)
    q_r[x_dofs] = ax - omega * (result.basis.doflocs[1, x_dofs] - float(y_r))
    q_r[y_dofs] = ay + omega * (result.basis.doflocs[0, y_dofs] - float(x_r))
    # For theta = dw/ds with w = n.(u, v), a plane rigid rotation has theta == omega.
    q_r[result.plate_rotation_dof_ids] = omega
    return q_r


# ---------------------------------------------------------------------------
# Runtime superposition and Hermite rendering
# ---------------------------------------------------------------------------


COLOR_UNITS = {
    "none": "",
    "normal_displacement": "length",
    "tangential_displacement": "length",
    "rotation": "rad",
    "curvature": "1/length",
    "bending_moment": "force·length",
    "shear_force": "force",
    "plate_tension": "force",
    "strain_energy_density": "energy/length",
}


def nodal_tangential_normal(
    u: np.ndarray,
    v: np.ndarray,
    mesh: PlateResponse,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodal (u_s, w) in node-averaged element frames.

    Each node uses the normalized mean of its adjacent element tangents, which
    reduces to the global plate tangent for a straight plate.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    acc = np.zeros((mesh.n_nodes, 2), dtype=float)
    conn = mesh.element_connectivity
    np.add.at(acc, conn[:, 0], mesh.element_tangents)
    np.add.at(acc, conn[:, 1], mesh.element_tangents)
    t = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    u_s = t[:, 0] * u + t[:, 1] * v
    w = -t[:, 1] * u + t[:, 0] * v
    return u_s, w


def sample_plate_centerline(
    mesh: PlateResponse,
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    plate_tension: np.ndarray,
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT,
) -> dict[str, np.ndarray]:
    """Hermite-sample the plate centerline without duplicating shared nodes."""
    if samples_per_element < 2:
        raise ValueError("samples_per_element must be at least 2.")
    u_s_nodes, w_nodes = nodal_tangential_normal(u, v, mesh)
    EI = float(mesh.EI_plate_Nm2_per_m)

    s_list: list[float] = []
    x_list: list[float] = []
    y_list: list[float] = []
    rot_list: list[float] = []
    curv_list: list[float] = []
    mom_list: list[float] = []
    shear_list: list[float] = []
    tension_list: list[float] = []
    w_list: list[float] = []
    us_list: list[float] = []
    eid_list: list[int] = []

    for e in range(mesh.n_elements):
        i = int(mesh.element_connectivity[e, 0])
        j = int(mesh.element_connectivity[e, 1])
        L_e = float(mesh.element_lengths[e])
        t_e = mesh.element_tangents[e]
        n_e = mesh.element_normals[e]
        # Endpoint values in the element frame.
        w_i = float(n_e[0] * u[i] + n_e[1] * v[i])
        w_j = float(n_e[0] * u[j] + n_e[1] * v[j])
        th_i = float(theta[i])
        th_j = float(theta[j])
        us_i = float(t_e[0] * u[i] + t_e[1] * v[i])
        us_j = float(t_e[0] * u[j] + t_e[1] * v[j])
        X_i = np.array([mesh.reference_x[i], mesh.reference_y[i]])
        # Interior samples only for e>0 at the left endpoint to avoid duplicates.
        n_samp = samples_per_element if e == 0 else samples_per_element - 1
        if e == 0:
            xi = np.linspace(0.0, 1.0, samples_per_element)
        else:
            xi = np.linspace(0.0, 1.0, samples_per_element)[1:]
        s_e = mesh.reference_arc_length[i] + xi * L_e
        w = hermite_deflection(xi, L_e, w_i, th_i, w_j, th_j)
        th = hermite_rotation(xi, L_e, w_i, th_i, w_j, th_j)
        kappa = hermite_curvature(xi, L_e, w_i, th_i, w_j, th_j)
        V = hermite_shear(L_e, w_i, th_i, w_j, th_j, EI)
        us = (1.0 - xi) * us_i + xi * us_j
        # Reference point along the element chord.
        X = X_i[None, :] + (xi * L_e)[:, None] * t_e[None, :]
        r = X + us[:, None] * t_e[None, :] + w[:, None] * n_e[None, :]
        s_list.extend(s_e.tolist())
        x_list.extend(r[:, 0].tolist())
        y_list.extend(r[:, 1].tolist())
        rot_list.extend(th.tolist())
        curv_list.extend(kappa.tolist())
        mom_list.extend((EI * kappa).tolist())
        shear_list.extend([V] * int(xi.size))
        tension_list.extend([float(plate_tension[e])] * int(xi.size))
        w_list.extend(w.tolist())
        us_list.extend(us.tolist())
        eid_list.extend([e] * int(xi.size))

    return {
        "s": np.asarray(s_list, dtype=float),
        "x_local": np.asarray(x_list, dtype=float),
        "y_local": np.asarray(y_list, dtype=float),
        "rotation_local": np.asarray(rot_list, dtype=float),
        "curvature": np.asarray(curv_list, dtype=float),
        "bending_moment": np.asarray(mom_list, dtype=float),
        "shear_force": np.asarray(shear_list, dtype=float),
        "plate_tension": np.asarray(tension_list, dtype=float),
        "w": np.asarray(w_list, dtype=float),
        "u_s": np.asarray(us_list, dtype=float),
        "element_id": np.asarray(eid_list, dtype=int),
        "u_s_nodes": u_s_nodes,
        "w_nodes": w_nodes,
    }


def build_plate_runtime_state(
    mesh: PlateResponse,
    u: np.ndarray,
    v: np.ndarray,
    th: np.ndarray,
    plate_tension: np.ndarray,
    *,
    anchor_x: float,
    d_ax: float,
    d_ay: float,
    varphi: float,
    elastic_scale: float = 1.0,
    samples_per_element: int = DEFAULT_SAMPLES_PER_ELEMENT,
    color_quantity: str = "none",
    anchor_y: float = 0.0,
    anchor: tuple[float, float] | None = None,
) -> PlateRuntimeState:
    """Build Hermite fixed-frame samples from local plate nodal fields.

    ``u, v, th`` are nodal (local frame), ``plate_tension`` per element. The
    fixed-frame placement is ``transform_to_fixed_frame`` about the material
    point ``(anchor_x, anchor_y)`` with local displacement ``(d_ax, d_ay)`` and
    true position ``anchor``; ``elastic_scale`` multiplies only the elastic part.
    """
    u, v, th = (np.asarray(a, dtype=float) for a in (u, v, th))
    plate_tension = np.asarray(plate_tension, dtype=float)
    u_s, w = nodal_tangential_normal(u, v, mesh)
    axial_diff = float(np.max(np.abs(np.diff(u_s)))) if u_s.size > 1 else 0.0
    # Constraint residual from nodal tangential differences (kinematic form).
    inextensibility_terms = np.zeros(mesh.n_elements)
    for e in range(mesh.n_elements):
        i, j = mesh.element_connectivity[e]
        t = mesh.element_tangents[e]
        du = np.array([u[j] - u[i], v[j] - v[i]])
        inextensibility_terms[e] = float(t @ du)
    inextensibility_residual_norm = float(np.linalg.norm(inextensibility_terms))

    samples = sample_plate_centerline(mesh, u, v, th, plate_tension, samples_per_element)
    # Reconstruct reference chord points and elastic displacements so the shared
    # foam/plate transform applies s_d only to the elastic part (never to varphi).
    x_ref = np.zeros_like(samples["x_local"])
    y_ref = np.zeros_like(samples["y_local"])
    for idx, e in enumerate(samples["element_id"]):
        i = int(mesh.element_connectivity[e, 0])
        L_e = float(mesh.element_lengths[e])
        t_e = mesh.element_tangents[e]
        s0 = float(mesh.reference_arc_length[i])
        xi = (float(samples["s"][idx]) - s0) / L_e if L_e > 0 else 0.0
        X_i = np.array([mesh.reference_x[i], mesh.reference_y[i]])
        X = X_i + xi * L_e * t_e
        x_ref[idx] = X[0]
        y_ref[idx] = X[1]
    u_samp = samples["x_local"] - x_ref
    v_samp = samples["y_local"] - y_ref
    x_fix, y_fix = transform_to_fixed_frame(
        x_ref,
        y_ref,
        u_samp,
        v_samp,
        float(anchor_x),
        float(d_ax),
        float(d_ay),
        float(varphi),
        anchor=anchor,
        elastic_scale=float(elastic_scale),
        y_anchor=float(anchor_y),
    )
    rot_fixed = np.asarray(samples["rotation_local"], dtype=float) + float(varphi)
    # Unwrap for continuous plotting.
    rot_fixed = np.unwrap(rot_fixed)

    color = str(color_quantity or "none").lower()
    color_map = {
        "none": None,
        "normal_displacement": samples["w"],
        "tangential_displacement": samples["u_s"],
        "rotation": samples["rotation_local"],
        "curvature": samples["curvature"],
        "bending_moment": samples["bending_moment"],
        "shear_force": samples["shear_force"],
        "plate_tension": samples["plate_tension"],
        # Inextensible plate: bending only, 1/2 M kappa = 1/2 EI kappa^2 per unit length.
        "strain_energy_density": 0.5 * samples["bending_moment"] * samples["curvature"],
    }
    if color not in color_map:
        raise ValueError(f"Unknown plate color quantity '{color_quantity}'.")
    color_values = color_map[color]

    # Summary diagnostics.
    abs_M = np.abs(samples["bending_moment"])
    i_max = int(np.argmax(abs_M)) if abs_M.size else 0
    summary = {
        "max_abs_normal_displacement": float(np.max(np.abs(w))) if w.size else 0.0,
        "max_abs_tangential_displacement_variation": axial_diff,
        "max_abs_local_rotation": float(np.max(np.abs(th))) if th.size else 0.0,
        "max_abs_curvature": float(np.max(np.abs(samples["curvature"]))) if samples["curvature"].size else 0.0,
        "max_abs_bending_moment": float(np.max(abs_M)) if abs_M.size else 0.0,
        "max_abs_shear_force": float(np.max(np.abs(samples["shear_force"]))) if samples["shear_force"].size else 0.0,
        "max_abs_plate_tension": float(np.max(np.abs(plate_tension))) if plate_tension.size else 0.0,
        "plate_constraint_residual": inextensibility_residual_norm,
        "max_moment_reference_s": float(samples["s"][i_max]) if samples["s"].size else float("nan"),
        "max_moment_local_x": float(samples["x_local"][i_max]) if samples["x_local"].size else float("nan"),
        "max_moment_local_y": float(samples["y_local"][i_max]) if samples["y_local"].size else float("nan"),
    }

    return PlateRuntimeState(
        mesh=mesh,
        u_local=u,
        v_local=v,
        rotation_local=th,
        plate_tension=plate_tension,
        u_tangential=u_s,
        w_normal=w,
        maximum_plate_axial_displacement_difference=axial_diff,
        plate_constraint_residual_norm=inextensibility_residual_norm,
        sample_s=samples["s"],
        sample_x_local=samples["x_local"],
        sample_y_local=samples["y_local"],
        sample_x_fixed=x_fix,
        sample_y_fixed=y_fix,
        sample_rotation_local=samples["rotation_local"],
        sample_rotation_fixed=rot_fixed,
        sample_curvature=samples["curvature"],
        sample_bending_moment=samples["bending_moment"],
        sample_shear_force=samples["shear_force"],
        sample_plate_tension=samples["plate_tension"],
        sample_w=samples["w"],
        sample_u_s=samples["u_s"],
        sample_element_id=samples["element_id"],
        color_quantity=color,
        color_values=None if color_values is None else np.asarray(color_values, dtype=float),
        color_units=COLOR_UNITS[color],
        summary=summary,
    )
