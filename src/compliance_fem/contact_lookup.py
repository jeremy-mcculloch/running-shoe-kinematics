"""Single-contiguous-interval contact lookup (schema v10).

Every lookup record is one contiguous contact interval ``I_ij = {i..j}`` of the
heel-to-toe ordered bottom nodes, ``0 <= i <= j < N_b``; all ``N_b (N_b + 1) / 2``
intervals are enumerated exactly once. Heel / toe / full contact are derived
labels (``i = 0`` / ``j = N_b - 1`` / both); interior intervals leave both the
heel and the toe free, which is what a concave or curved sole needs.

Per interval the existing boundary-compliance system is assembled from the
full-domain compliance blocks (no new solver), factored once, and solved for
six right-hand sides: the curved-sole closure (affine constant, column 0) and
the five linear basis modes ``[top_shape_alpha, contact_translation_x,
contact_translation_y, contact_rotation_x, contact_rotation_y]``. Every stored
response tensor therefore has a six-column affine axis whose column 0 always
has coefficient exactly 1; ``corotation.contract_basis`` applies it once.

The rotation-mode anchor is the interval midpoint ``x_a = (x_i + x_j)/2`` with
``y_a`` interpolated from the reference bottom profile; it is a numerical
anchor, not a physical contact edge. The physical contact edges are
``l_h = x_i`` and ``l_t = x_j``.

Old heel/toe/full schemas (v6-v8) only hold ``2 N_b - 1`` of the intervals and
use edge or ``L/2`` anchors without the curved-sole closure; they are rejected
with a regeneration message rather than silently treated as complete interval
support.
"""

from __future__ import annotations

import csv
import json
import threading
import time
import warnings
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

from compliance_fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    component_major_indices,
    split_uv,
    vector_rigid_mode_matrix,
)
from compliance_fem.contact_basis import (
    AFFINE_COLUMN_NAMES,
    BASIS_MODE_NAMES,
    BASIS_ORDER,
    COL_ALPHA,
    COL_BX,
    COL_BY,
    CURVED_SOLE_CLOSURE_CONVENTION,
    MODE_BX,
    MODE_BY,
    N_AFFINE_COLUMNS,
    N_BASIS_MODES,
    SHAPE_MODE_DEFINITION,
    SHAPE_MODE_NORMALIZATION,
    build_contact_affine_matrix,
    build_top_affine_matrix,
    shape_mode_phi1,
)
from compliance_fem.contact_topology import (
    ABSENT_NODE_ID,
    CONTACT_SET_MODEL,
    CONTACT_TYPE_CODES,
    ContactInterval,
    ContactType,
    code_of,
    enumerate_intervals,
    interval_anchor_x,
    interval_anchor_y,
    interval_row,
    labels_from_arrays,
    n_intervals,
    type_from_code,
)
from compliance_fem.corotation import SHAPE_MODE_SIGN, reference_chord_angle
from compliance_fem.measured_render import (
    MEASURED_RENDER_FLOAT_KEYS,
    MEASURED_RENDER_INT_KEYS,
    MEASURED_RENDER_KEYS,
    build_measured_render_section,
    is_measured_result,
    render_section_to_payload,
)
from compliance_fem.plate_response import (
    CONSTRAINT_ROW_NORMALIZATION,
    COORDINATE_FRAME,
    PLATE_AXIAL_MODEL,
    PLATE_MODEL,
    PLATE_NORMAL_SIGN,
    PLATE_ROTATION_SIGN,
    PlateInfluence,
    build_plate_influence,
    empty_plate_basis,
    extract_plate_mesh_info,
    plate_basis_from_influence,
)
from compliance_fem.toe_spring import (
    TOE_GENERALIZED_FORCE_DEFINITION,
    RearfootGeometry,
    q_alpha_shoe_on_foot_basis,
)

if TYPE_CHECKING:
    from compliance_fem.compliance import ComplianceResult

LOOKUP_SCHEMA_VERSION = 10
# v9 files (same interval/affine layout, no rigid-amplitude basis or measured
# geometry section) are migrated on load; older layouts are rejected.
MIGRATABLE_SCHEMA_VERSIONS = (9,)
REJECTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8)
GROUND_GEOMETRY = "flat ground y = 0 with upward normal n_g = (0, 1)"
CONTACT_LAW = (
    "perfect sticking on the active interval: every active node is prescribed the full "
    "rotating-frame displacement d_k = d_a + (Q^T - I)(x_k - x_a, 0) + (0, -(y_k - y_a)), "
    "which places it at (x_k, 0) on the ground; free nodes are traction-free"
)
CONTACT_ANCHOR_DEFINITION = "interval_midpoint"
FORCE_SIGN_CONVENTION = (
    "top forces are foot-on-shoe nodal loads (load term of K u = f), rotating frame; "
    "Fx positive heel->toe, Fy positive up (compression is negative)"
)
GAP_SIGN_CONVENTION = (
    "g_k = n_g . r_k^F, the fixed-frame height of a bottom node above the ground; "
    "positive = open, negative = penetration"
)
NORMAL_REACTION_SIGN_CONVENTION = (
    "R_n,k = n_g . Q(varphi) R_k^T with R_k the nodal ground-on-shoe contact force; "
    "positive = compressive, negative = tensile"
)

# Scalar lookup fields per record and affine column.
SCALAR_LOOKUP_FIELDS = ("Fx", "Fy", "Mv", "Mz", "M_toe", "M_toe_vertical")
N_SCALAR_FIELDS = len(SCALAR_LOOKUP_FIELDS)
SCALAR_FX = 0
SCALAR_FY = 1
SCALAR_MV = 2
SCALAR_MZ = 3
SCALAR_TOE = 4
SCALAR_TOE_VERT = 5

CONTACT_TOTAL_FIELDS = ("Fx_contact", "Fy_contact", "Mz_contact")

# Per-record, per-column edge / adjacent-node responses (NaN where absent).
EDGE_RESPONSE_FIELDS = (
    "heel_edge_reaction_local_x",
    "heel_edge_reaction_local_y",
    "toe_edge_reaction_local_x",
    "toe_edge_reaction_local_y",
    "heel_adjacent_free_u",
    "heel_adjacent_free_v",
    "toe_adjacent_free_u",
    "toe_adjacent_free_v",
)

SOLVER_YR = 0.0
MOMENT_ORIGIN = (0.0, 0.0)
# Boundary matrices are flagged rank deficient below this reciprocal condition
# estimate; the numerical rank is then confirmed by an SVD of the actual matrix.
RANK_RCOND_TOL = 1.0e-12
SOLVE_RESIDUAL_TOL = 1.0e-6
EQUILIBRIUM_REL_TOL = 1.0e-8

REJECT_RANK_DEFICIENT = "rank_deficient_boundary_matrix"
REJECT_SOLVE_FAILED = "boundary_solve_failed"
REJECT_SOLVE_RESIDUAL = "solve_residual_exceeds_tolerance"
REJECT_BC_RESIDUAL = "boundary_condition_residual_exceeds_tolerance"
REJECT_EQUILIBRIUM = "equilibrium_residual_exceeds_tolerance"
REJECT_NONFINITE = "nonfinite_response"
REJECT_PLATE = "plate_recovery_residual_exceeds_tolerance"

NODAL_FIELD_NAMES = ("bottom_u", "bottom_v", "reaction_x", "reaction_y", "top_force_x", "top_force_y")
PLATE_FIELD_NAMES = (
    "plate_u_local",
    "plate_v_local",
    "plate_rotation_local",
    "plate_constraint_multiplier",
    "plate_axial_force",
)
NODAL_FIELD_ARRAY_KEYS = tuple(f"{name}_basis" for name in NODAL_FIELD_NAMES)
PLATE_BASIS_ARRAY_KEYS = tuple(f"{name}_basis" for name in PLATE_FIELD_NAMES)
FIELD_SOLVER_COMPLIANCE_KEY = "field_solver_compliance"
PLATE_INFLUENCE_ARRAY_FIELDS = (
    "u", "v", "theta", "lam", "bp", "top", "bottom",
    "rigid_u", "rigid_v", "rigid_theta", "rigid_bp", "rigid_top", "rigid_bottom",
)
PLATE_INFLUENCE_KEY_PREFIX = "plate_influence_"
# Solved records kept per lookup when nodal fields are computed on demand.
FIELD_CACHE_ROWS = 2048

PLATE_MESH_ARRAY_KEYS = (
    "plate_bp_residual",
    "plate_node_ids",
    "plate_reference_x",
    "plate_reference_y",
    "plate_reference_arc_length",
    "plate_element_connectivity",
    "plate_element_lengths",
    "plate_element_tangents",
    "plate_element_normals",
    "plate_u_dof_ids",
    "plate_v_dof_ids",
    "plate_rotation_dof_ids",
    "plate_constraint_ids",
    "plate_tangent",
    "plate_normal",
)
PLATE_RESPONSE_ARRAY_KEYS = PLATE_BASIS_ARRAY_KEYS + PLATE_MESH_ARRAY_KEYS

REGENERATE_LOOKUP_MESSAGE = (
    "This contact-lookup file is not schema_version=10 (or a migratable v9 file) "
    "with single contiguous contact intervals and the six-column affine curved-sole "
    "basis. Older heel/toe/full tables (schema v6-v8) store only 2*N_b-1 of the N_b*(N_b+1)/2 intervals, use "
    "contact-edge or L/2 anchors, and have no curved-sole closure response, so they "
    "cannot be migrated to arbitrary interval support. Regenerate with the current "
    "contact_lookup CLI."
)
REGENERATE_COMPLIANCE_MESSAGE = (
    "This compliance NPZ is not a vector (component-major u,v) file. "
    "Regenerate FEM with the current pipeline so Ctt/Ctb/Cbt/Cbb are 2n x 2n."
)


def contact_type_code(contact_type: ContactType | str) -> int:
    """Integer code for a topology label."""
    return code_of(ContactType(contact_type))


def contact_type_from_code(code: int) -> ContactType:
    """Topology label from its stored integer code."""
    return type_from_code(code)


def record_counts(n_bottom: int) -> dict[str, int]:
    """Theoretical per-label and total interval counts for ``n_bottom`` nodes."""
    n = int(n_bottom)
    total = n_intervals(n)
    if n == 1:
        return {"heel": 0, "toe": 0, "full": 1, "interior": 0, "total": 1}
    return {
        "heel": n - 1,
        "toe": n - 1,
        "full": 1,
        "interior": total - (2 * n - 1),
        "total": total,
    }


def ramp_virtual_displacement(x_top: np.ndarray, a: float) -> np.ndarray:
    """Return rho_a[j] = max(0, x_top[j] - a)."""
    return np.maximum(0.0, np.asarray(x_top, dtype=float) - float(a))


def interpolate_top_height(x_top: np.ndarray, y_top: np.ndarray, a: float) -> float:
    """Height of the top surface at x=a, interpolated from nodal y_top."""
    return float(np.interp(float(a), np.asarray(x_top, dtype=float), np.asarray(y_top, dtype=float)))


def ramp_horizontal_lever(x_top: np.ndarray, y_top: np.ndarray, a: float, H_a: float) -> np.ndarray:
    """eta_a[j] = (y_j - H_a) for x_j >= a, else 0. Flat top => eta = 0."""
    x_top = np.asarray(x_top, dtype=float)
    y_top = np.asarray(y_top, dtype=float)
    eta = np.zeros_like(x_top, dtype=float)
    mask = x_top >= float(a)
    eta[mask] = y_top[mask] - float(H_a)
    return eta


def compute_toe_basis(top_reactions: np.ndarray, x_top: np.ndarray, a: float) -> np.ndarray:
    """Discrete vertical toe moment T = rho_a^T f_{t,y} for one or more RHS columns."""
    rho = ramp_virtual_displacement(x_top, a)
    F = np.asarray(top_reactions, dtype=float)
    if F.ndim == 1:
        return float(rho @ F)
    return rho @ F


def compute_toe_moment(
    f_tx: np.ndarray,
    f_ty: np.ndarray,
    x_top: np.ndarray,
    y_top: np.ndarray,
    a: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Toe moment about P_toe = (a, H_a): T = rho^T f_ty - eta^T f_tx.

    Returns ``(T_toe, T_toe_vertical, H_a)``. Columns of ``f_tx`` / ``f_ty`` are RHS.
    """
    f_tx = np.asarray(f_tx, dtype=float)
    f_ty = np.asarray(f_ty, dtype=float)
    squeeze = f_tx.ndim == 1
    if squeeze:
        f_tx = f_tx[:, None]
        f_ty = f_ty[:, None]
    H_a = interpolate_top_height(x_top, y_top, a)
    rho = ramp_virtual_displacement(x_top, a)
    eta = ramp_horizontal_lever(x_top, y_top, a, H_a)
    T = rho @ f_ty - eta @ f_tx
    T_vert = rho @ f_ty
    if squeeze:
        return np.asarray(T).reshape(-1)[0], np.asarray(T_vert).reshape(-1)[0], H_a
    return np.asarray(T, dtype=float), np.asarray(T_vert, dtype=float), H_a


def warn_if_a_not_on_top_grid(x_top: np.ndarray, a: float) -> None:
    """Warn when a is not a top-node coordinate (nodal ramp is used; no remesh)."""
    x_top = np.asarray(x_top, dtype=float)
    if not np.any(np.isclose(x_top, float(a), rtol=0.0, atol=1e-12)):
        warnings.warn(
            f"Softplus location a={a} is not a top-node coordinate; "
            "toe moment uses the existing nodal ramp without remeshing.",
            UserWarning,
            stacklevel=2,
        )


@dataclass
class ComplianceBlocks:
    """Nodal-force compliance blocks and boundary coordinates."""

    Ctt: np.ndarray
    Ctb: np.ndarray
    Cbt: np.ndarray
    Cbb: np.ndarray
    x_top: np.ndarray
    x_bottom: np.ndarray
    L: float
    H: float
    E: float
    nu: float
    reciprocity_error: float = 0.0
    nx: int | None = None
    ny: int | None = None
    order: int | None = None
    y_top: np.ndarray | None = None
    y_bottom: np.ndarray | None = None
    x_plate: np.ndarray | None = None
    y_plate: np.ndarray | None = None
    geometry_type: str = "rectangle"
    dof_ordering: str = DOF_ORDERING_COMPONENT_MAJOR_UV
    n_top_nodes: int | None = None
    n_bottom_nodes: int | None = None


def _boundary_y(blocks: ComplianceBlocks) -> tuple[np.ndarray, np.ndarray]:
    n_t = len(blocks.x_top)
    n_b = len(blocks.x_bottom)
    y_t = (
        np.asarray(blocks.y_top, dtype=float)
        if blocks.y_top is not None
        else np.full(n_t, float(blocks.H))
    )
    y_b = (
        np.asarray(blocks.y_bottom, dtype=float)
        if blocks.y_bottom is not None
        else np.zeros(n_b, dtype=float)
    )
    return y_t, y_b


def _csr(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([int(np.asarray(v).size) for v in lists])
    flat = (
        np.concatenate([np.asarray(v, dtype=np.int32) for v in lists])
        if lists
        else np.zeros(0, dtype=np.int32)
    )
    return flat.astype(np.int32), offsets


@dataclass
class ContactLookupResult:
    """Interval lookup table: one record per contiguous contact interval.

    Rows follow :func:`contact_topology.enumerate_intervals` (``start`` then
    ``end``); ``interval_row(i, j, N_b)`` maps endpoints to a row. Rejected
    intervals keep their row with ``record_valid = False``, a structured
    ``rejection_reason`` and NaN responses.

    Response tensors have a six-column affine axis ``[closure, alpha, dx, dy,
    rx, ry]`` (``AFFINE_COLUMN_NAMES``); column 0 has coefficient 1.
    """

    contact_start_index: np.ndarray
    contact_end_index: np.ndarray
    n_free: np.ndarray
    n_contact: np.ndarray
    status: list[str]
    x_top: np.ndarray
    x_bottom: np.ndarray
    basis_top_displacements: np.ndarray  # (2 n_t, 6)
    scalar_lookup: np.ndarray  # (n_records, 6, N_SCALAR_FIELDS)
    softplus_a: float
    softplus_kappa: float
    L: float
    H: float
    E: float
    nu: float
    condition_estimates: np.ndarray
    solve_residuals: np.ndarray
    top_displacement_residuals: np.ndarray
    contact_displacement_residuals: np.ndarray
    force_equilibrium_residuals: np.ndarray
    moment_equilibrium_residuals: np.ndarray
    reciprocity_error: float
    record_valid: np.ndarray | None = None
    rejection_reason: list[str] = field(default_factory=list)
    boundary_matrix_rank: np.ndarray | None = None
    boundary_matrix_size: np.ndarray | None = None
    contact_type_codes: np.ndarray | None = None
    contact_start_x: np.ndarray | None = None
    contact_end_x: np.ndarray | None = None
    contact_anchor_reference_x: np.ndarray | None = None
    contact_anchor_reference_y: np.ndarray | None = None
    heel_contact_edge_node_id: np.ndarray | None = None
    toe_contact_edge_node_id: np.ndarray | None = None
    heel_adjacent_free_node_id: np.ndarray | None = None
    toe_adjacent_free_node_id: np.ndarray | None = None
    contact_mask: np.ndarray | None = None
    bottom_u_basis: np.ndarray | None = None  # (n_records, 6, n_b), all bottom nodes
    bottom_v_basis: np.ndarray | None = None
    reaction_x_basis: np.ndarray | None = None  # (n_records, 6, n_b), zero on free nodes
    reaction_y_basis: np.ndarray | None = None
    top_force_x_basis: np.ndarray | None = None  # (n_records, 6, n_t)
    top_force_y_basis: np.ndarray | None = None
    contact_force_total: np.ndarray | None = None  # (n_records, 6, 3)
    balance_residuals: np.ndarray | None = None
    edge_responses: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_records, 6)
    Q_alpha_shoe_on_foot_basis: np.ndarray | None = None  # (n_records, 6)
    toe_generalized_force_source: str = "generated"
    kf_matrix: np.ndarray | None = None
    kf_svals: np.ndarray | None = None
    kf_cond: np.ndarray | None = None
    kf_det: np.ndarray | None = None
    y_top: np.ndarray | None = None
    y_bottom: np.ndarray | None = None
    x_plate: np.ndarray | None = None
    y_plate: np.ndarray | None = None
    geometry_type: str = "rectangle"
    dof_ordering: str = DOF_ORDERING_COMPONENT_MAJOR_UV
    n_top_nodes: int = 0
    n_bottom_nodes: int = 0
    moment_origin: tuple[float, float] = MOMENT_ORIGIN
    solver_xr: float = 0.0
    solver_yr: float = SOLVER_YR
    phi_ref: float = 0.0
    build_time_s: float = float("nan")
    metadata: dict = field(default_factory=dict)
    schema_version: int = LOOKUP_SCHEMA_VERSION
    # Optional plate_response section (layered lookups only).
    has_plate_response: bool = False
    plate_node_ids: np.ndarray | None = None
    plate_reference_x: np.ndarray | None = None
    plate_reference_y: np.ndarray | None = None
    plate_reference_arc_length: np.ndarray | None = None
    plate_element_connectivity: np.ndarray | None = None
    plate_element_lengths: np.ndarray | None = None
    plate_element_tangents: np.ndarray | None = None
    plate_element_normals: np.ndarray | None = None
    plate_u_dof_ids: np.ndarray | None = None
    plate_v_dof_ids: np.ndarray | None = None
    plate_rotation_dof_ids: np.ndarray | None = None
    plate_constraint_ids: np.ndarray | None = None
    plate_tangent: np.ndarray | None = None
    plate_normal: np.ndarray | None = None
    EI_plate: float = float("nan")
    plate_u_local_basis: np.ndarray | None = None  # (n_records, 6, n_plate)
    plate_v_local_basis: np.ndarray | None = None
    plate_rotation_local_basis: np.ndarray | None = None
    plate_constraint_multiplier_basis: np.ndarray | None = None  # (n_records, 6, n_el)
    plate_axial_force_basis: np.ndarray | None = None
    plate_bp_residual: np.ndarray | None = None  # (n_records, 6)
    plate_metadata: dict = field(default_factory=dict)
    # Schema v10: rigid amplitudes about (solver_xr, solver_yr) per affine column.
    rigid_alpha_basis: np.ndarray | None = None  # (n_records, 6, 3)
    # Measured-sole geometry section (empty for rectangle / layered lookups).
    measured_render: dict = field(default_factory=dict)
    geometry_metadata: dict = field(default_factory=dict)
    migrated_from_schema: int | None = None
    # Recomputes per-record nodal fields when they are not stored.
    field_solver: RecordFieldSolver | None = field(default=None, repr=False, compare=False)

    @property
    def is_measured(self) -> bool:
        return self.geometry_type == "measured_sole"

    # --- per-record nodal fields ---------------------------------------------
    @property
    def nodal_fields_stored(self) -> bool:
        return all(getattr(self, key) is not None for key in NODAL_FIELD_ARRAY_KEYS)

    @property
    def has_nodal_fields(self) -> bool:
        return self.nodal_fields_stored or self.field_solver is not None

    def record_fields(self, rows, *, plate: bool = False) -> dict[str, np.ndarray]:
        """Six-column nodal bases ``name -> (len(rows), 6, n)`` for the given records.

        Names are :data:`NODAL_FIELD_NAMES` (plus :data:`PLATE_FIELD_NAMES` with
        ``plate=True``). Stored arrays are sliced; otherwise each record is re-solved
        from the stored compliance blocks (bitwise identical to generation) and cached.
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=int))
        names = NODAL_FIELD_NAMES + (PLATE_FIELD_NAMES if plate else ())
        if plate and not self.has_plate_response:
            raise ValueError("Lookup has no plate response.")
        out: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for name in names:
            arr = getattr(self, f"{name}_basis")
            if arr is None:
                missing.append(name)
            else:
                out[name] = np.asarray(arr)[rows]
        if missing:
            if self.field_solver is None:
                raise ValueError(
                    f"{REGENERATE_LOOKUP_MESSAGE} The lookup stores neither the nodal fields "
                    f"{missing} nor the compliance blocks needed to recompute them."
                )
            per_row = [self.field_solver.fields(self, int(r), plate=plate) for r in rows]
            for name in missing:
                out[name] = np.stack([f[name] for f in per_row]) if per_row else np.zeros((0, N_AFFINE_COLUMNS, 0))
        return out

    # --- record geometry -------------------------------------------------
    @property
    def n_records(self) -> int:
        return int(len(self.contact_start_index))

    @property
    def basis_order(self) -> list[str]:
        return list(BASIS_ORDER)

    @property
    def anchor_reference_x(self) -> np.ndarray:
        return np.asarray(self.contact_anchor_reference_x, dtype=float)

    @property
    def anchor_reference_y(self) -> np.ndarray:
        return np.asarray(self.contact_anchor_reference_y, dtype=float)

    @property
    def valid_mask(self) -> np.ndarray:
        if self.record_valid is None:
            return np.ones(self.n_records, dtype=bool)
        return np.asarray(self.record_valid, dtype=bool)

    @property
    def valid_rows(self) -> np.ndarray:
        return np.flatnonzero(self.valid_mask)

    def interval(self, row: int) -> ContactInterval:
        r = int(row)
        return ContactInterval(
            int(self.contact_start_index[r]), int(self.contact_end_index[r]), int(self.n_bottom_nodes)
        )

    def row_of(self, start: int, end: int) -> int:
        return interval_row(int(start), int(end), int(self.n_bottom_nodes))

    def contact_type(self, row: int) -> ContactType:
        if self.contact_type_codes is None:
            raise ValueError(REGENERATE_LOOKUP_MESSAGE)
        return contact_type_from_code(int(self.contact_type_codes[int(row)]))

    def contact_types(self) -> tuple[ContactType, ...]:
        return tuple(self.contact_type(row) for row in range(self.n_records))

    def record_sets(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``(free, contact)`` bottom-node ids of one record."""
        return self.interval(row).sets()

    def contact_node_ids(self, row: int) -> np.ndarray:
        return self.interval(row).contact_node_ids

    def free_node_ids(self, row: int) -> np.ndarray:
        return self.interval(row).free_node_ids

    def rows_for(self, contact_type: ContactType | str) -> np.ndarray:
        if self.contact_type_codes is None:
            raise ValueError(REGENERATE_LOOKUP_MESSAGE)
        code = contact_type_code(contact_type)
        return np.flatnonzero(np.asarray(self.contact_type_codes, dtype=int) == code)

    def full_contact_row(self) -> int:
        return self.row_of(0, int(self.n_bottom_nodes) - 1)

    def edge_response(self, name: str) -> np.ndarray:
        if name not in self.edge_responses:
            raise KeyError(f"Unknown edge response '{name}'; expected one of {EDGE_RESPONSE_FIELDS}.")
        return self.edge_responses[name]

    # --- legacy aliases (same arrays) --------------------------------------
    @property
    def gap_u_basis(self) -> np.ndarray | None:
        return self.bottom_u_basis

    @property
    def gap_v_basis(self) -> np.ndarray | None:
        return self.bottom_v_basis

    @property
    def gap_basis(self) -> np.ndarray | None:
        return self.bottom_v_basis

    @property
    def reaction_basis(self) -> np.ndarray | None:
        return self.reaction_y_basis

    @property
    def top_force_basis(self) -> np.ndarray | None:
        return self.top_force_y_basis

    def rejection_summary(self) -> dict[str, int]:
        counts = Counter(r for r, ok in zip(self.rejection_reason, self.valid_mask) if not ok)
        return dict(sorted(counts.items()))


def _decode_flag(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def _optional_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _require_vector_compliance_blocks(
    Ctt: np.ndarray,
    x_top: np.ndarray,
    x_bottom: np.ndarray,
    dof_ordering: str,
) -> None:
    n_t = int(len(x_top))
    n_b = int(len(x_bottom))
    if Ctt.ndim != 2 or Ctt.shape[0] % 2 != 0 or Ctt.shape[1] % 2 != 0:
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)
    if Ctt.shape != (2 * n_t, 2 * n_t):
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)
    if n_b < 1:
        raise ValueError("Compliance file has no bottom nodes.")
    if dof_ordering != DOF_ORDERING_COMPONENT_MAJOR_UV:
        raise ValueError(REGENERATE_COMPLIANCE_MESSAGE)


def from_compliance_result(result: ComplianceResult) -> ComplianceBlocks:
    """Extract nodal-force blocks from an in-memory ComplianceResult."""
    cfg = result.config
    dof_ordering = str(getattr(result, "dof_ordering", DOF_ORDERING_COMPONENT_MAJOR_UV))
    Ctt = np.asarray(result.Ctt_force, dtype=float)
    x_top = np.asarray(result.x_top, dtype=float)
    x_bottom = np.asarray(result.x_bottom, dtype=float)
    _require_vector_compliance_blocks(Ctt, x_top, x_bottom, dof_ordering)
    return ComplianceBlocks(
        Ctt=Ctt,
        Ctb=np.asarray(result.Ctb_force, dtype=float),
        Cbt=np.asarray(result.Cbt_force, dtype=float),
        Cbb=np.asarray(result.Cbb_force, dtype=float),
        x_top=x_top,
        x_bottom=x_bottom,
        L=float(cfg.L),
        H=float(cfg.H),
        E=float(cfg.E),
        nu=float(cfg.nu),
        reciprocity_error=float(result.reciprocity_error),
        nx=int(cfg.nx),
        ny=int(cfg.ny),
        order=int(cfg.order),
        y_top=_optional_array(result.y_top),
        y_bottom=_optional_array(result.y_bottom),
        x_plate=_optional_array(result.x_plate),
        y_plate=_optional_array(result.y_plate),
        geometry_type=str(getattr(cfg, "geometry_type", "rectangle")),
        dof_ordering=dof_ordering,
        n_top_nodes=int(getattr(result, "n_top_nodes", len(x_top))),
        n_bottom_nodes=int(getattr(result, "n_bottom_nodes", len(x_bottom))),
    )


def load_force_compliance(path: Path | str) -> ComplianceBlocks:
    """Load nodal-force compliance blocks from a compressed NPZ file."""
    data = np.load(path)
    required = ("Ctt_force", "Ctb_force", "Cbt_force", "Cbb_force", "x_top", "x_bottom", "L", "H", "E", "nu")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Compliance NPZ missing keys: {missing}")
    reciprocity = float(data["reciprocity_error"]) if "reciprocity_error" in data.files else 0.0
    geometry_type = _decode_flag(data["geometry_type"]) if "geometry_type" in data.files else "rectangle"
    dof_ordering = _decode_flag(data["dof_ordering"]) if "dof_ordering" in data.files else ""
    x_top = np.asarray(data["x_top"], dtype=float)
    x_bottom = np.asarray(data["x_bottom"], dtype=float)
    Ctt = np.asarray(data["Ctt_force"], dtype=float)
    _require_vector_compliance_blocks(Ctt, x_top, x_bottom, dof_ordering)
    return ComplianceBlocks(
        Ctt=Ctt,
        Ctb=np.asarray(data["Ctb_force"], dtype=float),
        Cbt=np.asarray(data["Cbt_force"], dtype=float),
        Cbb=np.asarray(data["Cbb_force"], dtype=float),
        x_top=x_top,
        x_bottom=x_bottom,
        L=float(data["L"]),
        H=float(data["H"]),
        E=float(data["E"]),
        nu=float(data["nu"]),
        reciprocity_error=reciprocity,
        nx=int(data["nx"]) if "nx" in data.files else None,
        ny=int(data["ny"]) if "ny" in data.files else None,
        order=int(data["order"]) if "order" in data.files else None,
        y_top=np.asarray(data["y_top"], dtype=float) if "y_top" in data.files else None,
        y_bottom=np.asarray(data["y_bottom"], dtype=float) if "y_bottom" in data.files else None,
        x_plate=np.asarray(data["x_plate"], dtype=float) if "x_plate" in data.files else None,
        y_plate=np.asarray(data["y_plate"], dtype=float) if "y_plate" in data.files else None,
        geometry_type=geometry_type,
        dof_ordering=dof_ordering,
        n_top_nodes=int(data["n_top_nodes"]) if "n_top_nodes" in data.files else int(len(x_top)),
        n_bottom_nodes=(
            int(data["n_bottom_nodes"]) if "n_bottom_nodes" in data.files else int(len(x_bottom))
        ),
    )


def _relative_asymmetry(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))


def prepare_compliance_blocks(
    blocks: ComplianceBlocks,
    reciprocity_tol: float = 1e-6,
) -> tuple[ComplianceBlocks, float]:
    """Verify reciprocity and symmetrize roundoff-level asymmetry."""
    Ctt = np.array(blocks.Ctt, copy=True)
    Ctb = np.array(blocks.Ctb, copy=True)
    Cbt = np.array(blocks.Cbt, copy=True)
    Cbb = np.array(blocks.Cbb, copy=True)
    _require_vector_compliance_blocks(Ctt, blocks.x_top, blocks.x_bottom, blocks.dof_ordering)
    err_tt = _relative_asymmetry(Ctt)
    err_bb = _relative_asymmetry(Cbb)
    err_bt = float(np.linalg.norm(Cbt - Ctb.T) / max(np.linalg.norm(Cbt), 1e-30))
    reported = max(err_tt, err_bb, err_bt, blocks.reciprocity_error)
    if reported > reciprocity_tol:
        raise ValueError(
            f"Compliance reciprocity error {reported:.3e} exceeds tolerance {reciprocity_tol:.3e}. "
            "Re-run FEM or pass a larger --reciprocity-tol if the residual is still a solver artifact."
        )
    n_t = Ctt.shape[0]
    full = np.block([[Ctt, Ctb], [Cbt, Cbb]])
    full = 0.5 * (full + full.T)
    prepared = ComplianceBlocks(
        Ctt=full[:n_t, :n_t],
        Ctb=full[:n_t, n_t:],
        Cbt=full[n_t:, :n_t],
        Cbb=full[n_t:, n_t:],
        x_top=blocks.x_top,
        x_bottom=blocks.x_bottom,
        L=blocks.L,
        H=blocks.H,
        E=blocks.E,
        nu=blocks.nu,
        reciprocity_error=reported,
        nx=blocks.nx,
        ny=blocks.ny,
        order=blocks.order,
        y_top=blocks.y_top,
        y_bottom=blocks.y_bottom,
        x_plate=blocks.x_plate,
        y_plate=blocks.y_plate,
        geometry_type=blocks.geometry_type,
        dof_ordering=blocks.dof_ordering,
        n_top_nodes=blocks.n_top_nodes,
        n_bottom_nodes=blocks.n_bottom_nodes,
    )
    return prepared, reported


def restricted_blocks(
    blocks: ComplianceBlocks,
    free: np.ndarray,
    contact: np.ndarray,
) -> dict[str, np.ndarray]:
    """Restricted compliance blocks for component-major free/contact partitions."""
    n_b = len(blocks.x_bottom)
    n_t_vec = blocks.Ctt.shape[0]
    contact_dofs = component_major_indices(contact, n_b)
    free_dofs = component_major_indices(free, n_b)
    n_c_vec = int(contact_dofs.size)
    n_f_vec = int(free_dofs.size)
    return {
        "C_tc": blocks.Ctb[:, contact_dofs],
        "C_ct": blocks.Cbt[contact_dofs, :],
        "C_cc": blocks.Cbb[np.ix_(contact_dofs, contact_dofs)],
        "C_ft": blocks.Cbt[free_dofs, :] if n_f_vec else np.zeros((0, n_t_vec)),
        "C_fc": blocks.Cbb[np.ix_(free_dofs, contact_dofs)] if n_f_vec else np.zeros((0, n_c_vec)),
    }


def build_boundary_matrix(
    blocks: ComplianceBlocks,
    free: np.ndarray,
    contact: np.ndarray,
    x_r: float,
    y_r: float = SOLVER_YR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Assemble the boundary-compliance matrix with three rigid columns for one interval."""
    rb = restricted_blocks(blocks, free, contact)
    y_top, y_bottom = _boundary_y(blocks)
    R_t = vector_rigid_mode_matrix(blocks.x_top, y_top, x_r, y_r)
    R_c = (
        vector_rigid_mode_matrix(blocks.x_bottom[contact], y_bottom[contact], x_r, y_r)
        if contact.size
        else np.zeros((0, 3), dtype=float)
    )
    R_f = (
        vector_rigid_mode_matrix(blocks.x_bottom[free], y_bottom[free], x_r, y_r)
        if free.size
        else np.zeros((0, 3), dtype=float)
    )
    n_t_vec = blocks.Ctt.shape[0]
    n_c_vec = int(R_c.shape[0])
    A = np.zeros((n_t_vec + n_c_vec + 3, n_t_vec + n_c_vec + 3), dtype=float)
    A[:n_t_vec, :n_t_vec] = blocks.Ctt
    A[:n_t_vec, n_t_vec : n_t_vec + n_c_vec] = rb["C_tc"]
    A[:n_t_vec, n_t_vec + n_c_vec :] = R_t
    A[n_t_vec : n_t_vec + n_c_vec, :n_t_vec] = rb["C_ct"]
    A[n_t_vec : n_t_vec + n_c_vec, n_t_vec : n_t_vec + n_c_vec] = rb["C_cc"]
    A[n_t_vec : n_t_vec + n_c_vec, n_t_vec + n_c_vec :] = R_c
    A[n_t_vec + n_c_vec :, :n_t_vec] = R_t.T
    A[n_t_vec + n_c_vec :, n_t_vec : n_t_vec + n_c_vec] = R_c.T
    rb["R_c"] = R_c
    return A, R_t, R_f, rb


def kf_from_scalars(scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Translation-to-force matrix K_F from a (6 or 5, n_fields) scalar table.

    ``K_F = [[F_x,Bx, F_x,By], [F_y,Bx, F_y,By]]`` maps the anchor translation
    ``(d_ax, d_ay)`` to the local top resultant. Returns ``(K_F, svals, cond, det)``.
    """
    S = np.asarray(scalars, dtype=float)
    bx, by = (COL_BX, COL_BY) if S.shape[0] == N_AFFINE_COLUMNS else (MODE_BX, MODE_BY)
    kf = np.array(
        [[S[bx, SCALAR_FX], S[by, SCALAR_FX]], [S[bx, SCALAR_FY], S[by, SCALAR_FY]]],
        dtype=float,
    )
    if not np.all(np.isfinite(kf)):
        return kf, np.full(2, np.nan), float("inf"), float("nan")
    det = float(np.linalg.det(kf))
    svals = np.linalg.svd(kf, compute_uv=False).astype(float)
    cond = float(svals[0] / svals[-1]) if svals[-1] > 0.0 else float("inf")
    return kf, svals, cond, det


def _as_affine_top(W: np.ndarray, n_t_vec: int) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if W.shape == (n_t_vec, N_AFFINE_COLUMNS):
        return W
    if W.shape == (n_t_vec, N_BASIS_MODES):
        return np.concatenate([np.zeros((n_t_vec, 1)), W], axis=1)
    raise ValueError(
        f"W has shape {W.shape}; expected ({n_t_vec}, {N_AFFINE_COLUMNS}) affine top matrix."
    )


def solve_candidate(
    blocks: ComplianceBlocks,
    spec: ContactInterval,
    W: np.ndarray,
    x_r: float,
    a: float,
    y_r: float = SOLVER_YR,
) -> dict:
    """Solve the six-column affine boundary system for one contact interval.

    Only the free/contact partition, the interval anchor, and the curved-sole
    closure depend on the interval; the full-domain compliance blocks are never
    rebuilt. The matrix is factored once and all six right-hand sides are solved
    against that factorization. Numerical rank is decided from the reciprocal
    condition estimate of the actual matrix and confirmed by an SVD.
    """
    if not isinstance(spec, ContactInterval):
        raise TypeError("solve_candidate expects a ContactInterval.")
    n_b = len(blocks.x_bottom)
    n_t = len(blocks.x_top)
    if spec.n_bottom != n_b:
        raise ValueError(f"Interval built for N_b={spec.n_bottom}; compliance has N_b={n_b}.")
    y_top, y_bottom = _boundary_y(blocks)
    free, contact = spec.sets()
    n_f, n_c = int(free.size), int(contact.size)
    n_t_vec = 2 * n_t
    n_c_vec = 2 * n_c
    W6 = _as_affine_top(W, n_t_vec)
    n_cols = N_AFFINE_COLUMNS

    x_a = interval_anchor_x(blocks.x_bottom, spec.start, spec.end)
    y_a = interval_anchor_y(blocks.x_bottom, y_bottom, x_a)
    W_c = build_contact_affine_matrix(blocks.x_bottom[contact], x_a, y_bottom[contact], y_a)

    A, R_t, R_f, rb = build_boundary_matrix(blocks, free, contact, x_r, y_r=y_r)
    rhs = np.zeros((A.shape[0], n_cols), dtype=float)
    rhs[:n_t_vec, :] = W6
    rhs[n_t_vec : n_t_vec + n_c_vec, :] = W_c

    out: dict = {
        "interval": spec,
        "contact_type": spec.topology_label,
        "x_anchor": x_a,
        "y_anchor": y_a,
        "free": free,
        "contact": contact,
        "n_free": n_f,
        "n_contact": n_c,
        "W_c": W_c,
        "A": A,
        "matrix_size": int(A.shape[0]),
        "valid": True,
        "rejection_reason": "",
    }

    anorm = float(np.linalg.norm(A, 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", linalg.LinAlgWarning)
            lu, piv = linalg.lu_factor(A, check_finite=True)
        rcond, info = lapack.dgecon(lu, anorm, norm="1")
        rcond = float(rcond) if info == 0 else 0.0
    except (linalg.LinAlgError, ValueError) as exc:
        out.update(valid=False, rejection_reason=f"{REJECT_SOLVE_FAILED}: {exc}", condition=np.inf)
        out["rank"] = int(np.linalg.matrix_rank(A))
        return out
    out["condition"] = float(1.0 / rcond) if rcond > 0.0 else float("inf")
    if not rcond > RANK_RCOND_TOL:
        rank = int(np.linalg.matrix_rank(A))
        out["rank"] = rank
        if rank < A.shape[0] or rcond == 0.0:
            out.update(
                valid=False,
                rejection_reason=f"{REJECT_RANK_DEFICIENT}: rank {rank} < {A.shape[0]}",
            )
            return out
    else:
        out["rank"] = int(A.shape[0])

    X = linalg.lu_solve((lu, piv), rhs, check_finite=True)
    F_t = X[:n_t_vec, :]
    F_c = X[n_t_vec : n_t_vec + n_c_vec, :]
    Alpha = X[n_t_vec + n_c_vec :, :]
    G_f = rb["C_ft"] @ F_t + rb["C_fc"] @ F_c + R_f @ Alpha if n_f else np.zeros((0, n_cols))

    f_tx, f_ty = split_uv(F_t, n_t)
    f_cx, f_cy = split_uv(F_c, n_c)
    u_f, v_f = split_uv(G_f, n_f)
    u_c, v_c = split_uv(W_c, n_c)

    bottom_u = np.zeros((n_cols, n_b), dtype=float)
    bottom_v = np.zeros((n_cols, n_b), dtype=float)
    reaction_x = np.zeros((n_cols, n_b), dtype=float)
    reaction_y = np.zeros((n_cols, n_b), dtype=float)
    bottom_u[:, free] = u_f.T
    bottom_v[:, free] = v_f.T
    bottom_u[:, contact] = u_c.T
    bottom_v[:, contact] = v_c.T
    reaction_x[:, contact] = f_cx.T
    reaction_y[:, contact] = f_cy.T

    T_toe, T_vert, _ = compute_toe_moment(f_tx, f_ty, blocks.x_top, y_top, a)
    scalars = np.column_stack(
        [
            f_tx.sum(axis=0),
            f_ty.sum(axis=0),
            blocks.x_top @ f_ty,
            blocks.x_top @ f_ty - y_top @ f_tx,
            np.atleast_1d(T_toe),
            np.atleast_1d(T_vert),
        ]
    )
    x_c = blocks.x_bottom[contact]
    y_c = y_bottom[contact]
    contact_total = np.column_stack(
        [f_cx.sum(axis=0), f_cy.sum(axis=0), x_c @ f_cy - y_c @ f_cx]
    )

    nan = np.full(n_cols, np.nan)
    heel_adj = spec.heel_adjacent_free_node_id
    toe_adj = spec.toe_adjacent_free_node_id
    edge = {
        "heel_edge_reaction_local_x": f_cx[0].copy(),
        "heel_edge_reaction_local_y": f_cy[0].copy(),
        "toe_edge_reaction_local_x": f_cx[-1].copy(),
        "toe_edge_reaction_local_y": f_cy[-1].copy(),
        "heel_adjacent_free_u": bottom_u[:, heel_adj].copy() if heel_adj >= 0 else nan.copy(),
        "heel_adjacent_free_v": bottom_v[:, heel_adj].copy() if heel_adj >= 0 else nan.copy(),
        "toe_adjacent_free_u": bottom_u[:, toe_adj].copy() if toe_adj >= 0 else nan.copy(),
        "toe_adjacent_free_v": bottom_v[:, toe_adj].copy() if toe_adj >= 0 else nan.copy(),
    }

    kf, kf_svals, kf_cond, kf_det = kf_from_scalars(scalars)

    residual_matrix = A @ X - rhs
    solve_residual = float(np.linalg.norm(residual_matrix) / max(np.linalg.norm(rhs), 1e-30))
    top_disp = blocks.Ctt @ F_t + rb["C_tc"] @ F_c + R_t @ Alpha
    top_res = float(np.linalg.norm(top_disp - W6) / max(np.linalg.norm(W6), 1e-30))
    contact_disp = rb["C_ct"] @ F_t + rb["C_cc"] @ F_c + rb["R_c"] @ Alpha
    contact_res = float(
        np.linalg.norm(contact_disp - W_c) / max(1.0, np.linalg.norm(W6), np.linalg.norm(W_c))
    )
    eq = R_t.T @ F_t + rb["R_c"].T @ F_c
    force_res = float(np.linalg.norm(eq[:2, :]))
    moment_res = float(np.linalg.norm(eq[2, :]))
    force_scale = max(float(np.linalg.norm(F_t)), float(np.linalg.norm(F_c)), 1e-30)
    lever_scale = max(float(np.max(np.abs(blocks.x_bottom - x_r))), 1e-30)
    equilibrium_rel = max(force_res, moment_res / lever_scale) / force_scale
    top_total = scalars[:, [SCALAR_FX, SCALAR_FY]]
    balance = float(
        np.max(np.abs(top_total + contact_total[:, :2])) / max(float(np.max(np.abs(top_total))), 1e-30)
    )

    out.update(
        F_t=F_t,
        F_c=F_c,
        Alpha=Alpha,
        G_f=G_f,
        bottom_u=bottom_u,
        bottom_v=bottom_v,
        reaction_x=reaction_x,
        reaction_y=reaction_y,
        top_force_x=f_tx.T.copy(),
        top_force_y=f_ty.T.copy(),
        scalars=scalars,
        contact_total=contact_total,
        edge=edge,
        kf_matrix=kf,
        kf_svals=kf_svals,
        kf_cond=kf_cond,
        kf_det=kf_det,
        solve_residual=solve_residual,
        top_residual=top_res,
        contact_residual=contact_res,
        force_residual=force_res,
        moment_residual=moment_res,
        equilibrium_rel=equilibrium_rel,
        balance_residual=balance,
    )
    finite = all(
        np.all(np.isfinite(v)) for v in (F_t, F_c, Alpha, G_f, scalars, contact_total)
    )
    if not finite:
        out.update(valid=False, rejection_reason=REJECT_NONFINITE)
    elif solve_residual > SOLVE_RESIDUAL_TOL:
        out.update(valid=False, rejection_reason=f"{REJECT_SOLVE_RESIDUAL}: {solve_residual:.3e}")
    elif max(top_res, contact_res) > SOLVE_RESIDUAL_TOL:
        out.update(
            valid=False,
            rejection_reason=f"{REJECT_BC_RESIDUAL}: {max(top_res, contact_res):.3e}",
        )
    elif equilibrium_rel > EQUILIBRIUM_REL_TOL:
        out.update(valid=False, rejection_reason=f"{REJECT_EQUILIBRIUM}: {equilibrium_rel:.3e}")
    return out


def _row_fields_from_solution(sol: dict) -> dict[str, np.ndarray]:
    return {
        "bottom_u": sol["bottom_u"],
        "bottom_v": sol["bottom_v"],
        "reaction_x": sol["reaction_x"],
        "reaction_y": sol["reaction_y"],
        "top_force_x": sol["top_force_x"],
        "top_force_y": sol["top_force_y"],
    }


def _plate_fields(plate) -> dict[str, np.ndarray]:
    return {
        "plate_u_local": plate.u_local,
        "plate_v_local": plate.v_local,
        "plate_rotation_local": plate.rotation_local,
        "plate_constraint_multiplier": plate.constraint_multiplier,
        "plate_axial_force": plate.axial_force,
    }


class RecordFieldSolver:
    """Re-solves one interval record for its nodal fields (thread-safe LRU cache).

    Holds the prepared compliance blocks, the affine top matrix and (for plated
    models) the plate influence matrices, i.e. exactly what generation used, so
    the fields equal the ones a ``store_fields=True`` lookup would have saved.
    """

    def __init__(
        self,
        blocks: ComplianceBlocks,
        W: np.ndarray,
        x_r: float,
        y_r: float,
        a: float,
        plate_influence: PlateInfluence | None = None,
        cache_rows: int = FIELD_CACHE_ROWS,
    ) -> None:
        self.blocks = blocks
        self.W = np.asarray(W, dtype=float)
        self.x_r = float(x_r)
        self.y_r = float(y_r)
        self.a = float(a)
        self.plate_influence = plate_influence
        self.cache_rows = int(cache_rows)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.n_solves = 0

    @property
    def compliance_matrix(self) -> np.ndarray:
        b = self.blocks
        return np.block([[b.Ctt, b.Ctb], [b.Cbt, b.Cbb]])

    def _solve(self, lookup: ContactLookupResult, row: int) -> dict:
        sol = solve_candidate(self.blocks, lookup.interval(row), self.W, self.x_r, a=self.a, y_r=self.y_r)
        self.n_solves += 1
        n_b, n_t = len(self.blocks.x_bottom), len(self.blocks.x_top)
        if "F_t" in sol:
            entry = _row_fields_from_solution(sol)
            entry["_solution"] = {k: sol[k] for k in ("F_t", "F_c", "Alpha", "W_c", "contact")}
        else:
            nan_b = np.full((N_AFFINE_COLUMNS, n_b), np.nan)
            nan_t = np.full((N_AFFINE_COLUMNS, n_t), np.nan)
            entry = {name: (nan_t if name.startswith("top") else nan_b).copy() for name in NODAL_FIELD_NAMES}
            entry["_solution"] = None
        return entry

    def _plate(self, lookup: ContactLookupResult, row: int, entry: dict) -> dict[str, np.ndarray]:
        n_p = int(np.asarray(lookup.plate_node_ids).size)
        n_el = int(np.asarray(lookup.plate_element_connectivity).shape[0])
        sol = entry["_solution"]
        if self.plate_influence is None:
            raise ValueError("Lookup has no plate influence matrices; plate fields cannot be recomputed.")
        if sol is None or not bool(lookup.valid_mask[row]):
            return _plate_fields(empty_plate_basis(n_p, n_el))
        plate = plate_basis_from_influence(
            self.plate_influence, sol["F_t"], sol["F_c"], np.asarray(sol["contact"], dtype=int),
            sol["Alpha"], self.W, sol["W_c"],
        )
        return _plate_fields(plate)

    def fields(self, lookup: ContactLookupResult, row: int, *, plate: bool = False) -> dict[str, np.ndarray]:
        row = int(row)
        with self._lock:
            entry = self._cache.get(row)
            if entry is not None:
                self._cache.move_to_end(row)
        if entry is None:
            entry = self._solve(lookup, row)
            with self._lock:
                self._cache[row] = entry
                while len(self._cache) > self.cache_rows:
                    self._cache.popitem(last=False)
        if plate and "plate_u_local" not in entry:
            entry.update(self._plate(lookup, row, entry))
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()


def _supports_plate_recovery(fem_result: ComplianceResult | None) -> bool:
    if fem_result is None:
        return False
    return (
        fem_result.B_p is not None
        and fem_result.factorization is not None
        and fem_result.x_plate is not None
        and fem_result.basis is not None
        and fem_result.n_primal is not None
    )


def rearfoot_geometry(
    x_top: np.ndarray,
    y_top: np.ndarray | None,
    L: float,
    a: float,
    kappa: float,
) -> RearfootGeometry:
    """Heel->MTP line of the reference top: ``a``, ``phi1(a)``, ``y_top(a) - y_top(0)``."""
    phi1_a = float(shape_mode_phi1(np.array([float(a)]), float(L), float(a), float(kappa))[0])
    if y_top is None:
        dy_a = 0.0
    else:
        dy_a = interpolate_top_height(x_top, y_top, a) - interpolate_top_height(x_top, y_top, 0.0)
    return RearfootGeometry(a=float(a), phi1_a=phi1_a, dy_a=float(dy_a))


def lookup_rearfoot_geometry(lookup: ContactLookupResult) -> RearfootGeometry:
    return rearfoot_geometry(
        np.asarray(lookup.x_top, dtype=float),
        None if lookup.y_top is None else np.asarray(lookup.y_top, dtype=float),
        float(lookup.L),
        float(lookup.softplus_a),
        float(lookup.softplus_kappa),
    )


def recompute_q_alpha_basis(lookup: ContactLookupResult) -> np.ndarray:
    """Recompute ``Q_alpha_shoe_on_foot_basis`` (-psi^T f_{t,y}) from the top forces."""
    if not lookup.has_nodal_fields:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} top_force_y_basis is missing, so the toe "
            "generalized force cannot be reconstructed."
        )
    top_fy = (
        np.asarray(lookup.top_force_y_basis, dtype=float)
        if lookup.top_force_y_basis is not None
        else lookup.record_fields(np.arange(lookup.n_records))["top_force_y"]
    )
    return q_alpha_shoe_on_foot_basis(
        top_fy,
        np.asarray(lookup.basis_top_displacements, dtype=float),
        len(lookup.x_top),
        np.asarray(lookup.x_top, dtype=float),
        lookup_rearfoot_geometry(lookup),
    )


def lookup_q_alpha_basis(lookup: ContactLookupResult) -> np.ndarray:
    """Stored ``Q_alpha_shoe_on_foot_basis`` (n_records, 6), or recompute it from top forces."""
    stored = getattr(lookup, "Q_alpha_shoe_on_foot_basis", None)
    if stored is not None:
        return np.asarray(stored, dtype=float)
    return recompute_q_alpha_basis(lookup)


def generate_contact_lookup(
    blocks: ComplianceBlocks,
    a: float,
    kappa: float,
    include_endpoints: bool = True,
    reciprocity_tol: float = 1e-6,
    fem_result: ComplianceResult | None = None,
    require_plate: bool = False,
    plate_bc_tol: float = 1e-5,
    plate_bp_tol: float = 1e-5,
    progress: Callable[[int, int], None] | None = None,
    store_fields: bool = False,
) -> ContactLookupResult:
    """Generate the phi-independent interval lookup (schema v10).

    All ``N_b (N_b + 1) / 2`` contiguous intervals are solved once each. Every
    record is factored once and solved for the six affine columns. Intervals
    whose boundary matrix is rank deficient (or whose solve fails its residual or
    equilibrium checks) are kept with ``record_valid = False`` and a structured
    rejection reason.

    By default only per-record scalars (``scalar_lookup``, edge responses, K_F,
    ``Q_alpha``, residuals) are kept; the per-node bases (bottom displacements,
    reactions, top forces, plate fields) are recomputed on demand by
    :class:`RecordFieldSolver`. ``store_fields=True`` also keeps the full
    ``(n_records, 6, n_nodes)`` arrays.

    When ``fem_result`` retains the layered factorization, plate fields are
    recovered for every record from precomputed plate influence matrices (one
    multi-RHS solve of the same augmented free-body operator).
    ``include_endpoints`` is accepted for CLI compatibility and ignored.
    """
    del include_endpoints
    t_start = time.perf_counter()
    prepared, reciprocity_error = prepare_compliance_blocks(blocks, reciprocity_tol=reciprocity_tol)
    n_b = len(prepared.x_bottom)
    n_t = len(prepared.x_top)
    intervals = enumerate_intervals(n_b)
    n_rec = len(intervals)
    x_r = 0.5 * prepared.L
    y_r = SOLVER_YR
    warn_if_a_not_on_top_grid(prepared.x_top, a)
    W = build_top_affine_matrix(prepared.x_top, prepared.L, a, kappa)
    phi_ref = reference_chord_angle(prepared.x_top, prepared.y_top)
    y_top, y_bottom = _boundary_y(prepared)

    recover_plate = _supports_plate_recovery(fem_result)
    if require_plate and not recover_plate:
        raise ValueError(
            "require_plate=True but fem_result is missing B_p, factorization, basis, "
            "or x_plate (rectangle / NPZ-only path cannot recover plate_response)."
        )
    plate_mesh = extract_plate_mesh_info(fem_result) if recover_plate else None
    influence: PlateInfluence | None = (
        build_plate_influence(fem_result, plate_mesh, x_r=x_r, y_r=y_r) if recover_plate else None
    )
    n_plate = int(plate_mesh.n_nodes) if plate_mesh is not None else 0
    n_el = int(plate_mesh.n_elements) if plate_mesh is not None else 0

    C = N_AFFINE_COLUMNS
    starts = np.array([iv.start for iv in intervals], dtype=int)
    ends = np.array([iv.end for iv in intervals], dtype=int)
    type_codes = labels_from_arrays(starts, ends, n_b)
    contact_mask = np.zeros((n_rec, n_b), dtype=bool)
    nanf = np.nan
    scalar_lookup = np.full((n_rec, C, N_SCALAR_FIELDS), nanf)
    contact_force_total = np.full((n_rec, C, 3), nanf)
    node_fields = (
        {name: np.full((n_rec, C, n_t if name.startswith("top") else n_b), nanf) for name in NODAL_FIELD_NAMES}
        if store_fields
        else None
    )
    q_alpha = np.full((n_rec, C), nanf)
    geom = rearfoot_geometry(prepared.x_top, prepared.y_top, prepared.L, a, kappa)
    edge_responses = {name: np.full((n_rec, C), nanf) for name in EDGE_RESPONSE_FIELDS}
    kf_matrix = np.full((n_rec, 2, 2), nanf)
    kf_svals = np.full((n_rec, 2), nanf)
    kf_cond = np.full(n_rec, np.inf)
    kf_det = np.full(n_rec, nanf)
    cond_est = np.full(n_rec, np.inf)
    solve_res = np.full(n_rec, nanf)
    top_res = np.full(n_rec, nanf)
    contact_res = np.full(n_rec, nanf)
    force_res = np.full(n_rec, nanf)
    moment_res = np.full(n_rec, nanf)
    balance_res = np.full(n_rec, nanf)
    rank = np.zeros(n_rec, dtype=int)
    size = np.zeros(n_rec, dtype=int)
    valid = np.zeros(n_rec, dtype=bool)
    reasons: list[str] = [""] * n_rec
    status: list[str] = [""] * n_rec
    anchor_x = np.zeros(n_rec)
    anchor_y = np.zeros(n_rec)
    store_plate = recover_plate and store_fields
    plate_u = np.full((n_rec, C, n_plate), nanf) if store_plate else None
    plate_v = np.full((n_rec, C, n_plate), nanf) if store_plate else None
    plate_th = np.full((n_rec, C, n_plate), nanf) if store_plate else None
    plate_lam = np.full((n_rec, C, n_el), nanf) if store_plate else None
    plate_ax = np.full((n_rec, C, n_el), nanf) if store_plate else None
    plate_bp = np.full((n_rec, C), nanf) if recover_plate else None
    rigid_alpha = np.full((n_rec, C, 3), nanf)

    report_every = max(1, n_rec // 20)
    for row, iv in enumerate(intervals):
        sol = solve_candidate(prepared, iv, W, x_r, a=a, y_r=y_r)
        contact_mask[row, iv.start : iv.end + 1] = True
        anchor_x[row] = float(sol["x_anchor"])
        anchor_y[row] = float(sol["y_anchor"])
        rank[row] = int(sol.get("rank", 0))
        size[row] = int(sol["matrix_size"])
        cond_est[row] = float(sol.get("condition", np.inf))
        if "F_t" in sol:
            scalar_lookup[row] = sol["scalars"]
            contact_force_total[row] = sol["contact_total"]
            if node_fields is not None:
                for name, value in _row_fields_from_solution(sol).items():
                    node_fields[name][row] = value
            for name in EDGE_RESPONSE_FIELDS:
                edge_responses[name][row] = sol["edge"][name]
            kf_matrix[row] = sol["kf_matrix"]
            kf_svals[row] = np.asarray(sol["kf_svals"], dtype=float)[:2]
            kf_cond[row] = float(sol["kf_cond"])
            kf_det[row] = float(sol["kf_det"])
            solve_res[row] = float(sol["solve_residual"])
            top_res[row] = float(sol["top_residual"])
            contact_res[row] = float(sol["contact_residual"])
            force_res[row] = float(sol["force_residual"])
            moment_res[row] = float(sol["moment_residual"])
            balance_res[row] = float(sol["balance_residual"])
            rigid_alpha[row] = np.asarray(sol["Alpha"], dtype=float).T
        ok = bool(sol["valid"])
        reason = str(sol["rejection_reason"])
        if ok and influence is not None:
            assert plate_mesh is not None
            plate = plate_basis_from_influence(
                influence,
                np.asarray(sol["F_t"]),
                np.asarray(sol["F_c"]),
                np.asarray(sol["contact"], dtype=int),
                np.asarray(sol["Alpha"]),
                W,
                np.asarray(sol["W_c"]),
            )
            bp_max = float(np.max(plate.bp_residual)) if plate.bp_residual.size else 0.0
            bc_max = max(
                float(np.max(plate.top_bc_residual)) if plate.top_bc_residual.size else 0.0,
                float(np.max(plate.contact_bc_residual)) if plate.contact_bc_residual.size else 0.0,
            )
            if bp_max > plate_bp_tol or bc_max > plate_bc_tol:
                ok = False
                reason = f"{REJECT_PLATE}: bp={bp_max:.3e}, bc={bc_max:.3e}"
            else:
                if store_plate:
                    plate_u[row] = plate.u_local
                    plate_v[row] = plate.v_local
                    plate_th[row] = plate.rotation_local
                    plate_lam[row] = plate.constraint_multiplier
                    plate_ax[row] = plate.axial_force
                plate_bp[row] = plate.bp_residual
        if ok:
            q_alpha[row] = q_alpha_shoe_on_foot_basis(sol["top_force_y"], W, n_t, prepared.x_top, geom)
        valid[row] = ok
        reasons[row] = "" if ok else reason
        status[row] = ("single_contact_node" if iv.is_single_node else "ok") if ok else "rejected"
        if progress is not None and ((row + 1) % report_every == 0 or row + 1 == n_rec):
            progress(row + 1, n_rec)

    counts = record_counts(n_b)
    rejected = Counter(r.split(":")[0] for r, ok in zip(reasons, valid) if not ok)
    build_time = time.perf_counter() - t_start

    measured_render: dict = {}
    geometry_metadata: dict = {}
    if is_measured_result(fem_result):
        measured_render = build_measured_render_section(fem_result, x_r=x_r, y_r=y_r)
        geometry_metadata = dict(fem_result.geometry_metadata)

    plate_metadata: dict = {"has_plate_response": False}
    if recover_plate and plate_mesh is not None:
        plate_metadata = {
            "has_plate_response": True,
            "plate_model": PLATE_MODEL,
            "EI_plate": float(plate_mesh.EI_plate),
            "plate_axial_model": PLATE_AXIAL_MODEL,
            "plate_rotation_sign": PLATE_ROTATION_SIGN,
            "plate_normal_sign": PLATE_NORMAL_SIGN,
            "constraint_row_normalization": CONSTRAINT_ROW_NORMALIZATION,
            "coordinate_frame": COORDINATE_FRAME,
            "n_plate_nodes": n_plate,
            "n_plate_elements": n_el,
            "plate_recovery": "precomputed influence matrices (one multi-RHS solve)",
        }

    metadata = {
        **interval_metadata(),
        "moment_reference": "Mz about (0,0); Mv = x_t^T f_{t,y}; xr=L/2, yr=0 inside solver",
        "toe_moment": "rho_a^T f_{t,y} - eta_a^T f_{t,x} about (a, H_a)",
        "toe_generalized_force": TOE_GENERALIZED_FORCE_DEFINITION,
        "toe_generalized_force_units": "N*m per metre width (alpha dimensionless)",
        "toe_generalized_force_source": "generated",
        "dof_ordering": DOF_ORDERING_COMPONENT_MAJOR_UV,
        "nx": prepared.nx,
        "ny": prepared.ny,
        "order": prepared.order,
        "geometry_type": prepared.geometry_type,
        "phi_ref": float(phi_ref),
        "n_bottom_nodes": n_b,
        "n_intervals_theoretical": int(counts["total"]),
        "n_intervals_valid": int(np.count_nonzero(valid)),
        "n_intervals_rejected": int(n_rec - np.count_nonzero(valid)),
        "rejection_counts": dict(sorted(rejected.items())),
        "expected_label_counts": counts,
        "build_time_s": float(build_time),
        "curved_sole": bool(np.any(np.abs(y_bottom) > 0.0)),
        "max_bottom_height": float(np.max(y_bottom)) if y_bottom.size else 0.0,
        "nodal_fields_stored": bool(store_fields),
        **plate_metadata,
    }
    nf = node_fields or {}

    return ContactLookupResult(
        contact_start_index=starts,
        contact_end_index=ends,
        n_free=n_b - (ends - starts + 1),
        n_contact=ends - starts + 1,
        status=status,
        x_top=prepared.x_top.copy(),
        x_bottom=prepared.x_bottom.copy(),
        basis_top_displacements=W,
        scalar_lookup=scalar_lookup,
        softplus_a=float(a),
        softplus_kappa=float(kappa),
        L=prepared.L,
        H=prepared.H,
        E=prepared.E,
        nu=prepared.nu,
        condition_estimates=cond_est,
        solve_residuals=solve_res,
        top_displacement_residuals=top_res,
        contact_displacement_residuals=contact_res,
        force_equilibrium_residuals=force_res,
        moment_equilibrium_residuals=moment_res,
        reciprocity_error=reciprocity_error,
        record_valid=valid,
        rejection_reason=reasons,
        boundary_matrix_rank=rank,
        boundary_matrix_size=size,
        contact_type_codes=type_codes,
        contact_start_x=prepared.x_bottom[starts].copy(),
        contact_end_x=prepared.x_bottom[ends].copy(),
        contact_anchor_reference_x=anchor_x,
        contact_anchor_reference_y=anchor_y,
        heel_contact_edge_node_id=starts.copy(),
        toe_contact_edge_node_id=ends.copy(),
        heel_adjacent_free_node_id=np.where(starts > 0, starts - 1, ABSENT_NODE_ID),
        toe_adjacent_free_node_id=np.where(ends < n_b - 1, ends + 1, ABSENT_NODE_ID),
        contact_mask=contact_mask,
        bottom_u_basis=nf.get("bottom_u"),
        bottom_v_basis=nf.get("bottom_v"),
        reaction_x_basis=nf.get("reaction_x"),
        reaction_y_basis=nf.get("reaction_y"),
        top_force_x_basis=nf.get("top_force_x"),
        top_force_y_basis=nf.get("top_force_y"),
        contact_force_total=contact_force_total,
        balance_residuals=balance_res,
        edge_responses=edge_responses,
        Q_alpha_shoe_on_foot_basis=q_alpha,
        toe_generalized_force_source="generated",
        kf_matrix=kf_matrix,
        kf_svals=kf_svals,
        kf_cond=kf_cond,
        kf_det=kf_det,
        y_top=None if prepared.y_top is None else np.asarray(prepared.y_top, dtype=float),
        y_bottom=np.asarray(y_bottom, dtype=float),
        x_plate=prepared.x_plate,
        y_plate=prepared.y_plate,
        geometry_type=prepared.geometry_type,
        dof_ordering=prepared.dof_ordering,
        n_top_nodes=n_t,
        n_bottom_nodes=n_b,
        moment_origin=MOMENT_ORIGIN,
        solver_xr=float(x_r),
        solver_yr=float(y_r),
        phi_ref=float(phi_ref),
        build_time_s=float(build_time),
        metadata=metadata,
        schema_version=LOOKUP_SCHEMA_VERSION,
        has_plate_response=bool(recover_plate),
        plate_node_ids=None if plate_mesh is None else np.asarray(plate_mesh.node_ids, dtype=int),
        plate_reference_x=None if plate_mesh is None else plate_mesh.reference_x.copy(),
        plate_reference_y=None if plate_mesh is None else plate_mesh.reference_y.copy(),
        plate_reference_arc_length=(
            None if plate_mesh is None else plate_mesh.reference_arc_length.copy()
        ),
        plate_element_connectivity=(
            None if plate_mesh is None else plate_mesh.element_connectivity.copy()
        ),
        plate_element_lengths=None if plate_mesh is None else plate_mesh.element_lengths.copy(),
        plate_element_tangents=None if plate_mesh is None else plate_mesh.element_tangents.copy(),
        plate_element_normals=None if plate_mesh is None else plate_mesh.element_normals.copy(),
        plate_u_dof_ids=None if plate_mesh is None else np.asarray(plate_mesh.u_dof_ids, dtype=int),
        plate_v_dof_ids=None if plate_mesh is None else np.asarray(plate_mesh.v_dof_ids, dtype=int),
        plate_rotation_dof_ids=(
            None if plate_mesh is None else np.asarray(plate_mesh.rotation_dof_ids, dtype=int)
        ),
        plate_constraint_ids=(
            None if plate_mesh is None else np.asarray(plate_mesh.constraint_ids, dtype=int)
        ),
        plate_tangent=None if plate_mesh is None else plate_mesh.plate_tangent.copy(),
        plate_normal=None if plate_mesh is None else plate_mesh.plate_normal.copy(),
        EI_plate=float("nan") if plate_mesh is None else float(plate_mesh.EI_plate),
        plate_u_local_basis=plate_u,
        plate_v_local_basis=plate_v,
        plate_rotation_local_basis=plate_th,
        plate_constraint_multiplier_basis=plate_lam,
        plate_axial_force_basis=plate_ax,
        plate_bp_residual=plate_bp,
        plate_metadata=plate_metadata,
        rigid_alpha_basis=rigid_alpha,
        measured_render=measured_render,
        geometry_metadata=geometry_metadata,
        field_solver=RecordFieldSolver(prepared, W, x_r, y_r, a, plate_influence=influence),
    )


def interval_metadata() -> dict:
    """Schema-level conventions written to NPZ / JSON and checked on load."""
    return {
        "schema_version": LOOKUP_SCHEMA_VERSION,
        "contact_set_model": CONTACT_SET_MODEL,
        "ground_geometry": GROUND_GEOMETRY,
        "contact_law": CONTACT_LAW,
        "contact_anchor_definition": CONTACT_ANCHOR_DEFINITION,
        "contact_anchor": (
            "x_a = (x_i + x_j)/2, y_a interpolated from the reference bottom profile; "
            "a numerical anchor, not a physical contact edge (edges are l_h = x_i, l_t = x_j)"
        ),
        "curved_sole_closure_convention": CURVED_SOLE_CLOSURE_CONVENTION,
        "affine_columns": list(AFFINE_COLUMN_NAMES),
        "basis_order": list(BASIS_ORDER),
        "force_sign_convention": FORCE_SIGN_CONVENTION,
        "gap_sign_convention": GAP_SIGN_CONVENTION,
        "normal_reaction_sign_convention": NORMAL_REACTION_SIGN_CONVENTION,
        "absent_node_id": ABSENT_NODE_ID,
        "absent_value": "NaN for adjacent-free responses beyond a domain endpoint",
        "one_node_interval": (
            "heel and toe contact edge are the same node; the heel/toe edge reaction "
            "fields both report that single nodal force (never summed twice)"
        ),
        "runtime_coefficients": (
            "gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)]; affine "
            "contraction z = z_closure + sum_k gamma_k z_k"
        ),
        "kf_matrix": "[[F_x,dx, F_x,dy], [F_y,dx, F_y,dy]] maps (d_ax, d_ay) to F^T",
        "n_shape_modes": 1,
        "shape_mode_definition": SHAPE_MODE_DEFINITION,
        "shape_mode_normalization": SHAPE_MODE_NORMALIZATION,
        "shape_mode_sign": float(SHAPE_MODE_SIGN),
        "force_frame": "rotating_local",
        "contact_type_codes": {t.value: c for t, c in CONTACT_TYPE_CODES.items()},
        "strain_model": (
            "Intentional infinitesimal strain and linear elasticity with exact "
            "finite-rotation boundary displacement (true sin/cos of varphi)."
        ),
    }


_EDGE_NPZ_KEYS = EDGE_RESPONSE_FIELDS


def save_contact_lookup(result: ContactLookupResult, output_dir: Path | str) -> Path:
    """Save NPZ, CSV, and JSON artifacts for an interval lookup table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "contact_lookup.npz"
    n_b = int(result.n_bottom_nodes)
    stored = result.nodal_fields_stored
    if not stored and result.field_solver is None:
        raise ValueError("Lookup has neither stored nodal fields nor a field solver; nothing to save them from.")
    affine_constant = np.stack(
        [result.scalar_lookup[:, 0, SCALAR_FX], result.scalar_lookup[:, 0, SCALAR_FY]], axis=1
    )
    meta = interval_metadata()

    payload = {
        "schema_version": int(LOOKUP_SCHEMA_VERSION),
        "contact_set_model": np.asarray(CONTACT_SET_MODEL),
        "ground_geometry": np.asarray(GROUND_GEOMETRY),
        "contact_law": np.asarray(CONTACT_LAW),
        "contact_anchor_definition": np.asarray(CONTACT_ANCHOR_DEFINITION),
        "curved_sole_closure_convention": np.asarray(CURVED_SOLE_CLOSURE_CONVENTION),
        "force_sign_convention": np.asarray(FORCE_SIGN_CONVENTION),
        "gap_sign_convention": np.asarray(GAP_SIGN_CONVENTION),
        "normal_reaction_sign_convention": np.asarray(NORMAL_REACTION_SIGN_CONVENTION),
        "basis_order": np.asarray(BASIS_ORDER, dtype=object),
        "affine_column_names": np.asarray(AFFINE_COLUMN_NAMES, dtype=object),
        "scalar_lookup_fields": np.asarray(SCALAR_LOOKUP_FIELDS, dtype=object),
        "contact_start_index": np.asarray(result.contact_start_index, dtype=int),
        "contact_end_index": np.asarray(result.contact_end_index, dtype=int),
        "contact_type_codes": np.asarray(result.contact_type_codes, dtype=int),
        "contact_start_x": np.asarray(result.contact_start_x, dtype=float),
        "contact_end_x": np.asarray(result.contact_end_x, dtype=float),
        "contact_anchor_reference_x": np.asarray(result.contact_anchor_reference_x, dtype=float),
        "contact_anchor_reference_y": np.asarray(result.contact_anchor_reference_y, dtype=float),
        "heel_contact_edge_node_id": np.asarray(result.heel_contact_edge_node_id, dtype=int),
        "toe_contact_edge_node_id": np.asarray(result.toe_contact_edge_node_id, dtype=int),
        "heel_adjacent_free_node_id": np.asarray(result.heel_adjacent_free_node_id, dtype=int),
        "toe_adjacent_free_node_id": np.asarray(result.toe_adjacent_free_node_id, dtype=int),
        "contact_mask": np.asarray(result.contact_mask, dtype=bool),
        "n_free": np.asarray(result.n_free, dtype=int),
        "n_contact": np.asarray(result.n_contact, dtype=int),
        "status": np.asarray(result.status, dtype=object),
        "record_valid": np.asarray(result.valid_mask, dtype=bool),
        "rejection_reason": np.asarray(result.rejection_reason, dtype=object),
        "boundary_matrix_rank": np.asarray(result.boundary_matrix_rank, dtype=int),
        "boundary_matrix_size": np.asarray(result.boundary_matrix_size, dtype=int),
        "x_top": result.x_top,
        "x_bottom": result.x_bottom,
        "y_bottom": np.asarray(
            result.y_bottom if result.y_bottom is not None else np.zeros(n_b), dtype=float
        ),
        "basis_top_displacements": result.basis_top_displacements,
        "scalar_lookup": result.scalar_lookup,
        "basis_response": result.scalar_lookup[:, 1:, :],
        "affine_constant_response": result.scalar_lookup[:, 0, :],
        "affine_constant_force": affine_constant,
        "contact_force_total": result.contact_force_total,
        "balance_residuals": result.balance_residuals,
        "nodal_fields_stored": bool(stored),
        "Q_alpha_shoe_on_foot_basis": np.asarray(lookup_q_alpha_basis(result), dtype=float),
        "kf_matrix": result.kf_matrix,
        "kf_svals": result.kf_svals,
        "kf_cond": result.kf_cond,
        "kf_det": result.kf_det,
        "softplus_a": result.softplus_a,
        "softplus_kappa": result.softplus_kappa,
        "L": result.L,
        "H": result.H,
        "E": result.E,
        "nu": result.nu,
        "condition_estimates": result.condition_estimates,
        "solve_residuals": result.solve_residuals,
        "top_displacement_residuals": result.top_displacement_residuals,
        "contact_displacement_residuals": result.contact_displacement_residuals,
        "force_equilibrium_residuals": result.force_equilibrium_residuals,
        "moment_equilibrium_residuals": result.moment_equilibrium_residuals,
        "reciprocity_error": result.reciprocity_error,
        "geometry_type": np.asarray(result.geometry_type),
        "dof_ordering": np.asarray(result.dof_ordering),
        "n_top_nodes": result.n_top_nodes,
        "n_bottom_nodes": result.n_bottom_nodes,
        "moment_origin": np.asarray(result.moment_origin, dtype=float),
        "solver_xr": result.solver_xr,
        "solver_yr": result.solver_yr,
        "phi_ref": result.phi_ref,
        "build_time_s": float(result.build_time_s),
        "has_plate_response": bool(result.has_plate_response),
        "EI_plate": float(result.EI_plate),
        "metadata_json": np.asarray(json.dumps(_jsonable(result.metadata))),
    }
    for name in _EDGE_NPZ_KEYS:
        payload[name] = np.asarray(result.edge_responses[name], dtype=float)
    for key in ("nx", "ny", "order"):
        value = result.metadata.get(key)
        if value is not None:
            payload[key] = int(value)
    if result.y_top is not None:
        payload["y_top"] = result.y_top
    if result.x_plate is not None:
        payload["x_plate"] = result.x_plate
    if result.y_plate is not None:
        payload["y_plate"] = result.y_plate
    if stored:
        intervals = [result.interval(r) for r in range(result.n_records)]
        for name, lists in (
            ("contact_node_ids", [iv.contact_node_ids for iv in intervals]),
            ("heel_free_node_ids", [iv.heel_free_node_ids for iv in intervals]),
            ("toe_free_node_ids", [iv.toe_free_node_ids for iv in intervals]),
            ("free_node_ids", [iv.free_node_ids for iv in intervals]),
        ):
            payload[f"{name}_flat"], payload[f"{name}_offsets"] = _csr(lists)
        for key in NODAL_FIELD_ARRAY_KEYS:
            payload[key] = np.asarray(getattr(result, key), dtype=float)
    solver = result.field_solver
    if solver is not None:
        payload[FIELD_SOLVER_COMPLIANCE_KEY] = solver.compliance_matrix
    if result.has_plate_response:
        plate_keys = PLATE_RESPONSE_ARRAY_KEYS if result.plate_u_local_basis is not None else PLATE_MESH_ARRAY_KEYS
        for key in plate_keys:
            payload[key] = np.asarray(getattr(result, key))
        if solver is not None and solver.plate_influence is not None:
            for name in PLATE_INFLUENCE_ARRAY_FIELDS:
                payload[PLATE_INFLUENCE_KEY_PREFIX + name] = np.asarray(getattr(solver.plate_influence, name))
    if result.rigid_alpha_basis is not None:
        payload["rigid_alpha_basis"] = np.asarray(result.rigid_alpha_basis, dtype=float)
    if result.measured_render:
        payload.update(render_section_to_payload(result.measured_render, result.geometry_metadata))
    np.savez_compressed(npz_path, **payload)

    _write_lookup_csv(result, output_dir / "contact_lookup.csv")

    summary = {
        **meta,
        "n_bottom_nodes": n_b,
        "n_intervals_theoretical": int(n_intervals(n_b)),
        "n_records": int(result.n_records),
        "n_valid": int(np.count_nonzero(result.valid_mask)),
        "rejections": result.rejection_summary(),
        "label_counts": {
            t.value: int(np.count_nonzero(np.asarray(result.contact_type_codes) == code_of(t)))
            for t in ContactType
        },
        "softplus_a": result.softplus_a,
        "softplus_kappa": result.softplus_kappa,
        "L": result.L,
        "H": result.H,
        "reciprocity_error": result.reciprocity_error,
        "build_time_s": float(result.build_time_s),
        "has_plate_response": bool(result.has_plate_response),
        "nodal_fields_stored": bool(stored),
        "metadata": result.metadata,
        "plate_metadata": result.plate_metadata,
        "geometry_metadata": result.geometry_metadata,
    }
    with (output_dir / "contact_lookup_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    return npz_path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def _write_lookup_csv(result: ContactLookupResult, path: Path) -> None:
    fieldnames = [
        "row",
        "contact_start_index",
        "contact_end_index",
        "topology_label",
        "contact_start_x",
        "contact_end_x",
        "contact_anchor_reference_x",
        "contact_anchor_reference_y",
        "heel_adjacent_free_node_id",
        "toe_adjacent_free_node_id",
        "n_free",
        "n_contact",
        "record_valid",
        "rejection_reason",
        "boundary_matrix_rank",
        "boundary_matrix_size",
        "kf_cond",
        "kf_det",
        "condition_estimate",
        "solve_residual",
        "force_equilibrium_residual",
        "moment_equilibrium_residual",
        "balance_residual",
    ]
    for col in AFFINE_COLUMN_NAMES:
        for name in SCALAR_LOOKUP_FIELDS:
            fieldnames.append(f"{name}_{col}")
    for col in AFFINE_COLUMN_NAMES:
        for name in EDGE_RESPONSE_FIELDS:
            fieldnames.append(f"{name}_{col}")
    for col in AFFINE_COLUMN_NAMES:
        fieldnames.append(f"Q_alpha_shoe_on_foot_{col}")
    q = lookup_q_alpha_basis(result)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(result.n_records):
            rec = {
                "row": row,
                "contact_start_index": int(result.contact_start_index[row]),
                "contact_end_index": int(result.contact_end_index[row]),
                "topology_label": result.contact_type(row).value,
                "contact_start_x": float(result.contact_start_x[row]),
                "contact_end_x": float(result.contact_end_x[row]),
                "contact_anchor_reference_x": float(result.contact_anchor_reference_x[row]),
                "contact_anchor_reference_y": float(result.contact_anchor_reference_y[row]),
                "heel_adjacent_free_node_id": int(result.heel_adjacent_free_node_id[row]),
                "toe_adjacent_free_node_id": int(result.toe_adjacent_free_node_id[row]),
                "n_free": int(result.n_free[row]),
                "n_contact": int(result.n_contact[row]),
                "record_valid": bool(result.valid_mask[row]),
                "rejection_reason": result.rejection_reason[row],
                "boundary_matrix_rank": int(result.boundary_matrix_rank[row]),
                "boundary_matrix_size": int(result.boundary_matrix_size[row]),
                "kf_cond": float(result.kf_cond[row]),
                "kf_det": float(result.kf_det[row]),
                "condition_estimate": float(result.condition_estimates[row]),
                "solve_residual": float(result.solve_residuals[row]),
                "force_equilibrium_residual": float(result.force_equilibrium_residuals[row]),
                "moment_equilibrium_residual": float(result.moment_equilibrium_residuals[row]),
                "balance_residual": float(result.balance_residuals[row]),
            }
            s = result.scalar_lookup[row]
            for k, col in enumerate(AFFINE_COLUMN_NAMES):
                for j, name in enumerate(SCALAR_LOOKUP_FIELDS):
                    rec[f"{name}_{col}"] = float(s[k, j])
                for name in EDGE_RESPONSE_FIELDS:
                    rec[f"{name}_{col}"] = float(result.edge_responses[name][row, k])
                rec[f"Q_alpha_shoe_on_foot_{col}"] = float(q[row, k])
            writer.writerow(rec)


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


def _plate_arrays_present(files: list[str]) -> bool:
    if not all(key in files for key in PLATE_MESH_ARRAY_KEYS):
        return False
    stored = all(key in files for key in PLATE_BASIS_ARRAY_KEYS)
    recomputable = all(PLATE_INFLUENCE_KEY_PREFIX + name in files for name in PLATE_INFLUENCE_ARRAY_FIELDS)
    return stored or recomputable


def _field_solver_from_npz(data, files: list[str], result: ContactLookupResult) -> RecordFieldSolver | None:
    if FIELD_SOLVER_COMPLIANCE_KEY not in files:
        return None
    full = np.asarray(data[FIELD_SOLVER_COMPLIANCE_KEY], dtype=float)
    n_tv = 2 * int(result.n_top_nodes)
    n_bv = 2 * int(result.n_bottom_nodes)
    if full.shape != (n_tv + n_bv, n_tv + n_bv):
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} {FIELD_SOLVER_COMPLIANCE_KEY} has shape {full.shape}.")
    blocks = ComplianceBlocks(
        Ctt=full[:n_tv, :n_tv],
        Ctb=full[:n_tv, n_tv:],
        Cbt=full[n_tv:, :n_tv],
        Cbb=full[n_tv:, n_tv:],
        x_top=np.asarray(result.x_top, dtype=float),
        x_bottom=np.asarray(result.x_bottom, dtype=float),
        L=float(result.L),
        H=float(result.H),
        E=float(result.E),
        nu=float(result.nu),
        reciprocity_error=float(result.reciprocity_error),
        y_top=None if result.y_top is None else np.asarray(result.y_top, dtype=float),
        y_bottom=None if result.y_bottom is None else np.asarray(result.y_bottom, dtype=float),
        geometry_type=result.geometry_type,
        dof_ordering=result.dof_ordering,
        n_top_nodes=int(result.n_top_nodes),
        n_bottom_nodes=int(result.n_bottom_nodes),
    )
    influence = None
    if result.has_plate_response and all(PLATE_INFLUENCE_KEY_PREFIX + n in files for n in PLATE_INFLUENCE_ARRAY_FIELDS):
        influence = PlateInfluence(
            n_top=int(result.n_top_nodes),
            n_bottom=int(result.n_bottom_nodes),
            **{n: np.asarray(data[PLATE_INFLUENCE_KEY_PREFIX + n], dtype=float) for n in PLATE_INFLUENCE_ARRAY_FIELDS},
        )
    return RecordFieldSolver(
        blocks,
        np.asarray(result.basis_top_displacements, dtype=float),
        float(result.solver_xr),
        float(result.solver_yr),
        float(result.softplus_a),
        plate_influence=influence,
    )


def load_contact_lookup(path: Path | str) -> ContactLookupResult:
    """Reload a schema-v10 interval lookup (v9 files are migrated) from NPZ.

    Files with an older schema (heel/toe/full topologies only) are rejected with
    :data:`REGENERATE_LOOKUP_MESSAGE`; they are never reinterpreted as interval
    support. The stored toe generalized-force basis is checked against the
    stored top forces.
    """
    path = resolve_contact_lookup_path(path)
    data = np.load(path, allow_pickle=True)
    files = list(data.files)
    schema_version = int(data["schema_version"]) if "schema_version" in files else 1
    if schema_version != LOOKUP_SCHEMA_VERSION and schema_version not in MIGRATABLE_SCHEMA_VERSIONS:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} (found schema_version={schema_version})")
    stored_geometry = _decode_flag(data["geometry_type"]) if "geometry_type" in files else ""
    if schema_version in MIGRATABLE_SCHEMA_VERSIONS and stored_geometry == "measured_sole":
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} (schema_version={schema_version} cannot contain "
            "measured-sole geometry; the file is inconsistent)"
        )
    model = _decode_flag(data["contact_set_model"]) if "contact_set_model" in files else ""
    if model != CONTACT_SET_MODEL:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} (contact_set_model={model!r})")
    anchor_def = (
        _decode_flag(data["contact_anchor_definition"])
        if "contact_anchor_definition" in files
        else ""
    )
    if anchor_def != CONTACT_ANCHOR_DEFINITION:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} (contact_anchor_definition={anchor_def!r})")
    basis_order = [str(s) for s in data["basis_order"].tolist()] if "basis_order" in files else []
    if basis_order != list(BASIS_ORDER):
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Saved basis order {basis_order} does not match {BASIS_ORDER}."
        )
    scalar = np.asarray(data["scalar_lookup"], dtype=float)
    scalar_fields = [str(s) for s in data["scalar_lookup_fields"].tolist()]
    if tuple(scalar_fields) != SCALAR_LOOKUP_FIELDS or scalar.ndim != 3 or scalar.shape[1] != N_AFFINE_COLUMNS:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Unexpected scalar_lookup layout.")
    n_b = int(len(np.asarray(data["x_bottom"])))
    starts = np.asarray(data["contact_start_index"], dtype=int)
    ends = np.asarray(data["contact_end_index"], dtype=int)
    if starts.size != n_intervals(n_b) or scalar.shape[0] != starts.size:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Expected {n_intervals(n_b)} interval records, found {starts.size}."
        )
    codes = np.asarray(data["contact_type_codes"], dtype=int)
    if not np.array_equal(codes, labels_from_arrays(starts, ends, n_b)):
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Topology labels disagree with endpoints.")

    def _arr(name: str) -> np.ndarray | None:
        if name not in files:
            return None
        arr = np.asarray(data[name], dtype=float)
        return arr if arr.size else None

    def _int(name: str) -> np.ndarray | None:
        if name not in files:
            return None
        return np.asarray(data[name], dtype=int)

    has_plate = _plate_arrays_present(files) and bool(
        np.asarray(data["has_plate_response"]).reshape(-1)[0]
    )
    if "has_plate_response" in files and bool(np.asarray(data["has_plate_response"]).reshape(-1)[0]) and not has_plate:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} has_plate_response=True but plate arrays are missing.")
    ei_plate = float(data["EI_plate"]) if "EI_plate" in files else float("nan")
    metadata = json.loads(_decode_flag(data["metadata_json"])) if "metadata_json" in files else {}
    metadata.update(interval_metadata())
    plate_metadata: dict = {"has_plate_response": False}
    if has_plate:
        plate_metadata = {
            "has_plate_response": True,
            "plate_model": PLATE_MODEL,
            "EI_plate": ei_plate,
            "plate_axial_model": PLATE_AXIAL_MODEL,
            "plate_rotation_sign": PLATE_ROTATION_SIGN,
            "plate_normal_sign": PLATE_NORMAL_SIGN,
            "constraint_row_normalization": CONSTRAINT_ROW_NORMALIZATION,
            "coordinate_frame": COORDINATE_FRAME,
            "n_plate_nodes": int(np.asarray(data["plate_node_ids"]).size),
            "n_plate_elements": int(np.asarray(data["plate_element_connectivity"]).shape[0]),
        }
    moment_origin = tuple(float(v) for v in np.asarray(data["moment_origin"], dtype=float).reshape(-1)[:2])

    result = ContactLookupResult(
        contact_start_index=starts,
        contact_end_index=ends,
        n_free=np.asarray(data["n_free"], dtype=int),
        n_contact=np.asarray(data["n_contact"], dtype=int),
        status=[str(s) for s in data["status"].tolist()],
        x_top=np.asarray(data["x_top"], dtype=float),
        x_bottom=np.asarray(data["x_bottom"], dtype=float),
        basis_top_displacements=np.asarray(data["basis_top_displacements"], dtype=float),
        scalar_lookup=scalar,
        softplus_a=float(data["softplus_a"]),
        softplus_kappa=float(data["softplus_kappa"]),
        L=float(data["L"]),
        H=float(data["H"]),
        E=float(data["E"]),
        nu=float(data["nu"]),
        condition_estimates=np.asarray(data["condition_estimates"], dtype=float),
        solve_residuals=np.asarray(data["solve_residuals"], dtype=float),
        top_displacement_residuals=np.asarray(data["top_displacement_residuals"], dtype=float),
        contact_displacement_residuals=np.asarray(data["contact_displacement_residuals"], dtype=float),
        force_equilibrium_residuals=np.asarray(data["force_equilibrium_residuals"], dtype=float),
        moment_equilibrium_residuals=np.asarray(data["moment_equilibrium_residuals"], dtype=float),
        reciprocity_error=float(data["reciprocity_error"]),
        record_valid=np.asarray(data["record_valid"], dtype=bool),
        rejection_reason=[str(s) for s in data["rejection_reason"].tolist()],
        boundary_matrix_rank=_int("boundary_matrix_rank"),
        boundary_matrix_size=_int("boundary_matrix_size"),
        contact_type_codes=codes,
        contact_start_x=np.asarray(data["contact_start_x"], dtype=float),
        contact_end_x=np.asarray(data["contact_end_x"], dtype=float),
        contact_anchor_reference_x=np.asarray(data["contact_anchor_reference_x"], dtype=float),
        contact_anchor_reference_y=np.asarray(data["contact_anchor_reference_y"], dtype=float),
        heel_contact_edge_node_id=_int("heel_contact_edge_node_id"),
        toe_contact_edge_node_id=_int("toe_contact_edge_node_id"),
        heel_adjacent_free_node_id=_int("heel_adjacent_free_node_id"),
        toe_adjacent_free_node_id=_int("toe_adjacent_free_node_id"),
        contact_mask=np.asarray(data["contact_mask"], dtype=bool),
        bottom_u_basis=_arr("bottom_u_basis"),
        bottom_v_basis=_arr("bottom_v_basis"),
        reaction_x_basis=_arr("reaction_x_basis"),
        reaction_y_basis=_arr("reaction_y_basis"),
        top_force_x_basis=_arr("top_force_x_basis"),
        top_force_y_basis=_arr("top_force_y_basis"),
        contact_force_total=_arr("contact_force_total"),
        balance_residuals=_arr("balance_residuals"),
        edge_responses={name: np.asarray(data[name], dtype=float) for name in EDGE_RESPONSE_FIELDS},
        kf_matrix=_arr("kf_matrix"),
        kf_svals=_arr("kf_svals"),
        kf_cond=_arr("kf_cond"),
        kf_det=_arr("kf_det"),
        y_top=_arr("y_top"),
        y_bottom=np.asarray(data["y_bottom"], dtype=float),
        x_plate=_arr("x_plate"),
        y_plate=_arr("y_plate"),
        geometry_type=_decode_flag(data["geometry_type"]),
        dof_ordering=_decode_flag(data["dof_ordering"]),
        n_top_nodes=int(data["n_top_nodes"]),
        n_bottom_nodes=int(data["n_bottom_nodes"]),
        moment_origin=moment_origin,
        solver_xr=float(data["solver_xr"]),
        solver_yr=float(data["solver_yr"]),
        phi_ref=float(data["phi_ref"]),
        build_time_s=float(data["build_time_s"]) if "build_time_s" in files else float("nan"),
        metadata={**metadata, **plate_metadata},
        schema_version=schema_version,
        has_plate_response=has_plate,
        plate_node_ids=_int("plate_node_ids") if has_plate else None,
        plate_reference_x=_arr("plate_reference_x") if has_plate else None,
        plate_reference_y=_arr("plate_reference_y") if has_plate else None,
        plate_reference_arc_length=_arr("plate_reference_arc_length") if has_plate else None,
        plate_element_connectivity=_int("plate_element_connectivity") if has_plate else None,
        plate_element_lengths=_arr("plate_element_lengths") if has_plate else None,
        plate_element_tangents=_arr("plate_element_tangents") if has_plate else None,
        plate_element_normals=_arr("plate_element_normals") if has_plate else None,
        plate_u_dof_ids=_int("plate_u_dof_ids") if has_plate else None,
        plate_v_dof_ids=_int("plate_v_dof_ids") if has_plate else None,
        plate_rotation_dof_ids=_int("plate_rotation_dof_ids") if has_plate else None,
        plate_constraint_ids=_int("plate_constraint_ids") if has_plate else None,
        plate_tangent=_arr("plate_tangent") if has_plate else None,
        plate_normal=_arr("plate_normal") if has_plate else None,
        EI_plate=ei_plate if has_plate else float("nan"),
        plate_u_local_basis=_arr("plate_u_local_basis") if has_plate else None,
        plate_v_local_basis=_arr("plate_v_local_basis") if has_plate else None,
        plate_rotation_local_basis=_arr("plate_rotation_local_basis") if has_plate else None,
        plate_constraint_multiplier_basis=_arr("plate_constraint_multiplier_basis") if has_plate else None,
        plate_axial_force_basis=_arr("plate_axial_force_basis") if has_plate else None,
        plate_bp_residual=_arr("plate_bp_residual") if has_plate else None,
        plate_metadata=plate_metadata,
        rigid_alpha_basis=_arr("rigid_alpha_basis"),
    )
    result.field_solver = _field_solver_from_npz(data, files, result)
    if not result.has_nodal_fields:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} The file stores neither the per-node bases nor "
            f"{FIELD_SOLVER_COMPLIANCE_KEY}."
        )
    if has_plate and result.plate_u_local_basis is None and (
        result.field_solver is None or result.field_solver.plate_influence is None
    ):
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Plate fields are neither stored nor recomputable.")
    result.metadata["nodal_fields_stored"] = result.nodal_fields_stored
    if schema_version != LOOKUP_SCHEMA_VERSION:
        result.migrated_from_schema = schema_version
        result.schema_version = LOOKUP_SCHEMA_VERSION
        result.metadata["migrated_from_schema"] = schema_version
    if result.geometry_type == "measured_sole":
        missing = [k for k in (*MEASURED_RENDER_KEYS, "geometry_metadata_json", "rigid_alpha_basis") if k not in files]
        if missing:
            raise ValueError(
                f"{REGENERATE_LOOKUP_MESSAGE} Measured-sole lookup is missing geometry arrays: {missing}."
            )
        result.measured_render = {
            **{k: np.asarray(data[k], dtype=int) for k in MEASURED_RENDER_INT_KEYS},
            **{k: np.asarray(data[k], dtype=float) for k in MEASURED_RENDER_FLOAT_KEYS},
        }
        result.geometry_metadata = json.loads(_decode_flag(data["geometry_metadata_json"]))
    stored_q = _arr("Q_alpha_shoe_on_foot_basis")
    if stored_q is None or stored_q.shape != (result.n_records, N_AFFINE_COLUMNS):
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Q_alpha_shoe_on_foot_basis is missing or misshapen.")
    # With on-demand fields only a few records are re-solved for this check.
    check_rows = result.valid_rows
    if not result.nodal_fields_stored and check_rows.size:
        check_rows = np.unique(check_rows[[0, check_rows.size // 2, -1]])
    mismatch = 0.0
    if check_rows.size:
        recomputed = q_alpha_shoe_on_foot_basis(
            result.record_fields(check_rows)["top_force_y"],
            result.basis_top_displacements,
            len(result.x_top),
            result.x_top,
            lookup_rearfoot_geometry(result),
        )
        scale = max(float(np.nanmax(np.abs(recomputed))), 1e-300)
        mismatch = float(np.nanmax(np.abs(stored_q[check_rows] - recomputed))) / scale
    if mismatch > 1e-9:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Stored Q_alpha_shoe_on_foot_basis disagrees with "
            f"-psi^T f_(t,y) from the stored top forces (relative {mismatch:.3e})."
        )
    result.Q_alpha_shoe_on_foot_basis = stored_q
    result.toe_generalized_force_source = "stored"
    result.metadata["toe_generalized_force"] = TOE_GENERALIZED_FORCE_DEFINITION
    result.metadata["toe_generalized_force_source"] = "stored"
    return result


if __name__ == "__main__":
    from compliance_fem.contact_lookup_cli import main

    main()
