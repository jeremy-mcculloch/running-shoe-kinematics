"""Contact-edge lookup table (schema v6) for heel, toe, and full topologies.

Schema v6 stores every contiguous contact topology once: heel candidates
``i = 0..N_b-2``, toe candidates ``i = 1..N_b-1``, and exactly one full-contact
record. Topology is explicit via ``contact_type_codes`` and never inferred from
the material edge coordinate ``l``. Layered lookups may also carry a
``plate_response`` section recovered from the in-process FEM factorization.
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from compliance_fem.boundaries import (
    DOF_ORDERING_COMPONENT_MAJOR_UV,
    component_major_indices,
    split_uv,
    vector_rigid_mode_matrix,
)
from compliance_fem.contact_basis import (
    BASIS_MODE_NAMES,
    MODE_BX,
    MODE_BY,
    N_BASIS_MODES,
    SHAPE_MODE_DEFINITION,
    SHAPE_MODE_NORMALIZATION,
    build_contact_displacement_matrix,
    build_top_displacement_matrix,
)
from compliance_fem.corotation import SHAPE_MODE_SIGN, reference_chord_angle

# ``free_contact_sets`` is re-exported so historical toe-family callers keep
# importing it from the lookup module.
from compliance_fem.contact_topology import (
    ContactRecordSpec,
    ContactType,
    anchor_reference_x,
    code_of,
    contact_free_sets,
    edge_node_pair,
    enumerate_records,
    free_contact_sets,
    n_records,
    type_from_code,
)
from compliance_fem.plate_response import (
    CONSTRAINT_ROW_NORMALIZATION,
    COORDINATE_FRAME,
    PLATE_AXIAL_MODEL,
    PLATE_MODEL,
    PLATE_NORMAL_SIGN,
    PLATE_ROTATION_SIGN,
    extract_plate_mesh_info,
    recover_plate_basis_for_record,
)

if TYPE_CHECKING:
    from compliance_fem.compliance import ComplianceResult

LOOKUP_SCHEMA_VERSION = 6
N_SCALAR_FIELDS = 10
# Topology-independent scalar names. ``edge_*`` refers to the record's contact
# edge and the free node adjoining it, which is (i, i+1) for heel contact and
# (i, i-1) for toe contact; all four are NaN for full contact.
SCALAR_LOOKUP_FIELDS = (
    "Fx",
    "Fy",
    "Mv",
    "Mz",
    "edge_free_gap_v",
    "edge_free_gap_u",
    "edge_contact_reaction_x",
    "edge_contact_reaction_y",
    "M_toe",
    "M_toe_vertical",
)
SCALAR_FX = 0
SCALAR_FY = 1
SCALAR_MV = 2
SCALAR_MZ = 3
SCALAR_GAP = 4
SCALAR_GAP_U = 5
SCALAR_RX = 6
SCALAR_RY = 7
SCALAR_TOE = 8
SCALAR_TOE_VERT = 9
# Longer aliases used by plotting / older call sites.
SCALAR_EDGE_GAP_V = SCALAR_GAP
SCALAR_EDGE_GAP_U = SCALAR_GAP_U
SCALAR_EDGE_RX = SCALAR_RX
SCALAR_EDGE_RY = SCALAR_RY

# Stable storage order for topology families (matches enumerate_records).
CONTACT_TYPE_NAMES = ("heel", "toe", "full")
HEEL_CORNER_NODE_ID = 0
FULL_CONTACT_ANCHOR_FRACTION = 0.5

# Corner axes of ``corner_reactions_local[record, basis, corner, component]``.
CORNER_NAMES = ("heel", "toe")
CORNER_COMPONENT_NAMES = ("x", "y")
CORNER_HEEL = 0
CORNER_TOE = 1
CORNER_REACTION_LAYOUT = (
    "[basis, corner={heel,toe}, component={x,y}]; local rotating-frame nodal "
    "forces. Finite only for the full-contact record."
)

# NPZ keys that mark a layered lookup as carrying plate_response fields.
PLATE_RESPONSE_ARRAY_KEYS = (
    "plate_u_local_basis",
    "plate_v_local_basis",
    "plate_rotation_local_basis",
    "plate_constraint_multiplier_basis",
    "plate_axial_force_basis",
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

SOLVER_YR = 0.0
MOMENT_ORIGIN = (0.0, 0.0)
REGENERATE_LOOKUP_MESSAGE = (
    "This contact-lookup file is not schema_version=6 (heel/full/toe contact topologies "
    "with the five-mode co-rotating basis and optional plate_response). Regenerate with "
    "the current contact_lookup CLI. Schema v4/v5 tables cannot be migrated: v4 stores "
    "only the toe-contact family, and v5 lacks plate_response fields."
)
REGENERATE_COMPLIANCE_MESSAGE = (
    "This compliance NPZ is not a vector (component-major u,v) file. "
    "Regenerate FEM with the current pipeline so Ctt/Ctb/Cbt/Cbb are 2n x 2n."
)


def contact_type_code(contact_type: ContactType | str) -> int:
    """Integer code for a topology (alias of :func:`code_of`)."""
    if isinstance(contact_type, str):
        contact_type = ContactType(contact_type)
    return code_of(contact_type)


def contact_type_from_code(code: int) -> ContactType:
    """Topology from stored integer code (alias of :func:`type_from_code`)."""
    return type_from_code(code)


def toe_corner_node_id(n_bottom: int) -> int:
    """Bottom-node id of the toe corner (``N_b - 1``)."""
    return int(n_bottom) - 1


def edge_reference_x(spec: ContactRecordSpec, x_bottom: np.ndarray) -> float:
    """Material contact-edge coordinate ``l_i``, or NaN for full contact."""
    if spec.contact_type is ContactType.FULL:
        return float("nan")
    return float(np.asarray(x_bottom, dtype=float)[int(spec.edge_node_id)])


def contact_mask_from_sets(contact: np.ndarray, n_bottom: int) -> np.ndarray:
    """Dense bool mask with True on contact nodes."""
    mask = np.zeros(int(n_bottom), dtype=bool)
    if np.asarray(contact).size:
        mask[np.asarray(contact, dtype=int)] = True
    return mask


def sets_from_contact_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(free, contact)`` node ids from a dense contact membership mask."""
    mask = np.asarray(mask, dtype=bool)
    contact = np.flatnonzero(mask)
    free = np.flatnonzero(~mask)
    return free, contact


def record_counts(n_bottom: int) -> dict[str, int]:
    """Expected per-topology and total record counts for ``n_bottom`` nodes."""
    n = int(n_bottom)
    return {
        "heel": n - 1,
        "toe": n - 1,
        "full": 1,
        "total": n_records(n),
    }


def ramp_virtual_displacement(x_top: np.ndarray, a: float) -> np.ndarray:
    """Return rho_a[j] = max(0, x_top[j] - a)."""
    return np.maximum(0.0, np.asarray(x_top, dtype=float) - float(a))


def interpolate_top_height(x_top: np.ndarray, y_top: np.ndarray, a: float) -> float:
    """Height of the top surface at x=a, interpolated from nodal y_top."""
    return float(np.interp(float(a), np.asarray(x_top, dtype=float), np.asarray(y_top, dtype=float)))


def ramp_horizontal_lever(x_top: np.ndarray, y_top: np.ndarray, a: float, H_a: float) -> np.ndarray:
    """eta_a[j] = (y_j - H_a) for x_j >= a, else 0. Flat top ⇒ eta = 0."""
    x_top = np.asarray(x_top, dtype=float)
    y_top = np.asarray(y_top, dtype=float)
    eta = np.zeros_like(x_top, dtype=float)
    mask = x_top >= float(a)
    eta[mask] = y_top[mask] - float(H_a)
    return eta


def compute_toe_basis(top_reactions: np.ndarray, x_top: np.ndarray, a: float) -> np.ndarray:
    """Discrete vertical toe moment T = rho_a^T f_{t,y} for one or more RHS columns.

    Parameters
    ----------
    top_reactions
        Shape (n_t,) or (n_t, n_rhs) vertical nodal forces.
    """
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
    if f_tx.ndim == 1:
        f_tx = f_tx[:, None]
        f_ty = f_ty[:, None]
        squeeze = True
    else:
        squeeze = False
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


@dataclass
class ContactLookupResult:
    """Contact-record lookup table and diagnostics for all three topologies.

    Rows are stored records, ordered heel (``i = 0..N_b-2``), toe
    (``i = 1..N_b-1``), then exactly one full-contact record. Topology is always
    read from ``contact_type_codes``, never inferred from ``candidate_l``.

    ``candidate_indices`` holds the contact-edge node id (``-1`` for full) and
    ``candidate_l`` the material contact-edge coordinate (``NaN`` for full).
    ``anchor_reference_x`` is defined for every record: the contact edge for
    heel/toe, the numerical anchor ``L/2`` for full.
    """

    candidate_indices: np.ndarray
    candidate_l: np.ndarray
    n_free: np.ndarray
    n_contact: np.ndarray
    status: list[str]
    x_top: np.ndarray
    x_bottom: np.ndarray
    basis_top_displacements: np.ndarray
    scalar_lookup: np.ndarray
    gap_basis: np.ndarray
    reaction_basis: np.ndarray
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
    include_endpoints: bool = True
    metadata: dict = field(default_factory=dict)
    schema_version: int = LOOKUP_SCHEMA_VERSION
    top_force_basis: np.ndarray | None = None
    top_force_x_basis: np.ndarray | None = None
    top_force_y_basis: np.ndarray | None = None
    gap_u_basis: np.ndarray | None = None
    gap_v_basis: np.ndarray | None = None
    reaction_x_basis: np.ndarray | None = None
    reaction_y_basis: np.ndarray | None = None
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
    contact_type_codes: np.ndarray | None = None
    edge_node_ids: np.ndarray | None = None
    anchor_reference_x: np.ndarray | None = None
    edge_contact_node_ids: np.ndarray | None = None
    edge_free_node_ids: np.ndarray | None = None
    has_free_edge: np.ndarray | None = None
    contact_mask: np.ndarray | None = None
    corner_reactions_local: np.ndarray | None = None
    heel_corner_node_id: int = HEEL_CORNER_NODE_ID
    toe_corner_node_id: int = -1
    full_contact_anchor_x: float = float("nan")
    # Optional plate_response section (layered lookups only).
    has_plate_response: bool = False
    plate_node_ids: np.ndarray | None = None
    plate_reference_x: np.ndarray | None = None
    plate_reference_y: np.ndarray | None = None
    plate_reference_arc_length: np.ndarray | None = None
    plate_element_connectivity: np.ndarray | None = None  # (n_el, 2)
    plate_element_lengths: np.ndarray | None = None
    plate_element_tangents: np.ndarray | None = None  # (n_el, 2)
    plate_element_normals: np.ndarray | None = None
    plate_u_dof_ids: np.ndarray | None = None
    plate_v_dof_ids: np.ndarray | None = None
    plate_rotation_dof_ids: np.ndarray | None = None
    plate_constraint_ids: np.ndarray | None = None
    plate_tangent: np.ndarray | None = None  # (2,)
    plate_normal: np.ndarray | None = None  # (2,)
    EI_plate: float = float("nan")
    plate_u_local_basis: np.ndarray | None = None  # (n_records, 5, n_plate)
    plate_v_local_basis: np.ndarray | None = None
    plate_rotation_local_basis: np.ndarray | None = None
    plate_constraint_multiplier_basis: np.ndarray | None = None  # (n_records, 5, n_el)
    plate_axial_force_basis: np.ndarray | None = None
    plate_bp_residual: np.ndarray | None = None  # (n_records, 5)
    plate_metadata: dict = field(default_factory=dict)

    def requires_regeneration_for_toe(self) -> bool:
        """True when this file is older than the current schema."""
        return int(self.schema_version) < LOOKUP_SCHEMA_VERSION

    @property
    def n_records(self) -> int:
        """Number of stored contact records."""
        return int(len(self.candidate_indices))

    def contact_type(self, row: int) -> ContactType:
        """Explicit topology of one record."""
        if self.contact_type_codes is None:
            raise ValueError(REGENERATE_LOOKUP_MESSAGE)
        return contact_type_from_code(int(self.contact_type_codes[int(row)]))

    def contact_types(self) -> tuple[ContactType, ...]:
        """Topology of every record, in storage order."""
        return tuple(self.contact_type(row) for row in range(self.n_records))

    def record_spec(self, row: int) -> ContactRecordSpec:
        """Rebuild the record spec for one row."""
        kind = self.contact_type(row)
        if kind is ContactType.FULL:
            return ContactRecordSpec(ContactType.FULL, None)
        return ContactRecordSpec(kind, int(self.edge_node_ids[int(row)]))

    def record_sets(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``(free, contact)`` bottom-node indices of one record, from the stored mask."""
        if self.contact_mask is None:
            raise ValueError(REGENERATE_LOOKUP_MESSAGE)
        return sets_from_contact_mask(self.contact_mask[int(row)])

    def rows_for(self, contact_type: ContactType | str) -> np.ndarray:
        """Row indices belonging to one topology family."""
        if self.contact_type_codes is None:
            raise ValueError(REGENERATE_LOOKUP_MESSAGE)
        code = contact_type_code(contact_type)
        return np.flatnonzero(np.asarray(self.contact_type_codes, dtype=int) == code)

    def full_contact_row(self) -> int:
        """Row index of the single full-contact record."""
        rows = self.rows_for(ContactType.FULL)
        if rows.size != 1:
            raise ValueError(
                f"Expected exactly one full-contact record, found {rows.size}. "
                + REGENERATE_LOOKUP_MESSAGE
            )
        return int(rows[0])


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
    geometry_type = "rectangle"
    if "geometry_type" in data.files:
        geometry_type = _decode_flag(data["geometry_type"])
    dof_ordering = (
        _decode_flag(data["dof_ordering"])
        if "dof_ordering" in data.files
        else ""
    )
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


def candidate_indices(n_bottom: int, include_endpoints: bool = True) -> np.ndarray:
    """Return contact-edge node indices for one partial-contact family."""
    if n_bottom < 3 and not include_endpoints:
        raise ValueError("Need at least three bottom nodes for interior candidates.")
    if include_endpoints:
        return np.arange(n_bottom, dtype=int)
    return np.arange(1, n_bottom - 1, dtype=int)


def contact_records(n_bottom: int, include_endpoints: bool = True) -> tuple[ContactRecordSpec, ...]:
    """Stored records: heel candidates, toe candidates, then exactly one full record.

    ``include_endpoints`` is retained for CLI compatibility but ignored: schema v6
    always enumerates every heel/toe/full topology via :func:`enumerate_records`.
    """
    del include_endpoints  # always enumerate all topologies
    return tuple(enumerate_records(n_bottom))


def _relative_asymmetry(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))


def prepare_compliance_blocks(
    blocks: ComplianceBlocks,
    reciprocity_tol: float = 1e-6,
) -> tuple[ComplianceBlocks, float]:
    """Verify reciprocity and symmetrize roundoff-level asymmetry.

    Large free-body saddles (especially the layered inextensible-plate
    operator) can leave relative C symmetry near 1e-7 after SuperLU even
    when column residuals are ~1e-10. That is still small enough to remove
    by averaging; errors above ``reciprocity_tol`` are rejected.
    Horizontal/vertical coupling does not imply Fx*=0 ⇒ c=0.
    """
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
    n_b = Cbb.shape[0]
    full = np.block([[Ctt, Ctb], [Cbt, Cbb]])
    full = 0.5 * (full + full.T)
    Ctt = full[:n_t, :n_t]
    Ctb = full[:n_t, n_t:]
    Cbt = full[n_t:, :n_t]
    Cbb = full[n_t:, n_t:]

    prepared = ComplianceBlocks(
        Ctt=Ctt,
        Ctb=Ctb,
        Cbt=Cbt,
        Cbb=Cbb,
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
    """Build restricted compliance blocks for component-major free/contact partitions."""
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
        "C_ft": (
            blocks.Cbt[free_dofs, :] if n_f_vec else np.zeros((0, n_t_vec))
        ),
        "C_fc": (
            blocks.Cbb[np.ix_(free_dofs, contact_dofs)]
            if n_f_vec
            else np.zeros((0, n_c_vec))
        ),
    }


def build_boundary_matrix(
    blocks: ComplianceBlocks,
    free: np.ndarray,
    contact: np.ndarray,
    x_r: float,
    y_r: float = SOLVER_YR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Assemble A_i with three rigid columns for one candidate."""
    rb = restricted_blocks(blocks, free, contact)
    y_top, y_bottom = _boundary_y(blocks)
    R_t = vector_rigid_mode_matrix(blocks.x_top, y_top, x_r, y_r)
    if contact.size:
        R_c = vector_rigid_mode_matrix(blocks.x_bottom[contact], y_bottom[contact], x_r, y_r)
    else:
        R_c = np.zeros((0, 3), dtype=float)
    if free.size:
        R_f = vector_rigid_mode_matrix(blocks.x_bottom[free], y_bottom[free], x_r, y_r)
    else:
        R_f = np.zeros((0, 3), dtype=float)

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
    return A, R_t, R_f, rb


def kf_from_scalars(scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build the translation-to-force matrix K_F from a (5, n_fields) scalar table.

    K_F = [[F_x,Bx, F_x,By], [F_y,Bx, F_y,By]] maps the anchor translation
    (d_ax, d_ay) to the local top resultant.

    Returns ``(K_F, singular_values, cond, det)``.
    """
    kf = np.array(
        [
            [float(scalars[MODE_BX, SCALAR_FX]), float(scalars[MODE_BY, SCALAR_FX])],
            [float(scalars[MODE_BX, SCALAR_FY]), float(scalars[MODE_BY, SCALAR_FY])],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(kf)):
        return kf, np.full(2, np.nan), float("inf"), float("nan")
    det = float(np.linalg.det(kf))
    try:
        svals = np.linalg.svd(kf, compute_uv=False).astype(float)
        cond = float(svals[0] / svals[-1]) if svals[-1] > 0.0 else float("inf")
    except np.linalg.LinAlgError:
        svals = np.full(2, np.nan)
        cond = float("inf")
    return kf, svals, cond, det


def solve_candidate(
    blocks: ComplianceBlocks,
    spec: ContactRecordSpec,
    W: np.ndarray,
    x_r: float,
    a: float,
    y_r: float = SOLVER_YR,
) -> dict[str, np.ndarray | float | str | int | None]:
    """Solve the five-mode vector boundary system for one contact record.

    ``spec`` carries the topology and, for heel or toe contact, the contact-edge
    node. Only the boundary selectors, the basis anchor ``x_anchor``, and the
    edge-node picks depend on the topology; the full-domain compliance blocks
    are never rebuilt.

    ``W`` is the top block W_t; the contact block W_c is built here from
    ``x_anchor``, which is ``l_i`` for partial contact and ``L/2`` for full
    contact. No co-rotation angle is involved.
    """
    n_b = len(blocks.x_bottom)
    n_t = len(blocks.x_top)
    y_top, _y_bottom = _boundary_y(blocks)
    # contact_free_sets returns (contact, free); build_boundary_matrix wants free then contact.
    contact, free = contact_free_sets(spec, n_b)
    n_f, n_c = free.size, contact.size
    n_t_vec = 2 * n_t
    n_c_vec = 2 * n_c
    n_modes = W.shape[1]
    if W.shape != (n_t_vec, N_BASIS_MODES):
        raise ValueError(
            f"W has shape {W.shape}; expected ({n_t_vec}, {N_BASIS_MODES}) component-major five-mode matrix."
        )

    is_full = spec.contact_type is ContactType.FULL
    if is_full:
        status = "full_contact"
    elif n_c < 2:
        status = "single_contact_node"
    else:
        status = "ok"

    x_anchor = anchor_reference_x(spec, blocks.x_bottom, blocks.L)
    l_i = edge_reference_x(spec, blocks.x_bottom)
    edge_contact_node_id, edge_free_node_id = edge_node_pair(spec, n_b)
    W_c = build_contact_displacement_matrix(blocks.x_bottom[contact], x_anchor)

    A, R_t, R_f, rb = build_boundary_matrix(blocks, free, contact, x_r, y_r=y_r)
    rhs = np.zeros((A.shape[0], n_modes), dtype=float)
    rhs[:n_t_vec, :] = W
    if n_c:
        rhs[n_t_vec : n_t_vec + n_c_vec, :] = W_c

    where = f"{spec.contact_type.value} edge={spec.edge_node_id}"
    # Factor the candidate matrix once and solve all five right-hand sides.
    try:
        lu, piv = linalg.lu_factor(A, check_finite=True)
        X = linalg.lu_solve((lu, piv), rhs, check_finite=True)
    except (linalg.LinAlgError, ValueError) as exc:
        raise RuntimeError(f"Boundary solve failed for {where}: {exc}") from exc

    F_t = X[:n_t_vec, :]
    F_c = X[n_t_vec : n_t_vec + n_c_vec, :] if n_c else np.zeros((0, n_modes))
    Alpha = X[n_t_vec + n_c_vec :, :]

    if n_f:
        G_f = rb["C_ft"] @ F_t + rb["C_fc"] @ F_c + R_f @ Alpha
    else:
        G_f = np.zeros((0, n_modes))

    f_tx, f_ty = split_uv(F_t, n_t)
    if n_c:
        f_cx, f_cy = split_uv(F_c, n_c)
    else:
        f_cx = np.zeros((0, n_modes))
        f_cy = np.zeros((0, n_modes))
    if n_f:
        u_f, v_f = split_uv(G_f, n_f)
    else:
        u_f = np.zeros((0, n_modes))
        v_f = np.zeros((0, n_modes))

    gap_u_full = np.zeros((n_modes, n_b), dtype=float)
    gap_v_full = np.zeros((n_modes, n_b), dtype=float)
    reaction_x_full = np.zeros((n_modes, n_b), dtype=float)
    reaction_y_full = np.zeros((n_modes, n_b), dtype=float)
    if n_f:
        gap_u_full[:, free] = u_f.T
        gap_v_full[:, free] = v_f.T
    if n_c:
        reaction_x_full[:, contact] = f_cx.T
        reaction_y_full[:, contact] = f_cy.T

    T_toe, T_vert, _H_a = compute_toe_moment(f_tx, f_ty, blocks.x_top, y_top, a)
    T_toe = np.atleast_1d(np.asarray(T_toe, dtype=float))
    T_vert = np.atleast_1d(np.asarray(T_vert, dtype=float))

    # Local indices of the topology edge nodes within the free/contact partitions.
    free_edge_local: int | None = None
    contact_edge_local: int | None = None
    if edge_free_node_id is not None and n_f:
        matches = np.flatnonzero(free == int(edge_free_node_id))
        if matches.size:
            free_edge_local = int(matches[0])
    if edge_contact_node_id is not None and n_c:
        matches = np.flatnonzero(contact == int(edge_contact_node_id))
        if matches.size:
            contact_edge_local = int(matches[0])

    scalars = np.full((n_modes, N_SCALAR_FIELDS), np.nan, dtype=float)
    for k in range(n_modes):
        fx = float(np.sum(f_tx[:, k]))
        fy = float(np.sum(f_ty[:, k]))
        Mv = float(np.dot(blocks.x_top, f_ty[:, k]))
        Mz = float(np.dot(blocks.x_top, f_ty[:, k]) - np.dot(y_top, f_tx[:, k]))
        # Edge quantities from the free/contact local indices of the topology
        # edge pair. Full contact has neither, so both stay NaN.
        if free_edge_local is None:
            edge_gap_v = np.nan
            edge_gap_u = np.nan
        else:
            edge_gap_v = float(v_f[free_edge_local, k])
            edge_gap_u = float(u_f[free_edge_local, k])
        if contact_edge_local is None:
            edge_rx = np.nan
            edge_ry = np.nan
        else:
            edge_rx = float(f_cx[contact_edge_local, k])
            edge_ry = float(f_cy[contact_edge_local, k])
        scalars[k] = [
            fx,
            fy,
            Mv,
            Mz,
            edge_gap_v,
            edge_gap_u,
            edge_rx,
            edge_ry,
            float(T_toe[k]),
            float(T_vert[k]),
        ]

    # Full contact has no lift-off edge; validity uses the two bottom corner
    # reactions (contact index 0 = heel node 0, index -1 = toe node N-1).
    corner_local = np.full((n_modes, 2, 2), np.nan, dtype=float)
    if is_full and n_c:
        for k in range(n_modes):
            corner_local[k, CORNER_HEEL, 0] = float(f_cx[0, k])
            corner_local[k, CORNER_HEEL, 1] = float(f_cy[0, k])
            corner_local[k, CORNER_TOE, 0] = float(f_cx[-1, k])
            corner_local[k, CORNER_TOE, 1] = float(f_cy[-1, k])

    kf, kf_svals, kf_cond, kf_det = kf_from_scalars(scalars)

    residual_matrix = A @ X - rhs
    solve_residual = float(np.linalg.norm(residual_matrix) / max(np.linalg.norm(rhs), 1e-30))

    top_disp = blocks.Ctt @ F_t + rb["C_tc"] @ F_c + R_t @ Alpha
    top_res = float(np.linalg.norm(top_disp - W) / max(np.linalg.norm(W), 1e-30))

    if n_c:
        y_c = _boundary_y(blocks)[1][contact]
        R_c = vector_rigid_mode_matrix(blocks.x_bottom[contact], y_c, x_r, y_r)
        contact_disp = rb["C_ct"] @ F_t + rb["C_cc"] @ F_c + R_c @ Alpha
        contact_res = float(
            np.linalg.norm(contact_disp - W_c)
            / max(1.0, np.linalg.norm(W), np.linalg.norm(W_c))
        )
    else:
        contact_res = 0.0

    if n_c:
        eq = R_t.T @ F_t + vector_rigid_mode_matrix(
            blocks.x_bottom[contact], _boundary_y(blocks)[1][contact], x_r, y_r
        ).T @ F_c
    else:
        eq = R_t.T @ F_t
    force_res = float(np.linalg.norm(eq[:2, :]))
    moment_res = float(np.linalg.norm(eq[2, :]))
    # R_t^T F_t + R_c^T F_c ~ 0 must hold for every basis response; report it
    # relative to the force magnitude so the check is mesh-scale independent.
    force_scale = max(float(np.linalg.norm(F_t)), float(np.linalg.norm(F_c)), 1e-30)
    lever_scale = max(float(np.max(np.abs(blocks.x_bottom - x_r))), 1e-30)
    equilibrium_rel = max(force_res, moment_res / lever_scale) / force_scale
    cond = float(np.linalg.cond(A))

    return {
        "status": status,
        "contact_type": spec.contact_type,
        "edge_node_id": -1 if spec.edge_node_id is None else int(spec.edge_node_id),
        "edge_reference_x": l_i,
        "x_anchor": x_anchor,
        "anchor_reference_x": x_anchor,
        "edge_contact_node_id": -1 if edge_contact_node_id is None else int(edge_contact_node_id),
        "edge_free_node_id": -1 if edge_free_node_id is None else int(edge_free_node_id),
        "has_free_edge": edge_free_node_id is not None,
        "contact_mask": contact_mask_from_sets(contact, n_b),
        "corner_reactions_local": corner_local,
        "n_free": n_f,
        "n_contact": n_c,
        "F_t": F_t,
        "F_c": F_c,
        "Alpha": Alpha,
        "G_f": G_f,
        "gap_full": gap_v_full,
        "gap_u_full": gap_u_full,
        "gap_v_full": gap_v_full,
        "reaction_full": reaction_y_full,
        "reaction_x_full": reaction_x_full,
        "reaction_y_full": reaction_y_full,
        "scalars": scalars,
        "W_c": W_c,
        "kf_matrix": kf,
        "kf_svals": kf_svals,
        "kf_cond": kf_cond,
        "kf_det": kf_det,
        "condition": cond,
        "solve_residual": solve_residual,
        "top_residual": top_res,
        "contact_residual": contact_res,
        "force_residual": force_res,
        "moment_residual": moment_res,
        "equilibrium_rel": equilibrium_rel,
        "A": A,
        "free": free,
        "contact": contact,
    }


def _supports_plate_recovery(fem_result: ComplianceResult | None) -> bool:
    """True when ``fem_result`` retains the layered factorization needed for plate recovery."""
    if fem_result is None:
        return False
    return (
        fem_result.B_p is not None
        and fem_result.factorization is not None
        and fem_result.x_plate is not None
        and fem_result.basis is not None
        and fem_result.n_primal is not None
    )


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
) -> ContactLookupResult:
    """Generate the phi-independent five-mode contact-edge lookup table.

    Nothing in this function takes or observes a co-rotation angle: the table is
    generated once and combined with ``phi`` only at runtime.

    ``include_endpoints`` is retained for CLI compatibility but ignored; schema v6
    always enumerates heel, toe, and full topologies via :func:`enumerate_records`.

    When ``fem_result`` is a layered :class:`ComplianceResult` retaining ``B_p``,
    factorization, and plate coordinates, each record also stores five-mode plate
    basis fields recovered from the same free-body operator. Rectangle / NPZ-only
    paths leave ``has_plate_response=False``.
    """
    del include_endpoints  # always enumerate all topologies
    prepared, reciprocity_error = prepare_compliance_blocks(blocks, reciprocity_tol=reciprocity_tol)
    n_b = len(prepared.x_bottom)
    n_t = len(prepared.x_top)
    records = enumerate_records(n_b)
    x_r = 0.5 * prepared.L
    y_r = SOLVER_YR
    warn_if_a_not_on_top_grid(prepared.x_top, a)
    W = build_top_displacement_matrix(prepared.x_top, prepared.L, a, kappa)
    phi_ref = reference_chord_angle(prepared.x_top, prepared.y_top)

    recover_plate = _supports_plate_recovery(fem_result)
    if require_plate and not recover_plate:
        raise ValueError(
            "require_plate=True but fem_result is missing B_p, factorization, basis, "
            "or x_plate (rectangle / NPZ-only path cannot recover plate_response)."
        )
    plate_mesh = extract_plate_mesh_info(fem_result) if recover_plate else None
    n_plate = int(plate_mesh.n_nodes) if plate_mesh is not None else 0
    n_el = int(plate_mesh.n_elements) if plate_mesh is not None else 0

    n_cand = len(records)
    n_modes = N_BASIS_MODES
    indices = np.array(
        [-1 if spec.edge_node_id is None else int(spec.edge_node_id) for spec in records],
        dtype=int,
    )
    type_codes = np.array([code_of(spec.contact_type) for spec in records], dtype=int)
    edge_l = np.array([edge_reference_x(spec, prepared.x_bottom) for spec in records], dtype=float)
    anchor_x = np.array(
        [anchor_reference_x(spec, prepared.x_bottom, prepared.L) for spec in records], dtype=float
    )
    edge_contact_ids = np.full(n_cand, -1, dtype=int)
    edge_free_ids = np.full(n_cand, -1, dtype=int)
    has_free_edge = np.zeros(n_cand, dtype=bool)
    contact_mask = np.zeros((n_cand, n_b), dtype=bool)
    corner_reactions_local = np.full((n_cand, n_modes, 2, 2), np.nan, dtype=float)
    scalar_lookup = np.full((n_cand, n_modes, N_SCALAR_FIELDS), np.nan, dtype=float)
    gap_u_basis = np.zeros((n_cand, n_modes, n_b), dtype=float)
    gap_v_basis = np.zeros((n_cand, n_modes, n_b), dtype=float)
    reaction_x_basis = np.zeros((n_cand, n_modes, n_b), dtype=float)
    reaction_y_basis = np.zeros((n_cand, n_modes, n_b), dtype=float)
    top_force_x_basis = np.zeros((n_cand, n_modes, n_t), dtype=float)
    top_force_y_basis = np.zeros((n_cand, n_modes, n_t), dtype=float)
    kf_matrix = np.full((n_cand, 2, 2), np.nan, dtype=float)
    kf_svals = np.full((n_cand, 2), np.nan, dtype=float)
    kf_cond = np.full(n_cand, np.nan)
    kf_det = np.full(n_cand, np.nan)
    condition_estimates = np.full(n_cand, np.nan)
    solve_residuals = np.full(n_cand, np.nan)
    top_residuals = np.full(n_cand, np.nan)
    contact_residuals = np.full(n_cand, np.nan)
    force_residuals = np.full(n_cand, np.nan)
    moment_residuals = np.full(n_cand, np.nan)
    n_free = np.zeros(n_cand, dtype=int)
    n_contact = np.zeros(n_cand, dtype=int)
    status: list[str] = []

    plate_u_local_basis = (
        np.zeros((n_cand, n_modes, n_plate), dtype=float) if recover_plate else None
    )
    plate_v_local_basis = (
        np.zeros((n_cand, n_modes, n_plate), dtype=float) if recover_plate else None
    )
    plate_rotation_local_basis = (
        np.zeros((n_cand, n_modes, n_plate), dtype=float) if recover_plate else None
    )
    plate_constraint_multiplier_basis = (
        np.zeros((n_cand, n_modes, n_el), dtype=float) if recover_plate else None
    )
    plate_axial_force_basis = (
        np.zeros((n_cand, n_modes, n_el), dtype=float) if recover_plate else None
    )
    plate_bp_residual = np.zeros((n_cand, n_modes), dtype=float) if recover_plate else None

    for row, spec in enumerate(records):
        sol = solve_candidate(prepared, spec, W, x_r, a=a, y_r=y_r)
        status.append(str(sol["status"]))
        edge_contact_ids[row] = int(sol["edge_contact_node_id"])
        edge_free_ids[row] = int(sol["edge_free_node_id"])
        has_free_edge[row] = bool(sol["has_free_edge"])
        contact_mask[row] = np.asarray(sol["contact_mask"], dtype=bool)
        corner_reactions_local[row] = np.asarray(sol["corner_reactions_local"], dtype=float)
        n_free[row] = int(sol["n_free"])
        n_contact[row] = int(sol["n_contact"])
        scalar_lookup[row] = sol["scalars"]
        gap_u_basis[row] = sol["gap_u_full"]
        gap_v_basis[row] = sol["gap_v_full"]
        reaction_x_basis[row] = sol["reaction_x_full"]
        reaction_y_basis[row] = sol["reaction_y_full"]
        F_t = np.asarray(sol["F_t"])
        f_tx, f_ty = split_uv(F_t, n_t)
        top_force_x_basis[row] = f_tx.T
        top_force_y_basis[row] = f_ty.T
        kf_matrix[row] = sol["kf_matrix"]
        kf_svals[row] = np.asarray(sol["kf_svals"], dtype=float).reshape(-1)[:2]
        kf_cond[row] = float(sol["kf_cond"])
        kf_det[row] = float(sol["kf_det"])
        condition_estimates[row] = float(sol["condition"])
        solve_residuals[row] = float(sol["solve_residual"])
        top_residuals[row] = float(sol["top_residual"])
        contact_residuals[row] = float(sol["contact_residual"])
        force_residuals[row] = float(sol["force_residual"])
        moment_residuals[row] = float(sol["moment_residual"])

        # Every record is a genuine boundary-value solve, including full contact
        # and the degenerate single-contact-node ends, so all of them must pass
        # the residual and equilibrium checks.
        where = f"{spec.contact_type.value} record i={spec.edge_node_id}"
        if float(sol["solve_residual"]) > 1e-6:
            raise ValueError(f"Large solve residual {sol['solve_residual']:.3e} at {where}.")
        if float(sol["top_residual"]) > 1e-6:
            raise ValueError(
                f"Large top-displacement residual {sol['top_residual']:.3e} at {where}."
            )
        if float(sol["contact_residual"]) > 1e-6:
            raise ValueError(
                f"Large contact-displacement residual {sol['contact_residual']:.3e} at {where}."
            )
        if float(sol["equilibrium_rel"]) > 1e-8:
            raise ValueError(
                f"Basis responses violate R_t^T F_t + R_c^T F_c = 0 by "
                f"{sol['equilibrium_rel']:.3e} (relative) at {where}."
            )

        if recover_plate:
            assert plate_mesh is not None and fem_result is not None
            contact_nodes = np.asarray(sol["contact"], dtype=int)
            W_c = np.asarray(sol["W_c"], dtype=float)
            x_contact = prepared.x_bottom[contact_nodes] if contact_nodes.size else np.zeros(0)
            plate = recover_plate_basis_for_record(
                fem_result,
                plate_mesh,
                np.asarray(sol["F_t"], dtype=float),
                np.asarray(sol["F_c"], dtype=float),
                contact_nodes,
                W,
                W_c,
                prepared.x_top,
                x_contact,
                Alpha=np.asarray(sol["Alpha"], dtype=float),
                x_r=x_r,
                y_r=y_r,
            )
            bp_max = float(np.max(plate.bp_residual)) if plate.bp_residual.size else 0.0
            top_bc_max = float(np.max(plate.top_bc_residual)) if plate.top_bc_residual.size else 0.0
            contact_bc_max = (
                float(np.max(plate.contact_bc_residual)) if plate.contact_bc_residual.size else 0.0
            )
            if bp_max > plate_bp_tol:
                raise ValueError(
                    f"Plate inextensibility residual {bp_max:.3e} exceeds plate_bp_tol="
                    f"{plate_bp_tol:.3e} at {where}."
                )
            if top_bc_max > plate_bc_tol:
                raise ValueError(
                    f"Plate top BC residual {top_bc_max:.3e} exceeds plate_bc_tol="
                    f"{plate_bc_tol:.3e} at {where}."
                )
            if contact_bc_max > plate_bc_tol:
                raise ValueError(
                    f"Plate contact BC residual {contact_bc_max:.3e} exceeds plate_bc_tol="
                    f"{plate_bc_tol:.3e} at {where}."
                )
            assert plate_u_local_basis is not None
            assert plate_v_local_basis is not None
            assert plate_rotation_local_basis is not None
            assert plate_constraint_multiplier_basis is not None
            assert plate_axial_force_basis is not None
            assert plate_bp_residual is not None
            plate_u_local_basis[row] = plate.u_local
            plate_v_local_basis[row] = plate.v_local
            plate_rotation_local_basis[row] = plate.rotation_local
            plate_constraint_multiplier_basis[row] = plate.constraint_multiplier
            plate_axial_force_basis[row] = plate.axial_force
            plate_bp_residual[row] = plate.bp_residual

    n_heel = int(np.count_nonzero(type_codes == code_of(ContactType.HEEL)))
    n_toe = int(np.count_nonzero(type_codes == code_of(ContactType.TOE)))
    n_full = int(np.count_nonzero(type_codes == code_of(ContactType.FULL)))
    if n_full != 1:
        raise ValueError(f"Expected exactly one full-contact record, generated {n_full}.")

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
        }

    return ContactLookupResult(
        candidate_indices=indices,
        candidate_l=edge_l,
        n_free=n_free,
        n_contact=n_contact,
        status=status,
        x_top=prepared.x_top.copy(),
        x_bottom=prepared.x_bottom.copy(),
        basis_top_displacements=W,
        scalar_lookup=scalar_lookup,
        gap_basis=gap_v_basis,
        reaction_basis=reaction_y_basis,
        softplus_a=float(a),
        softplus_kappa=float(kappa),
        L=prepared.L,
        H=prepared.H,
        E=prepared.E,
        nu=prepared.nu,
        condition_estimates=condition_estimates,
        solve_residuals=solve_residuals,
        top_displacement_residuals=top_residuals,
        contact_displacement_residuals=contact_residuals,
        force_equilibrium_residuals=force_residuals,
        moment_equilibrium_residuals=moment_residuals,
        reciprocity_error=reciprocity_error,
        include_endpoints=True,
        schema_version=LOOKUP_SCHEMA_VERSION,
        top_force_basis=top_force_y_basis,
        top_force_x_basis=top_force_x_basis,
        top_force_y_basis=top_force_y_basis,
        gap_u_basis=gap_u_basis,
        gap_v_basis=gap_v_basis,
        reaction_x_basis=reaction_x_basis,
        reaction_y_basis=reaction_y_basis,
        kf_matrix=kf_matrix,
        kf_svals=kf_svals,
        kf_cond=kf_cond,
        kf_det=kf_det,
        metadata={
            "moment_reference": "Mz about (0,0); Mv = x_t^T f_{t,y}; xr=L/2, yr=0 inside solver",
            "edge_node_in_contact": True,
            "reaction_is_nodal_force": True,
            "toe_moment": "rho_a^T f_{t,y} - eta_a^T f_{t,x} about (a, H_a)",
            "dof_ordering": DOF_ORDERING_COMPONENT_MAJOR_UV,
            "schema_version": LOOKUP_SCHEMA_VERSION,
            "nx": prepared.nx,
            "ny": prepared.ny,
            "order": prepared.order,
            "geometry_type": prepared.geometry_type,
            "n_shape_modes": 1,
            "shape_mode_definition": SHAPE_MODE_DEFINITION,
            "shape_mode_normalization": SHAPE_MODE_NORMALIZATION,
            "shape_mode_sign": float(SHAPE_MODE_SIGN),
            "basis_order": list(BASIS_MODE_NAMES),
            "force_frame": "rotating_local",
            "angle_conventions": (
                "phi is the absolute fixed-frame heel-to-toe chord angle "
                "(counterclockwise positive); varphi = phi - phi_ref is the rigid "
                "rotation relative to the reference mesh; alpha = tan(theta) is the "
                "top shape amplitude with positive theta bending the toe up"
            ),
            "phi_ref": float(phi_ref),
            "contact_coordinate": (
                "candidate_l is the material/reference coordinate l_i (NaN for full "
                "contact); the rotated coordinate x_contact_rot = l_i + d_lx is only "
                "known at runtime"
            ),
            "contact_topologies": list(CONTACT_TYPE_NAMES),
            "n_heel": n_heel,
            "n_toe": n_toe,
            "n_full": n_full,
            "heel_node_sets": "contact {0..i}, free {i+1..N_b-1}, i = 0..N_b-2",
            "toe_node_sets": "contact {i..N_b-1}, free {0..i-1}, i = 1..N_b-1",
            "full_node_sets": "contact {0..N_b-1}, free {}",
            "transition_node_in_contact": True,
            "full_contact_anchor": "L/2",
            "corner_reaction_layout": CORNER_REACTION_LAYOUT,
            "edge_field_semantics": (
                "edge_contact_reaction_* is at node i; edge_free_gap_* is at node i+1 "
                "for heel contact and node i-1 for toe contact; both are NaN for full "
                "contact, which has no free edge"
            ),
            "full_contact_validity": (
                "corner-only: valid when the fixed-frame normal reactions at x=0 and "
                "x=L are both >= -tau_R; interior reactions stay diagnostic"
            ),
            "strain_model": (
                "Intentional infinitesimal strain and linear elasticity with exact "
                "finite-rotation boundary displacement (true sin/cos of varphi). Fast "
                "but not objective for locally large rotations."
            ),
            **plate_metadata,
        },
        y_top=prepared.y_top,
        y_bottom=prepared.y_bottom,
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
        contact_type_codes=type_codes,
        edge_node_ids=indices.copy(),
        anchor_reference_x=anchor_x,
        edge_contact_node_ids=edge_contact_ids,
        edge_free_node_ids=edge_free_ids,
        has_free_edge=has_free_edge,
        contact_mask=contact_mask,
        corner_reactions_local=corner_reactions_local,
        heel_corner_node_id=HEEL_CORNER_NODE_ID,
        toe_corner_node_id=toe_corner_node_id(n_b),
        full_contact_anchor_x=FULL_CONTACT_ANCHOR_FRACTION * float(prepared.L),
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
        plate_u_local_basis=plate_u_local_basis,
        plate_v_local_basis=plate_v_local_basis,
        plate_rotation_local_basis=plate_rotation_local_basis,
        plate_constraint_multiplier_basis=plate_constraint_multiplier_basis,
        plate_axial_force_basis=plate_axial_force_basis,
        plate_bp_residual=plate_bp_residual,
        plate_metadata=plate_metadata,
    )



def save_contact_lookup(result: ContactLookupResult, output_dir: Path | str) -> Path:
    """Save NPZ, CSV, and JSON artifacts for a contact lookup table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "contact_lookup.npz"

    payload = {
        "candidate_indices": result.candidate_indices,
        "candidate_l": result.candidate_l,
        "n_free": result.n_free,
        "n_contact": result.n_contact,
        "status": np.asarray(result.status, dtype=object),
        "x_top": result.x_top,
        "x_bottom": result.x_bottom,
        "basis_top_displacements": result.basis_top_displacements,
        "scalar_lookup": result.scalar_lookup,
        "gap_basis": result.gap_basis,
        "reaction_basis": result.reaction_basis,
        "gap_u_basis": result.gap_u_basis if result.gap_u_basis is not None else np.array([]),
        "gap_v_basis": result.gap_v_basis if result.gap_v_basis is not None else result.gap_basis,
        "reaction_x_basis": (
            result.reaction_x_basis if result.reaction_x_basis is not None else np.array([])
        ),
        "reaction_y_basis": (
            result.reaction_y_basis if result.reaction_y_basis is not None else result.reaction_basis
        ),
        "top_force_x_basis": (
            result.top_force_x_basis if result.top_force_x_basis is not None else np.array([])
        ),
        "top_force_y_basis": (
            result.top_force_y_basis if result.top_force_y_basis is not None else np.array([])
        ),
        "top_force_basis": (
            result.top_force_y_basis
            if result.top_force_y_basis is not None
            else (result.top_force_basis if result.top_force_basis is not None else np.array([]))
        ),
        "kf_matrix": result.kf_matrix if result.kf_matrix is not None else np.array([]),
        "kf_svals": result.kf_svals if result.kf_svals is not None else np.array([]),
        "kf_cond": result.kf_cond if result.kf_cond is not None else np.array([]),
        "kf_det": result.kf_det if result.kf_det is not None else np.array([]),
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
        "include_endpoints": result.include_endpoints,
        "schema_version": result.schema_version,
        "geometry_type": np.asarray(result.geometry_type),
        "dof_ordering": np.asarray(result.dof_ordering),
        "n_top_nodes": result.n_top_nodes,
        "n_bottom_nodes": result.n_bottom_nodes,
        "moment_origin": np.asarray(result.moment_origin, dtype=float),
        "solver_xr": result.solver_xr,
        "solver_yr": result.solver_yr,
        "phi_ref": result.phi_ref,
        "basis_mode_names": np.asarray(BASIS_MODE_NAMES, dtype=object),
        "scalar_lookup_fields": np.asarray(SCALAR_LOOKUP_FIELDS, dtype=object),
        "contact_type_codes": np.asarray(result.contact_type_codes, dtype=int),
        "contact_type_names": np.asarray(CONTACT_TYPE_NAMES, dtype=object),
        "edge_node_ids": np.asarray(result.edge_node_ids, dtype=int),
        "anchor_reference_x": np.asarray(result.anchor_reference_x, dtype=float),
        "edge_contact_node_ids": np.asarray(result.edge_contact_node_ids, dtype=int),
        "edge_free_node_ids": np.asarray(result.edge_free_node_ids, dtype=int),
        "has_free_edge": np.asarray(result.has_free_edge, dtype=bool),
        "contact_mask": np.asarray(result.contact_mask, dtype=bool),
        "corner_reactions_local": np.asarray(result.corner_reactions_local, dtype=float),
        "corner_names": np.asarray(CORNER_NAMES, dtype=object),
        "corner_component_names": np.asarray(CORNER_COMPONENT_NAMES, dtype=object),
        "heel_corner_node_id": int(result.heel_corner_node_id),
        "toe_corner_node_id": int(result.toe_corner_node_id),
        "full_contact_anchor_x": float(result.full_contact_anchor_x),
        "has_plate_response": bool(result.has_plate_response),
        "EI_plate": float(result.EI_plate),
    }
    for key in ("nx", "ny", "order"):
        value = result.metadata.get(key)
        if value is not None:
            payload[key] = int(value)
    if result.y_top is not None:
        payload["y_top"] = result.y_top
    if result.y_bottom is not None:
        payload["y_bottom"] = result.y_bottom
    if result.x_plate is not None:
        payload["x_plate"] = result.x_plate
    if result.y_plate is not None:
        payload["y_plate"] = result.y_plate
    if result.has_plate_response:
        assert result.plate_u_local_basis is not None
        payload.update(
            {
                "plate_node_ids": np.asarray(result.plate_node_ids, dtype=int),
                "plate_reference_x": np.asarray(result.plate_reference_x, dtype=float),
                "plate_reference_y": np.asarray(result.plate_reference_y, dtype=float),
                "plate_reference_arc_length": np.asarray(
                    result.plate_reference_arc_length, dtype=float
                ),
                "plate_element_connectivity": np.asarray(
                    result.plate_element_connectivity, dtype=int
                ),
                "plate_element_lengths": np.asarray(result.plate_element_lengths, dtype=float),
                "plate_element_tangents": np.asarray(result.plate_element_tangents, dtype=float),
                "plate_element_normals": np.asarray(result.plate_element_normals, dtype=float),
                "plate_u_dof_ids": np.asarray(result.plate_u_dof_ids, dtype=int),
                "plate_v_dof_ids": np.asarray(result.plate_v_dof_ids, dtype=int),
                "plate_rotation_dof_ids": np.asarray(result.plate_rotation_dof_ids, dtype=int),
                "plate_constraint_ids": np.asarray(result.plate_constraint_ids, dtype=int),
                "plate_tangent": np.asarray(result.plate_tangent, dtype=float),
                "plate_normal": np.asarray(result.plate_normal, dtype=float),
                "plate_u_local_basis": np.asarray(result.plate_u_local_basis, dtype=float),
                "plate_v_local_basis": np.asarray(result.plate_v_local_basis, dtype=float),
                "plate_rotation_local_basis": np.asarray(
                    result.plate_rotation_local_basis, dtype=float
                ),
                "plate_constraint_multiplier_basis": np.asarray(
                    result.plate_constraint_multiplier_basis, dtype=float
                ),
                "plate_axial_force_basis": np.asarray(result.plate_axial_force_basis, dtype=float),
                "plate_bp_residual": np.asarray(result.plate_bp_residual, dtype=float),
            }
        )
    np.savez_compressed(npz_path, **payload)

    csv_path = output_dir / "contact_lookup.csv"
    fieldnames = [
        "contact_type",
        "edge_node_id",
        "l",
        "anchor_reference_x",
        "edge_contact_node_id",
        "edge_free_node_id",
        "has_free_edge",
        "n_free",
        "n_contact",
        "status",
        "kf_cond",
        "kf_det",
        "kf_smin",
        "condition_estimate",
        "solve_residual",
        "top_displacement_residual",
        "contact_displacement_residual",
        "force_equilibrium_residual",
        "moment_equilibrium_residual",
    ]
    for mode in BASIS_MODE_NAMES:
        for name in SCALAR_LOOKUP_FIELDS:
            fieldnames.append(f"{name}_{mode}")
    for mode in BASIS_MODE_NAMES:
        for corner in CORNER_NAMES:
            for comp in CORNER_COMPONENT_NAMES:
                fieldnames.append(f"corner_R{comp}_{corner}_{mode}")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(len(result.candidate_indices)):
            s = result.scalar_lookup[row]
            record = {
                "contact_type": result.contact_type(row).value,
                "edge_node_id": int(result.edge_node_ids[row]),
                "l": float(result.candidate_l[row]),
                "anchor_reference_x": float(result.anchor_reference_x[row]),
                "edge_contact_node_id": int(result.edge_contact_node_ids[row]),
                "edge_free_node_id": int(result.edge_free_node_ids[row]),
                "has_free_edge": bool(result.has_free_edge[row]),
                "n_free": int(result.n_free[row]),
                "n_contact": int(result.n_contact[row]),
                "status": result.status[row],
                "kf_cond": (
                    float(result.kf_cond[row]) if result.kf_cond is not None else float("nan")
                ),
                "kf_det": (
                    float(result.kf_det[row]) if result.kf_det is not None else float("nan")
                ),
                "kf_smin": (
                    float(result.kf_svals[row, -1])
                    if result.kf_svals is not None
                    else float("nan")
                ),
                "condition_estimate": float(result.condition_estimates[row]),
                "solve_residual": float(result.solve_residuals[row]),
                "top_displacement_residual": float(result.top_displacement_residuals[row]),
                "contact_displacement_residual": float(result.contact_displacement_residuals[row]),
                "force_equilibrium_residual": float(result.force_equilibrium_residuals[row]),
                "moment_equilibrium_residual": float(result.moment_equilibrium_residuals[row]),
            }
            for k, mode in enumerate(BASIS_MODE_NAMES):
                for j, name in enumerate(SCALAR_LOOKUP_FIELDS):
                    record[f"{name}_{mode}"] = float(s[k, j])
            corners = result.corner_reactions_local[row]
            for k, mode in enumerate(BASIS_MODE_NAMES):
                for c, corner in enumerate(CORNER_NAMES):
                    for j, comp in enumerate(CORNER_COMPONENT_NAMES):
                        record[f"corner_R{comp}_{corner}_{mode}"] = float(corners[k, c, j])
            writer.writerow(record)

    meta = {
        "schema_version": result.schema_version,
        "conventions": {
            "dof_ordering": result.dof_ordering,
            "positive_gap": "upward open gap (positive bottom vertical displacement)",
            "positive_reaction": "upward compressive reaction from the ground (nodal force)",
            "positive_Fy": "upward top resultant force (compression typically negative)",
            "positive_Fx": "rightward top resultant force",
            "edge_node_in_contact": True,
            "Mv": "x_t^T f_{t,y} (vertical center-of-effort moment)",
            "Mz": "moment of top nodal forces about (0,0): x^T fy - y^T fx",
            "toe_moment": "rho_a^T f_{t,y} - eta_a^T f_{t,x} about (a, H_a)",
            "solver_moment_reference": f"xr = {result.solver_xr}, yr = {result.solver_yr}",
            "contact": (
                "perfect stick on the record's contact interval ([0,l] heel, [l,L] toe, "
                "[0,L] full); the rotating-frame contact displacement is "
                "d_c^T(x) = d_a^T + (Q(varphi)^T - I)(X - X_a) with X_a the record "
                "anchor, spanned by the four contact-motion basis modes"
            ),
            "bilateral_vs_unilateral": (
                "Lookup solves a bilateral support interval; unilateral admissibility "
                "is checked after the force-controlled 2x2 solve using fixed-frame "
                "normal gaps g^F and fixed-frame normal reactions R_n^F only."
            ),
            "scalar_lookup_axis": list(SCALAR_LOOKUP_FIELDS),
            "basis": list(BASIS_MODE_NAMES),
            "kf_matrix": "[[F_x,Bx, F_x,By], [F_y,Bx, F_y,By]] maps (d_ax, d_ay) to F^T",
            "runtime_coefficients": (
                "gamma = [tan(theta), d_ax, d_ay, cos(varphi) - 1, -sin(varphi)], where "
                "(d_ax, d_ay) is the contact-edge translation d_l for heel/toe contact "
                "and the anchor translation d_a for full contact"
            ),
            "contact_topology": (
                "explicit per record in contact_type_codes (0=heel, 1=toe, 2=full); "
                "never inferred from l"
            ),
            "node_sets": (
                "contact_mask[record, node] is the dense contact membership; "
                "contact_node_ids = flatnonzero(mask), free_node_ids = flatnonzero(~mask)"
            ),
            "corner_reactions": CORNER_REACTION_LAYOUT,
            "limitation": (
                "On large layered saddles, stacked-C reciprocity may sit near 1e-7 "
                "(symmetrized if below --reciprocity-tol). Horizontal/vertical coupling "
                "does not imply Fx*=0 ⇒ d_ax=0."
            ),
        },
        "contact_records": {
            "n_heel": int(
                np.count_nonzero(np.asarray(result.contact_type_codes) == code_of(ContactType.HEEL))
            ),
            "n_toe": int(
                np.count_nonzero(np.asarray(result.contact_type_codes) == code_of(ContactType.TOE))
            ),
            "n_full": int(
                np.count_nonzero(np.asarray(result.contact_type_codes) == code_of(ContactType.FULL))
            ),
            "n_records": int(result.n_records),
            "expected": record_counts(int(result.n_bottom_nodes)),
            "order": "heel (i=0..N_b-2), toe (i=1..N_b-1), full (once)",
        },
        "full_contact_anchor_x": float(result.full_contact_anchor_x),
        "heel_corner_node_id": int(result.heel_corner_node_id),
        "toe_corner_node_id": int(result.toe_corner_node_id),
        "n_shape_modes": 1,
        "shape_mode_definition": SHAPE_MODE_DEFINITION,
        "shape_mode_normalization": SHAPE_MODE_NORMALIZATION,
        "shape_mode_sign": float(SHAPE_MODE_SIGN),
        "basis_order": list(BASIS_MODE_NAMES),
        "force_frame": "rotating_local",
        "phi_ref": result.phi_ref,
        "softplus_a": result.softplus_a,
        "softplus_kappa": result.softplus_kappa,
        "L": result.L,
        "H": result.H,
        "E": result.E,
        "nu": result.nu,
        "reciprocity_error": result.reciprocity_error,
        "include_endpoints": result.include_endpoints,
        "metadata": result.metadata,
        "geometry_type": result.geometry_type,
        "n_top_nodes": result.n_top_nodes,
        "n_bottom_nodes": result.n_bottom_nodes,
        "solver_xr": result.solver_xr,
        "solver_yr": result.solver_yr,
        "moment_origin": list(result.moment_origin),
        "has_plate_response": bool(result.has_plate_response),
        "plate_model": result.plate_metadata.get("plate_model", PLATE_MODEL),
        "EI_plate": float(result.EI_plate),
        "plate_axial_model": result.plate_metadata.get("plate_axial_model", PLATE_AXIAL_MODEL),
        "plate_rotation_sign": result.plate_metadata.get(
            "plate_rotation_sign", PLATE_ROTATION_SIGN
        ),
        "plate_normal_sign": result.plate_metadata.get("plate_normal_sign", PLATE_NORMAL_SIGN),
        "constraint_row_normalization": result.plate_metadata.get(
            "constraint_row_normalization", CONSTRAINT_ROW_NORMALIZATION
        ),
        "n_plate_nodes": (
            int(result.plate_node_ids.size)
            if result.has_plate_response and result.plate_node_ids is not None
            else 0
        ),
        "plate_metadata": result.plate_metadata,
    }
    with (output_dir / "contact_lookup_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return npz_path


def _require_lookup_schema_v6(
    schema_version: int,
    scalar: np.ndarray,
    files: list[str],
) -> None:
    if int(schema_version) != LOOKUP_SCHEMA_VERSION:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    if scalar.ndim != 3 or scalar.shape[1] != N_BASIS_MODES or scalar.shape[2] != N_SCALAR_FIELDS:
        raise ValueError(REGENERATE_LOOKUP_MESSAGE)
    required = (
        "contact_type_codes",
        "edge_node_ids",
        "anchor_reference_x",
        "contact_mask",
        "corner_reactions_local",
        "has_free_edge",
    )
    missing = [key for key in required if key not in files]
    if missing:
        raise ValueError(f"{REGENERATE_LOOKUP_MESSAGE} Missing topology keys: {missing}.")


def _plate_arrays_present(files: list[str]) -> bool:
    return all(key in files for key in PLATE_RESPONSE_ARRAY_KEYS)


def load_contact_lookup(path: Path | str) -> ContactLookupResult:
    """Reload a contact lookup table from NPZ (schema v6 only)."""
    data = np.load(path, allow_pickle=True)
    status = [str(s) for s in data["status"].tolist()]
    scalar = np.asarray(data["scalar_lookup"], dtype=float)
    schema_version = int(data["schema_version"]) if "schema_version" in data.files else 1
    _require_lookup_schema_v6(schema_version, scalar, list(data.files))

    def _arr(name: str) -> np.ndarray | None:
        if name not in data.files:
            return None
        arr = np.asarray(data[name], dtype=float)
        return arr if arr.size else None

    def _int_arr(name: str) -> np.ndarray | None:
        if name not in data.files:
            return None
        arr = np.asarray(data[name], dtype=int)
        return arr if arr.size else None

    gap_v = _arr("gap_v_basis")
    if gap_v is None:
        gap_v = np.asarray(data["gap_basis"], dtype=float)
    reaction_y = _arr("reaction_y_basis")
    if reaction_y is None:
        reaction_y = np.asarray(data["reaction_basis"], dtype=float)
    top_fy = _arr("top_force_y_basis")
    if top_fy is None:
        top_fy = _arr("top_force_basis")
    moment_origin = MOMENT_ORIGIN
    if "moment_origin" in data.files:
        mo = np.asarray(data["moment_origin"], dtype=float).reshape(-1)
        if mo.size >= 2:
            moment_origin = (float(mo[0]), float(mo[1]))

    basis_order = (
        [str(s) for s in data["basis_mode_names"].tolist()]
        if "basis_mode_names" in data.files
        else list(BASIS_MODE_NAMES)
    )
    if tuple(basis_order) != BASIS_MODE_NAMES:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Saved basis order {basis_order} does not match "
            f"{list(BASIS_MODE_NAMES)}."
        )
    scalar_fields = (
        [str(s) for s in data["scalar_lookup_fields"].tolist()]
        if "scalar_lookup_fields" in data.files
        else list(SCALAR_LOOKUP_FIELDS)
    )
    if tuple(scalar_fields) != SCALAR_LOOKUP_FIELDS:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Saved scalar fields {scalar_fields} do not match "
            f"{list(SCALAR_LOOKUP_FIELDS)}."
        )

    type_codes = np.asarray(data["contact_type_codes"], dtype=int)
    n_full = int(np.count_nonzero(type_codes == code_of(ContactType.FULL)))
    if n_full != 1:
        raise ValueError(
            f"{REGENERATE_LOOKUP_MESSAGE} Found {n_full} full-contact records; expected exactly one."
        )
    n_bottom_loaded = int(len(np.asarray(data["x_bottom"], dtype=float)))

    has_plate = _plate_arrays_present(list(data.files))
    if "has_plate_response" in data.files:
        flagged = bool(np.asarray(data["has_plate_response"]).reshape(-1)[0])
        if flagged and not has_plate:
            raise ValueError(
                f"{REGENERATE_LOOKUP_MESSAGE} has_plate_response=True but plate arrays are missing."
            )
        has_plate = flagged and has_plate

    plate_metadata: dict = {"has_plate_response": False}
    ei_plate = float(data["EI_plate"]) if "EI_plate" in data.files else float("nan")
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

    return ContactLookupResult(
        candidate_indices=np.asarray(data["candidate_indices"], dtype=int),
        candidate_l=np.asarray(data["candidate_l"], dtype=float),
        n_free=np.asarray(data["n_free"], dtype=int),
        n_contact=np.asarray(data["n_contact"], dtype=int),
        status=status,
        x_top=np.asarray(data["x_top"], dtype=float),
        x_bottom=np.asarray(data["x_bottom"], dtype=float),
        basis_top_displacements=np.asarray(data["basis_top_displacements"], dtype=float),
        scalar_lookup=scalar,
        gap_basis=gap_v,
        reaction_basis=reaction_y,
        softplus_a=float(data["softplus_a"]),
        softplus_kappa=float(data["softplus_kappa"]),
        L=float(data["L"]),
        H=float(data["H"]),
        E=float(data["E"]),
        nu=float(data["nu"]),
        condition_estimates=np.asarray(data["condition_estimates"], dtype=float),
        solve_residuals=np.asarray(data["solve_residuals"], dtype=float),
        top_displacement_residuals=np.asarray(data["top_displacement_residuals"], dtype=float),
        contact_displacement_residuals=np.asarray(
            data["contact_displacement_residuals"], dtype=float
        ),
        force_equilibrium_residuals=np.asarray(data["force_equilibrium_residuals"], dtype=float),
        moment_equilibrium_residuals=np.asarray(data["moment_equilibrium_residuals"], dtype=float),
        reciprocity_error=float(data["reciprocity_error"]),
        include_endpoints=bool(data["include_endpoints"]),
        schema_version=schema_version,
        metadata={
            "basis_order": basis_order,
            "scalar_lookup_axis": scalar_fields,
            "n_shape_modes": 1,
            "shape_mode_definition": SHAPE_MODE_DEFINITION,
            "shape_mode_normalization": SHAPE_MODE_NORMALIZATION,
            "force_frame": "rotating_local",
            "phi_ref": float(data["phi_ref"]) if "phi_ref" in data.files else 0.0,
            "nx": int(data["nx"]) if "nx" in data.files else None,
            "ny": int(data["ny"]) if "ny" in data.files else None,
            "order": int(data["order"]) if "order" in data.files else None,
            "contact_topologies": list(CONTACT_TYPE_NAMES),
            "n_heel": int(np.count_nonzero(type_codes == code_of(ContactType.HEEL))),
            "n_toe": int(np.count_nonzero(type_codes == code_of(ContactType.TOE))),
            "n_full": int(np.count_nonzero(type_codes == code_of(ContactType.FULL))),
            "corner_reaction_layout": CORNER_REACTION_LAYOUT,
            "full_contact_anchor": "L/2",
            **plate_metadata,
        },
        top_force_basis=top_fy,
        top_force_x_basis=_arr("top_force_x_basis"),
        top_force_y_basis=top_fy,
        gap_u_basis=_arr("gap_u_basis"),
        gap_v_basis=gap_v,
        reaction_x_basis=_arr("reaction_x_basis"),
        reaction_y_basis=reaction_y,
        kf_matrix=_arr("kf_matrix"),
        kf_svals=_arr("kf_svals"),
        kf_cond=_arr("kf_cond"),
        kf_det=_arr("kf_det"),
        y_top=np.asarray(data["y_top"], dtype=float) if "y_top" in data.files else None,
        y_bottom=np.asarray(data["y_bottom"], dtype=float) if "y_bottom" in data.files else None,
        x_plate=np.asarray(data["x_plate"], dtype=float) if "x_plate" in data.files else None,
        y_plate=np.asarray(data["y_plate"], dtype=float) if "y_plate" in data.files else None,
        geometry_type=(
            _decode_flag(data["geometry_type"]) if "geometry_type" in data.files else "rectangle"
        ),
        dof_ordering=(
            _decode_flag(data["dof_ordering"])
            if "dof_ordering" in data.files
            else DOF_ORDERING_COMPONENT_MAJOR_UV
        ),
        n_top_nodes=int(data["n_top_nodes"]) if "n_top_nodes" in data.files else int(len(data["x_top"])),
        n_bottom_nodes=(
            int(data["n_bottom_nodes"]) if "n_bottom_nodes" in data.files else int(len(data["x_bottom"]))
        ),
        moment_origin=moment_origin,
        solver_xr=float(data["solver_xr"]) if "solver_xr" in data.files else 0.5 * float(data["L"]),
        solver_yr=float(data["solver_yr"]) if "solver_yr" in data.files else SOLVER_YR,
        phi_ref=float(data["phi_ref"]) if "phi_ref" in data.files else 0.0,
        contact_type_codes=type_codes,
        edge_node_ids=np.asarray(data["edge_node_ids"], dtype=int),
        anchor_reference_x=np.asarray(data["anchor_reference_x"], dtype=float),
        edge_contact_node_ids=np.asarray(data["edge_contact_node_ids"], dtype=int),
        edge_free_node_ids=np.asarray(data["edge_free_node_ids"], dtype=int),
        has_free_edge=np.asarray(data["has_free_edge"], dtype=bool),
        contact_mask=np.asarray(data["contact_mask"], dtype=bool),
        corner_reactions_local=np.asarray(data["corner_reactions_local"], dtype=float),
        heel_corner_node_id=(
            int(data["heel_corner_node_id"])
            if "heel_corner_node_id" in data.files
            else HEEL_CORNER_NODE_ID
        ),
        toe_corner_node_id=(
            int(data["toe_corner_node_id"])
            if "toe_corner_node_id" in data.files
            else toe_corner_node_id(n_bottom_loaded)
        ),
        full_contact_anchor_x=(
            float(data["full_contact_anchor_x"])
            if "full_contact_anchor_x" in data.files
            else FULL_CONTACT_ANCHOR_FRACTION * float(data["L"])
        ),
        has_plate_response=has_plate,
        plate_node_ids=_int_arr("plate_node_ids") if has_plate else None,
        plate_reference_x=_arr("plate_reference_x") if has_plate else None,
        plate_reference_y=_arr("plate_reference_y") if has_plate else None,
        plate_reference_arc_length=_arr("plate_reference_arc_length") if has_plate else None,
        plate_element_connectivity=_int_arr("plate_element_connectivity") if has_plate else None,
        plate_element_lengths=_arr("plate_element_lengths") if has_plate else None,
        plate_element_tangents=_arr("plate_element_tangents") if has_plate else None,
        plate_element_normals=_arr("plate_element_normals") if has_plate else None,
        plate_u_dof_ids=_int_arr("plate_u_dof_ids") if has_plate else None,
        plate_v_dof_ids=_int_arr("plate_v_dof_ids") if has_plate else None,
        plate_rotation_dof_ids=_int_arr("plate_rotation_dof_ids") if has_plate else None,
        plate_constraint_ids=_int_arr("plate_constraint_ids") if has_plate else None,
        plate_tangent=_arr("plate_tangent") if has_plate else None,
        plate_normal=_arr("plate_normal") if has_plate else None,
        EI_plate=ei_plate if has_plate else float("nan"),
        plate_u_local_basis=_arr("plate_u_local_basis") if has_plate else None,
        plate_v_local_basis=_arr("plate_v_local_basis") if has_plate else None,
        plate_rotation_local_basis=_arr("plate_rotation_local_basis") if has_plate else None,
        plate_constraint_multiplier_basis=(
            _arr("plate_constraint_multiplier_basis") if has_plate else None
        ),
        plate_axial_force_basis=_arr("plate_axial_force_basis") if has_plate else None,
        plate_bp_residual=_arr("plate_bp_residual") if has_plate else None,
        plate_metadata=plate_metadata,
    )


if __name__ == "__main__":
    from compliance_fem.contact_lookup_cli import main

    main()
