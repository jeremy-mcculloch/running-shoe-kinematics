"""Image-derived (measured) sole geometry: normalized CSV loader and SI topology.

Normalized CSV convention
-------------------------
Columns ``record_type, name, index, x_over_length, y_over_length``:

* ``curve`` rows are samples of ``top_surface``, ``bottom_surface`` and
  ``foam_interface`` with consecutive integer ``index`` from 0, each ordered heel
  to toe with strictly increasing ``x``;
* two ``plate`` rows, ``heel_end`` and ``toe_end``, give the plate extent as
  ``x_over_length`` only (``index`` and ``y_over_length`` empty); each end lies
  on the foam interface at that ``x``.

Coordinates are in the project frame: ``+x`` heel to toe, ``+y`` up, both axes
divided by the projected heel-bottom-to-toe-tip length. The bottom surface
starts at the origin (heel bottom) and ends at the toe tip, ``x = 1``, where the
top surface also ends. The heel top is the first top sample, and the foam
interface endpoints are its first and last samples. Nothing here flips an axis,
reverses a curve, or rescales ``y`` on its own.

SI conversion
-------------
``x_m = (shoe_length_mm / 1000) x_over_length`` and the same factor for ``y``.
``shoe_length_mm`` is the projected heel-to-toe length, not the outsole arc length.

Topology
--------
The closed exterior boundary is, counterclockwise: the bottom surface
(heel_bottom -> toe_tip), the top surface traversed toe -> heel
(toe_tip -> heel_top), and the straight heel edge (heel_top -> heel_bottom).
The top and bottom surfaces meet at the single point toe ``toe_tip``. The foam
interface endpoints are projected onto the exterior boundary and inserted there;
the interface splits the sole into the upper and lower foam regions. The plate
endpoints are inserted as interface vertices, splitting the interface into heel
foam-only, plate, and toe foam-only segments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compliance_fem.contact.sole_config import validate_shoe_length_mm

REQUIRED_COLUMNS = ("record_type", "name", "index", "x_over_length", "y_over_length")
REQUIRED_CURVES = ("top_surface", "bottom_surface", "foam_interface")
PLATE_ENDS = ("heel_end", "toe_end")

COORDINATE_SYSTEM = "heel_to_toe_x_up_y"
NORMALIZATION = "projected_heel_bottom_to_toe_x_span"
SHARED_TOE_POLICY = "bottom_contact_owns_toe_vertex"

TAG_TOP = "TOP_SURFACE"
TAG_BOTTOM = "BOTTOM_SURFACE"
TAG_HEEL = "HEEL_EDGE"
TAG_TOE = "TOE_TIP"
TAG_INTERFACE = "FOAM_INTERFACE"
TAG_PLATE = "PLATE_SEGMENT"
BOUNDARY_TAGS = (TAG_TOP, TAG_BOTTOM, TAG_HEEL, TAG_TOE, TAG_INTERFACE, TAG_PLATE)
REGION_UPPER = "upper_foam"
REGION_LOWER = "lower_foam"
FOAM_REGION_TAGS = (REGION_UPPER, REGION_LOWER)

# Normalized-coordinate tolerance for the origin / toe-tip conventions.
NORMALIZED_ENDPOINT_TOL = 1.0e-6
# Strict monotonicity of x along each curve (normalized units).
NORMALIZED_MONOTONE_TOL = 1.0e-9


class GeometryFileError(ValueError):
    """Malformed or inconsistent normalized geometry file."""


@dataclass(frozen=True)
class NormalizedSoleGeometry:
    """Dimensionless heel-to-toe curves and plate extent, exactly as supplied."""

    curves: dict[str, np.ndarray]
    plate_x: tuple[float, float]
    source_filename: str = ""

    def to_payload(self) -> dict:
        return {
            "curves": {k: np.asarray(v, dtype=float).tolist() for k, v in self.curves.items()},
            "plate_x": [float(v) for v in self.plate_x],
            "source_filename": self.source_filename,
        }


def _parse_float(text: str | None, where: str) -> float:
    if text is None:
        raise GeometryFileError(f"{where}: coordinate {text!r} is not a number.")
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise GeometryFileError(f"{where}: coordinate {text!r} is not a number.") from exc
    if not np.isfinite(value):
        raise GeometryFileError(f"{where}: coordinate {text!r} is not finite.")
    return value


def load_normalized_sole_csv(path: str | Path) -> NormalizedSoleGeometry:
    """Parse a normalized sole CSV by semantic record names and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Normalized sole geometry CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = tuple(h.strip() for h in (reader.fieldnames or ()))
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise GeometryFileError(f"{path.name}: missing columns {missing}; found {list(header)}.")
        plate: dict[str, float] = {}
        raw_curves: dict[str, dict[int, tuple[float, float]]] = {}
        for line_no, raw in enumerate(reader, start=2):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            kind = (row.get("record_type") or "").lower()
            name = row.get("name") or ""
            where = f"{path.name}:{line_no} ({kind} {name})"
            if kind == "plate":
                if name not in PLATE_ENDS:
                    raise GeometryFileError(f"{where}: plate row name must be one of {PLATE_ENDS}.")
                if name in plate:
                    raise GeometryFileError(f"{where}: duplicate plate row {name!r}.")
                plate[name] = _parse_float(row.get("x_over_length"), where)
            elif kind == "curve":
                if name not in REQUIRED_CURVES:
                    raise GeometryFileError(f"{where}: unknown curve {name!r}; expected one of {REQUIRED_CURVES}.")
                index_text = row.get("index")
                try:
                    index = int(str(index_text))
                except (TypeError, ValueError) as exc:
                    raise GeometryFileError(f"{where}: curve index {index_text!r} is not an integer.") from exc
                bucket = raw_curves.setdefault(name, {})
                if index in bucket:
                    raise GeometryFileError(f"{where}: duplicate index {index} on curve {name!r}.")
                bucket[index] = (
                    _parse_float(row.get("x_over_length"), where),
                    _parse_float(row.get("y_over_length"), where),
                )
            elif kind == "point":
                raise GeometryFileError(
                    f"{where}: landmark 'point' rows are no longer used; endpoints come from the curves. "
                    "Give the plate extent as 'plate,heel_end,,x,' and 'plate,toe_end,,x,' rows."
                )
            else:
                raise GeometryFileError(f"{where}: unknown record_type {kind!r}.")

    missing_plate = [x for x in PLATE_ENDS if x not in plate]
    if missing_plate:
        raise GeometryFileError(f"{path.name}: missing plate rows {missing_plate}.")
    missing_curves = [x for x in REQUIRED_CURVES if x not in raw_curves]
    if missing_curves:
        raise GeometryFileError(f"{path.name}: missing curves {missing_curves}.")

    curves: dict[str, np.ndarray] = {}
    for name in REQUIRED_CURVES:
        bucket = raw_curves[name]
        indices = sorted(bucket)
        if indices != list(range(len(indices))):
            raise GeometryFileError(
                f"{path.name}: curve {name!r} indices must be consecutive from 0; got "
                f"{indices[:5]}...{indices[-3:]}."
            )
        curves[name] = np.array([bucket[i] for i in indices], dtype=float)

    geom = NormalizedSoleGeometry(
        curves=curves,
        plate_x=(plate["heel_end"], plate["toe_end"]),
        source_filename=path.name,
    )
    validate_normalized_geometry(geom)
    return geom


def validate_normalized_geometry(geom: NormalizedSoleGeometry) -> None:
    """Check sampling, heel-to-toe ordering, the normalization and the plate extent."""

    # Check that the curves are well-formed and strictly heel-to-toe ordered in x.
    tol = NORMALIZED_ENDPOINT_TOL
    for name in REQUIRED_CURVES:
        c = np.asarray(geom.curves[name], dtype=float)
        if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 2:
            raise GeometryFileError(f"Curve {name!r} must have at least two (x, y) samples.")
        if not np.all(np.isfinite(c)):
            raise GeometryFileError(f"Curve {name!r} has nonfinite coordinates.")
        dx = np.diff(c[:, 0])
        if np.any(dx <= NORMALIZED_MONOTONE_TOL):
            k = int(np.argmax(dx <= NORMALIZED_MONOTONE_TOL))
            raise GeometryFileError(
                f"Curve {name!r} is not strictly heel-to-toe ordered in x at samples {k}->{k + 1} "
                f"(dx={dx[k]:.3e}); duplicate or reversed samples are rejected."
            )
    top = np.asarray(geom.curves["top_surface"], dtype=float)
    bottom = np.asarray(geom.curves["bottom_surface"], dtype=float)
    iface = np.asarray(geom.curves["foam_interface"], dtype=float)
    # Check that the curves are normalized
    if float(np.max(np.abs(bottom[0]))) > tol:
        raise GeometryFileError(f"bottom_surface must start at the origin (0, 0); got {bottom[0].tolist()}.")
    if abs(float(bottom[-1, 0]) - 1.0) > tol:
        raise GeometryFileError(
            f"bottom_surface must end at the toe tip x_over_length = 1 ({NORMALIZATION}); got {bottom[-1, 0]}."
        )
    if float(np.max(np.abs(top[-1] - bottom[-1]))) > tol:
        raise GeometryFileError(
            f"top_surface and bottom_surface must end at the same toe tip; got {top[-1].tolist()} "
            f"and {bottom[-1].tolist()}."
        )

    # Check that the plate does not extend beyond the interface endpoints
    heel_x, toe_x = geom.plate_x
    if not iface[0, 0] < heel_x < toe_x < iface[-1, 0]:
        raise GeometryFileError(
            f"Plate ends x = ({heel_x}, {toe_x}) must satisfy interface start {iface[0, 0]} < heel < toe "
            f"< interface end {iface[-1, 0]}."
        )


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------


def polygon_signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polyline_length(poly: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(np.asarray(poly, dtype=float), axis=0).T)))


def closest_point_on_polyline(poly: np.ndarray, point: np.ndarray) -> tuple[int, float, np.ndarray, float]:
    """Return ``(segment, t, projection, distance)`` of the closest polyline point."""
    P = np.asarray(poly, dtype=float)
    q = np.asarray(point, dtype=float).reshape(2)
    a = P[:-1]
    d = P[1:] - a
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", q - a, d) / np.where(dd > 0.0, dd, 1.0), 0.0, 1.0)
    proj = a + t[:, None] * d
    dist = np.hypot(*(proj - q).T)
    k = int(np.argmin(dist))
    return k, float(t[k]), proj[k], float(dist[k])


def distance_to_polyline(poly: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.array([closest_point_on_polyline(poly, q)[3] for q in pts], dtype=float)


def insert_point_on_polyline(
    poly: np.ndarray,
    point: np.ndarray,
    tol: float,
    *,
    use_projection: bool,
    coincide_tol: float,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Insert ``point`` (or its projection) as a polyline vertex.

    Returns ``(new_poly, vertex_index, distance, inserted_xy)``. A point already
    within ``coincide_tol`` of an existing vertex reuses that vertex; otherwise
    nothing is snapped to an existing sample.
    """
    P = np.asarray(poly, dtype=float)
    seg, t, proj, dist = closest_point_on_polyline(P, point)
    if dist > tol:
        raise GeometryFileError(
            f"Point {np.asarray(point).tolist()} lies {dist:.3e} m from the curve (tolerance {tol:.3e} m)."
        )
    target = proj if use_projection else np.asarray(point, dtype=float).reshape(2)
    vertex_d = np.hypot(*(P - target).T)
    k = int(np.argmin(vertex_d))
    if vertex_d[k] <= coincide_tol:
        return P.copy(), k, dist, P[k].copy()
    # Placed inside segment ``seg`` and away from its vertices, so the order is kept.
    return np.insert(P, seg + 1, target, axis=0), seg + 1, dist, target.copy()


def turning_angles(poly: np.ndarray) -> np.ndarray:
    """Signed turning angle at each interior vertex of an open polyline (rad)."""
    P = np.asarray(poly, dtype=float)
    out = np.zeros(len(P))
    if len(P) < 3:
        return out
    d1 = P[1:-1] - P[:-2]
    d2 = P[2:] - P[1:-1]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dot = np.einsum("ij,ij->i", d1, d2)
    out[1:-1] = np.arctan2(cross, dot)
    return out


# ---------------------------------------------------------------------------
# Physical geometry
# ---------------------------------------------------------------------------


@dataclass
class SoleGeometry:
    """SI sole geometry with inserted topology vertices and two foam regions.

    Curves are heel-to-toe ordered polylines in metres (``heel_edge`` runs
    heel_top -> heel_bottom). ``bottom`` / ``top`` / ``heel_edge`` contain the
    inserted foam-interface endpoints; ``interface`` contains the inserted plate
    endpoints at ``plate_heel_index`` and ``plate_toe_index``.
    """

    normalized: NormalizedSoleGeometry
    shoe_length_mm: float
    top: np.ndarray
    bottom: np.ndarray
    heel_edge: np.ndarray
    interface: np.ndarray
    plate_heel_index: int
    plate_toe_index: int
    region_polygons: dict[str, np.ndarray]
    corner_angles_deg: dict[str, float]

    @property
    def heel_bottom(self) -> np.ndarray:
        return self.bottom[0]

    @property
    def heel_top(self) -> np.ndarray:
        return self.top[0]

    @property
    def toe_tip(self) -> np.ndarray:
        return self.bottom[-1]

    @property
    def shoe_length_m(self) -> float:
        return float(self.shoe_length_mm) / 1000.0

    @property
    def plate_arc_length(self) -> float:
        return polyline_length(self.interface[self.plate_heel_index : self.plate_toe_index + 1])

    @property
    def region_areas(self) -> dict[str, float]:
        return {k: polygon_signed_area(v) for k, v in self.region_polygons.items()}

    @property
    def total_area(self) -> float:
        return sum(self.region_areas.values())


def _exterior_insert(
    curves: dict[str, np.ndarray], point: np.ndarray, tol: float, coincide: float, label: str
) -> np.ndarray:
    """Insert ``point`` projected onto the nearest exterior curve and return the inserted vertex."""
    name = min(curves, key=lambda n: closest_point_on_polyline(curves[n], point)[3])
    try:
        curves[name], _, _, xy = insert_point_on_polyline(
            curves[name], point, tol, use_projection=True, coincide_tol=coincide
        )
    except GeometryFileError as exc:
        raise GeometryFileError(f"{label}: {exc}") from exc
    return xy


def build_sole_geometry(
    normalized: NormalizedSoleGeometry,
    shoe_length_mm: float,
    landmark_tolerance: float = 2.0e-3,
) -> SoleGeometry:
    """Scale geometry from normalized coordinates to metres."""
    shoe_length_mm = validate_shoe_length_mm(shoe_length_mm)
    s = shoe_length_mm / 1000.0
    tol = float(landmark_tolerance) * s
    coincide = 1.0e-9 * s

    top = s * np.asarray(normalized.curves["top_surface"], dtype=float)
    bottom = s * np.asarray(normalized.curves["bottom_surface"], dtype=float)
    interface = s * np.asarray(normalized.curves["foam_interface"], dtype=float)
    top[-1] = bottom[-1]  # bit-identical shared toe vertex

    # Insert the interface endpoints into the exterior; the heel edge is built
    # heel_bottom -> heel_top here and stored heel_top -> heel_bottom.
    exterior_curves = {TAG_TOP: top, TAG_BOTTOM: bottom, TAG_HEEL: np.vstack([bottom[0], top[0]])}
    interface[0] = _exterior_insert(exterior_curves, interface[0], tol, coincide, "foam_interface heel end")
    interface[-1] = _exterior_insert(exterior_curves, interface[-1], tol, coincide, "foam_interface toe end")
    top, bottom = exterior_curves[TAG_TOP], exterior_curves[TAG_BOTTOM]
    heel_edge = exterior_curves[TAG_HEEL][::-1].copy()

    # Add the plate endpoints to the interface curve and record their indices.
    # Important to insert the heel end first so that the toe end index is not shifted by the insertion.
    plate_index = []
    for x in normalized.plate_x:
        point = np.array([s * x, np.interp(s * x, interface[:, 0], interface[:, 1])])
        interface, idx, _, _ = insert_point_on_polyline(interface, point, tol, use_projection=True, coincide_tol=coincide)
        plate_index.append(idx)
    plate_heel_index, plate_toe_index = plate_index

    # Verify that that top is above interface is above bottom everywhere
    def y_at(curve: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.interp(x, curve[:, 0], curve[:, 1])
    if np.any(y_at(top, bottom[1:-1, 0]) <= bottom[1:-1, 1]) or np.any(top[:-1, 1] <= y_at(bottom, top[:-1, 0])):
        raise GeometryFileError("top_surface must lie above bottom_surface everywhere heel-side of the toe tip.")
    inner = interface[1:-1]
    if np.any(inner[:, 1] <= y_at(bottom, inner[:, 0])) or np.any(inner[:, 1] >= y_at(top, inner[:, 0])):
        raise GeometryFileError("Foam-interface samples must lie strictly between bottom_surface and top_surface.")

    # Exterior loop (counterclockwise): bottom, top reversed, heel edge
    exterior = np.vstack([bottom, top[::-1][1:], heel_edge[1:-1]])
    area = polygon_signed_area(exterior)
    if area <= 0.0:
        raise GeometryFileError(
            f"Exterior boundary bottom -> top -> heel edge has nonpositive signed area {area:.3e}; "
            "the curves are not in the heel-to-toe x / up y convention."
        )

    def loop_index(p: np.ndarray) -> int:
        idx = np.flatnonzero(np.all(exterior == p, axis=1))
        if idx.size != 1:
            raise GeometryFileError(f"Vertex {p.tolist()} is not uniquely on the exterior loop.")
        return int(idx[0])
    # Compute the interior angles at the three exterior corners (heel_bottom, toe_tip, heel_top) for diagnostics purposes.
    n_ext = len(exterior)
    corners: dict[str, float] = {}
    for label, p in (("heel_bottom", bottom[0]), ("toe_tip", bottom[-1]), ("heel_top", top[0])):
        k = loop_index(p)
        a, b, c = exterior[(k - 1) % n_ext], exterior[k], exterior[(k + 1) % n_ext]
        d1, d2 = b - a, c - b
        turn = float(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2)))
        if turn <= 0.0:
            raise GeometryFileError(f"Exterior corner {label!r} is not convex (turn {np.degrees(turn):.2f} deg).")
        corners[label] = float(180.0 - np.degrees(turn))

    # Create curves that define the upper and lower foam regions, and check that they have positive area.
    def arc(i0: int, i1: int) -> np.ndarray:
        return exterior[(np.arange((i1 - i0) % n_ext + 1) + i0) % n_ext]

    idx_heel_interface, index_toe_interface = loop_index(interface[0]), loop_index(interface[-1])
    regions = {
        REGION_LOWER: np.vstack([arc(idx_heel_interface, index_toe_interface), interface[::-1][1:-1]]),
        REGION_UPPER: np.vstack([arc(index_toe_interface, idx_heel_interface), interface[1:-1]]),
    }
    region_area = {k: polygon_signed_area(v) for k, v in regions.items()}
    if min(region_area.values()) <= 0.0 or abs(sum(region_area.values()) - area) > 1e-9 * area:
        raise GeometryFileError(f"Foam interface does not split the sole into two valid regions {region_area}.")

    return SoleGeometry(
        normalized=normalized,
        shoe_length_mm=float(shoe_length_mm),
        top=top,
        bottom=bottom,
        heel_edge=heel_edge,
        interface=interface,
        plate_heel_index=plate_heel_index,
        plate_toe_index=plate_toe_index,
        region_polygons=regions,
        corner_angles_deg=corners,
    )


def geometry_metadata(geom: SoleGeometry, config=None) -> dict:
    """JSON-serializable audit record of the measured geometry."""
    meta = {
        "source_geometry_filename": geom.normalized.source_filename,
        "shoe_length_mm": float(geom.shoe_length_mm),
        "shoe_length_m": float(geom.shoe_length_m),
        "shoe_length_definition": "projected heel_bottom -> toe_tip x distance (not outsole arc length)",
        "coordinate_system": COORDINATE_SYSTEM,
        "normalization": NORMALIZATION,
        "si_conversion": "x_m = (shoe_length_mm/1000) x_over_length; y_m = (shoe_length_mm/1000) y_over_length",
        "plate_x_over_length": [float(v) for v in geom.normalized.plate_x],
        "physical_landmarks": {
            "heel_bottom": geom.heel_bottom.tolist(),
            "heel_top": geom.heel_top.tolist(),
            "toe_tip": geom.toe_tip.tolist(),
            "plate_heel_endpoint": geom.interface[geom.plate_heel_index].tolist(),
            "plate_toe_endpoint": geom.interface[geom.plate_toe_index].tolist(),
        },
        "top_surface_reference_coordinates": np.asarray(geom.top, dtype=float).tolist(),
        "bottom_surface_reference_coordinates": np.asarray(geom.bottom, dtype=float).tolist(),
        "heel_edge_reference_coordinates": np.asarray(geom.heel_edge, dtype=float).tolist(),
        "foam_interface_reference_coordinates": np.asarray(geom.interface, dtype=float).tolist(),
        "plate_start_reference_coordinate": geom.interface[geom.plate_heel_index].tolist(),
        "plate_end_reference_coordinate": geom.interface[geom.plate_toe_index].tolist(),
        "plate_arc_length_m": geom.plate_arc_length,
        "foam_region_tags": list(FOAM_REGION_TAGS),
        "boundary_tags": list(BOUNDARY_TAGS),
        "shared_toe_policy": SHARED_TOE_POLICY,
        "corner_interior_angles_deg": dict(geom.corner_angles_deg),
        "total_area_m2": geom.total_area,
        "region_areas_m2": geom.region_areas,
        "exterior_corners": "heel_bottom, toe_tip (point toe), heel_top; all convex",
    }
    if config is not None:
        meta.update(
            {
                "upper_foam_material": config.upper_foam_material,
                "lower_foam_material": config.lower_foam_material,
                "materials": {
                    name: {"E_heel": m.E_heel, "E_toe": m.E_toe, "nu": m.nu}
                    for name, m in (
                        (n, config.material(n)) for n in {config.upper_foam_material, config.lower_foam_material}
                    )
                },
                "continuum_model": "plane_strain",
                "EI_plate_Nm2_per_m": float(config.EI_plate_Nm2_per_m),
            }
        )
    return meta
