"""Image-derived (measured) sole geometry: normalized CSV loader and SI topology.

Normalized CSV convention
-------------------------
Columns ``record_type, name, index, x_over_length, y_over_length``. Rows with
``record_type = point`` are landmarks (``index`` empty); rows with
``record_type = curve`` are samples of a named curve with consecutive integer
``index`` starting at 0. Coordinates are already in the project frame:

* ``+x`` heel to toe, ``+y`` up, heel-bottom landmark at ``(0, 0)``;
* both axes divided by the same projected heel-bottom-to-toe-tip pixel length,
  so the toe tip has ``x/L = 1``;
* every curve is ordered heel to toe.

Nothing here flips an axis, reverses a curve, or rescales ``y`` on its own.
Rows are found by ``(record_type, name)``, never by row number; names outside
the required set (for example historical corner labels) are ignored and not
interpreted.

SI conversion
-------------
``x_m = (shoe_length_mm / 1000) x_over_length`` and the same factor for ``y``.
``shoe_length_mm`` is the projected heel-to-toe length, not the outsole arc length.

Topology
--------
The closed exterior boundary is, counterclockwise: the measured bottom surface
(heel_bottom -> toe_tip), the measured top surface traversed toe -> heel
(toe_tip -> heel_top), and the straight heel edge (heel_top -> heel_bottom).
The top and bottom surfaces meet at the single point toe ``toe_tip``. The foam
interface is an internal material boundary whose two endpoints lie on the
exterior boundary; it splits the sole into the upper and lower foam regions.
The plate endpoints are projected onto the interface polyline and inserted as
vertices, splitting the interface into heel foam-only, plate, and toe foam-only
segments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from compliance_fem.config import validate_shoe_length_mm

REQUIRED_COLUMNS = ("record_type", "name", "index", "x_over_length", "y_over_length")
REQUIRED_LANDMARKS = (
    "toe_tip",
    "heel_top",
    "heel_bottom",
    "foam_interface_toe_endpoint",
    "foam_interface_heel_endpoint",
    "plate_toe_endpoint",
    "plate_heel_endpoint",
)
REQUIRED_CURVES = ("top_surface", "bottom_surface", "foam_interface")
CURVE_ENDPOINTS = {
    "top_surface": ("heel_top", "toe_tip"),
    "bottom_surface": ("heel_bottom", "toe_tip"),
    "foam_interface": ("foam_interface_heel_endpoint", "foam_interface_toe_endpoint"),
}

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

# Normalized-coordinate tolerance for exact landmark/curve-endpoint agreement.
NORMALIZED_ENDPOINT_TOL = 1.0e-6
# Strict monotonicity of x along each curve (normalized units).
NORMALIZED_MONOTONE_TOL = 1.0e-9


class GeometryFileError(ValueError):
    """Malformed or inconsistent normalized geometry file."""


@dataclass(frozen=True)
class NormalizedSoleGeometry:
    """Dimensionless landmarks and heel-to-toe curves, exactly as supplied."""

    landmarks: dict[str, tuple[float, float]]
    curves: dict[str, np.ndarray]
    source_filename: str = ""
    ignored_names: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        return {
            "landmarks": {k: [float(v[0]), float(v[1])] for k, v in self.landmarks.items()},
            "curves": {k: np.asarray(v, dtype=float).tolist() for k, v in self.curves.items()},
            "source_filename": self.source_filename,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> NormalizedSoleGeometry:
        geom = cls(
            landmarks={k: (float(v[0]), float(v[1])) for k, v in payload["landmarks"].items()},
            curves={k: np.asarray(v, dtype=float).reshape(-1, 2) for k, v in payload["curves"].items()},
            source_filename=str(payload.get("source_filename", "")),
        )
        validate_normalized_geometry(geom)
        return geom


def _parse_float(text: str, where: str) -> float:
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
        landmarks: dict[str, tuple[float, float]] = {}
        samples: dict[str, dict[int, tuple[float, float]]] = {}
        ignored: set[str] = set()
        for line_no, raw in enumerate(reader, start=2):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            kind = (row.get("record_type") or "").lower()
            name = row.get("name") or ""
            where = f"{path.name}:{line_no} ({kind} {name})"
            if kind == "point":
                if name not in REQUIRED_LANDMARKS:
                    ignored.add(name)
                    continue
                if name in landmarks:
                    raise GeometryFileError(f"{where}: duplicate landmark {name!r}.")
                landmarks[name] = (
                    _parse_float(row.get("x_over_length"), where),
                    _parse_float(row.get("y_over_length"), where),
                )
            elif kind == "curve":
                if name not in REQUIRED_CURVES:
                    ignored.add(name)
                    continue
                index_text = row.get("index")
                try:
                    index = int(str(index_text))
                except (TypeError, ValueError) as exc:
                    raise GeometryFileError(f"{where}: curve index {index_text!r} is not an integer.") from exc
                bucket = samples.setdefault(name, {})
                if index in bucket:
                    raise GeometryFileError(f"{where}: duplicate index {index} on curve {name!r}.")
                bucket[index] = (
                    _parse_float(row.get("x_over_length"), where),
                    _parse_float(row.get("y_over_length"), where),
                )
            else:
                raise GeometryFileError(f"{where}: unknown record_type {kind!r}.")

    missing_landmarks = [n for n in REQUIRED_LANDMARKS if n not in landmarks]
    if missing_landmarks:
        raise GeometryFileError(f"{path.name}: missing landmarks {missing_landmarks}.")
    missing_curves = [n for n in REQUIRED_CURVES if n not in samples]
    if missing_curves:
        raise GeometryFileError(f"{path.name}: missing curves {missing_curves}.")

    curves: dict[str, np.ndarray] = {}
    for name in REQUIRED_CURVES:
        bucket = samples[name]
        indices = sorted(bucket)
        if indices != list(range(len(indices))):
            raise GeometryFileError(
                f"{path.name}: curve {name!r} indices must be consecutive from 0; got "
                f"{indices[:5]}...{indices[-3:]}."
            )
        curves[name] = np.array([bucket[i] for i in indices], dtype=float)

    geom = NormalizedSoleGeometry(
        landmarks=landmarks,
        curves=curves,
        source_filename=path.name,
        ignored_names=tuple(sorted(ignored)),
    )
    validate_normalized_geometry(geom)
    return geom


def validate_normalized_geometry(geom: NormalizedSoleGeometry) -> None:
    """Check ordering, endpoints, and the normalization convention."""
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
        start, end = CURVE_ENDPOINTS[name]
        for label, idx in ((start, 0), (end, -1)):
            p = np.asarray(geom.landmarks[label], dtype=float)
            if float(np.max(np.abs(c[idx] - p))) > tol:
                raise GeometryFileError(
                    f"Curve {name!r} {'first' if idx == 0 else 'last'} sample {c[idx].tolist()} does "
                    f"not match landmark {label!r} {p.tolist()}."
                )
    heel_bottom = np.asarray(geom.landmarks["heel_bottom"], dtype=float)
    if float(np.max(np.abs(heel_bottom))) > tol:
        raise GeometryFileError(f"heel_bottom must be the origin (0, 0); got {heel_bottom.tolist()}.")
    toe = np.asarray(geom.landmarks["toe_tip"], dtype=float)
    if abs(float(toe[0]) - 1.0) > tol:
        raise GeometryFileError(
            f"toe_tip must have x_over_length = 1 ({NORMALIZATION}); got {toe[0]}."
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
    new = np.insert(P, seg + 1, target, axis=0)
    if np.any(np.diff(new[:, 0]) <= 0.0):
        raise GeometryFileError(
            f"Inserting {target.tolist()} breaks heel-to-toe ordering of the curve."
        )
    return new, seg + 1, dist, target.copy()


def _segments_intersect(p1, p2, q1, q2, eps: float) -> bool:
    """Proper or touching intersection of two closed segments (within eps)."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1, o2 = orient(p1, p2, q1), orient(p1, p2, q2)
    o3, o4 = orient(q1, q2, p1), orient(q1, q2, p2)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return False


def polyline_self_intersections(poly: np.ndarray, closed: bool, eps: float) -> list[tuple[int, int]]:
    """Index pairs of non-adjacent segments that properly intersect."""
    P = np.asarray(poly, dtype=float)
    segs = list(zip(P[:-1], P[1:]))
    if closed:
        segs.append((P[-1], P[0]))
    n = len(segs)
    hits: list[tuple[int, int]] = []
    lo = np.array([np.minimum(a, b) for a, b in segs])
    hi = np.array([np.maximum(a, b) for a, b in segs])
    for i in range(n):
        cand = np.flatnonzero(
            (lo[:, 0] <= hi[i, 0]) & (hi[:, 0] >= lo[i, 0]) & (lo[:, 1] <= hi[i, 1]) & (hi[:, 1] >= lo[i, 1])
        )
        for j in cand:
            if j <= i + 1 or (closed and i == 0 and j == n - 1):
                continue
            if _segments_intersect(*segs[i], *segs[j], eps):
                hits.append((i, int(j)))
    return hits


def crossing_pairs(a: np.ndarray, b: np.ndarray, eps: float) -> list[tuple[int, int]]:
    """Proper crossings between segments of open polylines ``a`` and ``b``."""
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    out = []
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            if _segments_intersect(A[i], A[i + 1], B[j], B[j + 1], eps):
                out.append((i, j))
    return out


def point_in_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test."""
    P = np.asarray(poly, dtype=float)
    Q = np.atleast_2d(np.asarray(pts, dtype=float))
    x, y = Q[:, 0][:, None], Q[:, 1][:, None]
    x1, y1 = P[:, 0][None, :], P[:, 1][None, :]
    x2, y2 = np.roll(P[:, 0], -1)[None, :], np.roll(P[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.sum(cond & (x < xin), axis=1) % 2 == 1


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
class MeasuredSoleGeometry:
    """SI sole geometry with inserted topology landmarks and two foam regions.

    Curves are heel-to-toe ordered polylines in metres. ``bottom`` / ``top`` /
    ``heel_edge`` contain the inserted foam-interface endpoints; ``interface``
    contains the inserted plate endpoints at ``plate_heel_index`` and
    ``plate_toe_index``.
    """

    normalized: NormalizedSoleGeometry
    shoe_length_mm: float
    tolerance_m: float
    landmarks: dict[str, np.ndarray]
    top: np.ndarray
    bottom: np.ndarray
    heel_edge: np.ndarray
    interface: np.ndarray
    plate_heel_index: int
    plate_toe_index: int
    interface_endpoint_curves: dict[str, str]
    projection_distances: dict[str, float]
    exterior: np.ndarray
    exterior_tags: list[str]
    region_polygons: dict[str, np.ndarray]
    corner_angles_deg: dict[str, float]
    diagnostics: dict = field(default_factory=dict)

    @property
    def scale(self) -> float:
        return float(self.shoe_length_mm) / 1000.0

    @property
    def shoe_length_m(self) -> float:
        return self.scale

    @property
    def plate(self) -> np.ndarray:
        return self.interface[self.plate_heel_index : self.plate_toe_index + 1]

    @property
    def plate_arc_length(self) -> float:
        return polyline_length(self.plate)

    @property
    def total_area(self) -> float:
        return polygon_signed_area(self.exterior)

    @property
    def region_areas(self) -> dict[str, float]:
        return {k: polygon_signed_area(v) for k, v in self.region_polygons.items()}


def build_measured_geometry(
    normalized: NormalizedSoleGeometry,
    shoe_length_mm: float,
    landmark_tolerance: float = 2.0e-3,
) -> MeasuredSoleGeometry:
    """Scale to SI with one common factor and construct the validated topology."""
    validate_normalized_geometry(normalized)
    shoe_length_mm = validate_shoe_length_mm(shoe_length_mm)
    s = shoe_length_mm / 1000.0
    tol = float(landmark_tolerance) * s
    coincide = 1.0e-9 * s
    eps = 1.0e-14 * s * s

    landmarks = {k: s * np.asarray(v, dtype=float) for k, v in normalized.landmarks.items()}
    top = s * np.asarray(normalized.curves["top_surface"], dtype=float)
    bottom = s * np.asarray(normalized.curves["bottom_surface"], dtype=float)
    interface = s * np.asarray(normalized.curves["foam_interface"], dtype=float)
    heel_edge = np.vstack([landmarks["heel_top"], landmarks["heel_bottom"]])
    # Snap shared vertices to bit-identical copies of their landmarks.
    top[0], top[-1] = landmarks["heel_top"], landmarks["toe_tip"]
    bottom[0], bottom[-1] = landmarks["heel_bottom"], landmarks["toe_tip"]
    interface[0] = landmarks["foam_interface_heel_endpoint"]
    interface[-1] = landmarks["foam_interface_toe_endpoint"]

    distances: dict[str, float] = {}
    endpoint_curves: dict[str, str] = {}
    curves = {TAG_TOP: top, TAG_BOTTOM: bottom, TAG_HEEL: heel_edge}
    for label in ("foam_interface_heel_endpoint", "foam_interface_toe_endpoint"):
        p = landmarks[label]
        best = min(
            ((name, closest_point_on_polyline(poly, p)[3]) for name, poly in curves.items()),
            key=lambda item: item[1],
        )
        name, dist = best
        if dist > tol:
            raise GeometryFileError(
                f"{label} lies {dist:.3e} m from the exterior boundary (tolerance {tol:.3e} m)."
            )
        # Project onto the exterior so the supplied boundary curves (and the straight
        # heel edge) stay exact; the interface endpoint moves by ``dist`` <= tol.
        poly = curves[name]
        if name == TAG_HEEL:
            # Ordering is heel_top -> heel_bottom (x decreasing), so insert by segment.
            seg, _, proj, _ = closest_point_on_polyline(poly, p)
            hit = np.flatnonzero(np.hypot(*(poly - proj).T) <= coincide)
            if hit.size:
                proj = poly[int(hit[0])].copy()
            else:
                poly = np.insert(poly, seg + 1, proj, axis=0)
        else:
            poly, _, _, proj = insert_point_on_polyline(poly, p, tol, use_projection=True, coincide_tol=coincide)
        curves[name] = poly
        endpoint_curves[label] = name
        distances[label] = float(dist)
        landmarks[f"{label}_projected"] = np.asarray(proj, dtype=float).copy()
    top, bottom, heel_edge = curves[TAG_TOP], curves[TAG_BOTTOM], curves[TAG_HEEL]
    interface[0] = landmarks["foam_interface_heel_endpoint_projected"]
    interface[-1] = landmarks["foam_interface_toe_endpoint_projected"]

    projected: dict[str, np.ndarray] = {}
    for label in ("plate_heel_endpoint", "plate_toe_endpoint"):
        interface, _, dist, xy = insert_point_on_polyline(
            interface, landmarks[label], tol, use_projection=True, coincide_tol=coincide
        )
        distances[label] = float(dist)
        projected[label] = xy
    ih = int(np.flatnonzero(np.all(interface == projected["plate_heel_endpoint"], axis=1))[0])
    it = int(np.flatnonzero(np.all(interface == projected["plate_toe_endpoint"], axis=1))[0])
    if ih >= it:
        raise GeometryFileError("plate_heel_endpoint must lie heel-side of plate_toe_endpoint.")
    landmarks["plate_heel_endpoint_projected"] = projected["plate_heel_endpoint"]
    landmarks["plate_toe_endpoint_projected"] = projected["plate_toe_endpoint"]

    # Exterior loop (counterclockwise): bottom, top reversed, heel edge interior.
    exterior = np.vstack([bottom, top[::-1][1:], heel_edge[1:-1]])
    tags = (
        [TAG_BOTTOM] * (len(bottom) - 1)
        + [TAG_TOP] * (len(top) - 1)
        + [TAG_HEEL] * (len(heel_edge) - 1)
    )
    area = polygon_signed_area(exterior)
    if area <= 0.0:
        raise GeometryFileError(
            f"Exterior boundary bottom -> top -> heel edge has nonpositive signed area {area:.3e}; "
            "the curves are not in the heel-to-toe x / up y convention."
        )
    hits = polyline_self_intersections(exterior, closed=True, eps=eps)
    if hits:
        raise GeometryFileError(f"Exterior boundary self-intersects at segment pairs {hits[:5]}.")

    def loop_index(p: np.ndarray) -> int:
        idx = np.flatnonzero(np.all(exterior == p, axis=1))
        if idx.size != 1:
            raise GeometryFileError(f"Vertex {p.tolist()} is not uniquely on the exterior loop.")
        return int(idx[0])

    n_ext = len(exterior)
    i_toe = loop_index(landmarks["toe_tip"])
    i_heel_top = loop_index(landmarks["heel_top"])
    i_heel_bot = loop_index(landmarks["heel_bottom"])
    corners: dict[str, float] = {}
    for label, k in (("heel_bottom", i_heel_bot), ("toe_tip", i_toe), ("heel_top", i_heel_top)):
        a, b, c = exterior[(k - 1) % n_ext], exterior[k], exterior[(k + 1) % n_ext]
        d1, d2 = b - a, c - b
        turn = float(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2)))
        if turn <= 0.0:
            raise GeometryFileError(f"Exterior corner {label!r} is not convex (turn {np.degrees(turn):.2f} deg).")
        corners[label] = float(180.0 - np.degrees(turn))

    # Interface: interior vertices strictly inside, no crossings with the exterior.
    inner = interface[1:-1]
    if inner.size and not np.all(point_in_polygon(exterior, inner)):
        bad = int(np.flatnonzero(~point_in_polygon(exterior, inner))[0]) + 1
        raise GeometryFileError(f"Foam-interface sample {bad} lies outside the sole.")
    ext_open = np.vstack([exterior, exterior[:1]])
    crossings = crossing_pairs(interface, ext_open, eps)
    if crossings:
        raise GeometryFileError(f"Foam interface crosses the exterior boundary at {crossings[:5]}.")
    if polyline_self_intersections(interface, closed=False, eps=eps):
        raise GeometryFileError("Foam interface self-intersects.")

    # Split the exterior at the interface endpoints into two counterclockwise regions.
    i_h = loop_index(interface[0])
    i_t = loop_index(interface[-1])

    def arc(i0: int, i1: int) -> tuple[np.ndarray, list[str]]:
        idx = [i0]
        k = i0
        while k != i1:
            k = (k + 1) % n_ext
            idx.append(k)
        edge_tags = [tags[idx[m]] for m in range(len(idx) - 1)]
        return exterior[idx], edge_tags

    arc_ht, tags_ht = arc(i_h, i_t)
    arc_th, tags_th = arc(i_t, i_h)
    poly_a = np.vstack([arc_ht, interface[::-1][1:-1]])  # region left of arc heel->toe
    poly_b = np.vstack([arc_th, interface[1:-1]])
    area_a, area_b = polygon_signed_area(poly_a), polygon_signed_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        raise GeometryFileError(
            f"Foam interface does not split the sole into two valid regions (areas {area_a:.3e}, {area_b:.3e})."
        )
    if abs(area_a + area_b - area) > 1e-9 * area:
        raise GeometryFileError("Foam-region areas do not sum to the sole area.")

    def tag_length(poly: np.ndarray, edge_tags: list[str], tag: str) -> float:
        seg = np.hypot(*np.diff(poly, axis=0).T)
        return float(sum(L for L, t in zip(seg, edge_tags) if t == tag))

    top_a, top_b = tag_length(arc_ht, tags_ht, TAG_TOP), tag_length(arc_th, tags_th, TAG_TOP)
    bot_a, bot_b = tag_length(arc_ht, tags_ht, TAG_BOTTOM), tag_length(arc_th, tags_th, TAG_BOTTOM)
    upper_is_b = top_b > top_a
    lower_is_a = bot_a > bot_b
    if upper_is_b != lower_is_a:
        raise GeometryFileError(
            "Cannot identify the upper (top-surface) and lower (bottom-surface) foam regions "
            "from the interface endpoints."
        )
    regions = (
        {REGION_UPPER: poly_b, REGION_LOWER: poly_a}
        if upper_is_b
        else {REGION_UPPER: poly_a, REGION_LOWER: poly_b}
    )

    return MeasuredSoleGeometry(
        normalized=normalized,
        shoe_length_mm=float(shoe_length_mm),
        tolerance_m=tol,
        landmarks=landmarks,
        top=top,
        bottom=bottom,
        heel_edge=heel_edge,
        interface=interface,
        plate_heel_index=ih,
        plate_toe_index=it,
        interface_endpoint_curves=endpoint_curves,
        projection_distances=distances,
        exterior=exterior,
        exterior_tags=tags,
        region_polygons=regions,
        corner_angles_deg=corners,
        diagnostics={
            "n_top_samples": int(len(normalized.curves["top_surface"])),
            "n_bottom_samples": int(len(normalized.curves["bottom_surface"])),
            "n_interface_samples": int(len(normalized.curves["foam_interface"])),
            "ignored_record_names": list(normalized.ignored_names),
        },
    )


def geometry_from_config(config) -> MeasuredSoleGeometry:
    return build_measured_geometry(
        config.normalized_geometry, config.shoe_length_mm, config.landmark_tolerance
    )


def geometry_metadata(geom: MeasuredSoleGeometry, config=None) -> dict:
    """JSON-serializable audit record of the measured geometry."""
    meta = {
        "geometry_type": "measured_sole",
        "source_geometry_filename": geom.normalized.source_filename,
        "shoe_length_mm": float(geom.shoe_length_mm),
        "shoe_length_m": float(geom.shoe_length_m),
        "shoe_length_definition": "projected heel_bottom -> toe_tip x distance (not outsole arc length)",
        "coordinate_system": COORDINATE_SYSTEM,
        "normalization": NORMALIZATION,
        "si_conversion": "x_m = (shoe_length_mm/1000) x_over_length; y_m = (shoe_length_mm/1000) y_over_length",
        "normalized_landmarks": {k: [float(v[0]), float(v[1])] for k, v in geom.normalized.landmarks.items()},
        "physical_landmarks": {k: [float(v[0]), float(v[1])] for k, v in geom.landmarks.items()},
        "top_surface_reference_coordinates": np.asarray(geom.top, dtype=float).tolist(),
        "bottom_surface_reference_coordinates": np.asarray(geom.bottom, dtype=float).tolist(),
        "heel_edge_reference_coordinates": np.asarray(geom.heel_edge, dtype=float).tolist(),
        "foam_interface_reference_coordinates": np.asarray(geom.interface, dtype=float).tolist(),
        "plate_reference_coordinates": np.asarray(geom.plate, dtype=float).tolist(),
        "plate_start_reference_coordinate": geom.interface[geom.plate_heel_index].tolist(),
        "plate_end_reference_coordinate": geom.interface[geom.plate_toe_index].tolist(),
        "plate_arc_length_m": geom.plate_arc_length,
        "projection_distances_m": dict(geom.projection_distances),
        "landmark_tolerance_m": float(geom.tolerance_m),
        "interface_endpoint_curves": dict(geom.interface_endpoint_curves),
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
                "EI_plate": float(config.EI_plate),
            }
        )
    return meta
