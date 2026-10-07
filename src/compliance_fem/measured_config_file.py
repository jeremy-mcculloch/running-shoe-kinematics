"""JSON configuration for the measured carbon-plated sole.

The normalized geometry CSV holds the shape only. Everything else that defines
a measured-sole model — shoe length, foam materials, plate stiffness, mesh,
softplus toe shape (``a``, ``kappa``), passive toe spring and runtime shoe
width — lives in one JSON file::

    {
      "schema_version": 1,
      "geometry":   {"csv": "../data/geometry/sole_geometry_normalized.csv", "shoe_length_mm": 270},
      "materials":  {"upper_foam": "FFTurbo", "lower_foam": "FFLeap",
                     "FFTurbo": {"E_Pa": 2.6e5, "nu": 0.113},
                     "FFLeap": {"E_heel_Pa": 3.54e5, "E_toe_Pa": 2.07e5, "nu": 0.113},
                     "plate": {"EI_Nm2_per_m": 2.0}},
      "mesh":       {"size_mm": 3.0, ...},
      "lookup":     {"a_over_length": 0.78, "kappa": 160, "output_dir": "../outputs/measured_sole_lookup"},
      "toe_spring": {"stiffness_Nm_per_rad": 25.0, ...},
      "runtime":    {"shoe_width_m": 0.10}
    }

Only ``geometry.csv`` and ``geometry.shoe_length_mm`` are required; every
other key falls back to the documented default. Unknown keys are rejected so a
typo cannot silently fall back to a default. Relative paths resolve against
the directory containing the JSON file.

Lookups built through :func:`build_measured_lookup` store the full setup and a
fingerprint of the build-relevant parameters in their metadata, so
:class:`compliance_fem.api.SoleModel` can detect a stale lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from compliance_fem.config import (
    FOAM_FFLEAP,
    FOAM_FFTURBO,
    FOAM_MATERIALS,
    MeasuredSoleConfig,
)
from compliance_fem.toe_spring import ToeSpringConfig

CONFIG_SCHEMA_VERSION = 1
DEFAULT_A_OVER_LENGTH = 0.78
DEFAULT_KAPPA = 160.0
DEFAULT_RECIPROCITY_TOL = 1.0e-6
DEFAULT_SHOE_WIDTH_M = 0.10
SETUP_METADATA_KEY = "measured_setup"
FINGERPRINT_METADATA_KEY = "measured_setup_fingerprint"


class ConfigFileError(ValueError):
    """Malformed measured-sole configuration file."""


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LookupSettings:
    """Contact-lookup generation settings. Exactly one of ``a_over_length`` /
    ``a_m`` locates the softplus toe transition."""

    a_over_length: float | None = DEFAULT_A_OVER_LENGTH
    a_m: float | None = None
    kappa: float = DEFAULT_KAPPA
    reciprocity_tol: float = DEFAULT_RECIPROCITY_TOL
    output_dir: str | None = None
    store_nodal_fields: bool = False

    def __post_init__(self) -> None:
        if (self.a_over_length is None) == (self.a_m is None):
            raise ConfigFileError("lookup: give exactly one of 'a_over_length' or 'a_m'.")
        if self.a_over_length is not None and not 0.0 < self.a_over_length < 1.0:
            raise ConfigFileError("lookup.a_over_length must lie in (0, 1).")
        if self.a_m is not None and not self.a_m > 0.0:
            raise ConfigFileError("lookup.a_m must be positive.")
        if not self.kappa > 0.0:
            raise ConfigFileError("lookup.kappa must be positive.")
        if not self.reciprocity_tol > 0.0:
            raise ConfigFileError("lookup.reciprocity_tol must be positive.")

    def a(self, L: float) -> float:
        a = float(self.a_m) if self.a_m is not None else float(self.a_over_length) * float(L)
        if not 0.0 < a < float(L):
            raise ConfigFileError(f"Softplus location a={a:.6g} m must lie inside (0, L={L:.6g} m).")
        return a


@dataclass(frozen=True)
class RuntimeSettings:
    shoe_width_m: float = DEFAULT_SHOE_WIDTH_M

    def __post_init__(self) -> None:
        if not (math.isfinite(self.shoe_width_m) and self.shoe_width_m > 0.0):
            raise ConfigFileError("runtime.shoe_width_m must be positive.")


@dataclass(frozen=True)
class MeasuredSoleSetup:
    """Everything needed to build and run a measured-sole model."""

    sole: MeasuredSoleConfig
    lookup: LookupSettings
    toe_spring: ToeSpringConfig
    runtime: RuntimeSettings
    source_path: str | None = None

    @property
    def a(self) -> float:
        return self.lookup.a(self.sole.L)

    @property
    def kappa(self) -> float:
        return float(self.lookup.kappa)

    @property
    def output_dir(self) -> Path | None:
        return None if self.lookup.output_dir is None else Path(self.lookup.output_dir)

    def fingerprint(self) -> str:
        return lookup_build_fingerprint(self.sole, self.a, self.kappa, self.lookup.reciprocity_tol)

    def to_dict(self) -> dict:
        """Canonical JSON dict (all values filled in; paths as resolved absolute strings)."""
        s, lk, t = self.sole, self.lookup, self.toe_spring
        lookup: dict = {"kappa": lk.kappa, "reciprocity_tol": lk.reciprocity_tol}
        if lk.a_m is not None:
            lookup["a_m"] = lk.a_m
        else:
            lookup["a_over_length"] = lk.a_over_length
        if lk.output_dir is not None:
            lookup["output_dir"] = str(lk.output_dir)
        lookup["store_nodal_fields"] = bool(lk.store_nodal_fields)
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "geometry": {
                "csv": None if s.geometry_csv is None else str(s.geometry_csv),
                "shoe_length_mm": float(s.shoe_length_mm),
                "landmark_tolerance": float(s.landmark_tolerance),
            },
            "materials": {
                "upper_foam": s.upper_foam_material,
                "lower_foam": s.lower_foam_material,
                FOAM_FFTURBO: {"E_Pa": float(s.ffturbo_E), "nu": float(s.ffturbo_nu)},
                FOAM_FFLEAP: {
                    "E_heel_Pa": float(s.ffleap_E_heel),
                    "E_toe_Pa": float(s.ffleap_E_toe),
                    "nu": float(s.ffleap_nu),
                },
                "plate": {"EI_Nm2_per_m": float(s.EI_plate)},
            },
            "mesh": {
                "size_mm": float(s.mesh_size) * 1000.0,
                "toe_refinement": float(s.toe_refinement),
                "heel_corner_refinement": float(s.heel_corner_refinement),
                "interface_refinement": float(s.interface_refinement),
                "plate_end_refinement": float(s.plate_end_refinement),
                "curvature_max_turn_deg": float(s.curvature_max_turn_deg),
                "min_angle_deg": float(s.min_angle_deg),
                "element_order": int(s.element_order),
            },
            "lookup": lookup,
            "toe_spring": {
                "model": t.toe_model,
                "stiffness_Nm_per_rad": float(t.toe_stiffness_Nm_per_rad),
                "neutral_angle_rad": float(t.toe_neutral_angle_rad),
                "damping_Nms_per_rad": float(t.toe_damping_Nms_per_rad),
                "angle_min_deg": float(t.toe_angle_min_deg),
                "angle_max_deg": float(t.toe_angle_max_deg),
                "low_force_threshold_N": float(t.toe_low_force_threshold_N),
                "equilibrium_abs_tol_Nm": float(t.toe_equilibrium_abs_tol_Nm),
                "equilibrium_rel_tol": float(t.toe_equilibrium_rel_tol),
                "root_scan_points": int(t.toe_root_scan_points),
            },
            "runtime": {"shoe_width_m": float(self.runtime.shoe_width_m)},
        }


# ---------------------------------------------------------------------------
# Key tables: JSON key -> (dataclass field, scale to SI)
# ---------------------------------------------------------------------------

_GEOMETRY_KEYS = {"csv", "shoe_length_mm", "landmark_tolerance"}
_MATERIAL_KEYS = {"upper_foam", "lower_foam", FOAM_FFTURBO, FOAM_FFLEAP, "plate"}
_FFTURBO_KEYS = {"E_Pa": "ffturbo_E", "nu": "ffturbo_nu"}
_FFLEAP_KEYS = {"E_heel_Pa": "ffleap_E_heel", "E_toe_Pa": "ffleap_E_toe", "nu": "ffleap_nu"}
_PLATE_KEYS = {"EI_Nm2_per_m": "EI_plate"}
_MESH_KEYS = {
    "size_mm": ("mesh_size", 1.0e-3),
    "toe_refinement": ("toe_refinement", 1.0),
    "heel_corner_refinement": ("heel_corner_refinement", 1.0),
    "interface_refinement": ("interface_refinement", 1.0),
    "plate_end_refinement": ("plate_end_refinement", 1.0),
    "curvature_max_turn_deg": ("curvature_max_turn_deg", 1.0),
    "min_angle_deg": ("min_angle_deg", 1.0),
    "element_order": ("element_order", None),
}
_LOOKUP_KEYS = {"a_over_length", "a_m", "kappa", "reciprocity_tol", "output_dir", "store_nodal_fields"}
_TOE_KEYS = {
    "model": "toe_model",
    "stiffness_Nm_per_rad": "toe_stiffness_Nm_per_rad",
    "neutral_angle_rad": "toe_neutral_angle_rad",
    "damping_Nms_per_rad": "toe_damping_Nms_per_rad",
    "angle_min_deg": "toe_angle_min_deg",
    "angle_max_deg": "toe_angle_max_deg",
    "low_force_threshold_N": "toe_low_force_threshold_N",
    "equilibrium_abs_tol_Nm": "toe_equilibrium_abs_tol_Nm",
    "equilibrium_rel_tol": "toe_equilibrium_rel_tol",
    "root_scan_points": "toe_root_scan_points",
}
_RUNTIME_KEYS = {"shoe_width_m"}
_TOP_KEYS = {"schema_version", "geometry", "materials", "mesh", "lookup", "toe_spring", "runtime", "description"}


def _section(data: dict, name: str, allowed, where: str) -> dict:
    sec = data.get(name, {})
    if sec is None:
        sec = {}
    if not isinstance(sec, dict):
        raise ConfigFileError(f"{where}{name} must be an object, got {type(sec).__name__}.")
    unknown = sorted(set(sec) - set(allowed))
    if unknown:
        raise ConfigFileError(f"{where}{name}: unknown key(s) {unknown}; allowed {sorted(allowed)}.")
    return sec


def _number(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigFileError(f"{key} must be a number, got {value!r}.")
    out = float(value)
    if not math.isfinite(out):
        raise ConfigFileError(f"{key} must be finite, got {value!r}.")
    return out


def _integer(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
        raise ConfigFileError(f"{key} must be an integer, got {value!r}.")
    return int(value)


def _resolve(path_text, base: Path | None, key: str) -> str:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ConfigFileError(f"{key} must be a non-empty path string.")
    p = Path(path_text).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return str(p.resolve())


# ---------------------------------------------------------------------------
# Parse / load / save
# ---------------------------------------------------------------------------


def setup_from_dict(data: dict, *, base_dir: Path | str | None = None, source_path: str | None = None) -> MeasuredSoleSetup:
    """Validate a config dict and build the setup. Relative paths resolve against ``base_dir``."""
    if not isinstance(data, dict):
        raise ConfigFileError("Configuration must be a JSON object.")
    base = None if base_dir is None else Path(base_dir).resolve()
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise ConfigFileError(f"Unknown top-level key(s) {unknown}; allowed {sorted(_TOP_KEYS)}.")
    version = data.get("schema_version", CONFIG_SCHEMA_VERSION)
    if _integer(version, "schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigFileError(
            f"Unsupported config schema_version={version}; this code reads schema_version={CONFIG_SCHEMA_VERSION}."
        )

    geo = _section(data, "geometry", _GEOMETRY_KEYS, "")
    for req in ("csv", "shoe_length_mm"):
        if req not in geo:
            raise ConfigFileError(f"geometry.{req} is required.")
    sole_kwargs: dict = {
        "geometry_csv": _resolve(geo["csv"], base, "geometry.csv"),
        "shoe_length_mm": _number(geo["shoe_length_mm"], "geometry.shoe_length_mm"),
    }
    if "landmark_tolerance" in geo:
        sole_kwargs["landmark_tolerance"] = _number(geo["landmark_tolerance"], "geometry.landmark_tolerance")

    mat = _section(data, "materials", _MATERIAL_KEYS, "")
    for key, field_name in (("upper_foam", "upper_foam_material"), ("lower_foam", "lower_foam_material")):
        if key in mat:
            if mat[key] not in FOAM_MATERIALS:
                raise ConfigFileError(f"materials.{key} must be one of {list(FOAM_MATERIALS)}, got {mat[key]!r}.")
            sole_kwargs[field_name] = mat[key]
    for name, table in ((FOAM_FFTURBO, _FFTURBO_KEYS), (FOAM_FFLEAP, _FFLEAP_KEYS), ("plate", _PLATE_KEYS)):
        sub = _section(mat, name, table, "materials.")
        for key, field_name in table.items():
            if key in sub:
                sole_kwargs[field_name] = _number(sub[key], f"materials.{name}.{key}")

    mesh = _section(data, "mesh", _MESH_KEYS, "")
    for key, (field_name, scale) in _MESH_KEYS.items():
        if key in mesh:
            if scale is None:
                sole_kwargs[field_name] = _integer(mesh[key], f"mesh.{key}")
            else:
                sole_kwargs[field_name] = _number(mesh[key], f"mesh.{key}") * scale

    lk = _section(data, "lookup", _LOOKUP_KEYS, "")
    lk_kwargs: dict = {}
    if "a_m" in lk:
        lk_kwargs["a_m"] = _number(lk["a_m"], "lookup.a_m")
        lk_kwargs["a_over_length"] = None
    if "a_over_length" in lk:
        lk_kwargs["a_over_length"] = _number(lk["a_over_length"], "lookup.a_over_length")
    for key in ("kappa", "reciprocity_tol"):
        if key in lk:
            lk_kwargs[key] = _number(lk[key], f"lookup.{key}")
    if "output_dir" in lk:
        lk_kwargs["output_dir"] = _resolve(lk["output_dir"], base, "lookup.output_dir")
    if "store_nodal_fields" in lk:
        if not isinstance(lk["store_nodal_fields"], bool):
            raise ConfigFileError(f"lookup.store_nodal_fields must be true or false, got {lk['store_nodal_fields']!r}.")
        lk_kwargs["store_nodal_fields"] = lk["store_nodal_fields"]

    toe = _section(data, "toe_spring", _TOE_KEYS, "")
    toe_kwargs: dict = {}
    for key, field_name in _TOE_KEYS.items():
        if key not in toe:
            continue
        if key == "model":
            toe_kwargs[field_name] = str(toe[key])
        elif key == "root_scan_points":
            toe_kwargs[field_name] = _integer(toe[key], f"toe_spring.{key}")
        else:
            toe_kwargs[field_name] = _number(toe[key], f"toe_spring.{key}")

    rt = _section(data, "runtime", _RUNTIME_KEYS, "")
    rt_kwargs = {"shoe_width_m": _number(rt["shoe_width_m"], "runtime.shoe_width_m")} if "shoe_width_m" in rt else {}

    try:
        sole = MeasuredSoleConfig(**sole_kwargs)
        toe_cfg = ToeSpringConfig(**toe_kwargs)
    except ConfigFileError:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise ConfigFileError(f"{source_path or 'config'}: {exc}") from exc
    setup = MeasuredSoleSetup(
        sole=sole,
        lookup=LookupSettings(**lk_kwargs),
        toe_spring=toe_cfg,
        runtime=RuntimeSettings(**rt_kwargs),
        source_path=source_path,
    )
    setup.a  # validates a inside (0, L)
    return setup


def load_measured_sole_config(path: str | Path) -> MeasuredSoleSetup:
    """Read and validate a measured-sole JSON config."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Measured-sole config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigFileError(f"{path}: invalid JSON ({exc}).") from exc
    return setup_from_dict(data, base_dir=path.parent, source_path=str(path.resolve()))


def save_measured_sole_config(setup: MeasuredSoleSetup, path: str | Path) -> Path:
    """Write the canonical config (absolute paths) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def default_setup(geometry_csv: str | Path, shoe_length_mm: float, **lookup) -> MeasuredSoleSetup:
    """Setup with every default for the given CSV and shoe length."""
    return MeasuredSoleSetup(
        sole=MeasuredSoleConfig(shoe_length_mm=shoe_length_mm, geometry_csv=str(Path(geometry_csv).resolve())),
        lookup=LookupSettings(**lookup),
        toe_spring=ToeSpringConfig(),
        runtime=RuntimeSettings(),
    )


def with_sole_overrides(setup: MeasuredSoleSetup, **overrides) -> MeasuredSoleSetup:
    """Replace ``MeasuredSoleConfig`` fields (reloading the CSV when it changes)."""
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return setup
    valid = {f.name for f in fields(MeasuredSoleConfig)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown MeasuredSoleConfig field(s) {unknown}.")
    if "geometry_csv" in overrides:
        overrides["geometry_csv"] = str(Path(overrides["geometry_csv"]).resolve())
        overrides["normalized_geometry"] = None
    return replace(setup, sole=replace(setup.sole, **overrides))


def with_lookup_overrides(
    setup: MeasuredSoleSetup, *, a_m=None, kappa=None, reciprocity_tol=None, output_dir=None, store_nodal_fields=None
):
    lk = setup.lookup
    kw: dict = {}
    if store_nodal_fields is not None:
        kw["store_nodal_fields"] = bool(store_nodal_fields)
    if a_m is not None:
        kw.update(a_m=float(a_m), a_over_length=None)
    if kappa is not None:
        kw["kappa"] = float(kappa)
    if reciprocity_tol is not None:
        kw["reciprocity_tol"] = float(reciprocity_tol)
    if output_dir is not None:
        kw["output_dir"] = str(Path(output_dir).resolve())
    return setup if not kw else replace(setup, lookup=replace(lk, **kw))


# ---------------------------------------------------------------------------
# Fingerprint and build
# ---------------------------------------------------------------------------


def _round(x: float) -> float:
    return float(f"{float(x):.12e}")


def lookup_build_fingerprint(sole: MeasuredSoleConfig, a: float, kappa: float, reciprocity_tol: float) -> str:
    """SHA-256 of every parameter that changes the lookup contents.

    The geometry enters through its normalized landmark/curve values, so the
    fingerprint does not depend on the CSV path or formatting. Runtime-only
    settings (toe spring, shoe width) are excluded.
    """
    geom = sole.normalized_geometry
    payload = {
        "landmarks": {k: [_round(v[0]), _round(v[1])] for k, v in sorted(geom.landmarks.items())},
        "curves": {k: [[_round(p[0]), _round(p[1])] for p in np.asarray(v)] for k, v in sorted(geom.curves.items())},
        "shoe_length_mm": _round(sole.shoe_length_mm),
        "landmark_tolerance": _round(sole.landmark_tolerance),
        "upper_foam": sole.upper_foam_material,
        "lower_foam": sole.lower_foam_material,
        "ffturbo": [_round(sole.ffturbo_E), _round(sole.ffturbo_nu)],
        "ffleap": [_round(sole.ffleap_E_heel), _round(sole.ffleap_E_toe), _round(sole.ffleap_nu)],
        "EI_plate": _round(sole.EI_plate),
        "mesh": [
            _round(sole.mesh_size), _round(sole.toe_refinement), _round(sole.heel_corner_refinement),
            _round(sole.interface_refinement), _round(sole.plate_end_refinement),
            _round(sole.curvature_max_turn_deg), _round(sole.min_angle_deg), int(sole.element_order),
        ],
        "a": _round(a),
        "kappa": _round(kappa),
        "reciprocity_tol": _round(reciprocity_tol),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_measured_lookup(setup: MeasuredSoleSetup, output_dir: str | Path | None = None, *, progress=None):
    """Compute FEM, generate the interval lookup with the setup's ``a`` / ``kappa``,
    record the setup and its fingerprint in the lookup metadata, and save it.

    Returns ``(fem_result, lookup, npz_path)``.
    """
    from compliance_fem.compliance import compute_compliance
    from compliance_fem.contact_lookup import from_compliance_result, generate_contact_lookup, save_contact_lookup

    out = Path(output_dir) if output_dir is not None else setup.output_dir
    if out is None:
        raise ConfigFileError("No lookup output directory: set lookup.output_dir or pass output_dir.")
    fem = compute_compliance(setup.sole)
    lookup = generate_contact_lookup(
        from_compliance_result(fem),
        a=setup.a,
        kappa=setup.kappa,
        reciprocity_tol=setup.lookup.reciprocity_tol,
        fem_result=fem,
        progress=progress,
        store_fields=setup.lookup.store_nodal_fields,
    )
    attach_setup_metadata(lookup, setup)
    npz_path = save_contact_lookup(lookup, out)
    save_measured_sole_config(setup, npz_path.parent / "measured_sole_config.json")
    return fem, lookup, npz_path


def attach_setup_metadata(lookup, setup: MeasuredSoleSetup) -> None:
    lookup.metadata[SETUP_METADATA_KEY] = setup.to_dict()
    lookup.metadata[FINGERPRINT_METADATA_KEY] = setup.fingerprint()


def lookup_matches_setup(lookup, setup: MeasuredSoleSetup) -> bool | None:
    """True / False when the lookup records a fingerprint, ``None`` when it does not."""
    stored = (lookup.metadata or {}).get(FINGERPRINT_METADATA_KEY)
    if stored is None:
        return None
    return str(stored) == setup.fingerprint()


def stored_lookup_fingerprint(path: str | Path) -> str | None:
    """Read only the recorded build fingerprint of a saved lookup (no full load)."""
    from compliance_fem.contact_lookup import resolve_contact_lookup_path

    npz = resolve_contact_lookup_path(path)
    with np.load(npz, allow_pickle=True) as data:
        if "metadata_json" not in data.files:
            return None
        meta = json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0]))
    value = meta.get(FINGERPRINT_METADATA_KEY)
    return None if value is None else str(value)


def lookup_stores_nodal_fields(path: str | Path) -> bool:
    """Whether a saved lookup keeps the per-node bases (no full load)."""
    from compliance_fem.contact_lookup import resolve_contact_lookup_path

    with np.load(resolve_contact_lookup_path(path), allow_pickle=True) as data:
        return "bottom_u_basis" in data.files


def matching_prebuilt_lookup(setup: MeasuredSoleSetup) -> Path | None:
    """The NPZ at ``lookup.output_dir`` if it exists, was built from ``setup`` and
    stores the nodal fields when ``lookup.store_nodal_fields`` asks for them."""
    from compliance_fem.contact_lookup import resolve_contact_lookup_path

    if setup.output_dir is None:
        return None
    try:
        npz = resolve_contact_lookup_path(setup.output_dir)
        if stored_lookup_fingerprint(npz) != setup.fingerprint():
            return None
        if setup.lookup.store_nodal_fields and not lookup_stores_nodal_fields(npz):
            return None
        return npz
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def ensure_measured_lookup(config: str | Path | MeasuredSoleSetup, *, force: bool = False, progress=None) -> Path:
    """Return the lookup NPZ for ``config``, (re)building it when missing or stale."""
    setup = config if isinstance(config, MeasuredSoleSetup) else load_measured_sole_config(config)
    out = setup.output_dir
    if out is None:
        raise ConfigFileError("ensure_measured_lookup requires lookup.output_dir in the config.")
    if not force:
        npz = matching_prebuilt_lookup(setup)
        if npz is not None:
            return npz
    _, _, npz = build_measured_lookup(setup, out, progress=progress)
    return npz


# ---------------------------------------------------------------------------
# CLI: write a template / validate a config
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measured-sole JSON config utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    tpl = sub.add_parser("template", help="Write a config with every default filled in.")
    tpl.add_argument("output", type=Path)
    tpl.add_argument("--geometry-csv", type=Path, default=Path("data/geometry/sole_geometry_normalized.csv"))
    tpl.add_argument("--shoe-length-mm", type=float, default=270.0)
    chk = sub.add_parser("check", help="Validate a config and print the resolved values.")
    chk.add_argument("config", type=Path)
    bld = sub.add_parser("build", help="Build (or refresh) the lookup described by a config.")
    bld.add_argument("config", type=Path)
    bld.add_argument("--force", action="store_true")
    bld.add_argument(
        "--store-fields", action=argparse.BooleanOptionalAction, default=None,
        help="Override lookup.store_nodal_fields (keep the per-node bases in the NPZ).",
    )
    args = parser.parse_args(argv)

    if args.command == "template":
        setup = default_setup(args.geometry_csv, args.shoe_length_mm)
        print(f"wrote {save_measured_sole_config(setup, args.output)}")
    elif args.command == "check":
        setup = load_measured_sole_config(args.config)
        print(json.dumps(setup.to_dict(), indent=2))
        print(f"a = {setup.a:.6g} m, L = {setup.sole.L:.6g} m, fingerprint {setup.fingerprint()[:16]}")
    else:
        def progress(done: int, total: int) -> None:
            if done == total or done % 500 == 0:
                print(f"  lookup records {done}/{total}", flush=True)

        setup = with_lookup_overrides(load_measured_sole_config(args.config), store_nodal_fields=args.store_fields)
        print(f"lookup: {ensure_measured_lookup(setup, force=args.force, progress=progress)}")


if __name__ == "__main__":
    main()
