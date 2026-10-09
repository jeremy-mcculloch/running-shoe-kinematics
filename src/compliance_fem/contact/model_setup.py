"""JSON configuration for the measured carbon-plated sole.

The normalized geometry CSV holds the shape only. Everything else that defines
a measured-sole model — shoe length, foam materials, plate stiffness, mesh,
toe length, minimum bend radius, passive toe spring and runtime shoe
width — lives in one JSON file whose layout mirrors :class:`ModelSetup`: one
section per settings dataclass, keyed by the dataclass field names::

    {
      "schema_version": 2,
      "sole":       {"geometry_csv": "../data/geometry/sole_geometry.csv",
                     "shoe_length_mm": 270, "mesh_size_m": 0.003, ...},
      "lookup":     {"toe_length_mm": 59.4, "min_bend_radius_mm": 6.75,
                     "output_dir": "../outputs/measured_sole_lookup"},
      "toe_spring": {"toe_stiffness_Nm_per_rad": 25.0, ...},
      "runtime":    {"shoe_width_m": 0.10}
    }

Only ``sole.geometry_csv`` and ``sole.shoe_length_mm`` are required; every
other key falls back to the dataclass default. Unknown keys are rejected so a
typo cannot silently fall back to a default. Relative paths resolve against
the directory containing the JSON file.

Lookups built through :func:`ensure_lookup_exists` store the full setup and a
fingerprint of the build-relevant parameters in their metadata, so
:class:`compliance_fem.api.SoleModel` can detect a stale lookup.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import types
from dataclasses import Field, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import numpy as np

from compliance_fem.contact.config import SoleConfig
from compliance_fem.contact.toe_spring import ToeSpringConfig

CONFIG_SCHEMA_VERSION = 2
DEFAULT_TOE_LENGTH_MM = 59.4
DEFAULT_MIN_BEND_RADIUS_MM = 6.75
DEFAULT_KAPPA = 160.0
DEFAULT_RECIPROCITY_TOL = 1.0e-6
DEFAULT_SHOE_WIDTH_M = 0.10
SETUP_METADATA_KEY = "model_setup"
FINGERPRINT_METADATA_KEY = "model_setup_fingerprint"
DEFAULT_LOOKUP_DIR = Path("outputs/contact_lookup")


class ConfigFileError(ValueError):
    """Malformed measured-sole configuration file."""


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LookupConfig:
    """Contact-lookup generation settings.

    ``toe_length_mm`` is the distance from the softplus joint to the toe tip,
    so the joint sits at ``x = L - toe_length`` from the heel.
    ``min_bend_radius_mm`` is the tightest radius of that shape at a 45° toe
    angle; the dimensionless softplus sharpness is ``κ = 4L / R``.
    """

    toe_length_mm: float = DEFAULT_TOE_LENGTH_MM
    min_bend_radius_mm: float = DEFAULT_MIN_BEND_RADIUS_MM
    reciprocity_tol: float = DEFAULT_RECIPROCITY_TOL
    output_dir: str | None = None
    store_nodal_fields: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.toe_length_mm) and self.toe_length_mm > 0.0):
            raise ConfigFileError("lookup.toe_length_mm must be positive.")
        if not (math.isfinite(self.min_bend_radius_mm) and self.min_bend_radius_mm > 0.0):
            raise ConfigFileError("lookup.min_bend_radius_mm must be positive.")
        if not self.reciprocity_tol > 0.0:
            raise ConfigFileError("lookup.reciprocity_tol must be positive.")

    def toe_length_m(self, L: float) -> float:
        """Toe length in metres, checked to lie inside (0, L)."""
        toe_length = float(self.toe_length_mm) / 1000.0
        if not 0.0 < toe_length < float(L):
            raise ConfigFileError(
                f"lookup.toe_length_mm = {self.toe_length_mm:.6g} mm must be shorter than the shoe "
                f"(L = {float(L) * 1000.0:.6g} mm)."
            )
        return toe_length


@dataclass(frozen=True)
class RuntimeConfig:
    shoe_width_m: float = DEFAULT_SHOE_WIDTH_M

    def __post_init__(self) -> None:
        if not (math.isfinite(self.shoe_width_m) and self.shoe_width_m > 0.0):
            raise ConfigFileError("runtime.shoe_width_m must be positive.")


@dataclass(frozen=True)
class ModelSetup:
    """Everything needed to build and run a measured-sole model."""

    sole: SoleConfig
    lookup: LookupConfig
    toe_spring: ToeSpringConfig
    runtime: RuntimeConfig
    source_path: str | None = None

    @property
    def toe_length_m(self) -> float:
        """Toe length in metres; the softplus joint sits at ``x = L - toe_length``."""
        return self.lookup.toe_length_m(self.sole.L)

    @property
    def softplus_kappa(self) -> float:
        """Dimensionless softplus sharpness, κ = 4 L / R, with both lengths in mm."""
        return 4.0 * float(self.sole.shoe_length_mm) / float(self.lookup.min_bend_radius_mm)

    @property
    def output_dir(self) -> Path | None:
        return None if self.lookup.output_dir is None else Path(self.lookup.output_dir)

    def fingerprint(self) -> str:
        return lookup_build_fingerprint(self.sole, self.toe_length_m, self.softplus_kappa, self.lookup.reciprocity_tol)

    def to_dict(self) -> dict:
        """Canonical JSON dict (all values filled in; paths as resolved absolute strings)."""
        out: dict = {"schema_version": CONFIG_SCHEMA_VERSION}
        for name, cls in _config_sections().items():
            section = getattr(self, name)
            out[name] = {f.name: getattr(section, f.name) for f in _file_fields(cls)}
        return out


# ---------------------------------------------------------------------------
# Config layout, derived from the ModelSetup dataclass fields
# ---------------------------------------------------------------------------

# Rebuilt from geometry_csv, so it never appears in the file.
_NOT_IN_FILE = {"normalized_geometry"}
_PATH_FIELDS = {"geometry_csv", "output_dir"}
_REQUIRED = (("sole", "geometry_csv"), ("sole", "shoe_length_mm"))
_EXTRA_TOP_KEYS = {"schema_version", "description"}


def _config_sections() -> dict[str, type]:
    """Returns dictionary of section name -> settings dataclass for each config section in ModelSetup."""
    hints = get_type_hints(ModelSetup)
    return {f.name: hints[f.name] for f in fields(ModelSetup) if is_dataclass(hints[f.name])}


def _file_fields(cls: type) -> list[Field]:
    return [f for f in fields(cls) if f.init and f.name not in _NOT_IN_FILE]


def _split_optional(tp) -> tuple[type, bool]:
    """``float | None`` -> ``(float, True)``; ``float`` -> ``(float, False)``."""
    if get_origin(tp) in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0], len(args) < len(get_args(tp))
    return tp, False


def _section(data: dict, name: str, allowed) -> dict:
    """Returns data[name] if it exists, otherwise empty dict."""
    sec = data.get(name, {})
    if sec is None:
        sec = {}
    if not isinstance(sec, dict):
        raise ConfigFileError(f"{name} must be an object, got {type(sec).__name__}.")
    unknown = sorted(set(sec) - set(allowed))
    if unknown:
        raise ConfigFileError(f"{name}: unknown key(s) {unknown}; allowed {sorted(allowed)}.")
    return sec


def _float(value, key: str) -> float:
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


def _resolve_path(path_text, base: Path | None, key: str) -> str:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ConfigFileError(f"{key} must be a non-empty path string.")
    p = Path(path_text).expanduser()
    if not p.is_absolute() and base is not None:
        p = base / p
    return str(p.resolve())


def _parse_value(value, tp, name: str, key: str, base: Path | None):
    """Validate one JSON value against the dataclass field type ``tp``."""
    kind, optional = _split_optional(tp)
    if value is None and optional:
        return None
    if name in _PATH_FIELDS:
        return _resolve_path(value, base, key)
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigFileError(f"{key} must be true or false, got {value!r}.")
        return value
    if kind is int:
        return _integer(value, key)
    if kind is float:
        return _float(value, key)
    if kind is str:
        if not isinstance(value, str):
            raise ConfigFileError(f"{key} must be a string, got {value!r}.")
        return value
    raise TypeError(f"No config parser for {key} of type {tp!r}.")


# ---------------------------------------------------------------------------
# Parse / load / save
# ---------------------------------------------------------------------------


def model_setup_from_dict(data: dict, *, base_dir: Path | str | None = None, source_path: str | None = None) -> ModelSetup:
    """Validate a config dict and build the setup. Relative paths resolve against ``base_dir``."""
    if not isinstance(data, dict):
        raise ConfigFileError("Configuration must be a JSON object.")
    base = None if base_dir is None else Path(base_dir).resolve()
    section_classes = _config_sections()

    # Ensures all top level keys are allowed.
    top_keys = _EXTRA_TOP_KEYS | set(section_classes)
    unknown = sorted(set(data) - top_keys)
    if unknown:
        raise ConfigFileError(f"Unknown top-level key(s) {unknown}; allowed {sorted(top_keys)}.")
    version = data.get("schema_version", CONFIG_SCHEMA_VERSION)
    if _integer(version, "schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigFileError(
            f"Unsupported config schema_version={version}; this code reads schema_version={CONFIG_SCHEMA_VERSION}."
        )

    kwargs: dict[str, dict] = {}
    for name, cls in section_classes.items():
        hints = get_type_hints(cls)
        file_fields = [f.name for f in _file_fields(cls)] # List of config class fields
        sec = _section(data, name, file_fields)
        kwargs[name] = {
            key: _parse_value(sec[key], hints[key], key, f"{name}.{key}", base) for key in file_fields if key in sec
        }
    for name, key in _REQUIRED:
        if key not in kwargs[name]:
            raise ConfigFileError(f"{name}.{key} is required.")

    try:
        sections = {name: cls(**kwargs[name]) for name, cls in section_classes.items()}
    except ConfigFileError:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise ConfigFileError(f"{source_path or 'config'}: {exc}") from exc
    setup = ModelSetup(**sections, source_path=source_path)
    setup.toe_length_m  # validates toe_length inside (0, L)
    return setup


def load_model_setup(path: str | Path) -> ModelSetup:
    """Read and validate a model setup JSON config."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model setup JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigFileError(f"{path}: invalid JSON ({exc}).") from exc
    return model_setup_from_dict(data, base_dir=path.parent, source_path=str(path.resolve()))


def save_model_setup(setup: ModelSetup, path: str | Path) -> Path:
    """Write the canonical config (absolute paths) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fingerprint and build
# ---------------------------------------------------------------------------


def _round(x: float) -> float:
    return float(f"{float(x):.12e}")


def lookup_build_fingerprint(sole: SoleConfig, toe_length: float, kappa: float, reciprocity_tol: float) -> str:
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
        "ffturbo": [_round(sole.ffturbo_E_Pa), _round(sole.ffturbo_nu)],
        "ffleap": [_round(sole.ffleap_E_heel_Pa), _round(sole.ffleap_E_toe_Pa), _round(sole.ffleap_nu)],
        "EI_plate_Nm2_per_m": _round(sole.EI_plate_Nm2_per_m),
        "mesh": [
            _round(sole.mesh_size_m), _round(sole.toe_refinement), _round(sole.heel_corner_refinement),
            _round(sole.interface_refinement), _round(sole.plate_end_refinement),
            _round(sole.curvature_max_turn_deg), _round(sole.min_angle_deg), int(sole.element_order),
        ],
        "toe_length": _round(toe_length),
        "kappa": _round(kappa),
        "reciprocity_tol": _round(reciprocity_tol),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def attach_setup_metadata(lookup, setup: ModelSetup) -> None:
    lookup.metadata[SETUP_METADATA_KEY] = setup.to_dict()
    lookup.metadata[FINGERPRINT_METADATA_KEY] = setup.fingerprint()


def lookup_matches_setup(lookup, setup: ModelSetup) -> bool | None:
    """True / False when the lookup records a fingerprint, ``None`` when it does not."""
    stored = (lookup.metadata or {}).get(FINGERPRINT_METADATA_KEY)
    if stored is None:
        return None
    return str(stored) == setup.fingerprint()


def stored_lookup_fingerprint(path: str | Path) -> str | None:
    """Read only the recorded build fingerprint of a saved lookup (no full load)."""
    from compliance_fem.contact.lookup import resolve_contact_lookup_path

    npz = resolve_contact_lookup_path(path)
    with np.load(npz, allow_pickle=True) as data:
        if "metadata_json" not in data.files:
            return None
        meta = json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0]))
    value = meta.get(FINGERPRINT_METADATA_KEY)
    return None if value is None else str(value)


def lookup_stores_nodal_fields(path: str | Path) -> bool:
    """Whether a saved lookup keeps the per-node bases (no full load)."""
    from compliance_fem.contact.lookup import resolve_contact_lookup_path

    with np.load(resolve_contact_lookup_path(path), allow_pickle=True) as data:
        return "bottom_u_basis" in data.files


def matching_prebuilt_lookup(setup: ModelSetup, output_dir: str | Path | None = None) -> Path | None:
    """The NPZ in ``output_dir`` (default ``lookup.output_dir``) if it exists, was built
    from ``setup`` and stores the nodal fields when ``lookup.store_nodal_fields`` asks for them."""
    from compliance_fem.contact.lookup import resolve_contact_lookup_path

    out = Path(output_dir) if output_dir is not None else setup.output_dir
    if out is None:
        return None
    try:
        npz = resolve_contact_lookup_path(out)
        if stored_lookup_fingerprint(npz) != setup.fingerprint():
            return None
        if setup.lookup.store_nodal_fields and not lookup_stores_nodal_fields(npz):
            return None
        return npz
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def ensure_lookup_exists(
    config: str | Path | ModelSetup,
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    save_fem: bool = False,
    skip_plots: bool = False,
    quiet: bool = False,
) -> Path:
    """Return the lookup NPZ for ``config``, building it when missing or stale.

    The lookup goes to ``output_dir``, else ``lookup.output_dir``, else
    ``outputs/contact_lookup``. A lookup there that was built from the same setup is
    reused unless ``force``. A build runs mesh -> FEM -> lookup, records the setup and
    its fingerprint in the lookup metadata, and writes ``contact_lookup.npz``,
    ``measured_sole_config.json``, ``generation_summary.json`` and the diagnostic
    plots. ``save_fem`` also writes the compliance NPZ, mesh, compliance heatmaps and
    ``fem_summary.json`` to ``<output>/fem`` (even when the lookup is reused).
    ``skip_plots`` leaves out the lookup diagnostic plots and compliance heatmaps.
    """
    from compliance_fem.contact.lookup import get_compliance_block_matrix, generate_contact_lookup, save_contact_lookup
    from compliance_fem.plotting.contact import save_lookup_plots

    setup = config if isinstance(config, ModelSetup) else load_model_setup(config)
    out = Path(output_dir) if output_dir is not None else (setup.output_dir or DEFAULT_LOOKUP_DIR)
    fem_dir = out / "fem" if save_fem else None
    say = (lambda msg: None) if quiet else print

    existing = None if force else matching_prebuilt_lookup(setup, out)
    if existing is not None:
        say(f"Contact lookup {existing} is up to date with {setup.source_path or 'the setup'}; pass force to rebuild.")
        if fem_dir is not None:
            _save_fem_outputs(_compute_fem(setup, fem_dir), setup, fem_dir, say, plots=not skip_plots)
        return existing

    fem = _compute_fem(setup, fem_dir)
    t0 = time.perf_counter()
    lookup = generate_contact_lookup(
        get_compliance_block_matrix(fem),
        toe_length=setup.toe_length_m,
        kappa=setup.softplus_kappa,
        reciprocity_tol=setup.lookup.reciprocity_tol,
        fem_result=fem,
        progress=None if quiet else (lambda done, total: _print_progress(done, total, t0)),
        store_fields=setup.lookup.store_nodal_fields,
    )
    attach_setup_metadata(lookup, setup)
    npz_path = save_contact_lookup(lookup, out)
    save_model_setup(setup, npz_path.parent / "measured_sole_config.json")
    if not skip_plots:
        save_lookup_plots(lookup, out)
    _write_lookup_summary(lookup, setup, npz_path, out, say)
    if fem_dir is not None:
        _save_fem_outputs(fem, setup, fem_dir, say, plots=not skip_plots)
    return npz_path


def _print_progress(done: int, total: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    eta = elapsed / max(done, 1) * (total - done)
    sys.stdout.write(f"\r  intervals {done}/{total}  elapsed {elapsed:7.1f} s  eta {eta:7.1f} s")
    if done == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _compute_fem(setup: ModelSetup, fem_dir: Path | None):
    """FEM compliance; with ``fem_dir`` the Gmsh mesh is also written there."""
    from compliance_fem.fem.compliance import compute_compliance
    from compliance_fem.geometry.mesh import generate_mesh

    if fem_dir is None:
        return compute_compliance(setup.sole)
    fem_dir.mkdir(parents=True, exist_ok=True)
    return compute_compliance(setup.sole, mesh_data=generate_mesh(setup.sole, output_msh=fem_dir / "mesh.msh"))


def _save_fem_outputs(fem, setup: ModelSetup, fem_dir: Path, say, *, plots: bool) -> None:
    """Compliance NPZ, ParaView mesh, compliance heatmaps and the FEM residual checks."""
    from compliance_fem.fem.compliance import save_compliance_npz
    from compliance_fem.plotting.compliance import plot_compliance_heatmap

    fem_dir.mkdir(parents=True, exist_ok=True)
    save_compliance_npz(fem, fem_dir / "compliance_results.npz")
    fem.mesh_data.mesh.save(str(fem_dir / "mesh.vtk"))
    if plots:
        plot_compliance_heatmap(fem.Cbt_force, r"$C_{bt}^F$", fem_dir / "Cbt_force_heatmap.png")
        plot_compliance_heatmap(fem.Cbb_force, r"$C_{bb}^F$", fem_dir / "Cbb_force_heatmap.png")
    summary = {
        "config_source": setup.source_path,
        "geometry": fem.geometry_metadata,
        "rigid_mode_error": fem.rigid_mode_error,
        "rigid_constraint_residual": fem.rigid_constraint_residual,
        "reciprocity_errors": asdict(fem.reciprocity),
        "solve_residuals": asdict(fem.solve_residuals),
        "inextensibility_residuals": asdict(fem.inextensibility_residuals),
        "gauge_residuals": asdict(fem.gauge_residuals),
        "number_of_plate_nodes": fem.n_plate_nodes,
        "number_of_plate_constraints": fem.n_lambda,
        "constraint_rank": fem.constraint_rank,
    }
    (fem_dir / "fem_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    say(f"FEM outputs written to {fem_dir}")


def _nanmax(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if arr.size else float("nan")


def _write_lookup_summary(lookup, setup: ModelSetup, npz_path: Path, out: Path, say) -> None:
    from compliance_fem.contact.topology import ContactType

    n_plate_nodes = (
        int(lookup.plate_node_ids.size) if lookup.has_plate_response and lookup.plate_node_ids is not None else 0
    )
    valid = lookup.valid_mask
    label_counts = {}
    for kind in (ContactType.HEEL, ContactType.INTERIOR, ContactType.TOE, ContactType.FULL):
        rows = lookup.rows_for(kind)
        label_counts[kind.value] = {"theoretical": int(rows.size), "valid": int(np.count_nonzero(valid[rows]))}
    n_b = int(lookup.n_bottom_nodes)
    summary = {
        "contact_set_model": "single_contiguous_interval",
        "n_bottom_nodes": n_b,
        "n_records_theoretical": n_b * (n_b + 1) // 2,
        "n_records": int(lookup.n_records),
        "n_valid_records": int(np.count_nonzero(valid)),
        "label_counts": label_counts,
        "rejection_counts": lookup.rejection_summary(),
        "build_time_s": float(lookup.build_time_s),
        "file_size_bytes": int(npz_path.stat().st_size),
        "softplus_toe_length": lookup.softplus_toe_length,
        "softplus_kappa": lookup.softplus_kappa,
        "reciprocity_error": lookup.reciprocity_error,
        "max_solve_residual": _nanmax(lookup.solve_residuals[valid]),
        "has_plate_response": bool(lookup.has_plate_response),
        "nodal_fields_stored": bool(lookup.nodal_fields_stored),
        "n_plate_nodes": n_plate_nodes,
        "EI_plate_Nm2_per_m": float(lookup.EI_plate_Nm2_per_m) if lookup.has_plate_response else None,
        "schema_version": int(lookup.schema_version),
        "geometry": lookup.geometry_metadata or None,
        "config_source": setup.source_path,
        "model_setup_fingerprint": setup.fingerprint(),
        "output": str(npz_path),
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    say(f"Contact lookup written to {out}")
    counts = ", ".join(f"{k}={v['valid']}/{v['theoretical']}" for k, v in label_counts.items())
    say(
        f"N_b={n_b}  records valid/theoretical={summary['n_valid_records']}/{summary['n_records_theoretical']} "
        f"({counts})  build={summary['build_time_s']:.1f} s  size={summary['file_size_bytes'] / 1e6:.1f} MB  "
        f"max_solve_residual={summary['max_solve_residual']:.3e}  plate_response={summary['has_plate_response']}  "
        f"nodal_fields={'stored' if lookup.nodal_fields_stored else 'on demand'}"
        + (f"  n_plate={n_plate_nodes}" if lookup.has_plate_response else "")
    )
    if summary["rejection_counts"]:
        say(f"rejections: {summary['rejection_counts']}")
