"""Wang / OpenSim TRC+MOT I/O, filename parsing, and trial discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

# Paper naming: Pxx_CV_TT with C in {p,n1,w1,...}, V in {w1,w2,r1,r2}, TT trial index.
# On disk this appears as underscores (P01_pr1_01) or spaces (P4 pr1 01); MOT
# files often append ``_force`` (P4 pr1 01_force.mot).
_TRIAL_RE = re.compile(
    r"^(?P<subject>P\d+)[_\s]+(?P<code>[-a-zA-Z0-9]+)[_\s]+(?P<trial>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrialSelection:
    subject: str
    code: str
    trial: str
    stem: str
    trc_path: Path | None = None
    mot_path: Path | None = None
    markers_csv: Path | None = None
    forces_csv: Path | None = None

    @property
    def is_preferred_r1(self) -> bool:
        """Preferred step width + 3.0 m/s running (code contains ``pr1``)."""
        return "pr1" in self.code.lower()


def _strip_force_suffix(stem: str) -> str:
    if stem.lower().endswith("_force"):
        return stem[: -len("_force")]
    return stem


def parse_wang_trial_name(name: str) -> tuple[str, str, str]:
    """Parse ``Pxx_code_TT`` stem into ``(subject, code, trial)``.

    Accepts underscore or space separators and an optional ``_force`` suffix
    (common on OpenSim MOT exports).
    """
    stem = _strip_force_suffix(Path(name).stem)
    m = _TRIAL_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Unrecognized Wang/OpenSim trial name {name!r}; expected Pxx_code_TT"
        )
    return m.group("subject"), m.group("code"), m.group("trial")


def discover_trials(root: str | Path) -> list[TrialSelection]:
    """Discover matched TRC/MOT (and optional CSV) pairs under ``root``.

    MOT files named ``…_force.mot`` are paired with the matching TRC stem
    without the ``_force`` suffix.
    """
    root = Path(root)
    if not root.exists():
        return []
    by_stem: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        raw_stem = path.stem
        try:
            subject, code, trial = parse_wang_trial_name(raw_stem)
        except ValueError:
            continue
        # Canonical stem without _force, preserving original separators.
        stem = _strip_force_suffix(raw_stem)
        slot = by_stem.setdefault(
            stem,
            {"subject": subject, "code": code, "trial": trial},
        )
        if suf == ".trc":
            slot["trc"] = path
        elif suf == ".mot":
            slot["mot"] = path
        elif suf == ".csv":
            lower = path.name.lower()
            if "force" in lower or "grf" in lower or "mot" in lower:
                slot["forces_csv"] = path
            else:
                slot["markers_csv"] = path

    out: list[TrialSelection] = []
    for stem, files in sorted(by_stem.items()):
        out.append(
            TrialSelection(
                subject=str(files["subject"]),
                code=str(files["code"]),
                trial=str(files["trial"]),
                stem=stem,
                trc_path=files.get("trc"),
                mot_path=files.get("mot"),
                markers_csv=files.get("markers_csv"),
                forces_csv=files.get("forces_csv"),
            )
        )
    return out


def _skip_opensim_header(lines: list[str], *, data_marker: str = "endheader") -> int:
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(data_marker):
            return i + 1
        if line.strip().lower() == "endheader":
            return i + 1
    # TRC often has a fixed multi-line header without endheader.
    return 0


def read_opensim_trc(
    path: str | Path,
    *,
    position_units: str = "auto",
) -> dict[str, Any]:
    """Read an OpenSim ``.trc`` marker file into SI metres.

    Laboratory axes (OpenSim-ready Wang): +x anterior, +y superior, +z right.
    """
    from compliance_fem.gait.units import resolve_position_units

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(text) < 6:
        raise ValueError(f"{path}: TRC file too short")

    meta_vals = text[2].split()
    rate_hz = float(meta_vals[0]) if meta_vals else 200.0
    units = "mm"
    if len(meta_vals) >= 5:
        units = meta_vals[4].lower()

    label_line = text[3].split("\t") if "\t" in text[3] else text[3].split()
    labels_raw = [t.strip() for t in label_line[2:] if t.strip()]
    labels: list[str] = []
    for lab in labels_raw:
        if lab and lab not in labels:
            labels.append(lab)

    data_start = 5
    rows = []
    for line in text[data_start:]:
        if not line.strip():
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: no TRC data rows")
    arr = np.asarray(rows, dtype=np.float64)
    times = arr[:, 1]
    coords = arr[:, 2:]
    n_markers = coords.shape[1] // 3
    if len(labels) < n_markers:
        labels = [f"M{i}" for i in range(n_markers)]

    pos_unit, pos_scale, pos_notes = resolve_position_units(
        declared=units,
        sample_positions=coords,
        mode=position_units,  # type: ignore[arg-type]
    )
    markers: dict[str, NDArray[np.float64]] = {}
    for i, lab in enumerate(labels[:n_markers]):
        markers[lab] = coords[:, 3 * i : 3 * i + 3] * pos_scale
    return {
        "path": path,
        "times": times,
        "markers": markers,
        "labels": labels[:n_markers],
        "rate_hz": rate_hz,
        "units_original": units,
        "units_si": "m",
        "source_format": "opensim_trc",
        "source_position_units": pos_unit,
        "position_scale": pos_scale,
        "unit_notes": list(pos_notes),
        "axis_mapping": {"x": "anterior", "y": "superior", "z": "right"},
    }


def read_opensim_mot(
    path: str | Path,
    *,
    position_units: str = "auto",
    moment_units: str = "auto",
) -> dict[str, Any]:
    """Read an OpenSim ``.mot`` / forces file into SI (N, N·m, m).

    MOT COP columns ``*_px,_py,_pz`` are **global laboratory** positions
    (anterior, superior, right), not heel-relative.
    """
    from compliance_fem.gait.units import resolve_moment_units, resolve_position_units

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_end = 0
    n_rows = None
    in_degrees = False
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("nrows"):
            n_rows = int(line.split("=")[-1].strip())
        if low.startswith("indegrees"):
            in_degrees = "yes" in low
        if low == "endheader":
            header_end = i + 1
            break
    if header_end == 0:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("time"):
                header_end = i
                break
    if header_end >= len(lines):
        raise ValueError(f"{path}: missing MOT header/data")

    colnames = lines[header_end].replace("\t", " ").split()
    data_lines = []
    for line in lines[header_end + 1 :]:
        if not line.strip() or line.strip().startswith("#"):
            continue
        data_lines.append([float(x) for x in line.replace("\t", " ").split()])
    data = np.asarray(data_lines, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(colnames):
        raise ValueError(
            f"{path}: data shape {data.shape} incompatible with columns {len(colnames)}"
        )
    times = data[:, 0]
    dt = np.diff(times)
    rate_hz = float(1.0 / np.median(dt)) if dt.size and np.median(dt) > 0 else 1000.0

    raw_cols: dict[str, NDArray[np.float64]] = {}
    for j, name in enumerate(colnames):
        if j == 0:
            continue
        raw_cols[name] = data[:, j].copy()

    # Collect samples for unit detection (once).
    cop_samples = []
    mom_samples = []
    for name, col in raw_cols.items():
        lname = name.lower()
        if "cop" in lname or lname.endswith("_px") or lname.endswith("_py") or lname.endswith("_pz"):
            cop_samples.append(col)
        if (
            "torque" in lname
            or "moment" in lname
            or lname.endswith("_mx")
            or lname.endswith("_my")
            or lname.endswith("_mz")
        ):
            mom_samples.append(col)
    pos_unit, pos_scale, pos_notes = resolve_position_units(
        declared=None,
        sample_positions=np.concatenate(cop_samples) if cop_samples else None,
        mode=position_units,  # type: ignore[arg-type]
    )
    mom_unit, mom_scale, mom_notes = resolve_moment_units(
        declared=None,
        sample_moments=np.concatenate(mom_samples) if mom_samples else None,
        mode=moment_units,  # type: ignore[arg-type]
    )

    columns: dict[str, NDArray[np.float64]] = {}
    for name, col in raw_cols.items():
        lname = name.lower()
        out = col.copy()
        if (
            "torque" in lname
            or "moment" in lname
            or lname.endswith("_mx")
            or lname.endswith("_my")
            or lname.endswith("_mz")
        ):
            out *= mom_scale
        elif "cop" in lname or lname.endswith("_px") or lname.endswith("_py") or lname.endswith("_pz"):
            out *= pos_scale
        columns[name] = out

    return {
        "path": path,
        "times": times,
        "columns": columns,
        "column_names": colnames[1:],
        "rate_hz": rate_hz,
        "in_degrees": in_degrees,
        "n_rows_header": n_rows,
        "units_si": {"force": "N", "moment": "N·m", "cop": "m", "time": "s"},
        "source_format": "opensim_mot",
        "source_position_units": pos_unit,
        "source_moment_units": mom_unit,
        "position_scale": pos_scale,
        "moment_scale": mom_scale,
        "unit_notes": list(pos_notes) + list(mom_notes),
        "axis_mapping": {
            "force_vx / px": "anterior",
            "force_vy / py": "superior",
            "force_vz / pz": "right",
            "cop_frame": "global_laboratory_ground",
        },
        "free_torque_interpretation": (
            "OpenSim plate torque_* columns are free moments at the COP in the "
            "laboratory frame. torque_z is the free sagittal moment about +z (right)."
        ),
    }


def read_wang_forces_csv(
    path: str | Path,
    *,
    position_units: str = "auto",
    moment_units: str = "auto",
    plate_index: int = 0,
) -> dict[str, Any]:
    """Read a raw Wang Qualisys CSV force export (Cx/Cy/Cz COP convention).

    Paper CSV ordering: Cx = mediolateral, Cy = anterior-posterior, Cz = vertical
    (normally zero). Forces Fx,Fy,Fz follow the device axes on the unit row.
    This is **not** the OpenSim MOT axis mapping.
    """
    from compliance_fem.gait.units import resolve_moment_units, resolve_position_units

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Find header row with Frame,Sub Frame,...
    header_i = None
    unit_i = None
    for i, line in enumerate(lines):
        if line.lower().startswith("frame"):
            header_i = i
            if i + 1 < len(lines):
                unit_i = i + 1
            break
    if header_i is None:
        raise ValueError(f"{path}: no Frame header row in Wang CSV")

    header = [c.strip() for c in lines[header_i].split(",")]
    units_row = [c.strip() for c in lines[unit_i].split(",")] if unit_i is not None else []
    data_start = (unit_i or header_i) + 1
    rows = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.split(",")
        try:
            rows.append([float(x) if x.strip() else float("nan") for x in parts])
        except ValueError:
            continue
    data = np.asarray(rows, dtype=np.float64)
    # Locate plate blocks: each has Fx,Fy,Fz,Mx,My,Mz,Cx,Cy,Cz
    blocks: list[dict[str, int]] = []
    i = 0
    while i < len(header):
        name = header[i].lower()
        if name == "fx":
            block = {"fx": i, "fy": i + 1, "fz": i + 2, "mx": i + 3, "my": i + 4, "mz": i + 5}
            # Cx,Cy,Cz follow moments in the paper export
            if i + 8 < len(header) and header[i + 6].lower() == "cx":
                block.update({"cx": i + 6, "cy": i + 7, "cz": i + 8})
                i += 9
            else:
                i += 6
            blocks.append(block)
        else:
            i += 1
    if not blocks:
        raise ValueError(f"{path}: no Fx..Cz force blocks found")
    if plate_index < 0 or plate_index >= len(blocks):
        raise IndexError(f"plate_index {plate_index} out of range 0..{len(blocks)-1}")
    b = blocks[plate_index]

    # Declared units from unit row if present
    decl_pos = None
    decl_mom = None
    if units_row and "cx" in b:
        u = units_row[b["cx"]].lower() if b["cx"] < len(units_row) else ""
        if "mm" in u:
            decl_pos = "mm"
        elif u == "m":
            decl_pos = "m"
    if units_row and "mz" in b:
        u = units_row[b["mz"]].lower().replace(" ", "") if b["mz"] < len(units_row) else ""
        if "mm.n" in u or "n.mm" in u or "nmm" in u or u == "mm.n":
            decl_mom = "N-mm"
        elif u in {"n.m", "n-m", "nm"}:
            decl_mom = "N-m"

    cx = data[:, b["cx"]] if "cx" in b else np.zeros(data.shape[0])
    cy = data[:, b["cy"]] if "cy" in b else np.zeros(data.shape[0])
    cz = data[:, b["cz"]] if "cz" in b else np.zeros(data.shape[0])
    pos_unit, pos_scale, pos_notes = resolve_position_units(
        declared=decl_pos,
        sample_positions=np.concatenate([cx, cy, cz]),
        mode=position_units,  # type: ignore[arg-type]
    )
    mom = data[:, b["mz"]]
    mom_unit, mom_scale, mom_notes = resolve_moment_units(
        declared=decl_mom,
        sample_moments=mom,
        mode=moment_units,  # type: ignore[arg-type]
    )

    # Map CSV device axes → OpenSim-like lab sagittal: px=anterior=Cy, py=vertical=Cz,
    # pz=ML=Cx (paper: Cx ML, Cy AP, Cz vertical).
    rate = 1000.0
    for line in lines[:header_i]:
        if line.strip().isdigit():
            rate = float(line.strip())
            break
    n = data.shape[0]
    times = np.arange(n, dtype=np.float64) / rate
    # Sub Frame column may refine time; use Frame index / rate for now.
    if header[0].lower() == "frame" and np.all(np.isfinite(data[:, 0])):
        times = (data[:, 0] - data[0, 0]) / rate

    fx = data[:, b["fx"]]  # device — keep as exported; caller must know device axes
    fy = data[:, b["fy"]]
    fz = data[:, b["fz"]]
    # Paper CSV often uses Fx,Fy,Fz with Fy vertical for Kistler; CoP Cy = AP.
    # Provide OpenSim-named channels after axis remap for extract_plate_wrench.
    columns = {
        "1_force_vx": cy * 0.0 + fx,  # placeholder; prefer explicit remap below
    }
    # Remap assuming Qualisys/Kistler CSV: Fx=ML, Fy=AP, Fz=vertical (common),
    # but paper says Cx=ML, Cy=AP, Cz=vertical. Force axes often match CoP.
    # Use: lab anterior = Fy (device), superior = Fz, right = Fx.
    columns = {
        "1_force_vx": fy.copy(),
        "1_force_vy": fz.copy(),
        "1_force_vz": fx.copy(),
        "1_force_px": cy * pos_scale,  # AP
        "1_force_py": cz * pos_scale,  # vertical
        "1_force_pz": cx * pos_scale,  # ML
        "1_torque_x": data[:, b["mx"]] * mom_scale,
        "1_torque_y": data[:, b["my"]] * mom_scale,
        "1_torque_z": data[:, b["mz"]] * mom_scale,
    }
    return {
        "path": path,
        "times": times,
        "columns": columns,
        "column_names": list(columns.keys()),
        "rate_hz": rate,
        "units_si": {"force": "N", "moment": "N·m", "cop": "m", "time": "s"},
        "source_format": "wang_qualisys_csv",
        "source_position_units": pos_unit,
        "source_moment_units": mom_unit,
        "position_scale": pos_scale,
        "moment_scale": mom_scale,
        "unit_notes": list(pos_notes) + list(mom_notes),
        "axis_mapping": {
            "csv_Cx": "mediolateral → lab pz",
            "csv_Cy": "anterior-posterior → lab px",
            "csv_Cz": "vertical → lab py",
            "csv_Fx": "mediolateral → lab fz",
            "csv_Fy": "anterior-posterior → lab fx",
            "csv_Fz": "vertical → lab fy",
            "cop_frame": "global_laboratory_ground",
        },
        "csv_raw_blocks": len(blocks),
        "plate_index": plate_index,
    }


def group_force_plate_channels(column_names: list[str]) -> dict[str, dict[str, str]]:
    """Map force-plate id → {fx,fy,fz,mx,my,mz,px,py,pz} column names when possible."""
    plates: dict[str, dict[str, str]] = {}
    for name in column_names:
        lname = name.lower()
        # Common OpenSim: ground_force_vx, ground_force_1_vx, ...
        m = re.search(r"(?:force|grf|plate)[_\s-]*(\d+)", lname)
        pid = m.group(1) if m else ("1" if "1" in lname else "0")
        slot = plates.setdefault(pid, {})
        if re.search(r"v?x$|_vx$|_fx$|force.*x", lname) and "m" not in lname.split("_")[-1]:
            if "fx" not in slot:
                slot["fx"] = name
        if re.search(r"v?y$|_vy$|_fy$|force.*y", lname) and "moment" not in lname and "torque" not in lname:
            if "fy" not in slot and ("force" in lname or "grf" in lname or "v" in lname):
                slot["fy"] = name
        if re.search(r"v?z$|_vz$|_fz$|force.*z", lname) and "moment" not in lname:
            if "fz" not in slot:
                slot["fz"] = name
        if re.search(r"(mx|_mx|moment.*x|torque.*x)", lname):
            slot["mx"] = name
        if re.search(r"(my|_my|moment.*y|torque.*y)", lname):
            slot["my"] = name
        if re.search(r"(mz|_mz|moment.*z|torque.*z)", lname):
            slot["mz"] = name
        if re.search(r"(px|_px|cop.*x)", lname):
            slot["px"] = name
        if re.search(r"(py|_py|cop.*y)", lname):
            slot["py"] = name
        if re.search(r"(pz|_pz|cop.*z)", lname):
            slot["pz"] = name
    return plates


def extract_plate_wrench(
    mot: dict[str, Any],
    plate_id: str,
) -> dict[str, NDArray[np.float64]]:
    """Extract SI wrench time series for one plate from a parsed MOT dict."""
    groups = group_force_plate_channels(list(mot["columns"].keys()))
    if plate_id not in groups and plate_id != "auto":
        # try integer variants
        if str(int(plate_id)) in groups:
            plate_id = str(int(plate_id))
        else:
            raise KeyError(f"force plate {plate_id!r} not found; have {sorted(groups)}")
    if plate_id == "auto":
        # Prefer plate with largest peak vertical force.
        best_id = None
        best_peak = -1.0
        for pid, slots in groups.items():
            fy_name = slots.get("fy") or slots.get("fz")
            if fy_name is None:
                continue
            peak = float(np.nanmax(np.abs(mot["columns"][fy_name])))
            if peak > best_peak:
                best_peak = peak
                best_id = pid
        if best_id is None:
            raise KeyError("no force plate channels detected in MOT")
        plate_id = best_id
    slots = groups[plate_id]
    cols = mot["columns"]
    n = mot["times"].size

    def _get(*keys: str, default: float = 0.0) -> NDArray[np.float64]:
        for k in keys:
            if k in slots and slots[k] in cols:
                return np.asarray(cols[slots[k]], dtype=np.float64)
        return np.full(n, default, dtype=np.float64)

    # OpenSim +y superior: vertical is often ground_force_vy.
    fx = _get("fx")
    fy = _get("fy")
    fz = _get("fz")
    # If fy looks empty but fz has content and naming is z-up, swap later in sagittal.
    return {
        "plate_id": plate_id,  # type: ignore[dict-item]
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "mx": _get("mx"),
        "my": _get("my"),
        "mz": _get("mz"),
        "px": _get("px"),
        "py": _get("py"),
        "pz": _get("pz"),
        "times": np.asarray(mot["times"], dtype=np.float64),
        "channel_map": slots,  # type: ignore[dict-item]
    }
