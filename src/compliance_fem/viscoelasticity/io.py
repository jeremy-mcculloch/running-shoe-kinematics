"""CSV I/O for viscoelastic force histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_force_history(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``time,Fx,Fy[,...]`` CSV.

    Returns
    -------
    times : (n,)
    FVE : (n, n_components)
    """
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is None:
        raise ValueError(f"{path}: expected a header row with column names")
    names = [n.lower() for n in data.dtype.names]
    if "time" not in names:
        raise ValueError(f"{path}: missing 'time' column")
    time_key = data.dtype.names[names.index("time")]
    times = np.asarray(data[time_key], dtype=np.float64).reshape(-1)

    force_cols: list[str] = []
    for key, lower in zip(data.dtype.names, names):
        if lower == "time":
            continue
        if lower.startswith("fx") or lower.startswith("fy") or lower.startswith("f"):
            force_cols.append(key)
        else:
            force_cols.append(key)
    # Prefer Fx, Fy ordering when present.
    preferred = []
    remaining = list(force_cols)
    for want in ("fx", "fy"):
        for key, lower in zip(data.dtype.names, names):
            if lower == want and key in remaining:
                preferred.append(key)
                remaining.remove(key)
                break
    force_cols = preferred + remaining
    if not force_cols:
        raise ValueError(f"{path}: no force columns found")

    FVE = np.column_stack([np.asarray(data[c], dtype=np.float64) for c in force_cols])
    if np.any(np.diff(times) < 0.0):
        raise ValueError(f"{path}: time column must be monotonically non-decreasing")
    return times, FVE


def write_force_history(
    path: str | Path,
    times: NDArray[np.float64],
    FVE: NDArray[np.float64],
    FE: NDArray[np.float64],
) -> None:
    """Write ``time,Fx_ve,Fy_ve,...,Fx_e,Fy_e,...`` CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    FVE = np.asarray(FVE, dtype=np.float64)
    FE = np.asarray(FE, dtype=np.float64)
    if FVE.ndim == 1:
        FVE = FVE.reshape(-1, 1)
    if FE.ndim == 1:
        FE = FE.reshape(-1, 1)
    n_comp = FVE.shape[1]
    if FE.shape != FVE.shape or times.size != FVE.shape[0]:
        raise ValueError("times / FVE / FE shape mismatch")

    if n_comp == 2:
        ve_names = ["Fx_ve", "Fy_ve"]
        e_names = ["Fx_e", "Fy_e"]
    else:
        ve_names = [f"F{i}_ve" for i in range(n_comp)]
        e_names = [f"F{i}_e" for i in range(n_comp)]

    header = ",".join(["time", *ve_names, *e_names])
    rows = np.column_stack([times, FVE, FE])
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.16g")
