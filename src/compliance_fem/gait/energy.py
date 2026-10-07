"""Conservative energy accounting helpers for gait replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnergyReport:
    absorbed_work: float
    returned_work: float
    net_work: float
    note: str


def integrate_top_power(
    times: NDArray[np.float64],
    f_top: NDArray[np.float64],
    u_top: NDArray[np.float64],
) -> EnergyReport:
    """Integrate ``P = Σ f·u_dot`` on the top boundary when nodal fields exist.

    Parameters
    ----------
    f_top, u_top
        Shape ``(n_times, n_dof)`` in a consistent physical frame.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    f_top = np.asarray(f_top, dtype=np.float64)
    u_top = np.asarray(u_top, dtype=np.float64)
    if f_top.shape != u_top.shape or f_top.shape[0] != times.size:
        raise ValueError("f_top/u_top/times shape mismatch")
    if times.size < 2:
        return EnergyReport(0.0, 0.0, 0.0, "insufficient samples")
    du = np.diff(u_top, axis=0)
    f_mid = 0.5 * (f_top[1:] + f_top[:-1])
    dW = np.sum(f_mid * du, axis=1)
    absorbed = float(np.sum(dW[dW > 0.0]))
    returned = float(-np.sum(dW[dW < 0.0]))
    return EnergyReport(
        absorbed_work=absorbed,
        returned_work=returned,
        net_work=float(np.sum(dW)),
        note=(
            "Boundary work from nodal f·du. Viscoelastic internal dissipation is "
            "not identified with input_work - elastic_energy unless internal "
            "states are evolved at the FEM/modal level."
        ),
    )
