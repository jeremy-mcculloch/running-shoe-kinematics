"""Stance detection from vertical GRF with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StanceInterval:
    index: int
    i0: int
    i1: int
    t0: float
    t1: float
    peak_fy: float


def detect_stances(
    times: NDArray[np.float64],
    fy: NDArray[np.float64],
    *,
    threshold: float = 20.0,
    hysteresis: float = 5.0,
    min_duration: float = 0.05,
) -> list[StanceInterval]:
    """Detect stance intervals where |Fy| rises above threshold with hysteresis.

    Uses ``on = threshold``, ``off = max(0, threshold - hysteresis)``.
    ``fy`` should be the vertical GRF component (ground-on-body, typically >0).
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    fy = np.asarray(fy, dtype=np.float64).reshape(-1)
    if times.shape != fy.shape:
        raise ValueError("times and fy must have the same shape")
    on = float(threshold)
    off = float(max(0.0, threshold - hysteresis))
    stances: list[StanceInterval] = []
    in_stance = False
    i0 = 0
    for i in range(fy.size):
        v = float(fy[i])
        if not in_stance and v >= on:
            in_stance = True
            i0 = i
        elif in_stance and v <= off:
            i1 = i
            t0 = float(times[i0])
            t1 = float(times[i1])
            if t1 - t0 >= min_duration:
                peak = float(np.max(fy[i0 : i1 + 1]))
                stances.append(
                    StanceInterval(
                        index=len(stances),
                        i0=i0,
                        i1=i1,
                        t0=t0,
                        t1=t1,
                        peak_fy=peak,
                    )
                )
            in_stance = False
    if in_stance:
        i1 = fy.size - 1
        t0 = float(times[i0])
        t1 = float(times[i1])
        if t1 - t0 >= min_duration:
            peak = float(np.max(fy[i0 : i1 + 1]))
            stances.append(
                StanceInterval(
                    index=len(stances),
                    i0=i0,
                    i1=i1,
                    t0=t0,
                    t1=t1,
                    peak_fy=peak,
                )
            )
    return stances


def pad_stance_window(
    times: NDArray[np.float64],
    stance: StanceInterval,
    *,
    pre_s: float = 0.05,
    post_s: float = 0.05,
) -> tuple[int, int]:
    """Expand stance indices by pre/post zero-load padding (clamped to array)."""
    t0 = stance.t0 - pre_s
    t1 = stance.t1 + post_s
    i0 = int(np.searchsorted(times, t0, side="left"))
    i1 = int(np.searchsorted(times, t1, side="right") - 1)
    i0 = max(0, i0)
    i1 = min(times.size - 1, max(i0, i1))
    return i0, i1
