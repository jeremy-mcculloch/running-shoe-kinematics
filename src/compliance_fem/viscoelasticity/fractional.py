"""Fractional power-law creep mapper via sum-of-exponentials."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper
from compliance_fem.viscoelasticity.exponential import (
    ExponentialSpectrum,
    fractional_kernel_weights,
    integrate_force_driven_exponentials,
    logspaced_taus,
)


class FractionalMaterial(ViscoelasticForceMapper):
    """Creep ``J(t) = (1/E0) (t/T)^α`` with ``0 < α < 1``.

    Online evaluation uses a sum-of-exponentials approximation

    ``t^{α-1} ≈ Σ_k a_k exp(-t/τ_k)``

    over ``[tau_min, tau_max]``, then maintains

    ``q_k(t) = ∫_0^t exp(-(t-τ)/τ_k) F_VE(τ) dτ``

    so

    ``F_e ≈ (α / (E0 T^α)) Σ_k a_k q_k``.

    A pure fractional spring-pot has no intrinsic τ bounds; any finite Prony
    approximation is accurate only over a finite time/frequency window.
    """

    def __init__(
        self,
        *,
        E0: float,
        T: float,
        alpha: float,
        num_modes: int = 32,
        tau_min: float | None = None,
        tau_max: float | None = None,
        n_components: int = 2,
    ) -> None:
        if E0 <= 0.0:
            raise ValueError("E0 must be > 0 (force/area)")
        if T <= 0.0:
            raise ValueError("T must be > 0 (time)")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")

        super().__init__(n_components=n_components)
        self.E0 = float(E0)
        self.T = float(T)
        self.alpha = float(alpha)
        self.num_modes = int(num_modes)
        # Default window: decades covering typical simulation times.
        self.tau_min = float(tau_min) if tau_min is not None else 1.0e-6 * self.T
        self.tau_max = float(tau_max) if tau_max is not None else 1.0e6 * self.T
        if not (self.tau_min > 0.0 and self.tau_max > self.tau_min):
            raise ValueError("require 0 < tau_min < tau_max")

        taus = logspaced_taus(self.tau_min, self.tau_max, self.num_modes)
        weights = fractional_kernel_weights(
            self.alpha,
            taus,
            t_min=self.tau_min * 3.0,
            t_max=self.tau_max / 3.0,
        )
        self.spectrum = ExponentialSpectrum(taus, weights)
        self._pref = self.alpha / (self.E0 * self.T**self.alpha)
        self._q = np.zeros((self.spectrum.n_modes, self.n_components), dtype=np.float64)
        self.reset()

    def _reset_internal(self) -> None:
        self._q.fill(0.0)

    def _step(
        self,
        FVE: NDArray[np.float64],
        t: float,
        dt: float,
    ) -> NDArray[np.float64]:
        self._q = integrate_force_driven_exponentials(
            F=FVE,
            F_prev=self._FVE,
            q=self._q,
            tau=self.spectrum.tau,
            dt=dt,
        )
        # F_e = pref * Σ a_k q_k
        return self._pref * np.einsum("m,mc->c", self.spectrum.weight, self._q)

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "model": "fractional",
            "E0": self.E0,
            "T": self.T,
            "alpha": self.alpha,
            "num_modes": self.spectrum.n_modes,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "n_components": self.n_components,
            "approximation": (
                "sum-of-exponentials for t^{α-1} on [tau_min, tau_max]; "
                "no intrinsic τ bounds for a pure spring-pot"
            ),
        }
