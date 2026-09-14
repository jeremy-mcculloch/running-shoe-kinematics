"""Fung continuous-spectrum viscoelasticity via logarithmic Prony quadrature."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expi

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper
from compliance_fem.viscoelasticity.exponential import (
    ExponentialSpectrum,
    fung_prony_weights,
    prony_forward_step,
    prony_inverse_step,
)


def fung_G_analytic(t: float | np.ndarray, C: float, tau1: float, tau2: float) -> np.ndarray:
    """Normalized Fung relaxation ``G(t)`` using ``E_1``.

    ``G(t) = [1 + C (E1(t/τ2) - E1(t/τ1))] / [1 + C ln(τ2/τ1)]``.

    At ``t=0``, ``E1`` diverges but the difference tends to ``ln(τ2/τ1)``,
    so ``G(0)=1``.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    denom = 1.0 + C * np.log(tau2 / tau1)
    out = np.empty_like(t_arr)
    zero = t_arr == 0.0
    out[zero] = 1.0
    tt = t_arr[~zero]
    # E1(z) = -Ei(-z) for z > 0
    e1_hi = -expi(-tt / tau2)
    e1_lo = -expi(-tt / tau1)
    out[~zero] = (1.0 + C * (e1_hi - e1_lo)) / denom
    return out


def fung_G_prony(
    t: float | np.ndarray,
    g_inf: float,
    h: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Discrete Prony approximation of Fung ``G(t)``."""
    t_arr = np.asarray(t, dtype=np.float64)
    G = np.full_like(t_arr, g_inf, dtype=np.float64)
    for hk, tk in zip(np.asarray(h, dtype=np.float64), np.asarray(tau, dtype=np.float64)):
        G = G + hk * np.exp(-t_arr / tk)
    return G


class FungMaterial(ViscoelasticForceMapper):
    """Fung bounded ``1/τ`` spectrum on ``[τ1, τ2]``.

    The continuous relaxation is approximated by a logarithmically spaced Prony
    series sharing the generic exponential inverse engine. Online evaluation
    inverts ``F_VE = G * dF_e`` without storing the full force history.
    """

    def __init__(
        self,
        *,
        C: float,
        tau1: float,
        tau2: float,
        num_modes: int = 32,
        n_components: int = 2,
    ) -> None:
        if C <= 0.0:
            raise ValueError("C must be > 0")
        if not (tau1 > 0.0 and tau2 > tau1):
            raise ValueError("require 0 < tau1 < tau2")
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")

        super().__init__(n_components=n_components)
        self.C = float(C)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.num_modes = int(num_modes)

        g_inf, h, taus = fung_prony_weights(self.C, self.tau1, self.tau2, self.num_modes)
        self.g_inf = g_inf
        self.spectrum = ExponentialSpectrum(taus, h, g_inf=g_inf)
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
        FE, self._q = prony_inverse_step(
            FVE=FVE,
            FVE_prev=self._FVE,
            FE_prev=self._FE,
            q=self._q,
            g_inf=self.g_inf,
            h=self.spectrum.weight,
            tau=self.spectrum.tau,
            dt=dt,
        )
        return FE

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "model": "fung",
            "C": self.C,
            "tau1": self.tau1,
            "tau2": self.tau2,
            "g_inf": self.g_inf,
            "num_modes": self.spectrum.n_modes,
            "n_components": self.n_components,
            "approximation": (
                "logarithmic quadrature of S(τ)=C/τ on [τ1, τ2] → Prony G; "
                "online inverse of F_VE = G * dF_e"
            ),
        }

    def forward_history(
        self,
        times: np.ndarray,
        FE_history: np.ndarray,
    ) -> np.ndarray:
        """Forward Prony map for round-trip tests."""
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        FE = np.asarray(FE_history, dtype=np.float64)
        if FE.ndim == 1:
            FE = FE.reshape(-1, 1)
        q = np.zeros((self.spectrum.n_modes, self.n_components), dtype=np.float64)
        out = np.empty_like(FE)
        FE_prev = np.zeros(self.n_components, dtype=np.float64)
        if float(t[0]) > 0.0:
            out[0], q = prony_forward_step(
                FE=FE[0],
                FE_prev=FE_prev,
                q=q,
                g_inf=self.g_inf,
                h=self.spectrum.weight,
                tau=self.spectrum.tau,
                dt=float(t[0]),
            )
            FE_prev = FE[0].copy()
        else:
            out[0] = FE[0].copy()
            FE_prev = FE[0].copy()
        for i in range(1, t.size):
            out[i], q = prony_forward_step(
                FE=FE[i],
                FE_prev=FE_prev,
                q=q,
                g_inf=self.g_inf,
                h=self.spectrum.weight,
                tau=self.spectrum.tau,
                dt=float(t[i] - t[i - 1]),
            )
            FE_prev = FE[i].copy()
        return out
