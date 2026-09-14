"""Standard linear solid (SLS) creep inverse mapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper
from compliance_fem.viscoelasticity.exponential import (
    prony_forward_step,
    stable_one_minus_exp_over_x,
)


class SLSMaterial(ViscoelasticForceMapper):
    """Normalized SLS with

    ``G(t) = g_inf + (1-g_inf) exp(-t/tau_r)``

    and creep

    ``J(t) = 1 + (1/g_inf - 1)(1 - exp(-t/tau_c))``, ``tau_c = tau_r / g_inf``.

    Online update uses the hereditary state

    ``s(t) = int_0^t exp(-(t-τ)/tau_c) dF_VE(τ)``

    so that ``F_e = F_VE / g_inf - (1/g_inf - 1) s``.
    """

    def __init__(
        self,
        *,
        g_inf: float,
        tau_r: float,
        n_components: int = 2,
    ) -> None:
        if not (0.0 < g_inf <= 1.0):
            raise ValueError("g_inf must satisfy 0 < g_inf <= 1")
        if tau_r <= 0.0:
            raise ValueError("tau_r must be > 0")
        super().__init__(n_components=n_components)
        self.g_inf = float(g_inf)
        self.tau_r = float(tau_r)
        self.tau_c = self.tau_r / self.g_inf
        self.beta = 1.0 / self.g_inf - 1.0  # J_inf - 1
        self._s = np.zeros(self.n_components, dtype=np.float64)
        self.reset()

    def _reset_internal(self) -> None:
        self._s.fill(0.0)

    def _step(
        self,
        FVE: NDArray[np.float64],
        t: float,
        dt: float,
    ) -> NDArray[np.float64]:
        if dt == 0.0:
            # Instantaneous glassy response of J: J(0)=1 => dFE = dFVE.
            return self._FE + (FVE - self._FVE)

        # Piecewise-linear F_VE: constant rate r = dFVE/dt.
        r = (FVE - self._FVE) / dt
        alpha = float(np.exp(-dt / self.tau_c))
        # s^+ = α s + r τ_c (1-α) = α s + ΔFVE * (1-α)/(dt/τ_c)
        phi = float(stable_one_minus_exp_over_x(dt / self.tau_c))
        self._s = alpha * self._s + (FVE - self._FVE) * phi
        # F_e = (1+β) F_VE - β s = F_VE/g_inf - β s
        return (1.0 / self.g_inf) * FVE - self.beta * self._s

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "model": "sls",
            "g_inf": self.g_inf,
            "tau_r": self.tau_r,
            "tau_c": self.tau_c,
            "n_components": self.n_components,
            "n_modes": 1,
        }

    def forward_history(
        self,
        times: np.ndarray,
        FE_history: np.ndarray,
    ) -> np.ndarray:
        """Reference forward map ``F_VE = G * dF_e`` via Prony recurrence."""
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        FE = np.asarray(FE_history, dtype=np.float64)
        if FE.ndim == 1:
            FE = FE.reshape(-1, 1)
        h = np.array([1.0 - self.g_inf], dtype=np.float64)
        tau = np.array([self.tau_r], dtype=np.float64)
        q = np.zeros((1, self.n_components), dtype=np.float64)
        out = np.empty_like(FE)
        FE_prev = np.zeros(self.n_components, dtype=np.float64)
        t_prev = float(t[0])
        # Seed first sample from zero IC at t=0 if needed.
        if t_prev > 0.0:
            FVE0, q = prony_forward_step(
                FE=FE[0],
                FE_prev=FE_prev,
                q=q,
                g_inf=self.g_inf,
                h=h,
                tau=tau,
                dt=t_prev,
            )
            out[0] = FVE0
            FE_prev = FE[0].copy()
        else:
            out[0] = FE[0].copy()  # G(0)=1 and zero history
            FE_prev = FE[0].copy()
        for i in range(1, t.size):
            dt = float(t[i] - t[i - 1])
            FVE_i, q = prony_forward_step(
                FE=FE[i],
                FE_prev=FE_prev,
                q=q,
                g_inf=self.g_inf,
                h=h,
                tau=tau,
                dt=dt,
            )
            out[i] = FVE_i
            FE_prev = FE[i].copy()
        return out
