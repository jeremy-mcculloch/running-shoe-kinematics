"""Abstract stateful viscoelastic force mapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ViscoelasticForceMapper(ABC):
    """Online mapper: ``F_VE(t) -> F_e(t)`` for a generalized force vector.

    The temporal constitutive model is applied componentwise. Coupling between
    components belongs in the downstream elastic operator, not here.

    Initial conditions are ``F_VE(0)=0`` and ``F_e(0)=0``. Call :meth:`reset`
    before each independent history.
    """

    def __init__(self, *, n_components: int = 2) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        self._initialized = False
        self._t: float = 0.0
        self._FVE = np.zeros(self.n_components, dtype=np.float64)
        self._FE = np.zeros(self.n_components, dtype=np.float64)

    @property
    def time(self) -> float:
        return float(self._t)

    @property
    def FVE(self) -> NDArray[np.float64]:
        return self._FVE.copy()

    @property
    def FE(self) -> NDArray[np.float64]:
        return self._FE.copy()

    def reset(self) -> None:
        """Reset to the zero initial state at ``t=0``."""
        self._t = 0.0
        self._FVE.fill(0.0)
        self._FE.fill(0.0)
        self._reset_internal()
        self._initialized = True

    @abstractmethod
    def _reset_internal(self) -> None:
        """Clear model-specific internal memory."""

    @abstractmethod
    def _step(
        self,
        FVE: NDArray[np.float64],
        t: float,
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance from the previous sample to ``(t, FVE)`` and return ``FE``."""

    def update(self, FVE_i: ArrayLike, t_i: float) -> NDArray[np.float64]:
        """Ingest one viscoelastic force sample and return the elastic force.

        Parameters
        ----------
        FVE_i
            Force components at time ``t_i`` (length ``n_components``).
        t_i
            Sample time. Must be ``>=`` the previous sample time.

        Returns
        -------
        ndarray
            Elastic force ``F_e(t_i)`` (copy, shape ``(n_components,)``).
        """
        if not self._initialized:
            self.reset()

        FVE = np.asarray(FVE_i, dtype=np.float64).reshape(-1)
        if FVE.shape != (self.n_components,):
            raise ValueError(
                f"FVE_i must have shape ({self.n_components},), got {FVE.shape}"
            )
        t = float(t_i)
        if not np.isfinite(t):
            raise ValueError(f"t_i must be finite, got {t_i!r}")
        if not np.all(np.isfinite(FVE)):
            raise ValueError("FVE_i must be finite")

        dt = t - self._t
        if dt < 0.0:
            raise ValueError(
                f"time must be non-decreasing: t={t} < previous t={self._t}"
            )

        if dt == 0.0 and t == self._t:
            # Zero timestep: keep hereditary state; allow force overwrite only
            # when this is a repeated stamp of the same instant.
            if np.allclose(FVE, self._FVE, rtol=0.0, atol=0.0):
                return self._FE.copy()
            # Same time, different force: treat as instantaneous update with
            # no additional hereditary advance (dt=0 path inside _step).
            FE = self._step(FVE, t, 0.0)
        else:
            FE = self._step(FVE, t, dt)

        self._t = t
        np.copyto(self._FVE, FVE)
        np.copyto(self._FE, FE)
        return FE.copy()

    def evaluate_history(
        self,
        times: ArrayLike,
        FVE_history: ArrayLike,
        *,
        reset: bool = True,
    ) -> NDArray[np.float64]:
        """Batch evaluation implemented exclusively via :meth:`update`.

        Parameters
        ----------
        times
            Monotone sample times, shape ``(n,)``.
        FVE_history
            Viscoelastic forces, shape ``(n, n_components)``.
        reset
            If True (default), call :meth:`reset` before the loop.
        """
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        FVE = np.asarray(FVE_history, dtype=np.float64)
        if FVE.ndim == 1:
            if self.n_components != 1:
                raise ValueError("1-D FVE_history requires n_components=1")
            FVE = FVE.reshape(-1, 1)
        if FVE.shape != (t.size, self.n_components):
            raise ValueError(
                f"FVE_history must have shape ({t.size}, {self.n_components}), "
                f"got {FVE.shape}"
            )
        if t.size == 0:
            return np.zeros((0, self.n_components), dtype=np.float64)
        if np.any(np.diff(t) < 0.0):
            raise ValueError("times must be monotonically non-decreasing")

        if reset:
            self.reset()

        out = np.empty((t.size, self.n_components), dtype=np.float64)
        for i in range(t.size):
            out[i] = self.update(FVE[i], float(t[i]))
        return out

    @abstractmethod
    def parameter_summary(self) -> dict[str, Any]:
        """Human-readable parameter dict for CLI / logging."""
