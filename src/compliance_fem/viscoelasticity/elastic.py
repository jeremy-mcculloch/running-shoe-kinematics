"""Elastic identity viscoelastic mapper (F_e = F_VE)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper


class ElasticMaterial(ViscoelasticForceMapper):
    """Pass-through constitutive map for purely elastic replay."""

    def __init__(self, *, n_components: int = 2) -> None:
        super().__init__(n_components=n_components)
        self.reset()

    def _reset_internal(self) -> None:
        return None

    def _step(
        self,
        FVE: NDArray[np.float64],
        t: float,
        dt: float,
    ) -> NDArray[np.float64]:
        return np.asarray(FVE, dtype=np.float64).copy()

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "model": "elastic",
            "n_components": self.n_components,
            "n_modes": 0,
            "note": "identity map F_e = F_VE",
        }
