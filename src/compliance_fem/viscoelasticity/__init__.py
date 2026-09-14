"""Viscoelastic force history → equivalent elastic force history.

Maps a viscoelastic generalized force F_VE(t) to the elastic force
F_e(t) satisfying F_VE = G * dF_e / dt via the creep convolution
F_e = J * dF_VE / dt.

Online API
----------
>>> material = create_material(SLSConfig(g_inf=0.5, tau_r=1.0))
>>> material.reset()
>>> Fe = material.update(FVE_i, t_i)

See ``docs/viscoelasticity.md`` for mathematics, approximations, and CLI usage.
"""

from __future__ import annotations

from compliance_fem.viscoelasticity.base import ViscoelasticForceMapper
from compliance_fem.viscoelasticity.factory import (
    FractionalConfig,
    FungConfig,
    MaterialConfig,
    SLSConfig,
    create_material,
)
from compliance_fem.viscoelasticity.fractional import FractionalMaterial
from compliance_fem.viscoelasticity.fung import FungMaterial
from compliance_fem.viscoelasticity.sls import SLSMaterial

__all__ = [
    "ViscoelasticForceMapper",
    "SLSMaterial",
    "FractionalMaterial",
    "FungMaterial",
    "SLSConfig",
    "FractionalConfig",
    "FungConfig",
    "MaterialConfig",
    "create_material",
]
