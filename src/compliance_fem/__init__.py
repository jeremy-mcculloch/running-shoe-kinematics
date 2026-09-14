"""Finite-element compliance matrices for plane-strain elastic bodies."""

from compliance_fem.config import LayeredPlateConfig, ProblemConfig
from compliance_fem.compliance import ComplianceResult, compute_compliance

__version__ = "0.2.0"
__all__ = [
    "ProblemConfig",
    "LayeredPlateConfig",
    "ComplianceResult",
    "compute_compliance",
    "__version__",
]
