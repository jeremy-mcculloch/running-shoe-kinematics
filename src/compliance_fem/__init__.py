"""Finite-element compliance matrices for plane-strain elastic bodies."""

from compliance_fem.contact.config import SoleConfig
from compliance_fem.fem.compliance import ComplianceResult, compute_compliance

__version__ = "0.2.0"
__all__ = [
    "SoleConfig",
    "ComplianceResult",
    "compute_compliance",
    "__version__",
]
