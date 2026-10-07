"""Shared small-mesh fixtures for the interval-contact tests.

``*_case`` and ``*_lookup_full`` keep the per-node bases in memory
(``store_fields=True``); ``*_lookup`` is the default compact lookup whose nodal
fields are recomputed on demand.
"""

from __future__ import annotations

import pytest

from compliance_fem.compliance import compute_compliance
from compliance_fem.config import ProblemConfig
from compliance_fem.contact_lookup import from_compliance_result, generate_contact_lookup

# a = 9 L / 12 lies on the nx = 12 top-node grid (no off-grid ramp warning).
SMALL_L = 0.25
SMALL_A = 0.1875
SMALL_KAPPA = 160.0


def small_config(rocker: float = 0.0, apex: float = 0.5, nx: int = 12, ny: int = 3) -> ProblemConfig:
    return ProblemConfig(
        L=SMALL_L, H=0.03, E=1.0e6, nu=0.3, nx=nx, ny=ny,
        sole_rocker_height=rocker, sole_rocker_apex=apex,
    )


def _build(config: ProblemConfig):
    result = compute_compliance(config)
    lookup = generate_contact_lookup(from_compliance_result(result), a=SMALL_A, kappa=SMALL_KAPPA, store_fields=True)
    return result, lookup


def _compact(case):
    result, _ = case
    return generate_contact_lookup(from_compliance_result(result), a=SMALL_A, kappa=SMALL_KAPPA)


@pytest.fixture(scope="session")
def flat_case():
    """Flat sole, N_b = 13."""
    return _build(small_config())


@pytest.fixture(scope="session")
def rocker_case():
    """Symmetric curved (rocker) sole: lowest at mid-length, so vertical load gives interior contact."""
    return _build(small_config(rocker=0.002, apex=0.5))


@pytest.fixture(scope="session")
def asym_case():
    """Asymmetric curved sole (apex at 0.35 L): off-centre interior contact."""
    return _build(small_config(rocker=0.002, apex=0.35))


@pytest.fixture(scope="session")
def flat_lookup_full(flat_case):
    return flat_case[1]


@pytest.fixture(scope="session")
def rocker_lookup_full(rocker_case):
    return rocker_case[1]


@pytest.fixture(scope="session")
def asym_lookup_full(asym_case):
    return asym_case[1]


@pytest.fixture(scope="session")
def flat_lookup(flat_case):
    return _compact(flat_case)


@pytest.fixture(scope="session")
def rocker_lookup(rocker_case):
    return _compact(rocker_case)


@pytest.fixture(scope="session")
def asym_lookup(asym_case):
    return _compact(asym_case)
