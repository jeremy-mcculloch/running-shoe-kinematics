"""Shared small-mesh fixtures for the interval-contact tests.

``*_case`` and ``*_lookup_full`` keep the per-node bases in memory
(``store_fields=True``); ``*_lookup`` is the default compact lookup whose nodal
fields are recomputed on demand.

The meshes are coarse synthetic measured soles (:func:`synthetic_sole_geometry`):
a flat-topped sole that drops to a point toe, with every sample on one
``x / L = k / 12`` grid so ``N_b = 13`` bottom and 12 top nodes are uniformly
spaced. A flat sole has ``y_b = 0``; curved soles set ``rocker``.
"""

from __future__ import annotations

import numpy as np
import pytest

from compliance_fem.fem.compliance import compute_compliance
from compliance_fem.contact.config import SoleConfig
from compliance_fem.contact.lookup import get_compliance_block_matrix, generate_contact_lookup
from compliance_fem.geometry.profile import NormalizedSoleGeometry, validate_normalized_geometry

# L - toe_length = 9 L / 12 lies on the top-node grid (no off-grid ramp warning).
SMALL_L = 0.25
SMALL_TOE_LENGTH = 0.0625
SMALL_KAPPA = 160.0
SMALL_N = 13


def synthetic_sole_geometry(
    rocker: float = 0.0,
    apex: float = 0.5,
    n: int = SMALL_N,
    height: float = 0.06,
    top_slope: float = 0.0,
) -> NormalizedSoleGeometry:
    """Normalized sole with ``n`` bottom and ``n - 1`` top samples on the ``x = k / (n - 1)`` grid.

    The bottom is ``y_b = rocker ((x - apex)^2 - apex^2) / max(apex, 1 - apex)^2``
    (flat for ``rocker = 0``; lowest at ``apex`` otherwise). The top is
    ``height - top_slope x`` (horizontal reference chord for ``top_slope = 0``) and
    drops in one straight segment to the point toe. The foam interface sits at ``height / 2`` and turns
    down to the second-to-last bottom sample; the plate spans interface samples
    ``1 .. n - 4``.
    """
    x = np.linspace(0.0, 1.0, n)
    half = max(apex, 1.0 - apex)
    bottom = rocker * (((x - apex) / half) ** 2 - (apex / half) ** 2)
    bottom[0] = 0.0
    toe_y = float(bottom[-1])
    top = height - top_slope * x
    top[-1] = toe_y
    xi = x[:-1]
    interface = np.full(n - 1, 0.5 * height)
    interface[-1] = bottom[-2]
    landmarks = {
        "toe_tip": (1.0, toe_y),
        "heel_top": (0.0, height),
        "heel_bottom": (0.0, 0.0),
        "foam_interface_toe_endpoint": (float(xi[-1]), float(interface[-1])),
        "foam_interface_heel_endpoint": (0.0, 0.5 * height),
        "plate_toe_endpoint": (float(xi[-3]), 0.5 * height),
        "plate_heel_endpoint": (float(xi[1]), 0.5 * height),
    }
    curves = {
        "top_surface": np.column_stack([x, top]),
        "bottom_surface": np.column_stack([x, bottom]),
        "foam_interface": np.column_stack([xi, interface]),
    }
    geom = NormalizedSoleGeometry(landmarks=landmarks, curves=curves, source_filename="synthetic")
    validate_normalized_geometry(geom)
    return geom


def small_config(
    rocker: float = 0.0,
    apex: float = 0.5,
    n: int = SMALL_N,
    L: float = SMALL_L,
    height: float = 0.06,
    top_slope: float = 0.0,
    **overrides,
) -> SoleConfig:
    """Coarse synthetic measured sole of length ``L``; ``overrides`` go to :class:`SoleConfig`.

    The mesh size slightly exceeds the toe-drop segment, so Gmsh adds no boundary
    nodes and the top / bottom nodes are exactly the geometry samples.
    """
    toe_drop = float(np.hypot(1.0 / (n - 1), height))
    params = dict(
        shoe_length_mm=L * 1000.0,
        normalized_geometry=synthetic_sole_geometry(rocker, apex, n, height, top_slope),
        mesh_size_m=1.05 * toe_drop * L,
        toe_refinement=1.0,
        heel_corner_refinement=1.0,
        interface_refinement=1.0,
        plate_end_refinement=1.0,
        curvature_max_turn_deg=90.0,
    )
    params.update(overrides)
    return SoleConfig(**params)


def _build(config: SoleConfig):
    result = compute_compliance(config)
    lookup = generate_contact_lookup(
        get_compliance_block_matrix(result),
        toe_length=SMALL_TOE_LENGTH,
        kappa=SMALL_KAPPA,
        fem_result=result,
        store_fields=True,
    )
    return result, lookup


def _compact(case):
    result, _ = case
    return generate_contact_lookup(
        get_compliance_block_matrix(result), toe_length=SMALL_TOE_LENGTH, kappa=SMALL_KAPPA, fem_result=result
    )


@pytest.fixture(scope="session")
def flat_case():
    """Flat sole, N_b = 13."""
    return _build(small_config())


@pytest.fixture(scope="session")
def rocker_case():
    """Symmetric curved (rocker) sole: lowest at mid-length, so vertical load gives interior contact."""
    return _build(small_config(rocker=0.008, apex=0.5))


@pytest.fixture(scope="session")
def asym_case():
    """Asymmetric curved sole (apex at 0.35 L): off-centre interior contact."""
    return _build(small_config(rocker=0.008, apex=0.35))


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
