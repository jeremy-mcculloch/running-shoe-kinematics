"""Narrowing helpers for scikit-fem attributes annotated ``Optional``.

scikit-fem types ``Basis.nodal_dofs`` (and the facet / edge / interior tables),
``Mesh.facets`` and ``Mesh.boundaries`` as possibly ``None``. A constructed basis
always fills its DOF tables, a 2D mesh always has facets, and the sole mesh always
tags its boundaries, so these helpers only turn that invariant into a type.
"""

from __future__ import annotations

import numpy as np
from skfem import Mesh


def dof_table(dofs: np.ndarray | None) -> np.ndarray:
    """Return a basis DOF table, which every constructed scikit-fem basis sets."""
    if dofs is None:
        raise RuntimeError("scikit-fem basis has no DOF table; it was not fully constructed.")
    return dofs


def mesh_facets(mesh: Mesh) -> np.ndarray:
    """Return the facet-to-node table of ``mesh``."""
    if mesh.facets is None:
        raise ValueError("Mesh has no facet table.")
    return mesh.facets


def mesh_boundaries(mesh: Mesh) -> dict[str, np.ndarray]:
    """Return the tagged boundary facets of ``mesh``."""
    if mesh.boundaries is None:
        raise ValueError("Mesh has no tagged boundaries.")
    return mesh.boundaries
