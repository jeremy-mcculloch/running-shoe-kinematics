"""Gmsh helpers that work under Streamlit / non-main threads."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator

import gmsh

# Gmsh's native API is process-global and not thread-safe.
_GMSH_LOCK = threading.RLock()


def initialize_gmsh(*, read_config_files: bool = True) -> None:
    """Initialize Gmsh without installing a SIGINT handler.

    Streamlit's script runner is not the main interpreter thread, so the
    default ``gmsh.initialize(interruptible=True)`` raises
    ``ValueError: signal only works in main thread of the main interpreter``.
    Batch meshing does not need that handler.

    If a previous failed init left Gmsh half-open (native API up, Python
    exception after), finalize first so the next init is clean.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=read_config_files, interruptible=False)


@contextmanager
def gmsh_session() -> Iterator[None]:
    """Serialize Gmsh initialize → mesh → finalize across threads."""
    with _GMSH_LOCK:
        yield
