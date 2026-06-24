"""Engine definition registry.

The repository currently ships only the native TRT Network API build path.
This module is kept as a small compatibility shim for older callers that
imported the optional backend registry.
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import BuildBackend

_engine_defs: Dict[str, BuildBackend] = {}


def _discover() -> None:
    return


def get_engine_def(name: str) -> Optional[BuildBackend]:
    """Look up an optional engine def by name. Returns None if not available."""
    _discover()
    return _engine_defs.get(name)


def list_engine_defs() -> Dict[str, BuildBackend]:
    """Return all available engine defs."""
    _discover()
    return dict(_engine_defs)


# Backward-compat aliases
get_backend = get_engine_def
list_backends = list_engine_defs
