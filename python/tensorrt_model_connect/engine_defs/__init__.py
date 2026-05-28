"""Engine definition registry -- discovers and dispatches to engine def methods.

The default method is 'trt' (TRT Network API). Optional methods like
'torch_trt' are discovered if their dependencies are installed.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Optional

from .base import BuildBackend

logger = logging.getLogger(__name__)

_engine_defs: Dict[str, BuildBackend] = {}
_discovered = False

# Known engine definition modules -- maps CLI name to importable module path.
# Each module must expose a module-level ``backend`` attribute implementing
# the BuildBackend protocol.
_ENGINE_DEF_MODULES = {
    "torch_trt": ".engine_defs.torch_trt",
}

# CLI aliases (user-facing choice string -> canonical engine def name)
_CLI_ALIASES = {
    "torchtrt": "torch_trt",
}


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    _discovered = True

    for name, mod_path in _ENGINE_DEF_MODULES.items():
        try:
            mod = importlib.import_module(mod_path, package="tensorrt_model_connect")
            backend = getattr(mod, "backend", None)
            if backend is not None and isinstance(backend, BuildBackend):
                _engine_defs[name] = backend
                logger.debug("Registered engine def: %s", name)
        except ImportError:
            logger.debug("Engine def %s not available (missing dependencies)", name)
        except Exception:
            logger.warning("Failed to load engine def %s", name, exc_info=True)


def get_engine_def(name: str) -> Optional[BuildBackend]:
    """Look up an engine def by name. Returns None if not available.

    Accepts both canonical names (``torch_trt``) and CLI aliases
    (``torchtrt``).
    """
    _discover()
    canonical = _CLI_ALIASES.get(name, name)
    return _engine_defs.get(canonical)


def list_engine_defs() -> Dict[str, BuildBackend]:
    """Return all available engine defs."""
    _discover()
    return dict(_engine_defs)


# Backward-compat aliases
get_backend = get_engine_def
list_backends = list_engine_defs
