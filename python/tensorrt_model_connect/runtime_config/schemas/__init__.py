"""Per-feature config schemas.

Each feature owns a module in this package and registers its namespace at
import time. To add a new feature, drop a new module here — nothing else
in the config machinery needs to change. That's the scalability contract.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def load_all() -> list[str]:
    """Import every schema module in this package and register their namespaces.

    Returns the list of namespaces that got registered. If a module is
    already in :data:`sys.modules` (e.g. a test cleared the registry
    between cases) it is re-imported via :func:`importlib.reload` so the
    module-level :func:`register_schema` call runs again.
    """
    from tensorrt_model_connect.runtime_config import registered_namespaces
    before = set(registered_namespaces())
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        qual = f"{__name__}.{info.name}"
        if qual in sys.modules:
            importlib.reload(sys.modules[qual])
        else:
            importlib.import_module(qual)
    after = set(registered_namespaces())
    return sorted(after - before)
