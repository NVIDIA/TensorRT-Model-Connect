# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-feature config schemas.

Shared features own modules in this package. Model-family specific features
own ``runtime_config_schema.py`` sidecars under
``tensorrt_model_connect/families/<family>/``. Loading sidecars by file path
keeps schema discovery generic without importing family model modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path


def _load_module_from_path(qualname: str, path: Path) -> None:
    if qualname in sys.modules:
        del sys.modules[qualname]
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime config schema module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)


def _iter_family_schema_modules() -> list[tuple[str, Path]]:
    import tensorrt_model_connect.families as families

    modules: list[tuple[str, Path]] = []
    for root in families.__path__:
        families_root = Path(root)
        for family_dir in sorted(path for path in families_root.iterdir() if path.is_dir()):
            if family_dir.name.startswith("_"):
                continue
            schema_path = family_dir / "runtime_config_schema.py"
            if schema_path.is_file():
                modules.append((
                    f"{families.__name__}.{family_dir.name}.runtime_config_schema",
                    schema_path,
                ))
    return modules


def load_all() -> list[str]:
    """Import every schema module in this package and register their namespaces.

    Returns the list of namespaces that got registered. If a module is
    already in :data:`sys.modules` (e.g. a test cleared the registry
    between cases), shared schema modules are reloaded and model-family
    schema sidecars are executed again so their module-level
    :func:`register_schema` calls run.
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
    for qual, path in _iter_family_schema_modules():
        _load_module_from_path(qual, path)
    after = set(registered_namespaces())
    return sorted(after - before)
