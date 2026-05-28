"""Auto-discover Torch-TRT family plugins from this package.

Same pattern as tensorrt_model_connect/families: any .py file in this directory
(excluding _-prefixed and base.py) that exposes a module-level ``plugin``
attribute is automatically registered.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from .base import TorchTrtFamilyPlugin

ALL_PLUGINS: list[TorchTrtFamilyPlugin] = []

_pkg_dir = str(Path(__file__).parent)
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name.startswith("_") or _name == "base":
        continue
    try:
        _mod = importlib.import_module(f"{__name__}.{_name}")
    except ImportError:
        continue
    _plugin = getattr(_mod, "plugin", None)
    if _plugin is not None:
        ALL_PLUGINS.append(_plugin)


def find_plugin(model_ref: Any) -> TorchTrtFamilyPlugin | None:
    """Find the first plugin that matches the given model reference.

    `model_ref` is usually either:
    - a raw `model_type` string, or
    - a parsed `ModelConfig`

    Most plugins only key off `model_type`. A few models, such as
    Chronos-Bolt, need additional config fields (`architectures`,
    `chronos_config`) to disambiguate them from generic T5 checkpoints.
    """
    model_type = str(getattr(model_ref, "model_type", model_ref))
    for p in ALL_PLUGINS:
        matches_config = getattr(p, "matches_config", None)
        if callable(matches_config) and matches_config(model_ref):
            return p
        if p.matches(model_type):
            return p
    return None
