"""Auto-discover family plugins from this package.

Any .py file or package in this directory (excluding _-prefixed and base.py)
that exposes a module-level ``plugin`` attribute is automatically registered.
Adding a new family = drop a .py file or package, zero edits to shared files.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FamilyPlugin


_PLUGINS_DISCOVERED = False


class _LazyPluginList(list["FamilyPlugin"]):
    def _materialize(self) -> None:
        _ensure_discovered()

    def __iter__(self):
        self._materialize()
        return super().__iter__()

    def __len__(self):
        self._materialize()
        return super().__len__()

    def __getitem__(self, index):
        self._materialize()
        return super().__getitem__(index)

    def __bool__(self):
        self._materialize()
        return super().__len__() != 0

    def __eq__(self, other):
        self._materialize()
        return super().__eq__(other)

    def __repr__(self):
        self._materialize()
        return super().__repr__()


_ALL_PLUGINS: list["FamilyPlugin"] = _LazyPluginList()


def _discover_plugins() -> None:
    # Scan every .py module or package in this directory.
    _pkg_dir = str(Path(__file__).parent)
    for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
        # Skip private modules and the base protocol definition.
        if _name.startswith("_") or _name == "base":
            continue
        try:
            _mod = importlib.import_module(f"{__name__}.{_name}")
        except ImportError:
            # Skip plugins whose dependencies (e.g. tensorrt) are not installed.
            continue
        _plugin = getattr(_mod, "plugin", None)
        if _plugin is not None:
            list.append(_ALL_PLUGINS, _plugin)


def _ensure_discovered() -> None:
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED or not isinstance(_ALL_PLUGINS, _LazyPluginList):
        return
    _PLUGINS_DISCOVERED = True
    _discover_plugins()


def find_plugin(model_type: object) -> "FamilyPlugin | None":
    """Find the first plugin that matches a model type or config object."""
    model_type_str = str(getattr(model_type, "model_type", model_type))
    for p in _ALL_PLUGINS:
        matches_config = getattr(p, "matches_config", None)
        if callable(matches_config) and matches_config(model_type):
            return p
        if p.matches(model_type_str):
            return p
    return None


def find_diffusion_plugin(pipeline_class: str) -> "FamilyPlugin | None":
    """Find the first plugin that handles the given diffusers pipeline class.

    Plugins declare supported pipeline classes via a ``pipeline_classes``
    attribute (list of class name strings). This enables auto-discovery
    without a hardcoded mapping dict.
    """
    for p in _ALL_PLUGINS:
        classes = getattr(p, 'pipeline_classes', None)
        if classes and pipeline_class in classes:
            return p
    return None
