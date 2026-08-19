# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy public surface for the LFM2 family."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any


_plugin_module = None


def _load_plugin_module():
    global _plugin_module
    if _plugin_module is None:
        _plugin_module = importlib.import_module(f"{__name__}.plugin")
        globals().update(
            {
                name: value
                for name, value in vars(_plugin_module).items()
                if not name.startswith("__")
            }
        )
    return _plugin_module


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    plugin_module = _load_plugin_module()
    if name == "plugin":
        return plugin_module.plugin
    try:
        return getattr(plugin_module, name)
    except AttributeError:
        raise AttributeError(name) from None


def __dir__() -> list[str]:
    plugin_module = _load_plugin_module()
    return sorted(
        set(globals()) | {name for name in vars(plugin_module) if not name.startswith("__")}
    )


class _FamilyModule(types.ModuleType):
    def __setattr__(self, name, value):
        if name == "plugin" and isinstance(value, types.ModuleType):
            super().__setattr__("_plugin_module", value)
            super().__setattr__("plugin", value.plugin)
            return
        super().__setattr__(name, value)
        if (
            not name.startswith("__")
            and name not in {"_plugin_module", "plugin"}
            and not isinstance(value, types.ModuleType)
        ):
            setattr(_load_plugin_module(), name, value)


sys.modules[__name__].__class__ = _FamilyModule
