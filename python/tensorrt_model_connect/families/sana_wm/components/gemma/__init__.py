# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

_plugin = None


def _load_plugin_module():
    global _plugin
    if _plugin is None:
        _plugin = importlib.import_module(f"{__name__}.plugin")
        globals().update({
            _name: _value
            for _name, _value in vars(_plugin).items()
            if not _name.startswith("__")
        })
    return _plugin


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    plugin_module = _load_plugin_module()
    if name == "plugin":
        return getattr(plugin_module, "plugin")
    try:
        return getattr(plugin_module, name)
    except AttributeError:
        raise AttributeError(name) from None


def __dir__() -> list[str]:
    plugin_module = _load_plugin_module()
    return sorted(set(globals()) | {
        _name for _name in vars(plugin_module) if not _name.startswith("__")
    })


class _FamilyModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            not name.startswith("__")
            and name not in {"_plugin", "plugin"}
            and not isinstance(value, types.ModuleType)
        ):
            setattr(_load_plugin_module(), name, value)


sys.modules[__name__].__class__ = _FamilyModule
