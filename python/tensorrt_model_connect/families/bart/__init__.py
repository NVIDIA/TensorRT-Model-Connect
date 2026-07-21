# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART family plugin, loaded lazily for TensorRT-free runtime discovery."""

from __future__ import annotations

import importlib
import sys
import types

_plugin_module = None


def _load_plugin_module():
    global _plugin_module
    if _plugin_module is None:
        _plugin_module = importlib.import_module(f"{__name__}.plugin")
    return _plugin_module


class _FamilyModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "plugin":
            return _load_plugin_module().plugin
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "plugin" and isinstance(value, types.ModuleType):
            super().__setattr__("_plugin_module", value)
            value = value.plugin
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _FamilyModule

__all__ = ["plugin"]
