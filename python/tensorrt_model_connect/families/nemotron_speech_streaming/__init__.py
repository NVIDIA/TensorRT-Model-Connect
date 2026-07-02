# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

from . import plugin as _plugin

globals().update({
    _name: _value
    for _name, _value in vars(_plugin).items()
    if not _name.startswith("__")
})
__all__ = [
    _name for _name in globals()
    if not _name.startswith("__") and _name != "_plugin"
]


class _FamilyModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not name.startswith("__") and name != "_plugin":
            setattr(_plugin, name, value)


sys.modules[__name__].__class__ = _FamilyModule
