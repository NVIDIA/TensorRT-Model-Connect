# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-discover check modules in this package."""

import importlib
import pkgutil
from pathlib import Path

_pkg_dir = str(Path(__file__).parent)
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_name}")
