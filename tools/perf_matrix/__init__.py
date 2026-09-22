# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
for p in (repo_root, repo_root / "core" / "builder", repo_root / "apps" / "benchmark"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inspect  # noqa: E402
from . import core, types  # noqa: E402

for mod in (core, types):
    for name, obj in inspect.getmembers(mod):
        globals()[name] = obj
