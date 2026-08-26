# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-origin guard for the physically isolated server package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_SERVER_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "python"
_TEST_INSTALLED_WHEEL = os.environ.get("TRTMC_TEST_INSTALLED_WHEEL") == "1"

if not _TEST_INSTALLED_WHEEL and str(_SERVER_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_SOURCE_ROOT))

if _TEST_INSTALLED_WHEEL:
    import trtmc_server as _installed_server

    installed_path = Path(_installed_server.__file__).resolve()
    if installed_path.is_relative_to(_SERVER_SOURCE_ROOT):
        raise RuntimeError(
            "TRTMC_TEST_INSTALLED_WHEEL=1 imported trtmc_server "
            f"from the source checkout: {installed_path}"
        )
