#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and report the TRTMC release performance matrix."""

import sys
from pathlib import Path

# Add the root directory and sources to path to allow importing benchmark modules
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "core" / "builder"))
sys.path.insert(0, str(repo_root / "apps" / "benchmark"))

from tools.perf_matrix.cli import main
from tools.perf_matrix.core import *
from tools.perf_matrix.types import *

if __name__ == "__main__":
    sys.exit(main())
