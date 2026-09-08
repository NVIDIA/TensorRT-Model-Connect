# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    """Replace the console adapter with its co-located native product CLI."""
    executable = Path(__file__).resolve().parent / "bin" / "trtmc"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"native trtmc executable is missing or not executable: {executable}")
    os.execv(executable, [str(executable), *sys.argv[1:]])
