#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one TRTMC-to-reference comparison through the validation backend."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation import engine  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Run comparison through the validation engine."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    return engine.main(["eval", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
