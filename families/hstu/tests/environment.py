# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the exact source-only HSTU oracle for selected GPU tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from .reference import REFERENCE_REVISION, _verify_source


_SOURCE_URL = "https://github.com/NVIDIA/recsys-examples.git"


def reference_source() -> Path:
    """Use an explicit checkout or fetch the pinned source into a local cache.

    No upstream package is installed. Only the test oracle reads these files;
    the model build and C++ deployment do not use this checkout.
    """
    explicit = os.environ.get("TRTMC_HSTU_REFERENCE_ROOT")
    if explicit:
        source = Path(explicit).resolve()
        _verify_source(source)
        return source
    cache = Path(os.environ.get(
        "TRTMC_HSTU_REFERENCE_CACHE",
        str(Path(tempfile.gettempdir()) / "trtmc-reference-cache" / "hstu"),
    ))
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / REFERENCE_REVISION
    if source.exists():
        _verify_source(source)
        return source
    with tempfile.TemporaryDirectory(prefix=".checkout-", dir=cache) as temporary:
        stage = Path(temporary) / "source"
        commands = (
            ["git", "init", "--quiet", str(stage)],
            ["git", "-C", str(stage), "fetch", "--quiet", "--no-tags", "--depth=1",
             _SOURCE_URL, REFERENCE_REVISION],
            ["git", "-C", str(stage), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        )
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(
                    "Cannot prepare pinned HSTU reference source; set "
                    "TRTMC_HSTU_REFERENCE_ROOT to an existing exact checkout.\n"
                    + result.stderr
                )
        _verify_source(stage)
        try:
            stage.rename(source)
        except OSError:
            # Another selected test process may have populated the same cache.
            if not source.is_dir():
                raise
            _verify_source(source)
    return source
