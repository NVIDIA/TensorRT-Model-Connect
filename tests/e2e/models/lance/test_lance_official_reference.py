# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance official-reference platform compatibility contracts."""

from __future__ import annotations

import os
import subprocess
import sys

from tests.e2e.models.lance.e2e_plugins.references.lance_official import (
    _image_reference_environment,
)


def test_lance_image_reference_provides_decord_import_without_video_support() -> None:
    environment = _image_reference_environment(
        {**os.environ, "PYTHONPATH": "/existing/python/path"}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import decord; "
                "assert callable(decord.cpu); "
                "decord.VideoReader('unused.mp4')"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "image-only Lance reference" in result.stderr
    assert environment["PYTHONPATH"].endswith(":/existing/python/path")
