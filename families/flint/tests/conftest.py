# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native tokenizer helper shared by asset-free and checkpoint test stages."""

import os
from pathlib import Path

import pytest

from families.flint.tests.inputs import native_binary


@pytest.fixture(scope="module")
def probe():
    path = os.environ.get("TRTMC_FLINT_TOKENIZER_PROBE")
    path = Path(path) if path is not None else native_binary("flint_tokenizer_probe")
    assert path.is_file()
    return path
