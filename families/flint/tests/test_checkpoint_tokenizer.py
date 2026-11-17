# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint parity runs in the family stage where pinned assets are staged."""

import os
from pathlib import Path

from families.flint.tests.inputs import checkpoint_dir
from families.flint.tests.test_tokenizer import compare_tokenizers


def test_pinned_checkpoint_tokenizer(probe):
    path = os.environ.get("TRTMC_FLINT_TOKENIZER")
    path = Path(path) if path is not None else checkpoint_dir() / "tokenizer.json"
    compare_tokenizers(probe, path)
