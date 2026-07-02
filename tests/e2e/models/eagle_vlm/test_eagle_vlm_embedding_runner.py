# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eagle VLM-owned tests for embedding runner behavior."""

from __future__ import annotations

from tests.e2e.models.eagle_vlm.e2e_plugins.runners import embedding


def test_embedding_parser_accepts_fragmented_json_from_mpirun() -> None:
    stdout = '{"embedding": [0.1, 0.2,\n 0.3], "dim": 3}\n'

    assert embedding._parse_embedding(stdout) == [0.1, 0.2, 0.3]
