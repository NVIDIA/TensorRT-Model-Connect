# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_nemotron_builder_owns_the_two_gib_workspace_limit() -> None:
    source = (Path(__file__).resolve().parents[1] / "utils.py").read_text(encoding="utf-8")
    assert "config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)" in source
    assert "workspace_bytes" not in source
