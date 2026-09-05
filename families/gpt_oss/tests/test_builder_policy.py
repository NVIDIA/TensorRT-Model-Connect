# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_builders_bound_workspace_to_exclude_oversized_tactics() -> None:
    family = Path(__file__).resolve().parents[1]

    for name in ("model.py", "tp_builder.py"):
        source = (family / name).read_text(encoding="utf-8")
        assert "set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)" in source
