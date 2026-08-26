# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source contract for allocation-first Qwen native-KV error handling."""

from pathlib import Path


PLUGIN = Path(__file__).parents[4] / "src" / "runtime" / "models" / "qwen" / "plugin.cpp"


def test_qwen_reports_memory_only_after_allocation_failure() -> None:
    source = PLUGIN.read_text(encoding="utf-8")

    assert "admit_native_kv_allocation" not in source
    assert "admission failed before allocation" not in source
    assert source.count("cudaMemGetInfo") == 1
    assert source.count("qwen_kv_cache_allocation_failure(") == 2

    allocation = source.rsplit("build_inference_state", maxsplit=1)[1].split(
        "static void log_kv_cache_sizing", maxsplit=1
    )[0]
    assert (
        allocation.index("std::make_unique<QwenKvCache>")
        < allocation.index("if (!state->ok())")
        < allocation.index("throw qwen_kv_cache_allocation_failure(sizing)")
    )

    reporter = source.split("std::runtime_error qwen_kv_cache_allocation_failure", maxsplit=1)[
        1
    ].split("void reject_native_kv_size_override", maxsplit=1)[0]
    assert "cudaMemGetInfo" in reporter
    assert "free after failure=" in reporter
    assert "CUDA memory diagnostics failed" in reporter
