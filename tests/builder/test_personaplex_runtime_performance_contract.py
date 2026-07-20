# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the PersonaPlex generation runtime."""

from pathlib import Path


RUNTIME_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "runtime"
    / "models"
    / "personaplex"
)


def test_temporal_and_depth_generation_do_not_use_synchronous_host_forward() -> None:
    source = (RUNTIME_DIR / "pipeline.cpp").read_text(encoding="utf-8")

    assert "temporal_->forward_async(inputs)" in source
    assert "engine.forward_async(inputs)" in source
    assert "temporal_->forward(inputs)" not in source
    assert "engine.forward(inputs)" not in source


def test_depth_cache_reset_does_not_synchronize_the_generation_stream() -> None:
    source = (RUNTIME_DIR / "kv_cache.cpp").read_text(encoding="utf-8")
    reset_body = source.split("void PersonaplexKvCache::reset()", maxsplit=1)[1].split(
        "std::size_t PersonaplexKvCache::device_memory_bytes()", maxsplit=1
    )[0]

    assert "cudaStreamSynchronize" not in reset_body


def test_personaplex_modules_share_one_ordered_generation_stream() -> None:
    source = (RUNTIME_DIR / "plugin.cpp").read_text(encoding="utf-8")

    assert "chained_opts.stream = stream" in source
    assert "load_depth_engines(ctx.backend, ctx.bundle, chained_opts)" in source
