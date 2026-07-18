# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Wan2.2-owned UMT5 CUDA plugins."""

from __future__ import annotations

import inspect
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v import (
    umt5_cuda_plugin_builder as plugin_builder,
)


def test_plugin_builder_resolves_the_packaged_aot_companion(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    companion = type("Companion", (), {"load_path": library})()
    monkeypatch.setattr(
        plugin_builder,
        "load_wan22_plugin_companion",
        lambda **_kwargs: companion,
    )

    assert plugin_builder.ensure_umt5_cuda_plugin() == library
    assert "subprocess" not in inspect.getsource(plugin_builder)


def test_plugin_sources_are_cuda_tensorrt_only() -> None:
    source_dir = Path(plugin_builder.__file__).with_name("umt5_cuda_plugins")
    cmake = (source_dir / "CMakeLists.txt").read_text()
    cuda = (source_dir / "wan22_umt5_gelu_plugin.cu").read_text()
    builder = inspect.getsource(plugin_builder)
    combined = "\n".join((cmake, cuda, builder)).lower()

    assert "find_package(torch" not in combined
    assert "torch/" not in combined
    assert "aten/" not in combined
    assert "libtorch" not in combined
    assert "import torch" not in combined
    assert "wan22umt5sourcegelu" in combined
    assert "wan22umt5sourcesoftmax" in combined
    assert "wan22umt5sourcermsnorm" in combined
    assert "wan22umt5bf16barrier" in combined
    assert "source_softmax_512_kernel" in cuda
    assert "kUmt5SoftmaxElements = 512" in cuda
    assert "kUmt5SoftmaxRows = 64 * 512" in cuda
    assert "__shfl_xor_sync" in cuda
    assert "std::exp" in cuda
    assert "source_rmsnorm_512x4096_kernel" in cuda
    assert "kUmt5RmsNormRows = 512" in cuda
    assert "kUmt5RmsNormElements = 4096" in cuda
    assert "__shfl_down_sync" in cuda
    assert "--use_fast_math" not in cmake
    assert 'cuda_architectures "103;110"' in cmake.lower()
