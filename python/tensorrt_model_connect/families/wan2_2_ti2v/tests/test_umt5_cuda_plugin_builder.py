# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Wan2.2-owned UMT5 CUDA plugins."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import (
    umt5_cuda_plugin_builder as plugin_builder,
)


def test_plugin_override_must_exist(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.so"
    monkeypatch.setenv(plugin_builder._PLUGIN_ENV, str(missing))

    with pytest.raises(FileNotFoundError, match=plugin_builder._PLUGIN_ENV):
        plugin_builder.ensure_umt5_cuda_plugin()


def test_plugin_override_is_returned_without_building(
    tmp_path: Path,
    monkeypatch,
) -> None:
    library = tmp_path / "libwan22_umt5_test.so"
    library.write_bytes(b"test plugin")
    monkeypatch.setenv(plugin_builder._PLUGIN_ENV, str(library))

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("override must not invoke CMake")

    monkeypatch.setattr(plugin_builder.subprocess, "run", unexpected_run)
    assert plugin_builder.ensure_umt5_cuda_plugin() == library.resolve()


def test_plugin_build_uses_a_content_addressed_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv(plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(plugin_builder, "_source_digest", lambda _path: "digest")

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[1] == "--build":
            output = Path(command[2]) / "libtrtmc_wan22_umt5_cuda_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plugin_builder.subprocess, "run", run)
    first = plugin_builder.ensure_umt5_cuda_plugin()
    second = plugin_builder.ensure_umt5_cuda_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert len(calls) == 2
    assert calls[0][0:2] == ["cmake", "-S"]
    assert calls[1][0:2] == ["cmake", "--build"]


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
