# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from families.qwen.edge_llm import build as edge_build
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC, BundleWriter


def _request(tmp_path: Path, **updates) -> BuildRequest:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    values = {
        "model_dir": model,
        "output_path": tmp_path / "qwen-edge.bundle",
        "family": "qwen",
        "task": "text_generation",
        "precision": "fp16",
        "backend": "edge_llm",
        "max_sequence_length": 4096,
    }
    values.update(updates)
    return BuildRequest(**values)


def _read_header(path: Path) -> dict:
    data = path.read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    header_size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    start = len(BUNDLE_MAGIC) + 8
    return json.loads(data[start : start + header_size])


def test_explicit_edge_build_runs_official_tools_and_streams_engine_files(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        edge_build.shutil,
        "which",
        lambda name: f"/tools/{name}",
    )

    def run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        if command[0].endswith("tensorrt-edgellm-export"):
            (Path(command[2]) / "llm").mkdir(parents=True)
            return
        engine_argument = next(value for value in command if value.startswith("--engineDir="))
        engine = Path(engine_argument.split("=", 1)[1])
        engine.mkdir()
        for name in edge_build._REQUIRED_FILES:
            (engine / name).write_bytes(name.encode())

    monkeypatch.setattr(edge_build.subprocess, "run", run)
    request = _request(tmp_path, verbose=True)
    writer = BundleWriter(request.output_path)

    edge_build.build(request, writer)
    writer.finish()

    header = _read_header(request.output_path)
    assert header["family"] == "qwen"
    assert header["task"] == "text_generation"
    assert header["backend"] == "edge_llm"
    assert set(header["sections"]) == {
        "edge_llm.json",
        *(f"edge_llm/{name}" for name in edge_build._REQUIRED_FILES),
    }
    assert all(".so" not in name for name in header["sections"])
    assert calls[0][0] == "/tools/tensorrt-edgellm-export"
    assert calls[0][-1] == "--dtype=float16"
    assert calls[1][0] == "/tools/llm_build"
    assert "--maxInputLen=1024" in calls[1]
    assert "--maxKVCacheCapacity=4096" in calls[1]
    assert "--maxBatchSize=1" in calls[1]
    assert "--debug" in calls[1]


def test_edge_build_fails_when_an_official_tool_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(edge_build.shutil, "which", lambda _name: None)
    request = _request(tmp_path)
    writer = BundleWriter(request.output_path)

    with pytest.raises(RuntimeError, match="on PATH"):
        edge_build.build(request, writer)
    writer.abort()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "trt", "backend=edge_llm"),
        ("precision", "bf16", "precision=fp16"),
        ("tensor_parallel_size", 2, "single-device"),
        ("quantization", "fp8", "unquantized"),
        ("dynamic_kv_cache", True, "dynamic_kv_cache"),
    ],
)
def test_edge_build_rejects_unsupported_requests(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    request = _request(tmp_path, **{field: value})
    writer = BundleWriter(request.output_path)

    with pytest.raises(ValueError, match=message):
        edge_build.build(request, writer)
    writer.abort()
