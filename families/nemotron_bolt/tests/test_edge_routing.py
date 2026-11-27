# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace

import pytest

from families.nemotron_bolt.edge_llm import dispatch, builder


def _request(path):
    return SimpleNamespace(
        output_path=path / "model.bundle",
        model_dir=path / "target",
        draft=path / "draft",
        task="vision_language_generation",
        precision="bf16",
    )


def test_discovery_failure_retries_original_pair(tmp_path, monkeypatch, caplog):
    request = _request(tmp_path)
    calls = []

    def unavailable():
        raise RuntimeError("unavailable Edge runtime")

    monkeypatch.setattr(dispatch, "detect_local_platform", unavailable)
    writer = object()
    dispatch.build(request, writer, request.draft, 4096, 8192, lambda *args: calls.append(args))
    assert calls == [(request, writer)] and "unavailable Edge runtime" in caplog.text


def test_native_wheel_does_not_use_external_sdk_receipt(tmp_path, monkeypatch):
    import tensorrt_model_connect

    active = tmp_path / "site-packages/tensorrt_model_connect"
    (active / "bin").mkdir(parents=True)
    (active / "bin/trtmc").touch()
    monkeypatch.setattr(tensorrt_model_connect.__spec__, "origin", str(active / "__init__.py"))

    def external_prefix():
        pytest.fail("native wheel looked outside its package")

    monkeypatch.setattr(
        importlib.import_module("tensorrt_model_connect.build"), "cmake_prefixes", external_prefix
    )
    with pytest.raises(FileNotFoundError, match="SDK receipt"):
        builder.sdk({"sm": 120}, None)


def test_malformed_prepared_output_falls_back_but_partial_write_does_not(
    tmp_path, monkeypatch, caplog
):
    request = _request(tmp_path)
    writer = object()
    calls = []
    monkeypatch.setattr(dispatch, "detect_local_platform", lambda: {"sm": 120})
    monkeypatch.setattr(builder, "sdk", lambda *_: ("python", "plugin", builder.REVISION))
    monkeypatch.setattr(builder, "run_build", lambda *_: None)
    dispatch.build(request, writer, request.draft, 4096, 8192, lambda *args: calls.append(args))
    assert calls == [(request, writer)] and "Falling back" in caplog.text
    monkeypatch.setattr(builder, "prepare_assets", lambda *_: ({}, {}))

    def broken(received_writer, *_):
        assert received_writer is writer
        raise OSError("partial writer")

    monkeypatch.setattr(builder, "publish", broken)
    with pytest.raises(OSError, match="partial writer"):
        dispatch.build(
            request,
            writer,
            request.draft,
            4096,
            8192,
            lambda *_: pytest.fail("partial writer retried"),
        )
