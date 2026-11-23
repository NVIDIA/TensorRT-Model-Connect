# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flash-Next complete-network routing contracts; no checkpoint download."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.qwen3_8_flash_next.edge_llm import builder, dispatch


def test_discovery_error_retries_the_unchanged_native_request(tmp_path, monkeypatch, caplog):
    def unavailable():
        raise RuntimeError("optional SDK platform unavailable")

    request = SimpleNamespace(output_path=tmp_path / "model.bundle")
    writer = object()
    calls = []
    monkeypatch.setattr(dispatch, "detect_local_platform", unavailable)
    dispatch.build(request, writer, 255, 256, lambda *args: calls.append(args))
    assert calls == [(request, writer)]
    assert "optional SDK platform unavailable" in caplog.text


def test_edge_publication_error_must_not_retry_native(tmp_path, monkeypatch):
    request = SimpleNamespace(
        output_path=tmp_path / "model.bundle", model_dir=tmp_path, task="text_generation"
    )
    writer = object()
    monkeypatch.setattr(dispatch, "detect_local_platform", lambda: {})
    monkeypatch.setattr(builder, "sdk", lambda *_: ("python", "plugin", builder.REVISION))
    monkeypatch.setattr(builder, "run_build", lambda *_: None)
    monkeypatch.setattr(builder, "prepare_assets", lambda *_: {})

    def partial_write(received_writer, *_):
        assert received_writer is writer
        raise OSError("partial bundle")

    monkeypatch.setattr(builder, "publish", partial_write)
    with pytest.raises(OSError, match="partial bundle"):
        dispatch.build(request, writer, 255, 256, lambda *_: pytest.fail("retried partial bundle"))


def test_runner_owns_sdk_even_without_a_packaged_receipt(tmp_path, monkeypatch):
    import tensorrt_model_connect

    module = tmp_path / "active/tensorrt_model_connect"
    spec = tensorrt_model_connect.__spec__
    monkeypatch.setattr(spec, "origin", str(module / "__init__.py"))
    standalone = tmp_path / "standalone"
    target = {"arch": "x86_64", "sm": 80, "cuda_version": "13.3", "tensorrt_version": "11.4.0.106"}

    def install(prefix):
        receipt = prefix / "share/trtmc/edge-llm.json"
        receipt.parent.mkdir(parents=True)
        (prefix / "plugin.so").touch()
        receipt.write_text(json.dumps({
            "version": builder.RELEASE, "schema_version": 2, "release_revision": builder.REVISION,
            "arch": target["arch"], "architectures": [target["sm"]], "cuda_version": target["cuda_version"],
            "tensorrt_version": target["tensorrt_version"], "plugin": "plugin.so",
        }))

    install(standalone)
    monkeypatch.setattr(importlib.import_module("tensorrt_model_connect.build"), "cmake_prefixes",
                        lambda: [standalone])
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps([builder.RELEASE, target["tensorrt_version"], "5.17.0", True])))
    runner = module / "bin/trtmc"
    runner.parent.mkdir(parents=True)
    runner.touch()
    with pytest.raises(FileNotFoundError, match="SDK receipt"):
        builder.sdk(target, Path(sys.executable))
    install(module)
    assert builder.sdk(target, None)[1] == module / "plugin.so"
    runner.unlink()
    assert builder.sdk(target, None)[1] == standalone / "plugin.so"
