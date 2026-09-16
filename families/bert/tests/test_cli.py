# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from families.bert import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import build_cli, family_cli


def test_declared_build_and_legacy_spelling_use_owner_request(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"bert"}')
    calls = []
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append((request, output)))
    monkeypatch.setattr(build_cli, "build", lambda legacy: cli.build_bundle(cli.coerce_request(legacy), legacy.output_path))
    arguments = ["build", str(tmp_path), "-o", str(tmp_path / "model.bundle"), "--task", "embedding", "--fp32-layer", "2"]
    assert family_cli.main(["bert", *arguments]) == 0
    assert build_cli.main(arguments) == 0
    assert calls[0] == calls[1]
    assert type(calls[0][0]) is cli.BuildRequest
    assert calls[0][0].task == "embedding" and calls[0][0].fp32_layers == (2,)
    assert not hasattr(calls[0][0], "video_num_frames")
    assert build_cli.main([*arguments, "--max-batch-size", "1", "--context-parallel-size", "1"]) == 0
    assert calls[-1] == calls[0]
    with pytest.raises(NotImplementedError):
        build_cli.main([*arguments, "--max-batch-size", "2"])


@pytest.mark.parametrize("changes", [{"image_height": 8}, {"quantization": "fp8"}, {"context_parallel_size": 2}, {"dynamic_kv_cache": True}])
def test_legacy_api_rejects_unsupported_nondefault_values(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "model.bundle", "bert", "encoding", "fp32")
    assert cli.coerce_request(legacy) == cli.BuildRequest(tmp_path)
    with pytest.raises((ValueError, NotImplementedError)):
        cli.coerce_request(replace(legacy, **changes))


def test_help_and_rejected_options_do_not_import_model(monkeypatch, capsys):
    original = family_cli.importlib.import_module
    def guarded(name, *args, **kwargs):
        assert name not in {"families.bert.model", "tensorrt", "tensorrt_rtx", "huggingface_hub"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    with pytest.raises(SystemExit) as caught:
        family_cli.main(["bert", "build", "--help"])
    assert caught.value.code == 0
    assert "--tensor-parallel-size" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        family_cli.main(["bert", "build", "checkpoint", "-o", "model.bundle", "--video-num-frames", "2"])


def test_bundle_selects_backend_before_import_and_aborts_on_owner_failure(monkeypatch, tmp_path):
    events = []
    def select(backend):
        events.append(backend)
        monkeypatch.setitem(sys.modules, "families.bert.model", SimpleNamespace(build=lambda *args: (_ for _ in ()).throw(RuntimeError("build failed"))))
    monkeypatch.setattr(cli, "select_backend", select)
    monkeypatch.setattr(cli, "BundleWriter", lambda output: SimpleNamespace(finish=lambda: events.append("finish"), abort=lambda: events.append("abort")))
    with pytest.raises(RuntimeError, match="build failed"):
        cli.build_bundle(cli.BuildRequest(tmp_path, backend="trt_rtx"), tmp_path / "out")
    assert events == ["trt_rtx", "abort"]
