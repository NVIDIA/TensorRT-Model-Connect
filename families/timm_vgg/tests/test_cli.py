# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the timm_vgg command boundary."""

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from families.timm_vgg import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import family_cli

FAMILY = "timm_vgg"
TASK = "classification"


def test_build_arguments_reach_the_narrow_owner_request(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "build_bundle", lambda request, output: calls.append((request, output))
    )
    output = tmp_path / "model.bundle"
    assert (
        family_cli.main([FAMILY, "build", str(tmp_path), "-o", str(output), "--precision", "fp16"])
        == 0
    )
    legacy = LegacyRequest(tmp_path, output, FAMILY, TASK, "fp16")
    assert calls == [(cli.coerce_request(legacy), output)]
    assert type(calls[0][0]) is cli.BuildRequest
    assert not hasattr(calls[0][0], "image_height")
    assert not hasattr(calls[0][0], "max_sequence_length")


def test_ignored_legacy_settings_do_not_become_new_cli_options(tmp_path):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", FAMILY, TASK, "fp32")
    request = cli.coerce_request(legacy)
    assert (
        cli.coerce_request(replace(legacy, max_sequence_length=1, quantization="none")) == request
    )
    with pytest.raises(NotImplementedError, match="max_sequence_length"):
        cli.coerce_request(replace(legacy, max_sequence_length=2))


@pytest.mark.parametrize(
    "changes",
    [
        {"dynamic_kv_cache": True},
        {"image_height": 2},
        {"image_width": 2},
        {"video_num_frames": 2},
        {"max_batch_size": 2},
        {"context_parallel_size": 2},
        {"quantization": "fp8"},
        {"fp32_layers": (0,)},
        {"tensor_parallel_size": 2},
    ],
)
def test_legacy_unsupported_options_remain_rejected(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", FAMILY, TASK, "fp32")
    with pytest.raises((ValueError, NotImplementedError)):
        cli.coerce_request(replace(legacy, **changes))


def test_unknown_python_inputs_are_rejected(tmp_path):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", FAMILY, TASK, "fp32")
    with pytest.raises(ValueError, match="unknown"):
        cli.coerce_request(SimpleNamespace(**vars(legacy), unknown_owner_input=1))


def test_help_and_rejection_do_not_import_heavy_builder(monkeypatch, capsys):
    original = family_cli.importlib.import_module

    def guarded(name, *args, **kwargs):
        assert name not in {
            f"families.{FAMILY}.model",
            "tensorrt",
            "tensorrt_rtx",
            "huggingface_hub",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    with pytest.raises(SystemExit) as caught:
        family_cli.main([FAMILY, "build", "--help"])
    assert caught.value.code == 0
    assert "--max-sequence-length" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as rejected:
        family_cli.main([FAMILY, "build", "checkpoint", "-o", "out", "--max-sequence-length", "1"])
    assert rejected.value.code == 2
    with pytest.raises(SystemExit) as classify:
        family_cli.main([FAMILY, "classify", "--help"])
    assert classify.value.code == 0
    help_text = capsys.readouterr().out
    assert "--runtime-cache" in help_text and "--cuda-graphs" in help_text


@pytest.mark.parametrize("fail", [False, True])
def test_backend_selection_and_bundle_lifecycle(monkeypatch, tmp_path, fail):
    events = []

    def run(request, writer):
        events.append("build")
        if fail:
            raise RuntimeError("owner build failed")

    def select(backend):
        events.append(backend)
        monkeypatch.setitem(sys.modules, f"families.{FAMILY}.model", SimpleNamespace(build=run))

    monkeypatch.setattr(cli, "select_backend", select)
    monkeypatch.setattr(
        cli,
        "BundleWriter",
        lambda output: SimpleNamespace(
            finish=lambda: events.append("finish"), abort=lambda: events.append("abort")
        ),
    )
    if fail:
        with pytest.raises(RuntimeError, match="owner build failed"):
            cli.build_bundle(cli.BuildRequest(tmp_path, backend="trt_rtx"), tmp_path / "out")
    else:
        cli.build_bundle(cli.BuildRequest(tmp_path, backend="trt_rtx"), tmp_path / "out")
    assert events == ["trt_rtx", "build", "abort" if fail else "finish"]
