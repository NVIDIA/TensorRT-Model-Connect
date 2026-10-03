# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from families.gpt_neox import cli, support
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import family_cli
from tensorrt_model_connect.model_support import ModelMetadata


def test_cli_defaults_match_support_and_publish_a_bundle(monkeypatch, tmp_path):
    calls = []

    def build_model(request, writer):
        calls.append(request)
        writer.set_header(family="gpt_neox", task=request.task, backend=request.backend)
        writer.add_bytes("engine.plan", b"PLAN")

    def select_backend(backend):
        assert backend == "trt"
        monkeypatch.setitem(
            sys.modules, "families.gpt_neox.model", SimpleNamespace(build=build_model)
        )

    monkeypatch.setattr(cli, "select_backend", select_backend)
    output = tmp_path / "model.bundle"
    declaration = support.describe(ModelMetadata({"model_type": "gpt_neox"}, {}))
    assert declaration is not None
    assert family_cli.main(["gpt_neox", "build", str(tmp_path), "-o", str(output)]) == 0
    assert len(calls) == 1 and type(calls[0]) is cli.BuildRequest
    assert calls[0].task == declaration.default_task == "text_generation"
    assert calls[0].precision == declaration.default_precision
    assert not hasattr(calls[0], "video_num_frames")
    assert output.read_bytes().startswith(b"BUNDLE\x01\x00")


def test_failed_owner_preserves_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "model.bundle"
    output.write_bytes(b"previous bundle")

    def fail(request, writer):
        writer.set_header(family="gpt_neox", task=request.task, backend=request.backend)
        writer.add_bytes("engine.plan", b"partial")
        raise RuntimeError("owner build failed")

    monkeypatch.setattr(cli, "select_backend", lambda backend: None)
    monkeypatch.setitem(sys.modules, "families.gpt_neox.model", SimpleNamespace(build=fail))
    with pytest.raises(RuntimeError, match="owner build failed"):
        cli.build_bundle(cli.BuildRequest(tmp_path), output)
    assert output.read_bytes() == b"previous bundle"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize(
    "changes",
    [
        {"dynamic_kv_cache": True},
        {"image_height": 8},
        {"video_num_frames": 2},
        {"max_batch_size": 2},
        {"context_parallel_size": 2},
        {"quantization": "fp8"},
        {"fp32_layers": (1,)},
    ],
)
def test_legacy_unsupported_inputs_are_still_rejected(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "gpt_neox", "text_generation", "fp32")
    assert cli.coerce_request(legacy) == cli.BuildRequest(tmp_path)
    assert cli.coerce_request(replace(legacy, quantization="none")) == cli.BuildRequest(tmp_path)
    with pytest.raises((NotImplementedError, ValueError)):
        cli.coerce_request(replace(legacy, **changes))


def test_invalid_input_is_rejected_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "build_bundle", lambda *args: pytest.fail("invalid input reached builder")
    )
    with pytest.raises(ValueError, match="max_sequence_length"):
        family_cli.main(
            [
                "gpt_neox",
                "build",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--max-sequence-length",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        family_cli.main(
            [
                "gpt_neox",
                "build",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--video-num-frames",
                "2",
            ]
        )


def test_help_is_lazy_and_owned(monkeypatch, capsys):
    original = family_cli.importlib.import_module

    def guarded(name, *args, **kwargs):
        assert name not in {
            "families.gpt_neox.cli",
            "families.gpt_neox.model",
            "tensorrt",
            "tensorrt_rtx",
            "huggingface_hub",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    with pytest.raises(SystemExit) as error:
        family_cli.main(["gpt_neox", "build", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--tensor-parallel-size" in help_text
    assert "--quantization" not in help_text
