# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import build_cli
from tensorrt_model_connect.model_support import FamilySupport


def test_build_command_forwards_only_direct_inputs(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(build_cli, "build", captured.append)
    monkeypatch.setattr(build_cli, "resolve_family", lambda metadata: (
        "example_owner", FamilySupport(("owner_default", "explicit_task"), "owner_default")
    ))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"example_model"}', encoding="utf-8")
    output = tmp_path / "model.bundle"

    assert (
        build_cli.main(
            [
                "build",
                str(model),
                "--output",
                str(output),
                "--task",
                "explicit_task",
                "--precision",
                "fp16",
                "--backend",
                "trt_rtx",
                "--max-sequence-length",
                "1024",
                "--image-height",
                "512",
                "--image-width",
                "768",
                "--video-num-frames",
                "17",
                "--max-batch-size",
                "3",
                "--tensor-parallel-size",
                "2",
                "--context-parallel-size",
                "4",
                "--quantization",
                "fp8",
                "--fp32-layer",
                "2",
                "--fp32-layer",
                "5",
                "--dynamic-kv-cache",
                "--verbose",
            ]
        )
        == 0
    )
    request = captured[0]
    assert request.model_dir == model
    assert request.output_path == output
    assert request.family == "example_owner"
    assert request.task == "explicit_task"
    assert request.precision == "fp16"
    assert request.backend == "trt_rtx"
    assert request.max_sequence_length == 1024
    assert request.image_height == 512
    assert request.image_width == 768
    assert request.video_num_frames == 17
    assert request.max_batch_size == 3
    assert request.tensor_parallel_size == 2
    assert request.context_parallel_size == 4
    assert request.quantization == "fp8"
    assert request.fp32_layers == (2, 5)
    assert request.dynamic_kv_cache is True
    assert request.verbose is True


def test_build_command_uses_the_family_owned_default_task(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(build_cli, "build", captured.append)
    monkeypatch.setattr(build_cli, "resolve_family", lambda metadata: (
        "example_owner", FamilySupport(("owner_default", "explicit_task"), "owner_default")
    ))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"example_model"}', encoding="utf-8")

    assert build_cli.main(["build", str(model), "--output", str(tmp_path / "out.bundle")]) == 0

    assert captured[0].family == "example_owner"
    assert captured[0].task == "owner_default"
    assert captured[0].precision == "fp32"


def test_console_build_reuses_the_existing_builder(monkeypatch) -> None:
    from tensorrt_model_connect import __main__ as launcher

    calls = []
    monkeypatch.setattr(build_cli, "main", lambda arguments: calls.append(arguments) or 7)
    monkeypatch.setattr(launcher.os, "execv", lambda *args: pytest.fail("build invoked runtime"))
    arguments = ["build", "example/model", "-o", "model.bundle"]
    assert launcher.main(arguments) == 7
    assert calls == [arguments]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "model.bundle", "--prompt", "Hello"],
        ["run", "model.bundle", "--runtime-root", "/explicit/runtime", "--prompt", ""],
        ["inspect", "model.bundle"],
        ["version"],
    ],
)
def test_console_runtime_executes_only_the_packaged_native_binary(
    monkeypatch, tmp_path: Path, arguments: list[str]
) -> None:
    from tensorrt_model_connect import __main__ as launcher

    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "package" / "__main__.py"))
    calls = []

    class Replaced(BaseException):
        pass

    def execute(path, argv):
        calls.append((path, argv))
        raise Replaced

    monkeypatch.setattr(launcher.os, "execv", execute)
    original = list(arguments)
    with pytest.raises(Replaced):
        launcher.main(arguments)
    native = tmp_path / "package" / "bin" / "trtmc"
    expected = list(arguments)
    if arguments[0] == "run" and "--runtime-root" not in arguments:
        expected += ["--runtime-root", str(native.parent)]
    assert calls == [(str(native), [str(native), *expected])]
    assert arguments == original


def test_console_missing_native_binary_fails_without_path_fallback(monkeypatch, capsys) -> None:
    from tensorrt_model_connect import __main__ as launcher

    calls = []

    def missing(path, argv):
        calls.append(path)
        raise FileNotFoundError("packaged binary is missing")

    monkeypatch.setattr(launcher.os, "execv", missing)
    assert launcher.main(["version"]) == 1
    assert len(calls) == 1
    assert "cannot execute packaged trtmc" in capsys.readouterr().err


def test_console_help_includes_build_and_delegates_native_help(monkeypatch, capsys) -> None:
    from tensorrt_model_connect import __main__ as launcher

    class Replaced(BaseException):
        pass

    def execute(path, argv):
        assert argv == [path, "help"]
        raise Replaced

    monkeypatch.setattr(launcher.os, "execv", execute)
    with pytest.raises(Replaced):
        launcher.main(["--help"])
    assert "trtmc build MODEL -o model.bundle" in capsys.readouterr().out


def test_hugging_face_model_id_resolves_to_a_local_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    assert build_cli._resolve_model("openai-community/gpt2", "revision-1") == tmp_path
    assert calls == [{"repo_id": "openai-community/gpt2", "revision": "revision-1"}]


def test_build_command_rejects_a_task_the_family_does_not_own(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"example_model"}', encoding="utf-8")
    monkeypatch.setattr(build_cli, "resolve_family", lambda metadata: (
        "example_owner", FamilySupport(("owner_default", "explicit_task"), "owner_default")
    ))
    monkeypatch.setattr(build_cli, "build", lambda request: pytest.fail("unsupported task reached build"))

    with pytest.raises(ValueError, match="does not support task 'unowned_task'"):
        build_cli.main(
            [
                "build",
                str(model),
                "--output",
                str(tmp_path / "out.bundle"),
                "--task",
                "unowned_task",
            ]
        )


def test_prepare_structure_dispatches_to_the_resolved_family(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"boltz2"}', encoding="utf-8")
    request = tmp_path / "request.yaml"
    request.write_text("version: 1\n", encoding="utf-8")
    output = tmp_path / "request.b2rq"
    cache = tmp_path / "cache"
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"family": "boltz2", "cache_hit": False}

    monkeypatch.setattr(
        build_cli,
        "_load_family",
        lambda family: SimpleNamespace(prepare_structure_request=prepare),
    )

    assert (
        build_cli.main(
            [
                "prepare-structure",
                str(model),
                "--input",
                str(request),
                "--output",
                str(output),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )
    assert calls == [((model, request, output), {"cache_dir": cache})]
    assert json.loads(capsys.readouterr().out) == {
        "cache_hit": False,
        "family": "boltz2",
    }


def test_prepare_structure_uses_the_family_hook_after_task_migration(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"example_model"}', encoding="utf-8")
    support = FamilySupport(
        ("molecular_document_to_structure",), "molecular_document_to_structure"
    )
    monkeypatch.setattr(build_cli, "resolve_family", lambda metadata: ("example_owner", support))
    monkeypatch.setattr(build_cli, "build", lambda request: pytest.fail("preparation invoked build"))
    request = tmp_path / "request.yaml"
    request.write_text("version: 1\n", encoding="utf-8")
    output = tmp_path / "prepared.request"
    calls = []

    def prepare(model_dir, input_path, output_path, *, cache_dir):
        calls.append((model_dir, input_path, output_path, cache_dir))
        output_path.write_bytes(input_path.read_bytes())
        return {"family": "example_owner", "output": str(output_path)}

    def load_family(family):
        assert family == "example_owner"
        return SimpleNamespace(prepare_structure_request=prepare)

    monkeypatch.setattr(build_cli, "_load_family", load_family)

    assert build_cli.main([
        "prepare-structure", str(model), "--input", str(request), "-o", str(output)
    ]) == 0
    assert calls == [(model, request, output, None)]
    assert output.read_bytes() == request.read_bytes()
    assert json.loads(capsys.readouterr().out) == {
        "family": "example_owner", "output": str(output)
    }


@pytest.mark.parametrize("hook", ["absent", None, "not callable"])
def test_prepare_structure_requires_a_callable_family_hook(
    monkeypatch, tmp_path: Path, hook
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"example_model"}', encoding="utf-8")
    support = FamilySupport(
        ("molecular_document_to_structure",), "molecular_document_to_structure"
    )
    monkeypatch.setattr(build_cli, "resolve_family", lambda metadata: ("example_owner", support))
    module = SimpleNamespace()
    if hook != "absent":
        module.prepare_structure_request = hook
    monkeypatch.setattr(build_cli, "_load_family", lambda family: module)
    monkeypatch.setattr(build_cli, "build", lambda request: pytest.fail("preparation invoked build"))
    output = tmp_path / "prepared.request"

    with pytest.raises(ValueError, match="family 'example_owner' does not support request preparation"):
        build_cli.main([
            "prepare-structure", str(model), "--input", str(tmp_path / "request.yaml"),
            "-o", str(output),
        ])
    assert not output.exists()
