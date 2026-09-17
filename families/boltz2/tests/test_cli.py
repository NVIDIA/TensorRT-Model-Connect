# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from families.boltz2 import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import build_cli, family_cli


def _legacy_prepare_module():
    """Load the actual legacy hooks without importing the TensorRT builder."""
    path = Path(cli.__file__).with_name("model.py")
    tree = ast.parse(path.read_text())
    names = {"prepare_structure_request", "add_prepare_structure_arguments", "prepare_structure_cli_options"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *functions], type_ignores=[])
    namespace = {"__package__": "families.boltz2"}
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_build_defaults_are_family_owned(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append((request, output)))
    assert family_cli.main(["boltz2", "build", str(tmp_path), "-o", str(tmp_path / "model.bundle")]) == 0
    assert calls[0][0] == cli.BuildRequest(tmp_path)
    assert calls[0][0].precision == "bf16"
    assert not hasattr(calls[0][0], "tensor_parallel_size")


@pytest.mark.parametrize("options, expected", [
    ([], {"sampling_steps": 200, "diffusion_samples": 1, "seed": 42,
          "affinity_sampling_steps": 200, "affinity_diffusion_samples": 5}),
    (["--num-steps", "300", "--num-samples", "2", "--seed", "0",
      "--affinity-num-steps", "400", "--affinity-num-samples", "3"],
     {"sampling_steps": 300, "diffusion_samples": 2, "seed": 0,
      "affinity_sampling_steps": 400, "affinity_diffusion_samples": 3}),
])
def test_prepare_command_and_old_spelling_preserve_request_controls(monkeypatch, tmp_path, capsys, options, expected):
    (tmp_path / "config.json").write_text('{"model_type":"boltz2"}')
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"output": str(args[2])}
    monkeypatch.setitem(sys.modules, "families.boltz2.request_preparation", SimpleNamespace(prepare_structure_request=prepare))
    monkeypatch.setattr(build_cli, "_load_family", lambda family: _legacy_prepare_module())
    arguments = ["prepare-structure", str(tmp_path), "--input", "request.yaml", "-o", "prepared.request", *options]
    assert family_cli.main(["boltz2", *arguments]) == 0
    assert build_cli.main(arguments) == 0
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0][1] == {"cache_dir": None, **expected}
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {"output": "prepared.request"}


@pytest.mark.parametrize("changes", [{"tensor_parallel_size": 2}, {"fp32_layers": (0,)}, {"image_width": 8}, {"dynamic_kv_cache": True}])
def test_legacy_api_rejects_unsupported_nondefault_values(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "boltz2", "structure_prediction", "bf16")
    assert cli.coerce_request(legacy).precision == "bf16"
    with pytest.raises(NotImplementedError):
        cli.coerce_request(replace(legacy, **changes))


def test_help_does_not_load_model_or_preparation(monkeypatch, capsys):
    original = family_cli.importlib.import_module
    def guarded(name, *args, **kwargs):
        assert name not in {"families.boltz2.model", "families.boltz2.request_preparation", "tensorrt", "huggingface_hub"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    with pytest.raises(SystemExit) as caught:
        family_cli.main(["boltz2", "prepare-structure", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--num-steps", "--num-samples", "--seed", "--affinity-num-steps", "--affinity-num-samples"):
        assert flag in help_text


@pytest.mark.parametrize("flag, value, error", [
    ("--num-steps", "9", "sampling steps"),
    ("--num-samples", "0", "structure samples"),
    ("--seed", "-1", "seed"),
    ("--affinity-num-steps", "9", "affinity sampling steps"),
    ("--affinity-num-samples", "0", "affinity samples"),
])
def test_prepare_command_preserves_owner_sampling_validation(monkeypatch, tmp_path, flag, value, error):
    from families.boltz2 import request_preparation

    request = tmp_path / "request.yaml"
    request.write_text("version: 1\n")
    monkeypatch.setattr(request_preparation, "resolve_package_root", lambda model: tmp_path)
    monkeypatch.setattr(request_preparation, "validate_structure_checkpoint", lambda path: None)
    monkeypatch.setattr(request_preparation, "validate_artifact", lambda *args: None)
    monkeypatch.setattr(request_preparation, "_request_inputs", lambda path: (b"request", SimpleNamespace(affinity=True), (), ()))
    with pytest.raises(ValueError, match=error):
        family_cli.main(["boltz2", "prepare-structure", str(tmp_path), "--input", str(request),
                         "-o", str(tmp_path / "prepared.request"), flag, value])
    assert not (tmp_path / "prepared.request").exists()
