# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from families.boltz2 import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import build_cli, family_cli


def test_build_defaults_are_family_owned(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append((request, output)))
    assert family_cli.main(["boltz2", "build", str(tmp_path), "-o", str(tmp_path / "model.bundle")]) == 0
    assert calls[0][0] == cli.BuildRequest(tmp_path)
    assert calls[0][0].precision == "bf16"
    assert not hasattr(calls[0][0], "tensor_parallel_size")


def test_prepare_command_and_old_spelling_only_load_preparation(monkeypatch, tmp_path, capsys):
    (tmp_path / "config.json").write_text('{"model_type":"boltz2"}')
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"output": str(args[2])}
    monkeypatch.setitem(sys.modules, "families.boltz2.request_preparation", SimpleNamespace(prepare_structure_request=prepare))
    monkeypatch.setattr(build_cli, "_load_family", lambda family: SimpleNamespace(prepare_structure_request=prepare))
    arguments = ["prepare-structure", str(tmp_path), "--input", "request.yaml", "-o", "prepared.request"]
    assert family_cli.main(["boltz2", *arguments]) == 0
    assert build_cli.main(arguments) == 0
    assert len(calls) == 2 and calls[0] == calls[1]
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == {"output": "prepared.request"}


@pytest.mark.parametrize("changes", [{"tensor_parallel_size": 2}, {"fp32_layers": (0,)}, {"image_width": 8}, {"dynamic_kv_cache": True}])
def test_legacy_api_rejects_unsupported_nondefault_values(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "boltz2", "structure_prediction", "bf16")
    assert cli.coerce_request(legacy).precision == "bf16"
    with pytest.raises(NotImplementedError):
        cli.coerce_request(replace(legacy, **changes))


def test_help_does_not_load_model_or_preparation(monkeypatch):
    original = family_cli.importlib.import_module
    def guarded(name, *args, **kwargs):
        assert name not in {"families.boltz2.model", "families.boltz2.request_preparation", "tensorrt", "huggingface_hub"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    with pytest.raises(SystemExit) as caught:
        family_cli.main(["boltz2", "prepare-structure", "--help"])
    assert caught.value.code == 0
