# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from families.timm_mobilenetv2 import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import family_cli
from trtmc_benchmark.builder import _build_command
from trtmc_benchmark.catalog import ManifestCatalog
from trtmc_benchmark.types import BenchmarkError


FAMILY = "timm_mobilenetv2"
TASK = "image_to_class_scores"


def legacy(tmp_path, **changes):
    request = LegacyRequest(tmp_path, tmp_path / "model.bundle", FAMILY, TASK, "fp32")
    return replace(request, **changes)


def test_owner_request_preserves_supported_legacy_defaults(tmp_path):
    assert cli.coerce_request(legacy(tmp_path)) == cli.BuildRequest(tmp_path)
    assert cli.coerce_request(legacy(tmp_path, quantization="none", max_sequence_length=1)) == cli.BuildRequest(tmp_path)
    assert set(cli.BuildRequest.__dataclass_fields__) == {"model_dir", "task", "precision", "backend", "verbose"}
    with pytest.raises(NotImplementedError):
        cli.coerce_request(legacy(tmp_path, max_sequence_length=2))


@pytest.mark.parametrize("changes", [
    {"dynamic_kv_cache": True}, {"image_height": 32}, {"image_width": 32},
    {"video_num_frames": 2}, {"max_batch_size": 2}, {"tensor_parallel_size": 2},
    {"context_parallel_size": 2}, {"quantization": "fp8"}, {"fp32_layers": (0,)},
])
def test_legacy_unsupported_values_are_still_rejected(tmp_path, changes):
    with pytest.raises(NotImplementedError):
        cli.coerce_request(legacy(tmp_path, **changes))


def test_wrong_task_precision_and_unknown_legacy_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        cli.coerce_request(legacy(tmp_path, task="text_generation"))
    with pytest.raises(ValueError):
        cli.BuildRequest(tmp_path, precision="bf16")
    with pytest.raises(ValueError):
        cli.BuildRequest(tmp_path, backend="unknown")
    request = SimpleNamespace(**vars(legacy(tmp_path)), invented=1)
    with pytest.raises(ValueError, match="unknown"):
        cli.coerce_request(request)


def test_cli_help_and_unknown_flags_do_not_load_the_model(monkeypatch, capsys):
    original = family_cli.importlib.import_module
    def guarded(name, *args, **kwargs):
        assert name not in {f"families.{FAMILY}.model", "tensorrt", "tensorrt_rtx", "huggingface_hub"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(family_cli.importlib, "import_module", guarded)
    for command in ("build", "classify"):
        with pytest.raises(SystemExit) as caught:
            family_cli.main([FAMILY, command, "--help"])
        assert caught.value.code == 0
    assert "--image" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        family_cli.main([FAMILY, "build", "unused", "-o", "unused.bundle", "--image-height", "32"])
    assert caught.value.code != 0


@pytest.mark.parametrize("fail", [False, True])
def test_backend_precedes_lazy_model_import_and_bundle_publication(monkeypatch, tmp_path, fail):
    events = []
    output = tmp_path / "output.bundle"
    output.write_bytes(b"existing")
    def build_model(request, writer):
        assert events == ["trt_rtx"]
        assert type(request) is cli.BuildRequest
        events.append("build")
        writer.set_header(family=FAMILY, task=TASK, backend=request.backend)
        writer.add_bytes("engine.plan", b"PLAN")
        if fail:
            raise RuntimeError("owner build failed")
    def select(backend):
        events.append(backend)
        monkeypatch.setitem(sys.modules, f"families.{FAMILY}.model", SimpleNamespace(build=build_model))
    monkeypatch.setattr(cli, "select_backend", select)
    request = cli.BuildRequest(tmp_path, backend="trt_rtx")
    if fail:
        with pytest.raises(RuntimeError, match="owner build failed"):
            cli.build_bundle(request, output)
        assert output.read_bytes() == b"existing"
    else:
        cli.build_bundle(request, output)
        assert output.read_bytes().startswith(b"BUNDLE\x01\x00")
    assert events == ["trt_rtx", "build"]
    assert not list(tmp_path.glob(".output.bundle.sections.*"))


def test_real_manifests_use_the_declared_build_and_serializer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append((request, output)))
    catalog = ManifestCatalog(root=Path(__file__).resolve().parents[2])
    manifests = sorted((Path(__file__).parent / "manifests").glob("*.json"))
    assert manifests
    for path in manifests:
        model = catalog.resolve(str(path))
        command = _build_command(model, tmp_path, tmp_path / "built.bundle", ())
        assert command[3:5] == (FAMILY, "build")
        assert family_cli.main(command[3:]) == 0
        assert type(calls[-1][0]) is cli.BuildRequest
        assert calls[-1][0].task == model.task and calls[-1][0].precision == model.precision
        invalid = json.loads(path.read_text())
        invalid["build"] = {"image_height": 32}
        invalid_path = tmp_path / "invalid-manifest.json"
        invalid_path.write_text(json.dumps(invalid))
        with pytest.raises(BenchmarkError, match="undeclared"):
            catalog.resolve(str(invalid_path))
