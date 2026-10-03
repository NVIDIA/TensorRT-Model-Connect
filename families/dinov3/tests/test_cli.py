# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from families.dinov3 import cli
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import family_cli
from trtmc_benchmark.catalog import ManifestCatalog
from trtmc_benchmark.types import BenchmarkError


def test_declared_build_uses_narrow_owner_request(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "resolve_model", lambda model, revision: tmp_path)
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append((request, output)))
    output = tmp_path / "out.bundle"
    assert family_cli.main(["dinov3", "build", "checkpoint", "-o", str(output)]) == 0
    legacy = LegacyRequest(tmp_path, output, "dinov3", "image_features", "fp32")
    assert calls == [(cli.coerce_request(legacy), output)]
    assert type(calls[0][0]) is cli.BuildRequest
    assert not hasattr(calls[0][0], "dynamic_kv_cache")
    assert not hasattr(calls[0][0], "tensor_parallel_size")
    assert not hasattr(calls[0][0], "output_path")


@pytest.mark.parametrize("changes", [
    {"dynamic_kv_cache": True}, {"video_num_frames": 2}, {"max_batch_size": 2},
    {"tensor_parallel_size": 2}, {"context_parallel_size": 2},
    {"quantization": "fp8"}, {"fp32_layers": (1,)},
])
def test_legacy_rejects_unsupported_nondefaults(tmp_path, changes):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "dinov3", "image_features", "fp32")
    with pytest.raises((ValueError, NotImplementedError)):
        cli.coerce_request(replace(legacy, **changes))


def test_unknown_legacy_fields_are_not_dropped(tmp_path):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "dinov3", "image_features", "fp32")
    with pytest.raises(ValueError, match="unknown dinov3 build inputs"):
        cli.coerce_request(SimpleNamespace(**vars(legacy), unrecognized_option=1))


def test_help_and_rejected_options_do_not_import_handlers(monkeypatch, capsys):
    discover_namespace = family_cli.importlib.import_module
    def reject_import(name, *args, **kwargs):
        if name == "families":
            return discover_namespace(name, *args, **kwargs)
        raise AssertionError(f"unexpected lazy import: {name}")
    monkeypatch.setattr(family_cli.importlib, "import_module", reject_import)
    for command in ("build", "extract-features"):
        with pytest.raises(SystemExit) as result:
            family_cli.main(["dinov3", command, "--help"])
        assert result.value.code == 0
    assert "--runtime-root" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        family_cli.main(["dinov3", "build", "checkpoint", "-o", "out.bundle", "--tensor-parallel-size", "2"])
    assert result.value.code == 2


def test_bundle_selects_backend_before_import_and_publishes_atomically(monkeypatch, tmp_path):
    events = []
    output = tmp_path / "model.bundle"
    def build(request, writer):
        events.append("build")
        writer.set_header(family="dinov3", task="image_features", backend=request.backend)
        writer.add_bytes("engine.plan", b"fixture")
    def select(backend):
        events.append(backend)
        monkeypatch.setitem(sys.modules, "families.dinov3.model", SimpleNamespace(build=build))
    monkeypatch.setattr(cli, "select_backend", select)
    cli.build_bundle(cli.BuildRequest(tmp_path, backend="trt_rtx"), output)
    assert events == ["trt_rtx", "build"]
    published = output.read_bytes()
    assert published.startswith(b"BUNDLE")
    def fail(request, writer):
        writer.set_header(family="dinov3", task="image_features", backend=request.backend)
        writer.add_bytes("engine.plan", b"partial")
        raise RuntimeError("owner build failed")
    monkeypatch.setattr(cli, "select_backend", lambda backend: monkeypatch.setitem(
        sys.modules, "families.dinov3.model", SimpleNamespace(build=fail)))
    with pytest.raises(RuntimeError, match="owner build failed"):
        cli.build_bundle(cli.BuildRequest(tmp_path), output)
    assert output.read_bytes() == published
    assert list(tmp_path.iterdir()) == [output]


def test_real_manifests_serialize_through_the_owner_handler(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(cli, "resolve_model", lambda model, revision: tmp_path)
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: captured.append((request, output)))
    spec = family_cli.load_family_cli("dinov3")["commands"][0]
    manifests = sorted((Path(__file__).parent / "manifests").glob("*.json"))
    assert manifests
    for path in manifests:
        model = ManifestCatalog._load(path)
        values = dict(model.build_settings)
        values.update(model=model.hf_id, output=str(tmp_path / model.bundle_name), task=model.task, precision=model.precision)
        argv = family_cli.serialize_arguments(spec, values)
        assert family_cli.main(["dinov3", "build", *argv]) == 0
        request, output = captured[-1]
        assert type(request) is cli.BuildRequest and request.precision == model.precision
        assert request.task == model.task and output == tmp_path / model.bundle_name
    raw = json.loads(manifests[0].read_text())
    raw["build"] = {"unknown_owner_option": 1}
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(BenchmarkError, match="undeclared build fields"):
        ManifestCatalog._load(invalid)


def test_legacy_sequence_limit_remains_accepted_but_unexposed(tmp_path):
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "dinov3", "image_features", "fp32", max_sequence_length=512)
    assert cli.coerce_request(legacy) == cli.BuildRequest(tmp_path)
    for field in ("image_height", "image_width"):
        with pytest.raises(NotImplementedError):
            cli.coerce_request(replace(legacy, **{field: 32}))


def test_e2e_build_preserves_single_device_rejection(monkeypatch, tmp_path):
    from families.dinov3.tests import test_e2e

    calls = []
    monkeypatch.setattr(test_e2e, "build_bundle", lambda *args: calls.append(args))
    manifest = json.loads(next((Path(__file__).parent / "manifests").glob("*.json")).read_text())
    manifest["tensor_parallel_size"] = 2
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        test_e2e._build(tmp_path, tmp_path / "out.bundle", manifest)
    assert not calls
