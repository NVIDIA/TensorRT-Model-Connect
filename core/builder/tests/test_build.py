# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import BuildRequest


build_core = importlib.import_module("tensorrt_model_connect.build")


def _request(tmp_path: Path, *, family: str = "example") -> BuildRequest:
    return BuildRequest(
        model_dir=tmp_path / "model",
        output_path=tmp_path / "model.bundle",
        precision="fp16",
        family=family,
        task="text_generation",
        tensor_parallel_size=2,
        context_parallel_size=3,
    )


def test_build_request_is_a_plain_frozen_dataclass(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(FrozenInstanceError):
        request.precision = "fp32"  # type: ignore[misc]

    assert BuildRequest.__bases__ == (object,)
    assert request.tensor_parallel_size == 2
    assert request.context_parallel_size == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_sequence_length", 0),
        ("image_height", 0),
        ("image_width", 0),
        ("video_num_frames", 0),
        ("max_batch_size", 0),
        ("tensor_parallel_size", 0),
        ("context_parallel_size", 0),
        ("quantization", ""),
        ("fp32_layers", (-1,)),
        ("graph_transform", object()),
    ],
)
def test_build_request_rejects_invalid_direct_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    kwargs = {
        "model_dir": tmp_path / "model",
        "output_path": tmp_path / "model.bundle",
        "family": "example",
        "task": "text_generation",
        "precision": "fp16",
        field: value,
    }
    with pytest.raises(ValueError):
        BuildRequest(**kwargs)  # type: ignore[arg-type]


def test_resolver_returns_only_the_explicit_family(tmp_path: Path) -> None:
    assert build_core._resolve_family(_request(tmp_path, family="exact_family")) == "exact_family"


@pytest.mark.parametrize("family", ["", "../other", "foo.bar", "MixedCase", "a-b"])
def test_family_rejects_names_that_are_not_safe_directories(tmp_path: Path, family: str) -> None:
    with pytest.raises(ValueError, match="lowercase identifier"):
        build_core._resolve_family(_request(tmp_path, family=family))


def test_load_family_imports_only_the_exact_model_module(monkeypatch) -> None:
    imported: list[str] = []
    expected = SimpleNamespace(build=lambda request, writer: None)

    def fake_import(name: str):
        imported.append(name)
        return expected

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert build_core._load_family("exact_family") is expected
    assert imported == ["families.exact_family.model"]


def test_family_internal_import_error_is_not_wrapped(monkeypatch) -> None:
    internal_error = ImportError("family dependency failed")

    def fail_import(_name: str):
        raise internal_error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as caught:
        build_core._load_family("exact_family")
    assert caught.value is internal_error


def test_family_internal_module_not_found_error_is_not_wrapped(monkeypatch) -> None:
    internal_error = ModuleNotFoundError(
        "No module named 'family_dependency'", name="family_dependency"
    )

    def fail_import(_name: str):
        raise internal_error

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError) as caught:
        build_core._load_family("exact_family")
    assert caught.value is internal_error


def test_build_finishes_after_family_returns(monkeypatch, tmp_path: Path) -> None:
    events: list[object] = []

    class FakeWriter:
        def __init__(self, destination: Path) -> None:
            events.append(("writer", destination))

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(request: BuildRequest, writer: FakeWriter) -> None:
        events.append(("build", request, writer))

    request = _request(tmp_path)
    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    assert build_core.build(request) is None
    assert events[0] == ("writer", request.output_path)
    assert events[1][0:2] == ("build", request)
    assert events[2:] == ["finish"]


def test_build_runs_graph_transform_before_family_engine_serialization(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[object] = []

    class FakeTrtBuilder:
        def __init__(self, _logger: object) -> None:
            pass

        def build_serialized_network(self, network: object, _config: object) -> bytes:
            events.append(("serialize", network))
            return b"engine"

    fake_trt = SimpleNamespace(Builder=FakeTrtBuilder)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(_request: BuildRequest, _writer: FakeWriter) -> None:
        network = SimpleNamespace(replaced=False)
        fake_trt.Builder("logger").build_serialized_network(network, "config")

    def transform(network: object, engine_index: int) -> None:
        setattr(network, "replaced", True)
        events.append(("transform", network, engine_index))

    request = replace(_request(tmp_path), graph_transform=transform)
    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    build_core.build(request)

    assert events[0][0] == "transform"
    assert events[0][1].replaced is True
    assert events[0][2] == 0
    assert events[1] == ("serialize", events[0][1])
    assert events[2] == "finish"
    assert fake_trt.Builder is FakeTrtBuilder


def test_build_aborts_and_preserves_family_error(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    family_error = RuntimeError("family failed")

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")

        def abort(self) -> None:
            events.append("abort")

    def family_build(_request: BuildRequest, _writer: FakeWriter) -> None:
        raise family_error

    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core, "_load_family", lambda family: SimpleNamespace(build=family_build)
    )

    with pytest.raises(RuntimeError) as caught:
        build_core.build(_request(tmp_path))
    assert caught.value is family_error
    assert events == ["abort"]


def test_build_aborts_if_finish_fails(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    class FakeWriter:
        def __init__(self, _destination: Path) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")
            raise OSError("publish failed")

        def abort(self) -> None:
            events.append("abort")

    monkeypatch.setattr(build_core, "BundleWriter", FakeWriter)
    monkeypatch.setattr(
        build_core,
        "_load_family",
        lambda family: SimpleNamespace(build=lambda request, writer: None),
    )

    with pytest.raises(OSError, match="publish failed"):
        build_core.build(_request(tmp_path))
    assert events == ["finish", "abort"]
