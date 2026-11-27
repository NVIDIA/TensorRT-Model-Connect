# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import numpy as np
import struct
import sys
import pytest
from families.hstu.config import Config, ServingProfile
from families.hstu.triton.native_client import (
    NativeClient,
    save_snapshot,
    snapshot_metadata,
)
from families.hstu.triton import native_repository


def test_snapshot_atomic_publication_and_bounded_metadata(tmp_path):
    p = tmp_path / "history.kv"
    save_snapshot(p, b"old")
    with p.open("rb") as reader:
        save_snapshot(p, b"new")
        assert reader.read() == b"old"
    assert p.read_bytes() == b"new" and list(tmp_path.iterdir()) == [p]
    header = json.dumps({"revisions": ["public-v1"]}).encode()
    assert snapshot_metadata(b"HSTUKV2\0" + struct.pack("<Q", len(header)) + header)[
        "revisions"
    ] == ["public-v1"]
    for bad in [
        b"",
        b"HSTUKV2\0" + struct.pack("<Q", 65537),
        b"HSTUKV2\0" + struct.pack("<Q", len(header) + 1) + header,
    ]:
        with pytest.raises(ValueError):
            snapshot_metadata(bad)


@pytest.mark.parametrize("option", ["max_entries", "ttl_seconds", "lease_seconds"])
def test_client_rejects_ignored_controls_without_loading(option):
    with pytest.raises(ValueError, match="hydration=True"):
        NativeClient("unused.bundle", "unused-sdk", **{option: 1})


def test_native_repository_uses_public_tensor_metadata(tmp_path, monkeypatch):
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"example bundle")
    monkeypatch.setattr(
        native_repository, "runtime_metadata", lambda *args: Config().runtime(4, "bf16")
    )
    out = native_repository.generate(
        bundle, tmp_path / "models", "hstu", parameters={"kv_cache_memory_ratio": 0.05}
    )
    text = (out / "config.pbtxt").read_text()
    assert 'backend: "tensorrt"' in text and 'platform: "model_connect"' in text
    assert "dims: [512, 512]" in text and "dims: [128, 512]" in text and "dims: [128]" in text
    assert " },\n]" not in text
    assert 'key: "kv_cache_memory_ratio" value { string_value: "0.05" }' in text
    assert (out / "1/model.bundle").read_bytes() == bundle.read_bytes()
    with pytest.raises(FileExistsError):
        native_repository.generate(bundle, tmp_path / "models", "hstu")


def test_fixed_batch_repository_exposes_explicit_batch_dimensions(tmp_path, monkeypatch):
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"example bundle")
    monkeypatch.setattr(
        native_repository,
        "runtime_metadata",
        lambda *args: Config(
            serving=ServingProfile(history_length=64, candidates=128, batching="fixed")
        ).runtime(4, "bf16"),
    )
    target = native_repository.generate(bundle, tmp_path / "models", "fixed")
    text = (target / "config.pbtxt").read_text()
    assert "max_batch_size: 0" in text
    assert "dims: [4, 64]" in text and "dims: [4, 128]" in text and "dims: [4, 1]" in text


@pytest.mark.parametrize("history", [False, True])
def test_python_repository_preserves_fixed_batch_contract(tmp_path, monkeypatch, history):
    from families.hstu.triton import generate_repository

    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"example bundle")
    monkeypatch.setattr(
        generate_repository,
        "runtime_metadata",
        lambda *args: Config(
            serving=ServingProfile(history_length=64, candidates=128, batching="fixed")
        ).runtime(4, "bf16"),
    )
    flags = ["--history-cache", "--kv-cache-size-bytes", "134217728"] if history else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_repository",
            "--bundle",
            str(bundle),
            "--repository",
            str(tmp_path / "models"),
            "--runtime-root",
            "/unused-sdk",
            *flags,
        ],
    )
    generate_repository.main()
    text = (tmp_path / "models/hstu/config.pbtxt").read_text()
    assert "max_batch_size: 0" in text
    assert "dims: [4, 64]" in text and "dims: [4, 128]" in text and "dims: [4, 1]" in text
    if history:
        assert 'name: "CACHE_CONTROL", data_type: TYPE_STRING, dims: [1]' in text
        assert 'name: "CACHE_SNAPSHOT", data_type: TYPE_UINT8, dims: [-1]' in text


@pytest.mark.parametrize(
    "flags",
    [
        ["--kv-cache-memory-ratio", "0"],
        ["--kv-cache-memory-ratio", "nan"],
        ["--kv-cache-memory-ratio", "1.1"],
        ["--kv-cache-size-bytes", "0"],
        ["--kv-cache-size-bytes", "8", "--kv-cache-memory-ratio", ".1"],
    ],
)
def test_cli_rejects_invalid_budgets_before_writing(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_repository",
            "--bundle",
            "unused.bundle",
            "--repository",
            str(tmp_path / "models"),
            "--runtime-root",
            "/unused-sdk",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        native_repository.main()
    assert error.value.code == 2 and not (tmp_path / "models").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        {"kv_cache_size_bytes": -1},
        {"kv_cache_size_bytes": "8388608junk"},
        {"kv_cache_size_bytes": True},
        {"kv_cache_size_bytes": 2**64},
        {"kv_cache_memory_ratio": "0.05junk"},
        {"kv_cache_memory_ratio": float("nan")},
        {"kv_cache_memory_ratio": 0.0},
        {"kv_cache_size_bytes": 8, "kv_cache_memory_ratio": 0.1},
        {"kv_cache_hydration_enabled": True},
    ],
)
def test_repository_api_rejects_invalid_options_before_writing(tmp_path, parameters):
    with pytest.raises(ValueError):
        native_repository.generate(
            tmp_path / "unused.bundle", tmp_path / "models", "hstu", parameters=parameters
        )
    assert not (tmp_path / "models").exists()


def test_prototype_headers_preserve_image_loader_and_cache_abi(tmp_path):
    from ..triton.build_image_plugin import prototype_headers

    root = Path(__file__).resolve().parents[3]
    prototype_headers(root, tmp_path)
    headers = tmp_path / "sdk-compat/trtmc/runtime"
    assert "kInt64,\n    kFloat8E4M3," in (headers / "tensor.h").read_text()
    assert "double kv_cache_memory_ratio{0.0};" in (headers / "family_factory.h").read_text()
    assert (headers / "family_loader.h").read_text().count(
        "double kv_cache_memory_ratio = 0.0"
    ) == 2


def test_example_help_documents_inputs_without_loading_model(monkeypatch, capsys):
    from ..examples.history_cache import main

    monkeypatch.setattr(sys, "argv", ["history_cache", "--help"])
    with pytest.raises(SystemExit) as status:
        main()
    assert status.value.code == 0
    assert "--inputs" in capsys.readouterr().out


def test_python_transport_requires_each_named_tensor(monkeypatch):
    import types

    monkeypatch.setitem(
        sys.modules,
        "triton_python_backend_utils",
        types.SimpleNamespace(get_input_tensor_by_name=lambda request, name: request.get(name)),
    )
    from ..triton.model import TritonPythonModel

    transport = TritonPythonModel()
    transport.batch_min, transport.batch_max = 1, 4
    with pytest.raises(ValueError, match="history_lengths"):
        transport.arrays({}, ["history_lengths"])
    tensor = types.SimpleNamespace(as_numpy=lambda: np.array([[1]], dtype=np.int32))
    arrays, descriptors = transport.arrays({"history_lengths": tensor}, ["history_lengths"])
    assert descriptors[0]["dtype"] == "int32" and descriptors[0]["shape"] == [1, 1]
    assert descriptors[0]["pointer"] == arrays[0].ctypes.data


def test_native_generation_failure_is_retryable(tmp_path, monkeypatch):
    bundle = tmp_path / "input.bundle"
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(
        native_repository, "runtime_metadata", lambda *_: Config().runtime(4, "fp32")
    )
    copy = native_repository.shutil.copyfile

    def fail_copy(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(native_repository.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        native_repository.generate(bundle, tmp_path / "models", "ranking")
    assert list((tmp_path / "models").iterdir()) == []
    monkeypatch.setattr(native_repository.shutil, "copyfile", copy)
    assert native_repository.generate(bundle, tmp_path / "models", "ranking").is_dir()


def test_explicit_dot_runtime_uses_exact_bridge(tmp_path, monkeypatch):
    import types
    from ..triton import generate_repository

    loaded = []
    library = types.SimpleNamespace(
        hstu_runtime_metadata=lambda _: b'{"task":"structured_ranking"}',
        hstu_last_error=lambda: b"unused",
    )
    monkeypatch.setattr(
        generate_repository.ctypes, "CDLL", lambda path: loaded.append(path) or library
    )
    monkeypatch.chdir(tmp_path)
    assert generate_repository.runtime_metadata("model.bundle", ".")["task"] == "structured_ranking"
    assert loaded == [str(tmp_path / "libhstu_triton.so")]


def test_python_transport_checks_batch_before_copy(monkeypatch):
    import types
    from ..triton import model

    monkeypatch.setattr(
        model, "pb", types.SimpleNamespace(get_input_tensor_by_name=lambda r, n: r[n])
    )
    transport = model.TritonPythonModel()
    transport.batch_min, transport.batch_max = 1, 4

    def unexpected_copy(_value):
        raise AssertionError("copied an invalid batch")

    monkeypatch.setattr(model.np, "ascontiguousarray", unexpected_copy)
    request = {"x": types.SimpleNamespace(as_numpy=lambda: np.zeros((5, 2), np.float32))}
    with pytest.raises(ValueError, match="batch"):
        transport.arrays(request, ["x"])


def test_metric_failure_preserves_inference_result_and_error(monkeypatch):
    import types
    from ..triton import model

    errors = []
    monkeypatch.setattr(
        model,
        "pb",
        types.SimpleNamespace(
            Logger=types.SimpleNamespace(log_error=errors.append),
            InferenceResponse=lambda **kwargs: kwargs,
            TritonError=str,
        ),
    )
    transport = model.TritonPythonModel()
    transport.hydration = True

    def fail_metrics():
        raise RuntimeError("metric unavailable")

    transport.publish = fail_metrics
    transport.history_execute = lambda _: ["committed handles"]
    request = types.SimpleNamespace(is_cancelled=lambda: False)
    assert transport.execute([request])[0]["output_tensors"] == ["committed handles"]

    def fail_request(_request):
        raise ValueError("invalid snapshot")

    transport.history_execute = fail_request
    assert transport.execute([request])[0]["error"] == "invalid snapshot"
    assert len(errors) == 2
