# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-precision contract tests for the VoiceChat EAR-TTS builder."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _native_tts(monkeypatch: pytest.MonkeyPatch):
    # The unit tests exercise graph wiring with a local TensorRT double and do
    # not require TensorRT to be installed in the test environment.
    module_name = "families.nemotron_voicechat.native_tts"
    package = importlib.import_module("families.nemotron_voicechat")
    missing = object()
    previous_module = sys.modules.pop(module_name, missing)
    previous_attribute = package.__dict__.pop("native_tts", missing)
    try:
        with monkeypatch.context() as isolated:
            isolated.setitem(sys.modules, "tensorrt", ModuleType("tensorrt"))
            native_tts = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        package.__dict__.pop("native_tts", None)
        if previous_module is not missing:
            sys.modules[module_name] = previous_module
        if previous_attribute is not missing:
            package.native_tts = previous_attribute
    return native_tts


def test_native_tts_stub_import_does_not_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "families.nemotron_voicechat.native_tts"
    native_tts = _native_tts(monkeypatch)
    package = importlib.import_module("families.nemotron_voicechat")

    assert sys.modules.get(module_name) is not native_tts
    assert getattr(package, "native_tts", None) is not native_tts


def test_fallback_tokenizer_download_is_revision_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_tts = _native_tts(monkeypatch)
    received: dict[str, object] = {}
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        received.update(kwargs)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert native_tts._resolve_tokenizer_snapshot(None) == tmp_path
    assert received == {
        "repo_id": native_tts.TEXT_MODEL_ID,
        "revision": native_tts.TEXT_MODEL_REVISION,
        "allow_patterns": ["tokenizer.json"],
    }


@pytest.mark.parametrize(
    ("linear_precision", "linear_dtype", "weight_dtype", "cast_dtypes"),
    [
        ("fp32", "float32", np.dtype(np.float32), []),
        ("fp16", "float16", np.dtype(np.float16), ["float16", "float32"]),
    ],
)
def test_static_linear_precision_is_confined_to_the_matmul(
    monkeypatch: pytest.MonkeyPatch,
    linear_precision: str,
    linear_dtype: str,
    weight_dtype: np.dtype,
    cast_dtypes: list[str],
) -> None:
    native_tts = _native_tts(monkeypatch)

    class Tensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype

    class Layer:
        def __init__(self, output):
            self.output = output

        def get_output(self, index):
            assert index == 0
            return self.output

    class Weights:
        def __init__(self, values):
            self.values = np.array(values, copy=True)

    class Trt:
        float16 = "float16"
        float32 = "float32"

        class MatrixOperation:
            NONE = "none"

        class ElementWiseOperation:
            SUM = "sum"

    Trt.Weights = Weights

    class Network:
        def __init__(self):
            self.constants = []
            self.casts = []
            self.matmuls = []

        def add_constant(self, shape, weights):
            dtype = Trt.float16 if weights.values.dtype == np.float16 else Trt.float32
            output = Tensor(shape, dtype)
            self.constants.append((tuple(shape), weights.values, output))
            return Layer(output)

        def add_cast(self, tensor, dtype):
            self.casts.append((tensor, dtype))
            return Layer(Tensor(tensor.shape, dtype))

        def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
            assert lhs_op == rhs_op == Trt.MatrixOperation.NONE
            assert lhs.dtype == rhs.dtype
            self.matmuls.append((lhs, rhs))
            return Layer(Tensor(lhs.shape[:-1] + (rhs.shape[-1],), lhs.dtype))

        def add_elementwise(self, lhs, rhs, operation):
            assert operation == Trt.ElementWiseOperation.SUM
            assert lhs.dtype == rhs.dtype == Trt.float32
            return Layer(Tensor(lhs.shape, lhs.dtype))

    network = Network()
    weights = native_tts.NativeTTSWeights(
        {
            "projection.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
            "projection.bias": np.arange(4, dtype=np.float32),
        }
    )
    context = native_tts._GraphContext(
        network,
        Trt,
        weights,
        Trt.float32,
        np.float32,
        Trt.float16 if linear_precision == "fp16" else Trt.float32,
        np.float16 if linear_precision == "fp16" else np.float32,
    )

    output = native_tts._linear(
        context,
        Tensor((2, 1, 3), Trt.float32),
        "projection.weight",
        "projection.bias",
    )

    lhs, rhs = network.matmuls[0]
    assert lhs.dtype == rhs.dtype == linear_dtype
    assert rhs.shape == (1, 3, 4)
    assert output.dtype == Trt.float32
    assert network.constants[0][1].dtype == weight_dtype
    assert network.constants[1][1].dtype == np.float32
    assert [dtype for _tensor, dtype in network.casts] == cast_dtypes


def test_tts_sections_normalize_and_forward_fp16(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    native_tts = _native_tts(monkeypatch)
    captured: dict[str, object] = {}

    def build_engine(model_dir, tokenizer_dir, **kwargs):
        captured.update(model_dir=model_dir, tokenizer_dir=tokenizer_dir, **kwargs)
        return b"tts-plan"

    monkeypatch.setattr(native_tts, "build_native_tts_engine", build_engine)
    monkeypatch.setattr(
        native_tts,
        "_load_runtime_code_assets",
        lambda _model_dir: (
            np.zeros(native_tts.EXACT_CONFIG.num_quantizers, dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        native_tts,
        "_load_aria_warmup_assets",
        lambda _model_dir: (
            np.zeros((37, native_tts.EXACT_CONFIG.hidden_size), dtype=np.float32),
            {},
        ),
    )

    sections = native_tts.build_tts_sections(
        tmp_path,
        tokenizer_dir=tmp_path / "tokenizer",
        max_cache_length=512,
        linear_precision="FP16",
    )

    assert sections[0] == ("tts.plan", b"tts-plan")
    assert captured["linear_precision"] == "fp16"
    assert captured["max_cache_length"] == 512
    with pytest.raises(ValueError, match="linear_precision must be 'fp32' or 'fp16'"):
        native_tts.build_tts_sections(tmp_path, linear_precision="bf16")
