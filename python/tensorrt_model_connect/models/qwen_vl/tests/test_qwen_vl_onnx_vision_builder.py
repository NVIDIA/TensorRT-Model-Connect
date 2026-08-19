# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned ONNX vision builder tests using deterministic TRT stubs."""

from __future__ import annotations

import importlib
import io
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


_PKG_ROOT = Path(__file__).resolve().parents[5] / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def _make_fake_trt_base() -> types.SimpleNamespace:
    class _Logger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    return types.SimpleNamespace(
        Logger=_Logger,
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        Permutation=lambda dims: tuple(dims),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def _drop_imported_module(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    package_name, _, attribute_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attribute_name):
        delattr(package, attribute_name)


@contextmanager
def _import_with_fake_trt(
    module_name: str,
    fake_trt: types.SimpleNamespace | None = None,
) -> Iterator[object]:
    if fake_trt is None:
        fake_trt = _make_fake_trt_base()
    _drop_imported_module(module_name)
    from tensorrt_model_connect import trt_compat

    previous_trt = sys.modules.get("tensorrt")
    previous_module = trt_compat._module
    sys.modules["tensorrt"] = fake_trt
    trt_compat._module = fake_trt
    try:
        yield importlib.import_module(module_name)
    finally:
        _drop_imported_module(module_name)
        trt_compat._module = previous_module
        if previous_trt is None:
            sys.modules.pop("tensorrt", None)
        else:
            sys.modules["tensorrt"] = previous_trt


@pytest.mark.unit
def test_onnx_builder_raises_parser_error_with_details() -> None:
    """Qwen-VL ONNX builder reports parser failures with details."""
    class _FakeLogger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    class _FakeNetwork:
        def __init__(self):
            self.num_layers = 7

    class _FakeBuilderConfig:
        def set_memory_pool_limit(self, *_args, **_kwargs):
            return None

    class _FakeBuilder:
        def __init__(self, _logger):
            self.network = _FakeNetwork()

        def create_network(self, _flags):
            return self.network

        def create_builder_config(self):
            return _FakeBuilderConfig()

        def build_serialized_network(self, _network, _config):
            return b"unused"

    class _FakeParser:
        def __init__(self, _network, _logger):
            self._errors = ["node conversion failed", "unsupported op"]

        def parse(self, _onnx_bytes: bytes) -> bool:
            return False

        @property
        def num_errors(self) -> int:
            return len(self._errors)

        def get_error(self, i: int) -> str:
            return self._errors[i]

    fake_trt = types.SimpleNamespace(
        Logger=_FakeLogger,
        Builder=_FakeBuilder,
        OnnxParser=_FakeParser,
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
    )
    with _import_with_fake_trt(
        "tensorrt_model_connect.models.qwen_vl.onnx_vision_builder",
        fake_trt=fake_trt,
    ) as mod:
        with pytest.raises(RuntimeError, match="ONNX parsing failed"):
            mod.build_vision_engine_from_onnx(b"bad-onnx")


@pytest.mark.unit
def test_onnx_builder_success_and_plan_none_branches() -> None:
    """Qwen-VL ONNX builder succeeds and reports TensorRT build failure."""
    class _FakeLogger:
        VERBOSE = 2
        WARNING = 1

        def __init__(self, _level):
            pass

    class _FakeNetwork:
        def __init__(self):
            self.num_layers = 5

    class _FakeBuilderConfig:
        def __init__(self):
            self.calls: list[tuple[object, int]] = []

        def set_memory_pool_limit(self, pool, size):
            self.calls.append((pool, size))

    class _FakeBuilder:
        last_instance = None
        plan_to_return = b"engine-plan"

        def __init__(self, _logger):
            self.network = _FakeNetwork()
            self.config = _FakeBuilderConfig()
            self.flags = None
            _FakeBuilder.last_instance = self

        def create_network(self, flags):
            self.flags = flags
            return self.network

        def create_builder_config(self):
            return self.config

        def build_serialized_network(self, _network, _config):
            return _FakeBuilder.plan_to_return

    class _FakeParser:
        def __init__(self, _network, _logger):
            self.onnx_bytes = b""

        def parse(self, onnx_bytes: bytes) -> bool:
            self.onnx_bytes = onnx_bytes
            return True

        @property
        def num_errors(self) -> int:
            return 0

        def get_error(self, i: int) -> str:
            raise IndexError(i)

    fake_trt = types.SimpleNamespace(
        Logger=_FakeLogger,
        Builder=_FakeBuilder,
        OnnxParser=_FakeParser,
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
    )
    with _import_with_fake_trt(
        "tensorrt_model_connect.models.qwen_vl.onnx_vision_builder",
        fake_trt=fake_trt,
    ) as mod:
        plan = mod.build_vision_engine_from_onnx(b"good-onnx", verbose=True)
        assert plan == b"engine-plan"
        assert _FakeBuilder.last_instance.flags == 3
        assert _FakeBuilder.last_instance.config.calls == [("workspace", 1 << 30)]

        _FakeBuilder.plan_to_return = None
        with pytest.raises(RuntimeError, match="TensorRT vision engine build failed"):
            mod.build_vision_engine_from_onnx(b"good-onnx")


@pytest.mark.unit
def test_trace_hf_vision_encoder_import_and_missing_encoder_branches() -> None:
    """Qwen-VL trace helper reports missing dependencies and vision module."""
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.randn = lambda *_shape: np.zeros((1, 3, 8, 8), dtype=np.float32)
    fake_torch.no_grad = lambda: _NoGrad()
    fake_torch.onnx = types.SimpleNamespace(export=lambda *_args, **_kwargs: None)

    config = types.SimpleNamespace(raw={"vision_config": {"image_size": 8}})

    class _AutoModelMissingVision:
        @staticmethod
        def from_pretrained(_model_dir, trust_remote_code=False):
            class _Model:
                pass

            return _Model()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _AutoModelMissingVision

    with _import_with_fake_trt(
        "tensorrt_model_connect.models.qwen_vl.onnx_vision_builder"
    ) as mod:
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "torch", fake_torch)
            mp.setitem(sys.modules, "transformers", None)
            with pytest.raises(ImportError, match="transformers is required"):
                mod.trace_hf_vision_encoder("unused", config)

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "torch", fake_torch)
            mp.setitem(sys.modules, "transformers", fake_transformers)
            with pytest.raises(RuntimeError, match="Could not find vision encoder"):
                mod.trace_hf_vision_encoder("unused", config)


@pytest.mark.unit
def test_trace_hf_vision_encoder_success_path_with_mocked_export() -> None:
    """Qwen-VL trace helper delegates exported ONNX bytes to TRT builder."""
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    export_calls: list[dict[str, object]] = []

    def _fake_export(_model, _dummy_input, onnx_buffer: io.BytesIO, **kwargs):
        export_calls.append(kwargs)
        onnx_buffer.write(b"onnx-bytes")

    fake_torch = types.ModuleType("torch")
    fake_torch.randn = lambda *_shape: np.zeros((1, 3, 16, 16), dtype=np.float32)
    fake_torch.no_grad = lambda: _NoGrad()
    fake_torch.onnx = types.SimpleNamespace(export=_fake_export)

    class _VisionModel:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True

    vision_model = _VisionModel()

    class _AutoModel:
        called: tuple[str, bool] | None = None

        @staticmethod
        def from_pretrained(model_dir, trust_remote_code=False):
            _AutoModel.called = (model_dir, trust_remote_code)
            return types.SimpleNamespace(vision_model=vision_model)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _AutoModel

    received: list[bytes] = []

    def _fake_build(onnx_bytes: bytes, *, verbose: bool = False) -> bytes:
        received.append(onnx_bytes)
        return b"engine-ok"

    with _import_with_fake_trt(
        "tensorrt_model_connect.models.qwen_vl.onnx_vision_builder"
    ) as mod:
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "torch", fake_torch)
            mp.setitem(sys.modules, "transformers", fake_transformers)
            mp.setattr(mod, "build_vision_engine_from_onnx", _fake_build)
            config = types.SimpleNamespace(raw={"vision_config": {"image_size": 16}})
            out = mod.trace_hf_vision_encoder("model-dir", config, verbose=True)

    assert out == b"engine-ok"
    assert received == [b"onnx-bytes"]
    assert _AutoModel.called == ("model-dir", False)
    assert vision_model.eval_called is True
    assert export_calls and export_calls[0]["opset_version"] == 17
