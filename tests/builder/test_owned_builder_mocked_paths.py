"""Unit tests for owned builder modules using deterministic mocks/stubs.

Trace: ARCH-ENG-001, UD-ENG-MOCKED
Intent: Validate builder module code paths using fake TRT stubs without real GPU execution
Preconditions: Fake tensorrt module is injected; no real TRT or GPU available
Postconditions: Builder modules execute expected code paths and produce correct intermediate structures
"""

from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# Ensure imports resolve to this workspace's Python package.
_PKG_ROOT = Path(__file__).resolve().parents[2] / "tensorrt_model_connect"
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


def _import_with_fake_trt(module_name: str, fake_trt: types.SimpleNamespace | None = None):
    if fake_trt is None:
        fake_trt = _make_fake_trt_base()
    # Remove cached modules that may have imported with a different trt object.
    _drop_imported_module(module_name)
    if module_name.endswith("encoder_builder"):
        _drop_imported_module("tensorrt_model_connect.graph_ops")
        _drop_imported_module("tensorrt_model_connect.graph_blocks")
    with patch.dict(sys.modules, {"tensorrt": fake_trt}):
        return importlib.import_module(module_name)


@pytest.mark.unit
def test_encodec_fuse_weight_norm_matches_manual_formula() -> None:
    """Intent: verify weight-norm fusion helper math is deterministic.

    Preconditions: `g` and `v` arrays have compatible output-channel shapes.
    Postconditions: Fused weights equal g*v/||v|| with float32 output dtype.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bark.encodec_builder")

    g = np.array([[[2.0]], [[4.0]]], dtype=np.float32)
    v = np.array(
        [
            [[3.0, 4.0]],  # norm = 5
            [[6.0, 8.0]],  # norm = 10
        ],
        dtype=np.float32,
    )

    fused = mod._fuse_weight_norm(g, v)
    expected = np.array([[[1.2, 1.6]], [[2.4, 3.2]]], dtype=np.float32)

    np.testing.assert_allclose(fused, expected, atol=1e-6)
    assert fused.dtype == np.float32


@pytest.mark.unit
def test_encoder_seq_layer_norm_uses_native_normalization() -> None:
    """Intent: verify layer-norm helper uses TRT native add_normalization_v2.

    Preconditions: Fake network implements add_normalization_v2 API.
    Postconditions: Function returns tensor from native normalization layer
        with correct epsilon and axis mask.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.bert.encoder_builder")

    class _FakeTensor:
        def __init__(self, name: str, dtype=np.float32):
            self.name = name
            self.dtype = dtype

    class _FakeNormLayer:
        def __init__(self):
            self.epsilon = 0.0
            self._out = _FakeTensor("norm_out")

        def get_output(self, _i: int):
            return self._out

    class _FakeNetwork:
        def __init__(self):
            self.norm_calls: list[tuple] = []

        def add_normalization_v2(self, inp, gamma, beta, axis_mask):
            self.norm_calls.append((inp, gamma, beta, axis_mask))
            return _FakeNormLayer()

    add_constant_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _fake_add_constant(_network, shape, values, **_kw):
        add_constant_calls.append((tuple(shape), tuple(np.asarray(values).shape)))
        return _FakeTensor(f"const_{len(add_constant_calls)}", _kw.get("dtype", np.float32))

    with patch.object(mod.graph_ops, "add_constant", side_effect=_fake_add_constant):
        net = _FakeNetwork()
        out = mod._add_seq_layer_norm(
            network=net,
            inp=_FakeTensor("in"),
            hidden_size=4,
            seq_length=3,
            gamma=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            beta=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            eps=1e-6,
        )

    assert isinstance(out, _FakeTensor)
    assert len(net.norm_calls) == 1  # single native normalization call
    assert net.norm_calls[0][3] == (1 << 1)  # axis_mask = hidden dim
    assert add_constant_calls[0][0] == (1, 4)  # gamma [1, hidden]
    assert add_constant_calls[1][0] == (1, 4)  # beta [1, hidden]


@pytest.mark.unit
def test_onnx_builder_raises_parser_error_with_details() -> None:
    """Intent: validate ONNX parser failure branch and error aggregation.

    Preconditions: Fake parser returns parse=False with two errors.
    Postconditions: RuntimeError includes ONNX parser failure text and details.
    """
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
    mod = _import_with_fake_trt("tensorrt_model_connect.families.qwen_vl.onnx_vision_builder", fake_trt=fake_trt)

    with pytest.raises(RuntimeError, match="ONNX parsing failed"):
        mod.build_vision_engine_from_onnx(b"bad-onnx")


@pytest.mark.unit
def test_onnx_builder_success_and_plan_none_branches() -> None:
    """Intent: validate success path and build-plan None failure path.

    Preconditions: Fake parser always succeeds; builder return value is configurable.
    Postconditions: Success returns bytes; None plan raises RuntimeError.
    """
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
    mod = _import_with_fake_trt("tensorrt_model_connect.families.qwen_vl.onnx_vision_builder", fake_trt=fake_trt)

    plan = mod.build_vision_engine_from_onnx(b"good-onnx", verbose=True)
    assert plan == b"engine-plan"
    # EXPLICIT_BATCH (1 << 0) | STRONGLY_TYPED (1 << 1) = 3
    assert _FakeBuilder.last_instance.flags == 3
    assert _FakeBuilder.last_instance.config.calls == [("workspace", 1 << 30)]

    _FakeBuilder.plan_to_return = None
    with pytest.raises(RuntimeError, match="TensorRT vision engine build failed"):
        mod.build_vision_engine_from_onnx(b"good-onnx")


@pytest.mark.unit
def test_trace_hf_vision_encoder_import_and_missing_encoder_branches() -> None:
    """Intent: validate trace helper handles missing transformers and missing vision submodule.

    Preconditions: torch and transformers modules are replaced by deterministic fakes.
    Postconditions: Missing dependency raises ImportError; missing vision attr raises RuntimeError.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.qwen_vl.onnx_vision_builder")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.randn = lambda *_shape: np.zeros((1, 3, 8, 8), dtype=np.float32)  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    fake_torch.onnx = types.SimpleNamespace(
        export=lambda *_args, **_kwargs: None,
    )

    config = types.SimpleNamespace(raw={"vision_config": {"image_size": 8}})

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": None}):
        with pytest.raises(ImportError, match="transformers is required"):
            mod.trace_hf_vision_encoder("unused", config)

    class _AutoModelMissingVision:
        @staticmethod
        def from_pretrained(_model_dir, trust_remote_code=False):
            class _Model:
                pass

            return _Model()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = _AutoModelMissingVision  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
        with pytest.raises(RuntimeError, match="Could not find vision encoder"):
            mod.trace_hf_vision_encoder("unused", config)


@pytest.mark.unit
def test_trace_hf_vision_encoder_success_path_with_mocked_export() -> None:
    """Intent: validate successful ONNX tracing path delegates bytes to TRT builder helper.

    Preconditions: Fake torch exporter writes deterministic ONNX bytes; fake transformers returns vision model.
    Postconditions: Returned engine bytes come from `build_vision_engine_from_onnx` call.
    """
    mod = _import_with_fake_trt("tensorrt_model_connect.families.qwen_vl.onnx_vision_builder")

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
    fake_torch.randn = lambda *_shape: np.zeros((1, 3, 16, 16), dtype=np.float32)  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
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
    fake_transformers.AutoModel = _AutoModel  # type: ignore[attr-defined]

    received: list[bytes] = []

    def _fake_build(onnx_bytes: bytes, *, verbose: bool = False) -> bytes:
        received.append(onnx_bytes)
        return b"engine-ok"

    with patch.object(mod, "build_vision_engine_from_onnx", side_effect=_fake_build), patch.dict(
        sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
    ):
        config = types.SimpleNamespace(raw={"vision_config": {"image_size": 16}})
        out = mod.trace_hf_vision_encoder("model-dir", config, verbose=True)

    assert out == b"engine-ok"
    assert received == [b"onnx-bytes"]
    assert _AutoModel.called == ("model-dir", False)
    assert vision_model.eval_called is True
    assert export_calls and export_calls[0]["opset_version"] == 17
