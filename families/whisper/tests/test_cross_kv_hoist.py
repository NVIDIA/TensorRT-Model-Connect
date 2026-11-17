# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper's cross-attention K/V projected once per request by the encoder engine.

A single-GPU bundle's encoder outputs every decoder layer's cross-attention K
and V, split into heads, and its decoder reads them in every step instead of
projecting the encoder output again. Tensor-parallel builds keep the raw
encoder output, which each rank's decoder shards and projects itself.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for Whisper engine builds")
pytest.importorskip("safetensors.numpy")

from .. import graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import WhisperQuantContext, calibrate_whisper  # noqa: E402
from .test_quantization import (  # noqa: E402
    D,
    HEADS,
    LAYERS,
    MEL,
    SRC,
    _all_scales,
    _write_checkpoint,
)

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

HEAD_DIM = D // HEADS
CROSS_SHAPE = (1, HEADS, SRC, HEAD_DIM)
# FP16 heads against a float64 reference, as an absolute error. The heads of
# the tiny checkpoint stay below 4 in magnitude, where FP16 steps by 2**-9 at
# most: four such steps.
FP16_TOLERANCE = 4 * 2.0**-9


def _require_gpu() -> None:
    try:
        trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(0)
    except Exception as error:  # noqa: BLE001 - any builder failure means no usable GPU
        pytest.skip(f"engine build needs a GPU ({error})")


def _tiny(path: Path, fp32_layers=()):
    """A random tiny Whisper checkpoint: (plugin, config, weights)."""
    config = _write_checkpoint(path)
    config.raw["_model_dir"] = str(path)
    config.raw["_fp32_layers"] = tuple(fp32_layers)
    whisper = model._WhisperModel()
    return whisper, config, whisper.load_weights(str(path), config)


def _io(plan: bytes) -> dict[str, tuple[trt.TensorIOMode, tuple[int, ...], trt.DataType]]:
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    tensors = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors[name] = (
            engine.get_tensor_mode(name),
            tuple(engine.get_tensor_shape(name)),
            engine.get_tensor_dtype(name),
        )
    return tensors


def _run(plan: bytes, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run one engine on host arrays; return every output as a host array."""
    cudart = pytest.importorskip("cuda.bindings.runtime")

    def check(result):
        error = result[0] if isinstance(result, tuple) else result
        assert error == cudart.cudaError_t.cudaSuccess, error
        return result[1] if isinstance(result, tuple) and len(result) > 1 else None

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    buffers, outputs = [], {}
    try:
        for name, array in inputs.items():
            array = np.ascontiguousarray(array)
            pointer = check(cudart.cudaMalloc(array.nbytes))
            buffers.append(pointer)
            check(
                cudart.cudaMemcpy(
                    pointer,
                    array.ctypes.data,
                    array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            context.set_tensor_address(name, pointer)
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            host = np.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=trt.nptype(engine.get_tensor_dtype(name)),
            )
            pointer = check(cudart.cudaMalloc(host.nbytes))
            buffers.append(pointer)
            context.set_tensor_address(name, pointer)
            outputs[name] = (host, pointer)
        assert context.execute_async_v3(0)
        check(cudart.cudaDeviceSynchronize())
        for host, pointer in outputs.values():
            check(
                cudart.cudaMemcpy(
                    host.ctypes.data,
                    pointer,
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
    finally:
        for pointer in buffers:
            cudart.cudaFree(pointer)
    return {name: host for name, (host, _) in outputs.items()}


def _mel(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((MEL, 2 * SRC)).astype(np.float32)


def _heads(rows: np.ndarray) -> np.ndarray:
    """[positions, hidden] -> [1, heads, positions, head_dim]."""
    return rows.reshape(SRC, HEADS, HEAD_DIM).transpose(1, 0, 2)[np.newaxis]


def _projection(weights, layer: int, kind: str, states: np.ndarray) -> np.ndarray:
    """One layer's K or V of the encoder states: what the decoder used to compute."""
    weight = weights[f"layer.{layer}.cross_w_{kind}"].astype(np.float64)
    bias = weights[f"layer.{layer}.cross_b_{kind}"].astype(np.float64)
    return _heads(states.astype(np.float64) @ weight + bias)


@pytest.mark.parametrize(
    ("precision", "fp32_layers", "cross_dtype"),
    [
        ("fp16", (), trt.float16),
        # An FP32 encoder still hands an FP16 decoder FP16 K/V.
        ("fp16", (0,), trt.float16),
        ("fp32", (), trt.float32),
    ],
)
def test_single_gpu_encoder_outputs_every_layers_cross_kv_heads(
    tmp_path: Path, precision, fp32_layers, cross_dtype
) -> None:
    _require_gpu()
    whisper, config, weights = _tiny(tmp_path, fp32_layers)
    encoder = _io(whisper.build_vision_engine(str(tmp_path), config, weights, precision=precision))
    decoder = _io(whisper.build_engine(config, weights, 8, precision=precision))

    assert "encoder_output" not in encoder
    assert encoder[model.ENCODER_PADDED_ROWS] == (trt.TensorIOMode.INPUT, (SRC, 1), trt.float32)
    for layer in range(LAYERS):
        for kind in ("cross_k", "cross_v"):
            name = f"{kind}_{layer}"
            assert encoder[name] == (trt.TensorIOMode.OUTPUT, CROSS_SHAPE, cross_dtype)
            assert decoder[name] == (trt.TensorIOMode.INPUT, CROSS_SHAPE, cross_dtype)


@pytest.mark.parametrize(
    ("precision", "fp32_layers", "tolerance"),
    [
        ("fp32", (), 1e-3),
        # FP16 K/V of an FP32 encoder's states: the whisper-tiny-fp16 manifest.
        ("fp16", (0,), FP16_TOLERANCE),
        # An FP16 encoder, the only one an FP8 build can have.
        ("fp16", (), FP16_TOLERANCE),
    ],
)
def test_projected_cross_kv_equals_the_projection_of_the_raw_encoder_output(
    tmp_path: Path, precision, fp32_layers, tolerance
) -> None:
    """Each cross_k_<i>/cross_v_<i> is that layer's K/V projection (weight and
    bias) of the encoder output, split into [1, heads, positions, head_dim]:
    what the decoder used to compute in every step. Catches a K/V swap, a wrong
    bias or layer, or a missing head transpose, which the I/O contract alone
    would not."""
    _require_gpu()
    whisper, config, weights = _tiny(tmp_path, fp32_layers)
    mel = _mel()
    raw = _run(
        whisper.build_vision_engine(
            str(tmp_path), config, weights, precision=precision, project_cross_kv=False
        ),
        {"mel_features": mel},
    )["encoder_output"]
    projected = _run(
        whisper.build_vision_engine(str(tmp_path), config, weights, precision=precision),
        {"mel_features": mel, model.ENCODER_PADDED_ROWS: np.zeros((SRC, 1), dtype=np.float32)},
    )

    for layer in range(LAYERS):
        for kind in ("k", "v"):
            np.testing.assert_allclose(
                projected[f"cross_{kind}_{layer}"].astype(np.float64),
                _projection(weights, layer, kind, raw),
                rtol=0,
                atol=tolerance,
            )


@pytest.mark.parametrize(
    ("precision", "fp32_layers", "tolerance", "valid"),
    [
        ("fp32", (), 1e-3, 1),
        ("fp32", (), 1e-3, SRC // 2),
        ("fp32", (), 1e-3, SRC - 1),
        # The row select sits before the cast to the decoder's dtype.
        ("fp16", (0,), FP16_TOLERANCE, SRC // 2),
        ("fp16", (), FP16_TOLERANCE, SRC // 2),
    ],
)
def test_rows_past_the_valid_length_project_as_zero_states(
    tmp_path: Path, precision, fp32_layers, tolerance, valid: int
) -> None:
    """A row marked 1 in encoder_padded_rows reads as a zero encoder state, as
    when the runtime zeroed the raw encoder output past a short request; every
    other row is untouched."""
    _require_gpu()
    whisper, config, weights = _tiny(tmp_path, fp32_layers)
    mel = _mel(5)
    raw = _run(
        whisper.build_vision_engine(
            str(tmp_path), config, weights, precision=precision, project_cross_kv=False
        ),
        {"mel_features": mel},
    )["encoder_output"].copy()
    raw[valid:] = 0.0
    padded_rows = np.zeros((SRC, 1), dtype=np.float32)
    padded_rows[valid:] = 1.0
    projected = _run(
        whisper.build_vision_engine(str(tmp_path), config, weights, precision=precision),
        {"mel_features": mel, model.ENCODER_PADDED_ROWS: padded_rows},
    )

    for layer in range(LAYERS):
        for kind in ("k", "v"):
            np.testing.assert_allclose(
                projected[f"cross_{kind}_{layer}"].astype(np.float64),
                _projection(weights, layer, kind, raw),
                rtol=0,
                atol=tolerance,
            )
        # Whisper's key projection has no bias: a zeroed state has a zero key.
        assert not projected[f"cross_k_{layer}"][:, :, valid:, :].any()


def _decoder_step(plan: bytes, cross: dict[str, np.ndarray]) -> np.ndarray:
    """The logits of one decoder step (the first token, an empty self-attention
    cache) for these cross-attention K and V heads."""
    inputs = {}
    for name, (mode, shape, dtype) in _io(plan).items():
        if mode != trt.TensorIOMode.INPUT:
            continue
        if name in cross:
            inputs[name] = cross[name].astype(trt.nptype(dtype))
        elif name == "token_id":
            inputs[name] = np.array([1], dtype=np.int32)
        elif name == "position_id":
            inputs[name] = np.array([0], dtype=np.int32)
        elif name == "attention_mask":
            mask = np.full(shape, -1.0e4, dtype=np.float32)
            mask[0, -1] = 0.0
            inputs[name] = mask
        else:
            inputs[name] = np.zeros(shape, dtype=trt.nptype(dtype))
    return _run(plan, inputs)["logits"].astype(np.float64).reshape(-1)


@pytest.mark.parametrize(("precision", "same"), [("fp16", 1e-2), ("fp32", 1e-4)])
def test_decoder_reads_k_as_keys_and_v_as_values(tmp_path: Path, precision, same) -> None:
    """The decoder attends over the heads it is given: K only weighs the
    positions and V only supplies what is weighed. Without a reference model:
    values that are the same at every position come out whatever the keys are,
    and zero keys weigh every position alike, so the order of the values does
    not count. Catches K and V swapped in the decoder, keys or values that are
    not read, and weights that do not sum to 1. It does not pin which layer
    reads which heads or the attention's scale: the transcript checks do."""
    _require_gpu()
    whisper, config, weights = _tiny(tmp_path)
    plan = whisper.build_engine(config, weights, 8, precision=precision)
    rng = np.random.default_rng(11)
    keys = {f"cross_k_{layer}": 20.0 * rng.standard_normal(CROSS_SHAPE) for layer in range(LAYERS)}
    values = {f"cross_v_{layer}": rng.standard_normal(CROSS_SHAPE) for layer in range(LAYERS)}
    no_keys = {name: np.zeros(CROSS_SHAPE) for name in keys}
    one_value = {
        name: np.broadcast_to(heads[:, :, :1, :], CROSS_SHAPE).copy()
        for name, heads in values.items()
    }
    different = 0.1  # ten times the FP16 "same" bound

    def logits(k, v) -> np.ndarray:
        return _decoder_step(plan, {**k, **v})

    def distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.abs(left - right).max())

    # The weights of a layer sum to 1: one value at every position is that value.
    assert distance(logits(keys, one_value), logits(no_keys, one_value)) < same

    # Zero keys weigh every position alike: the order of the values does not count.
    uniform = logits(no_keys, values)
    order = rng.permutation(SRC)
    shuffled = {name: heads[:, :, order, :] for name, heads in values.items()}
    assert distance(logits(no_keys, shuffled), uniform) < same

    # The keys do weigh the positions, and every layer's values input is read.
    assert distance(logits(keys, values), uniform) > different
    for layer in range(LAYERS):
        name = f"cross_v_{layer}"
        assert distance(logits(no_keys, {**values, name: -values[name]}), uniform) > different


def test_tensor_parallel_encoder_keeps_the_raw_output(tmp_path: Path) -> None:
    _require_gpu()
    whisper, config, weights = _tiny(tmp_path)
    encoder = _io(
        whisper.build_vision_engine(
            str(tmp_path), config, weights, precision="fp16", project_cross_kv=False
        )
    )
    assert encoder["encoder_output"] == (trt.TensorIOMode.OUTPUT, (SRC, D), trt.float32)
    assert model.ENCODER_PADDED_ROWS not in encoder
    assert not any(name.startswith("cross_") for name in encoder)


def test_fp8_build_quantizes_the_cross_kv_projections_in_the_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    """An FP8 build quantizes every layer's K and V projection under the weight
    name its activation scale is keyed by, in the encoder engine that now runs
    them, and the decoder engine no longer builds them."""
    _require_gpu()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: _all_scales(cfg))
    whisper, config, weights = _tiny(tmp_path)
    ctx = calibrate_whisper(tmp_path, graph_ops, "fp8", config)

    quantized: list[str] = []
    quantized_matmul = WhisperQuantContext.maybe_quantized_matmul

    def record(
        self, network, lhs, lhs_width, rhs_width, rhs_weights, weight_name, dtype=np.float32
    ):
        if self.should_quantize(weight_name):
            quantized.append(weight_name)
        return quantized_matmul(
            self, network, lhs, lhs_width, rhs_width, rhs_weights, weight_name, dtype=dtype
        )

    # The context is a frozen dataclass: wrap the method on the class.
    monkeypatch.setattr(WhisperQuantContext, "maybe_quantized_matmul", record)
    cross_kv = {f"layer.{layer}.cross_w_{kind}" for layer in range(LAYERS) for kind in "kv"}
    cross_query = {f"layer.{layer}.cross_w_q" for layer in range(LAYERS)}

    encoder = _io(
        whisper.build_vision_engine(
            str(tmp_path), config, weights, precision="fp16", quant_ctx=ctx
        )
    )
    assert cross_kv <= set(quantized)
    assert encoder["cross_k_0"] == (trt.TensorIOMode.OUTPUT, CROSS_SHAPE, trt.float16)

    quantized.clear()
    whisper.build_engine(config, weights, 8, precision="fp16", quant_ctx=ctx)
    assert not cross_kv & set(quantized)
    assert cross_query <= set(quantized)


@pytest.mark.parametrize(("tensor_parallel_size", "projected"), [(1, True), (2, False)])
def test_build_projects_cross_kv_for_single_gpu_bundles_only(
    tmp_path: Path, monkeypatch, tensor_parallel_size: int, projected: bool
) -> None:
    """build() asks the encoder for projected K/V on a single-GPU bundle and for
    the raw encoder states on a tensor-parallel one, whose rank decoders project
    their own head shards. No engine is built here."""
    from tensorrt_model_connect import BuildRequest

    _write_checkpoint(tmp_path)
    encoder_builds = []

    def build_vision_engine(self, *args, **kwargs):
        encoder_builds.append(kwargs)
        return b"encoder"

    monkeypatch.setattr(model._WhisperModel, "load_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(model._WhisperModel, "build_engine", lambda *args, **kwargs: b"decoder")
    monkeypatch.setattr(model._WhisperModel, "build_vision_engine", build_vision_engine)
    monkeypatch.setattr(
        model._WhisperModel,
        "build_extra_engines",
        lambda *args, **kwargs: {"mel_filterbank": b"mel"},
    )
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda path: {})
    sections = {}
    writer = SimpleNamespace(
        set_header=lambda **value: None,
        add_bytes=lambda name, value: sections.__setitem__(name, value),
        add_json=lambda name, value: sections.__setitem__(name, value),
    )

    model.build(
        BuildRequest(
            model_dir=tmp_path,
            output_path=tmp_path / "whisper.bundle",
            family="whisper",
            task="transcription",
            precision="fp16",
            tensor_parallel_size=tensor_parallel_size,
        ),
        writer,
    )

    assert [build["project_cross_kv"] for build in encoder_builds] == [projected]
    decoders = {name for name in sections if name.startswith("engine.")}
    assert decoders == (
        {"engine.plan"}
        if tensor_parallel_size == 1
        else {f"engine.rank{rank}.plan" for rank in range(tensor_parallel_size)}
    )
