# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary's cross-attention K/V projected once per request by the encoder engine.

A single-GPU bundle's encoder outputs every decoder layer's cross-attention K
and V, split into heads, and its decoder reads them in every step instead of
projecting the encoder output again. Tensor-parallel builds keep the raw
encoder output, which each rank's decoder shards and projects itself.
"""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for Canary engine builds")

from .. import model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from .test_quantization import (  # noqa: E402
    DEC_LAYERS,
    ENC_LAYERS,
    ES,
    HEADS,
    MEL,
    ML,
    D,
    _tiny_weights,
)

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _require_gpu() -> None:
    try:
        trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(0)
    except Exception as error:  # noqa: BLE001 - any builder failure means no usable GPU
        pytest.skip(f"engine build needs a GPU ({error})")


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
            context.set_input_shape(name, array.shape)
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


def _config(fp32_layers=()) -> ModelConfig:
    return ModelConfig(model_type="canary", raw={"_fp32_layers": tuple(fp32_layers)})


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_single_gpu_encoder_outputs_every_layers_cross_kv_heads(precision) -> None:
    _require_gpu()
    weights = _tiny_weights()
    canary = model._CanaryModel()
    encoder = _io(
        canary.build_vision_engine(
            "", _config(), weights, precision=precision, project_cross_kv=True
        )
    )
    decoder = _io(canary.build_engine(_config(), weights, 8, precision=precision))
    work = trt.float16 if precision == "fp16" else trt.float32

    assert "encoder_output" not in encoder
    for layer in range(DEC_LAYERS):
        for kind in ("cross_k", "cross_v"):
            name = f"{kind}_{layer}"
            mode, shape, dtype = encoder[name]
            assert mode == trt.TensorIOMode.OUTPUT
            assert shape == (-1, HEADS, ES, D // HEADS)
            assert dtype == work
            assert decoder[name] == (trt.TensorIOMode.INPUT, shape, dtype)


def test_projected_cross_kv_equals_the_projection_of_the_raw_encoder_output() -> None:
    """Each cross_k_<i>/cross_v_<i> is that layer's K/V projection (weight and
    bias) of the encoder output, split into [B, heads, S, head_dim]: what the
    decoder used to compute in every step. Catches a K/V swap, a wrong bias or
    a missing head transpose, which the I/O contract alone would not."""
    _require_gpu()
    weights = _tiny_weights()
    canary = model._CanaryModel()
    rng = np.random.default_rng(3)
    inputs = {
        "mel_features": rng.standard_normal((2, MEL, ML)).astype(np.float32),
        "encoder_mask": np.zeros((2, 1, 1, ES), dtype=np.float32),
    }
    raw = _run(
        canary.build_vision_engine(
            "", _config(), weights, precision="fp32", project_cross_kv=False
        ),
        inputs,
    )["encoder_output"].astype(np.float64)
    projected = _run(
        canary.build_vision_engine("", _config(), weights, precision="fp32", project_cross_kv=True),
        inputs,
    )

    def heads(rows: np.ndarray) -> np.ndarray:
        return rows.reshape(2, ES, HEADS, D // HEADS).transpose(0, 2, 1, 3)

    for layer in range(DEC_LAYERS):
        for kind, weight, bias in (("cross_k", "xw_k", "xb_k"), ("cross_v", "xw_v", "xb_v")):
            expected = heads(
                raw @ weights[f"layer.{layer}.{weight}"].astype(np.float64)
                + weights[f"layer.{layer}.{bias}"].astype(np.float64)
            )
            np.testing.assert_allclose(projected[f"{kind}_{layer}"], expected, rtol=1e-3, atol=1e-3)


def test_fp32_decoder_layers_receive_fp32_cross_kv_heads() -> None:
    _require_gpu()
    weights = _tiny_weights()
    canary = model._CanaryModel()
    # Global layer index ENC_LAYERS is decoder layer 0.
    config = _config(fp32_layers=(ENC_LAYERS,))
    encoder = _io(
        canary.build_vision_engine("", config, weights, precision="fp16", project_cross_kv=True)
    )
    decoder = _io(canary.build_engine(config, weights, 8, precision="fp16"))

    for kind in ("cross_k", "cross_v"):
        assert encoder[f"{kind}_0"][2] == trt.float32
        assert decoder[f"{kind}_0"][2] == trt.float32
        assert encoder[f"{kind}_1"][2] == trt.float16
        assert decoder[f"{kind}_1"][2] == trt.float16


def test_tensor_parallel_encoder_keeps_the_raw_output() -> None:
    _require_gpu()
    weights = _tiny_weights()
    encoder = _io(
        model._CanaryModel().build_vision_engine(
            "", _config(), weights, precision="fp16", project_cross_kv=False
        )
    )
    assert encoder["encoder_output"][1] == (-1, ES, D)
    assert not any(name.startswith("cross_") for name in encoder)
