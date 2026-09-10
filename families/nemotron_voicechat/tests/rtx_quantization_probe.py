# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and execute the family W8A8 GEMM on TensorRT-RTX with a CPU reference.

Run as a module from the repository root in an environment containing
TensorRT-RTX, cuda-python, and NumPy. This is an explicit GPU probe so ordinary
unit test discovery does not import the RTX backend into a TensorRT process.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import tensorrt_rtx as trt

sys.modules["tensorrt"] = trt

from families.nemotron_voicechat import graph_ops, quantization  # noqa: E402


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run() -> dict:
    from cuda.bindings import driver as cuda

    rng = np.random.default_rng(1218)
    inputs = rng.normal(0, 0.2, size=(4, 64)).astype(np.float32)
    inputs[0] = 0  # Verify the guarded absmax scale for silence.
    weights = rng.normal(0, 0.2, size=(64, 32)).astype(np.float32)
    scales = quantization.derive_weight_scale(weights)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("input", trt.float32, inputs.shape)
    output = quantization._wrap_int8_matmul(
        network, activation, weights, scales,
        lhs_width=64, rhs_width=32, graph_ops=graph_ops,
    )
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    plan = quantization.build_serialized_network(builder, network, config)
    if plan is None:
        raise RuntimeError("TensorRT-RTX rejected the family W8A8 graph")

    checked(cuda.cuInit(0))
    device = checked(cuda.cuDeviceGet(0))
    primary_context = checked(cuda.cuDevicePrimaryCtxRetain(device))
    checked(cuda.cuCtxSetCurrent(primary_context))
    allocations = []
    stream = None
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("TensorRT-RTX cannot deserialize the family W8A8 graph")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT-RTX cannot create a W8A8 execution context")
        actual = np.empty((4, 32), dtype=np.float32)
        for name, array in (("input", inputs), ("output", actual)):
            pointer = checked(cuda.cuMemAlloc(array.nbytes))
            allocations.append(pointer)
            if not context.set_tensor_address(name, int(pointer)):
                raise RuntimeError(f"Cannot bind RTX tensor {name}")
        checked(cuda.cuMemcpyHtoD(allocations[0], inputs.ctypes.data, inputs.nbytes))
        stream = checked(cuda.cuStreamCreate(0))
        if not context.execute_async_v3(int(stream)):
            raise RuntimeError("TensorRT-RTX W8A8 execution failed")
        checked(cuda.cuStreamSynchronize(stream))
        checked(cuda.cuMemcpyDtoH(actual.ctypes.data, allocations[1], actual.nbytes))

        packed, scales = quantization.quantize_int8_per_output_channel(
            weights, scales, lhs_width=64, rhs_width=32,
        )
        dynamic = np.maximum(np.max(np.abs(inputs), axis=1, keepdims=True),
                             np.float32(127 * np.finfo(np.float32).tiny)) / np.float32(127)
        quantized_inputs = np.clip(np.rint(inputs / dynamic), -128, 127).astype(np.int8)
        reference = ((quantized_inputs.astype(np.float32) @
                      (packed.astype(np.float32) * scales)) * dynamic)
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        return {"backend": "trt_rtx", "version": trt.__version__,
                "planBytes": len(bytes(plan)),
                "maxAbsoluteError": float(np.max(np.abs(actual - reference))),
                "silenceExactZero": bool(np.all(actual[0] == 0)), "passed": True}
    finally:
        # Release TRT objects while their CUDA context is still current.
        context = None
        engine = None
        runtime = None
        if stream is not None:
            checked(cuda.cuStreamDestroy(stream))
        for pointer in allocations:
            checked(cuda.cuMemFree(pointer))
        checked(cuda.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
