# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small TensorRT executor for exact family-local numerical diagnostics."""

import numpy as np
import ml_dtypes
import tensorrt as trt
from cuda.bindings import runtime as cuda


def check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA error: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run(plan, inputs):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("cannot deserialize Flint diagnostic engine")
    context = engine.create_execution_context()
    allocations = []
    arrays = {}
    outputs = {}
    stream = check(cuda.cudaStreamCreate())
    try:
        for name, value in inputs.items():
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                value = np.ascontiguousarray(value)
                arrays[name] = value
                context.set_input_shape(name, value.shape)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype = (
                    ml_dtypes.bfloat16
                    if engine.get_tensor_dtype(name) == trt.bfloat16
                    else trt.nptype(engine.get_tensor_dtype(name))
                )
                arrays[name] = np.empty(tuple(context.get_tensor_shape(name)), dtype=dtype)
                outputs[name] = arrays[name]
            value = arrays[name]
            pointer = check(cuda.cudaMalloc(value.nbytes))
            allocations.append(pointer)
            context.set_tensor_address(name, pointer)
            if name not in outputs:
                check(
                    cuda.cudaMemcpyAsync(
                        pointer,
                        value.ctypes.data,
                        value.nbytes,
                        cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )
        if not context.execute_async_v3(stream):
            raise RuntimeError("Flint TensorRT diagnostic execution failed")
        for name, value in outputs.items():
            check(
                cuda.cudaMemcpyAsync(
                    value.ctypes.data,
                    context.get_tensor_address(name),
                    value.nbytes,
                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
        check(cuda.cudaStreamSynchronize(stream))
        return outputs
    finally:
        for pointer in allocations:
            check(cuda.cudaFree(pointer))
        check(cuda.cudaStreamDestroy(stream))
