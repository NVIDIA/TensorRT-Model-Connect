# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the built family plans against independent reference tensors."""

import json
import struct

import numpy as np


def section(bundle, name):
    with bundle.open("rb") as stream:
        if stream.read(8) != b"BUNDLE\x01\x00":
            raise ValueError("Phonon probe received an invalid bundle header")
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        info = header["sections"][name]
        stream.seek(16 + size + info["offset"])
        return stream.read(info["length"])


def execute(plan, inputs):
    import tensorrt as trt
    from cuda.bindings import runtime as cuda

    def checked(result):
        if int(result[0]) != 0:
            raise RuntimeError(f"CUDA operation failed: {result[0]}")
        return result[1] if len(result) > 1 else None

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Phonon probe could not deserialize the engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Phonon probe could not create an execution context")
    for name, value in inputs.items():
        if not context.set_input_shape(name, value.shape):
            raise RuntimeError(f"Phonon probe rejected input shape for {name}")
    buffers, outputs, arrays = [], {}, {}
    try:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if name in inputs:
                value = np.ascontiguousarray(inputs[name])
            else:
                value = np.empty(
                    context.get_tensor_shape(name), dtype=trt.nptype(engine.get_tensor_dtype(name))
                )
                outputs[name] = value
            arrays[name] = value
            pointer = checked(cuda.cudaMalloc(value.nbytes))
            buffers.append(pointer)
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Phonon probe could not bind tensor {name}")
            if name in inputs:
                checked(
                    cuda.cudaMemcpy(
                        pointer,
                        value.ctypes.data,
                        value.nbytes,
                        cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                )
        if not context.execute_async_v3(0):
            raise RuntimeError("Phonon probe engine execution failed")
        checked(cuda.cudaDeviceSynchronize())
        for name, value in outputs.items():
            checked(
                cuda.cudaMemcpy(
                    value.ctypes.data,
                    context.get_tensor_address(name),
                    value.nbytes,
                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
        return outputs
    finally:
        for pointer in buffers:
            checked(cuda.cudaFree(pointer))
