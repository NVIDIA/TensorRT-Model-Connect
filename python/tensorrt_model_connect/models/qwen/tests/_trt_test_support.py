# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned TensorRT graph execution support."""

from __future__ import annotations

import numpy as np
import pytest


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401

        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]

        status, count = cudart.cudaGetDeviceCount()
        return int(status) == 0 and int(count) > 0
    except (ImportError, RuntimeError):
        return False


def _gpu_trt_skipif(condition: bool, reason: str):
    def decorator(obj):
        obj = pytest.mark.skipif(condition, reason=reason)(obj)
        obj = pytest.mark.gpu(obj)
        obj = pytest.mark.trt(obj)
        return obj

    return decorator


requires_trt = _gpu_trt_skipif(
    not _trt_available(),
    "TensorRT + CUDA not available",
)


def _check_cuda(status) -> None:
    """Raise on CUDA error."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]

    if hasattr(cudart, "cudaError_t"):
        ok = cudart.cudaError_t.cudaSuccess
    else:
        ok = 0
    if status != ok:
        raise RuntimeError(f"CUDA error: {status}")


def run_trt_graph(
    build_fn,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build a TensorRT engine, feed inputs, and return outputs."""
    import tensorrt as trt

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.clear_flag(trt.BuilderFlag.TF32)

    trt_inputs = {}
    for name, array in inputs.items():
        if array.dtype == np.float32:
            data_type = trt.float32
        elif array.dtype == np.float16:
            data_type = trt.float16
        else:
            data_type = trt.int32
        trt_inputs[name] = network.add_input(name, data_type, tuple(array.shape))

    trt_outputs = build_fn(network, trt_inputs)
    for name, tensor in trt_outputs.items():
        tensor.name = name
        if tensor.dtype != trt.float32:
            raise TypeError(
                f"Expected TRT graph output {name!r} to be float32, got {tensor.dtype}"
            )
        network.mark_output(tensor)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    error, stream = cudart.cudaStreamCreate()
    _check_cuda(error)

    device_buffers = {}
    host_outputs = {}
    for index in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(tensor_name))
        mode = engine.get_tensor_mode(tensor_name)
        byte_count = (
            inputs[tensor_name].nbytes
            if mode == trt.TensorIOMode.INPUT
            else int(np.prod(shape)) * 4
        )
        error, pointer = cudart.cudaMallocAsync(byte_count, stream)
        _check_cuda(error)
        device_buffers[tensor_name] = pointer
        if mode == trt.TensorIOMode.INPUT:
            array = inputs[tensor_name]
            cudart.cudaMemcpyAsync(
                pointer,
                array.ctypes.data,
                byte_count,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
        else:
            host_outputs[tensor_name] = np.zeros(shape, dtype=np.float32)
        context.set_tensor_address(tensor_name, pointer)

    context.execute_async_v3(stream)

    for name, array in host_outputs.items():
        cudart.cudaMemcpyAsync(
            array.ctypes.data,
            device_buffers[name],
            array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )

    cudart.cudaStreamSynchronize(stream)
    for pointer in device_buffers.values():
        cudart.cudaFreeAsync(pointer, stream)
    cudart.cudaStreamDestroy(stream)

    return host_outputs
