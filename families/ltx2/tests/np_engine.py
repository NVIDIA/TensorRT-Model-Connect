# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-free TensorRT plan runner (cuda-python + NumPy) for multi-rank tests.

Multi-rank tests keep each rank light: no torch, only the plan and the NCCL communicator.
Waits poll ``cudaStreamQuery`` with a deadline and abort the NCCL communicator instead of
blocking forever, so a hung collective never leaves kernels running on the GPUs (killing
a rank with NCCL kernels in flight can leave the devices busy until a reset).
"""

from __future__ import annotations

import time

import ml_dtypes
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as rt

from families.ltx2.graph import make_logger

_NP = {trt.float32: np.float32, trt.float16: np.float16, trt.bfloat16: ml_dtypes.bfloat16, trt.int32: np.int32,
       trt.bool: np.bool_}


def ck(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA error {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


class NpEngine:
    def __init__(self, plan: bytes, communicator=None, on_timeout=None):
        self.runtime = trt.Runtime(make_logger())
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("failed to deserialize plan")
        self.context = self.engine.create_execution_context()
        if communicator is not None and not self.context.set_communicator(communicator):
            raise RuntimeError("set_communicator failed")
        self.stream = int(ck(rt.cudaStreamCreate()))
        self.on_timeout = on_timeout
        self.buffers: dict[str, tuple[int, tuple, object]] = {}
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        for name in self.inputs:  # run-time dimensions: allocate the profile maximum
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 in shape:
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2])
                self.context.set_input_shape(name, shape)
            self._allocate(name, shape)
        for name in names:
            if name not in self.inputs:
                self._allocate(name, tuple(self.context.get_tensor_shape(name)))

    def _allocate(self, name: str, shape: tuple) -> None:
        dtype = _NP[self.engine.get_tensor_dtype(name)]
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        ptr = int(ck(rt.cudaMalloc(max(nbytes, 1))))
        self.buffers[name] = (ptr, shape, dtype)
        self.context.set_tensor_address(name, ptr)

    def __call__(self, inputs: dict[str, np.ndarray], *, timeout_s: float = 120.0) -> dict[str, np.ndarray]:
        outs = {}
        for name in self.inputs:
            ptr, shape, dtype = self.buffers[name]
            a = np.ascontiguousarray(np.asarray(inputs[name]).astype(dtype))
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                if a.size > int(np.prod(shape)):
                    raise ValueError(f"{name}: shape {a.shape} exceeds the profile maximum {shape}")
                self.context.set_input_shape(name, a.shape)
            elif a.shape != shape:
                raise ValueError(f"{name}: shape {a.shape} != engine {shape}")
            ck(rt.cudaMemcpy(ptr, a.ctypes.data, a.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")
        deadline = time.monotonic() + timeout_s
        while int(rt.cudaStreamQuery(self.stream)[0]) != 0:
            if time.monotonic() > deadline:
                if self.on_timeout is not None:
                    self.on_timeout()
                raise TimeoutError(f"engine did not finish within {timeout_s} s")
            time.sleep(0.002)
        for name, (ptr, _, dtype) in self.buffers.items():
            if name not in self.inputs:
                a = np.empty(tuple(self.context.get_tensor_shape(name)), dtype=dtype)
                ck(rt.cudaMemcpy(a.ctypes.data, ptr, a.nbytes, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost))
                outs[name] = a.astype(np.float32) if dtype is not np.int32 else a
        return outs


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
