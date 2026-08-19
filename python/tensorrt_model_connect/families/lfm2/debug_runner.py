# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned fixed-step debug runner for native LFM2 bundles."""

from __future__ import annotations

import json
import struct

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt() if trt_compat.is_available() else None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - TRT-free source jobs
        cudart = None  # type: ignore[assignment]


def _require_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for the LFM2 debug runner")
    if cudart is None:
        raise ImportError("cuda-python is required for the LFM2 debug runner")


def _check_cuda(status) -> None:
    if cudart is None:
        raise RuntimeError("cuda-python is required for debug execution")
    success = cudart.cudaError_t.cudaSuccess if hasattr(cudart, "cudaError_t") else 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


def _trt_numpy_dtype(dtype):
    try:
        return trt.nptype(dtype)
    except TypeError:
        if dtype == trt.bfloat16:
            return np.uint16
        raise


def _tensor_nbytes(engine, name: str) -> int:
    shape = tuple(int(value) for value in engine.get_tensor_shape(name))
    if any(value <= 0 for value in shape):
        raise ValueError(f"LFM2 debug tensor {name!r} must have a static shape")
    dtype = np.dtype(_trt_numpy_dtype(engine.get_tensor_dtype(name)))
    return int(np.prod(shape)) * dtype.itemsize


def load_engine_from_bundle(
    bundle_path: str,
    section_name: str = "engine_plan",
) -> tuple[bytes, dict]:
    """Return the requested TensorRT plan and parsed bundle header."""

    with open(bundle_path, "rb") as handle:
        if handle.read(8) != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
        section = header.get("sections", {}).get(section_name)
        if not isinstance(section, dict):
            raise KeyError(f"Bundle {bundle_path!r} has no section {section_name!r}")
        handle.seek(16 + header_length + int(section["offset"]))
        plan = handle.read(int(section["size"]))
    return plan, header


def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    with open(bundle_path, "rb") as handle:
        if handle.read(8) != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
        section = header.get("sections", {}).get(section_name)
        if not isinstance(section, dict):
            return None
        handle.seek(16 + header_length + int(section["offset"]))
        return handle.read(int(section["size"]))


def load_config_from_bundle(bundle_path: str) -> dict:
    data = load_section_from_bundle(bundle_path, "config.json")
    return json.loads(data.decode("utf-8")) if data is not None else {}


class Lfm2TrtRunner:
    """One-token native-KV runner with double-buffered short-conv state."""

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_conv_layers: int,
        num_attention_layers: int,
        distributed_communicator: object | None = None,
    ) -> None:
        _require_runtime()
        self._closed = True
        self.max_cache_length = int(max_cache_length)
        self.num_conv_layers = int(num_conv_layers)
        self.num_attention_layers = int(num_attention_layers)
        if self.max_cache_length <= 0:
            raise ValueError("LFM2 debug max_cache_length must be positive")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize LFM2 TensorRT engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create LFM2 TensorRT execution context")
        if distributed_communicator is not None:
            setter = getattr(self.context, "set_communicator", None)
            if setter is None or not setter(distributed_communicator):
                raise RuntimeError("Failed to set TensorRT distributed communicator")

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)
        self.cache_length = 0

        self._host_scalars = {
            "token_id": np.zeros((1,), dtype=np.int32),
            "position_id": np.zeros((1,), dtype=np.int32),
            "cache_write_indices": np.zeros((1,), dtype=np.int32),
            "key_value_lengths": np.zeros((1,), dtype=np.int32),
        }
        self._device_scalars: dict[str, int] = {}
        for name in self._host_scalars:
            err, pointer = cudart.cudaMalloc(4)
            _check_cuda(err)
            self._device_scalars[name] = pointer

        self._conv_bytes = (
            _tensor_nbytes(self.engine, "conv_state_0") if self.num_conv_layers else 0
        )
        self._conv_state: list[int] = []
        self._present_conv: list[int] = []
        for _ in range(self.num_conv_layers):
            for storage in (self._conv_state, self._present_conv):
                err, pointer = cudart.cudaMalloc(self._conv_bytes)
                _check_cuda(err)
                storage.append(pointer)

        self._cache_bytes = (
            _tensor_nbytes(self.engine, "cache_k_0") if self.num_attention_layers else 0
        )
        self._cache_k: list[int] = []
        self._cache_v: list[int] = []
        for _ in range(self.num_attention_layers):
            for storage in (self._cache_k, self._cache_v):
                err, pointer = cudart.cudaMalloc(self._cache_bytes)
                _check_cuda(err)
                storage.append(pointer)

        logits_shape = tuple(int(value) for value in self.engine.get_tensor_shape("logits"))
        self._host_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._device_logits = cudart.cudaMalloc(self._host_logits.nbytes)
        _check_cuda(err)

        self._debug_names: list[str] = []
        self._host_debug: dict[str, np.ndarray] = {}
        self._device_debug: dict[str, int] = {}
        state_prefixes = ("present_k_", "present_v_", "present_conv_")
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            if name == "logits" or name.startswith(state_prefixes):
                continue
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            dtype = np.dtype(_trt_numpy_dtype(self.engine.get_tensor_dtype(name)))
            host = np.zeros(shape, dtype=dtype)
            err, pointer = cudart.cudaMalloc(host.nbytes)
            _check_cuda(err)
            self._debug_names.append(name)
            self._host_debug[name] = host
            self._device_debug[name] = pointer

        self.reset()
        self._closed = False

    def _copy_scalar(self, name: str, value: int) -> None:
        host = self._host_scalars[name]
        host[0] = int(value)
        cudart.cudaMemcpyAsync(
            self._device_scalars[name],
            host.ctypes.data,
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        if self.cache_length >= self.max_cache_length:
            raise RuntimeError("LFM2 sequence exceeds the fixed native KV capacity")
        self._copy_scalar("token_id", token_id)
        self._copy_scalar("position_id", self.cache_length)
        self._copy_scalar("cache_write_indices", self.cache_length)
        self._copy_scalar("key_value_lengths", self.cache_length + 1)

        for name, pointer in self._device_scalars.items():
            self.context.set_tensor_address(name, pointer)
        self.context.set_tensor_address("logits", self._device_logits)

        for index in range(self.num_conv_layers):
            self.context.set_tensor_address(f"conv_state_{index}", self._conv_state[index])
            self.context.set_tensor_address(f"present_conv_{index}", self._present_conv[index])
        for index in range(self.num_attention_layers):
            # IKVCacheUpdate requires cache input and present output to alias.
            self.context.set_tensor_address(f"cache_k_{index}", self._cache_k[index])
            self.context.set_tensor_address(f"present_k_{index}", self._cache_k[index])
            self.context.set_tensor_address(f"cache_v_{index}", self._cache_v[index])
            self.context.set_tensor_address(f"present_v_{index}", self._cache_v[index])
        for name in self._debug_names:
            self.context.set_tensor_address(name, self._device_debug[name])

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("LFM2 TensorRT execution failed")

        for index in range(self.num_conv_layers):
            cudart.cudaMemcpyAsync(
                self._conv_state[index],
                self._present_conv[index],
                self._conv_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self.stream,
            )
        cudart.cudaMemcpyAsync(
            self._host_logits.ctypes.data,
            self._device_logits,
            self._host_logits.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        for name in self._debug_names:
            host = self._host_debug[name]
            cudart.cudaMemcpyAsync(
                host.ctypes.data,
                self._device_debug[name],
                host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        _check_cuda(cudart.cudaStreamSynchronize(self.stream)[0])
        self.cache_length += 1

        outputs = {"logits": self._host_logits.copy()}
        outputs.update({name: self._host_debug[name].copy() for name in self._debug_names})
        return outputs

    def reset(self) -> None:
        self.cache_length = 0
        for pointer in (*self._conv_state, *self._present_conv):
            _check_cuda(
                cudart.cudaMemsetAsync(
                    pointer,
                    0,
                    self._conv_bytes,
                    self.stream,
                )[0]
            )
        for pointer in (*self._cache_k, *self._cache_v):
            _check_cuda(
                cudart.cudaMemsetAsync(
                    pointer,
                    0,
                    self._cache_bytes,
                    self.stream,
                )[0]
            )
        _check_cuda(cudart.cudaStreamSynchronize(self.stream)[0])

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        if not input_ids:
            return []
        outputs = [self.step(token_id) for token_id in input_ids]
        for _ in range(max_new_tokens):
            token_id = int(np.argmax(outputs[-1]["logits"].reshape(-1)))
            outputs.append(self.step(token_id))
        return outputs

    def close(self) -> None:
        if cudart is None or getattr(self, "_closed", True):
            return
        self._closed = True
        pointers = list(self._device_scalars.values())
        pointers.extend(self._conv_state)
        pointers.extend(self._present_conv)
        pointers.extend(self._cache_k)
        pointers.extend(self._cache_v)
        pointers.append(self._device_logits)
        pointers.extend(self._device_debug.values())
        for pointer in pointers:
            cudart.cudaFree(pointer)
        self._device_scalars.clear()
        self._conv_state.clear()
        self._present_conv.clear()
        self._cache_k.clear()
        self._cache_v.clear()
        self._device_debug.clear()
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
            del self.stream

    def __del__(self):
        self.close()


def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> Lfm2TrtRunner:
    del bundle_path
    expected = "lfm2_hybrid_conv_attention"
    if runtime_strategy != expected:
        raise ValueError(
            f"LFM2 debug runner requires runtime_strategy={expected!r}, got {runtime_strategy!r}"
        )
    return Lfm2TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=int(header["max_cache_length"]),
        num_conv_layers=int(config.get("num_conv_layers", 0)),
        num_attention_layers=int(config.get("num_attention_layers", 0)),
        distributed_communicator=distributed_communicator,
    )
