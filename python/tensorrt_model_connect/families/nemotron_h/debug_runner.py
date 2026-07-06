# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned debug runner implementation."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt() if trt_compat.is_available() else None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exercised in TRT-free test envs
        cudart = None  # type: ignore[assignment]

def _trt_nptype_safe(dtype: trt.DataType):
    """Resolve TRT dtype to a NumPy dtype, including BF16 fallback."""
    try:
        return trt.nptype(dtype)
    except TypeError:
        if dtype == trt.bfloat16:
            return np.uint16
        raise

def _trt_itemsize(dtype: trt.DataType) -> int:
    return np.dtype(_trt_nptype_safe(dtype)).itemsize



def _check_cuda(status):
    if cudart is None:
        raise RuntimeError("cuda-python is required for debug_runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for debug_runner execution")


def load_engine_from_bundle(
    bundle_path: str,
    section_name: str = "engine_plan",
) -> tuple[bytes, dict]:
    """Load this family's engine plan bytes and bundle metadata."""
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        engine_meta = sections.get(section_name)
        if engine_meta is None:
            raise KeyError(
                f"Bundle {bundle_path!r} does not contain section {section_name!r}")
        f.seek(16 + header_len + engine_meta["offset"])
        engine_plan = f.read(engine_meta["size"])

    return engine_plan, header

def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named raw section from this family's .trtfb bundle."""
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        meta = sections.get(section_name)
        if meta is None:
            return None
        f.seek(16 + header_len + meta["offset"])
        return f.read(meta["size"])

def load_config_from_bundle(bundle_path: str) -> dict:
    """Load and parse this family's config.json from a .trtfb bundle."""
    import json

    data = load_section_from_bundle(bundle_path, "config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


class HybridTrtRunner:
    """Device-resident hybrid TRT inference runner for models with mixed
    recurrent (DeltaNet/Mamba) + attention layers.

    Combines recurrent conv/SSM state management with KV cache + position
    tracking for hybrid recurrent-attention models.
    """

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_mamba_layers: int,
        num_attention_layers: int,
        distributed_communicator: object | None = None,
    ):
        _require_trt_runtime()
        self.max_cache_length = max_cache_length
        self.num_mamba_layers = num_mamba_layers
        self.num_attention_layers = num_attention_layers
        self._distributed_communicator = distributed_communicator

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()
        if distributed_communicator is not None:
            set_communicator = getattr(self.context, "set_communicator", None)
            if set_communicator is None:
                raise RuntimeError(
                    "TensorRT distributed execution requires TRT 11.0+ "
                    "IExecutionContext.set_communicator"
                )
            if not set_communicator(distributed_communicator):
                raise RuntimeError("Failed to set TRT distributed communicator")

        # Auto-detect state dimensions from engine tensor shapes
        if num_mamba_layers > 0:
            conv_shape = tuple(self.engine.get_tensor_shape("conv_state_0"))
            ssm_shape = tuple(self.engine.get_tensor_shape("ssm_state_0"))
        else:
            conv_shape = (0,)
            ssm_shape = (0,)

        if num_attention_layers > 0:
            cache_shape = tuple(self.engine.get_tensor_shape("cache_k_0"))
            self.attention_size = cache_shape[1]
            cache_dtype = self.engine.get_tensor_dtype("cache_k_0")
            self._cache_elem_bytes = _trt_itemsize(cache_dtype)
        else:
            self.attention_size = 0
            self._cache_elem_bytes = 4

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        self.cache_length = 0
        attention_window = max_cache_length + 1

        # Discover debug output tensor names
        self._output_names: list[str] = []
        self._output_shapes: dict[str, tuple] = {}
        self._debug_output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self._output_names.append(name)
                self._output_shapes[name] = shape
                if (name != "logits"
                        and not name.startswith("present_conv_")
                        and not name.startswith("present_ssm_")
                        and not name.startswith("present_k_")
                        and not name.startswith("present_v_")):
                    self._debug_output_names.append(name)

        # --- Mamba/DeltaNet state buffers ---
        self._conv_state_bytes = int(np.prod(conv_shape)) * 4 if num_mamba_layers > 0 else 0
        self._ssm_state_bytes = int(np.prod(ssm_shape)) * 4 if num_mamba_layers > 0 else 0

        self._d_conv_state: list[int] = []
        self._d_ssm_state: list[int] = []
        self._d_present_conv: list[int] = []
        self._d_present_ssm: list[int] = []
        for _ in range(num_mamba_layers):
            for lst, sz in [(self._d_conv_state, self._conv_state_bytes),
                            (self._d_ssm_state, self._ssm_state_bytes),
                            (self._d_present_conv, self._conv_state_bytes),
                            (self._d_present_ssm, self._ssm_state_bytes)]:
                err, ptr = cudart.cudaMalloc(sz)
                _check_cuda(err)
                lst.append(ptr)

        # --- KV cache buffers ---
        row_bytes = self.attention_size * self._cache_elem_bytes
        cache_bytes = max_cache_length * row_bytes

        self._d_cache_k: list[int] = []
        self._d_cache_v: list[int] = []
        self._d_present_k: list[int] = []
        self._d_present_v: list[int] = []
        for _ in range(num_attention_layers):
            for lst, sz in [(self._d_cache_k, cache_bytes),
                            (self._d_cache_v, cache_bytes),
                            (self._d_present_k, row_bytes),
                            (self._d_present_v, row_bytes)]:
                err, ptr = cudart.cudaMalloc(sz)
                _check_cuda(err)
                lst.append(ptr)

        # --- Small I/O ---
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        self._h_position_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)
        err, self._d_position_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        self._h_mask = np.zeros((1, attention_window), dtype=np.float32)
        err, self._d_mask = cudart.cudaMalloc(attention_window * 4)
        _check_cuda(err)

        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # Debug output buffers
        self._d_debug: dict[str, int] = {}
        self._h_debug: dict[str, np.ndarray] = {}
        for name in self._debug_output_names:
            shape = self._output_shapes[name]
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = _trt_nptype_safe(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize
            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._d_debug[name] = d_ptr
            self._h_debug[name] = np.zeros(shape, dtype=dtype_np)

        # Zero-init all state
        for i in range(num_mamba_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, self._conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, self._ssm_state_bytes, self.stream)[0])
        for i in range(num_attention_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one hybrid decode step."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        # Build attention mask (matches C++ build_attention_mask)
        position_id = min(self.cache_length, self.max_cache_length)
        self._h_mask[:] = -1e9
        valid = min(self.cache_length, self.max_cache_length)
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        # H2D: small inputs
        cudart.cudaMemcpyAsync(
            self._d_token_id, self._h_token_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_position_id, self._h_position_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_mask, self._h_mask.ctypes.data,
            attention_window * 4, H2D, stream)

        # Set tensor addresses
        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("position_id", self._d_position_id)
        self.context.set_tensor_address("attention_mask", self._d_mask)
        self.context.set_tensor_address("logits", self._d_logits)

        for i in range(self.num_mamba_layers):
            self.context.set_tensor_address(
                f"conv_state_{i}", self._d_conv_state[i])
            self.context.set_tensor_address(
                f"ssm_state_{i}", self._d_ssm_state[i])
            self.context.set_tensor_address(
                f"present_conv_{i}", self._d_present_conv[i])
            self.context.set_tensor_address(
                f"present_ssm_{i}", self._d_present_ssm[i])

        for i in range(self.num_attention_layers):
            self.context.set_tensor_address(
                f"cache_k_{i}", self._d_cache_k[i])
            self.context.set_tensor_address(
                f"cache_v_{i}", self._d_cache_v[i])
            self.context.set_tensor_address(
                f"present_k_{i}", self._d_present_k[i])
            self.context.set_tensor_address(
                f"present_v_{i}", self._d_present_v[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        # Execute
        self.context.execute_async_v3(stream)

        # D2D state update: conv/ssm (direct replacement)
        for i in range(self.num_mamba_layers):
            cudart.cudaMemcpyAsync(
                self._d_conv_state[i], self._d_present_conv[i],
                self._conv_state_bytes, D2D, stream)
            cudart.cudaMemcpyAsync(
                self._d_ssm_state[i], self._d_present_ssm[i],
                self._ssm_state_bytes, D2D, stream)

        # D2D cache update: KV (append or shift)
        row_bytes = self.attention_size * self._cache_elem_bytes
        for i in range(self.num_attention_layers):
            for cache_buf, present_buf in [
                (self._d_cache_k[i], self._d_present_k[i]),
                (self._d_cache_v[i], self._d_present_v[i]),
            ]:
                if self.cache_length < self.max_cache_length:
                    offset = self.cache_length * row_bytes
                    cudart.cudaMemcpyAsync(
                        cache_buf + offset, present_buf,
                        row_bytes, D2D, stream)
                else:
                    cudart.cudaMemcpyAsync(
                        cache_buf, cache_buf + row_bytes,
                        (self.max_cache_length - 1) * row_bytes,
                        D2D, stream)
                    offset = (self.max_cache_length - 1) * row_bytes
                    cudart.cudaMemcpyAsync(
                        cache_buf + offset, present_buf,
                        row_bytes, D2D, stream)

        # D2H: logits + debug outputs
        cudart.cudaMemcpyAsync(
            self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        for name in self._debug_output_names:
            h_buf = self._h_debug[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, self._d_debug[name],
                h_buf.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)

        results: dict[str, np.ndarray] = {"logits": self._h_logits.copy()}
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        """Zero all device state buffers and reset cache_length."""
        cache_bytes = self.max_cache_length * self.attention_size * self._cache_elem_bytes
        for i in range(self.num_mamba_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, self._conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, self._ssm_state_bytes, self.stream)[0])
        for i in range(self.num_attention_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)
        self.cache_length = 0

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        """Run autoregressive generation."""
        all_results = []
        for tid in input_ids:
            all_results.append(self.step(tid))
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(all_results[-1]["logits"].flatten()))
            all_results.append(self.step(next_token))
        return all_results

    def __del__(self):
        if cudart is None:
            return
        if not hasattr(self, "_d_token_id"):
            return
        bufs = [self._d_token_id, self._d_position_id, self._d_mask,
                self._d_logits]
        bufs.extend(self._d_conv_state)
        bufs.extend(self._d_ssm_state)
        bufs.extend(self._d_present_conv)
        bufs.extend(self._d_present_ssm)
        bufs.extend(self._d_cache_k)
        bufs.extend(self._d_cache_v)
        bufs.extend(self._d_present_k)
        bufs.extend(self._d_present_v)
        for d_ptr in self._d_debug.values():
            bufs.append(d_ptr)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine

def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> HybridTrtRunner:
    del runtime_strategy, bundle_path
    pattern = str(config.get("hybrid_override_pattern", ""))
    num_mamba_layers = int(config.get("num_mamba_layers", 0) or 0)
    num_attention_layers = int(config.get("num_attention_layers", 0) or 0)
    if pattern:
        num_mamba_layers = sum(char == "M" for char in pattern)
        num_attention_layers = sum(char == "*" for char in pattern)
    return HybridTrtRunner(
        engine_plan=engine_plan,
        max_cache_length=header["max_cache_length"],
        num_mamba_layers=num_mamba_layers,
        num_attention_layers=num_attention_layers,
        distributed_communicator=distributed_communicator,
    )
