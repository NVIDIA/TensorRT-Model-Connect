# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned VL debug runner used by this model family.

This file intentionally duplicates the Python VL debug path so changes to one
family's runner do not couple sibling model families.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np
from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt() if trt_compat.is_available() else None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - TRT-free test envs
        cudart = None  # type: ignore[assignment]


def _check_cuda(status):
    if cudart is None:
        raise RuntimeError("cuda-python is required for VL debug runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


def _trt_nptype_safe(dtype: trt.DataType):
    try:
        return trt.nptype(dtype)
    except TypeError:
        if dtype == trt.bfloat16:
            return np.uint16
        raise


def _trt_itemsize(dtype: trt.DataType) -> int:
    return np.dtype(_trt_nptype_safe(dtype)).itemsize


def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for VL debug runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for VL debug runner execution")


def _profile_min_shape(engine: Any, name: str, profile_index: int) -> tuple[int, ...]:
    """Resolve a concrete single-step shape for a possibly dynamic tensor."""
    declared = tuple(int(dim) for dim in engine.get_tensor_shape(name))
    if all(dim >= 0 for dim in declared):
        return declared
    profile_shapes = engine.get_tensor_profile_shape(name, profile_index)
    if not profile_shapes:
        raise RuntimeError(f"Missing optimization profile shape for {name!r}")
    resolved = tuple(int(dim) for dim in profile_shapes[0])
    if any(dim < 0 for dim in resolved):
        raise RuntimeError(f"Invalid optimization profile shape for {name!r}: {resolved}")
    return resolved


class TrtRunner:
    """Device-resident TRT inference runner for debugging and diff testing.

    Keeps KV cache on-device. Only transfers token_id/position_id/mask
    (H2D, ~1 KB) and logits + debug outputs (D2H) per step. Cache updates
    are D2D memcpy. Matches the C++ DeviceKvCache behavior exactly.
    """

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_layers: int,
        attention_size: int | None = None,
        distributed_communicator: object | None = None,
    ):
        _require_trt_runtime()
        self.max_cache_length = max_cache_length
        self.num_layers = num_layers
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

        # Dual-profile engines (built by build_dual_profile_decoder_engine)
        # carry one prefill profile (profile 0, Sq dynamic) followed by one
        # decode profile (profile 1, Sq=1). For per-step decode runs the
        # debug_runner must select the decode profile and call
        # set_input_shape on every dynamic input before each execute. We
        # detect dual-profile via num_optimization_profiles > 1 and resolve
        # every dynamic input from the selected profile's minimum shape.
        self._dynamic_input_shapes: dict[str, tuple[int, ...]] = {}
        self._is_dual_profile = self.engine.num_optimization_profiles > 1
        self._profile_index = 1 if self._is_dual_profile else 0
        for index in range(self.engine.num_io_tensors):
            input_name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(input_name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(self.engine.get_tensor_shape(input_name))
            if any(dim < 0 for dim in shape):
                self._dynamic_input_shapes[input_name] = _profile_min_shape(
                    self.engine, input_name, self._profile_index)
        if self._is_dual_profile:
            # Profile 1 = decode (Sq=1). step() always runs single-token, so
            # we lock the context to that profile once and never switch.
            err, decode_stream = cudart.cudaStreamCreate()
            _check_cuda(err)
            self.context.set_optimization_profile_async(1, decode_stream)
            cudart.cudaStreamSynchronize(decode_stream)
            cudart.cudaStreamDestroy(decode_stream)

        # Auto-detect attention_size from cache_k_0 shape
        if attention_size is None:
            cache_shape = tuple(self.engine.get_tensor_shape("cache_k_0"))
            attention_size = cache_shape[1]  # (max_cache_length, attention_size)
        self.attention_size = attention_size

        # Detect cache element size from engine dtype (fp16=2, fp32=4)
        cache_dtype = self.engine.get_tensor_dtype("cache_k_0")
        self._cache_elem_bytes = _trt_itemsize(cache_dtype)

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        self.cache_length = 0
        attention_window = max_cache_length + 1
        row_bytes = self.attention_size * self._cache_elem_bytes
        self._row_bytes = row_bytes

        # Discover IO tensor metadata and identify debug/extra outputs
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
                # Debug outputs: anything that's not logits/present_k/present_v
                if (name != "logits"
                        and not name.startswith("present_k_")
                        and not name.startswith("present_v_")):
                    self._debug_output_names.append(name)

        # --- Persistent device cache buffers (not copied per step) ---
        cache_bytes = max_cache_length * row_bytes
        self._d_cache_k: list[int] = []
        self._d_cache_v: list[int] = []
        for _ in range(num_layers):
            err, dk = cudart.cudaMalloc(cache_bytes)
            _check_cuda(err)
            self._d_cache_k.append(dk)
            err, dv = cudart.cudaMalloc(cache_bytes)
            _check_cuda(err)
            self._d_cache_v.append(dv)

        # --- Device buffers for present_k/v outputs (single-row each) ---
        self._d_present_k: list[int] = []
        self._d_present_v: list[int] = []
        for _ in range(num_layers):
            err, pk = cudart.cudaMalloc(row_bytes)
            _check_cuda(err)
            self._d_present_k.append(pk)
            err, pv = cudart.cudaMalloc(row_bytes)
            _check_cuda(err)
            self._d_present_v.append(pv)

        # --- Small I/O: device + host buffers ---
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        self._h_position_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)
        err, self._d_position_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        # Qwen3-VL decoders require rank-2 multimodal rotary positions even
        # for a single text/decode token. Keep all three axes equal for this
        # text-only debug step, matching the production decode path.
        self._h_mrope_position_ids: np.ndarray | None = None
        self._d_mrope_position_ids = 0
        if any(
            self.engine.get_tensor_name(index) == "mrope_position_ids"
            for index in range(self.engine.num_io_tensors)
        ):
            mrope_shape = _profile_min_shape(
                self.engine, "mrope_position_ids", self._profile_index)
            self._h_mrope_position_ids = np.zeros(
                mrope_shape, dtype=np.int32)
            err, self._d_mrope_position_ids = cudart.cudaMalloc(
                self._h_mrope_position_ids.nbytes)
            _check_cuda(err)

        # attention_mask
        self._h_mask = np.zeros((1, attention_window), dtype=np.float32)
        err, self._d_mask = cudart.cudaMalloc(attention_window * 4)
        _check_cuda(err)

        # logits
        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # VL embed input support
        self._has_embed_input = False
        self._d_input_embed = 0
        self._d_use_input_embed = 0
        self._h_input_embed: np.ndarray | None = None
        self._h_use_input_embed: np.ndarray | None = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name == "input_embed":
                self._has_embed_input = True
                embed_shape = _profile_min_shape(
                    self.engine, name, self._profile_index)
                embed_bytes = int(np.prod(embed_shape)) * 4
                self._h_input_embed = np.zeros(embed_shape, dtype=np.float32)
                err, self._d_input_embed = cudart.cudaMalloc(embed_bytes)
                _check_cuda(err)
            elif name == "use_input_embed":
                selector_shape = _profile_min_shape(
                    self.engine, name, self._profile_index)
                self._h_use_input_embed = np.zeros(selector_shape, dtype=np.float32)
                err, self._d_use_input_embed = cudart.cudaMalloc(
                    self._h_use_input_embed.nbytes)
                _check_cuda(err)

        # DeepStack inputs (auto-detected from engine bindings)
        self._deepstack_names: list[str] = []
        self._d_deepstack: dict[str, int] = {}
        self._h_deepstack: dict[str, np.ndarray] = {}
        self._d_deepstack_active = 0
        self._h_deepstack_active: np.ndarray | None = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name.startswith("deepstack_embed_"):
                self._deepstack_names.append(name)
                shape = _profile_min_shape(
                    self.engine, name, self._profile_index)
                nbytes = int(np.prod(shape)) * 4
                self._h_deepstack[name] = np.zeros(shape, dtype=np.float32)
                err, d_ptr = cudart.cudaMalloc(nbytes)
                _check_cuda(err)
                self._d_deepstack[name] = d_ptr
            elif name == "deepstack_active":
                active_shape = _profile_min_shape(
                    self.engine, name, self._profile_index)
                self._h_deepstack_active = np.zeros(active_shape, dtype=np.float32)
                err, self._d_deepstack_active = cudart.cudaMalloc(
                    self._h_deepstack_active.nbytes)
                _check_cuda(err)

        # Debug output device/host buffers
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

        # Zero-init device cache
        for i in range(num_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    @property
    def has_embed_input(self) -> bool:
        """True if the engine has input_embed and use_input_embed inputs."""
        return self._has_embed_input

    def step(
        self,
        token_id: int,
        input_embed: np.ndarray | None = None,
        use_input_embed: float = 0.0,
        deepstack_embeds: list[np.ndarray] | None = None,
        deepstack_active: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Run one decode step (manages position and cache internally).

        Args:
            token_id: Input token ID.
            input_embed: Optional pre-computed embedding [1, hidden] for VL prefill.
            use_input_embed: 0.0 = use token_id lookup, 1.0 = use input_embed.
            deepstack_embeds: Optional per-level DeepStack embeddings for VL prefill.
            deepstack_active: 0.0 = inactive, 1.0 = inject DeepStack.

        Returns:
            Dict with 'logits' and any debug outputs (e.g. 'debug_hidden_0').
        """
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        # Build attention mask (matches C++ build_attention_mask exactly)
        position_id = min(self.cache_length, self.max_cache_length)
        self._h_mask[:] = -1e9
        valid = min(self.cache_length, self.max_cache_length)
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        # Prepare small host buffers
        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        # H2D: small inputs only
        cudart.cudaMemcpyAsync(
            self._d_token_id, self._h_token_id.ctypes.data,
            4, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_position_id, self._h_position_id.ctypes.data,
            4, H2D, stream)
        if self._h_mrope_position_ids is not None:
            self._h_mrope_position_ids.fill(position_id)
            cudart.cudaMemcpyAsync(
                self._d_mrope_position_ids,
                self._h_mrope_position_ids.ctypes.data,
                self._h_mrope_position_ids.nbytes, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_mask, self._h_mask.ctypes.data,
            attention_window * 4, H2D, stream)

        # VL embed_input support
        if self._has_embed_input:
            if input_embed is not None and use_input_embed > 0.5:
                self._h_input_embed[:] = input_embed.astype(np.float32)
                self._h_use_input_embed[0] = use_input_embed
            else:
                self._h_input_embed[:] = 0.0
                self._h_use_input_embed[0] = 0.0
            cudart.cudaMemcpyAsync(
                self._d_input_embed, self._h_input_embed.ctypes.data,
                self._h_input_embed.nbytes, H2D, stream)
            cudart.cudaMemcpyAsync(
                self._d_use_input_embed, self._h_use_input_embed.ctypes.data,
                self._h_use_input_embed.nbytes, H2D, stream)

        # DeepStack H2D transfers
        if self._deepstack_names:
            for idx, ds_name in enumerate(self._deepstack_names):
                if (deepstack_embeds is not None and idx < len(deepstack_embeds)
                        and deepstack_active > 0.5):
                    self._h_deepstack[ds_name][:] = deepstack_embeds[idx].astype(np.float32)
                else:
                    self._h_deepstack[ds_name][:] = 0.0
                cudart.cudaMemcpyAsync(
                    self._d_deepstack[ds_name],
                    self._h_deepstack[ds_name].ctypes.data,
                    self._h_deepstack[ds_name].nbytes, H2D, stream)
            if self._h_deepstack_active is not None:
                self._h_deepstack_active[0] = deepstack_active
                cudart.cudaMemcpyAsync(
                    self._d_deepstack_active,
                    self._h_deepstack_active.ctypes.data,
                    self._h_deepstack_active.nbytes, H2D, stream)

        # Set tensor addresses
        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("position_id", self._d_position_id)
        if self._d_mrope_position_ids:
            self.context.set_tensor_address(
                "mrope_position_ids", self._d_mrope_position_ids)
        self.context.set_tensor_address("attention_mask", self._d_mask)
        self.context.set_tensor_address("logits", self._d_logits)

        if self._has_embed_input:
            self.context.set_tensor_address("input_embed", self._d_input_embed)
            self.context.set_tensor_address(
                "use_input_embed", self._d_use_input_embed)

        # DeepStack tensor binding (zeroed by default, set during VL prefill)
        for ds_name in self._deepstack_names:
            self.context.set_tensor_address(ds_name, self._d_deepstack[ds_name])
        if self._d_deepstack_active:
            self.context.set_tensor_address(
                "deepstack_active", self._d_deepstack_active)

        for i in range(self.num_layers):
            self.context.set_tensor_address(f"cache_k_{i}", self._d_cache_k[i])
            self.context.set_tensor_address(f"cache_v_{i}", self._d_cache_v[i])
            self.context.set_tensor_address(f"present_k_{i}", self._d_present_k[i])
            self.context.set_tensor_address(f"present_v_{i}", self._d_present_v[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        # Dynamic inputs use the selected profile's minimum shapes. Both the
        # prefill and decode profiles have Sq=1 minima, so this covers token,
        # mask, image embedding, selector, and DeepStack inputs uniformly.
        for name, shape in self._dynamic_input_shapes.items():
            self.context.set_input_shape(name, shape)

        # Execute
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT Qwen-VL debug decoder execution failed")

        # D2D cache update
        row_bytes = self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
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

        # Collect results
        results: dict[str, np.ndarray] = {
            "logits": self._h_logits.copy(),
        }
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        """Zero all device cache buffers and reset cache_length."""
        cache_bytes = self.max_cache_length * self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
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
        """Run autoregressive generation.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Number of tokens to generate after the prompt.

        Returns:
            List of per-step result dicts. Each contains 'logits' and debug
            outputs. The list includes both prefill steps (one per input token)
            and generation steps.
        """
        all_results = []

        # Prefill: process input tokens one by one
        for tid in input_ids:
            result = self.step(tid)
            all_results.append(result)

        # Generate: autoregressive decoding
        for _ in range(max_new_tokens):
            last_logits = all_results[-1]["logits"].flatten()
            next_token = int(np.argmax(last_logits))
            result = self.step(next_token)
            all_results.append(result)

        return all_results

    def __del__(self):
        if cudart is None:
            return
        bufs = []
        for name in ("_d_token_id", "_d_position_id", "_d_mrope_position_ids",
                     "_d_mask", "_d_logits", "_d_input_embed",
                     "_d_use_input_embed", "_d_deepstack_active"):
            d_ptr = getattr(self, name, 0)
            if d_ptr:
                bufs.append(d_ptr)
        for name in ("_d_cache_k", "_d_cache_v", "_d_present_k", "_d_present_v"):
            bufs.extend(getattr(self, name, []))
        bufs.extend(getattr(self, "_d_deepstack", {}).values())
        bufs.extend(getattr(self, "_d_debug", {}).values())
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine


_NCCL_UNIQUE_ID_BYTES = 128
_NCCL_SUCCESS = 0

def load_engine_from_bundle(
    bundle_path: str,
    section_name: str = "engine_plan",
) -> tuple[bytes, dict]:
    """Load engine plan bytes and metadata from a .bundle artifact.

    Returns:
        (engine_plan_bytes, header_dict)
    """

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
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

def load_vision_engine_from_bundle(bundle_path: str) -> tuple[bytes | None, dict]:
    """Load vision engine plan bytes from a .bundle artifact.

    Returns:
        (vision_engine_plan_bytes_or_None, header_dict)
    """

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        vision_meta = sections.get("vision_engine_plan")
        if vision_meta is None:
            return None, header
        f.seek(16 + header_len + vision_meta["offset"])
        vision_plan = f.read(vision_meta["size"])

    return vision_plan, header

def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named section's raw bytes from a .bundle artifact.

    Returns None if the section doesn't exist.
    """

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid .bundle artifact: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        meta = sections.get(section_name)
        if meta is None:
            return None
        f.seek(16 + header_len + meta["offset"])
        return f.read(meta["size"])


def load_config_from_bundle(bundle_path: str) -> dict:
    """Load and parse config.json from a .bundle artifact."""
    data = load_section_from_bundle(bundle_path, "config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


def load_preprocessor_config_from_bundle(bundle_path: str) -> dict:
    """Load and parse preprocessor_config.json from a .bundle artifact."""
    data = load_section_from_bundle(bundle_path, "preprocessor_config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


def runner_from_bundle(
    bundle_path: str | None = None,
    *,
    runtime_strategy: str | None = None,
    config: dict | None = None,
    header: dict | None = None,
    engine_plan: bytes | None = None,
    distributed_communicator: object | None = None,
) -> object | None:
    """Create the Qwen-VL text debug runner from parsed bundle data."""
    if runtime_strategy != "qwen_vl_vision_language":
        return None
    if config is None or header is None or engine_plan is None:
        if bundle_path is None:
            raise TypeError(
                "bundle_path is required when parsed Qwen-VL data is absent")
        config = load_config_from_bundle(bundle_path)
        engine_plan, header = load_engine_from_bundle(bundle_path)
    return TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=int(header["max_cache_length"]),
        num_layers=int(header.get(
            "num_layers", config.get("num_hidden_layers", 1))),
        distributed_communicator=distributed_communicator,
    )
