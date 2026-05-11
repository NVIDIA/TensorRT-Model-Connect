"""Pure-Python TRT inference runner for debugging and diff testing.

No C++ binary needed. Deserializes a TRT engine plan and runs
single-step autoregressive decoding with KV cache management.

Supports both standard decoder (KV cache) and Mamba/SSM (recurrent state).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from tensorrt_model_connect import trt_compat


from .triattention_runtime import TriAttentionRuntimeConfig, TriAttentionSelector

trt = trt_compat.get_trt() if trt_compat.is_available() else None

# cuda-python >= 13 uses cuda.bindings.runtime; older versions use cuda.cudart.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exercised in TRT-free test envs
        cudart = None  # type: ignore[assignment]


def _check_cuda(status):
    """Raise on CUDA error."""
    if cudart is None:
        raise RuntimeError("cuda-python is required for debug_runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


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


def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for debug_runner execution")


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
    ):
        _require_trt_runtime()
        self.max_cache_length = max_cache_length
        self.num_layers = num_layers

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

        # Dual-profile engines (built by build_dual_profile_decoder_engine)
        # carry one prefill profile (profile 0, Sq dynamic) followed by one
        # decode profile (profile 1, Sq=1). For per-step decode runs the
        # debug_runner must select the decode profile and call
        # set_input_shape on every dynamic input before each execute. We
        # detect dual-profile via num_optimization_profiles > 1 and the
        # presence of -1 dims on token_id / position_id / attention_mask.
        self._dynamic_inputs: list[str] = []
        self._is_dual_profile = self.engine.num_optimization_profiles > 1
        for input_name in ("token_id", "position_id", "attention_mask"):
            try:
                shape = tuple(self.engine.get_tensor_shape(input_name))
            except Exception:
                continue
            if any(d < 0 for d in shape):
                self._dynamic_inputs.append(input_name)
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
                embed_shape = tuple(self.engine.get_tensor_shape(name))
                embed_bytes = int(np.prod(embed_shape)) * 4
                self._h_input_embed = np.zeros(embed_shape, dtype=np.float32)
                err, self._d_input_embed = cudart.cudaMalloc(embed_bytes)
                _check_cuda(err)
            elif name == "use_input_embed":
                self._h_use_input_embed = np.zeros((1,), dtype=np.float32)
                err, self._d_use_input_embed = cudart.cudaMalloc(4)
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
                shape = tuple(self.engine.get_tensor_shape(name))
                nbytes = int(np.prod(shape)) * 4
                self._h_deepstack[name] = np.zeros(shape, dtype=np.float32)
                err, d_ptr = cudart.cudaMalloc(nbytes)
                _check_cuda(err)
                self._d_deepstack[name] = d_ptr
            elif name == "deepstack_active":
                self._h_deepstack_active = np.zeros((1,), dtype=np.float32)
                err, self._d_deepstack_active = cudart.cudaMalloc(4)
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
                4, H2D, stream)

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
                    4, H2D, stream)

        # Set tensor addresses
        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("position_id", self._d_position_id)
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

        # Dual-profile engines need explicit shapes for the dynamic
        # inputs every step. step() is single-token decode, so all three
        # shapes are fixed: Sq=1 and K = max_cache_length + 1.
        if self._dynamic_inputs:
            for name in self._dynamic_inputs:
                if name == "attention_mask":
                    self.context.set_input_shape(name, (1, attention_window))
                else:
                    self.context.set_input_shape(name, (1,))

        # Execute
        self.context.execute_async_v3(stream)

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
        if not hasattr(self, "_d_token_id"):
            return
        bufs = [self._d_token_id, self._d_position_id, self._d_mask,
                self._d_logits]
        bufs.extend(self._d_cache_k)
        bufs.extend(self._d_cache_v)
        bufs.extend(self._d_present_k)
        bufs.extend(self._d_present_v)
        if self._d_input_embed:
            bufs.append(self._d_input_embed)
        if self._d_use_input_embed:
            bufs.append(self._d_use_input_embed)
        for d_ptr in self._d_deepstack.values():
            bufs.append(d_ptr)
        if self._d_deepstack_active:
            bufs.append(self._d_deepstack_active)
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


class TriAttentionTrtRunner(TrtRunner):
    """Experimental TriAttention runner for dense shared-row KV caches.

    This keeps the base TensorRT engine unchanged and applies cache selection
    in the Python debug path. Unlike the upstream vLLM integration, this MVP
    uses a single shared token set across all layers/heads.
    """

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_layers: int,
        triattention_config: TriAttentionRuntimeConfig,
        triattention_stats_payload: dict[str, Any],
        attention_size: int | None = None,
    ):
        super().__init__(
            engine_plan=engine_plan,
            max_cache_length=max_cache_length,
            num_layers=num_layers,
            attention_size=attention_size,
        )
        if triattention_config.kv_budget < 1:
            raise ValueError("TriAttention kv_budget must be >= 1")
        if triattention_config.kv_budget > max_cache_length:
            raise ValueError(
                "TriAttention kv_budget cannot exceed the engine max_cache_length"
            )
        self.triattention_config = triattention_config
        self.triattention_selector = TriAttentionSelector(
            triattention_stats_payload,
            triattention_config,
        )
        self.absolute_position = 0
        self.cache_positions: list[int] = []
        self._tri_prompt_end_position = 0

    def _copy_layer_cache_to_host(self, device_ptr: int, rows: int) -> np.ndarray:
        host = np.zeros((rows, self.attention_size), dtype=self._cache_dtype)
        if rows <= 0:
            return host
        cudart.cudaMemcpyAsync(
            host.ctypes.data,
            device_ptr,
            rows * self._row_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)
        return host

    def _write_layer_cache_from_host(self, device_ptr: int, host: np.ndarray) -> None:
        if host.size == 0:
            return
        cudart.cudaMemcpyAsync(
            device_ptr,
            np.ascontiguousarray(host).ctypes.data,
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)

    def _compact_existing_cache(self) -> None:
        if self.cache_length < self.triattention_config.kv_budget:
            return

        sampled_layers = sorted({int(layer) for layer, _ in self.triattention_selector.sampled_heads})
        sampled_k_cache: dict[int, np.ndarray] = {}
        for layer in sampled_layers:
            if layer >= self.num_layers:
                continue
            sampled_k_cache[layer] = self._copy_layer_cache_to_host(
                self._d_cache_k[layer],
                self.cache_length,
            )

        keep_indices = self.triattention_selector.select_keep_indices(
            sampled_k_cache,
            cache_positions=self.cache_positions,
            next_position=self.absolute_position,
            prefix_length=self._tri_prompt_end_position,
        )
        keep_count = int(keep_indices.size)
        for i in range(self.num_layers):
            host_k = self._copy_layer_cache_to_host(self._d_cache_k[i], self.cache_length)
            host_v = self._copy_layer_cache_to_host(self._d_cache_v[i], self.cache_length)
            self._write_layer_cache_from_host(self._d_cache_k[i], host_k[keep_indices])
            self._write_layer_cache_from_host(self._d_cache_v[i], host_v[keep_indices])

        self.cache_positions = [self.cache_positions[int(idx)] for idx in keep_indices]
        self.cache_length = keep_count

    def step(
        self,
        token_id: int,
        input_embed: np.ndarray | None = None,
        use_input_embed: float = 0.0,
        deepstack_embeds: list[np.ndarray] | None = None,
        deepstack_active: float = 0.0,
    ) -> dict[str, np.ndarray]:
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        position_id = self.absolute_position
        self._h_mask[:] = -1e9
        valid = self.cache_length
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        cudart.cudaMemcpyAsync(
            self._d_token_id, self._h_token_id.ctypes.data,
            4, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_position_id, self._h_position_id.ctypes.data,
            4, H2D, stream)
        cudart.cudaMemcpyAsync(
            self._d_mask, self._h_mask.ctypes.data,
            attention_window * 4, H2D, stream)

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
                4, H2D, stream)

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
                    4, H2D, stream)

        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("position_id", self._d_position_id)
        self.context.set_tensor_address("attention_mask", self._d_mask)
        self.context.set_tensor_address("logits", self._d_logits)

        if self._has_embed_input:
            self.context.set_tensor_address("input_embed", self._d_input_embed)
            self.context.set_tensor_address(
                "use_input_embed", self._d_use_input_embed)

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

        self.context.execute_async_v3(stream)

        self._compact_existing_cache()

        for i in range(self.num_layers):
            for cache_buf, present_buf in [
                (self._d_cache_k[i], self._d_present_k[i]),
                (self._d_cache_v[i], self._d_present_v[i]),
            ]:
                offset = self.cache_length * self._row_bytes
                cudart.cudaMemcpyAsync(
                    cache_buf + offset,
                    present_buf,
                    self._row_bytes,
                    D2D,
                    stream,
                )

        cudart.cudaMemcpyAsync(
            self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        for name in self._debug_output_names:
            h_buf = self._h_debug[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, self._d_debug[name],
                h_buf.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)
        self.cache_positions.append(self.absolute_position)
        self.cache_length += 1
        self.absolute_position += 1

        results: dict[str, np.ndarray] = {
            "logits": self._h_logits.copy(),
        }
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        super().reset()
        self.absolute_position = 0
        self.cache_positions = []
        self._tri_prompt_end_position = 0

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        all_results = []
        self._tri_prompt_end_position = self.absolute_position + len(input_ids)

        for tid in input_ids:
            result = self.step(tid)
            all_results.append(result)

        for _ in range(max_new_tokens):
            last_logits = all_results[-1]["logits"].flatten()
            next_token = int(np.argmax(last_logits))
            result = self.step(next_token)
            all_results.append(result)

        return all_results


class MambaTrtRunner:
    """Device-resident Mamba/SSM TRT inference runner.

    Keeps conv_state and ssm_state on-device. Only transfers token_id
    (H2D, 4 B) and logits + debug outputs (D2H) per step. State updates
    are D2D memcpy. Matches the C++ MambaBackend behavior exactly.
    """

    def __init__(
        self,
        engine_plan: bytes,
        num_layers: int,
        d_inner: int | None = None,
        state_size: int | None = None,
        conv_kernel: int | None = None,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

        # Auto-detect state dimensions
        if d_inner is None or conv_kernel is None:
            conv_shape = tuple(self.engine.get_tensor_shape("conv_state_0"))
            d_inner = conv_shape[0]
            conv_kernel = conv_shape[1]
        if state_size is None:
            ssm_shape = tuple(self.engine.get_tensor_shape("ssm_state_0"))
            state_size = ssm_shape[1]

        self.d_inner = d_inner
        self.state_size = state_size
        self.conv_kernel = conv_kernel

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        conv_state_bytes = d_inner * conv_kernel * 4
        ssm_state_bytes = d_inner * state_size * 4

        # Discover debug/extra output tensor names
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
                        and not name.startswith("present_ssm_")):
                    self._debug_output_names.append(name)

        # --- Persistent device state buffers ---
        self._d_conv_state: list[int] = []
        self._d_ssm_state: list[int] = []
        for _ in range(num_layers):
            err, dc = cudart.cudaMalloc(conv_state_bytes)
            _check_cuda(err)
            self._d_conv_state.append(dc)
            err, ds = cudart.cudaMalloc(ssm_state_bytes)
            _check_cuda(err)
            self._d_ssm_state.append(ds)

        # --- Device buffers for present outputs ---
        self._d_present_conv: list[int] = []
        self._d_present_ssm: list[int] = []
        for _ in range(num_layers):
            err, pc = cudart.cudaMalloc(conv_state_bytes)
            _check_cuda(err)
            self._d_present_conv.append(pc)
            err, ps = cudart.cudaMalloc(ssm_state_bytes)
            _check_cuda(err)
            self._d_present_ssm.append(ps)

        # --- Small I/O ---
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
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

        # Zero-init device state
        for i in range(num_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, ssm_state_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one Mamba decode step.

        Args:
            token_id: Input token ID.

        Returns:
            Dict with 'logits' and any debug outputs.
        """
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream

        self._h_token_id[0] = token_id
        cudart.cudaMemcpyAsync(
            self._d_token_id, self._h_token_id.ctypes.data,
            4, H2D, stream)

        # Set tensor addresses
        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("logits", self._d_logits)

        conv_state_bytes = self.d_inner * self.conv_kernel * 4
        ssm_state_bytes = self.d_inner * self.state_size * 4

        for i in range(self.num_layers):
            self.context.set_tensor_address(
                f"conv_state_{i}", self._d_conv_state[i])
            self.context.set_tensor_address(
                f"ssm_state_{i}", self._d_ssm_state[i])
            self.context.set_tensor_address(
                f"present_conv_{i}", self._d_present_conv[i])
            self.context.set_tensor_address(
                f"present_ssm_{i}", self._d_present_ssm[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        # Execute
        self.context.execute_async_v3(stream)

        # D2D state update (direct replacement — Mamba state is overwritten)
        for i in range(self.num_layers):
            cudart.cudaMemcpyAsync(
                self._d_conv_state[i], self._d_present_conv[i],
                conv_state_bytes, D2D, stream)
            cudart.cudaMemcpyAsync(
                self._d_ssm_state[i], self._d_present_ssm[i],
                ssm_state_bytes, D2D, stream)

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

        # Collect results
        results: dict[str, np.ndarray] = {
            "logits": self._h_logits.copy(),
        }
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        """Zero all device state buffers."""
        conv_state_bytes = self.d_inner * self.conv_kernel * 4
        ssm_state_bytes = self.d_inner * self.state_size * 4
        for i in range(self.num_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, ssm_state_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        """Run autoregressive generation with Mamba.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Number of tokens to generate after the prompt.

        Returns:
            List of per-step result dicts.
        """
        all_results = []

        # Prefill: process input tokens one by one (Mamba is recurrent)
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
        if not hasattr(self, "_d_token_id"):
            return
        bufs = []
        for attr in ("_d_token_id", "_d_logits"):
            value = getattr(self, attr, None)
            if value is not None:
                bufs.append(value)
        for attr in ("_d_conv_state", "_d_ssm_state", "_d_present_conv", "_d_present_ssm"):
            bufs.extend(getattr(self, attr, []))
        for d_ptr in getattr(self, "_d_debug", {}).values():
            bufs.append(d_ptr)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine


class RwkvTrtRunner:
    """Device-resident RWKV TRT inference runner.

    Keeps 5 state tensors per layer on-device:
      attn_state, ff_state, num_state, den_state, max_state
    Only transfers token_id (H2D) and logits (D2H) per step.
    """

    def __init__(self, engine_plan: bytes, num_layers: int):
        _require_trt_runtime()
        self.num_layers = num_layers

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

        # Auto-detect hidden_size from attn_state_0 shape [1, hidden]
        state_shape = tuple(self.engine.get_tensor_shape("attn_state_0"))
        self.hidden_size = state_shape[1] if len(state_shape) == 2 else state_shape[0]
        state_bytes = self.hidden_size * 4

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Per-layer device state buffers (5 per layer)
        self._d_attn = []
        self._d_ff = []
        self._d_num = []
        self._d_den = []
        self._d_max = []
        self._d_p_attn = []
        self._d_p_ff = []
        self._d_p_num = []
        self._d_p_den = []
        self._d_p_max = []
        for _ in range(num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den,
                        self._d_max, self._d_p_attn, self._d_p_ff,
                        self._d_p_num, self._d_p_den, self._d_p_max]:
                err, ptr = cudart.cudaMalloc(state_bytes)
                _check_cuda(err)
                lst.append(ptr)

        self._h_token_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # Discover debug outputs
        self._debug_output_names = []
        self._output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self._output_shapes[name] = shape
                if (name != "logits"
                        and not name.startswith("present_attn_")
                        and not name.startswith("present_ff_")
                        and not name.startswith("present_num_")
                        and not name.startswith("present_den_")
                        and not name.startswith("present_max_")):
                    self._debug_output_names.append(name)

        self._d_debug = {}
        self._h_debug = {}
        for name in self._debug_output_names:
            shape = self._output_shapes[name]
            nbytes = int(np.prod(shape)) * 4
            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._d_debug[name] = d_ptr
            self._h_debug[name] = np.zeros(shape, dtype=np.float32)

        # Zero-init all states (max_state should be -1e38 for numerical stability)
        for i in range(num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den]:
                _check_cuda(cudart.cudaMemsetAsync(lst[i], 0, state_bytes, self.stream)[0])
            # max_state init: -1e38
            h_neg_inf = np.full(self.hidden_size, -1e38, dtype=np.float32)
            cudart.cudaMemcpyAsync(
                self._d_max[i], h_neg_inf.ctypes.data, state_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        state_bytes = self.hidden_size * 4

        self._h_token_id[0] = token_id
        cudart.cudaMemcpyAsync(self._d_token_id, self._h_token_id.ctypes.data, 4, H2D, stream)

        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("logits", self._d_logits)

        state_map = [
            ("attn_state", self._d_attn, "present_attn", self._d_p_attn),
            ("ff_state", self._d_ff, "present_ff", self._d_p_ff),
            ("num_state", self._d_num, "present_num", self._d_p_num),
            ("den_state", self._d_den, "present_den", self._d_p_den),
            ("max_state", self._d_max, "present_max", self._d_p_max),
        ]
        for i in range(self.num_layers):
            for in_stem, in_lst, out_stem, out_lst in state_map:
                self.context.set_tensor_address(f"{in_stem}_{i}", in_lst[i])
                self.context.set_tensor_address(f"{out_stem}_{i}", out_lst[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        self.context.execute_async_v3(stream)

        # D2D state update
        for i in range(self.num_layers):
            for _, in_lst, _, out_lst in state_map:
                cudart.cudaMemcpyAsync(in_lst[i], out_lst[i], state_bytes, D2D, stream)

        cudart.cudaMemcpyAsync(
            self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        for name in self._debug_output_names:
            h = self._h_debug[name]
            cudart.cudaMemcpyAsync(h.ctypes.data, self._d_debug[name], h.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)
        results = {"logits": self._h_logits.copy()}
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        state_bytes = self.hidden_size * 4
        for i in range(self.num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den]:
                _check_cuda(cudart.cudaMemsetAsync(lst[i], 0, state_bytes, self.stream)[0])
            h_neg_inf = np.full(self.hidden_size, -1e38, dtype=np.float32)
            cudart.cudaMemcpyAsync(
                self._d_max[i], h_neg_inf.ctypes.data, state_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[dict[str, np.ndarray]]:
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
        bufs = [self._d_token_id, self._d_logits]
        for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den, self._d_max,
                    self._d_p_attn, self._d_p_ff, self._d_p_num, self._d_p_den, self._d_p_max]:
            bufs.extend(lst)
        for d_ptr in self._d_debug.values():
            bufs.append(d_ptr)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


class WhisperTrtRunner:
    """Device-resident Whisper TRT inference runner for diff testing.

    Runs encoder on mel features, then feeds encoder output as cross_k/cross_v
    to a standard KV-cache decoder. Compares per-step decoder logits.
    """

    def __init__(
        self,
        decoder_plan: bytes,
        encoder_plan: bytes,
        num_layers: int,
        max_cache_length: int,
        max_source_positions: int = 1500,
        hidden_size: int | None = None,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers
        self.max_cache_length = max_cache_length
        self.max_source_positions = max_source_positions

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        # Decoder engine
        self.dec_engine = runtime.deserialize_cuda_engine(decoder_plan)
        if self.dec_engine is None:
            raise RuntimeError("Failed to deserialize Whisper decoder TRT engine")
        self.dec_context = self.dec_engine.create_execution_context()

        # Encoder engine
        self.enc_engine = runtime.deserialize_cuda_engine(encoder_plan)
        if self.enc_engine is None:
            raise RuntimeError("Failed to deserialize Whisper encoder TRT engine")
        self.enc_context = self.enc_engine.create_execution_context()

        # Auto-detect dimensions
        cache_shape = tuple(self.dec_engine.get_tensor_shape("cache_k_0"))
        self.attention_size = cache_shape[1]
        if hidden_size is None:
            cross_shape = tuple(self.dec_engine.get_tensor_shape("cross_k_0"))
            hidden_size = cross_shape[1]
        self.hidden_size = hidden_size

        cache_dtype = self.dec_engine.get_tensor_dtype("cache_k_0")
        self._cache_elem_bytes = _trt_itemsize(cache_dtype)

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        self.cache_length = 0
        attention_window = max_cache_length + 1
        row_bytes = self.attention_size * self._cache_elem_bytes

        # ----- Decoder device buffers -----
        cache_bytes = max_cache_length * row_bytes
        self._d_cache_k = []
        self._d_cache_v = []
        self._d_present_k = []
        self._d_present_v = []
        for _ in range(num_layers):
            for lst, sz in [(self._d_cache_k, cache_bytes), (self._d_cache_v, cache_bytes),
                            (self._d_present_k, row_bytes), (self._d_present_v, row_bytes)]:
                err, ptr = cudart.cudaMalloc(sz)
                _check_cuda(err)
                lst.append(ptr)

        # Cross-attention K/V (per layer, [max_source_positions, hidden_size])
        cross_bytes = max_source_positions * hidden_size * 4
        self._d_cross_k = []
        self._d_cross_v = []
        for _ in range(num_layers):
            for lst in [self._d_cross_k, self._d_cross_v]:
                err, ptr = cudart.cudaMalloc(cross_bytes)
                _check_cuda(err)
                lst.append(ptr)

        # Small I/O
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        self._h_position_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)
        err, self._d_position_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        self._h_mask = np.zeros((1, attention_window), dtype=np.float32)
        err, self._d_mask = cudart.cudaMalloc(attention_window * 4)
        _check_cuda(err)

        logits_shape = tuple(self.dec_engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # ----- Encoder device buffers -----
        mel_shape = tuple(self.enc_engine.get_tensor_shape("mel_features"))
        self._mel_shape = mel_shape
        mel_bytes = int(np.prod(mel_shape)) * 4
        err, self._d_mel = cudart.cudaMalloc(mel_bytes)
        _check_cuda(err)

        enc_out_bytes = max_source_positions * hidden_size * 4
        err, self._d_enc_out = cudart.cudaMalloc(enc_out_bytes)
        _check_cuda(err)

        # Zero-init caches
        for i in range(num_layers):
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def run_encoder(self, mel_features: np.ndarray):
        """Run encoder on mel features, populate cross_k/cross_v buffers."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream

        h_mel = np.ascontiguousarray(mel_features, dtype=np.float32)
        cudart.cudaMemcpyAsync(self._d_mel, h_mel.ctypes.data, h_mel.nbytes, H2D, stream)

        self.enc_context.set_tensor_address("mel_features", self._d_mel)
        self.enc_context.set_tensor_address("encoder_output", self._d_enc_out)
        self.enc_context.execute_async_v3(stream)

        # Copy encoder output to all cross_k/cross_v (decoder graph applies K/V projections)
        enc_bytes = self.max_source_positions * self.hidden_size * 4
        for i in range(self.num_layers):
            cudart.cudaMemcpyAsync(self._d_cross_k[i], self._d_enc_out, enc_bytes, D2D, stream)
            cudart.cudaMemcpyAsync(self._d_cross_v[i], self._d_enc_out, enc_bytes, D2D, stream)
        cudart.cudaStreamSynchronize(stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one decoder step with KV cache + cross attention."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        # Build attention mask
        position_id = min(self.cache_length, self.max_cache_length)
        self._h_mask[:] = -1e9
        valid = min(self.cache_length, self.max_cache_length)
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        cudart.cudaMemcpyAsync(self._d_token_id, self._h_token_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_position_id, self._h_position_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_mask, self._h_mask.ctypes.data, attention_window * 4, H2D, stream)

        # Bind decoder tensors
        self.dec_context.set_tensor_address("token_id", self._d_token_id)
        self.dec_context.set_tensor_address("position_id", self._d_position_id)
        self.dec_context.set_tensor_address("attention_mask", self._d_mask)
        self.dec_context.set_tensor_address("logits", self._d_logits)

        for i in range(self.num_layers):
            self.dec_context.set_tensor_address(f"cache_k_{i}", self._d_cache_k[i])
            self.dec_context.set_tensor_address(f"cache_v_{i}", self._d_cache_v[i])
            self.dec_context.set_tensor_address(f"present_k_{i}", self._d_present_k[i])
            self.dec_context.set_tensor_address(f"present_v_{i}", self._d_present_v[i])
            self.dec_context.set_tensor_address(f"cross_k_{i}", self._d_cross_k[i])
            self.dec_context.set_tensor_address(f"cross_v_{i}", self._d_cross_v[i])

        self.dec_context.execute_async_v3(stream)

        # D2D cache update
        row_bytes = self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
            for cache_buf, present_buf in [
                (self._d_cache_k[i], self._d_present_k[i]),
                (self._d_cache_v[i], self._d_present_v[i]),
            ]:
                if self.cache_length < self.max_cache_length:
                    offset = self.cache_length * row_bytes
                    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)
                else:
                    cudart.cudaMemcpyAsync(cache_buf, cache_buf + row_bytes,
                        (self.max_cache_length - 1) * row_bytes, D2D, stream)
                    offset = (self.max_cache_length - 1) * row_bytes
                    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)

        cudart.cudaMemcpyAsync(self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)

        return {"logits": self._h_logits.copy()}

    def reset(self):
        cache_bytes = self.max_cache_length * self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)
        self.cache_length = 0

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[dict[str, np.ndarray]]:
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
        bufs = [self._d_token_id, self._d_position_id, self._d_mask, self._d_logits,
                self._d_mel, self._d_enc_out]
        bufs.extend(self._d_cache_k)
        bufs.extend(self._d_cache_v)
        bufs.extend(self._d_present_k)
        bufs.extend(self._d_present_v)
        bufs.extend(self._d_cross_k)
        bufs.extend(self._d_cross_v)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


class HybridTrtRunner:
    """Device-resident hybrid TRT inference runner for models with mixed
    recurrent (DeltaNet/Mamba) + attention layers.

    Combines MambaTrtRunner's conv/ssm state management with TrtRunner's
    KV cache + position tracking. Used for Qwen3.5 (DeltaNet + attention)
    and NemotronH (Mamba-2 + attention).
    """

    def __init__(
        self,
        engine_plan: bytes,
        max_cache_length: int,
        num_mamba_layers: int,
        num_attention_layers: int,
    ):
        _require_trt_runtime()
        self.max_cache_length = max_cache_length
        self.num_mamba_layers = num_mamba_layers
        self.num_attention_layers = num_attention_layers

        # Deserialize engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()

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


class Seq2SeqTrtRunner:
    """Device-resident encoder-decoder TRT inference runner for seq2seq models.

    Runs encoder on text input tokens, then feeds encoder output as cross_k/cross_v
    to a standard KV-cache decoder. Used for BART, T5, Marian, and other
    encoder-decoder text models.
    """

    def __init__(
        self,
        decoder_plan: bytes,
        encoder_plan: bytes,
        num_layers: int,
        max_cache_length: int,
        max_source_positions: int,
        hidden_size: int | None = None,
        decoder_start_token_id: int = 2,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers
        self.max_cache_length = max_cache_length
        self.max_source_positions = max_source_positions
        self.decoder_start_token_id = decoder_start_token_id

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        # Decoder engine
        self.dec_engine = runtime.deserialize_cuda_engine(decoder_plan)
        if self.dec_engine is None:
            raise RuntimeError("Failed to deserialize seq2seq decoder TRT engine")
        self.dec_context = self.dec_engine.create_execution_context()

        # Encoder engine
        self.enc_engine = runtime.deserialize_cuda_engine(encoder_plan)
        if self.enc_engine is None:
            raise RuntimeError("Failed to deserialize seq2seq encoder TRT engine")
        self.enc_context = self.enc_engine.create_execution_context()

        # Auto-detect dimensions from decoder engine
        cache_shape = tuple(self.dec_engine.get_tensor_shape("cache_k_0"))
        self.attention_size = cache_shape[1]
        if hidden_size is None:
            cross_shape = tuple(self.dec_engine.get_tensor_shape("cross_k_0"))
            hidden_size = cross_shape[1]
        self.hidden_size = hidden_size

        cache_dtype = self.dec_engine.get_tensor_dtype("cache_k_0")
        self._cache_elem_bytes = _trt_itemsize(cache_dtype)

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        self.cache_length = 0
        attention_window = max_cache_length + 1
        row_bytes = self.attention_size * self._cache_elem_bytes

        # ----- Decoder device buffers -----
        cache_bytes = max_cache_length * row_bytes
        self._d_cache_k = []
        self._d_cache_v = []
        self._d_present_k = []
        self._d_present_v = []
        for _ in range(num_layers):
            for lst, sz in [(self._d_cache_k, cache_bytes), (self._d_cache_v, cache_bytes),
                            (self._d_present_k, row_bytes), (self._d_present_v, row_bytes)]:
                err, ptr = cudart.cudaMalloc(sz)
                _check_cuda(err)
                lst.append(ptr)

        # Cross-attention K/V (per layer, [max_source_positions, hidden_size])
        cross_bytes = max_source_positions * hidden_size * 4
        self._d_cross_k = []
        self._d_cross_v = []
        for _ in range(num_layers):
            for lst in [self._d_cross_k, self._d_cross_v]:
                err, ptr = cudart.cudaMalloc(cross_bytes)
                _check_cuda(err)
                lst.append(ptr)

        # Small I/O buffers
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        self._h_position_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)
        err, self._d_position_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        self._h_mask = np.zeros((1, attention_window), dtype=np.float32)
        err, self._d_mask = cudart.cudaMalloc(attention_window * 4)
        _check_cuda(err)

        logits_shape = tuple(self.dec_engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # ----- Encoder device buffers -----
        enc_input_bytes = max_source_positions * 4  # int32 tokens
        err, self._d_enc_input_ids = cudart.cudaMalloc(enc_input_bytes)
        _check_cuda(err)

        enc_mask_bytes = max_source_positions * 4  # float32 mask
        err, self._d_enc_mask = cudart.cudaMalloc(enc_mask_bytes)
        _check_cuda(err)

        enc_out_bytes = max_source_positions * hidden_size * 4
        err, self._d_enc_out = cudart.cudaMalloc(enc_out_bytes)
        _check_cuda(err)

        # Zero-init caches
        for i in range(num_layers):
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def run_encoder(self, input_ids: list[int]):
        """Run encoder on input token IDs, populate cross_k/cross_v buffers."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream

        # Pad input_ids to max_source_positions
        padded = np.zeros((self.max_source_positions,), dtype=np.int32)
        copy_len = min(len(input_ids), self.max_source_positions)
        padded[:copy_len] = input_ids[:copy_len]

        # Build encoder attention mask: 0.0 for valid, -1e9 for padding
        enc_mask = np.full((self.max_source_positions,), -1e9, dtype=np.float32)
        enc_mask[:copy_len] = 0.0

        cudart.cudaMemcpyAsync(self._d_enc_input_ids, padded.ctypes.data,
                               padded.nbytes, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_enc_mask, enc_mask.ctypes.data,
                               enc_mask.nbytes, H2D, stream)

        self.enc_context.set_tensor_address("input_ids", self._d_enc_input_ids)
        # Only set attention_mask if the encoder expects it
        has_mask = False
        for i in range(self.enc_engine.num_io_tensors):
            if self.enc_engine.get_tensor_name(i) == "attention_mask":
                has_mask = True
                break
        if has_mask:
            self.enc_context.set_tensor_address("attention_mask", self._d_enc_mask)
        self.enc_context.set_tensor_address("encoder_output", self._d_enc_out)
        self.enc_context.execute_async_v3(stream)

        # Copy encoder output to all cross_k/cross_v
        enc_bytes = self.max_source_positions * self.hidden_size * 4
        for i in range(self.num_layers):
            cudart.cudaMemcpyAsync(self._d_cross_k[i], self._d_enc_out, enc_bytes, D2D, stream)
            cudart.cudaMemcpyAsync(self._d_cross_v[i], self._d_enc_out, enc_bytes, D2D, stream)
        cudart.cudaStreamSynchronize(stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one decoder step with KV cache + cross attention."""
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        attention_window = self.max_cache_length + 1

        # Build attention mask
        position_id = min(self.cache_length, self.max_cache_length)
        self._h_mask[:] = -1e9
        valid = min(self.cache_length, self.max_cache_length)
        self._h_mask[0, :valid] = 0.0
        self._h_mask[0, -1] = 0.0

        self._h_token_id[0] = token_id
        self._h_position_id[0] = position_id

        cudart.cudaMemcpyAsync(self._d_token_id, self._h_token_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_position_id, self._h_position_id.ctypes.data, 4, H2D, stream)
        cudart.cudaMemcpyAsync(self._d_mask, self._h_mask.ctypes.data, attention_window * 4, H2D, stream)

        # Bind decoder tensors
        self.dec_context.set_tensor_address("token_id", self._d_token_id)
        self.dec_context.set_tensor_address("position_id", self._d_position_id)
        self.dec_context.set_tensor_address("attention_mask", self._d_mask)
        self.dec_context.set_tensor_address("logits", self._d_logits)

        for i in range(self.num_layers):
            self.dec_context.set_tensor_address(f"cache_k_{i}", self._d_cache_k[i])
            self.dec_context.set_tensor_address(f"cache_v_{i}", self._d_cache_v[i])
            self.dec_context.set_tensor_address(f"present_k_{i}", self._d_present_k[i])
            self.dec_context.set_tensor_address(f"present_v_{i}", self._d_present_v[i])
            self.dec_context.set_tensor_address(f"cross_k_{i}", self._d_cross_k[i])
            self.dec_context.set_tensor_address(f"cross_v_{i}", self._d_cross_v[i])

        self.dec_context.execute_async_v3(stream)

        # D2D cache update
        row_bytes = self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
            for cache_buf, present_buf in [
                (self._d_cache_k[i], self._d_present_k[i]),
                (self._d_cache_v[i], self._d_present_v[i]),
            ]:
                if self.cache_length < self.max_cache_length:
                    offset = self.cache_length * row_bytes
                    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)
                else:
                    cudart.cudaMemcpyAsync(cache_buf, cache_buf + row_bytes,
                        (self.max_cache_length - 1) * row_bytes, D2D, stream)
                    offset = (self.max_cache_length - 1) * row_bytes
                    cudart.cudaMemcpyAsync(cache_buf + offset, present_buf, row_bytes, D2D, stream)

        cudart.cudaMemcpyAsync(self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)

        return {"logits": self._h_logits.copy()}

    def reset(self):
        cache_bytes = self.max_cache_length * self.attention_size * self._cache_elem_bytes
        for i in range(self.num_layers):
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_k[i], 0, cache_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(self._d_cache_v[i], 0, cache_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)
        self.cache_length = 0

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[dict[str, np.ndarray]]:
        """Run encoder on input_ids, then decode autoregressively.

        Returns per-step decoder logits (one dict per decoder step).
        """
        self.run_encoder(input_ids)
        self.reset()

        results = []
        token = self.decoder_start_token_id
        for _ in range(max_new_tokens):
            result = self.step(token)
            results.append(result)
            token = int(np.argmax(result["logits"].flatten()))
        return results

    def __del__(self):
        if cudart is None:
            return
        if not hasattr(self, "_d_token_id"):
            return
        bufs = [self._d_token_id, self._d_position_id, self._d_mask, self._d_logits,
                self._d_enc_input_ids, self._d_enc_mask, self._d_enc_out]
        bufs.extend(self._d_cache_k)
        bufs.extend(self._d_cache_v)
        bufs.extend(self._d_present_k)
        bufs.extend(self._d_present_v)
        bufs.extend(self._d_cross_k)
        bufs.extend(self._d_cross_v)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


def load_engine_from_bundle(bundle_path: str) -> tuple[bytes, dict]:
    """Load engine plan bytes and metadata from a .trtfb bundle.

    Returns:
        (engine_plan_bytes, header_dict)
    """
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        engine_meta = sections.get("engine_plan", {})
        f.seek(16 + header_len + engine_meta["offset"])
        engine_plan = f.read(engine_meta["size"])

    return engine_plan, header


def runner_from_bundle(bundle_path: str) -> TrtRunner:
    """Create the appropriate TrtRunner from a .trtfb bundle file.

    Dispatches to Seq2SeqTrtRunner, HybridTrtRunner, MambaTrtRunner,
    RwkvTrtRunner, or TrtRunner based on the runtime_strategy in the
    bundle config.
    """
    config = load_config_from_bundle(bundle_path)
    engine_plan, header = load_engine_from_bundle(bundle_path)
    runtime_strategy = config.get("runtime_strategy", "decoder_kv_cache")
    num_layers = header.get("num_layers", config.get("num_hidden_layers", 1))
    tri_cfg = config.get("triattention", {}) or {}

    if runtime_strategy in ("seq2seq_encoder_decoder", "text_to_text", "marian_translation"):
        encoder_plan, _ = load_vision_engine_from_bundle(bundle_path)
        if encoder_plan is not None:
            dec_layers = config.get("decoder_layers", num_layers)
            decoder_start = config.get("decoder_start_token_id", 2)
            return Seq2SeqTrtRunner(
                decoder_plan=engine_plan,
                encoder_plan=encoder_plan,
                num_layers=dec_layers,
                max_cache_length=header["max_cache_length"],
                max_source_positions=header["max_cache_length"],
                decoder_start_token_id=decoder_start,
            )

    if runtime_strategy == "hybrid_mamba_attention":
        num_mamba = config.get("num_mamba_layers", 0)
        num_attn = config.get("num_attention_layers", 0)
        return HybridTrtRunner(
            engine_plan=engine_plan,
            max_cache_length=header["max_cache_length"],
            num_mamba_layers=num_mamba,
            num_attention_layers=num_attn,
        )
    if runtime_strategy == "ssm_recurrent":
        return MambaTrtRunner(
            engine_plan=engine_plan,
            num_layers=num_layers,
        )
    if runtime_strategy == "rwkv_recurrent":
        return RwkvTrtRunner(
            engine_plan=engine_plan,
            num_layers=num_layers,
        )
    if tri_cfg.get("enabled") and runtime_strategy in ("decoder_kv_cache", "decoder_moe"):
        tri_stats = load_triattention_stats_from_bundle(
            bundle_path,
            tri_cfg.get("stats_section", "triattention_stats.json"),
        )
        tri_runtime_cfg = TriAttentionRuntimeConfig.from_bundle_config(
            tri_cfg,
            rope_style=str(tri_stats.get("rope_style", "half")),
            max_cache_length=header["max_cache_length"],
        )
        return TriAttentionTrtRunner(
            engine_plan=engine_plan,
            max_cache_length=header["max_cache_length"],
            num_layers=num_layers,
            triattention_config=tri_runtime_cfg,
            triattention_stats_payload=tri_stats,
        )

    return TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=header["max_cache_length"],
        num_layers=num_layers,
    )


def load_vision_engine_from_bundle(bundle_path: str) -> tuple[bytes | None, dict]:
    """Load vision engine plan bytes from a .trtfb bundle.

    Returns:
        (vision_engine_plan_bytes_or_None, header_dict)
    """
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        vision_meta = sections.get("vision_engine_plan")
        if vision_meta is None:
            return None, header
        f.seek(16 + header_len + vision_meta["offset"])
        vision_plan = f.read(vision_meta["size"])

    return vision_plan, header


class VisionTrtRunner:
    """Single-pass TRT inference for vision encoder. Validation only.

    Deserializes a vision TRT engine and runs a single forward pass.
    No KV cache, no autoregressive loop.
    """

    def __init__(self, engine_plan: bytes):
        _require_trt_runtime()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize vision TRT engine")
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Discover IO tensors
        self._input_names = []
        self._output_names = []
        self._device_buffers: dict[str, int] = {}
        self._host_buffers: dict[str, np.ndarray] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = _trt_nptype_safe(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize

            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._device_buffers[name] = d_ptr
            self._host_buffers[name] = np.zeros(shape, dtype=dtype_np)

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def encode(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Run a single forward pass through the vision encoder.

        Args:
            **inputs: Named input arrays (e.g. patch_embeds=...).

        Returns:
            Dict of output name -> numpy array.
        """
        # Set input values
        for name, value in inputs.items():
            if name in self._host_buffers:
                self._host_buffers[name][:] = value.astype(
                    self._host_buffers[name].dtype)

        # Copy inputs to device
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            self.context.set_tensor_address(name, self._device_buffers[name])
            if mode == trt.TensorIOMode.INPUT:
                h_buf = self._host_buffers[name]
                cudart.cudaMemcpyAsync(
                    self._device_buffers[name],
                    h_buf.ctypes.data,
                    h_buf.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )

        self.context.execute_async_v3(self.stream)

        # Copy outputs
        results: dict[str, np.ndarray] = {}
        for name in self._output_names:
            h_buf = self._host_buffers[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data,
                self._device_buffers[name],
                h_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )

        cudart.cudaStreamSynchronize(self.stream)

        for name in self._output_names:
            results[name] = self._host_buffers[name].copy()

        return results

    def __del__(self):
        if cudart is None:
            return
        for d_ptr in self._device_buffers.values():
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


# ---------------------------------------------------------------------------
# Bundle section utilities
# ---------------------------------------------------------------------------

def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named section's raw bytes from a .trtfb bundle.

    Returns None if the section doesn't exist.
    """
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
    """Load and parse config.json from a .trtfb bundle."""
    import json
    data = load_section_from_bundle(bundle_path, "config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


def load_triattention_stats_from_bundle(
    bundle_path: str,
    section_name: str = "triattention_stats.json",
) -> dict[str, Any]:
    """Load and parse embedded TriAttention stats from a .trtfb bundle."""
    import json

    data = load_section_from_bundle(bundle_path, section_name)
    if data is None:
        raise ValueError(
            "TriAttention is enabled in the bundle config but the stats section "
            f"{section_name!r} is missing."
        )
    return json.loads(data.decode("utf-8"))


def load_preprocessor_config_from_bundle(bundle_path: str) -> dict:
    """Load and parse preprocessor_config.json from a .trtfb bundle."""
    import json
    data = load_section_from_bundle(bundle_path, "preprocessor_config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# VL combined runner
# ---------------------------------------------------------------------------

def _resolve_pil_interpolation(mode: str):
    """Map interpolation mode string to PIL constant."""
    from PIL import Image
    _map = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }
    return _map.get(mode, Image.BICUBIC)


def _preprocess_qwen_merge_group(
    image_path: str,
    fixed_image_size: int = 448,
    temporal_patch_size: int = 2,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    patch_size: int = 14,
    merge_size: int = 2,
    interpolation: str = "bicubic",
) -> np.ndarray:
    """Qwen merge-group preprocessing: [C*T, H, W] with patch permutation.

    The TRT vision engine's Conv2D produces patches in raster order of the
    input image. To match HF's pipeline (where patches come out in merge-group
    order after the processor's reshape+transpose), we spatially rearrange
    the image at the 14x14 patch level: patches that should be in merge-group
    positions are placed at the corresponding raster positions.

    Channel layout: [R_t0, R_t1, G_t0, G_t1, B_t0, B_t1] matching Conv3D
    weight layout [out, C, T, kH, kW] reshaped to Conv2D [out, C*T, kH, kW].
    """
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((fixed_image_size, fixed_image_size), resample)
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Normalize per channel
    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    # HWC -> CHW
    img_chw = img_np.transpose(2, 0, 1)  # [C, H, W]
    C = img_chw.shape[0]
    T = temporal_patch_size
    H = W = fixed_image_size
    grid_h = H // patch_size
    grid_w = W // patch_size

    # Extract 14x14 patches from the original image: [gH, gW, C, pH, pW]
    orig_patches = img_chw.reshape(
        C, grid_h, patch_size, grid_w, patch_size
    ).transpose(1, 3, 0, 2, 4)  # [gH, gW, C, pH, pW]

    # Build merge-group ordering: maps merge-group index -> (orig_h, orig_w)
    # Merge groups iterate: (mh, mw, dh, dw) with orig = (mh*2+dh, mw*2+dw)
    merge_h = grid_h // merge_size
    merge_w = grid_w // merge_size
    merge_idx = np.zeros((grid_h * grid_w, 2), dtype=np.int32)
    idx = 0
    for mh in range(merge_h):
        for mw in range(merge_w):
            for dh in range(merge_size):
                for dw in range(merge_size):
                    merge_idx[idx] = [mh * merge_size + dh, mw * merge_size + dw]
                    idx += 1

    # Place merge-group-ordered patches at raster positions in pseudo-image.
    # Patch at raster position i in the pseudo-image gets the content from
    # the original position merge_idx[i], so Conv2D output patch i matches
    # HF's i-th merge-group-ordered patch.
    pseudo_patches = np.zeros_like(orig_patches)  # [gH, gW, C, pH, pW]
    for i in range(grid_h * grid_w):
        ph = i // grid_w
        pw = i % grid_w
        oh, ow = merge_idx[i]
        pseudo_patches[ph, pw] = orig_patches[oh, ow]

    # Reconstruct pseudo-image: [C, H, W]
    pseudo_img = pseudo_patches.transpose(2, 0, 3, 1, 4).reshape(C, H, W)

    # Temporal duplication with [C, T] channel layout:
    # [R_t0, R_t1, G_t0, G_t1, B_t0, B_t1]
    pixel_values = np.repeat(pseudo_img, T, axis=0)  # [C*T, H, W]
    return pixel_values.astype(np.float32)


def _preprocess_simple_chw(
    image_path: str,
    fixed_image_size: int = 448,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    interpolation: str = "bicubic",
    **_kwargs: Any,
) -> np.ndarray:
    """Standard resize + normalize preprocessing: [C, H, W].

    No patch permutation, no temporal duplication. Works for LLaVA,
    InternVL, Phi-3-Vision, and other standard ViT-based VL models.
    """
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((fixed_image_size, fixed_image_size), resample)
    img_np = np.array(img, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    # HWC -> CHW
    return img_np.transpose(2, 0, 1).astype(np.float32)


def _preprocess_center_crop_chw(
    image_path: str,
    fixed_image_size: int = 448,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    interpolation: str = "bicubic",
    **_kwargs: Any,
) -> np.ndarray:
    """Center-crop to square, then resize + normalize: [C, H, W].

    For traditional CLIP and DINOv2-based VL models that center-crop
    before resize.
    """
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")

    # Center-crop to square
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    img = img.resize((fixed_image_size, fixed_image_size), resample)
    img_np = np.array(img, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    return img_np.transpose(2, 0, 1).astype(np.float32)


def _preprocess_aspect_preserve_chw(
    image_path: str,
    fixed_image_size: int = 448,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    interpolation: str = "bicubic",
    **_kwargs: Any,
) -> np.ndarray:
    """Aspect-ratio-preserving resize + zero-pad to square: [C, H, W].

    Fits image into fixed_image_size x fixed_image_size without distortion,
    padding the remainder with zeros. For InternVL v2 and similar models.
    """
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")

    # Compute scaled dimensions fitting inside target square
    w, h = img.size
    scale = fixed_image_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.resize((new_w, new_h), resample)

    # Zero-pad to target square (top-left aligned)
    padded = Image.new("RGB", (fixed_image_size, fixed_image_size), (0, 0, 0))
    padded.paste(img, (0, 0))

    img_np = np.array(padded, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    return img_np.transpose(2, 0, 1).astype(np.float32)


def _preprocess_pad_center_chw(
    image_path: str,
    fixed_image_size: int = 448,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    interpolation: str = "bicubic",
    **_kwargs: Any,
) -> np.ndarray:
    """Aspect-ratio-preserving resize + center-pad with mean color: [C, H, W]."""
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    scale = min(fixed_image_size / w, fixed_image_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), resample)

    pad_color = tuple(int(float(value) * 255.0) for value in image_mean[:3])
    padded = Image.new("RGB", (fixed_image_size, fixed_image_size), pad_color)
    x_off = (fixed_image_size - new_w) // 2
    y_off = (fixed_image_size - new_h) // 2
    padded.paste(img, (x_off, y_off))

    img_np = np.array(padded, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    return img_np.transpose(2, 0, 1).astype(np.float32)


def preprocess_image_for_trt(
    image_path: str,
    preprocessor_type: str = "qwen_merge_group",
    **kwargs: Any,
) -> np.ndarray:
    """Load and preprocess an image for the TRT vision engine.

    Dispatches to the appropriate strategy based on preprocessor_type:
      "qwen_merge_group":    [C*T, H, W] with merge-group patch permutation
      "simple_chw":          [C, H, W] standard resize + normalize
      "center_crop_chw":     [C, H, W] center-crop to square, then resize + normalize
      "aspect_preserve_chw": [C, H, W] aspect-preserving resize + zero-pad
      "pad_center_chw":      [C, H, W] aspect-preserving resize + center mean-pad
    """
    temporal = kwargs.get("temporal_patch_size", 1)
    if preprocessor_type == "simple_chw":
        result = _preprocess_simple_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return result
    if preprocessor_type == "center_crop_chw":
        result = _preprocess_center_crop_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return result
    if preprocessor_type == "aspect_preserve_chw":
        result = _preprocess_aspect_preserve_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return result
    if preprocessor_type == "pad_center_chw":
        result = _preprocess_pad_center_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return result
    if preprocessor_type != "qwen_merge_group":
        warnings.warn(
            f"Unknown preprocessor_type {preprocessor_type!r}, "
            f"falling back to qwen_merge_group",
            stacklevel=2,
        )
    return _preprocess_qwen_merge_group(image_path, **kwargs)


class SegmentationTrtRunner:
    """Single-pass TRT inference for segmentation models.

    No KV cache, no autoregressive loop. Takes pixel_values [1, 3, H, W]
    and returns logits [1, num_classes, H/4, W/4].
    """

    def __init__(self, engine_plan: bytes):
        _require_trt_runtime()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize segmentation TRT engine")
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Discover IO tensors
        self._device_buffers: dict[str, int] = {}
        self._host_buffers: dict[str, np.ndarray] = {}
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = _trt_nptype_safe(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize

            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._device_buffers[name] = d_ptr
            self._host_buffers[name] = np.zeros(shape, dtype=dtype_np)

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def forward(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Run a single forward pass.

        Args:
            pixel_values: [1, 3, H, W] float32 normalized image.

        Returns:
            Dict with 'logits' [1, num_classes, H/4, W/4].
        """
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        stream = self.stream

        # Set input
        self._host_buffers["pixel_values"][:] = pixel_values.astype(np.float32)

        # H2D all inputs
        for name in self._input_names:
            h_buf = self._host_buffers[name]
            self.context.set_tensor_address(name, self._device_buffers[name])
            cudart.cudaMemcpyAsync(
                self._device_buffers[name], h_buf.ctypes.data,
                h_buf.nbytes, H2D, stream)

        for name in self._output_names:
            self.context.set_tensor_address(name, self._device_buffers[name])

        self.context.execute_async_v3(stream)

        # D2H outputs
        results: dict[str, np.ndarray] = {}
        for name in self._output_names:
            h_buf = self._host_buffers[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, self._device_buffers[name],
                h_buf.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)

        for name in self._output_names:
            results[name] = self._host_buffers[name].copy()

        return results

    def __del__(self):
        if cudart is None:
            return
        for d_ptr in self._device_buffers.values():
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


class VLTrtRunner:
    """Full VL pipeline runner combining vision encoder + text decoder.

    Runs: preprocess image -> vision TRT -> build prompt -> text TRT decode.
    Matches the C++ VLBackendFastPath pipeline exactly.
    """

    def __init__(
        self,
        bundle_path: str,
        tokenizer=None,
    ):
        self.bundle_path = bundle_path
        self.config = load_config_from_bundle(bundle_path)
        self.preproc_config = load_preprocessor_config_from_bundle(bundle_path)

        # Load text decoder engine
        engine_plan, header = load_engine_from_bundle(bundle_path)
        self.text_runner = TrtRunner(
            engine_plan=engine_plan,
            max_cache_length=header["max_cache_length"],
            num_layers=header["num_layers"],
        )

        # Load vision engine
        vision_plan, _ = load_vision_engine_from_bundle(bundle_path)
        self.vision_runner = VisionTrtRunner(vision_plan) if vision_plan else None

        # VL config from bundle
        self.image_token_id = self.config.get("image_token_id", -1)
        self.num_image_pad_tokens = self.config.get("num_image_pad_tokens", 256)
        self.vl_prompt_template = self.config.get("vl_prompt_template", "")
        self.image_token_str = self.config.get("image_token_str", "")
        self.fixed_image_size = self.config.get("fixed_image_size", 448)
        self.preprocessor_type = self.config.get(
            "preprocessor_type", "qwen_merge_group")

        # Preprocessor config
        self.temporal_patch_size = self.preproc_config.get("temporal_patch_size", 2)
        self.patch_size = self.preproc_config.get("patch_size", 14)
        self.merge_size = self.preproc_config.get("merge_size", 2)
        self.image_mean = tuple(self.preproc_config.get(
            "image_mean", [0.48145466, 0.4578275, 0.40821073]))
        self.image_std = tuple(self.preproc_config.get(
            "image_std", [0.26862954, 0.26130258, 0.27577711]))
        self.interpolation = self.config.get("interpolation", "bicubic")

        self.tokenizer = tokenizer

    def encode_image(self, image_path: str) -> np.ndarray:
        """Run the vision encoder on a single image. Returns [N, dim] features.

        Only single-image input is supported. Pass a single path string.
        """
        if isinstance(image_path, (list, tuple)):
            raise NotImplementedError(
                "Multi-image input is not yet supported. "
                "Pass a single image path string.")
        if self.vision_runner is None:
            raise RuntimeError("No vision engine in bundle")

        pixel_values = preprocess_image_for_trt(
            image_path,
            preprocessor_type=self.preprocessor_type,
            fixed_image_size=self.fixed_image_size,
            temporal_patch_size=self.temporal_patch_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
            patch_size=self.patch_size,
            merge_size=self.merge_size,
            interpolation=self.interpolation,
        )
        results = self.vision_runner.encode(pixel_values=pixel_values)
        return results["image_features"]

    def format_prompt(self, user_prompt: str) -> str:
        """Format the VL prompt with image pad tokens."""
        image_pads = self.image_token_str * self.num_image_pad_tokens
        result = self.vl_prompt_template
        result = result.replace("{image_pads}", image_pads)
        result = result.replace("{prompt}", user_prompt)
        return result

    def generate_vl(
        self,
        input_ids: list[int],
        image_features: np.ndarray,
        max_new_tokens: int,
    ) -> list[int]:
        """Run VL generation with pre-computed image features.

        Matches C++ VLBackendFastPath::generate_vl exactly:
        - During prefill, image_token_id tokens are replaced with image features.
        - During decode, normal autoregressive generation.
        """
        feat_idx = 0
        output_ids = list(input_ids)

        # Prefill: all but last token
        for tid in input_ids[:-1]:
            embed = None
            use_embed = 0.0
            if tid == self.image_token_id and feat_idx < len(image_features):
                embed = image_features[feat_idx:feat_idx+1]  # [1, dim]
                use_embed = 1.0
                feat_idx += 1
            self.text_runner.step(tid, input_embed=embed, use_input_embed=use_embed)

        # Last prefill token
        last_tid = input_ids[-1]
        embed = None
        use_embed = 0.0
        if last_tid == self.image_token_id and feat_idx < len(image_features):
            embed = image_features[feat_idx:feat_idx+1]
            use_embed = 1.0
            feat_idx += 1
        result = self.text_runner.step(last_tid, input_embed=embed, use_input_embed=use_embed)

        # Decode
        for _ in range(max_new_tokens):
            logits = result["logits"].flatten()
            next_token = int(np.argmax(logits))
            output_ids.append(next_token)
            eos_ids = self.config.get("eos_token_id", [])
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            if next_token in eos_ids:
                break
            result = self.text_runner.step(next_token)

        return output_ids
