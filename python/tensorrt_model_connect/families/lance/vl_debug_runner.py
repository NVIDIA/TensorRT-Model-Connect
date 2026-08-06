# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned VL debug runner used by this model family.

This file intentionally duplicates the Python VL debug path so changes to one
family's runner do not couple sibling model families.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import tempfile
import time
import warnings
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
        bufs = []
        for name in ("_d_token_id", "_d_position_id", "_d_mask", "_d_logits",
                     "_d_input_embed", "_d_use_input_embed", "_d_deepstack_active"):
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


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * _NCCL_UNIQUE_ID_BYTES)]


def _env_int(names: tuple[str, ...], default: int | None = None) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _mpi_rank_info_from_env() -> tuple[int, int]:
    rank = _env_int(("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "RANK"), 0)
    world_size = _env_int(
        ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "WORLD_SIZE"), 1)
    return int(rank or 0), int(world_size or 1)


def _default_nccl_rendezvous_path() -> str:
    path = os.environ.get("TRTMC_NCCL_RENDEZVOUS")
    if path:
        return path
    job_id = (
        os.environ.get("OMPI_COMM_WORLD_JOBID")
        or os.environ.get("PMIX_NAMESPACE")
        or os.environ.get("SLURM_JOB_ID")
        or f"pid{os.getppid()}"
    )
    safe_job_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_id)
    return os.path.join(tempfile.gettempdir(), f"trtmc_nccl_{safe_job_id}.bin")


def _load_nccl_library() -> ctypes.CDLL:
    errors: list[str] = []
    for name in ("libnccl.so.2", "libnccl.so"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError("Unable to load NCCL library: " + "; ".join(errors))

    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
    lib.ncclGetUniqueId.restype = ctypes.c_int
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        _NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommInitRank.restype = ctypes.c_int
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommDestroy.restype = ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib


def _nccl_error_string(lib: ctypes.CDLL, status: int) -> str:
    try:
        msg = lib.ncclGetErrorString(status)
    except Exception:
        msg = None
    if msg:
        return msg.decode("utf-8", errors="replace")
    return f"NCCL error {status}"


def _capsule_from_pointer(ptr: int):
    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return pycapsule_new(ctypes.c_void_p(ptr), None, None)


class TensorParallelNcclGroup:
    """Small NCCL group helper for TensorRT distributed debug execution.

    The caller must destroy TRT execution contexts before closing this group.
    """

    def __init__(
        self,
        world_size: int | None = None,
        rendezvous_path: str | None = None,
        timeout_s: float = 60.0,
        set_device: bool = True,
    ):
        _require_trt_runtime()
        self.rank, detected_world_size = _mpi_rank_info_from_env()
        self.world_size = int(world_size or detected_world_size)
        if self.world_size <= 1:
            raise RuntimeError("TensorParallelNcclGroup requires world_size > 1")
        if detected_world_size != self.world_size:
            raise RuntimeError(
                f"MPI world size {detected_world_size} does not match "
                f"requested tensor parallel size {self.world_size}"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise RuntimeError(
                f"MPI rank {self.rank} is outside world size {self.world_size}")

        if set_device:
            status = cudart.cudaSetDevice(self.rank)
            _check_cuda(status[0] if isinstance(status, tuple) else status)

        self.rendezvous_path = rendezvous_path or _default_nccl_rendezvous_path()
        self._lib = _load_nccl_library()
        self._comm = ctypes.c_void_p()
        unique_id = self._exchange_unique_id(timeout_s=timeout_s)
        self._check(
            self._lib.ncclCommInitRank(
                ctypes.byref(self._comm),
                self.world_size,
                unique_id,
                self.rank,
            ),
            "ncclCommInitRank",
        )
        if not self._comm.value:
            raise RuntimeError("NCCL returned a null communicator")
        self._communicator_capsule = _capsule_from_pointer(int(self._comm.value))
        self._closed = False

    @property
    def communicator(self):
        """PyCapsule wrapping the ncclComm_t pointer for TensorRT Python."""
        return self._communicator_capsule

    def _check(self, status: int, op: str) -> None:
        if int(status) != _NCCL_SUCCESS:
            raise RuntimeError(f"{op} failed: {_nccl_error_string(self._lib, status)}")

    def _exchange_unique_id(self, timeout_s: float) -> _NcclUniqueId:
        path = self.rendezvous_path
        if self.rank == 0:
            unique_id = _NcclUniqueId()
            self._check(self._lib.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                f.write(ctypes.string_at(ctypes.byref(unique_id), _NCCL_UNIQUE_ID_BYTES))
            os.replace(tmp_path, path)
            return unique_id

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if len(data) == _NCCL_UNIQUE_ID_BYTES:
                    unique_id = _NcclUniqueId()
                    ctypes.memmove(ctypes.byref(unique_id), data, _NCCL_UNIQUE_ID_BYTES)
                    return unique_id
            except FileNotFoundError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for NCCL rendezvous file {path!r}")
            time.sleep(0.05)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if self._comm.value:
            self._check(self._lib.ncclCommDestroy(self._comm), "ncclCommDestroy")
            self._comm = ctypes.c_void_p()

    def __enter__(self) -> "TensorParallelNcclGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
                host = self._host_buffers[name]
                converted = value.astype(host.dtype)
                if converted.shape != host.shape:
                    if (
                        converted.ndim == 3
                        and host.ndim == 3
                        and converted.shape[1:] == host.shape[1:]
                        and host.shape[0] % converted.shape[0] == 0
                    ):
                        converted = np.tile(
                            converted,
                            (host.shape[0] // converted.shape[0], 1, 1),
                        )
                    else:
                        raise ValueError(
                            f"vision input {name!r} has shape {converted.shape}; "
                            f"engine expects {host.shape}")
                host[:] = converted

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

def _resolve_pil_interpolation(mode: str):
    """Map interpolation mode string to PIL constant."""
    from PIL import Image
    _map = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }
    return _map.get(mode, Image.BICUBIC)


def _preprocess_merge_group_chw(
    image_path: str,
    fixed_image_size: int = 448,
    temporal_patch_size: int = 2,
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    patch_size: int = 14,
    merge_size: int = 2,
    interpolation: str = "bicubic",
) -> np.ndarray:
    """Merge-group preprocessing: [C*T, H, W] with patch permutation.

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

    No patch permutation, no temporal duplication. Works for standard
    ViT-based VL models.
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


def _preprocess_patchify_chw(
    image_path: str,
    fixed_image_size: int = 448,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
    patch_size: int = 14,
    interpolation: str = "bicubic",
    **_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Patchified CHW preprocessing: [N, C, pH, pW] plus [1, 2] grid."""
    from PIL import Image

    if patch_size <= 0 or fixed_image_size % patch_size != 0:
        raise ValueError(
            "fixed_image_size must be divisible by patch_size")

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((fixed_image_size, fixed_image_size), resample)
    img_np = np.array(img, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    img_chw = img_np.transpose(2, 0, 1)
    channels = img_chw.shape[0]
    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    pixel_values = img_chw.reshape(
        channels, grid_h, patch_size, grid_w, patch_size
    ).transpose(1, 3, 0, 2, 4).reshape(
        grid_h * grid_w, channels, patch_size, patch_size
    )
    image_grid_hws = np.array([[grid_h, grid_w]], dtype=np.int32)
    return pixel_values.astype(np.float32), image_grid_hws


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
    padding the remainder with zeros.
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
    """Aspect-ratio-preserving resize + centered zero-pad: [C, H, W]."""
    from PIL import Image

    resample = _resolve_pil_interpolation(interpolation)
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    scale = fixed_image_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.resize((new_w, new_h), resample)

    padded = Image.new("RGB", (fixed_image_size, fixed_image_size), (0, 0, 0))
    left = (fixed_image_size - new_w) // 2
    top = (fixed_image_size - new_h) // 2
    padded.paste(img, (left, top))

    img_np = np.array(padded, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std

    return img_np.transpose(2, 0, 1).astype(np.float32)


def preprocess_image_inputs_for_trt(
    image_path: str,
    preprocessor_type: str = "merge_group_chw",
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Load and preprocess image inputs for a TRT vision engine.

    Returns named arrays keyed by TensorRT input name. Most models only need
    pixel_values; patchified preprocessing also returns image_grid_hws.
    """
    temporal = kwargs.get("temporal_patch_size", 1)

    if preprocessor_type == "patchify_chw":
        pixel_values, image_grid_hws = _preprocess_patchify_chw(
            image_path, **kwargs)
        return {
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }

    if preprocessor_type == "simple_chw":
        result = _preprocess_simple_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return {"pixel_values": result}
    if preprocessor_type == "center_crop_chw":
        result = _preprocess_center_crop_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return {"pixel_values": result}
    if preprocessor_type == "aspect_preserve_chw":
        result = _preprocess_aspect_preserve_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return {"pixel_values": result}
    if preprocessor_type == "pad_center_chw":
        result = _preprocess_pad_center_chw(image_path, **kwargs)
        if temporal > 1 and result.shape[0] < temporal * 3:
            result = np.tile(result, (temporal, 1, 1))
        return {"pixel_values": result}
    if preprocessor_type != "merge_group_chw":
        warnings.warn(
            f"Unknown preprocessor_type {preprocessor_type!r}, "
            f"falling back to merge_group_chw",
            stacklevel=2,
        )
    return {"pixel_values": _preprocess_merge_group_chw(image_path, **kwargs)}


def preprocess_image_for_trt(
    image_path: str,
    preprocessor_type: str = "merge_group_chw",
    **kwargs: Any,
) -> np.ndarray:
    """Load and preprocess an image for the TRT vision engine.

    Compatibility wrapper returning only pixel_values. Use
    preprocess_image_inputs_for_trt when the engine has auxiliary inputs.
    """
    return preprocess_image_inputs_for_trt(
        image_path, preprocessor_type=preprocessor_type, **kwargs)["pixel_values"]

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
            "preprocessor_type", "merge_group_chw")

        # Preprocessor config
        self.temporal_patch_size = self.preproc_config.get(
            "temporal_patch_size", self.config.get("temporal_patch_size", 2))
        self.patch_size = self.preproc_config.get(
            "patch_size", self.config.get("patch_size", 14))
        self.merge_size = self.preproc_config.get(
            "merge_size", self.config.get("merge_size", 2))
        self.image_mean = tuple(self.preproc_config.get(
            "image_mean", self.config.get(
                "image_mean", [0.48145466, 0.4578275, 0.40821073])))
        self.image_std = tuple(self.preproc_config.get(
            "image_std", self.config.get(
                "image_std", [0.26862954, 0.26130258, 0.27577711])))
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

        vision_inputs = preprocess_image_inputs_for_trt(
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
        results = self.vision_runner.encode(**vision_inputs)
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
