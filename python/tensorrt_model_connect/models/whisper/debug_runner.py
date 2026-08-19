# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper-owned TRT debug runner."""

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

def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named raw section from this family's .bundle artifact."""
    import json
    import struct

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
    """Load and parse this family's config.json from a .bundle artifact."""
    import json

    data = load_section_from_bundle(bundle_path, "config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


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
