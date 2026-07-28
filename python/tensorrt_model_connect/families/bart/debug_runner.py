# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART-owned debug runner adapter."""

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

def _check_cuda(status):
    """Raise on CUDA error."""
    if cudart is None:
        raise RuntimeError("cuda-python is required for family debug_runner execution")
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
        raise ImportError("tensorrt is required for family debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for family debug_runner execution")


def _decoder_cross_attention_mask_name(engine) -> str | None:
    tensor_names = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
    }
    for name in ("cross_attention_mask", "encoder_mask"):
        if name in tensor_names:
            return name
    return None


def load_vision_engine_from_bundle(bundle_path: str) -> tuple[bytes | None, dict]:
    """Load vision engine plan bytes from this family's .trtfb bundle."""
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


class Seq2SeqTrtRunner:
    """Device-resident encoder-decoder TRT inference runner for seq2seq models.

    Runs encoder on text input tokens, then feeds encoder output as cross_k/cross_v
    to a standard KV-cache decoder. Used for encoder-decoder text models.
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
        distributed_communicator: object | None = None,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers
        self.max_cache_length = max_cache_length
        self.max_source_positions = max_source_positions
        self.decoder_start_token_id = decoder_start_token_id
        self._distributed_communicator = distributed_communicator

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        # Decoder engine
        self.dec_engine = runtime.deserialize_cuda_engine(decoder_plan)
        if self.dec_engine is None:
            raise RuntimeError("Failed to deserialize seq2seq decoder TRT engine")
        self.dec_context = self.dec_engine.create_execution_context()
        if distributed_communicator is not None:
            set_communicator = getattr(self.dec_context, "set_communicator", None)
            if set_communicator is None:
                raise RuntimeError(
                    "TensorRT distributed seq2seq debug execution requires "
                    "IExecutionContext.set_communicator"
                )
            if not set_communicator(distributed_communicator):
                raise RuntimeError("Failed to set TensorRT distributed communicator")

        # Encoder engine
        self.enc_engine = runtime.deserialize_cuda_engine(encoder_plan)
        if self.enc_engine is None:
            raise RuntimeError("Failed to deserialize seq2seq encoder TRT engine")
        self.enc_context = self.enc_engine.create_execution_context()

        # Auto-detect dimensions from decoder engine
        cache_shape = tuple(self.dec_engine.get_tensor_shape("cache_k_0"))
        self.attention_size = cache_shape[1]
        self._decoder_cross_attention_mask_name = (
            _decoder_cross_attention_mask_name(self.dec_engine)
        )
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

        self._debug_output_names = []
        self._h_debug = {}
        self._d_debug = {}
        for index in range(self.dec_engine.num_io_tensors):
            name = self.dec_engine.get_tensor_name(index)
            if not name.startswith("debug_"):
                continue
            if self.dec_engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.dec_engine.get_tensor_shape(name))
            dtype = _trt_nptype_safe(self.dec_engine.get_tensor_dtype(name))
            host = np.zeros(shape, dtype=dtype)
            err, device = cudart.cudaMalloc(host.nbytes)
            _check_cuda(err)
            self._debug_output_names.append(name)
            self._h_debug[name] = host
            self._d_debug[name] = device

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
        for name in self._debug_output_names:
            self.dec_context.set_tensor_address(name, self._d_debug[name])

        for i in range(self.num_layers):
            self.dec_context.set_tensor_address(f"cache_k_{i}", self._d_cache_k[i])
            self.dec_context.set_tensor_address(f"cache_v_{i}", self._d_cache_v[i])
            self.dec_context.set_tensor_address(f"present_k_{i}", self._d_present_k[i])
            self.dec_context.set_tensor_address(f"present_v_{i}", self._d_present_v[i])
            self.dec_context.set_tensor_address(f"cross_k_{i}", self._d_cross_k[i])
            self.dec_context.set_tensor_address(f"cross_v_{i}", self._d_cross_v[i])
        if self._decoder_cross_attention_mask_name is not None:
            self.dec_context.set_tensor_address(
                self._decoder_cross_attention_mask_name, self._d_enc_mask)

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
        for name in self._debug_output_names:
            host = self._h_debug[name]
            cudart.cudaMemcpyAsync(
                host.ctypes.data, self._d_debug[name], host.nbytes, D2H, stream)
        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)

        outputs = {"logits": self._h_logits.copy()}
        outputs.update({
            name: self._h_debug[name].copy()
            for name in self._debug_output_names
        })
        return outputs

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
        bufs.extend(self._d_debug.values())
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)


def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> object | None:
    del runtime_strategy
    encoder_plan, _ = load_vision_engine_from_bundle(bundle_path)
    if encoder_plan is None:
        return None
    dec_layers = config.get("decoder_layers", header.get("num_layers", 1))
    decoder_start = config.get("decoder_start_token_id", 2)
    return Seq2SeqTrtRunner(
        decoder_plan=engine_plan,
        encoder_plan=encoder_plan,
        num_layers=dec_layers,
        max_cache_length=header["max_cache_length"],
        max_source_positions=header["max_cache_length"],
        decoder_start_token_id=decoder_start,
        distributed_communicator=distributed_communicator,
    )
