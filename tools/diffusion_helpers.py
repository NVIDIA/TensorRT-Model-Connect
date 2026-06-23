"""Shared utilities for diffusion pipeline tools.

Used by diffusion debug and component validation tools.
"""
from __future__ import annotations

import json
import struct

import numpy as np


def silu(x):
    """SiLU (Swish) activation function."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def gelu_tanh(x):
    """GELU with tanh approximation."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def load_pp_weights(bundle_path: str) -> dict[str, np.ndarray]:
    """Load preprocessor weights from bundle."""
    with open(bundle_path, "rb") as f:
        f.read(8)
        jl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(jl))
        ds = 16 + jl
    sec = hdr["sections"]["preprocessor_weights"]
    with open(bundle_path, "rb") as f:
        f.seek(ds + sec["offset"])
        ppd = f.read(sec["size"])
    il = struct.unpack("<I", ppd[:4])[0]
    ppx = json.loads(ppd[4 : 4 + il])
    blob = ppd[4 + il :]
    def load(k):
        i = ppx[k]
        c = int(np.prod(i["shape"]))
        return np.frombuffer(blob, np.float32, c, i["offset"]).reshape(i["shape"])
    return {k: load(k) for k in ppx}


def load_bundle_config(bundle_path: str) -> dict:
    """Load config.json from bundle."""
    with open(bundle_path, "rb") as f:
        f.read(8)
        jl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(jl))
        ds = 16 + jl
    sec = hdr["sections"]["config.json"]
    with open(bundle_path, "rb") as f:
        f.seek(ds + sec["offset"])
        return json.loads(f.read(sec["size"]).decode("utf-8"))


def compute_timestep_embedding(timestep: float, pp: dict,
                                freq_dim: int = 256):
    """Compute timestep embedding using preprocessor weights.

    Returns (temb_6d, time_embed).
    """
    half = freq_dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    embed = np.concatenate([
        np.cos(timestep * freqs), np.sin(timestep * freqs)
    ]).astype(np.float32).reshape(1, freq_dim)

    h = embed @ pp["condition_embedder.time_embedding.0.weight"] + pp["condition_embedder.time_embedding.0.bias"]
    h = silu(h)
    time_embed = h @ pp["condition_embedder.time_embedding.2.weight"] + pp["condition_embedder.time_embedding.2.bias"]

    silu_te = silu(time_embed.copy())
    temb_6d = silu_te @ pp["condition_embedder.time_proj.weight"] + pp["condition_embedder.time_proj.bias"]

    return temb_6d, time_embed


def project_text(text_emb: np.ndarray, pp: dict):
    """Project text embeddings to denoiser hidden dimension."""
    seq_len = text_emb.shape[0] if text_emb.ndim == 2 else text_emb.shape[1]
    flat = text_emb.reshape(seq_len, -1)
    out = flat @ pp["condition_embedder.text_embedding.weight"] + pp["condition_embedder.text_embedding.bias"]
    out = gelu_tanh(out)
    out = out @ pp["condition_embedder.text_embedding_2.weight"] + pp["condition_embedder.text_embedding_2.bias"]
    return out


def get_cudart():
    """Import cudart from whichever cuda-python is installed."""
    try:
        from cuda import cudart
        return cudart
    except ImportError:
        pass
    try:
        from cuda.bindings import runtime as cudart
        return cudart
    except ImportError:
        raise ImportError("No cuda-python runtime found. Install cuda-python.")


def run_trt_engine(plan: bytes, inputs: dict[str, np.ndarray],
                    output_specs: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Run a TRT engine with given inputs/outputs via CUDA."""
    import tensorrt as trt
    cudart = get_cudart()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    stream = cudart.cudaStreamCreate()[1]
    device_ptrs = {}

    for name, arr in inputs.items():
        arr = np.ascontiguousarray(arr)
        d_ptr = cudart.cudaMalloc(arr.nbytes)[1]
        cudart.cudaMemcpyAsync(
            d_ptr, arr.ctypes.data, arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.set_tensor_address(name, d_ptr)
        device_ptrs[name] = d_ptr

    outputs = {}
    for name, (shape, dtype) in output_specs.items():
        h_out = np.empty(shape, dtype=dtype)
        d_ptr = cudart.cudaMalloc(h_out.nbytes)[1]
        context.set_tensor_address(name, d_ptr)
        device_ptrs[name] = d_ptr
        outputs[name] = (h_out, d_ptr)

    context.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    results = {}
    for name, (h_out, d_ptr) in outputs.items():
        cudart.cudaMemcpy(
            h_out.ctypes.data, d_ptr, h_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        results[name] = h_out

    for d_ptr in device_ptrs.values():
        cudart.cudaFree(d_ptr)
    cudart.cudaStreamDestroy(stream)

    return results
