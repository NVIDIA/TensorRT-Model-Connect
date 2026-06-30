#!/usr/bin/env python3
"""TRT decomposed attention vs FlashInfer TVM-FFI plugin — layer-level E2E benchmark.

Builds two TRT engines with Qwen3-0.6B attention dimensions:
  1. Decomposed: Q@K^T -> scale -> softmax -> @V (standard graph_ops)
  2. FlashInfer: single fused kernel via TvmFfiKernel plugin

Runs 100 decode steps (simulating autoregressive generation) and compares latency.
"""
if __name__ != "__main__":
    import pytest

    pytest.skip(
        "FlashInfer TVM-FFI benchmark script requires explicit direct execution.",
        allow_module_level=True,
    )

import ctypes as ct
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- Load TVM-FFI and register FlashInfer decode kernel ---
import tvm_ffi  # noqa: F401

# Load shared lib for plugin registration
REPO_ROOT = Path(__file__).resolve().parents[4]
shared_lib = str(REPO_ROOT / "build_shared" / "libtrtmc_core.so")
if not os.path.exists(shared_lib):
    print(f"SKIP: {shared_lib} not found")
    sys.exit(0)

lib = ct.CDLL(shared_lib, mode=ct.RTLD_GLOBAL)
lib.tvm_ffi_plugin_force_link()

import tensorrt as trt  # noqa: E402

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("SKIP: FlashInfer not available")
    sys.exit(0)

import torch  # noqa: E402

# Qwen3-0.6B dimensions
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 1024
KV_SIZE = NUM_KV_HEADS * HEAD_DIM    # 512
CACHE_LEN = 128
SCALE = 1.0 / (HEAD_DIM ** 0.5)


# --- Register FlashInfer kernel via TVM-FFI C API ---

libtvm = ct.CDLL("libtvm_ffi.so", mode=ct.RTLD_GLOBAL)

SAFE_CALL_TYPE = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p)

class DLDevice(ct.Structure):
    _fields_ = [("device_type", ct.c_int32), ("device_id", ct.c_int32)]

class DLDataType(ct.Structure):
    _fields_ = [("code", ct.c_uint8), ("bits", ct.c_uint8), ("lanes", ct.c_uint16)]

class DLTensor(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p), ("device", DLDevice), ("ndim", ct.c_int32),
        ("dtype", DLDataType), ("shape", ct.POINTER(ct.c_int64)),
        ("strides", ct.POINTER(ct.c_int64)), ("byte_offset", ct.c_uint64),
    ]

class TVMFFIAny(ct.Structure):
    _fields_ = [("type_index", ct.c_int32), ("padding", ct.c_int32), ("v_int64", ct.c_int64)]

class TVMFFIByteArray(ct.Structure):
    _fields_ = [("data", ct.c_char_p), ("size", ct.c_int64)]

kTVMFFIDLTensorPtr = 7
kTVMFFINone = 0
kTVMFFIOpaquePtr = 4


def _flashinfer_decode_callback(handle, args_ptr, num_args, result_ptr):
    """FlashInfer single_decode_with_kv_cache via TVM-FFI.

    Receives: q DLTensor, k DLTensor, v DLTensor, output DLTensor, stream ptr.
    Uses device-to-device torch tensor wrapping via cudaMemcpy.
    """
    args = ct.cast(args_ptr, ct.POINTER(TVMFFIAny))
    result = ct.cast(result_ptr, ct.POINTER(TVMFFIAny))

    q_dl = ct.cast(args[0].v_int64, ct.POINTER(DLTensor)).contents
    k_dl = ct.cast(args[1].v_int64, ct.POINTER(DLTensor)).contents
    v_dl = ct.cast(args[2].v_int64, ct.POINTER(DLTensor)).contents
    out_dl = ct.cast(args[3].v_int64, ct.POINTER(DLTensor)).contents

    q_shape = tuple(q_dl.shape[i] for i in range(q_dl.ndim))
    k_shape = tuple(k_dl.shape[i] for i in range(k_dl.ndim))

    q_numel = int(np.prod(q_shape))
    k_numel = int(np.prod(k_shape))

    # FlashInfer requires fp16. Copy from TRT fp32 buffers, cast, run, cast back.
    q_t = torch.empty(q_shape, device="cuda", dtype=torch.float32)
    k_t = torch.empty(k_shape, device="cuda", dtype=torch.float32)
    v_t = torch.empty(k_shape, device="cuda", dtype=torch.float32)

    cudart.cudaMemcpy(q_t.data_ptr(), q_dl.data, q_numel * 4,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    cudart.cudaMemcpy(k_t.data_ptr(), k_dl.data, k_numel * 4,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    cudart.cudaMemcpy(v_t.data_ptr(), v_dl.data, k_numel * 4,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    # Cast to fp16 for FlashInfer
    out_t = flashinfer.single_decode_with_kv_cache(
        q_t.half(), k_t.half(), v_t.half(), sm_scale=SCALE,
    ).float()

    # Copy result back to TRT output buffer
    cudart.cudaMemcpy(out_dl.data, out_t.data_ptr(), q_numel * 4,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    result[0].type_index = kTVMFFINone
    return 0


def _compute_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


_fi_callback = SAFE_CALL_TYPE(_flashinfer_decode_callback)
fn_handle = ct.c_void_p()
ret = libtvm.TVMFFIFunctionCreate(None, _fi_callback, None, ct.byref(fn_handle))
assert ret == 0
name = b"flashinfer.decode_attention"
name_arr = TVMFFIByteArray(name, len(name))
ret = libtvm.TVMFFIFunctionSetGlobal(ct.byref(name_arr), fn_handle, ct.c_int(1))
assert ret == 0
print("Registered FlashInfer decode attention kernel")


# --- Helper: check CUDA errors ---
def _check_cuda(status):
    if hasattr(cudart, "cudaError_t"):
        ok = cudart.cudaError_t.cudaSuccess
    else:
        ok = 0
    if status != ok:
        raise RuntimeError(f"CUDA error: {status}")


# --- Build TRT engine with decomposed attention ---
def build_decomposed_attention_engine():
    """Build TRT engine: Q@K^T * scale -> softmax -> @V."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.clear_flag(trt.BuilderFlag.TF32)

    # Inputs: q [num_heads, 1, head_dim], k [num_kv_heads, cache_len, head_dim], v same
    q = network.add_input("q", trt.float32, (NUM_HEADS, 1, HEAD_DIM))
    k = network.add_input("k", trt.float32, (NUM_KV_HEADS, CACHE_LEN, HEAD_DIM))
    v = network.add_input("v", trt.float32, (NUM_KV_HEADS, CACHE_LEN, HEAD_DIM))

    # GQA: expand KV heads to match Q heads
    repeats = NUM_HEADS // NUM_KV_HEADS
    if repeats > 1:
        # Tile KV: [kv_heads, cache, dim] -> [heads, cache, dim]
        # Use slice + concat
        k_slices = []
        v_slices = []
        for i in range(NUM_KV_HEADS):
            k_s = network.add_slice(k, (i, 0, 0), (1, CACHE_LEN, HEAD_DIM), (1, 1, 1))
            v_s = network.add_slice(v, (i, 0, 0), (1, CACHE_LEN, HEAD_DIM), (1, 1, 1))
            for _ in range(repeats):
                k_slices.append(k_s.get_output(0))
                v_slices.append(v_s.get_output(0))
        k_exp = network.add_concatenation(k_slices)
        k_exp.axis = 0
        v_exp = network.add_concatenation(v_slices)
        v_exp.axis = 0
        k_tensor = k_exp.get_output(0)
        v_tensor = v_exp.get_output(0)
    else:
        k_tensor = k
        v_tensor = v

    # score = Q @ K^T: [heads, 1, dim] @ [heads, dim, cache] -> [heads, 1, cache]
    score = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE,
        k_tensor, trt.MatrixOperation.TRANSPOSE,
    )

    # scale
    scale_weights = trt.Weights(np.array([SCALE], dtype=np.float32))
    scale_const = network.add_constant((1, 1, 1), scale_weights)
    scaled = network.add_elementwise(
        score.get_output(0), scale_const.get_output(0),
        trt.ElementWiseOperation.PROD,
    )

    # softmax
    softmax = network.add_softmax(scaled.get_output(0))
    softmax.axes = 1 << 2  # axis 2

    # context = softmax @ V: [heads, 1, cache] @ [heads, cache, dim] -> [heads, 1, dim]
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        v_tensor, trt.MatrixOperation.NONE,
    )

    out = context.get_output(0)
    out.name = "output"
    network.mark_output(out)

    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(plan)
    return engine


# --- Build TRT engine with FlashInfer plugin ---
def build_flashinfer_attention_engine():
    """Build TRT engine using FlashInfer via TvmFfiKernel plugin."""
    from tensorrt_model_connect.families.qwen.model.model import add_tvm_ffi_kernel

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.clear_flag(trt.BuilderFlag.TF32)

    # FlashInfer format: q [num_heads, head_dim], k [cache, kv_heads, head_dim], v same
    q = network.add_input("q", trt.float32, (NUM_HEADS, HEAD_DIM))
    k = network.add_input("k", trt.float32, (CACHE_LEN, NUM_KV_HEADS, HEAD_DIM))
    v = network.add_input("v", trt.float32, (CACHE_LEN, NUM_KV_HEADS, HEAD_DIM))

    outputs = add_tvm_ffi_kernel(
        network,
        kernel_name="flashinfer.decode_attention",
        inputs=[q, k, v],
        output_specs=[{"dims": [NUM_HEADS, HEAD_DIM], "dtype": "float32"}],
    )

    out = outputs[0]
    out.name = "output"
    network.mark_output(out)

    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    rt = trt.Runtime(logger)
    engine = rt.deserialize_cuda_engine(plan)
    return engine


# --- Run engine ---
def run_engine(engine, inputs_dict, output_name, output_shape):
    ctx = engine.create_execution_context()
    assert ctx is not None

    err, stream = cudart.cudaStreamCreate()

    device_bufs = {}
    for name, arr in inputs_dict.items():
        nbytes = arr.nbytes
        err, ptr = cudart.cudaMalloc(nbytes)
        cudart.cudaMemcpyAsync(ptr, arr.ctypes.data, nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        device_bufs[name] = ptr
        ctx.set_tensor_address(name, ptr)

    out_arr = np.zeros(output_shape, dtype=np.float32)
    err, out_ptr = cudart.cudaMalloc(out_arr.nbytes)
    device_bufs[output_name] = out_ptr
    ctx.set_tensor_address(output_name, out_ptr)

    # Warmup
    for _ in range(5):
        ctx.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    # Benchmark
    iters = 200
    start = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / iters) * 1000.0

    # Read output
    cudart.cudaMemcpyAsync(out_arr.ctypes.data, out_ptr, out_arr.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    for ptr in device_bufs.values():
        cudart.cudaFree(ptr)
    cudart.cudaStreamDestroy(stream)

    return latency_ms, out_arr


# --- Main benchmark ---
print("\nQwen3-0.6B attention dimensions:")
print(f"  num_heads={NUM_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
print(f"  cache_len={CACHE_LEN}, scale={SCALE:.4f}")

# Random inputs
np.random.seed(42)
q_decomp = np.random.randn(NUM_HEADS, 1, HEAD_DIM).astype(np.float32)
k_decomp = np.random.randn(NUM_KV_HEADS, CACHE_LEN, HEAD_DIM).astype(np.float32)
v_decomp = np.random.randn(NUM_KV_HEADS, CACHE_LEN, HEAD_DIM).astype(np.float32)

# FlashInfer format
q_fi = q_decomp.reshape(NUM_HEADS, HEAD_DIM)
k_fi = k_decomp.transpose(1, 0, 2)  # [cache, kv_heads, head_dim]
v_fi = v_decomp.transpose(1, 0, 2)

# 1. Decomposed attention
print("\nBuilding decomposed attention engine...")
engine_decomp = build_decomposed_attention_engine()
lat_decomp, out_decomp = run_engine(
    engine_decomp,
    {"q": q_decomp, "k": k_decomp, "v": v_decomp},
    "output", (NUM_HEADS, 1, HEAD_DIM),
)
print(f"  Decomposed latency: {lat_decomp:.4f} ms/step")

# 2. FlashInfer attention
print("\nBuilding FlashInfer attention engine...")
engine_fi = build_flashinfer_attention_engine()
lat_fi, out_fi = run_engine(
    engine_fi,
    {"q": q_fi, "k": k_fi, "v": v_fi},
    "output", (NUM_HEADS, HEAD_DIM),
)
print(f"  FlashInfer latency: {lat_fi:.4f} ms/step")

# Results
print("\n" + "=" * 60)
print(f"Decomposed attention: {lat_decomp:.4f} ms/step")
print(f"FlashInfer attention: {lat_fi:.4f} ms/step")
speedup = lat_decomp / lat_fi if lat_fi > 0 else 0
print(f"Speedup: {speedup:.2f}x")
print("=" * 60)

if speedup > 1.0:
    print(f"PASS: FlashInfer plugin shows {speedup:.2f}x improvement")
else:
    print(f"INFO: No speedup observed ({speedup:.2f}x)")
    print("  (Host-roundtrip kernel adds overhead; real GPU kernel would be faster)")
