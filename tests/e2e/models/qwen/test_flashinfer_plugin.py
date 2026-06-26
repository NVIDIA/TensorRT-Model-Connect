#!/usr/bin/env python3
"""FlashInfer attention via TVM-FFI plugin — native kernel, zero Python callback.

JIT-compiles the FlashInfer single_decode kernel (native CUDA), registers it
as a TVM-FFI global function, then builds a TRT engine with the TvmFfiKernel
plugin. The plugin calls the FlashInfer kernel directly via TVMFFIFunctionCall
in C++ — no Python in the hot path.

Extra scalar arguments (sm_scale, rope params, etc.) are baked into shape_spec
JSON and passed by the C++ enqueue() as TVMFFIAny scalars.
"""
if __name__ != "__main__":
    import pytest

    pytest.skip(
        "FlashInfer TVM-FFI smoke script requires explicit direct execution.",
        allow_module_level=True,
    )

import ctypes as ct
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tvm_ffi
import flashinfer
import flashinfer.decode as fi_dec

NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
CACHE_LEN = 128
SCALE = 1.0 / (HEAD_DIM ** 0.5)

# ---------------------------------------------------------------------------
# 1. JIT compile FlashInfer kernel and register as TVM-FFI global function
# ---------------------------------------------------------------------------

print("JIT compiling FlashInfer single_decode kernel (native CUDA)...")
fi_module = fi_dec.gen_single_decode_module(
    torch.float16, torch.float16, torch.float16,
    HEAD_DIM, HEAD_DIM,
    pos_encoding_mode=0,  # no rope in kernel (we apply rope separately)
    use_sliding_window=False,
    use_logits_soft_cap=False,
).build_and_load()

# fi_module.run is a native tvm_ffi.Function (compiled CUDA, NOT a Python callback).
# Signature: (q, k, v, tmp, o, maybe_lse, kv_layout_code, window_left,
#              alibi_slopes, logits_soft_cap, sm_scale, rope_rcp_scale, rope_rcp_theta)
fi_run_native = fi_module.run
print(f"  fi_run type: {type(fi_run_native)}")

# Register as a TVM-FFI global function so C++ can find it via TVMFFIFunctionGetGlobal
tvm_ffi.register_global_func("flashinfer.decode_f16_d64", fi_run_native, override=True)

# Verify C++ can find it
found = tvm_ffi.get_global_func("flashinfer.decode_f16_d64")
assert found is not None, "Failed to register globally"
print("  Registered as flashinfer.decode_f16_d64")

# Pre-allocate workspace tensor (FlashInfer needs tmp buffer)
_tmp_buf = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")

# ---------------------------------------------------------------------------
# 2. Load plugin shared library
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
shared_lib = str(REPO_ROOT / "build_shared" / "libtrtmc_core.so")
if not os.path.exists(shared_lib):
    shared_lib = str(REPO_ROOT / "build" / "libtrtmc_tvm_ffi_plugin.so")
if not os.path.exists(shared_lib):
    print("SKIP: No plugin .so found")
    sys.exit(0)

lib = ct.CDLL(shared_lib, mode=ct.RTLD_GLOBAL)
lib.tvm_ffi_plugin_force_link()
print(f"Loaded plugin: {shared_lib}")

# ---------------------------------------------------------------------------
# 3. Build TRT engine
# ---------------------------------------------------------------------------

import tensorrt as trt  # noqa: E402

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
config.set_flag(trt.BuilderFlag.FP16)  # Force fp16 to match FlashInfer

# Plugin I/O:
#   Inputs:  q [num_heads, head_dim], k [cache, kv_heads, head_dim],
#            v [cache, kv_heads, head_dim], tmp [32M], o [num_heads, head_dim]
#   (tmp and o are "inputs" to the plugin because FlashInfer writes into them)
#   Outputs: o_alias [num_heads, head_dim] (copy of o after kernel writes it)
#
# Actually, for simplicity: make q, k, v as TRT inputs; tmp as a constant;
# o as the TRT output. The plugin has 3 inputs + 1 output.
# But FlashInfer's fi_run expects: q, k, v, tmp, o, lse, kv_layout, window,
#   alibi, softcap, sm_scale, rope_rcp_scale, rope_rcp_theta
# That's 5 tensors + 8 scalars = 13 args total.
#
# Plugin convention: num_inputs=3 TRT inputs, num_outputs=1 TRT output.
# Extra args via shape_spec: tmp (tensor via workspace), then scalars.
# But workspace is a void* from TRT — not a DLTensor.
#
# Simpler approach: make tmp a TRT input too (4 inputs, 1 output).
# Then extra_args = [None(lse), 0(kv_layout), -1(window), None(alibi),
#                    0.0(softcap), scale, 1.0(rope_rcp_scale), 1/10000(rope_rcp_theta)]

TMP_SIZE = 32 * 1024 * 1024  # 32 MB workspace for FlashInfer

q_in = network.add_input("q", trt.float16, (NUM_HEADS, HEAD_DIM))
k_in = network.add_input("k", trt.float16, (CACHE_LEN, NUM_KV_HEADS, HEAD_DIM))
v_in = network.add_input("v", trt.float16, (CACHE_LEN, NUM_KV_HEADS, HEAD_DIM))

# tmp buffer passed as a float16 "input" of size TMP_SIZE/2 elements
# (reinterpreted as raw bytes by FlashInfer)
tmp_elems = TMP_SIZE // 2
tmp_in = network.add_input("tmp", trt.float16, (tmp_elems,))

kernel_name = "flashinfer.decode_f16_d64"
shape_spec = json.dumps({
    "num_inputs": 4,  # q, k, v, tmp
    "num_outputs": 1,  # o
    "outputs": [{"dims": [NUM_HEADS, HEAD_DIM], "dtype": "float16"}],
    "workspace_bytes": 0,
    "extra_args": [
        {"type": "none"},             # maybe_lse = None
        {"type": "int", "value": 0},   # kv_layout_code = NHD
        {"type": "int", "value": -1},  # window_left = -1 (no sliding window)
        {"type": "none"},             # alibi_slopes = None
        {"type": "float", "value": 0.0},    # logits_soft_cap
        {"type": "float", "value": SCALE},   # sm_scale
        {"type": "float", "value": 1.0},     # rope_rcp_scale
        {"type": "float", "value": 0.0001},  # rope_rcp_theta (1/10000)
    ],
})

registry = trt.get_plugin_registry()
creator = registry.get_plugin_creator("TvmFfiKernel", "1", "")
assert creator is not None, "TvmFfiKernel creator not found"

fields = [
    trt.PluginField("kernel_name", kernel_name.encode(), trt.PluginFieldType.CHAR),
    trt.PluginField("shape_spec", shape_spec.encode(), trt.PluginFieldType.CHAR),
]
fc = trt.PluginFieldCollection(fields)
plugin = creator.create_plugin("fi_attn", fc)

layer = network.add_plugin_v2([q_in, k_in, v_in, tmp_in], plugin)
out = layer.get_output(0)
out.name = "output"
network.mark_output(out)

print("Building TRT engine with FlashInfer plugin (native, no Python callback)...")
plan = builder.build_serialized_network(network, config)
assert plan is not None, "TRT build failed"

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan)
assert engine is not None
ctx = engine.create_execution_context()
assert ctx is not None
print("Engine ready")

# ---------------------------------------------------------------------------
# 4. Run inference
# ---------------------------------------------------------------------------

from cuda.bindings import runtime as cudart  # noqa: E402

def _check(status):
    if hasattr(cudart, "cudaError_t"):
        ok = cudart.cudaError_t.cudaSuccess
    else:
        ok = 0
    if status != ok:
        raise RuntimeError(f"CUDA error: {status}")

err, stream = cudart.cudaStreamCreate()

np.random.seed(42)
q_np = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float16)
k_np = np.random.randn(CACHE_LEN, NUM_KV_HEADS, HEAD_DIM).astype(np.float16)
v_np = np.random.randn(CACHE_LEN, NUM_KV_HEADS, HEAD_DIM).astype(np.float16)
tmp_np = np.zeros(tmp_elems, dtype=np.float16)
o_np = np.zeros((NUM_HEADS, HEAD_DIM), dtype=np.float16)

bufs = {}
for name, arr in [("q", q_np), ("k", k_np), ("v", v_np), ("tmp", tmp_np), ("output", o_np)]:
    err, ptr = cudart.cudaMalloc(arr.nbytes)
    if name != "output":
        cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    ctx.set_tensor_address(name, ptr)
    bufs[name] = (ptr, arr)

# Warmup
for _ in range(5):
    ctx.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)

# Benchmark
iters = 500
cudart.cudaStreamSynchronize(stream)
t0 = time.perf_counter()
for _ in range(iters):
    ctx.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)
plugin_latency = (time.perf_counter() - t0) / iters * 1000

# Read output
cudart.cudaMemcpy(o_np.ctypes.data, bufs["output"][0], o_np.nbytes,
                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

# Verify correctness
q_t = torch.from_numpy(q_np).cuda()
k_t = torch.from_numpy(k_np).cuda()
v_t = torch.from_numpy(v_np).cuda()
ref = flashinfer.single_decode_with_kv_cache(q_t, k_t, v_t, sm_scale=SCALE)
ref_np = ref.cpu().numpy()

max_diff = np.max(np.abs(o_np.astype(np.float32) - ref_np.astype(np.float32)))

# Benchmark native FlashInfer
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    flashinfer.single_decode_with_kv_cache(q_t, k_t, v_t, sm_scale=SCALE)
torch.cuda.synchronize()
native_latency = (time.perf_counter() - t0) / iters * 1000

# Benchmark PyTorch SDPA
q_sdpa = q_t.unsqueeze(0).unsqueeze(2)
k_sdpa = k_t.unsqueeze(0).permute(0, 2, 1, 3)
v_sdpa = v_t.unsqueeze(0).permute(0, 2, 1, 3)
rep = NUM_HEADS // NUM_KV_HEADS
k_exp = k_sdpa.repeat(1, rep, 1, 1)
v_exp = v_sdpa.repeat(1, rep, 1, 1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_exp, v_exp, scale=SCALE)
torch.cuda.synchronize()
sdpa_latency = (time.perf_counter() - t0) / iters * 1000

# Cleanup
for ptr, _ in bufs.values():
    cudart.cudaFree(ptr)
cudart.cudaStreamDestroy(stream)

# Results
print(f"\n{'='*70}")
print("FlashInfer TRT Plugin — NATIVE (zero Python callback)")
print(f"  heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, cache={CACHE_LEN}")
print(f"{'='*70}")
print(f"  PyTorch SDPA:              {sdpa_latency:.4f} ms/step")
print(f"  FlashInfer (native):       {native_latency:.4f} ms/step")
print(f"  FlashInfer (TRT plugin):   {plugin_latency:.4f} ms/step")
print(f"{'='*70}")
print(f"  Plugin vs SDPA speedup:    {sdpa_latency/plugin_latency:.2f}x")
print(f"  Plugin vs native overhead: {plugin_latency/native_latency:.2f}x")
print(f"  Correctness: max_diff={max_diff:.6f} {'PASS' if max_diff < 0.01 else 'FAIL'}")
print(f"{'='*70}")
