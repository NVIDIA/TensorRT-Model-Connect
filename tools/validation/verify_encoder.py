#!/usr/bin/env python3
"""Verify MagpieTTS TRT encoder output against NeMo reference.

Usage (inside trtmc-magpie container):
    source .venv/bin/activate
    python3 tools/validation/verify_encoder.py
"""

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CUDA helpers via ctypes (no cuda-python or pycuda needed)
# ---------------------------------------------------------------------------

_cudart = ctypes.CDLL("libcudart.so")

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    ret = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    assert ret == 0, f"cudaMalloc failed: {ret}"
    return ptr

def cuda_free(ptr):
    _cudart.cudaFree(ptr)

def cuda_memcpy_h2d(dst, src_np):
    ret = _cudart.cudaMemcpy(
        dst, src_np.ctypes.data, ctypes.c_size_t(src_np.nbytes),
        ctypes.c_int(1))  # cudaMemcpyHostToDevice = 1
    assert ret == 0, f"cudaMemcpy H2D failed: {ret}"

def cuda_memcpy_d2h(dst_np, src):
    ret = _cudart.cudaMemcpy(
        dst_np.ctypes.data, src, ctypes.c_size_t(dst_np.nbytes),
        ctypes.c_int(2))  # cudaMemcpyDeviceToHost = 2
    assert ret == 0, f"cudaMemcpy D2H failed: {ret}"

def cuda_stream_create():
    stream = ctypes.c_void_p()
    ret = _cudart.cudaStreamCreate(ctypes.byref(stream))
    assert ret == 0, f"cudaStreamCreate failed: {ret}"
    return stream

def cuda_stream_sync(stream):
    ret = _cudart.cudaStreamSynchronize(stream)
    assert ret == 0, f"cudaStreamSynchronize failed: {ret}"


# ---------------------------------------------------------------------------
# 1. Load reference data
# ---------------------------------------------------------------------------

ref_tokens = np.load("/tmp/ref_text_tokens.npy").flatten().astype(np.int64)
ref_output = np.load("/tmp/ref_encoder_output.npy")  # [1, seq_len, 768]
if ref_output.ndim == 3:
    ref_output = ref_output[0]  # -> [seq_len, 768]

seq_len = ref_tokens.shape[0]
print(f"Reference: {seq_len} tokens, output shape {ref_output.shape}")

# ---------------------------------------------------------------------------
# 2. Load plugin and weights from NeMo archive
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tensorrt_model_connect"))
from tensorrt_model_connect.families.magpie_tts import MagpieTTSPlugin
from tensorrt_model_connect.config import ModelConfig

plugin = MagpieTTSPlugin()

# Create minimal ModelConfig
config = ModelConfig.__new__(ModelConfig)
config.hidden_size = 768
config.num_attention_heads = 12
config.num_hidden_layers = 6
config.intermediate_size = 3072
config.vocab_size = 2380
config.model_type = "magpie_tts"

print("Loading weights from NeMo archive ...")
t0 = time.time()
weights = plugin.load_weights("/tmp/magpie_tts", config)
print(f"  Loaded in {time.time() - t0:.1f}s")

hidden = weights["_hidden_size"]
max_pos = weights["_max_source_positions"]
print(f"  hidden={hidden}, max_source_positions={max_pos}, "
      f"enc_layers={weights['_enc_layers']}, enc_heads={weights['_enc_heads']}")

# ---------------------------------------------------------------------------
# 3. Build encoder TRT engine
# ---------------------------------------------------------------------------

print("Building encoder TRT engine ...")
t0 = time.time()
engine_plan = plugin.build_vision_engine(
    model_dir="/tmp/magpie_tts", config=config, weights=weights, verbose=False)
print(f"  Built in {time.time() - t0:.1f}s, plan size={len(engine_plan)} bytes")

# ---------------------------------------------------------------------------
# 4. Run inference with TensorRT
# ---------------------------------------------------------------------------

import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_plan)
context = engine.create_execution_context()

# Prepare input: pad tokens to max_source_positions with zeros
input_ids = np.zeros(max_pos, dtype=np.int32)
input_ids[:seq_len] = ref_tokens.astype(np.int32)

# Allocate output buffer
output_buf = np.zeros((max_pos, hidden), dtype=np.float32)

# Find tensor info
num_io = engine.num_io_tensors
tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
print(f"  Engine tensors: {tensor_names}")

# Allocate device memory and copy input
d_input = cuda_malloc(input_ids.nbytes)
d_output = cuda_malloc(output_buf.nbytes)
cuda_memcpy_h2d(d_input, np.ascontiguousarray(input_ids))

# Set tensor addresses
context.set_tensor_address("input_ids", d_input.value)
context.set_tensor_address("encoder_output", d_output.value)

# Create stream and execute
stream = cuda_stream_create()

print("Running encoder inference ...")
t0 = time.time()
ok = context.execute_async_v3(stream_handle=stream.value)
cuda_stream_sync(stream)
elapsed = time.time() - t0
print(f"  Inference completed in {elapsed*1000:.1f}ms, success={ok}")

if not ok:
    print("ERROR: TRT inference failed!")
    sys.exit(1)

# Copy output back
cuda_memcpy_d2h(output_buf, d_output)

# Free device memory
cuda_free(d_input)
cuda_free(d_output)

# ---------------------------------------------------------------------------
# 5. Compare against reference
# ---------------------------------------------------------------------------

# Extract only the valid sequence positions (first seq_len positions)
trt_output = output_buf[:seq_len]  # [seq_len, hidden]
print(f"\nTRT output shape: {trt_output.shape}")
print(f"TRT output range: [{trt_output.min():.4f}, {trt_output.max():.4f}]")
print(f"Ref output shape: {ref_output.shape}")
print(f"Ref output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")

# Cosine similarity
def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Overall cosine similarity (entire tensor)
overall_cos = cosine_sim(trt_output, ref_output)

# Per-position cosine similarity
per_pos_cos = []
for i in range(seq_len):
    per_pos_cos.append(cosine_sim(trt_output[i], ref_output[i]))
per_pos_cos = np.array(per_pos_cos)

# Absolute error
abs_error = np.abs(trt_output - ref_output)
max_abs_err = abs_error.max()
mean_abs_err = abs_error.mean()

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Overall cosine similarity: {overall_cos:.6f}")
print(f"Per-position cosine sim:   min={per_pos_cos.min():.6f}, "
      f"mean={per_pos_cos.mean():.6f}, max={per_pos_cos.max():.6f}")
print(f"Max absolute error:        {max_abs_err:.6f}")
print(f"Mean absolute error:       {mean_abs_err:.6f}")
print(f"{'='*60}")

if overall_cos > 0.999:
    print("PASS: cosine similarity > 0.999")
    sys.exit(0)
else:
    print(f"FAIL: cosine similarity {overall_cos:.6f} <= 0.999")
    sys.exit(1)
