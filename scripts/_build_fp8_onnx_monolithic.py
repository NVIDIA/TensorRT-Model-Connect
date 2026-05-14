#!/usr/bin/env python3
"""Build FP8 TRT engine from ONNX with Q/DQ nodes via ONNX parser.

Key insight: the ONNX parser keeps the BF16 graph in a compact TensorRT compiler
partition. Injecting FP8 Q/DQ into the ONNX lets TensorRT fuse quantization into
MatMul parameters before backend partitioning, preserving a compact kernel plan.

The API path creates many separate kernels because explicit Cast nodes create
backend boundaries.
"""
import tensorrt as trt
import time
import os
import sys
import numpy as np

# FP8 ONNX from _inject_fp8_qdq_proto.py
ONNX_FP8 = "/tmp/flux2_fp8_onnx_proto/flux2_dit_fp8.onnx"
# BF16 ONNX baseline for comparison
ONNX_BF16 = "/tmp/flux2_dit_onnx/flux2_dit.onnx"
ENGINE_OUT = "/tmp/flux2_dit_onnx/flux2_dit_fp8_monolithic.engine"

# Choose which ONNX to build
onnx_path = ONNX_FP8
if "--bf16" in sys.argv:
    onnx_path = ONNX_BF16
    ENGINE_OUT = "/tmp/flux2_dit_onnx/flux2_dit_bf16_check.engine"
    print("Building BF16 baseline for comparison")

print(f"ONNX: {onnx_path}")
print(f"Engine: {ENGINE_OUT}")

if not os.path.exists(onnx_path):
    print(f"ERROR: {onnx_path} does not exist")
    print("Run _inject_fp8_qdq_proto.py first to create FP8 ONNX")
    sys.exit(1)

logger = trt.Logger(trt.Logger.VERBOSE if "--verbose" in sys.argv else trt.Logger.INFO)
builder = trt.Builder(logger)

# NON-STRONGLY_TYPED: Let TRT handle type inference naturally
# This is critical - STRONGLY_TYPED forces explicit type boundaries
# that fragment TensorRT compiler partitioning into many kernels.
# The ONNX parser + non-STRONGLY_TYPED compiles to 1 monolithic kernel.
flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
if "--strongly-typed" in sys.argv:
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    print("WARNING: Using STRONGLY_TYPED - this may fragment the graph!")

network = builder.create_network(flags)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)

# With STRONGLY_TYPED, types come from the graph (Q/DQ nodes declare FP8).
# Builder precision flags are not allowed.
# Without STRONGLY_TYPED, we need BF16+FP8 flags.
if "--strongly-typed" not in sys.argv:
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.FP8)

# Parse ONNX
parser = trt.OnnxParser(network, logger)
print(f"Parsing {onnx_path}...", flush=True)
t0 = time.monotonic()
success = parser.parse_from_file(onnx_path)
if not success:
    for i in range(min(parser.num_errors, 10)):
        print(f"  Parse error {i}: {parser.get_error(i)}", flush=True)
    sys.exit(1)
parse_time = time.monotonic() - t0
print(f"Parsed in {parse_time:.1f}s: {network.num_layers} layers", flush=True)

# Report layer types
layer_types = {}
for i in range(network.num_layers):
    lt = network.get_layer(i).type
    layer_types[lt] = layer_types.get(lt, 0) + 1
for lt, count in sorted(layer_types.items(), key=lambda x: -x[1])[:15]:
    print(f"  {lt}: {count}")

# Count Q/DQ layers specifically
qdq_count = sum(1 for i in range(network.num_layers)
                if network.get_layer(i).type in (trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE))
print(f"  Q/DQ layers: {qdq_count}")

# Build engine
print("\nBuilding engine (BF16+FP8, non-STRONGLY_TYPED)...", flush=True)
t0 = time.monotonic()
plan = builder.build_serialized_network(network, config)
if plan is None:
    print("BUILD FAILED!", flush=True)
    sys.exit(1)
plan_bytes = bytes(plan)
build_time = time.monotonic() - t0
print(f"Engine built in {build_time:.0f}s ({len(plan_bytes)/(1024**3):.1f} GB)", flush=True)

# Save
with open(ENGINE_OUT, "wb") as f:
    f.write(plan_bytes)
print(f"Saved: {ENGINE_OUT}")

# Inspect engine layers/profiles
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan_bytes)
print("\nEngine stats:")
print(f"  I/O tensors: {engine.num_io_tensors}")
print(f"  Layers: {engine.num_layers}")

# List I/O
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    print(f"  {'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'}: {name} {shape} {dtype}")

# Benchmark
print("\nBenchmarking...", flush=True)
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

ctx = engine.create_execution_context()
stream = cudart.cudaStreamCreate()[1]

# Allocate and fill I/O buffers
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(max(1, s) for s in engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    nb = int(np.prod(shape)) * np.dtype(dtype).itemsize
    d = cudart.cudaMalloc(nb)[1]
    h = (np.random.randn(*shape) * 0.01).astype(dtype)
    cudart.cudaMemcpyAsync(d, h.ctypes.data, nb,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    ctx.set_tensor_address(name, d)
cudart.cudaStreamSynchronize(stream)

# Warmup
for _ in range(3):
    ctx.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)

# Timed runs
ev_s = cudart.cudaEventCreate()[1]
ev_e = cudart.cudaEventCreate()[1]
N = 20
cudart.cudaEventRecord(ev_s, stream)
for _ in range(N):
    ctx.execute_async_v3(stream)
cudart.cudaEventRecord(ev_e, stream)
cudart.cudaStreamSynchronize(stream)
ms = cudart.cudaEventElapsedTime(ev_s, ev_e)[1]

per_step = ms / N
print(f"\n{'='*60}")
print(f"FP8 ONNX monolithic engine: {per_step:.1f}ms/step")
print(f"28 steps: {per_step*28/1000:.2f}s")
print(f"BF16 baseline: 185ms/step → Speedup: {185.0/per_step:.2f}x")
print(f"Engine size: {len(plan_bytes)/(1024**3):.1f} GB")
print(f"Engine layers: {engine.num_layers}")
print(f"{'='*60}")
