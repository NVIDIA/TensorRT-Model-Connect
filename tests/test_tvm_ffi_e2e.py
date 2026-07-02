#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: TVM-FFI plugin round-trip via shared library.

Loads libtrtmc_core.so to register the TvmFfiKernel plugin, registers a
trivial add_one kernel via tvm_ffi, builds a TRT engine, and verifies
that input [1,2,3,4] -> output [2,3,4,5].
"""
import ctypes
import json
import os
import sys

import numpy as np

# 1. Load tvm_ffi FIRST so its symbols are global
import tvm_ffi  # noqa: F401

# 2. Register the add_one kernel
# Register add_one kernel via C API (not Python) to avoid DLTensor wrapping issues.
# The C++ plugin passes raw DLTensor* as kTVMFFIDLTensorPtr args, which tvm_ffi
# wraps as tvm_ffi.Tensor objects. But these wrap stack-allocated DLTensors that
# become invalid during DLPack operations. Instead, we register using the C API
# which receives raw TVMFFIAny* args and can access DLTensor* directly.
import ctypes as ct

libtvm = ct.CDLL("libtvm_ffi.so", mode=ct.RTLD_GLOBAL)

# C callback signature: int(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
SAFE_CALL_TYPE = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p)

# DLTensor struct for ctypes
class DLDevice(ct.Structure):
    _fields_ = [("device_type", ct.c_int32), ("device_id", ct.c_int32)]

class DLDataType(ct.Structure):
    _fields_ = [("code", ct.c_uint8), ("bits", ct.c_uint8), ("lanes", ct.c_uint16)]

class DLTensor(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("device", DLDevice),
        ("ndim", ct.c_int32),
        ("dtype", DLDataType),
        ("shape", ct.POINTER(ct.c_int64)),
        ("strides", ct.POINTER(ct.c_int64)),
        ("byte_offset", ct.c_uint64),
    ]

# TVMFFIAny struct (simplified)
class TVMFFIAny(ct.Structure):
    _fields_ = [
        ("type_index", ct.c_int32),
        ("padding", ct.c_int32),
        # Union - we access v_ptr for DLTensor*
        ("v_int64", ct.c_int64),
    ]

kTVMFFIDLTensorPtr = 7
kTVMFFINone = 0

def _add_one_callback(handle, args_ptr, num_args, result_ptr):
    """C-level add_one kernel: reads raw DLTensor pointers from TVMFFIAny args."""
    from cuda.bindings import runtime as cudart

    args = ct.cast(args_ptr, ct.POINTER(TVMFFIAny))
    result = ct.cast(result_ptr, ct.POINTER(TVMFFIAny))

    # args[0] = input DLTensor*, args[1] = output DLTensor*
    inp_dl = ct.cast(args[0].v_int64, ct.POINTER(DLTensor)).contents
    out_dl = ct.cast(args[1].v_int64, ct.POINTER(DLTensor)).contents

    numel = 1
    for i in range(inp_dl.ndim):
        numel *= inp_dl.shape[i]
    nbytes = numel * 4  # float32

    host = np.empty(numel, dtype=np.float32)
    cudart.cudaMemcpy(host.ctypes.data, inp_dl.data, nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    host += 1.0
    cudart.cudaMemcpy(out_dl.data, host.ctypes.data, nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    result[0].type_index = kTVMFFINone
    return 0

# Keep reference alive
_callback = SAFE_CALL_TYPE(_add_one_callback)

# Register via TVM-FFI C API
class TVMFFIByteArray(ct.Structure):
    _fields_ = [("data", ct.c_char_p), ("size", ct.c_int64)]

fn_handle = ct.c_void_p()
ret = libtvm.TVMFFIFunctionCreate(None, _callback, None, ct.byref(fn_handle))
assert ret == 0, f"TVMFFIFunctionCreate failed: {ret}"

name = b"tvm_ffi_test.add_one_e2e"
name_arr = TVMFFIByteArray(name, len(name))
ret = libtvm.TVMFFIFunctionSetGlobal(ct.byref(name_arr), fn_handle, ct.c_int(1))
assert ret == 0, f"TVMFFIFunctionSetGlobal failed: {ret}"
print("Registered add_one kernel via C API")

# 3. Load our shared library (uses already-loaded libtvm_ffi symbols)
shared_lib = os.path.join(os.path.dirname(__file__), "..", "build_shared", "libtrtmc_core.so")
if not os.path.exists(shared_lib):
    print(f"SKIP: {shared_lib} not found (build with BUILD_SHARED_LIBS=ON)")
    sys.exit(0)

lib = ctypes.CDLL(shared_lib, mode=ctypes.RTLD_GLOBAL)
lib.tvm_ffi_plugin_force_link()

# 4. Build TRT engine with the plugin
import tensorrt as trt  # noqa: E402

registry = trt.get_plugin_registry()
creator = registry.get_creator("TvmFfiKernel", "1", "")
if creator is None:
    print("FAIL: TvmFfiKernel creator not found")
    sys.exit(1)

kernel_name = "tvm_ffi_test.add_one_e2e"
shape_spec = json.dumps({
    "num_inputs": 1,
    "num_outputs": 1,
    "outputs": [{"dims": "same_as_input_0", "dtype": "float32"}],
    "workspace_bytes": 0,
})

fields = [
    trt.PluginField("kernel_name", kernel_name.encode(), trt.PluginFieldType.CHAR),
    trt.PluginField("shape_spec", shape_spec.encode(), trt.PluginFieldType.CHAR),
]
fc = trt.PluginFieldCollection(fields)
plugin = creator.create_plugin("tvm_ffi_add_one", fc, trt.TensorRTPhase.BUILD)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)

inp_tensor = network.add_input("input", trt.float32, (4,))
layer = network.add_plugin_v3([inp_tensor], [], plugin)
out_tensor = layer.get_output(0)
out_tensor.name = "output"
network.mark_output(out_tensor)

print("Building TRT engine...")
plan = builder.build_serialized_network(network, config)
assert plan is not None, "TRT build failed"

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan)
assert engine is not None, "Engine deserialization failed"

ctx = engine.create_execution_context()
assert ctx is not None, "createExecutionContext failed"

# 5. Run inference
from cuda.bindings import runtime as cudart  # noqa: E402

err, stream = cudart.cudaStreamCreate()

h_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
h_output = np.zeros(4, dtype=np.float32)

err, d_in = cudart.cudaMalloc(h_input.nbytes)
err, d_out = cudart.cudaMalloc(h_output.nbytes)
cudart.cudaMemcpy(d_in, h_input.ctypes.data, h_input.nbytes,
                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

ctx.set_tensor_address("input", d_in)
ctx.set_tensor_address("output", d_out)
ok = ctx.execute_async_v3(stream)

cudart.cudaMemcpy(h_output.ctypes.data, d_out, h_output.nbytes,
                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
cudart.cudaStreamSynchronize(stream)

cudart.cudaFree(d_in)
cudart.cudaFree(d_out)
cudart.cudaStreamDestroy(stream)

# 6. Verify
expected = h_input + 1.0
print(f"Input:    {h_input}")
print(f"Output:   {h_output}")
print(f"Expected: {expected}")

if np.allclose(h_output, expected, atol=1e-5):
    print("PASS: TVM-FFI plugin round-trip succeeded!")
    sys.exit(0)
else:
    print(f"FAIL: Output mismatch! Execute returned: {ok}")
    sys.exit(1)
