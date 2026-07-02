# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TVM-FFI kernel bridge plugin — Python round-trip test.

Intent:
    Validates the add_tvm_ffi_kernel() graph_ops function by registering a
    trivial "add_one" kernel via tvm.ffi, building a TRT engine through the
    existing trt_runner fixture, and verifying output = input + 1.0.

Preconditions:
    - TensorRT + CUDA available (requires_trt marker)
    - tvm.ffi Python package available (requires_tvm_ffi marker)
    - TvmFfiKernel creator registered in TRT plugin registry

Postconditions:
    - Output tensor matches input + 1.0 within atol=1e-5

Trace IDs: ARCH-TVM-FFI-001, UD-TVM-FFI-GRAPHOPS-001, UT-TVM-FFI-PYTHON-001
"""

from __future__ import annotations

import numpy as np
import pytest

# --- Skip checks ---

try:
    import tvm.ffi as _tvm_ffi  # noqa: F401
    _has_tvm_ffi = True
except ImportError:
    _has_tvm_ffi = False

try:
    import tensorrt as trt
    _has_trt = True
except ImportError:
    _has_trt = False


def _tvm_ffi_plugin_registered() -> bool:
    """Check that TvmFfiKernel is in the TRT plugin registry."""
    if not _has_trt:
        return False
    try:
        registry = trt.get_plugin_registry()
        creator = registry.get_creator("TvmFfiKernel", "1", "")
        return creator is not None
    except Exception:
        return False


requires_tvm_ffi = pytest.mark.skipif(
    not (_has_tvm_ffi and _has_trt and _tvm_ffi_plugin_registered()),
    reason="TVM-FFI + TRT + TvmFfiKernel plugin not available",
)


# --- Tests ---

@requires_tvm_ffi
def test_add_one_roundtrip(trt_runner):
    """Build TRT engine with TVM-FFI add_one kernel, verify output."""
    import tvm.ffi
    from tests.builder.owned_graph_modules import load_graph_ops
    add_tvm_ffi_kernel = load_graph_ops().add_tvm_ffi_kernel

    # Register trivial add_one kernel
    @tvm.ffi.register_func("tvm_ffi_test.py_add_one")
    def py_add_one(inp, out):
        """Host-roundtrip add_one: copy to numpy, add 1, copy back."""
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]

        numel = 1
        for i in range(inp.ndim):
            numel *= inp.shape[i]
        nbytes = numel * 4  # float32

        host = np.empty(numel, dtype=np.float32)
        cudart.cudaMemcpy(
            host.ctypes.data, inp.data_ptr, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        host += 1.0

        cudart.cudaMemcpy(
            out.data_ptr, host.ctypes.data, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    def build_fn(network, trt_inputs):
        inp = trt_inputs["x"]
        outputs = add_tvm_ffi_kernel(
            network,
            kernel_name="tvm_ffi_test.py_add_one",
            inputs=[inp],
            output_specs=[{"dims": "same_as_input_0", "dtype": "float32"}],
        )
        return {"y": outputs[0]}

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = trt_runner(build_fn, {"x": x})
    expected = x + 1.0

    np.testing.assert_allclose(result["y"].flatten(), expected, atol=1e-5)


@requires_tvm_ffi
def test_add_one_multi_shape(trt_runner):
    """Verify add_one works with a 2D input shape."""
    import tvm.ffi
    from tests.builder.owned_graph_modules import load_graph_ops
    add_tvm_ffi_kernel = load_graph_ops().add_tvm_ffi_kernel

    # Kernel already registered from test above (same process),
    # but register again to be safe (idempotent).
    @tvm.ffi.register_func("tvm_ffi_test.py_add_one", override=True)
    def py_add_one(inp, out):
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]

        numel = 1
        for i in range(inp.ndim):
            numel *= inp.shape[i]
        nbytes = numel * 4

        host = np.empty(numel, dtype=np.float32)
        cudart.cudaMemcpy(
            host.ctypes.data, inp.data_ptr, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        host += 1.0
        cudart.cudaMemcpy(
            out.data_ptr, host.ctypes.data, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    def build_fn(network, trt_inputs):
        inp = trt_inputs["x"]
        outputs = add_tvm_ffi_kernel(
            network,
            kernel_name="tvm_ffi_test.py_add_one",
            inputs=[inp],
            output_specs=[{"dims": "same_as_input_0", "dtype": "float32"}],
        )
        return {"y": outputs[0]}

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = trt_runner(build_fn, {"x": x})
    expected = x + 1.0

    np.testing.assert_allclose(result["y"], expected, atol=1e-5)
