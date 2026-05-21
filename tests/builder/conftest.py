"""Shared fixtures and markers for tensorrt_model_connect tests."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


_PKG_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


# ---------------------------------------------------------------------------
# Bundle round-trip helper (shared across bundle_writer / schema tests)
# ---------------------------------------------------------------------------

def read_trtfb_bundle(path: str | Path) -> tuple[dict, dict[str, bytes]]:
    """Parse a .trtfb file into (header_dict, sections_data).

    Verifies BUNDLE_MAGIC, reads the JSON header, then seeks to each
    section payload by offset/size. Used by tests that need to inspect
    bundle contents without depending on a C++ reader.
    """
    from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC

    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == BUNDLE_MAGIC, f"bad magic: {magic!r}"
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 16 + header_len
        sections_data: dict[str, bytes] = {}
        for name, meta in header.get("sections", {}).items():
            f.seek(data_start + meta["offset"])
            sections_data[name] = f.read(meta["size"])
    return header, sections_data


# ---------------------------------------------------------------------------
# tensorrt_model_connect package availability check
# ---------------------------------------------------------------------------

def _tensorrt_model_connect_importable() -> bool:
    """Check if tensorrt_model_connect can be imported (requires tensorrt in the chain)."""
    try:
        import tensorrt_model_connect  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TRT availability check (for engine execution)
# ---------------------------------------------------------------------------

def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        try:
            from cuda.bindings import runtime as cudart  # noqa: F401
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_trt_skipif(condition: bool, reason: str):
    def decorator(obj):
        obj = pytest.mark.skipif(condition, reason=reason)(obj)
        obj = pytest.mark.gpu(obj)
        obj = pytest.mark.trt(obj)
        return obj
    return decorator


requires_trt = _gpu_trt_skipif(
    not _trt_available(), "TensorRT + CUDA not available"
)

requires_tensorrt_model_connect = pytest.mark.skipif(
    not _tensorrt_model_connect_importable(),
    reason="tensorrt_model_connect not importable (TensorRT not installed)"
)


# ---------------------------------------------------------------------------
# TVM-FFI availability check (for TVM-FFI kernel bridge tests)
# ---------------------------------------------------------------------------

def _tvm_ffi_available() -> bool:
    """Check if tvm.ffi is importable and TvmFfiKernel plugin is registered."""
    try:
        import tvm.ffi  # noqa: F401
        import tensorrt as trt
        registry = trt.get_plugin_registry()
        creator = registry.get_creator("TvmFfiKernel", "1", "")
        return creator is not None
    except (ImportError, Exception):
        return False


requires_tvm_ffi = _gpu_trt_skipif(
    not _tvm_ffi_available(),
    "TVM-FFI + TvmFfiKernel TRT plugin not available",
)


# ---------------------------------------------------------------------------
# TRT engine runner fixture
# ---------------------------------------------------------------------------

def _check_cuda(status):
    """Raise on CUDA error."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]
    if hasattr(cudart, "cudaError_t"):
        ok = cudart.cudaError_t.cudaSuccess
    else:
        ok = 0
    if status != ok:
        raise RuntimeError(f"CUDA error: {status}")


def run_trt_graph(build_fn, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build a TRT engine from build_fn, feed inputs, return outputs.

    build_fn(network, trt_inputs) -> dict[str, ITensor]
    """
    import tensorrt as trt
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.clear_flag(trt.BuilderFlag.TF32)

    trt_inputs = {}
    for name, arr in inputs.items():
        dt = trt.float32 if arr.dtype == np.float32 else trt.int32
        t = network.add_input(name, dt, tuple(arr.shape))
        trt_inputs[name] = t

    trt_outputs = build_fn(network, trt_inputs)

    for name, tensor in trt_outputs.items():
        tensor.name = name
        network.mark_output(tensor)
        tensor.dtype = trt.float32

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    err, stream = cudart.cudaStreamCreate()
    _check_cuda(err)

    device_bufs = {}
    host_out = {}
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(tname))
        nbytes = int(np.prod(shape)) * 4
        err, ptr = cudart.cudaMallocAsync(nbytes, stream)
        _check_cuda(err)
        device_bufs[tname] = ptr
        mode = engine.get_tensor_mode(tname)
        if mode == trt.TensorIOMode.INPUT:
            arr = inputs[tname]
            cudart.cudaMemcpyAsync(
                ptr, arr.ctypes.data, nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        else:
            host_out[tname] = np.zeros(shape, dtype=np.float32)
        ctx.set_tensor_address(tname, ptr)

    ctx.execute_async_v3(stream)

    for name, arr in host_out.items():
        cudart.cudaMemcpyAsync(
            arr.ctypes.data, device_bufs[name], arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)
    for ptr in device_bufs.values():
        cudart.cudaFreeAsync(ptr, stream)
    cudart.cudaStreamDestroy(stream)

    return host_out


@pytest.fixture
def trt_runner():
    """Fixture providing the run_trt_graph helper."""
    return run_trt_graph
