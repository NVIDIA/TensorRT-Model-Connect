"""SegFormer-owned debug runner for segmentation diffing."""

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

def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for family debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for family debug_runner execution")



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


class VisionTrtRunner:
    """Single-pass TRT inference for vision encoder. Validation only.

    Deserializes a vision TRT engine and runs a single forward pass.
    No KV cache, no autoregressive loop.
    """

    def __init__(self, engine_plan: bytes):
        _require_trt_runtime()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize vision TRT engine")
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Discover IO tensors
        self._input_names = []
        self._output_names = []
        self._device_buffers: dict[str, int] = {}
        self._host_buffers: dict[str, np.ndarray] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = _trt_nptype_safe(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize

            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._device_buffers[name] = d_ptr
            self._host_buffers[name] = np.zeros(shape, dtype=dtype_np)

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def encode(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Run a single forward pass through the vision encoder.

        Args:
            **inputs: Named input arrays (e.g. patch_embeds=...).

        Returns:
            Dict of output name -> numpy array.
        """
        # Set input values
        for name, value in inputs.items():
            if name in self._host_buffers:
                self._host_buffers[name][:] = value.astype(
                    self._host_buffers[name].dtype)

        # Copy inputs to device
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            self.context.set_tensor_address(name, self._device_buffers[name])
            if mode == trt.TensorIOMode.INPUT:
                h_buf = self._host_buffers[name]
                cudart.cudaMemcpyAsync(
                    self._device_buffers[name],
                    h_buf.ctypes.data,
                    h_buf.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )

        self.context.execute_async_v3(self.stream)

        # Copy outputs
        results: dict[str, np.ndarray] = {}
        for name in self._output_names:
            h_buf = self._host_buffers[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data,
                self._device_buffers[name],
                h_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )

        cudart.cudaStreamSynchronize(self.stream)

        for name in self._output_names:
            results[name] = self._host_buffers[name].copy()

        return results

    def __del__(self):
        if cudart is None:
            return
        for d_ptr in self._device_buffers.values():
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)

