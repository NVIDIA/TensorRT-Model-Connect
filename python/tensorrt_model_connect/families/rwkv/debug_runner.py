"""Family-owned debug runner implementation."""

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
    if cudart is None:
        raise RuntimeError("cuda-python is required for debug_runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")


def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for debug_runner execution")


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


class RwkvTrtRunner:
    """Device-resident RWKV TRT inference runner.

    Keeps 5 state tensors per layer on-device:
      attn_state, ff_state, num_state, den_state, max_state
    Only transfers token_id (H2D) and logits (D2H) per step.
    """

    def __init__(
        self,
        engine_plan: bytes,
        num_layers: int,
        distributed_communicator: object | None = None,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers
        self._distributed_communicator = distributed_communicator

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()
        if distributed_communicator is not None:
            set_communicator = getattr(self.context, "set_communicator", None)
            if set_communicator is None:
                raise RuntimeError(
                    "TensorRT distributed execution requires TRT 11.0+ "
                    "IExecutionContext.set_communicator"
                )
            if not set_communicator(distributed_communicator):
                raise RuntimeError("Failed to set TRT distributed communicator")

        # Auto-detect hidden_size from attn_state_0 shape [1, hidden]
        state_shape = tuple(self.engine.get_tensor_shape("attn_state_0"))
        self.hidden_size = state_shape[1] if len(state_shape) == 2 else state_shape[0]
        state_bytes = self.hidden_size * 4

        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # Per-layer device state buffers (5 per layer)
        self._d_attn = []
        self._d_ff = []
        self._d_num = []
        self._d_den = []
        self._d_max = []
        self._d_p_attn = []
        self._d_p_ff = []
        self._d_p_num = []
        self._d_p_den = []
        self._d_p_max = []
        for _ in range(num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den,
                        self._d_max, self._d_p_attn, self._d_p_ff,
                        self._d_p_num, self._d_p_den, self._d_p_max]:
                err, ptr = cudart.cudaMalloc(state_bytes)
                _check_cuda(err)
                lst.append(ptr)

        self._h_token_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # Discover debug outputs
        self._debug_output_names = []
        self._output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self._output_shapes[name] = shape
                if (name != "logits"
                        and not name.startswith("present_attn_")
                        and not name.startswith("present_ff_")
                        and not name.startswith("present_num_")
                        and not name.startswith("present_den_")
                        and not name.startswith("present_max_")):
                    self._debug_output_names.append(name)

        self._d_debug = {}
        self._h_debug = {}
        for name in self._debug_output_names:
            shape = self._output_shapes[name]
            nbytes = int(np.prod(shape)) * 4
            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._d_debug[name] = d_ptr
            self._h_debug[name] = np.zeros(shape, dtype=np.float32)

        # Zero-init all states (max_state should be -1e38 for numerical stability)
        for i in range(num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den]:
                _check_cuda(cudart.cudaMemsetAsync(lst[i], 0, state_bytes, self.stream)[0])
            # max_state init: -1e38
            h_neg_inf = np.full(self.hidden_size, -1e38, dtype=np.float32)
            cudart.cudaMemcpyAsync(
                self._d_max[i], h_neg_inf.ctypes.data, state_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream
        state_bytes = self.hidden_size * 4

        self._h_token_id[0] = token_id
        cudart.cudaMemcpyAsync(self._d_token_id, self._h_token_id.ctypes.data, 4, H2D, stream)

        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("logits", self._d_logits)

        state_map = [
            ("attn_state", self._d_attn, "present_attn", self._d_p_attn),
            ("ff_state", self._d_ff, "present_ff", self._d_p_ff),
            ("num_state", self._d_num, "present_num", self._d_p_num),
            ("den_state", self._d_den, "present_den", self._d_p_den),
            ("max_state", self._d_max, "present_max", self._d_p_max),
        ]
        for i in range(self.num_layers):
            for in_stem, in_lst, out_stem, out_lst in state_map:
                self.context.set_tensor_address(f"{in_stem}_{i}", in_lst[i])
                self.context.set_tensor_address(f"{out_stem}_{i}", out_lst[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        self.context.execute_async_v3(stream)

        # D2D state update
        for i in range(self.num_layers):
            for _, in_lst, _, out_lst in state_map:
                cudart.cudaMemcpyAsync(in_lst[i], out_lst[i], state_bytes, D2D, stream)

        cudart.cudaMemcpyAsync(
            self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        for name in self._debug_output_names:
            h = self._h_debug[name]
            cudart.cudaMemcpyAsync(h.ctypes.data, self._d_debug[name], h.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)
        results = {"logits": self._h_logits.copy()}
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        state_bytes = self.hidden_size * 4
        for i in range(self.num_layers):
            for lst in [self._d_attn, self._d_ff, self._d_num, self._d_den]:
                _check_cuda(cudart.cudaMemsetAsync(lst[i], 0, state_bytes, self.stream)[0])
            h_neg_inf = np.full(self.hidden_size, -1e38, dtype=np.float32)
            cudart.cudaMemcpyAsync(
                self._d_max[i], h_neg_inf.ctypes.data, state_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[dict[str, np.ndarray]]:
        all_results = []
        for tid in input_ids:
            all_results.append(self.step(tid))
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(all_results[-1]["logits"].flatten()))
            all_results.append(self.step(next_token))
        return all_results

    def __del__(self):
        if cudart is None:
            return
        if not hasattr(self, "_d_token_id"):
            return
        bufs = [
            getattr(self, "_d_token_id", None),
            getattr(self, "_d_logits", None),
        ]
        for lst in [
            getattr(self, "_d_attn", []),
            getattr(self, "_d_ff", []),
            getattr(self, "_d_num", []),
            getattr(self, "_d_den", []),
            getattr(self, "_d_max", []),
            getattr(self, "_d_p_attn", []),
            getattr(self, "_d_p_ff", []),
            getattr(self, "_d_p_num", []),
            getattr(self, "_d_p_den", []),
            getattr(self, "_d_p_max", []),
        ]:
            bufs.extend(lst)
        for d_ptr in getattr(self, "_d_debug", {}).values():
            bufs.append(d_ptr)
        for d_ptr in bufs:
            if d_ptr:
                cudart.cudaFree(d_ptr)
        self._d_token_id = None
        self._d_logits = None
        for attr in (
            "_d_attn",
            "_d_ff",
            "_d_num",
            "_d_den",
            "_d_max",
            "_d_p_attn",
            "_d_p_ff",
            "_d_p_num",
            "_d_p_den",
            "_d_p_max",
        ):
            setattr(self, attr, [])
        self._d_debug = {}
        stream = getattr(self, "stream", None)
        if stream:
            cudart.cudaStreamDestroy(stream)
            self.stream = None

def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> RwkvTrtRunner:
    del runtime_strategy, bundle_path
    return RwkvTrtRunner(
        engine_plan=engine_plan,
        num_layers=header.get("num_layers", config.get("num_hidden_layers", 1)),
        distributed_communicator=distributed_communicator,
    )
