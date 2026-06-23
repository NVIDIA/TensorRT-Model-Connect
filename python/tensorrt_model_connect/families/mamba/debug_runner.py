"""Family-owned debug runner implementation."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.debug_runner import (
    _trt_nptype_safe,
    cudart,
    trt,
)


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


class MambaTrtRunner:
    """Device-resident Mamba/SSM TRT inference runner.

    Keeps conv_state and ssm_state on-device. Only transfers token_id
    (H2D, 4 B) and logits + debug outputs (D2H) per step. State updates
    are D2D memcpy. Matches the C++ MambaBackend behavior exactly.
    """

    def __init__(
        self,
        engine_plan: bytes,
        num_layers: int,
        d_inner: int | None = None,
        state_size: int | None = None,
        conv_kernel: int | None = None,
        distributed_communicator: object | None = None,
    ):
        _require_trt_runtime()
        self.num_layers = num_layers
        self._distributed_communicator = distributed_communicator

        # Deserialize engine
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
                    "TensorRT distributed Mamba debug execution requires "
                    "IExecutionContext.set_communicator"
                )
            if not set_communicator(distributed_communicator):
                raise RuntimeError("Failed to set TensorRT distributed communicator")

        # Auto-detect state dimensions
        if d_inner is None or conv_kernel is None:
            conv_shape = tuple(self.engine.get_tensor_shape("conv_state_0"))
            d_inner = conv_shape[0]
            conv_kernel = conv_shape[1]
        if state_size is None:
            ssm_shape = tuple(self.engine.get_tensor_shape("ssm_state_0"))
            state_size = ssm_shape[1]

        self.d_inner = d_inner
        self.state_size = state_size
        self.conv_kernel = conv_kernel

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        conv_state_bytes = d_inner * conv_kernel * 4
        ssm_state_bytes = d_inner * state_size * 4

        # Discover debug/extra output tensor names
        self._output_names: list[str] = []
        self._output_shapes: dict[str, tuple] = {}
        self._debug_output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self._output_names.append(name)
                self._output_shapes[name] = shape
                if (name != "logits"
                        and not name.startswith("present_conv_")
                        and not name.startswith("present_ssm_")):
                    self._debug_output_names.append(name)

        # --- Persistent device state buffers ---
        self._d_conv_state: list[int] = []
        self._d_ssm_state: list[int] = []
        for _ in range(num_layers):
            err, dc = cudart.cudaMalloc(conv_state_bytes)
            _check_cuda(err)
            self._d_conv_state.append(dc)
            err, ds = cudart.cudaMalloc(ssm_state_bytes)
            _check_cuda(err)
            self._d_ssm_state.append(ds)

        # --- Device buffers for present outputs ---
        self._d_present_conv: list[int] = []
        self._d_present_ssm: list[int] = []
        for _ in range(num_layers):
            err, pc = cudart.cudaMalloc(conv_state_bytes)
            _check_cuda(err)
            self._d_present_conv.append(pc)
            err, ps = cudart.cudaMalloc(ssm_state_bytes)
            _check_cuda(err)
            self._d_present_ssm.append(ps)

        # --- Small I/O ---
        self._h_token_id = np.zeros((1,), dtype=np.int32)
        err, self._d_token_id = cudart.cudaMalloc(4)
        _check_cuda(err)

        logits_shape = tuple(self.engine.get_tensor_shape("logits"))
        self._logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(logits_shape, dtype=np.float32)
        err, self._d_logits = cudart.cudaMalloc(self._logits_numel * 4)
        _check_cuda(err)

        # Debug output device/host buffers
        self._d_debug: dict[str, int] = {}
        self._h_debug: dict[str, np.ndarray] = {}
        for name in self._debug_output_names:
            shape = self._output_shapes[name]
            dtype_trt = self.engine.get_tensor_dtype(name)
            dtype_np = _trt_nptype_safe(dtype_trt)
            nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize
            err, d_ptr = cudart.cudaMalloc(nbytes)
            _check_cuda(err)
            self._d_debug[name] = d_ptr
            self._h_debug[name] = np.zeros(shape, dtype=dtype_np)

        # Zero-init device state
        for i in range(num_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, ssm_state_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def step(self, token_id: int) -> dict[str, np.ndarray]:
        """Run one Mamba decode step.

        Args:
            token_id: Input token ID.

        Returns:
            Dict with 'logits' and any debug outputs.
        """
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        stream = self.stream

        self._h_token_id[0] = token_id
        cudart.cudaMemcpyAsync(
            self._d_token_id, self._h_token_id.ctypes.data,
            4, H2D, stream)

        # Set tensor addresses
        self.context.set_tensor_address("token_id", self._d_token_id)
        self.context.set_tensor_address("logits", self._d_logits)

        conv_state_bytes = self.d_inner * self.conv_kernel * 4
        ssm_state_bytes = self.d_inner * self.state_size * 4

        for i in range(self.num_layers):
            self.context.set_tensor_address(
                f"conv_state_{i}", self._d_conv_state[i])
            self.context.set_tensor_address(
                f"ssm_state_{i}", self._d_ssm_state[i])
            self.context.set_tensor_address(
                f"present_conv_{i}", self._d_present_conv[i])
            self.context.set_tensor_address(
                f"present_ssm_{i}", self._d_present_ssm[i])

        for name in self._debug_output_names:
            self.context.set_tensor_address(name, self._d_debug[name])

        # Execute
        self.context.execute_async_v3(stream)

        # D2D state update (direct replacement — Mamba state is overwritten)
        for i in range(self.num_layers):
            cudart.cudaMemcpyAsync(
                self._d_conv_state[i], self._d_present_conv[i],
                conv_state_bytes, D2D, stream)
            cudart.cudaMemcpyAsync(
                self._d_ssm_state[i], self._d_present_ssm[i],
                ssm_state_bytes, D2D, stream)

        # D2H: logits + debug outputs
        cudart.cudaMemcpyAsync(
            self._h_logits.ctypes.data, self._d_logits,
            self._logits_numel * 4, D2H, stream)
        for name in self._debug_output_names:
            h_buf = self._h_debug[name]
            cudart.cudaMemcpyAsync(
                h_buf.ctypes.data, self._d_debug[name],
                h_buf.nbytes, D2H, stream)

        cudart.cudaStreamSynchronize(stream)

        # Collect results
        results: dict[str, np.ndarray] = {
            "logits": self._h_logits.copy(),
        }
        for name in self._debug_output_names:
            results[name] = self._h_debug[name].copy()
        return results

    def reset(self):
        """Zero all device state buffers."""
        conv_state_bytes = self.d_inner * self.conv_kernel * 4
        ssm_state_bytes = self.d_inner * self.state_size * 4
        for i in range(self.num_layers):
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_conv_state[i], 0, conv_state_bytes, self.stream)[0])
            _check_cuda(cudart.cudaMemsetAsync(
                self._d_ssm_state[i], 0, ssm_state_bytes, self.stream)[0])
        cudart.cudaStreamSynchronize(self.stream)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> list[dict[str, np.ndarray]]:
        """Run autoregressive generation with Mamba.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Number of tokens to generate after the prompt.

        Returns:
            List of per-step result dicts.
        """
        all_results = []

        # Prefill: process input tokens one by one (Mamba is recurrent)
        for tid in input_ids:
            result = self.step(tid)
            all_results.append(result)

        # Generate: autoregressive decoding
        for _ in range(max_new_tokens):
            last_logits = all_results[-1]["logits"].flatten()
            next_token = int(np.argmax(last_logits))
            result = self.step(next_token)
            all_results.append(result)

        return all_results

    def __del__(self):
        if cudart is None:
            return
        if not hasattr(self, "_d_token_id"):
            return
        bufs = []
        for attr in ("_d_token_id", "_d_logits"):
            value = getattr(self, attr, None)
            if value is not None:
                bufs.append(value)
        for attr in ("_d_conv_state", "_d_ssm_state", "_d_present_conv", "_d_present_ssm"):
            bufs.extend(getattr(self, attr, []))
        for d_ptr in getattr(self, "_d_debug", {}).values():
            bufs.append(d_ptr)
        for d_ptr in bufs:
            cudart.cudaFree(d_ptr)
        if hasattr(self, "stream"):
            cudart.cudaStreamDestroy(self.stream)
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine

def runner_from_bundle(
    *,
    runtime_strategy: str,
    config: dict,
    header: dict,
    engine_plan: bytes,
    bundle_path: str,
    distributed_communicator: object | None = None,
) -> MambaTrtRunner:
    del runtime_strategy, bundle_path
    return MambaTrtRunner(
        engine_plan=engine_plan,
        num_layers=header.get("num_layers", config.get("num_hidden_layers", 1)),
        distributed_communicator=distributed_communicator,
    )
