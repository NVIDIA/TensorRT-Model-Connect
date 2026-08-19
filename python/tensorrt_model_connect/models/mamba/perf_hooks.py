# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance and profiling hooks owned by the Mamba family."""

from __future__ import annotations

import sys
import time

import numpy as np

# cuda-python >= 13 uses cuda.bindings.runtime; older versions use cuda.cudart.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exercised in CPU-only test envs
        cudart = None  # type: ignore[assignment]


def handles_runtime_strategy(runtime_strategy: str) -> bool:
    return runtime_strategy == "ssm_recurrent"


def backend_label() -> str:
    return "TRT-Mamba"


def perf_report_note() -> str:
    return (
        "* Prefill: HF processes full sequence; TRT is token-by-token\n"
        "  Decode: both token-by-token with recurrent state"
    )


def supports_hf_compile() -> bool:
    return False


def supports_layer_profile() -> bool:
    return False


def supports_cpu_phase_profile() -> bool:
    return False


def layer_profile_skip_message() -> str:
    return (
        "[profile] Family runtime does not support per-layer IProfiler; "
        "skipping."
    )


def cpu_profile_runner_type() -> str:
    return "mamba"


def bench_trt(
    *,
    engine_plan: bytes,
    num_layers: int,
    max_cache_length: int,
    input_ids: list[int],
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    eos_token_id: int | None,
    verbose: bool,
) -> dict:
    """Benchmark TRT inference for Mamba/SSM recurrent state."""
    del max_cache_length
    from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

    runner = MambaTrtRunner(
        engine_plan=engine_plan,
        num_layers=num_layers,
    )

    prefill_times: list[float] = []
    decode_times: list[float] = []
    decode_token_counts: list[int] = []
    gen_ids: list[int] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        is_warmup = run_idx < warmup
        runner.reset()

        t0 = time.perf_counter()
        for tid in input_ids:
            result = runner.step(tid)
        logits = result["logits"].flatten()
        prefill_ms = (time.perf_counter() - t0) * 1000

        tokens_generated = 0
        run_gen_ids: list[int] = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_token = int(np.argmax(logits))
            run_gen_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            result = runner.step(next_token)
            logits = result["logits"].flatten()
            tokens_generated += 1
        decode_ms = (time.perf_counter() - t0) * 1000

        if not is_warmup:
            prefill_times.append(prefill_ms)
            decode_times.append(decode_ms)
            decode_token_counts.append(tokens_generated)
            gen_ids = run_gen_ids

        if verbose:
            tag = "warmup" if is_warmup else f"iter {run_idx - warmup + 1}"
            print(f"  [trt {tag}] prefill={prefill_ms:.2f}ms "
                  f"decode={decode_ms:.2f}ms ({tokens_generated} tokens)",
                  file=sys.stderr)

    return {
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "decode_token_counts": decode_token_counts,
        "gen_ids": gen_ids,
    }


class TimedCpuProfileRunner:
    """Timed CPU-phase profiler for Mamba recurrent state execution."""

    PHASES = ("h2d", "tensor_bind", "execute", "d2d_state", "d2h", "argmax")

    def __init__(self, engine_plan: bytes, num_layers: int):
        from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

        self._runner = MambaTrtRunner(
            engine_plan=engine_plan,
            num_layers=num_layers,
        )
        self._phase_times: dict[str, list[float]] = {p: [] for p in self.PHASES}

    def reset(self) -> None:
        self._runner.reset()

    def reset_timing(self) -> None:
        for phase in self.PHASES:
            self._phase_times[phase].clear()

    def step(self, token_id: int) -> np.ndarray:
        return self._runner.step(token_id)["logits"].flatten()

    def timed_step(self, token_id: int) -> np.ndarray:
        if cudart is None:
            raise ImportError("cuda-python is required for CPU phase profiling")
        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        d2d = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        runner = self._runner
        stream = runner.stream
        conv_state_bytes = runner.d_inner * runner.conv_kernel * 4
        ssm_state_bytes = runner.d_inner * runner.state_size * 4

        start = time.perf_counter()
        runner._h_token_id[0] = token_id
        cudart.cudaMemcpyAsync(
            runner._d_token_id, runner._h_token_id.ctypes.data, 4, h2d, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["h2d"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        runner.context.set_tensor_address("token_id", runner._d_token_id)
        runner.context.set_tensor_address("logits", runner._d_logits)
        for layer in range(runner.num_layers):
            runner.context.set_tensor_address(
                f"conv_state_{layer}", runner._d_conv_state[layer])
            runner.context.set_tensor_address(
                f"ssm_state_{layer}", runner._d_ssm_state[layer])
            runner.context.set_tensor_address(
                f"present_conv_{layer}", runner._d_present_conv[layer])
            runner.context.set_tensor_address(
                f"present_ssm_{layer}", runner._d_present_ssm[layer])
        for name in runner._debug_output_names:
            runner.context.set_tensor_address(name, runner._d_debug[name])
        self._phase_times["tensor_bind"].append(
            (time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        runner.context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["execute"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        for layer in range(runner.num_layers):
            cudart.cudaMemcpyAsync(
                runner._d_conv_state[layer], runner._d_present_conv[layer],
                conv_state_bytes, d2d, stream)
            cudart.cudaMemcpyAsync(
                runner._d_ssm_state[layer], runner._d_present_ssm[layer],
                ssm_state_bytes, d2d, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["d2d_state"].append(
            (time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        cudart.cudaMemcpyAsync(
            runner._h_logits.ctypes.data, runner._d_logits,
            runner._logits_numel * 4, d2h, stream)
        cudart.cudaStreamSynchronize(stream)
        self._phase_times["d2h"].append((time.perf_counter() - start) * 1000)

        logits = runner._h_logits.flatten()
        start = time.perf_counter()
        int(np.argmax(logits))
        self._phase_times["argmax"].append((time.perf_counter() - start) * 1000)

        return logits

    @property
    def phase_times(self) -> dict[str, list[float]]:
        return self._phase_times


def make_cpu_profile_runner(
    *,
    engine_plan: bytes,
    num_layers: int,
    max_cache_length: int,
):
    del max_cache_length
    return TimedCpuProfileRunner(engine_plan, num_layers)
