#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT split-prefill/decode reproducer for issue #428."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt

from infer_engine import (
    DEFAULT_TOKEN_IDS,
    MASKED_SCORE,
    check_cuda,
    cuda_status,
    cuda_success,
    cuda_value,
    parse_token_ids,
    tensor_nbytes,
)

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]


class PlanSession:
    def __init__(
        self,
        runtime: trt.Runtime,
        plan_path: Path,
        stream: int,
        *,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        external_buffers: dict[str, int] | None = None,
    ) -> None:
        self.name = plan_path.stem
        self.stream = stream
        self.engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context for {plan_path}")
        for name, shape in (input_shapes or {}).items():
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"{self.name}: set_input_shape({name}, {shape}) failed")
        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"{self.name}: unresolved tensors: {unresolved}")

        external_buffers = external_buffers or {}
        self.buffers: dict[str, int] = {}
        self.owned_buffers: dict[str, int] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if name in external_buffers:
                pointer = external_buffers[name]
            else:
                nbytes = tensor_nbytes(self.engine, self.context, name)
                pointer = cuda_value(
                    cudart.cudaMalloc(nbytes), f"{self.name}: cudaMalloc({name}, {nbytes})"
                )
                self.owned_buffers[name] = pointer
            self.buffers[name] = pointer
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"{self.name}: set_tensor_address({name}) failed")

    def nbytes(self, name: str) -> int:
        return tensor_nbytes(self.engine, self.context, name)

    def copy_host_to_device(self, name: str, array: np.ndarray) -> None:
        expected = self.nbytes(name)
        if array.nbytes != expected:
            raise RuntimeError(
                f"{self.name}: {name} byte mismatch: host={array.nbytes} engine={expected}"
            )
        check_cuda(
            cudart.cudaMemcpyAsync(
                self.buffers[name],
                array.ctypes.data,
                array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            f"{self.name}: cudaMemcpyAsync({name}, H2D)",
        )

    def copy_device_to_host(self, name: str) -> np.ndarray:
        shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
        output = np.empty(shape, dtype=dtype)
        check_cuda(
            cudart.cudaMemcpyAsync(
                output.ctypes.data,
                self.buffers[name],
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            f"{self.name}: cudaMemcpyAsync({name}, D2H)",
        )
        check_cuda(cudart.cudaStreamSynchronize(self.stream), f"{self.name}: output sync")
        return output

    def execute(self) -> object:
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError(f"{self.name}: execute_async_v3 returned false")
        status = cuda_status(cudart.cudaStreamSynchronize(self.stream))
        print(f"[pure-trt] {self.name}: execute_sync_status={status}", flush=True)
        return status

    def destroy(self) -> bool:
        clean = True
        first_error: tuple[str, object] | None = None
        for name, pointer in reversed(tuple(self.owned_buffers.items())):
            status = cuda_status(cudart.cudaFree(pointer))
            if status != cuda_success():
                clean = False
                if first_error is None:
                    first_error = name, status
        if first_error is not None:
            name, status = first_error
            print(
                f"[pure-trt] {self.name}: cleanup first_error="
                f"cudaFree({name}) status={status}",
                file=sys.stderr,
                flush=True,
            )
        self.owned_buffers.clear()
        self.buffers.clear()
        del self.context
        gc.collect()
        print(f"[pure-trt] {self.name}: context_destroyed", flush=True)
        del self.engine
        gc.collect()
        print(f"[pure-trt] {self.name}: engine_destroyed", flush=True)
        return clean


def cache_input_names(engine: trt.ICudaEngine) -> tuple[str, ...]:
    names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if (
            engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            and (name.startswith("cache_k_") or name.startswith("cache_v_"))
        ):
            names.append(name)
    return tuple(names)


def allocate_shared_cache(
    engine: trt.ICudaEngine, context: trt.IExecutionContext
) -> dict[str, int]:
    buffers = {}
    for name in cache_input_names(engine):
        nbytes = tensor_nbytes(engine, context, name)
        buffers[name] = cuda_value(cudart.cudaMalloc(nbytes), f"cudaMalloc({name}, {nbytes})")
    return buffers


def make_decode_mask(prompt_length: int, cache_length: int) -> np.ndarray:
    mask = np.full((1, cache_length + 1), MASKED_SCORE, dtype=np.float32)
    mask[0, :prompt_length] = 0.0
    mask[0, -1] = 0.0
    return mask


def free_shared_cache(buffers: dict[str, int]) -> bool:
    clean = True
    first_error: tuple[str, object] | None = None
    for name, pointer in reversed(tuple(buffers.items())):
        status = cuda_status(cudart.cudaFree(pointer))
        if status != cuda_success():
            clean = False
            if first_error is None:
                first_error = name, status
    if first_error is not None:
        name, status = first_error
        print(
            f"[pure-trt] shared-cache: cleanup first_error="
            f"cudaFree({name}) status={status}",
            file=sys.stderr,
            flush=True,
        )
    buffers.clear()
    return clean


def run(
    prefill_plan: Path,
    decode_plan: Path,
    token_ids: tuple[int, ...],
    cache_length: int,
) -> int:
    print(f"[pure-trt] TensorRT={trt.__version__}", flush=True)
    print(
        f"[pure-trt] prefill_plan={prefill_plan} decode_plan={decode_plan} "
        f"cache_length={cache_length} token_ids={','.join(map(str, token_ids))}",
        flush=True,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    stream = cuda_value(cudart.cudaStreamCreate(), "cudaStreamCreate")

    # Load decode once to discover the shared fixed-size cache contract.
    decode_engine_probe = runtime.deserialize_cuda_engine(decode_plan.read_bytes())
    if decode_engine_probe is None:
        raise RuntimeError(f"failed to deserialize {decode_plan}")
    decode_context_probe = decode_engine_probe.create_execution_context()
    if decode_context_probe is None:
        raise RuntimeError("failed to create decode probe context")
    shared_cache = allocate_shared_cache(decode_engine_probe, decode_context_probe)
    del decode_context_probe
    del decode_engine_probe
    gc.collect()

    sequence_length = len(token_ids)
    prefill_shapes = {
        "token_id": (sequence_length,),
        "position_id": (sequence_length,),
        "attention_mask": (sequence_length, cache_length + sequence_length),
    }
    prefill = PlanSession(
        runtime,
        prefill_plan,
        stream,
        input_shapes=prefill_shapes,
        external_buffers=shared_cache,
    )
    decode = PlanSession(
        runtime,
        decode_plan,
        stream,
        external_buffers=shared_cache,
    )

    try:
        for pointer_name, pointer in shared_cache.items():
            check_cuda(
                cudart.cudaMemsetAsync(pointer, 0, decode.nbytes(pointer_name), stream),
                f"cudaMemsetAsync({pointer_name})",
            )

        prefill.copy_host_to_device("token_id", np.asarray(token_ids, dtype=np.int32))
        prefill.copy_host_to_device(
            "position_id", np.arange(sequence_length, dtype=np.int32)
        )
        prefill_mask = np.full(
            (sequence_length, cache_length + sequence_length),
            MASKED_SCORE,
            dtype=np.float32,
        )
        for row in range(sequence_length):
            prefill_mask[row, cache_length : cache_length + row + 1] = 0.0
        prefill.copy_host_to_device("attention_mask", prefill_mask)
        prefill_status = prefill.execute()
        if prefill_status != cuda_success():
            raise RuntimeError(f"prefill execution failed with CUDA status {prefill_status}")

        prefill_logits = prefill.copy_device_to_host("logits")
        next_token = int(np.argmax(prefill_logits[-1]))
        print(f"[pure-trt] prefill_top_token={next_token}", flush=True)

        device_to_device = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        for cache_name, cache_pointer in shared_cache.items():
            present_name = cache_name.replace("cache_", "present_", 1)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    cache_pointer,
                    prefill.buffers[present_name],
                    prefill.nbytes(present_name),
                    device_to_device,
                    stream,
                ),
                f"prefill-to-cache copy {present_name}",
            )
        check_cuda(cudart.cudaStreamSynchronize(stream), "prefill-to-cache sync")
        print("[pure-trt] prefill_to_decode_cache=OK", flush=True)

        decode.copy_host_to_device("token_id", np.asarray([next_token], dtype=np.int32))
        decode.copy_host_to_device(
            "position_id", np.asarray([sequence_length], dtype=np.int32)
        )
        decode.copy_host_to_device(
            "attention_mask", make_decode_mask(sequence_length, cache_length)
        )
        decode_status = decode.execute()
        reproduced = decode_status != cuda_success()
        if reproduced:
            print(
                f"[pure-trt] ISSUE_428_REPRODUCED decode_sync_status={decode_status}",
                flush=True,
            )
        else:
            decode_logits = decode.copy_device_to_host("logits")
            print(
                f"[pure-trt] decode_top_token={int(np.argmax(decode_logits[-1]))}",
                flush=True,
            )

            row_bytes = decode.nbytes("present_k_0")
            cache_offset = sequence_length * row_bytes
            for cache_name, cache_pointer in shared_cache.items():
                present_name = cache_name.replace("cache_", "present_", 1)
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        cache_pointer + cache_offset,
                        decode.buffers[present_name],
                        row_bytes,
                        device_to_device,
                        stream,
                    ),
                    f"decode-to-cache copy {present_name}",
                )
            check_cuda(cudart.cudaStreamSynchronize(stream), "decode-to-cache sync")
            print("[pure-trt] decode_to_cache=OK", flush=True)

        # Mirror pipeline member teardown: state, prefill module, decode module.
        clean = free_shared_cache(shared_cache)
        clean = prefill.destroy() and clean
        clean = decode.destroy() and clean
        del runtime
        gc.collect()
        print("[pure-trt] runtime_destroyed", flush=True)
        stream_status = cuda_status(cudart.cudaStreamDestroy(stream))
        if stream_status != cuda_success():
            clean = False
            print(
                f"[pure-trt] cudaStreamDestroy status={stream_status}",
                file=sys.stderr,
                flush=True,
            )
        if reproduced:
            return 42
        if clean:
            print("[pure-trt] CLEAN_EXIT", flush=True)
            return 0
        return 10
    except BaseException:
        if shared_cache:
            free_shared_cache(shared_cache)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pure TensorRT split-engine reproducer for issue #428"
    )
    parser.add_argument("--prefill-plan", type=Path, required=True)
    parser.add_argument("--decode-plan", type=Path, required=True)
    parser.add_argument("--cache-length", type=int, default=1741)
    parser.add_argument(
        "--token-ids",
        type=parse_token_ids,
        default=DEFAULT_TOKEN_IDS,
        help='Comma-separated token IDs (default: Gemma encoding of "Hello")',
    )
    args = parser.parse_args()
    return run(args.prefill_plan, args.decode_plan, args.token_ids, args.cache_length)


if __name__ == "__main__":
    raise SystemExit(main())
