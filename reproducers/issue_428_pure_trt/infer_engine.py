#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the issue #428 prefill engine through TensorRT without TRTMC runtime code."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart  # type: ignore[no-redef]


DEFAULT_TOKEN_IDS = (2, 4521)  # Gemma tokenizer encoding for "Hello".
MASKED_SCORE = -1.0e4


def cuda_success() -> object:
    error_type = getattr(cudart, "cudaError_t", None)
    return error_type.cudaSuccess if error_type is not None else 0


def cuda_status(result: object) -> object:
    return result[0] if isinstance(result, tuple) else result


def cuda_value(result: tuple[object, ...], operation: str) -> int:
    status = cuda_status(result)
    if status != cuda_success():
        raise RuntimeError(f"{operation} failed with CUDA status {status}")
    return int(result[1])


def check_cuda(result: object, operation: str) -> None:
    status = cuda_status(result)
    if status != cuda_success():
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


def parse_token_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one token ID is required")
    return values


def tensor_nbytes(engine: trt.ICudaEngine, context: trt.IExecutionContext, name: str) -> int:
    shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
    if not shape or any(dim < 0 for dim in shape):
        raise RuntimeError(f"unresolved shape for {name}: {shape}")
    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def make_prefill_mask(sequence_length: int, cache_length: int) -> np.ndarray:
    """Match GemmaKvCache::write_batched_mask at position zero."""
    kv_length = cache_length + sequence_length
    mask = np.full((sequence_length, kv_length), MASKED_SCORE, dtype=np.float32)
    for row in range(sequence_length):
        mask[row, cache_length : cache_length + row + 1] = 0.0
    return mask


def free_cuda_buffers(buffers: dict[str, int], stream: int) -> bool:
    clean = True
    for name, pointer in reversed(tuple(buffers.items())):
        status = cuda_status(cudart.cudaFree(pointer))
        if status != cuda_success():
            clean = False
            print(f"[pure-trt] cudaFree({name}) status={status}", file=sys.stderr, flush=True)
    status = cuda_status(cudart.cudaStreamDestroy(stream))
    if status != cuda_success():
        clean = False
        print(f"[pure-trt] cudaStreamDestroy status={status}", file=sys.stderr, flush=True)
    return clean


def run(plan_path: Path, token_ids: tuple[int, ...], cache_length: int) -> int:
    print(f"[pure-trt] TensorRT={trt.__version__}", flush=True)
    print(
        f"[pure-trt] plan={plan_path} cache_length={cache_length} "
        f"token_ids={','.join(str(value) for value in token_ids)}",
        flush=True,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {plan_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    sequence_length = len(token_ids)
    input_shapes = {
        "token_id": (sequence_length,),
        "position_id": (sequence_length,),
        "attention_mask": (sequence_length, cache_length + sequence_length),
    }
    for name, shape in input_shapes.items():
        if not context.set_input_shape(name, shape):
            raise RuntimeError(f"set_input_shape({name}, {shape}) failed")

    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"TensorRT reports unresolved tensors: {unresolved}")

    buffers: dict[str, int] = {}
    stream = cuda_value(cudart.cudaStreamCreate(), "cudaStreamCreate")
    try:
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            nbytes = tensor_nbytes(engine, context, name)
            pointer = cuda_value(cudart.cudaMalloc(nbytes), f"cudaMalloc({name}, {nbytes})")
            buffers[name] = pointer
            if not context.set_tensor_address(name, pointer):
                raise RuntimeError(f"set_tensor_address({name}) failed")

        host_inputs = {
            "token_id": np.asarray(token_ids, dtype=np.int32),
            "position_id": np.arange(sequence_length, dtype=np.int32),
            "attention_mask": make_prefill_mask(sequence_length, cache_length),
        }
        host_to_device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        for name, host_array in host_inputs.items():
            check_cuda(
                cudart.cudaMemcpyAsync(
                    buffers[name],
                    host_array.ctypes.data,
                    host_array.nbytes,
                    host_to_device,
                    stream,
                ),
                f"cudaMemcpyAsync({name}, H2D)",
            )

        for layer in range(26):
            for prefix in ("cache_k", "cache_v"):
                name = f"{prefix}_{layer}"
                check_cuda(
                    cudart.cudaMemsetAsync(
                        buffers[name], 0, tensor_nbytes(engine, context, name), stream
                    ),
                    f"cudaMemsetAsync({name})",
                )

        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        print("[pure-trt] enqueue=OK", flush=True)

        sync_status = cuda_status(cudart.cudaStreamSynchronize(stream))
        print(f"[pure-trt] execute_sync_status={sync_status}", flush=True)

        if sync_status == cuda_success():
            logits_shape = tuple(int(dim) for dim in context.get_tensor_shape("logits"))
            logits = np.empty(logits_shape, dtype=np.float32)
            device_to_host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            check_cuda(
                cudart.cudaMemcpyAsync(
                    logits.ctypes.data,
                    buffers["logits"],
                    logits.nbytes,
                    device_to_host,
                    stream,
                ),
                "cudaMemcpyAsync(logits, D2H)",
            )
            copy_status = cuda_status(cudart.cudaStreamSynchronize(stream))
            print(f"[pure-trt] logits_sync_status={copy_status}", flush=True)
            if copy_status == cuda_success():
                print(f"[pure-trt] top_token={int(np.argmax(logits[-1]))}", flush=True)
            else:
                sync_status = copy_status

        cleanup_ok = free_cuda_buffers(buffers, stream)
        buffers.clear()

        # Mirror TRTMC teardown: buffers first, then context and engine.
        del context
        gc.collect()
        print("[pure-trt] context_destroyed", flush=True)
        del engine
        gc.collect()
        print("[pure-trt] engine_destroyed", flush=True)
        del runtime
        gc.collect()
        print("[pure-trt] runtime_destroyed", flush=True)

        if sync_status != cuda_success() or not cleanup_ok:
            return 10
        print("[pure-trt] CLEAN_EXIT", flush=True)
        return 0
    except BaseException:
        if buffers:
            free_cuda_buffers(buffers, stream)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pure TensorRT inference reproducer for issue #428"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-length", type=int, default=1741)
    parser.add_argument(
        "--token-ids",
        type=parse_token_ids,
        default=DEFAULT_TOKEN_IDS,
        help='Comma-separated token IDs (default: Gemma encoding of "Hello")',
    )
    args = parser.parse_args()
    return run(args.plan, args.token_ids, args.cache_length)


if __name__ == "__main__":
    raise SystemExit(main())
