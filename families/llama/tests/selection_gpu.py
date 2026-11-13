# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU numerical contract probe: python -m families.llama.tests.selection_gpu."""

import json

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda

from ..speculative.selection import build_selection


def checked(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(str(result))
    return result[1] if len(result) == 2 else result[1:]


def run(vocab, ranks):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(build_selection(vocab, 9, ranks))
    assert engine is not None
    context = engine.create_execution_context()
    stream = checked(cuda.cudaStreamCreate())
    source = checked(cuda.cudaMalloc(9 * vocab * 4))
    destination = checked(cuda.cudaMalloc(9 * (ranks + 1) * 4))
    rng = np.random.default_rng(49)
    checks = 0
    try:
        assert context.set_tensor_address("logits", source)
        assert context.set_tensor_address("selection", destination)
        # Alternate row shapes to exercise output strides and retained buffers.
        for rows in (1, 9, 5, 1):
            cases = [rng.normal(size=(rows, vocab)).astype(np.float32),
                     np.full((rows, vocab), -5, np.float32),
                     np.zeros((rows, vocab), np.float32)]
            cases[-1][:, ::2] = -0.0
            extremes = cases[0].copy()
            extremes[:, 0] = np.finfo(np.float32).min
            extremes[:, -1] = np.finfo(np.float32).max
            cases.append(extremes)
            ties = cases[0].copy()
            ties[:, 1] = ties[:, -1] = 9
            cases.append(ties)
            for bad in (np.nan, np.inf, -np.inf):
                for position in (0, vocab // 2, vocab - 1):
                    poisoned = cases[0].copy()
                    poisoned[-1, position] = bad
                    cases.append(poisoned)
            for values in cases:
                assert context.set_input_shape("logits", values.shape)
                checked(cuda.cudaMemcpyAsync(source, values.ctypes.data, values.nbytes,
                                             cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
                assert context.execute_async_v3(stream)
                actual = np.empty((rows, ranks + 1), np.int32)
                checked(cuda.cudaMemcpyAsync(actual.ctypes.data, destination, actual.nbytes,
                                             cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
                checked(cuda.cudaStreamSynchronize(stream))
                valid = np.isfinite(values).all(axis=1)
                np.testing.assert_array_equal(actual[:, -1], valid.astype(np.int32))
                # Stable ordering is the independent first-index tie oracle.
                expected = np.argsort(-values[valid], axis=1, kind="stable")[:, :ranks]
                np.testing.assert_array_equal(actual[valid, :ranks], expected)
                checks += 1
        return {"vocab": vocab, "ranks": ranks, "cases": checks, "passed": True}
    finally:
        checked(cuda.cudaStreamSynchronize(stream))
        checked(cuda.cudaFree(source))
        checked(cuda.cudaFree(destination))
        checked(cuda.cudaStreamDestroy(stream))


if __name__ == "__main__":
    print(json.dumps([run(128256, 1), run(32000, 2), run(2, 2)]))
