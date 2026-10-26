# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL kernel for the toy user kernel tests: ``out = a * b + c + 1`` on any static shape.

The kernel indexes the tensors with a flat coordinate, one thread per element. The launch grid is
``ceil(numel / 128)`` blocks of 128 threads, which the test passes to ``UserKernel.from_cutedsl``.
This module needs ``cutlass.cute``. The tests import it only after checking that it is installed.
"""

import cutlass.cute as cute

THREADS = 128


@cute.kernel
def _fma_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, out: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < cute.size(out):
        out[i] = a[i] * b[i] + c[i] + 1.0


@cute.jit
def fma_run(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, out: cute.Tensor):
    n = cute.size(out)
    _fma_kernel(a, b, c, out).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1))
