<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local server

This directory owns the optional process behind `trtmc serve`:

- `python/trtmc_server/` provides the HTTP and WebSocket control plane.
- `native/` adapts the public `load_task()` and `ITask` contracts to a private
  JSONL worker process.
- `tests/` owns server API, process-lifecycle, protocol, and dependency checks.

The dependency is one-way: the server may use public library contracts, while
core and model families never depend on server implementation. Applications
consume the `trtmc serve` process and its HTTP/WebSocket APIs; they do not
import `trtmc_server`.

Concurrency is a fixed set of serial worker lanes configured at startup. The
server has no waiting queue, dynamic placement, continuous batching, cluster
scheduler, or worker restart. Saturation fails immediately, failed lanes leave
the model degraded while another lane remains healthy, and recovery belongs to
an external supervisor. One server process is one local placement domain, not
a generic distributed serving system.

Replica counts above one apply only to independently loadable single-process
bundles. MPI/NCCL distributed bundles are not supported by `trtmc serve`.
For multiple GPUs, run independent single-process server instances and pin
each instance with `CUDA_VISIBLE_DEVICES`; external routing remains the
supervisor's responsibility.
