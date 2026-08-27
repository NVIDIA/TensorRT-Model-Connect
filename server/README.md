<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local server

This directory owns the optional local serving process behind `trtmc serve`.

- `python/trtmc_server/` provides the HTTP and WebSocket control plane.
- `native/` provides the persistent JSONL worker linked to `trtmc_core`.
- `tests/` contains the server-owned Python and native contract tests.

The dependency direction is one-way: the server may use public TensorRT-Model-Connect APIs,
but the core library and its Python package must not import or link the server. The root CLI,
package metadata, and CI files are intentionally thin integration seams; applications consume
the server through its process, HTTP, and WebSocket contracts.
