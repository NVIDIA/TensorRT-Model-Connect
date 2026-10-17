# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned whole-network Edge build; no native graph is implemented."""


def build(request, writer) -> None:
    """Build the selected task through this family's reviewed Edge route.

    Args:
        request: Original Model Connect request.
        writer: Unpublished output bundle writer.

    Raises:
        Exception: Unsupported request, unavailable dependency, or build failure.
    """
    from .build_request import coerce_request
    from .edge_llm.dispatch import build as dispatch_build

    request = coerce_request(request)

    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_5_moe does not support dynamic_kv_cache")
    writer.set_header(family="qwen3_5_moe", task=request.task, backend=request.backend)
    dispatch_build(request, writer)
