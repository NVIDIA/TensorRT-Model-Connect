# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer family-owned complete-network build selection."""

from __future__ import annotations


def build(request, writer) -> None:
    """Build the exact qualified Muse-Glimmer profile through Edge-LLM."""
    from .build_request import coerce_request

    request = coerce_request(request)
    writer.set_header(family="muse_glimmer", task=request.task, backend=request.backend)

    from .edge_llm.dispatch import build as dispatch_build

    def native(_request, _writer) -> None:
        raise NotImplementedError(
            "Muse-Glimmer currently requires the qualified Edge-LLM 0.11.0 NVFP4 route"
        )

    dispatch_build(request, writer, native)
