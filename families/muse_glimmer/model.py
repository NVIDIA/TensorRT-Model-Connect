# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer family-owned complete-network build selection."""

from __future__ import annotations


def build(request, writer) -> None:
    """Select complete Edge offload or the owning native implementation."""
    from .build_request import coerce_request

    request = coerce_request(request)
    writer.set_header(family="muse_glimmer", task=request.task, backend=request.backend)

    from .edge_llm.dispatch import build as dispatch_build

    def native(native_request, native_writer) -> None:
        from .native_model import build as native_build

        native_build(native_request, native_writer)

    dispatch_build(request, writer, native)
