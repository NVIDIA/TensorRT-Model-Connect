# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer-owned build command; importing this module is CPU-only."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import graph_transform
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model

from .build_request import BuildRequest


def build_bundle(request: BuildRequest, output: Path) -> None:
    """Build and publish through the owning model with atomic failure."""
    request = replace(request, output_path=output)
    select_backend(request.backend)
    from .model import build as build_model

    writer = BundleWriter(output)
    try:
        with graph_transform(request.graph_transform):
            build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    task: str = "text_generation",
    precision: str = "fp16",
    backend: str = "trt",
    max_sequence_length: int | None = None,
    quantization: str = "nvfp4",
    execution_variant: str = "autoregressive",
    companion: str | None = None,
    verbose: bool = False,
) -> int:
    """Resolve one exact checkpoint and invoke its family-owned Edge route."""
    model_dir = resolve_model(model, revision)
    resolve_family(load_model_metadata(model_dir), "muse_glimmer")
    request = BuildRequest(
        model_dir=model_dir,
        output_path=output,
        family="muse_glimmer",
        task=task,
        precision=precision,
        backend=backend,
        max_sequence_length=max_sequence_length,
        quantization=quantization,
        execution_variant=execution_variant,
        companion=resolve_model(companion, None) if companion is not None else None,
        verbose=verbose,
    )
    build_bundle(request, output)
    return 0
