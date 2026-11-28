# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash-Next-specific build arguments; no shared command-line switches."""

from dataclasses import dataclass
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model


@dataclass(frozen=True)
class FlashNextRequest:
    """Only the controls exposed by this family-owned command."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str
    max_sequence_length: int | None
    verbose: bool
    edge_builder_python: Path | None = None
    edge_ple_residency: str = "resident"


def build_bundle(request: FlashNextRequest) -> None:
    select_backend(request.backend)
    from .model import build as build_model

    writer = BundleWriter(request.output_path)
    try:
        writer.record_build(request)
        build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def build(
    *,
    model,
    output,
    task="vision_language_generation",
    precision="fp16",
    backend="trt",
    revision=None,
    max_sequence_length=None,
    verbose=False,
    edge_builder_python=None,
    edge_ple_residency="resident",
):
    local = resolve_model(model, revision)
    resolve_family(load_model_metadata(local), "qwen3_8_flash_next")
    build_bundle(
        FlashNextRequest(
            model_dir=local,
            output_path=output,
            family="qwen3_8_flash_next",
            task=task,
            precision=precision,
            backend=backend,
            max_sequence_length=max_sequence_length,
            edge_builder_python=edge_builder_python,
            edge_ple_residency=edge_ple_residency,
            verbose=verbose,
        )
    )
    return 0
