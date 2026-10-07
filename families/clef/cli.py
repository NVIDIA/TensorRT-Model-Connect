# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef owns its build arguments and decision command."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "structured_decision"
    precision: str = "bf16"
    backend: str = "trt"
    max_sequence_length: int = 16384
    max_questions: int = 64
    max_options: int = 512
    verbose: bool = False

    def __post_init__(self):
        if self.task != "structured_decision" or self.precision != "bf16" or self.backend != "trt":
            raise ValueError("Clef requires structured_decision, bf16, and the trt backend")
        if (
            type(self.max_sequence_length) is not int
            or not 512 <= self.max_sequence_length <= 16384
        ):
            raise ValueError("max_sequence_length must be between 512 and 16384")
        if self.max_sequence_length % 64:
            raise ValueError("max_sequence_length must be a multiple of 64")
        if self.max_questions < 3 or self.max_options < 8:
            raise ValueError("max_questions must be at least 3 and max_options at least 8")


def build_bundle(request: BuildRequest, output: Path):
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from .model import build as build_model

    writer = BundleWriter(output)
    try:
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
    max_sequence_length: int = 16384,
    max_questions: int = 64,
    max_options: int = 512,
    verbose: bool = False,
):
    from tensorrt_model_connect.model_support import resolve_model

    build_bundle(
        BuildRequest(
            resolve_model(model, revision),
            max_sequence_length=max_sequence_length,
            max_questions=max_questions,
            max_options=max_options,
            verbose=verbose,
        ),
        output,
    )
    return 0
