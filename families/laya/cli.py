# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Laya's checkpoint, capacity, and native decision commands."""

from dataclasses import dataclass
from pathlib import Path


VARIANTS = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    variant: str = "english"
    max_sequence_length: int | None = None
    max_batch_size: int = 8
    max_options: int = 256
    precision: str = "bf16"
    task: str = "structured_decision"
    backend: str = "trt"

    def __post_init__(self):
        if self.variant not in (*VARIANTS, "router"):
            raise ValueError("unknown Laya variant")
        if self.precision != "bf16" or self.task != "structured_decision" or self.backend != "trt":
            raise ValueError(
                "Laya requires BF16 matrix precision, structured_decision, and TensorRT"
            )
        for value in (self.max_batch_size, self.max_options):
            if type(value) is not int or value < 1:
                raise ValueError("batch and option capacities must be positive integers")
        if self.max_sequence_length is not None and (
            type(self.max_sequence_length) is not int or not 16 <= self.max_sequence_length <= 8192
        ):
            raise ValueError("sequence capacity must be between 16 and 8192")


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    variant: str = "english",
    max_sequence_length: int | None = None,
    max_batch_size: int = 8,
    max_options: int = 256,
):
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from tensorrt_model_connect.model_support import resolve_model
    from .model import build as build_model

    request = BuildRequest(
        resolve_model(model, revision), variant, max_sequence_length, max_batch_size, max_options
    )
    writer = BundleWriter(output)
    try:
        build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return 0
