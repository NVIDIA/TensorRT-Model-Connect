# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen's CPU-only CLI entry point and atomic bundle publication."""

from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import graph_transform
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model

from .build_request import BuildRequest


def build_bundle(request: BuildRequest) -> None:
    """Build one Qwen request without extending shared build inputs."""
    if request.family != 'qwen':
        raise ValueError('Qwen build requires family=qwen')
    if request.strip_weights and request.backend != 'trt':
        raise ValueError('strip_weights requires backend=trt')
    select_backend(request.backend)
    from .model import build as build_model

    writer = BundleWriter(request.output_path)
    writer.record_build(request)
    try:
        with graph_transform(request.graph_transform):
            build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def build(*, model: str, output: Path, revision: str | None = None,
          task: str | None = None, precision: str | None = None,
          backend: str = 'trt', max_sequence_length: int | None = None,
          max_batch_size: int = 1, tensor_parallel_size: int = 1,
          quantization: str | None = None, strip_weights: bool = False,
          verbose: bool = False) -> int:
    """Resolve the declared checkpoint, then hand control to the Qwen builder."""
    model_dir = resolve_model(model, revision)
    _, support = resolve_family(load_model_metadata(model_dir), 'qwen')
    task = task or support.default_task
    if task not in support.tasks:
        raise ValueError(f'Qwen checkpoint does not support task={task}')
    request = BuildRequest(
        model_dir=model_dir, output_path=output, family='qwen', task=task,
        precision=precision or ('bf16' if strip_weights else support.default_precision),
        backend=backend, max_sequence_length=max_sequence_length,
        max_batch_size=max_batch_size, tensor_parallel_size=tensor_parallel_size,
        quantization=quantization, strip_weights=strip_weights, verbose=verbose,
    )
    build_bundle(request)
    return 0
