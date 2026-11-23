# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bolt-specific build arguments; no shared command-line switches."""

from dataclasses import dataclass
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model


@dataclass(frozen=True)
class BoltRequest:
    """Only the controls exposed by this family-owned command."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str
    max_sequence_length: int | None
    verbose: bool
    draft: Path | None = None
    edge_builder_python: Path | None = None


def build_bundle(request: BoltRequest) -> None:
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
    precision="bf16",
    backend="trt",
    revision=None,
    max_sequence_length=None,
    verbose=False,
    execution_variant=None,
    companion=(),
    edge_builder_python=None,
):
    local = resolve_model(model, revision)
    resolve_family(load_model_metadata(local), "nemotron_bolt")
    draft = None
    if execution_variant is not None or companion:
        if execution_variant != "lilicorr" or len(companion or ()) != 1:
            raise ValueError("LiLiCorr requires exactly one explicit --companion draft=LOCAL_DIR")
        role, sep, name = companion[0].partition("=")
        if role != "draft" or not sep or not name or "://" in name or not Path(name).is_dir():
            raise ValueError("Bolt requires draft=LOCAL_DIR; it does not fetch or guess a draft")
        draft = Path(name).resolve()
    build_bundle(
        BoltRequest(
            model_dir=local,
            output_path=output,
            family="nemotron_bolt",
            task=task,
            precision=precision,
            backend=backend,
            max_sequence_length=max_sequence_length,
            draft=draft,
            edge_builder_python=edge_builder_python,
            verbose=verbose,
        )
    )
    return 0
