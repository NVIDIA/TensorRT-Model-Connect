# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned LFM2-VL build command."""

from pathlib import Path


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    precision: str = "bf16",
    max_sequence_length: int = 32768,
    verbose: bool = False,
) -> int:
    from tensorrt_model_connect import BuildRequest, build as build_bundle
    from tensorrt_model_connect.model_support import resolve_model

    build_bundle(
        BuildRequest(
            model_dir=resolve_model(model, revision),
            output_path=output,
            family="lfm2_vl",
            task="images_text_to_text",
            precision=precision,
            max_sequence_length=max_sequence_length,
            verbose=verbose,
        )
    )
    return 0
