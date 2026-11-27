# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One owner for Bolt's ordinary and LiLiCorr builds."""

from pathlib import Path

from .config import validate_inputs
from .support import TASKS


def build(request, writer):
    if request.family != "nemotron_bolt" or request.task not in TASKS:
        raise ValueError("Unsupported Bolt build task")
    if (
        request.backend != "trt"
        or request.precision != "bf16"
        or getattr(request, "quantization", None) not in (None, "none")
    ):
        raise ValueError(
            "Bolt preserves its prepared IQ2, Q8 and NVFP4 weights with BF16 model compute"
        )
    if (
        getattr(request, "max_batch_size", 1) != 1
        or getattr(request, "tensor_parallel_size", 1) != 1
        or getattr(request, "context_parallel_size", 1) != 1
        or getattr(request, "dynamic_kv_cache", False)
        or getattr(request, "graph_transform", None) is not None
        or getattr(request, "fp32_layers", ())
        or any(
            getattr(request, k, None) is not None
            for k in ("mtp_seq_len", "image_height", "image_width", "video_num_frames")
        )
    ):
        raise ValueError(
            "Bolt supports one request on one device with the checkpoint's native geometry"
        )
    from .edge_llm.dispatch import build as dispatch

    draft = getattr(request, "draft", None)
    cfg = validate_inputs(request.model_dir, draft)
    limit = 8192 if request.max_sequence_length is None else request.max_sequence_length
    if (
        type(limit) is not int
        or not (8 if draft else 1) < limit <= cfg["llm_config"]["max_position_embeddings"]
    ):
        raise ValueError("Invalid Bolt context capacity")
    writer.set_header(family="nemotron_bolt", task=request.task, backend=request.backend)
    # Native plans and Edge bundles both contain checkpoint-derived material.
    writer.add_bytes(
        "notices/OpenMDW-1.1.txt",
        (Path(__file__).parent / "licenses" / "OpenMDW-1.1.txt").read_bytes(),
    )
    for role, checkpoint in (("target", request.model_dir), ("draft", draft)):
        if checkpoint is None:
            continue
        for asset in ("LICENSE", "NOTICE", "README.md"):
            source = checkpoint / asset
            if source.is_file():
                writer.add_bytes(f"notices/{role}/{asset}", source.read_bytes())
    dispatch(request, writer, draft, min(4096, limit), limit, _native)


def _native(request, writer):
    from .native_builder import build_native

    build_native(request, writer)
