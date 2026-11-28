# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary prepared Flash-Next: FP16 activations, original NVFP4 experts and FP8 lexical rows."""

import json
from pathlib import Path

from .support import TASKS, is_flash_next


def build(request, writer):
    if (
        request.family != "qwen3_8_flash_next"
        or request.task not in TASKS
        or request.backend != "trt"
    ):
        raise ValueError("Unsupported Flash-Next family, task, or backend")
    if request.precision != "fp16" or getattr(request, "quantization", None) not in (None, "none"):
        raise ValueError(
            "Flash-Next preserves prepared NVFP4 expert weights and FP16 runtime activations"
        )
    if (
        any(
            getattr(request, k, 1) != 1
            for k in ("max_batch_size", "tensor_parallel_size", "context_parallel_size")
        )
        or getattr(request, "dynamic_kv_cache", False)
        or getattr(request, "fp32_layers", ())
        or any(
            getattr(request, k, None) is not None
            for k in (
                "graph_transform",
                "mtp_seq_len",
                "image_height",
                "image_width",
                "video_num_frames",
                "draft",
                "execution_variant",
            )
        )
    ):
        raise ValueError("Flash-Next supports the ordinary single-device prepared model only")
    cfg = json.loads((request.model_dir / "config.json").read_text())
    if not is_flash_next(cfg):
        raise ValueError("Expected the documented Flash-Next NVFP4 VLM target")
    capacity = 8192 if request.max_sequence_length is None else request.max_sequence_length
    if (
        type(capacity) is not int
        or not 4 <= capacity <= cfg["text_config"]["max_position_embeddings"]
        or capacity % 4
    ):
        raise ValueError("Flash-Next KV capacity must be at least four and divisible by four")
    input_limit = min(4096, capacity - 1)
    writer.set_header(family="qwen3_8_flash_next", task=request.task, backend=request.backend)
    writer.add_bytes("notices/flash_next.txt", Path(__file__).with_name("NOTICE").read_bytes())
    for name in ("LICENSE", "NOTICE", "README.md"):
        p = request.model_dir / name
        if p.is_file():
            writer.add_bytes(f"notices/checkpoint/{name}", p.read_bytes())
    from .edge_llm.dispatch import build as dispatch

    def native(req, out):
        from .native_builder import write_native

        write_native(req, out, cfg, input_limit, capacity)

    dispatch(request, writer, input_limit, capacity, native)
