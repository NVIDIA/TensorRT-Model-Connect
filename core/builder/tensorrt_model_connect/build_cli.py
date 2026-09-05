# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal command-line entrypoint for family-owned builds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .build import BuildRequest, build
from .model_support import load_model_metadata, resolve_family


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trtmc")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build one TensorRT bundle")
    build_parser.add_argument("model", help="Hugging Face model ID or local snapshot")
    build_parser.add_argument("-o", "--output", type=Path, required=True)
    build_parser.add_argument("--task", help="Override the family-owned default task")
    build_parser.add_argument("--revision", help="Hugging Face model revision")
    build_parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp32")
    build_parser.add_argument("--backend", choices=("trt", "trt_rtx"), default="trt")
    build_parser.add_argument("--max-sequence-length", type=int)
    build_parser.add_argument("--image-height", type=int)
    build_parser.add_argument("--image-width", type=int)
    build_parser.add_argument("--video-num-frames", type=int)
    build_parser.add_argument("--max-batch-size", type=int, default=1)
    build_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    build_parser.add_argument("--context-parallel-size", type=int, default=1)
    build_parser.add_argument("--quantization")
    build_parser.add_argument("--fp32-layer", type=int, action="append", default=[])
    build_parser.add_argument("--dynamic-kv-cache", action="store_true")
    build_parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "build":
        raise AssertionError(f"unhandled command: {args.command}")
    model_dir = _resolve_model(args.model, args.revision)
    family, support = resolve_family(load_model_metadata(model_dir))
    task = args.task or support.default_task
    if task not in support.tasks:
        raise ValueError(
            f"family {family!r} does not support task {task!r}; "
            f"choose one of: {', '.join(support.tasks)}"
        )
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=args.output,
            precision=args.precision,
            backend=args.backend,
            family=family,
            task=task,
            max_sequence_length=args.max_sequence_length,
            image_height=args.image_height,
            image_width=args.image_width,
            video_num_frames=args.video_num_frames,
            max_batch_size=args.max_batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            context_parallel_size=args.context_parallel_size,
            quantization=args.quantization,
            fp32_layers=tuple(args.fp32_layer),
            dynamic_kv_cache=args.dynamic_kv_cache,
            verbose=args.verbose,
        )
    )
    return 0


def _resolve_model(model: str, revision: str | None) -> Path:
    local = Path(model)
    if local.is_dir():
        return local
    if local.exists():
        raise ValueError(f"model path is not a directory: {local}")

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model, revision=revision))
