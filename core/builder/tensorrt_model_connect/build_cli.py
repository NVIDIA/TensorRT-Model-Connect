# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal command-line entrypoint for family-owned builds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from .build import BuildRequest, build
from .model_support import load_model_metadata, resolve_family


_OPTION = re.compile(r"([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)=(.*)\Z", re.DOTALL)


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
    build_parser.add_argument(
        "--set",
        dest="family_options",
        action="append",
        default=[],
        metavar="FAMILY.KEY=VALUE",
        help="Set one family-owned scalar build option",
    )
    return parser


def _parse_family_options(
    values: Sequence[str], family: str
) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    result: list[tuple[str, str | int | float | bool | None]] = []
    names: set[str] = set()
    for value in values:
        match = _OPTION.fullmatch(value)
        if match is None:
            raise ValueError("--set must use FAMILY.KEY=VALUE")
        namespace, name, raw = match.groups()
        if namespace != family:
            raise ValueError(
                f"--set namespace {namespace!r} does not match resolved family {family!r}"
            )
        if name in names:
            raise ValueError(f"duplicate --set option: {family}.{name}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        if not isinstance(parsed, (str, int, float, bool, type(None))):
            raise ValueError("--set values must be JSON scalars")
        result.append((name, parsed))
        names.add(name)
    return tuple(result)


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
            family_options=_parse_family_options(args.family_options, family),
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
