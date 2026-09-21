# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal command-line entrypoint for family-owned builds."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .build import BuildRequest, _load_family, build
from .model_support import (
    AmbiguousFamilyError,
    FamilyResolutionError,
    load_model_metadata,
    resolve_family,
    resolve_model,
)


def _parser(prepare_family: object | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trtmc")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build one TensorRT bundle")
    build_parser.add_argument("model", help="Hugging Face model ID or local snapshot")
    build_parser.add_argument("-o", "--output", type=Path, required=True)
    build_parser.add_argument(
        "--family", help="Select one compatible family instead of automatic resolution"
    )
    build_parser.add_argument("--task", help="Override the family-owned default task")
    build_parser.add_argument("--revision", help="Hugging Face model revision")
    build_parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"))
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
    prepare_parser = commands.add_parser(
        "prepare-structure",
        help="Prepare one structure request without rebuilding its model bundle",
    )
    prepare_parser.add_argument("model", help="Local model or build package")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("-o", "--output", type=Path, required=True)
    prepare_parser.add_argument("--revision", help="Hugging Face model revision")
    prepare_parser.add_argument(
        "--family", help="Select one compatible family instead of automatic resolution"
    )
    prepare_parser.add_argument("--cache-dir", type=Path)
    add_arguments = getattr(prepare_family, "add_prepare_structure_arguments", None)
    if callable(add_arguments):
        add_arguments(prepare_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    base_parser = _parser()
    if (
        len(arguments) > 1
        and arguments[0] == "prepare-structure"
        and arguments[1] not in {"-h", "--help"}
        and arguments[1].startswith("-")
    ):
        base_parser.error("MODEL must immediately follow prepare-structure")
    preliminary, _ = base_parser.parse_known_args(arguments)
    model_dir = _resolve_model(preliminary.model, preliminary.revision)
    metadata = load_model_metadata(model_dir)
    try:
        family, support = (
            resolve_family(metadata, preliminary.family)
            if preliminary.family is not None
            else resolve_family(metadata)
        )
    except FamilyResolutionError as error:
        _print_family_error(error, arguments)
        return 2
    family_module = _load_family(family) if preliminary.command == "prepare-structure" else None
    args = _parser(family_module).parse_args(arguments)
    if args.command == "prepare-structure":
        prepare = getattr(family_module, "prepare_structure_request", None)
        if not callable(prepare):
            raise ValueError(f"family {family!r} does not support request preparation")
        cli_options = getattr(family_module, "prepare_structure_cli_options", None)
        options = cli_options(args) if callable(cli_options) else {}
        result = prepare(
            model_dir,
            args.input,
            args.output,
            cache_dir=args.cache_dir,
            **options,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command != "build":
        raise AssertionError(f"unhandled command: {args.command}")
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
            precision=args.precision or support.default_precision,
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
    return resolve_model(model, revision)


def _print_family_error(error: FamilyResolutionError, arguments: Sequence[str]) -> None:
    print(f"trtmc: error: {error}", file=sys.stderr)
    if not isinstance(error, AmbiguousFamilyError):
        return
    print("\nCompatible families:", file=sys.stderr)
    for family, support in error.matches:
        print(f"  {family} (Tasks: {', '.join(support.tasks)})", file=sys.stderr)
    print("\nChoose one explicitly:", file=sys.stderr)
    for family, _ in error.matches:
        command = shlex.join(["trtmc", *arguments, "--family", family])
        print(f"  {command}", file=sys.stderr)
