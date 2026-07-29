# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline CLI for listing and selecting nodes from a TensorRT graph snapshot."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys

from .graph_patch import load_snapshot, select_region, write_selection


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _extra_arg(value: str) -> dict[str, object]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"must be a JSON object: {exc.msg}") from exc
    if type(result) is not dict:
        raise argparse.ArgumentTypeError("must be a JSON object")
    if result.get("type") not in {"none", "int", "float", "ptr"}:
        raise argparse.ArgumentTypeError("type must be one of: none, int, float, ptr")
    return result


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="graph_command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Capture the raw TensorRT graph before engine compilation",
    )
    inspect_parser.add_argument("--snapshot", required=True, help="Output graph JSON")
    inspect_parser.add_argument(
        "--engine-role",
        choices=["prefill", "decode", "dual_profile"],
        default="decode",
    )
    inspect_parser.add_argument(
        "build_args",
        nargs=argparse.REMAINDER,
        help="Model and options passed verbatim to 'trtmc build'",
    )

    list_parser = commands.add_parser(
        "list",
        help="List node IDs in a saved graph snapshot",
    )
    list_parser.add_argument("snapshot", help="Graph snapshot JSON")
    list_parser.add_argument(
        "--match",
        default="*",
        metavar="GLOB",
        help="Show only IDs, ops, or names matching this glob (default: *)",
    )

    select_parser = commands.add_parser(
        "select",
        help="Select explicit node IDs for replacement",
    )
    select_parser.add_argument("snapshot", help="Graph snapshot JSON")
    select_parser.add_argument(
        "--nodes",
        nargs="+",
        required=True,
        metavar="NODE_ID",
        help="Node IDs copied from 'graph list'",
    )
    select_parser.add_argument(
        "--binding-id",
        required=True,
        help="FFI binding ID used by the replacement ([A-Za-z0-9_.@-]+)",
    )
    select_parser.add_argument(
        "--workspace-bytes",
        type=_nonnegative_int,
        default=0,
        help="Replacement workspace size in bytes, 0..2147483647 (default: 0)",
    )
    select_parser.add_argument(
        "--output-shape-like-input",
        type=_nonnegative_int,
        metavar="INPUT_INDEX",
        help="For a dynamic output, copy dimensions from this boundary input",
    )
    select_parser.add_argument(
        "--extra-arg",
        action="append",
        type=_extra_arg,
        default=[],
        metavar="JSON",
        help='Extra FFI argument, e.g. \'{"type":"int","value":32}\' (repeatable)',
    )
    select_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output selection JSON",
    )


def _print_snapshot(snapshot: object, pattern: str) -> None:
    print(f"graph: {snapshot.fingerprint}")
    print("ID\tOP\tNAME\tINPUTS\tOUTPUTS")
    for node in snapshot.nodes:
        lowered = pattern.lower()
        if not any(
            fnmatch.fnmatchcase(value.lower(), lowered)
            for value in (node.id, node.op, node.name)
        ):
            continue
        print(
            "\t".join(
                (
                    node.id,
                    node.op,
                    node.name,
                    ",".join(value if value is not None else "-" for value in node.inputs),
                    ",".join(value if value is not None else "-" for value in node.outputs),
                )
            )
        )


def run(arguments: argparse.Namespace) -> int:
    try:
        if arguments.graph_command == "inspect":
            build_args = list(arguments.build_args)
            if build_args[:1] == ["--"]:
                build_args.pop(0)
            if not build_args:
                raise ValueError("graph inspect requires model and build arguments")
            command = [
                sys.executable,
                "-m",
                "tensorrt_model_connect",
                "build",
                *build_args,
                "-o",
                os.devnull,
                "--graph-snapshot",
                arguments.snapshot,
                "--graph-role",
                arguments.engine_role,
            ]
            return subprocess.run(command, check=False).returncode

        snapshot = load_snapshot(arguments.snapshot)
        if arguments.graph_command == "list":
            _print_snapshot(snapshot, arguments.match)
            return 0
        if arguments.graph_command != "select":
            return 1

        selection = select_region(
            snapshot,
            arguments.nodes,
            binding_id=arguments.binding_id,
            workspace_bytes=arguments.workspace_bytes,
            extra_args=arguments.extra_arg,
            output_shape_input=arguments.output_shape_like_input,
        )
        write_selection(selection, arguments.output)
        print(f"Wrote selection to {arguments.output}")
        tensors = {tensor.id: tensor for tensor in snapshot.tensors}
        for kind, tensor_ids in (
            ("input", selection.input_tensor_ids),
            ("output", selection.output_tensor_ids),
        ):
            for index, tensor_id in enumerate(tensor_ids):
                tensor = tensors[tensor_id]
                print(
                    f"{kind}[{index}]: {tensor_id} name={tensor.name!r} "
                    f"dtype={tensor.dtype} shape={list(tensor.shape)}"
                )
        print(f"abi_sha256: {selection.abi_sha256}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
