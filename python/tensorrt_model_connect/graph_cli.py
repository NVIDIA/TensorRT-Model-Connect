# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline CLI for selecting recipe or explicit TensorRT graph regions."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from typing import Any

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

    recipe_parser = commands.add_parser(
        "recipe",
        help="Use a family-owned shortcut for an exact graph region",
    )
    recipe_commands = recipe_parser.add_subparsers(
        dest="recipe_command",
        required=True,
    )
    recipe_list = recipe_commands.add_parser(
        "list",
        help="List recipes recorded in a graph snapshot",
    )
    recipe_list.add_argument("snapshot", help="Graph snapshot JSON")
    recipe_apply = recipe_commands.add_parser(
        "apply",
        help="Turn one recipe instance into a normal selection JSON",
    )
    recipe_apply.add_argument("snapshot", help="Graph snapshot JSON")
    recipe_apply.add_argument("recipe", help="Exact versioned recipe ID")
    recipe_apply.add_argument(
        "--instance",
        required=True,
        help="Exact recipe instance shown by 'graph recipe list'",
    )
    recipe_apply.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output selection JSON",
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


def _recipes(snapshot: object) -> list[dict[str, Any]]:
    raw = snapshot.metadata.get("graph_recipes", [])
    if type(raw) is not list:
        raise ValueError("snapshot metadata.graph_recipes must be an array")
    fields = {
        "id",
        "instance",
        "node_ids",
        "workspace_bytes",
        "extra_args",
        "output_shape_input",
    }
    result = []
    for index, value in enumerate(raw):
        where = f"metadata.graph_recipes[{index}]"
        if type(value) is not dict or set(value) != fields:
            raise ValueError(f"{where} has invalid fields")
        if type(value["id"]) is not str or not value["id"]:
            raise ValueError(f"{where}.id must be a non-empty string")
        if type(value["instance"]) is not str or not value["instance"]:
            raise ValueError(f"{where}.instance must be a non-empty string")
        node_ids = value["node_ids"]
        if (
            type(node_ids) is not list
            or not node_ids
            or any(type(node_id) is not str or not node_id for node_id in node_ids)
            or len(set(node_ids)) != len(node_ids)
        ):
            raise ValueError(f"{where}.node_ids must be unique non-empty strings")
        result.append(value)
    identities = [(value["id"], value["instance"]) for value in result]
    if len(identities) != len(set(identities)):
        raise ValueError("snapshot graph recipes contain a duplicate instance")
    return result


def _print_recipes(snapshot: object) -> None:
    print(f"graph: {snapshot.fingerprint}")
    print("RECIPE\tINSTANCE\tNODES")
    for recipe in _recipes(snapshot):
        print(
            f"{recipe['id']}\t{recipe['instance']}\t"
            f"{','.join(recipe['node_ids'])}"
        )


def _write_and_print_selection(snapshot: object, selection: object, output: str) -> None:
    write_selection(selection, output)
    print(f"Wrote selection to {output}")
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
        if arguments.graph_command == "recipe":
            if arguments.recipe_command == "list":
                _print_recipes(snapshot)
                return 0
            matches = [
                recipe
                for recipe in _recipes(snapshot)
                if recipe["id"] == arguments.recipe
                and recipe["instance"] == arguments.instance
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one graph recipe {arguments.recipe!r} instance "
                    f"{arguments.instance!r}, found {len(matches)}"
                )
            recipe = matches[0]
            selection = select_region(
                snapshot,
                recipe["node_ids"],
                binding_id=recipe["id"],
                workspace_bytes=recipe["workspace_bytes"],
                extra_args=recipe["extra_args"],
                output_shape_input=recipe["output_shape_input"],
            )
            _write_and_print_selection(snapshot, selection, arguments.output)
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
        _write_and_print_selection(snapshot, selection, arguments.output)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
