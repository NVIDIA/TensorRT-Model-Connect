# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for discovering family-owned external-kernel slots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .kernel_slots import KernelSlot, load_family_kernel_slots


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="kernel_command", required=True)
    slots = commands.add_parser(
        "slots",
        help="List the external-kernel slots published by a model family",
    )
    slots.add_argument(
        "model",
        help="HF repo ID or local model directory",
    )
    slots.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face model revision",
    )


def _shape(value: tuple[int | str, ...] | str) -> str:
    if type(value) is str:
        return value
    return "[" + ", ".join(str(dimension) for dimension in value) + "]"


def _print_slot(slot: KernelSlot, config: object) -> None:
    print(slot.id)
    print(f"  {slot.description}")
    print("  inputs:")
    for tensor in slot.inputs:
        print(f"    {tensor.name}: {tensor.dtype} {_shape(tensor.shape)}")
    print("  outputs:")
    for tensor in slot.outputs:
        print(f"    {tensor.name}: {tensor.dtype} {_shape(tensor.shape)}")
    print(f"  workspace: {slot.workspace_bytes} bytes")
    instance_ids = slot.instances(config)
    print(f"  instances ({len(instance_ids)}):")
    for instance_id in instance_ids:
        print(f"    {instance_id}")
    if slot.model_arguments:
        print("  model arguments:")
        for argument in slot.model_arguments:
            print(f"    {argument.name}: {argument.type}")
    print("  call order: inputs, workspace (when nonzero), outputs, model arguments")


def _resolve_model_config(model: str, revision: str | None) -> tuple[object, str]:
    from ..config import ModelConfig
    from ..families import resolve_family_id

    model_path = Path(model)
    if model_path.is_dir():
        config = ModelConfig.from_dir(model_path)
    else:
        from huggingface_hub import hf_hub_download

        kwargs = {"revision": revision} if revision else {}
        config_path = hf_hub_download(
            repo_id=model,
            filename="config.json",
            **kwargs,
        )
        config = ModelConfig.from_json(Path(config_path).read_text(encoding="utf-8"))
    family = resolve_family_id(config)
    if family is None:
        raise ValueError(f"No Model Connect family recognizes {model!r}")
    return config, family


def run(arguments: argparse.Namespace) -> int:
    if arguments.kernel_command != "slots":
        return 1
    try:
        revision = getattr(arguments, "model_revision", None)
        config, family = _resolve_model_config(
            arguments.model,
            revision,
        )
        slots = load_family_kernel_slots(family)
        if not slots:
            print(f"Model family {family!r} publishes no kernel slots")
            return 0
        for index, slot in enumerate(slots):
            if index:
                print()
            _print_slot(slot, config)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
