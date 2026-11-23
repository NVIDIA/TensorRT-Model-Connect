# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a single-file WavLM state dictionary to safetensors before building."""

from __future__ import annotations

import argparse
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(source: Path, destination: Path) -> None:
    """Write the state dictionary in ``source`` to ``destination``, never replacing a file.

    The tensors go to a temporary file next to the destination, which a hard
    link then publishes: the link fails instead of replacing a destination that
    another conversion created meanwhile. The first check only saves loading the
    checkpoint when the destination is already there.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise ValueError("wavlm checkpoint must contain a nonempty tensor state dictionary")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file({name: value.contiguous() for name, value in state.items()}, str(temporary))
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
