# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an older single-file XLSR state dictionary before building."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise ValueError("wav2vec2 checkpoint must contain a nonempty tensor state dictionary")
    save_file({name: value.contiguous() for name, value in state.items()}, str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
