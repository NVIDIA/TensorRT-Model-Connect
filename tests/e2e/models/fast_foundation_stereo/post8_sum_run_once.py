# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one standalone post8 candidate inference for CUDA sanitizers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tests.e2e.models.fast_foundation_stereo.post8_sum_oracle import (
    _Runner,
    _SHAPE,
    _pin_plugin_library,
)
from tests.e2e.models.fast_foundation_stereo.trt_runner import (
    load_native_plugin_libraries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, required=True)
    args = parser.parse_args()

    plugin_library = _pin_plugin_library(args.plugin_library)
    load_native_plugin_libraries([plugin_library])
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    linear = torch.randn(_SHAPE, generator=generator, dtype=torch.float16, device="cuda")
    skip = torch.randn(_SHAPE, generator=generator, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    runner = _Runner(args.engine.resolve(), ("candidate",))
    runner.run_once(linear, skip)


if __name__ == "__main__":
    main()
