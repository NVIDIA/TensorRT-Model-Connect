# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-delimited TensorRT post enqueue for Nsight Compute."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

try:
    from .trt_runner import SplitTensorRTRunner, load_native_plugin_libraries
except ImportError:  # Direct execution: python tests/.../profile_ncu.py
    from trt_runner import SplitTensorRTRunner, load_native_plugin_libraries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("feature_engine", type=Path)
    parser.add_argument("post_engine", type=Path)
    parser.add_argument(
        "--plugin-library",
        action="append",
        default=[],
        type=Path,
        help="family plugin DSO to load before TensorRT engine deserialization",
    )
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    root = args.model_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    load_native_plugin_libraries(args.plugin_library)
    model = SplitTensorRTRunner(args.feature_engine, args.post_engine)
    left = torch.rand((1, 3, 704, 704), device="cuda") * 255
    right = torch.rand_like(left) * 255

    def prepare_inputs():
        return model.run_feature(left, right)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        inputs = prepare_inputs()
        for _ in range(args.warmup):
            model.run_post(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    with torch.inference_mode(), torch.cuda.stream(stream):
        torch.cuda.nvtx.range_push("ffs_post")
        try:
            model.run_post(inputs)
            stream.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    print("profiled one post-engine enqueue")


if __name__ == "__main__":
    main()
