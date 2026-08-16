# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-delimited TensorRT post enqueue for Nsight Compute."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("feature_engine", type=Path)
    parser.add_argument("post_engine", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    root = args.model_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import tensorrt as trt
    from core.foundation_stereo import TrtRunner
    from core.submodule import build_gwc_volume_triton

    checkpoint = root / "weights/23-36-37/model_best_bp2_serialize.pth"
    checkpoint_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_model.args.valid_iters = 8
    checkpoint_model.args.max_disp = 192
    model = TrtRunner(checkpoint_model.args, str(args.feature_engine), str(args.post_engine)).cuda()
    del checkpoint_model
    left = torch.rand((1, 3, 704, 704), device="cuda") * 255
    right = torch.rand_like(left) * 255
    input_names = model.get_io_tensor_names(model.post_engine, trt.TensorIOMode.INPUT)

    def prepare_inputs():
        feature = model.run_trt(
            model.feature_engine,
            model.feature_context,
            {"left": left, "right": right},
        )
        feature["gwc_volume"] = build_gwc_volume_triton(
            feature["features_left_04"].half(),
            feature["features_right_04"].half(),
            48,
            8,
            normalize=True,
        )
        return {name: value for name, value in feature.items() if name in input_names}

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        inputs = prepare_inputs()
        for _ in range(args.warmup):
            model.run_trt(model.post_engine, model.post_context, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    with torch.inference_mode(), torch.cuda.stream(stream):
        torch.cuda.nvtx.range_push("ffs_post")
        try:
            model.run_trt(model.post_engine, model.post_context, inputs)
            stream.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    print("profiled one post-engine enqueue")


if __name__ == "__main__":
    main()
