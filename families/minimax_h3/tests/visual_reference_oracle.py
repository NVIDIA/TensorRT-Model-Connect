# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the native host moment stitcher with pinned upstream methods."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3


class UpstreamStitch:
    # Invoke the actual upstream methods without constructing the unrelated
    # multi-GB decoder. No mathematical operation is reimplemented here.
    _blend = AutoencoderKLMiniMaxH3._blend
    _stitch_tiles = AutoencoderKLMiniMaxH3._stitch_tiles
    _split_tiles = AutoencoderKLMiniMaxH3._split_tiles
    spatial_compression_ratio = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    raw = subprocess.check_output([str(args.executable), "--dump-upstream-fixture"])
    actual = np.frombuffer(raw, dtype=np.float32).reshape(1, 48, 1, 17, 18)
    index = torch.arange(48 * 16 * 16)
    tiles = [
        [
            (((index * 13 + row * 37 + col * 101) % 251 - 125).float() / 32).reshape(
                1, 48, 1, 16, 16
            )
            for col in range(2)
        ]
        for row in range(2)
    ]
    oracle = UpstreamStitch()
    _, _, vertical = oracle._split_tiles(272, 256, 64)
    _, _, horizontal = oracle._split_tiles(288, 256, 64)
    expected = oracle._stitch_tiles(
        tiles, [value // 16 for value in vertical], [value // 16 for value in horizontal]
    ).numpy()
    args.output.mkdir(mode=0o700)
    np.save(args.output / "upstream-stitched-moments.npy", expected, allow_pickle=False)
    np.save(args.output / "native-stitched-moments.npy", actual, allow_pickle=False)
    report = dict(
        upstream="diffusers==0.40.0",
        max_abs_error=float(np.abs(actual - expected).max()),
        compared_elements=actual.size,
        rtol=1.3e-6,
        atol=1e-5,
        posterior_moments_only=True,
        sampling_qualified=False,
    )
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    np.testing.assert_allclose(actual, expected, rtol=1.3e-6, atol=1e-5, equal_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
