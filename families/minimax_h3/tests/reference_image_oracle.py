# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native H3 image setup against pinned Pillow, before model execution."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def fixture(height, width, pattern):
    index = np.arange(height * width * 3, dtype=np.int64).reshape(height, width, 3)
    if pattern == 0:
        return ((index * 37 + index // 7 * 13) % 256).astype(np.uint8)
    y, x = np.indices((height, width))
    if pattern == 1:
        values = (x + y) % 2 * 255
    elif pattern == 2:
        values = ((x == width // 2) & (y == height // 2)) * 255
    else:
        values = np.full((height, width), 173)
    return np.repeat(values[..., None], 3, axis=-1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("Pillow") == "12.2.0"
    # Identity, either single-axis pass, mixed up/down, heavily downsampled,
    # non-square kernels, negative lobes/clipping, and production H3 geometries.
    cases = [
        (1, 1, 17, 33),
        (3, 5, 3, 5),
        (7, 11, 7, 29),
        (7, 11, 19, 11),
        (17, 29, 37, 13),
        (257, 263, 13, 7),
        (129, 64, 32, 128),
        (880, 1184, 2048, 2752),
        (513, 1025, 2048, 4096),
    ]
    comparisons = []
    for height, width, target_h, target_w in cases:
        for pattern in range(4):
            rgb = fixture(height, width, pattern)
            expected = np.asarray(
                Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.LANCZOS)
            )
            actual = np.frombuffer(
                subprocess.check_output(
                    [
                        str(args.executable),
                        "--resize-fixture",
                        *map(str, (height, width, target_h, target_w, pattern)),
                    ]
                ),
                dtype=np.uint8,
            ).reshape(target_h, target_w, 3)
            np.testing.assert_array_equal(actual, expected)
            comparisons.append(
                dict(
                    input=[height, width],
                    output=[target_h, target_w],
                    pattern=pattern,
                    compared_bytes=actual.size,
                    mismatched_bytes=0,
                )
            )
    actual = np.frombuffer(
        subprocess.check_output([str(args.executable), "--tile-fixture"]), dtype=np.float32
    ).reshape(3, 1, 32, 48)
    pixels = fixture(64, 96, 0)[16:48, 32:80].transpose(2, 0, 1).astype(np.float32)[:, None]
    mean = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1, 1)
    expected = (pixels / np.float32(255) - mean) / std
    np.testing.assert_array_equal(actual, expected)
    # Exercise the actual H3 setup entry point, not only the low-level resize.
    prepared_cases = []
    for height, width in ((880, 1184), (128, 129), (128, 131), (2048, 2048)):
        scale = 2048 / min(height, width)
        target_h = max(32, round(height * scale / 32) * 32)
        target_w = max(32, round(width * scale / 32) * 32)
        raw = subprocess.check_output(
            [str(args.executable), "--prepare-fixture", str(height), str(width), "0"]
        )
        np.testing.assert_array_equal(np.frombuffer(raw[:8], np.int32), [target_h, target_w])
        actual = np.frombuffer(raw[8:], np.uint8).reshape(target_h, target_w, 3)
        expected = np.asarray(
            Image.fromarray(fixture(height, width, 0)).resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
        )
        np.testing.assert_array_equal(actual, expected)
        prepared_cases.append(dict(input=[height, width], output=[target_h, target_w], exact=True))
    report = dict(
        pillow="12.2.0",
        resizes=comparisons,
        h3_prepared_images=prepared_cases,
        vae_input_exact=True,
        posterior_sampling_qualified=False,
        native_ref2va_video_qualified=False,
    )
    args.output.mkdir(mode=0o700)
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(dict(resize_cases=len(comparisons), vae_input_exact=True)))


if __name__ == "__main__":
    main()
