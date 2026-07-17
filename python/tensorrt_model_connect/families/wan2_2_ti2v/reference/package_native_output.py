# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package native Wan2.2 PNG frames as video and a visual comparison sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--comparison-frames-dir", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0, 30, 60, 90, 120],
    )
    return parser.parse_args()


def _frame_paths(frames_dir: Path) -> list[Path]:
    paths = sorted(frames_dir.resolve().glob("frame_*.png"))
    if not paths:
        raise ValueError(f"No frame_*.png files found in {frames_dir}")
    expected = [frames_dir.resolve() / f"frame_{index:04d}.png" for index in range(len(paths))]
    if paths != expected:
        raise ValueError(f"Frame sequence in {frames_dir} is not contiguous from frame 0")
    return paths


def _write_video(paths: list[Path], output: Path, fps: int) -> None:
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for path in paths:
            with Image.open(path) as image:
                writer.append_data(np.asarray(image.convert("RGB")))
    finally:
        writer.close()


def _write_contact_sheet(
    native_paths: list[Path],
    reference_dir: Path,
    output: Path,
    indices: list[int],
) -> None:
    if not indices:
        raise ValueError("At least one contact-sheet sample index is required")
    invalid = [index for index in indices if index < 0 or index >= len(native_paths)]
    if invalid:
        raise ValueError(f"Contact-sheet indices are out of range: {invalid}")

    tile_width = 320
    tile_height = 176
    label_height = 24
    sheet = Image.new(
        "RGB",
        (tile_width * len(indices), (tile_height + label_height) * 2),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    rows = (
        ("Official", reference_dir.resolve()),
        ("TensorRT C++", native_paths[0].parent),
    )
    for row, (label, directory) in enumerate(rows):
        y = row * (tile_height + label_height)
        for column, index in enumerate(indices):
            path = directory / f"frame_{index:04d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                tile = image.convert("RGB").resize(
                    (tile_width, tile_height),
                    resample=Image.Resampling.LANCZOS,
                )
            x = column * tile_width
            sheet.paste(tile, (x, y + label_height))
            draw.text((x + 4, y + 4), f"{label} frame {index}", fill="white")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = _parse_args()
    paths = _frame_paths(args.frames_dir)
    _write_video(paths, args.output_video, args.fps)
    if (args.comparison_frames_dir is None) != (args.contact_sheet is None):
        raise ValueError("--comparison-frames-dir and --contact-sheet must be provided together")
    if args.comparison_frames_dir is not None:
        _write_contact_sheet(
            paths,
            args.comparison_frames_dir,
            args.contact_sheet,
            args.sample_indices,
        )
    print(f"Packaged {len(paths)} frames at {args.fps} fps to {args.output_video.resolve()}")


if __name__ == "__main__":
    main()
