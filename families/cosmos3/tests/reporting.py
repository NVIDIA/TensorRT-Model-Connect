# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned image previews from the outputs already used for comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.e2e_evidence import evidence_enabled, record_evidence


_MAX_FRAMES = 6


def _indices(count: int) -> list[int]:
    samples = min(_MAX_FRAMES, count)
    return (
        [round(index * (count - 1) / (samples - 1)) for index in range(samples)]
        if samples > 1
        else list(range(samples))
    )


def _files(output: Path) -> list[Path]:
    return (
        sorted(
            path
            for path in output.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm"}
        )
        if output.is_dir()
        else [output]
    )


def _thumbnail(source, path: Path) -> None:
    from PIL import Image

    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            image = image.convert("RGB")
    else:
        values = np.asarray(source)
        if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError("preview requires a finite HWC RGB frame")
        if values.dtype != np.uint8:
            values = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(values)
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def record_native_preview(output: Path) -> None:
    if not evidence_enabled():
        return
    try:
        files = _files(output)
        views = []
        for index in _indices(len(files)):
            image = output.parent / "native-report-views" / f"frame-{index:04d}.png"
            _thumbnail(files[index], image)
            views.append(
                {
                    "title": f"Native frame {index}",
                    "image": image,
                    "caption": f"Frame {index} of {len(files)}. Original file: {files[index]}",
                }
            )
        record_evidence("views", views)
        record_evidence(
            "native_artifacts",
            {
                "frame_count": len(files),
                "sampled_indices": _indices(len(files)),
                "original_location": f"Original files: {output}",
            },
        )
    except (ImportError, KeyError, TypeError, ValueError, OSError) as error:
        record_evidence("views", {"unavailable": str(error)})
