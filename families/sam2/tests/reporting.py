# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native mask views for the fixed SAM2 public-core operational probe."""

from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence

# The raw output layout is owned by tests/cpp/operational_probe.cpp.
_FRAME_COUNT, _HEIGHT, _WIDTH = 5, 1280, 1088


def record_mask_views(masks_path: Path, directory: Path) -> None:
    if not evidence_enabled():
        return
    views = []
    try:
        import numpy as np
        from PIL import Image

        if masks_path.stat().st_size != _FRAME_COUNT * _HEIGHT * _WIDTH:
            raise ValueError("native mask file does not match the operational probe layout")
        masks = np.memmap(
            masks_path, dtype=np.uint8, mode="r", shape=(_FRAME_COUNT, _HEIGHT, _WIDTH)
        )
        directory.mkdir(parents=True, exist_ok=True)
        for index, mask in enumerate(masks):
            picture = Image.fromarray((mask != 0).astype(np.uint8) * 255)
            picture.thumbnail((768, 768), Image.Resampling.NEAREST)
            path = directory / f"native-frame-{index}-mask.png"
            picture.save(path)
            views.append(
                {
                    "title": f"Native frame {index} mask",
                    "image": path,
                    "caption": "White: nonzero native mask. Contract-only operational "
                    "probe; no reference masks or numerical parity claim.",
                }
            )
        record_evidence("native_artifacts", {"masks": masks_path})
    except Exception as error:
        views.append(
            {
                "title": "Native masks",
                "caption": f"Visualization unavailable: {type(error).__name__}: {error}",
            }
        )
    record_evidence("views", views)
