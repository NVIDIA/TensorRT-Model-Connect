# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the already-computed independent reference visible in the CI report."""

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_image_views(reference, directory):
    if not evidence_enabled():
        return
    try:
        import numpy as np
        from PIL import Image

        path = directory / "reference-preview.png"
        Image.fromarray((np.clip(reference, 0, 1) * 255).astype(np.uint8)).save(path)
        record_evidence("reference", {
            "image": path,
            "preview": "Display-only clipping to [0, 1] and conversion to RGB8; comparisons use the original floating-point image.",
        })
        record_evidence("views", [
            {"title": "Sample 0 / Native", "image": directory / "native.png", "caption": "Native output image."},
            {"title": "Sample 0 / Reference", "image": path, "caption": "Independent reference output, converted to RGB8 for display only."},
        ])
    except Exception as error:
        record_evidence("reference", {"preview_unavailable": f"{type(error).__name__}: {error}"})
