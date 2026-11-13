# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display both compared depth maps on one numeric color scale."""

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_depth_views(native, reference, height, width, directory):
    if not evidence_enabled():
        return
    views = []
    try:
        import numpy as np
        from PIL import Image

        left = np.asarray(native).reshape(height, width)
        right = np.asarray(reference).reshape(height, width)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Nonfinite depth values cannot be shown on a finite shared scale")
        low, high = min(float(left.min()), float(right.min())), max(float(left.max()), float(right.max()))
        span = high - low if high > low else 1.0
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in (("Native", left), ("Reference", right)):
            level = np.clip((values - low) / span, 0, 1)
            rgb = np.stack((level, 1 - np.abs(2 * level - 1), 1 - level), axis=-1)
            path = directory / f"{name.lower()}-depth.png"
            Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
            views.append({"title": f"Sample 0 / {name}", "image": path,
                          "caption": f"Shared numeric scale {low:.6g} (blue) to {high:.6g} (red); original {width} x {height} layout. Display only; no metric units are assumed."})
        record_evidence("views", views)
    except Exception as error:
        record_evidence("views", views + [{"title": "Depth preview unavailable",
                                          "caption": f"{type(error).__name__}: {error}"}])
