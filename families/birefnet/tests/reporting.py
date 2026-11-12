# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display the exact binary masks compared by the BiRefNet testcase."""

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_mask_views(actual, expected, directory):
    if not evidence_enabled():
        return
    views = []
    try:
        import numpy as np
        from PIL import Image

        native = np.asarray(actual["mask"], dtype=np.int32).reshape(actual["height"], actual["width"])
        reference = np.asarray(expected)
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in (("Native", native), ("Reference", reference)):
            path = directory / f"{name.lower()}-mask.png"
            Image.fromarray((values == 1).astype(np.uint8) * 255).save(path)
            views.append({"title": f"Sample 0 / {name}", "image": path,
                          "caption": f"White: foreground ID 1; black: background. Original {values.shape[1]} x {values.shape[0]} mask; display only."})
        if native.shape == reference.shape:
            path = directory / "mask-disagreement.png"
            pixels = np.zeros((*native.shape, 3), dtype=np.uint8)
            pixels[native != reference] = (255, 70, 70)
            Image.fromarray(pixels).save(path)
            views.append({"title": "Mask disagreement", "image": path,
                          "caption": "Red: different mask IDs; black: agreement. The original IoU comparison determines pass/fail."})
        record_evidence("views", views)
    except Exception as error:
        record_evidence("views", views + [{"title": "Mask preview unavailable",
                                          "caption": f"{type(error).__name__}: {error}"}])
