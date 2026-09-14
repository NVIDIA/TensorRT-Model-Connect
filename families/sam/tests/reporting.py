# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded visual diagnostics for this family's existing mask comparison."""

from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


def _selected_masks(output):
    import numpy as np

    masks = np.asarray(output["masks"])
    if masks.ndim == 1:
        masks = masks.reshape(int(output["num_masks"]), int(output["height"]), int(output["width"]))
    if masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"unsupported mask dimensions: {masks.shape}")
    return masks[:2] > 0.0, int(masks.shape[0])


def record_mask_views(actual: dict, expected: dict, source: Path, directory: Path) -> None:
    if not evidence_enabled():
        return
    views, selected = [], {}
    for role, output in (("Native", actual), ("Reference", expected)):
        try:
            import numpy as np
            from PIL import Image

            masks, count = _selected_masks(output)
            selected[role] = masks
            record_evidence(
                "mask_view_summary",
                {
                    role.lower(): {
                        "total_masks": count,
                        "displayed_masks": len(masks),
                        "boundary": "foreground logits > 0",
                    }
                },
            )
            if count == 0:
                views.append({"title": f"{role} masks", "caption": "No masks were produced."})
            directory.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                base = image.convert("RGB")
                base.thumbnail((768, 768))
            for index, mask in enumerate(masks):
                rendered = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    base.size, Image.Resampling.NEAREST
                )
                overlay = Image.composite(
                    Image.blend(base, Image.new("RGB", base.size, (45, 220, 115)), 0.55),
                    base,
                    rendered,
                )
                path = directory / f"{role.lower()}-mask-{index}.png"
                overlay.save(path)
                views.append(
                    {
                        "title": f"{role} mask {index}",
                        "image": path,
                        "caption": f"Green: foreground; {int(mask.sum())} pixels; "
                        f"{mask.shape[1]} x {mask.shape[0]}. Showing up to 2 of {count} masks. "
                        "Display resized with nearest-neighbor masks; foreground logits > 0.",
                    }
                )
        except Exception as error:
            views.append(
                {
                    "title": f"{role} masks",
                    "caption": f"Visualization unavailable: {type(error).__name__}: {error}",
                }
            )
    try:
        for index, (left, right) in enumerate(
            zip(selected.get("Native", []), selected.get("Reference", []))
        ):
            if left.shape != right.shape:
                left = np.asarray(
                    Image.fromarray(left.astype(np.uint8) * 255).resize(
                        (right.shape[1], right.shape[0]), Image.Resampling.NEAREST
                    )
                ).astype(bool)
            mismatch = left != right
            pixels = np.zeros((*right.shape, 3), dtype=np.uint8)
            pixels[mismatch] = (255, 70, 70)
            picture = Image.fromarray(pixels)
            picture.thumbnail((768, 768), Image.Resampling.NEAREST)
            path = directory / f"mask-{index}-disagreement.png"
            picture.save(path)
            views.append(
                {
                    "title": f"Mask {index} disagreement",
                    "image": path,
                    "caption": f"Red: {int(mismatch.sum())} differing pixels out of {mismatch.size}; "
                    "black: agreement. The original comparator determines pass/fail.",
                }
            )
    except Exception as error:
        record_evidence(
            "mask_view_summary", {"disagreement_unavailable": f"{type(error).__name__}: {error}"}
        )
    record_evidence("views", views[:6])
