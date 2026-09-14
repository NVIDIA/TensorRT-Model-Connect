# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class label and disagreement views of the masks used by the comparator."""

from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


def _label_colors(labels):
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    return np.stack(
        ((labels * 37 + 31) % 256, (labels * 73 + 67) % 256, (labels * 109 + 101) % 256), axis=-1
    ).astype(np.uint8)


def record_mask_views(actual: dict, expected: dict, source: Path, directory: Path) -> None:
    if not evidence_enabled():
        return
    views, masks = [], {}
    for role, output in (("Native", actual), ("Reference", expected)):
        try:
            import numpy as np
            from PIL import Image

            mask = np.asarray(output["mask"])
            if mask.ndim != 2:
                raise ValueError(f"unsupported label map dimensions: {mask.shape}")
            masks[role] = mask
            directory.mkdir(parents=True, exist_ok=True)
            picture = Image.fromarray(_label_colors(mask))
            picture.thumbnail((768, 768), Image.Resampling.NEAREST)
            path = directory / f"{role.lower()}-labels.png"
            picture.save(path)
            views.append(
                {
                    "title": f"{role} class labels",
                    "image": path,
                    "caption": f"{mask.shape[1]} x {mask.shape[0]}; "
                    "identical class IDs use identical colors in both views.",
                }
            )
            with Image.open(source) as image:
                base = image.convert("RGB")
                base.thumbnail((768, 768))
            overlay = Image.blend(base, picture.resize(base.size, Image.Resampling.NEAREST), 0.5)
            path = directory / f"{role.lower()}-overlay.png"
            overlay.save(path)
            views.append(
                {
                    "title": f"{role} overlay",
                    "image": path,
                    "caption": "Source image with 50% class colors; display only.",
                }
            )
        except Exception as error:
            views.append(
                {
                    "title": f"{role} labels",
                    "caption": f"Visualization unavailable: {type(error).__name__}: {error}",
                }
            )
    try:
        from PIL import ImageDraw

        left, right = masks.get("Native"), masks.get("Reference")
        if left is not None and right is not None and left.shape == right.shape:
            mismatch = left != right
            pixels = np.zeros((*right.shape, 3), dtype=np.uint8)
            pixels[mismatch] = (255, 70, 70)
            picture = Image.fromarray(pixels)
            picture.thumbnail((768, 768), Image.Resampling.NEAREST)
            path = directory / "label-disagreement.png"
            picture.save(path)
            views.append(
                {
                    "title": "Pixel disagreement",
                    "image": path,
                    "caption": f"Red: {int(mismatch.sum())} differing pixels out of "
                    f"{mismatch.size}; black: agreement. Original pixel accuracy and "
                    "mIoU assertions determine pass/fail.",
                }
            )
        if masks:
            labels = np.unique(np.concatenate([np.unique(mask) for mask in masks.values()]))
            shown = labels[:64]
            legend = Image.new("RGB", (640, max(1, (len(shown) + 3) // 4) * 24), "white")
            drawing = ImageDraw.Draw(legend)
            for index, label in enumerate(shown):
                x, y = (index % 4) * 160, (index // 4) * 24
                color = tuple(int(value) for value in _label_colors(label))
                drawing.rectangle((x + 2, y + 4, x + 20, y + 20), fill=color)
                drawing.text((x + 26, y + 5), f"class {int(label)}", fill="black")
            path = directory / "class-color-key.png"
            legend.save(path)
            views.append(
                {
                    "title": "Class color key",
                    "image": path,
                    "caption": f"Showing {len(shown)} of {len(labels)} class IDs. "
                    "Colors repeat after 256 IDs; numeric IDs remain authoritative.",
                }
            )
    except Exception as error:
        record_evidence(
            "mask_view_summary", {"disagreement_unavailable": f"{type(error).__name__}: {error}"}
        )
    record_evidence("views", views[:6])
