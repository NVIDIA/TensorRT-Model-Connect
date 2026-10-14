# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate integer-image preprocessing receipts from the pinned processor."""

import argparse
import json
import lzma
from pathlib import Path

if __package__:
    from .reference_source import reference_source
else:
    from reference_source import reference_source

CASES = [
    ("square", 256, 256, 64, 256, 10, True),
    ("wide", 128, 512, 64, 256, 10, True),
    ("tall", 777, 211, 64, 256, 10, True),
    ("large", 1024, 1536, 64, 256, 10, True),
    ("budget", 1024, 1536, 64, 128, 4, True),
    ("nosplit", 1024, 1536, 64, 256, 10, False),
]


def main():
    import numpy as np
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "fixtures/preprocessing-reference.json",
    )
    args = parser.parse_args()
    options, provenance = reference_source(args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint, **options)
    fixture_dir = args.output.parent
    data_dir = fixture_dir / "preprocessing"
    data_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, h, w, minimum, maximum, tiles, split in CASES:
        y, x = np.indices((h, w))
        pixels = np.stack(
            [(x * 13 + y * 7) % 256, (x * 3 + y * 17) % 256, (x + y * 31) % 256], axis=-1
        ).astype("uint8")
        output = processor(
            text="<image>",
            images=[Image.fromarray(pixels)],
            return_tensors="pt",
            min_image_tokens=minimum,
            max_image_tokens=maximum,
            max_tiles=tiles,
            do_image_splitting=split,
        )
        tokens = processor.tokenizer.decode(output.input_ids[0], skip_special_tokens=False)
        record = {
            "name": name,
            "arguments": [h, w, minimum, maximum, tiles, int(split)],
            "placeholders": tokens,
            "tiles": [],
        }
        for i, (rows, cols) in enumerate(output.spatial_shapes.tolist()):
            pixel_path = data_dir / f"{name}-{i}-pixels.xz.bin"
            mask_path = data_dir / f"{name}-{i}-mask.xz.bin"
            pixel_path.write_bytes(
                lzma.compress(output.pixel_values[i].numpy().tobytes(), format=lzma.FORMAT_XZ)
            )
            mask_path.write_bytes(
                lzma.compress(
                    output.pixel_attention_mask[i].numpy().astype("<i4").tobytes(),
                    format=lzma.FORMAT_XZ,
                )
            )
            record["tiles"].append(
                {
                    "shape": [rows, cols],
                    "pixels": str(pixel_path.relative_to(fixture_dir)),
                    "mask": str(mask_path.relative_to(fixture_dir)),
                }
            )
        results.append(record)
    args.output.write_text(
        json.dumps(
            {
                **provenance,
                "transformers": transformers.__version__,
                "compression": "xz",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
