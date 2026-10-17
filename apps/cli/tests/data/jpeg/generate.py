# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the JPEG and lossless RGB golden pairs with Pillow.

The goldens contain Pillow's decoded pixels, not the uncompressed source.
Small odd dimensions and sharp chroma transitions exercise decoder rounding,
subsampling boundaries, and progressive scans independently of any model.
"""

from pathlib import Path

from PIL import Image

root = Path(__file__).parent
image = Image.new("RGB", (37, 29))
image.putdata([((x * 37 + y * 19) % 256, (x * 11 + y * 71) % 256,
                (x * 83 + y * 7) % 256)
               for y in range(image.height) for x in range(image.width)])
for name, source, options in [
    ("rgb-444", image, {"subsampling": 0}),
    ("rgb-422", image, {"subsampling": 1}),
    ("rgb-420", image, {"subsampling": 2}),
    ("progressive-420", image, {"subsampling": 2, "progressive": True}),
    ("grayscale", image.convert("L"), {}),
]:
    source.save(root / f"{name}.jpg", quality=93, **options)
    with Image.open(root / f"{name}.jpg") as decoded:
        decoded.convert("RGB").save(root / f"{name}.png")
