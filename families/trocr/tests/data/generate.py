# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the printed single-line images of the TrOCR E2E cases.

usage: python generate.py

TrOCR reads one line of text per image. The printed lines are rendered black on
white with Pillow's built-in default font (Aileron Regular, CC0 1.0), in the
style of the receipts the checkpoint was fine-tuned on. ``printed_tall.png``
puts a line on a canvas taller than the encoder's 384 rows and taller than
wide, which Pillow resizes vertically first.

``screenshot_rope.png`` and ``screenshot_swiglu.png`` are not written here: they
are single-line crops of the project-created screenshot listed in
ASSET_LICENSES.md (boxes 28,287,532,319 and 3,173,379,205, left, top, right,
bottom), saved as PNG once and committed as they are.

The checked-in PNG files are the test inputs; rerunning this script with a
different Pillow or FreeType version may rasterize the lines slightly
differently.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

PRINTED = {
    "printed_total.png": "TOTAL AMOUNT: RM 25.80",
    "printed_thanks.png": "Thank you, please come again!",
    "printed_receipt.png": "2X ICED LATTE 12.50 SST 6% 0.75 CASH 20.00 CHANGE 6.75",
}
# name: (text, canvas width, canvas height)
TALL = {"printed_tall.png": ("RM 12.50", 240, 420)}


def _printed(text: str) -> Image.Image:
    font = ImageFont.load_default(size=32)
    left, top, right, bottom = font.getbbox(text)
    margin = 12
    image = Image.new("RGB", (right - left + 2 * margin, bottom - top + 2 * margin), "white")
    ImageDraw.Draw(image).text((margin - left, margin - top), text, fill="black", font=font)
    return image


def main() -> None:
    for name, text in PRINTED.items():
        _printed(text).save(HERE / name)
    for name, (text, width, height) in TALL.items():
        line = _printed(text)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(line, ((width - line.width) // 2, (height - line.height) // 2))
        canvas.save(HERE / name)


if __name__ == "__main__":
    main()
