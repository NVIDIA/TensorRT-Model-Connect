# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the printed single-line images of the TrOCR E2E cases.

usage: python generate.py

TrOCR reads one line of text per image. The lines are rendered with Pillow's
built-in default font (Aileron Regular, CC0 1.0), in the style of the receipts
the checkpoint was fine-tuned on. Their sizes cover the resize regimes of the
encoder's 384 x 384 input: lines wider than 384 pixels (shrunk horizontally),
``printed_small.png`` (smaller grey text on an off-white ground, enlarged along
both axes), ``printed_tight.png`` (a line with narrow margins, enlarged about
fifteen times vertically) and ``printed_tall.png`` (a canvas taller than 384
rows and taller than wide, which Pillow resizes vertically first).

The checked-in PNG files are the test inputs; rerunning this script with a
different Pillow or FreeType version may rasterize the lines slightly
differently.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# name: (text, font size, margin, text colour, background colour)
LINES = {
    "printed_total.png": ("TOTAL AMOUNT: RM 25.80", 32, 12, "black", "white"),
    "printed_thanks.png": ("Thank you, please come again!", 32, 12, "black", "white"),
    "printed_receipt.png": ("2X ICED LATTE 12.50 SST 6% 0.75 CASH 20.00 CHANGE 6.75", 32, 12,
                            "black", "white"),
    "printed_small.png": ("Subtotal 41.20 Rounding -0.02", 18, 6, (64, 64, 64), (242, 242, 242)),
    "printed_tight.png": ("CARD VISA 4821 APPROVED 41.18", 28, 3, "black", "white"),
}
# name: (text, canvas width, canvas height)
TALL = {"printed_tall.png": ("RM 12.50", 240, 420)}


def _printed(text: str, size: int = 32, margin: int = 12, fill="black",
             background="white") -> Image.Image:
    font = ImageFont.load_default(size=size)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGB", (right - left + 2 * margin, bottom - top + 2 * margin), background)
    ImageDraw.Draw(image).text((margin - left, margin - top), text, fill=fill, font=font)
    return image


def main() -> None:
    for name, (text, size, margin, fill, background) in LINES.items():
        _printed(text, size, margin, fill, background).save(HERE / name)
    for name, (text, width, height) in TALL.items():
        line = _printed(text)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(line, ((width - line.width) // 2, (height - line.height) // 2))
        canvas.save(HERE / name)


if __name__ == "__main__":
    main()
