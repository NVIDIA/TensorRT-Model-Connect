# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the project-created OCR invoice; no third-party document is used."""

from pathlib import Path


def main():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (672, 448), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    draw.text((36, 22), "LIGHTON OCR", font=title, fill="black")
    draw.text((36, 76), "INVOICE 1042", font=font, fill="black")
    draw.text((36, 119), "Date: 2026-11-11", font=font, fill="black")
    draw.line((36, 170, 636, 170), fill="black", width=2)
    for y, item, amount in (
        (186, "Item", "Amount"),
        (237, "Paper", "12.00"),
        (288, "Ink", "18.00"),
        (360, "TOTAL", "30.00"),
    ):
        draw.text((36, y), item, font=font, fill="black")
        draw.text((475, y), amount, font=font, fill="black")
    draw.line((36, 341, 636, 341), fill="black", width=2)
    image.save(Path(__file__).with_name("invoice.png"))


if __name__ == "__main__":
    main()
