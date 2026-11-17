# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the synthetic document images of the layoutlm E2E cases.

usage: python generate.py DEJAVU_SANS_TTF DEJAVU_SANS_BOLD_TTF

The documents are invented for this test: a one-page invoice whose words fit
one 512-token window, a dense annual-report page whose words need several
overlapping windows, with facts near its start, middle and end, and a blank
page on which OCR finds no words. They are black
text on white, rendered with DejaVu Sans, and saved as 8-bit RGB PNG so that the
native runtime and the reference read identical pixels (a lossy format would let
two JPEG decoders disagree before OCR runs).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (90, 90, 90)


def _invoice(regular: str, bold: str) -> Image.Image:
    image = Image.new("RGB", (1240, 1754), WHITE)
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(bold, 56)
    heading = ImageFont.truetype(bold, 28)
    body = ImageFont.truetype(regular, 28)

    draw.text((90, 90), "Brightwater Supply Co.", font=heading, fill=BLACK)
    for row, line in enumerate(
        ("415 Meridian Avenue", "Portland, OR 97204", "billing@brightwater.example")
    ):
        draw.text((90, 136 + 40 * row), line, font=body, fill=GRAY)
    draw.text((830, 90), "INVOICE", font=title, fill=BLACK)

    fields = (
        ("Invoice No:", "BW-20417"),
        ("Invoice Date:", "March 3, 2026"),
        ("Due Date:", "April 2, 2026"),
        ("PO Number:", "7731-A"),
    )
    for row, (label, value) in enumerate(fields):
        draw.text((720, 200 + 44 * row), label, font=heading, fill=BLACK)
        draw.text((960, 200 + 44 * row), value, font=body, fill=BLACK)

    draw.text((90, 430), "Bill To:", font=heading, fill=BLACK)
    for row, line in enumerate(
        ("Northwind Traders", "1200 Harbor Road", "Seattle, WA 98101")
    ):
        draw.text((90, 476 + 40 * row), line, font=body, fill=BLACK)

    columns = (90, 640, 800, 1000)
    top = 700
    for x, label in zip(columns, ("Description", "Qty", "Unit Price", "Amount")):
        draw.text((x, top), label, font=heading, fill=BLACK)
    draw.line((90, top + 46, 1150, top + 46), fill=BLACK, width=3)
    items = (
        ("Stainless steel hinges", "40", "$12.50", "$500.00"),
        ("Oak cabinet doors", "12", "$185.00", "$2,220.00"),
        ("Brass drawer pulls", "60", "$7.25", "$435.00"),
        ("Installation labor", "16", "$65.00", "$1,040.00"),
    )
    for row, values in enumerate(items):
        for x, value in zip(columns, values):
            draw.text((x, top + 70 + 52 * row), value, font=body, fill=BLACK)
    totals_top = top + 70 + 52 * len(items) + 40
    draw.line((640, totals_top - 16, 1150, totals_top - 16), fill=BLACK, width=2)
    totals = (("Subtotal", "$4,195.00"), ("Sales Tax (8%)", "$335.60"), ("Total Due", "$4,530.60"))
    for row, (label, value) in enumerate(totals):
        font = heading if label == "Total Due" else body
        draw.text((640, totals_top + 48 * row), label, font=font, fill=BLACK)
        draw.text((1000, totals_top + 48 * row), value, font=font, fill=BLACK)

    notes = (
        "Payment terms: net 30 days. Please include the invoice number with your payment.",
        "Make checks payable to Brightwater Supply Co. or pay by bank transfer to account 0042-7781.",
        "Thank you for your business.",
    )
    for row, line in enumerate(notes):
        draw.text((90, 1420 + 44 * row), line, font=ImageFont.truetype(regular, 24), fill=GRAY)
    return image


_REPORT = (
    (
        "Harbor Lane Robotics - 2025 Annual Report",
        None,
    ),
    (
        "Letter to shareholders",
        "Harbor Lane Robotics designs autonomous carts that move parts between workstations in "
        "small and mid-sized factories. The company was founded in 2017 in Tacoma, Washington, "
        "and today serves customers in eleven countries. Our chief executive officer is Dana "
        "Whitfield and our chief financial officer is Marcus Oyelaran, who joined the company "
        "in January 2024 after twelve years at a regional logistics bank. This report covers "
        "the fiscal year that ended on December 31, 2025.",
    ),
    (
        "Operations",
        "During the year we shipped 2,340 carts, up from 1,615 in the previous year. The second "
        "assembly line in Tacoma opened in May and doubled our weekly capacity. We closed the "
        "pilot site in Boise because its lease could not be renewed on acceptable terms, and we "
        "moved its six engineers to the Tacoma campus. Average delivery time from order to "
        "installation fell from nine weeks to five weeks. Warranty claims stayed below two "
        "percent of units shipped for the third consecutive year, and the field service team "
        "resolved most requests within one business day.",
    ),
    (
        "People",
        "At year end the company had 418 employees, compared with 302 a year earlier. Most of "
        "the new positions were in manufacturing and field service. Voluntary turnover was "
        "seven percent. We introduced a tuition program for technicians and opened a training "
        "center next to the new assembly line. The board added two independent directors, "
        "bringing its size to seven members.",
    ),
    (
        "Products",
        "The Model C cart, launched in March, carries loads of up to 600 kilograms and can "
        "share aisles with forklifts. It accounted for forty percent of units shipped by the "
        "fourth quarter. A software update released in September added route planning across "
        "multiple buildings and reduced idle time by eighteen percent at our largest customer "
        "sites. The older Model A cart will be retired at the end of 2026, and existing owners "
        "will keep receiving software support until 2029.",
    ),
    (
        "Financial results",
        "Net revenue for the full year was $61.2 million, an increase of 38 percent. Gross "
        "margin improved to 41 percent from 36 percent, mainly because of lower battery costs "
        "and higher volume on the second line. Operating expenses grew more slowly than revenue. "
        "Net revenue in the fourth quarter was $18.4 million, the highest quarterly result in the "
        "history of the company. Cash and short-term investments at year end were $27.9 million "
        "and the company had no debt.",
    ),
    (
        "Outlook",
        "For 2026 we expect revenue between $78 million and $84 million. We plan to open a "
        "service hub in Rotterdam in the second quarter to support customers in Europe. Capital "
        "spending will focus on a third assembly line and on testing equipment for the next "
        "generation of batteries.",
    ),
)


def _report(regular: str, bold: str) -> Image.Image:
    image = Image.new("RGB", (1700, 2200), WHITE)
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(bold, 44)
    heading = ImageFont.truetype(bold, 30)
    body = ImageFont.truetype(regular, 26)
    y = 90
    for header, paragraph in _REPORT:
        if paragraph is None:
            draw.text((100, y), header, font=title, fill=BLACK)
            y += 90
            continue
        draw.text((100, y), header, font=heading, fill=BLACK)
        y += 50
        for line in textwrap.wrap(paragraph, width=104):
            draw.text((100, y), line, font=body, fill=BLACK)
            y += 38
        y += 30
    if y > image.height - 60:
        raise RuntimeError(f"report text overflows the page: {y}")
    return image


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    regular, bold = sys.argv[1], sys.argv[2]
    _invoice(regular, bold).save(HERE / "invoice.png", format="PNG", optimize=True)
    _report(regular, bold).save(HERE / "annual_report.png", format="PNG", optimize=True)
    Image.new("RGB", (400, 300), WHITE).save(HERE / "blank_page.png", format="PNG", optimize=True)


if __name__ == "__main__":
    main()
