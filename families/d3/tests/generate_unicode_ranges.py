# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the frozen Unicode 15 general-category ranges for D3 tokenization."""
from pathlib import Path
import unicodedata


def ranges(prefix):
    result = []
    start = previous = None
    for codepoint in range(0x110000):
        if not unicodedata.category(chr(codepoint)).startswith(prefix):
            continue
        if previous is None or codepoint != previous + 1:
            if previous is not None:
                result.append((start, previous))
            start = codepoint
        previous = codepoint
    if previous is not None:
        result.append((start, previous))
    return result


def main():
    if unicodedata.unidata_version != "15.0.0":
        raise RuntimeError("generation requires Python Unicode database 15.0.0")
    header = [
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "// Generated from Unicode 15.0.0 General_Category by tests/generate_unicode_ranges.py.",
        "#pragma once",
        "",
        "// clang-format off",
    ]
    for name, category in (("kLetterRanges", "L"), ("kMarkRanges", "M"), ("kNumberRanges", "N")):
        header.append(f"constexpr UnicodeRange {name}[] = {{")
        header.extend(f"    {{0x{lo:X}, 0x{hi:X}}}," for lo, hi in ranges(category))
        header.extend(("};", ""))
    header.append("// clang-format on")
    target = Path(__file__).resolve().parent.parent / "runtime/unicode_categories.h"
    target.write_text("\n".join(header) + "\n")


if __name__ == "__main__":
    main()
