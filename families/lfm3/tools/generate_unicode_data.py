# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the pinned Unicode 16.0 L/N ranges from UnicodeData.txt."""

from __future__ import annotations

import argparse
from pathlib import Path


def categories(path: Path) -> list[str]:
    result = ["."] * 0x110000
    first = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(";")
        codepoint = int(fields[0], 16)
        category = fields[2][0]
        category = category if category in "LN" else "."
        if fields[1].endswith(", First>"):
            first = (codepoint, category)
        elif fields[1].endswith(", Last>"):
            start, value = first
            result[start : codepoint + 1] = [value] * (codepoint - start + 1)
            first = None
        else:
            result[codepoint] = category
    return result


def ranges(values: list[str], category: str) -> list[tuple[int, int]]:
    result = []
    for codepoint, value in enumerate(values):
        if value != category:
            continue
        if result and result[-1][1] == codepoint - 1:
            result[-1] = (result[-1][0], codepoint)
        else:
            result.append((codepoint, codepoint))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unicode_data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values = categories(args.unicode_data)
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "",
        "// Note (Jiaxin Deng): Generated Unicode 16.0 categories match the pinned regex engine.",
        "// Regeneration and Unicode data attribution are documented in the family README.",
        "namespace trtmc::lfm3::unicode_data {",
        "struct Range {",
        "    char32_t first;",
        "    char32_t last;",
        "};",
    ]
    for category, name in [("L", "kLetters"), ("N", "kNumbers")]:
        entries = ranges(values, category)
        lines.append(f"inline constexpr std::array<Range, {len(entries)}> {name}{{{{")
        lines.extend(f"    {{0x{start:X}, 0x{end:X}}}," for start, end in entries)
        lines.append("}};")
    lines.append("} // namespace trtmc::lfm3::unicode_data")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
