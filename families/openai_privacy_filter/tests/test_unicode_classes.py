# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The o200k pre-tokenizer's code-point classes: generator and checked-in table."""

from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

from families.openai_privacy_filter.tools import generate_unicode_classes as generator

TABLE = Path(__file__).resolve().parents[1] / "runtime" / "unicode_classes.h"


def _table() -> list[tuple[int, int]]:
    text = TABLE.read_text(encoding="utf-8")
    return [
        (int(start, 16), int(cls)) for start, cls in re.findall(r"\{0x([0-9A-F]+), (\d)\}", text)
    ]


def _class_of(runs: list[tuple[int, int]], codepoint: int) -> int:
    return runs[bisect_right([start for start, _ in runs], codepoint) - 1][1]


def test_generator_expands_ranges_and_overrides_white_space(tmp_path):
    unicode_data = tmp_path / "UnicodeData.txt"
    unicode_data.write_text(
        "0009;<control>;Cc\n"
        "0020;SPACE;Zs\n"
        "0030;DIGIT ZERO;Nd\n"
        "0041;LATIN CAPITAL LETTER A;Lu\n"
        "0061;LATIN SMALL LETTER A;Ll\n"
        "01C5;LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON;Lt\n"
        "02B0;MODIFIER LETTER SMALL H;Lm\n"
        "0301;COMBINING ACUTE ACCENT;Mn\n"
        "3400;<CJK Ideograph Extension A, First>;Lo\n"
        "3402;<CJK Ideograph Extension A, Last>;Lo\n",
        encoding="utf-8",
    )
    prop_list = tmp_path / "PropList.txt"
    prop_list.write_text(
        "# comment\n0009..000D    ; White_Space # Cc\n0020          ; White_Space # Zs\n"
        "0041          ; ASCII_Hex_Digit # Lu\n",
        encoding="utf-8",
    )
    values = generator.classes(unicode_data, prop_list)
    assert [values[cp] for cp in (0x09, 0x0D, 0x20, 0x30, 0x41, 0x61, 0x1C5, 0x2B0, 0x301)] == [
        generator.SPACE,
        generator.SPACE,
        generator.SPACE,
        generator.NUMBER,
        generator.UPPER,
        generator.LOWER,
        generator.UPPER,
        generator.LETTER,
        generator.MARK,
    ]
    assert values[0x3400:0x3403] == [generator.LETTER] * 3
    assert values[0x3403] == values[0x10FFFF] == generator.OTHER
    assert generator.runs([0, 0, 1, 1, 0]) == [(0, 0), (2, 1), (4, 0)]
    rendered = generator.render(values)
    assert rendered.startswith("/*\n * SPDX-FileCopyrightText:")
    count = len(generator.runs(values))
    assert f"inline constexpr std::array<Run, {count}> kRuns{{{{" in rendered
    assert rendered.endswith(
        "// clang-format on\n} // namespace trtmc::openai_privacy_filter::unicode_classes\n"
    )
    assert "{0x00009, 6}," in rendered and "{0x03400, 3}," in rendered


def test_checked_in_table_is_well_formed():
    runs = _table()
    assert runs[0] == (0, generator.OTHER)
    assert all(left[0] < right[0] for left, right in zip(runs, runs[1:]))
    assert all(left[1] != right[1] for left, right in zip(runs, runs[1:]))
    assert {cls for _, cls in runs} == set(range(7))
    assert runs[-1][0] <= 0x10FFFF


def test_checked_in_table_follows_unicode_16():
    runs = _table()
    expected = {
        0x41: generator.UPPER,
        0x1C5: generator.UPPER,
        0x61: generator.LOWER,
        0x17F: generator.LOWER,
        0x2B0: generator.LETTER,
        0x4E2D: generator.LETTER,
        0x301: generator.MARK,
        0x94D: generator.MARK,
        0x30: generator.NUMBER,
        0x2160: generator.NUMBER,
        0xB2: generator.NUMBER,
        0x20: generator.SPACE,
        0x85: generator.SPACE,
        0xA0: generator.SPACE,
        0x3000: generator.SPACE,
        0x1C: generator.OTHER,
        0x200B: generator.OTHER,
        0xFEFF: generator.OTHER,
        0x1F44D: generator.OTHER,
        0xE000: generator.OTHER,
        # Assigned in Unicode 16.0: CYRILLIC CAPITAL/SMALL LETTER TJE.
        0x1C89: generator.UPPER,
        0x1C8A: generator.LOWER,
    }
    assert {cp: _class_of(runs, cp) for cp in expected} == expected
