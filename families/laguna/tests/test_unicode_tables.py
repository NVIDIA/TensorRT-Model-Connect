# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated Unicode tables of the Laguna pre-tokenizer."""

from __future__ import annotations

import bisect
import re
from pathlib import Path

from families.laguna.tools import generate_unicode_tables


_HEADER = Path(__file__).resolve().parents[1] / "runtime" / "unicode_tables.h"

# Code points with their Unicode 16.0.0 major category: common scripts, characters
# inside First/Last blocks (CJK, Hangul), and characters first assigned in Unicode
# 16.0 (Cyrillic TJE, Garay, Kirat Rai, Todhri, Sunuwar, Gurung Khema, outlined
# digits), which tell these tables apart from ones generated from an older database.
_LETTERS = (
    0x41,
    0x7A,
    0xDF,
    0xE9,
    0x1C5,
    0x2B0,
    0x3A9,
    0x416,
    0x5D0,
    0x628,
    0x915,
    0xE01,
    0x3042,
    0x4E2D,
    0xAC00,
    0x20000,
    0x1C89,
    0x10D50,
    0x16D43,
    0x105C0,
    0x11BC0,
    0x16100,
)
_NUMBERS = (0x30, 0x39, 0xB2, 0xBD, 0x663, 0x216B, 0x3007, 0x10D40, 0x1CCF9)
_NEITHER = (0x20, 0x21, 0x2D, 0x5F, 0x301, 0x378, 0x200D, 0x20AC, 0xD800, 0xE000, 0x1F600)

_SAMPLE = """0030;DIGIT ZERO;Nd;0;EN;;0;0;0;N;;;;;
0031;DIGIT ONE;Nd;0;EN;;1;1;1;N;;;;;
0041;LATIN CAPITAL LETTER A;Lu;0;L;;;;;N;;;;0061;
0042;LATIN CAPITAL LETTER B;Lu;0;L;;;;;N;;;;0062;
0300;COMBINING GRAVE ACCENT;Mn;230;NSM;;;;;N;NON-SPACING GRAVE;;;;
3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
4DBF;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
"""


def _header_tables() -> dict[str, list[tuple[int, int]]]:
    text = _HEADER.read_text(encoding="utf-8")
    tables = {}
    for name in ("kLetterRanges", "kNumberRanges"):
        body = text.split(f"{name}[] = {{", 1)[1].split("};", 1)[0]
        tables[name] = [
            (int(lo, 16), int(hi, 16))
            for lo, hi in re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", body)
        ]
    return tables


def test_generator_expands_first_last_blocks_and_merges_runs() -> None:
    values = generate_unicode_tables.categories(_SAMPLE)
    assert generate_unicode_tables.ranges(values, "L") == [(0x41, 0x42), (0x3400, 0x4DBF)]
    assert generate_unicode_tables.ranges(values, "N") == [(0x30, 0x31)]
    rendered = generate_unicode_tables.render(_SAMPLE, "test")
    assert "{0x3400, 0x4DBF}" in rendered and "0x300" not in rendered


def test_committed_tables_are_sorted_and_disjoint() -> None:
    for name, ranges in _header_tables().items():
        assert ranges, name
        assert all(lo <= hi for lo, hi in ranges), name
        assert all(prev[1] + 1 < cur[0] for prev, cur in zip(ranges, ranges[1:])), name


def _contains(ranges: list[tuple[int, int]], codepoint: int) -> bool:
    index = bisect.bisect_right(ranges, (codepoint, 0x10FFFF)) - 1
    return index >= 0 and ranges[index][0] <= codepoint <= ranges[index][1]


def _major_category(tables: dict[str, list[tuple[int, int]]], codepoint: int) -> str:
    in_letters = _contains(tables["kLetterRanges"], codepoint)
    in_numbers = _contains(tables["kNumberRanges"], codepoint)
    assert not (in_letters and in_numbers), hex(codepoint)
    return "L" if in_letters else "N" if in_numbers else ""


def test_committed_tables_classify_reference_code_points() -> None:
    tables = _header_tables()
    for expected, codepoints in (("L", _LETTERS), ("N", _NUMBERS), ("", _NEITHER)):
        for codepoint in codepoints:
            assert _major_category(tables, codepoint) == expected, hex(codepoint)
