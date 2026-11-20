# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated letter and number tables of the native pre-tokenizer.

``runtime/unicode_categories.h`` answers ``\\p{L}`` and ``\\p{N}`` for the tokenizer. The generator
is tested on small inputs; the committed header is checked for its structure and against the
classes of the reference tokenizer's own regular-expression engine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from families.qwen_image2_1.tools import generate_unicode_categories as generator

HEADER = Path(__file__).resolve().parents[1] / "runtime" / "unicode_categories.h"

# General_Category L* and N* in Unicode 16.0.0.
LETTERS = 141028
NUMBERS = 1911


def _table(name: str) -> list[tuple[int, int]]:
    text = HEADER.read_text(encoding="utf-8")
    declared = re.search(rf"std::array<CodePointRange, (\d+)> {name}\{{\{{(.*?)\}}\}};", text, re.S)
    assert declared is not None, name
    entries = [
        (int(first, 16), int(last, 16))
        for first, last in re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", declared[2])
    ]
    assert len(entries) == int(declared[1])
    return entries


def _members(entries: list[tuple[int, int]]) -> set[int]:
    return {code_point for first, last in entries for code_point in range(first, last + 1)}


def test_generator_reads_single_entries_and_first_last_ranges() -> None:
    categories = generator.major_categories(
        "0030;DIGIT ZERO;Nd\n"
        "0031;DIGIT ONE;Nd\n"
        "0041;LATIN CAPITAL LETTER A;Lu\n"
        "0042;LATIN CAPITAL LETTER B;Lu\n"
        "00B2;SUPERSCRIPT TWO;No\n"
        "0300;COMBINING GRAVE ACCENT;Mn\n"
        "216B;ROMAN NUMERAL TWELVE;Nl\n"
        "3400;<CJK Ideograph Extension A, First>;Lo\n"
        "3402;<CJK Ideograph Extension A, Last>;Lo\n"
    )
    assert len(categories) == 0x110000
    assert generator.ranges(categories, "L") == [(0x41, 0x42), (0x3400, 0x3402)]
    assert generator.ranges(categories, "N") == [(0x30, 0x31), (0xB2, 0xB2), (0x216B, 0x216B)]
    assert categories[0x300] == "M" and categories[0x3401] == "L"
    assert categories[0x3403] == "C" and categories[0x10FFFF] == "C"


@pytest.mark.parametrize(
    ("unicode_data", "message"),
    [
        ("3400;<CJK, First>;Lo\n", "never closed"),
        ("3402;<CJK, Last>;Lo\n", "without a matching start"),
        ("3400;<CJK, First>;Lo\n3402;<Other, Last>;Nd\n", "without a matching start"),
        ("3400;<CJK, First>;Lo\n3401;CJK;Lo\n3402;<CJK, Last>;Lo\n", "inside an open range"),
        ("3400;<CJK, First>;Lo\n3500;<CJK, First>;Lo\n", "opened twice"),
    ],
)
def test_generator_rejects_broken_ranges(unicode_data: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generator.major_categories(unicode_data)


def test_generator_writes_both_tables(tmp_path) -> None:
    source = tmp_path / "UnicodeData.txt"
    source.write_text("0030;DIGIT ZERO;Nd\n0041;LATIN CAPITAL LETTER A;Lu\n", encoding="utf-8")
    output = tmp_path / "unicode_categories.h"
    generator.main([str(source), str(output)])
    header = output.read_text(encoding="utf-8")
    assert "std::array<CodePointRange, 1> kLetterRanges{{\n    {0x0041, 0x0041},\n}};" in header
    assert "std::array<CodePointRange, 1> kNumberRanges{{\n    {0x0030, 0x0030},\n}};" in header
    assert f"Unicode Character Database {generator.UNICODE_VERSION}" in header
    assert header.endswith("} // namespace trtmc::qwen_image2_1::unicode\n")


@pytest.mark.parametrize(("name", "size"), [("kLetterRanges", LETTERS), ("kNumberRanges", NUMBERS)])
def test_committed_tables_are_sorted_maximal_runs_of_the_expected_size(name: str, size: int) -> None:
    entries = _table(name)
    assert all(first <= last for first, last in entries)
    # Sorted, disjoint, and never adjacent: the binary search in the runtime relies on the order.
    assert all(left[1] + 1 < right[0] for left, right in zip(entries, entries[1:]))
    assert entries[-1][1] <= 0x10FFFF
    assert sum(last - first + 1 for first, last in entries) == size


def test_committed_tables_do_not_overlap() -> None:
    assert not _members(_table("kLetterRanges")) & _members(_table("kNumberRanges"))


def _reference_class(pattern: str) -> set[int]:
    """The code points the reference tokenizer's regex engine matches with ``pattern``."""
    tokenizers = pytest.importorskip("tokenizers")
    marker = "\U0010fffd"  # private use: neither a letter nor a number
    replace = tokenizers.normalizers.Replace(tokenizers.Regex(pattern), marker)
    result: set[int] = set()
    for start in range(0, 0x110000, 0x4000):
        points = [cp for cp in range(start, start + 0x4000) if not 0xD800 <= cp <= 0xDFFF]
        replaced = replace.normalize_str("".join(map(chr, points)))
        assert len(replaced) == len(points)
        result.update(cp for cp, ch in zip(points, replaced) if ch == marker and cp != ord(marker))
    return result


@pytest.mark.parametrize(
    ("name", "pattern", "other"),
    [("kLetterRanges", r"\p{L}", r"\p{N}"), ("kNumberRanges", r"\p{N}", r"\p{L}")],
)
def test_committed_tables_agree_with_the_reference_regex_classes(
    name: str, pattern: str, other: str
) -> None:
    table = _members(_table(name))
    # A reference built on a later Unicode version may know more letters and numbers; it must
    # never disagree about a code point the table classifies. Together with the pinned sizes
    # above, a reference on the table's Unicode version makes this an equality.
    assert table <= _reference_class(pattern)
    assert not table & _reference_class(other)
