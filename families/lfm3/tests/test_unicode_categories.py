# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline generation of the native pretokenizer's Unicode ranges."""

from __future__ import annotations

from families.lfm3.tools import generate_unicode_data


def test_categories_expand_blocks_and_merge_only_adjacent_matches(tmp_path):
    source = tmp_path / "UnicodeData.txt"
    source.write_text(
        "0030;DIGIT ZERO;Nd\n"
        "0031;DIGIT ONE;Nd\n"
        "0041;LATIN CAPITAL LETTER A;Lu\n"
        "0042;LATIN CAPITAL LETTER B;Lu\n"
        "00B2;SUPERSCRIPT TWO;No\n"
        "0300;COMBINING GRAVE ACCENT;Mn\n"
        "216B;ROMAN NUMERAL TWELVE;Nl\n"
        "3400;<CJK Ideograph, First>;Lo\n"
        "3402;<CJK Ideograph, Last>;Lo\n",
        encoding="utf-8",
    )
    values = generate_unicode_data.categories(source)
    assert generate_unicode_data.ranges(values, "L") == [(0x41, 0x42), (0x3400, 0x3402)]
    assert generate_unicode_data.ranges(values, "N") == [
        (0x30, 0x31),
        (0xB2, 0xB2),
        (0x216B, 0x216B),
    ]
    assert values[0x300] == "."
    assert values[0x10FFFF] == "."


def test_cli_emits_both_native_tables(tmp_path, monkeypatch):
    source = tmp_path / "UnicodeData.txt"
    source.write_text("0041;LATIN CAPITAL LETTER A;Lu\n0030;DIGIT ZERO;Nd\n", encoding="utf-8")
    output = tmp_path / "unicode_categories.h"
    monkeypatch.setattr("sys.argv", ["generate_unicode_data", str(source), str(output)])
    generate_unicode_data.main()
    header = output.read_text(encoding="utf-8")
    assert "std::array<Range, 1> kLetters" in header
    assert "std::array<Range, 1> kNumbers" in header
    assert "{0x41, 0x41}" in header
    assert "{0x30, 0x30}" in header
    assert header.endswith("\n")
