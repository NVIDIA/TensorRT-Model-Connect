# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Unicode table generator and the generated runtime/unicode_tables.h."""

from __future__ import annotations

import re
from pathlib import Path

from families.glm_ocr.tools import generate_unicode_tables

TABLES = Path(__file__).resolve().parents[1] / "runtime" / "unicode_tables.h"

# A UnicodeData.txt excerpt: single code points, a First/Last range and categories the
# classes leave out (Cc outside U+0009-U+000D and U+0085, Mn, Po).
_SAMPLE = """\
0008;<control>;Cc;0;BN;;;;;N;BACKSPACE;;;;
0009;<control>;Cc;0;S;;;;;N;CHARACTER TABULATION;;;;
000A;<control>;Cc;0;B;;;;;N;LINE FEED (LF);;;;
0020;SPACE;Zs;0;WS;;;;;N;;;;;
0021;EXCLAMATION MARK;Po;0;ON;;;;;N;;;;;
0031;DIGIT ONE;Nd;0;EN;;1;1;1;N;;;;;
0041;LATIN CAPITAL LETTER A;Lu;0;L;;;;;N;;;;0061;
0042;LATIN CAPITAL LETTER B;Lu;0;L;;;;;N;;;;0062;
0085;<control>;Cc;0;B;;;;;N;NEXT LINE (NEL);;;;
0301;COMBINING ACUTE ACCENT;Mn;230;NSM;;;;;N;NON-SPACING ACUTE;;;;
2028;LINE SEPARATOR;Zl;0;WS;;;;;N;;;;;
2160;ROMAN NUMERAL ONE;Nl;0;L;<compat> 0049;;;1;N;;;;2170;
3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
4DBF;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
"""


def _tables(text: str) -> dict[str, list[tuple[int, int]]]:
    tables = {}
    for name, body in re.findall(r"inline constexpr Range (\w+)\[\] = \{(.*?)\};", text, re.S):
        tables[name] = [(int(first, 16), int(last, 16)) for first, last in
                        re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", body)]
    return tables


def test_generator_classes_and_layout(tmp_path, monkeypatch) -> None:
    source = tmp_path / "UnicodeData.txt"
    source.write_text(_SAMPLE, encoding="utf-8")
    output = tmp_path / "unicode_tables.h"
    monkeypatch.setattr("sys.argv", ["generate_unicode_tables", str(source), str(output)])
    generate_unicode_tables.main()
    text = output.read_text(encoding="utf-8")
    assert _tables(text) == {
        "kLetters": [(0x41, 0x42), (0x3400, 0x4DBF)],
        "kNumbers": [(0x31, 0x31), (0x2160, 0x2160)],
        "kSpaces": [(0x9, 0xD), (0x20, 0x20), (0x85, 0x85), (0x2028, 0x2028)],
    }
    assert "Unicode 16.0.0 character classes" in text
    assert all(len(line) <= 100 for line in text.splitlines())


def test_generated_tables_are_sorted_disjoint_unicode_16() -> None:
    text = TABLES.read_text(encoding="utf-8")
    assert "Unicode 16.0.0 character classes" in text
    tables = _tables(text)
    assert set(tables) == {"kLetters", "kNumbers", "kSpaces"}
    for ranges in tables.values():
        assert all(first <= last for first, last in ranges)
        assert all(left[1] + 1 < right[0] for left, right in zip(ranges, ranges[1:]))
    # \s of the tokenizers regex engine: U+0009-U+000D, U+0085 and Zs, Zl, Zp.
    assert tables["kSpaces"] == [
        (0x9, 0xD), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680),
        (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F),
        (0x3000, 0x3000)]
    letters = sum(last - first + 1 for first, last in tables["kLetters"])
    numbers = sum(last - first + 1 for first, last in tables["kNumbers"])
    assert (letters, numbers) == (141028, 1911)
