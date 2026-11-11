# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated Unicode tables of the Laguna pre-tokenizer."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from families.laguna.tools import generate_unicode_tables


_HEADER = Path(__file__).resolve().parents[1] / "runtime" / "unicode_tables.h"

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


@pytest.mark.skipif(
    unicodedata.unidata_version != "16.0.0", reason="needs the Unicode 16.0.0 database"
)
def test_committed_tables_match_the_unicode_database() -> None:
    expected = {
        major: generate_unicode_tables.ranges(
            {cp: unicodedata.category(chr(cp))[0] for cp in range(0x110000)}, major
        )
        for major in ("L", "N")
    }
    tables = _header_tables()
    assert tables["kLetterRanges"] == expected["L"]
    assert tables["kNumberRanges"] == expected["N"]
