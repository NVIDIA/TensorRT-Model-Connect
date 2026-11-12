# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed Unicode tables are what the pinned reference library produces.

Regenerates ``runtime/unicode_tables.h`` and ``reorderable_marks.py`` in memory
from the installed ``tokenizers`` package and requires them to equal the
committed files byte for byte. The tables are only correct for the reference
version they were read from, so another version is a failure here, not a skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from families.mobilebert.tests import unicode_tables

_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _pinned_tokenizers() -> str:
    for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("tokenizers=="):
            return line.split("==", 1)[1].strip()
    raise AssertionError("requirements.txt does not pin tokenizers")


def test_tables_match_the_pinned_reference() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    assert tokenizers.__version__ == _pinned_tokenizers(), (
        f"the tables are read from tokenizers {_pinned_tokenizers()}; "
        f"{tokenizers.__version__} is installed (pip install -r {_REQUIREMENTS})")
    header, marks = unicode_tables.render()
    stale = "is stale: run python -m families.mobilebert.tests.unicode_tables"
    assert header == unicode_tables.OUTPUT.read_text(encoding="utf-8"), f"the header {stale}"
    assert marks == unicode_tables.MARKS_OUTPUT.read_text(encoding="utf-8"), (
        f"reorderable_marks.py {stale}")


def test_tables_are_sorted_and_disjoint() -> None:
    """The runtime looks every table up by binary search."""
    import re

    text = unicode_tables.OUTPUT.read_text(encoding="utf-8")
    blocks = re.findall(r"inline constexpr (\w+) (\w+)\[\] = \{(.*?)\n\};", text, re.S)
    assert {name for _, name, _ in blocks} == {
        "kDropped", "kSpaced", "kIsolated", "kShifted", "kExpanded", "kReorderable",
        "kWhitespace", "kPunctuation"}
    for kind, name, body in blocks:
        if kind == "CodeExpansion":
            codes = [int(code, 16)
                     for code in re.findall(r"^ +\{(0x[0-9A-F]+), \d+, \{", body, re.M)]
            assert codes and codes == sorted(set(codes)), name
            continue
        spans = [(int(lo, 16), int(hi, 16))
                 for lo, hi in re.findall(r"\{(0x[0-9A-F]+), (0x[0-9A-F]+)", body)]
        assert spans and all(lo <= hi for lo, hi in spans), name
        assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1)), name
