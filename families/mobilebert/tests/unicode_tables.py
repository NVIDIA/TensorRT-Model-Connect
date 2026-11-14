# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ``runtime/unicode_tables.h`` from the reference tokenizer library.

The native tokenizer normalises and splits text with tables, not with a Unicode
library. The tables are read off the reference implementation itself - the
``tokenizers`` package pinned in ``requirements.txt`` - one code point at a
time, over the whole code space:

* ``BertNormalizer(clean_text=True, handle_chinese_chars=True,
  strip_accents=None, lowercase=True)``, the uncased normaliser of the
  published checkpoint, maps each code point on its own to nothing (control and
  non-spacing marks), to a space (whitespace), to itself padded with spaces
  (CJK ideographs), or to one or more other code points (decomposition,
  lowercase). The normaliser works character by character, so the per-code-point
  outputs concatenate to its output on a string.
* ``BertPreTokenizer`` decides, per code point of the normalised text, whether
  it separates words (whitespace) or is a word of its own (punctuation).

Hangul syllables decompose algorithmically and are checked here rather than
tabulated.

The one thing the normaliser does to a string that it does not do to each code
point alone is canonical reordering: inside a run of combining marks it sorts
those with a nonzero combining class. Marks that the accent stripping removes
cannot change the output that way, so only the marks that survive matter. They
are listed as ``kReorderable``, found with the reference's own NFD (a mark is a
non-starter when NFD moves it across U+0345 or U+0334). A word holding such a
mark becomes one unknown token when no vocabulary token contains it, whatever
the order; the native tokenizer therefore refuses a vocabulary that contains
one, and is exact on every vocabulary it accepts. The same ranges are written to
``reorderable_marks.py`` so that the builder refuses such a vocabulary before it
writes a bundle.

Run ``python -m families.mobilebert.tests.unicode_tables`` from the repository
root to rewrite both files; ``test_unicode_tables.py`` regenerates them in
memory and requires the committed files to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "runtime" / "unicode_tables.h"
MARKS_OUTPUT = Path(__file__).resolve().parents[1] / "reorderable_marks.py"
_SURROGATES = range(0xD800, 0xE000)
_HANGUL_FIRST, _HANGUL_LAST = 0xAC00, 0xD7A3


def _hangul(cp: int) -> str:
    index = cp - _HANGUL_FIRST
    lead, vowel, tail = index // 588, (index % 588) // 28, index % 28
    return chr(0x1100 + lead) + chr(0x1161 + vowel) + (chr(0x11A7 + tail) if tail else "")


def _ranges(points: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for cp in points:
        if ranges and ranges[-1][1] == cp - 1:
            ranges[-1] = (ranges[-1][0], cp)
        else:
            ranges.append((cp, cp))
    return ranges


def _code_points():
    return (cp for cp in range(0x110000) if cp not in _SURROGATES)


def _normalizer():
    from tokenizers.normalizers import BertNormalizer

    return BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None,
                          lowercase=True)


def _normalizer_tables():

    normalizer = _normalizer()
    dropped, spaced, isolated, runs, expanded = [], [], [], [], []
    for cp in _code_points():
        char = chr(cp)
        out = normalizer.normalize_str(char)
        if out == char:
            continue
        if _HANGUL_FIRST <= cp <= _HANGUL_LAST:
            if out != _hangul(cp):
                raise AssertionError(f"Hangul U+{cp:04X} does not decompose algorithmically")
            continue
        if out == "":
            dropped.append(cp)
        elif out == " ":
            spaced.append(cp)
        elif out == f" {char} ":
            isolated.append(cp)
        else:
            padded = len(out) == 3 and out[0] == out[2] == " "
            body = out[1] if padded else out
            if len(body) == 1:
                delta = ord(body) - cp
                if runs and runs[-1][1] == cp - 1 and runs[-1][2:] == (delta, padded):
                    runs[-1] = (runs[-1][0], cp, delta, padded)
                else:
                    runs.append((cp, cp, delta, padded))
            else:
                if len(out) > 4:
                    raise AssertionError(f"U+{cp:04X} expands to {len(out)} code points")
                expanded.append((cp, out))
    return _ranges(dropped), _ranges(spaced), _ranges(isolated), runs, expanded


def _reorderable_marks(normalizer_output) -> list[tuple[int, int]]:
    """Surviving code points that the reference's NFD treats as non-starters."""
    from tokenizers.normalizers import NFD

    nfd = NFD()
    marks = []
    for cp in _code_points():
        char = chr(cp)
        if normalizer_output(char) != char:
            continue
        # A non-starter moves before U+0345 (class 240) or after U+0334 (class 1).
        if (nfd.normalize_str(f"x\u0345{char}") == f"x{char}\u0345"
                or nfd.normalize_str(f"x{char}\u0334") == f"x\u0334{char}"):
            marks.append(cp)
    return _ranges(marks)


def _pre_tokenizer_tables():
    from tokenizers.pre_tokenizers import BertPreTokenizer

    pre_tokenizer = BertPreTokenizer()
    whitespace, punctuation = [], []
    for cp in _code_points():
        char = chr(cp)
        words = [word for word, _ in pre_tokenizer.pre_tokenize_str(f"a{char}b")]
        if words == ["a", "b"]:
            whitespace.append(cp)
        elif words == ["a", char, "b"]:
            punctuation.append(cp)
        elif words != [f"a{char}b"]:
            raise AssertionError(f"U+{cp:04X} splits as {words}")
    return _ranges(whitespace), _ranges(punctuation)


def _range_rows(name: str, ranges) -> list[str]:
    rows = [f"inline constexpr CodeRange {name}[] = {{"]
    for start in range(0, len(ranges), 4):
        rows.append("    " + " ".join(f"{{0x{lo:04X}, 0x{hi:04X}}},"
                                      for lo, hi in ranges[start:start + 4]))
    return rows + ["};"]


def render_marks(reorderable) -> str:
    """The Python copy of ``kReorderable`` that the builder checks vocabularies with."""
    import tokenizers

    rows = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights "
        "reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        '"""``kReorderable`` of ``runtime/unicode_tables.h`` as inclusive code point ranges.',
        "",
        "Generated by ``families/mobilebert/tests/unicode_tables.py`` from tokenizers "
        f"{tokenizers.__version__}.",
        'Do not edit by hand."""',
        "",
        "RANGES = (",
    ]
    for start in range(0, len(reorderable), 4):
        rows.append("    " + " ".join(f"(0x{lo:04X}, 0x{hi:04X}),"
                                      for lo, hi in reorderable[start:start + 4]))
    return "\n".join(rows + [")", ""])


def render() -> tuple[str, str]:
    """The header and the Python ranges, from the installed reference library."""
    import tokenizers

    dropped, spaced, isolated, runs, expanded = _normalizer_tables()
    reorderable = _reorderable_marks(_normalizer().normalize_str)
    whitespace, punctuation = _pre_tokenizer_tables()
    lines = [
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights "
        "reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " */",
        "",
        "#pragma once",
        "",
        "// Generated by families/mobilebert/tests/unicode_tables.py from tokenizers "
        f"{tokenizers.__version__}:",
        "// BertNormalizer(clean_text, handle_chinese_chars, strip_accents=None, lowercase) and",
        "// BertPreTokenizer, read one code point at a time. Do not edit by hand. The Unicode",
        "// notice is in families/mobilebert/UNICODE_LICENSE.txt.",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace trtmc::mobilebert_unicode {",
        "",
        "struct CodeRange {",
        "    char32_t lo, hi;",
        "};",
        "struct CodeShift {",
        "    char32_t lo, hi;",
        "    std::int32_t delta;",
        "    bool isolated;",
        "};",
        "struct CodeExpansion {",
        "    char32_t code;",
        "    std::size_t count;",
        "    char32_t out[4];",
        "};",
        "",
        "// clang-format off",
        "",
        "// Code points the normaliser removes.",
        *_range_rows("kDropped", dropped),
        "",
        "// Code points the normaliser turns into one space.",
        *_range_rows("kSpaced", spaced),
        "",
        "// Code points the normaliser keeps and surrounds with spaces.",
        *_range_rows("kIsolated", isolated),
        "",
        "// Runs mapped to code point + delta, surrounded by spaces when isolated.",
        "inline constexpr CodeShift kShifted[] = {",
    ]
    for lo, hi, delta, padded in runs:
        lines.append(f"    {{0x{lo:04X}, 0x{hi:04X}, {delta}, {'true' if padded else 'false'}}},")
    lines += ["};", "", "// Code points that normalise to several code points.",
              "inline constexpr CodeExpansion kExpanded[] = {"]
    for cp, out in expanded:
        body = ", ".join(f"0x{ord(char):04X}" for char in out)
        lines.append(f"    {{0x{cp:04X}, {len(out)}, {{{body}}}}},")
    lines += ["};", "", "// Surviving non-starters, which canonical reordering may move.",
              *_range_rows("kReorderable", reorderable), "",
              "// Code points of normalised text that separate words.",
              *_range_rows("kWhitespace", whitespace), "",
              "// Code points of normalised text that are a word of their own.",
              *_range_rows("kPunctuation", punctuation), "",
              "// clang-format on", "",
              "} // namespace trtmc::mobilebert_unicode", ""]
    return "\n".join(lines), render_marks(reorderable)


def main() -> int:
    header, marks = render()
    OUTPUT.write_text(header, encoding="utf-8")
    MARKS_OUTPUT.write_text(marks, encoding="utf-8")
    print(f"wrote {OUTPUT} and {MARKS_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
