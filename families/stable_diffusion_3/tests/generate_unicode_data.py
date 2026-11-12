# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ``runtime/unicode_data.h`` from Python's ``unicodedata``.

The tokenizers need a few Unicode properties: simple lowercase mappings, the
general categories behind ``\\p{L}``, ``\\p{N}`` and the combining marks, and the
canonical decompositions, combining classes and primary compositions that NFC
uses. Run ``python3 families/stable_diffusion_3/tests/generate_unicode_data.py``
and commit the output; ``test_unicode_data.py`` checks that the committed
header matches this generator for the same Unicode version.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

HEADER = Path(__file__).resolve().parents[1] / "runtime" / "unicode_data.h"
LIMIT = 0x110000
# Hangul syllables decompose and compose algorithmically.
HANGUL = range(0xAC00, 0xAC00 + 11172)
# The generated header's SPDX lines, kept out of comment syntax here so that
# this script carries exactly one SPDX header of its own.
_COPYRIGHT = ("SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
              "All rights reserved.")
_LICENSE = "SPDX-License-Identifier: Apache-2.0"


def _ranges(predicate) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for cp in range(LIMIT):
        if not predicate(cp):
            continue
        if result and result[-1][1] == cp - 1:
            result[-1] = (result[-1][0], cp)
        else:
            result.append((cp, cp))
    return result


def _category(prefix: str):
    return lambda cp: unicodedata.category(chr(cp)).startswith(prefix)


# Format characters that are not Grapheme_Cluster_Break=Control (UAX #29): ZWNJ
# and ZWJ, the Prepended_Concatenation_Mark characters and the tag characters,
# which unicodedata does not expose.
_NOT_CONTROL = {0x200C, 0x200D, 0x06DD, 0x070F, 0x0890, 0x0891, 0x08E2, 0x110BD, 0x110CD,
                *range(0x0600, 0x0606), *range(0xE0020, 0xE0080)}


# Grapheme_Cluster_Break=Prepend (UAX #29; Unicode 15.0 values), which
# unicodedata does not expose either.
_PREPEND = {*range(0x0600, 0x0606), 0x06DD, 0x070F, 0x0890, 0x0891, 0x08E2, 0x0D4E, 0x110BD,
            0x110CD, 0x111C2, 0x111C3, 0x1193F, 0x11941, 0x11A3A, *range(0x11A84, 0x11A8A),
            0x11D46, 0x11F02}


# What UAX #29 joins to the cluster before it beyond general category M
# (Unicode 15.0 values): ZWNJ, ZWJ, the SpacingMark letters U+0E33 / U+0EB3, the
# halfwidth sound marks and tag characters (Grapheme_Extend), the emoji
# modifiers; and the spacing marks that are not SpacingMark, which start one.
_EXTEND_EXTRA = {0x200C, 0x200D, 0x0E33, 0x0EB3, 0xFF9E, 0xFF9F, *range(0x1F3FB, 0x1F400),
                 *range(0xE0020, 0xE0080)}
_SPACING_NOT_EXTENDING = {0x102B, 0x102C, 0x1038, *range(0x1062, 0x1065), *range(0x1067, 0x106E),
                          0x1083, *range(0x1087, 0x108D), 0x108F, *range(0x109A, 0x109D), 0x1A61,
                          0x1A63, 0x1A64, 0xAA7B, 0xAA7D, 0x11720, 0x11721}


def _grapheme_extend(cp: int) -> bool:
    """Grapheme_Cluster_Break Extend, ZWJ or SpacingMark."""
    if cp in _EXTEND_EXTRA:
        return True
    return unicodedata.category(chr(cp)).startswith("M") and cp not in _SPACING_NOT_EXTENDING


def _grapheme_control(cp: int) -> bool:
    """Grapheme_Cluster_Break Control, CR or LF: Cc, Zl, Zp and Cf less the exceptions."""
    category = unicodedata.category(chr(cp))
    return category in ("Cc", "Zl", "Zp") or (category == "Cf" and cp not in _NOT_CONTROL)


def lowercase_ranges() -> list[tuple[int, int, int, int]]:
    """(lo, hi, delta, stride): cp in [lo, hi] with (cp - lo) % stride == 0 maps to cp + delta."""
    ranges: list[list[int]] = []
    for cp in range(LIMIT):
        lower = chr(cp).lower()
        if len(lower) != 1 or lower == chr(cp):
            continue
        delta = ord(lower) - cp
        if ranges:
            lo, hi, last_delta, stride = ranges[-1]
            step = cp - hi
            if last_delta == delta and step in (1, 2) and (stride in (0, step)):
                ranges[-1] = [lo, cp, delta, step]
                continue
        ranges.append([cp, cp, delta, 0])
    return [(lo, hi, delta, stride or 1) for lo, hi, delta, stride in ranges]


def combining_classes() -> list[tuple[int, int, int]]:
    result: list[list[int]] = []
    for cp in range(LIMIT):
        value = unicodedata.combining(chr(cp))
        if not value:
            continue
        if result and result[-1][1] == cp - 1 and result[-1][2] == value:
            result[-1][1] = cp
        else:
            result.append([cp, cp, value])
    return [tuple(item) for item in result]


def _canonical(cp: int) -> list[int] | None:
    text = unicodedata.decomposition(chr(cp))
    if not text or text.startswith("<"):
        return None
    return [int(part, 16) for part in text.split()]


def decompositions() -> list[tuple[int, int, int]]:
    """Single-step canonical decompositions (cp, first, second or 0)."""
    result = []
    for cp in range(LIMIT):
        if cp in HANGUL:
            continue
        parts = _canonical(cp)
        if parts is None:
            continue
        assert len(parts) in (1, 2), hex(cp)
        result.append((cp, parts[0], parts[1] if len(parts) == 2 else 0))
    return result


def compositions() -> list[tuple[int, int, int]]:
    """Primary composites: (first, second) -> cp, sorted for binary search."""
    result = []
    for cp in range(LIMIT):
        if cp in HANGUL:
            continue
        parts = _canonical(cp)
        if parts is None or len(parts) != 2:
            continue
        if unicodedata.normalize("NFC", unicodedata.normalize("NFD", chr(cp))) != chr(cp):
            continue  # a composition exclusion
        result.append((parts[0], parts[1], cp))
    return sorted(result)


def render() -> str:
    # Text below U+0300 has no combining marks and nothing NFC would change.
    assert all(unicodedata.normalize("NFC", chr(cp)) == chr(cp) for cp in range(0x300))
    assert all(second >= 0x300 for _, second, _ in compositions())
    multi = [cp for cp in range(LIMIT) if len(chr(cp).lower()) > 1]
    assert multi == [0x130], multi

    def rows(items):
        out = []
        for item in items:
            out.append("    {" + ", ".join(f"0x{v:X}" if v >= 0 else str(v) for v in item) + "},")
        return "\n".join(out)

    def signed_rows(items):
        return "\n".join(f"    {{0x{lo:X}, 0x{hi:X}, {delta}, {stride}}},"
                         for lo, hi, delta, stride in items)

    return f"""/*
 * {_COPYRIGHT}
 * {_LICENSE}
 */

// clang-format off
// Generated by families/stable_diffusion_3/tests/generate_unicode_data.py from
// Python's unicodedata (Unicode {unicodedata.unidata_version}). Do not edit by hand.
// Unicode Character Database data, redistributed under the Unicode License v3
// (families/stable_diffusion_3/UNICODE_LICENSE.txt).

#pragma once

#include <cstdint>

namespace trtmc::stable_diffusion_3::unicode_data {{

constexpr const char* kUnicodeVersion = "{unicodedata.unidata_version}";

struct Range {{
    char32_t lo, hi;
}};

// Simple lowercase: cp in [lo, hi] with (cp - lo) % stride == 0 maps to cp + delta.
// U+0130 is the only code point with a multi-code-point lowercase (i + U+0307).
struct CaseRange {{
    char32_t lo, hi;
    std::int32_t delta;
    std::int32_t stride;
}};

constexpr CaseRange kLowercase[] = {{
{signed_rows(lowercase_ranges())}
}};

// General category L* (\\p{{L}}).
constexpr Range kLetters[] = {{
{rows(_ranges(_category("L")))}
}};

// General category N* (\\p{{N}}).
constexpr Range kNumbers[] = {{
{rows(_ranges(_category("N")))}
}};

// General category M* (combining marks).
constexpr Range kMarks[] = {{
{rows(_ranges(_category("M")))}
}};

// Grapheme_Cluster_Break Control, CR and LF: Cc, Zl, Zp and Cf except ZWNJ,
// ZWJ, the prepended concatenation marks and the tag characters.
constexpr Range kGraphemeControls[] = {{
{rows(_ranges(_grapheme_control))}
}};

// Grapheme_Cluster_Break Prepend.
constexpr Range kGraphemePrepend[] = {{
{rows(_ranges(lambda cp: cp in _PREPEND))}
}};

// Grapheme_Cluster_Break Extend, ZWJ and SpacingMark: what joins the cluster before it.
constexpr Range kGraphemeExtend[] = {{
{rows(_ranges(_grapheme_extend))}
}};

// Canonical combining class of every code point whose class is not 0.
struct ClassRange {{
    char32_t lo, hi;
    std::uint8_t ccc;
}};

constexpr ClassRange kCombiningClasses[] = {{
{rows(combining_classes())}
}};

// Single-step canonical decompositions (Hangul is algorithmic); second 0 = singleton.
struct Decomposition {{
    char32_t cp, first, second;
}};

constexpr Decomposition kDecompositions[] = {{
{rows(decompositions())}
}};

// Primary composites (first, second) -> composite, sorted by (first, second).
struct Composition {{
    char32_t first, second, composite;
}};

constexpr Composition kCompositions[] = {{
{rows(compositions())}
}};

}} // namespace trtmc::stable_diffusion_3::unicode_data
"""


def main() -> None:
    text = render()
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        sys.exit(0 if HEADER.read_text(encoding="utf-8") == text else 1)
    HEADER.write_text(text, encoding="utf-8")
    print(f"wrote {HEADER} (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    main()
