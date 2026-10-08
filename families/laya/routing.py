# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package the released Router's language tables for native request routing."""

import unicodedata


def routing_data():
    from laya import lang
    from laya.router import _ALIASES, _TYPED_DECISION_WORKFLOWS

    properties, lowercase = [], []
    start, previous = 0, None
    for code in range(0x110000):
        char = chr(code)
        flags = (
            int(char.isalpha())
            | int(char.isalnum() or char == "_") << 1
            | int(char.isdecimal()) << 2
            | int(char.isupper()) << 3
            | int(bool(unicodedata.combining(char))) << 4
            | int(char.isprintable()) << 5
        )
        if flags != previous:
            if previous:
                properties.append([start, code - 1, previous])
            start, previous = code, flags
        if char.lower() != char:
            lowercase.append([code, [ord(c) for c in char.lower()]])
    if previous:
        properties.append([start, 0x10FFFF, previous])
    return {
        "properties": properties,
        "lowercase": lowercase,
        "scripts": lang._SCRIPT_RANGES,
        "stopwords": {key: sorted(values) for key, values in lang._STOP.items()},
        "shared_words": sorted(lang._SHARED_WORDS),
        "diacritics": sorted(ord(c) for c in lang._NON_EN_DIACRITICS),
        "aliases": _ALIASES,
        "workflows": {key: sorted(values) for key, values in _TYPED_DECISION_WORKFLOWS.items()},
    }
