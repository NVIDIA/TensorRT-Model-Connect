# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package canonical normalization data for the native Laya tokenizer."""

import json


def tokenizer_data(path):
    import unicodedata2 as unicode
    from tokenizers.normalizers import NFC, NFD

    data = json.loads(path.read_text())
    if data.get("normalizer", {}).get("type") == "NFC":
        decomposition, composition, combining = [], [], []
        nfc, nfd = NFC(), NFD()
        for code in range(0x110000):
            character = chr(code)
            order = unicode.combining(character)
            if order:
                # The released Rust tokenizer can use an older Unicode table
                # than Python. Unassigned combining marks must stay starters.
                probe = "a" + character + "\u0334" if order > 1 else "a\u0301" + character
                if nfd.normalize_str(probe) != probe:
                    combining.append([code, order])
            parts = unicode.decomposition(character)
            if parts and not parts.startswith("<") and nfd.normalize_str(character) != character:
                values = [int(part, 16) for part in parts.split()]
                decomposition.append([code, *values])
                if len(values) == 2 and nfc.normalize_str("".join(map(chr, values))) == character:
                    composition.append([*values, code])
        data["laya_nfc"] = {
            "version": unicode.unidata_version,
            "decomposition": decomposition,
            "composition": composition,
            "combining": combining,
        }
    return json.dumps(data, ensure_ascii=False).encode()
