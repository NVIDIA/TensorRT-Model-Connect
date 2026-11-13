# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    "prompts",
    [
        [
            "IT'S",
            "WE'RE THEY'VE I'LL I'D I'M DON'T",
            "rock'n'roll",
            "'hello 'n 'roll",
            "'ſome",
            "English punctuation!?\r\nNext",
            "",
            "123456789",
        ],
        [
            "שלום עולם",
            "বাংলা ভাষা ১২৩",
            "தமிழ் மொழி ௧௨௩",
            "తెలుగు భాష ౧౨౩",
            "አማርኛ ፩፪፫",
            "한글",
            "العربية، ١٢٣",
            "हिन्दी। १२३",
            "ภาษาไทย ๑๒๓",
            "Café Cafe\u0301",
            "漢字 カタカナ Ⅻ ²",
            "🎹music 🐱!",
        ],
        [
            f"left{chr(cp)}{tail}"
            for cp in [
                *range(0x09, 0x0E),
                0x20,
                0x85,
                0xA0,
                0x1680,
                *range(0x2000, 0x200B),
                0x2028,
                0x2029,
                0x202F,
                0x205F,
                0x3000,
            ]
            for tail in ("right", "!", "'r", "াব")
        ]
        + [
            "word \t\r\n next",
            "word\u2003\u2003next",
            "trailing\u202f\u202f",
            "word\n \nnext",
            "word\n\t\r\n next",
        ],
    ],
    ids=["apostrophes", "multilingual", "unicode-whitespace"],
)
def test_native_tokenizer_matches_hugging_face(prompts):
    from tokenizers import Tokenizer

    directory = os.environ.get(
        "TRTMC_MOLMO2_TOKENIZER_DIR", os.environ.get("TRTMC_MOLMO2_MODEL_DIR")
    )
    if not directory:
        pytest.skip(
            "native tokenizer parity requires the pinned tokenizer assets and native target"
        )
    tokenizer_json = Path(directory) / "tokenizer.json"
    reference = Tokenizer.from_file(str(tokenizer_json))
    binary = Path(os.environ["TRTMC_RUNTIME_ROOT"]) / "families/molmo2/test_molmo2_tokenizer"
    result = subprocess.run(
        [str(binary), str(tokenizer_json), *prompts], check=True, capture_output=True, text=True
    )
    actual = json.loads(result.stdout)
    expected = [reference.encode(prompt, add_special_tokens=False).ids for prompt in prompts]
    assert len(actual) == len(expected)
    for prompt, native, official in zip(prompts, actual, expected, strict=True):
        assert native == official, repr(prompt)
