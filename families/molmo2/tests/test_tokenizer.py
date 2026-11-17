# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess

import pytest

from families.molmo2.tests.inputs import native_tool, tokenizer_file

MANIFEST = json.loads((Path(__file__).parent / "manifests/molmo2-8b.json").read_text())


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

    tokenizer_json = tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"])
    reference = Tokenizer.from_file(str(tokenizer_json))
    binary = native_tool("test_molmo2_tokenizer")
    result = subprocess.run(
        [str(binary), str(tokenizer_json), *prompts], check=True, capture_output=True, text=True
    )
    actual = json.loads(result.stdout)
    expected = [reference.encode(prompt, add_special_tokens=False).ids for prompt in prompts]
    assert len(actual) == len(expected)
    for prompt, native, official in zip(prompts, actual, expected, strict=True):
        assert native == official, repr(prompt)


@pytest.mark.parametrize(
    "sequences",
    [
        ["c3", "e2", "e282", "f0", "f09f", "f09f98"],
        ["c3a9", "e282ac", "f09f9880", "e6bca2", "41c3a942e282ac43f09f9880"],
        ["80", "c0af", "e080af", "eda080", "f08080af", "f4908080", "ff", "e228a1"],
        ["", "41", "41c3", "e28241", "f09f9841", "e282c3a9", "c3a9e282ac80"],
        [f"{byte:02x}" for byte in range(256)],
    ],
    ids=["truncated", "cross-token", "invalid", "mixed", "all-single-bytes"],
)
def test_native_byte_decoding_matches_hugging_face(sequences):
    from tokenizers import Tokenizer

    tokenizer_json = tokenizer_file(MANIFEST["hf_id"], MANIFEST["hf_revision"])
    reference = Tokenizer.from_file(str(tokenizer_json))
    # GPT-2's byte alphabet maps bytes excluded from its visible ranges to
    # codepoints starting at U+0100. Select individual byte tokens so valid
    # multibyte sequences must be reconstructed across token boundaries.
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    alphabet = {byte: chr(byte) for byte in visible}
    alphabet.update(
        (byte, chr(256 + index))
        for index, byte in enumerate(byte for byte in range(256) if byte not in alphabet)
    )
    ids = [
        [reference.token_to_id(alphabet[byte]) for byte in bytes.fromhex(sequence)]
        for sequence in sequences
    ]
    assert all(token is not None for sequence in ids for token in sequence)
    completed = subprocess.run(
        [str(native_tool("test_molmo2_tokenizer")), str(tokenizer_json), "--decode"],
        input=json.dumps(ids),
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout) == [
        reference.decode(sequence, skip_special_tokens=True) for sequence in ids
    ]
