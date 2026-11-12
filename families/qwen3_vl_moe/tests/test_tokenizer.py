# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Unicode pretokenization and BPE against the official tokenizers engine."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TEXTS = [
    "שלום, עולם!",
    "বাংলা ১২৩।",
    "தமிழ் தமிழ்",
    "తెలుగు ౧౨",
    "አማርኛ።",
    "العربية، ١٢٣",
    "हिन्दी १२३।",
    "ไทย ๑๒๓",
    "ＡＢＣ ａｂｃ",
    "﨑塚 中文。",
    "ⱠꞀ letters",
    "a\u0301 e\u0301",
    "Ⅻ ½ ² ₃",
    "'hello 'world",
    "I'M WE'RE THEY'LL",
    "\n    \n",
    " \r\n\t \n   ",
    "\t}",
    "\t\t}",
    "\u00a0}",
    "\u00a0\u00a0}",
    "\tword",
    "\t\tword",
    "  hello  world ",
    "function() {\n\n    return 'hello';\n}",
] + [
    "left" + chr(point) + "right"
    for point in [0x85, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
]

NEWLINE_CASES = [
    pytest.param("\n    \n", ["\n    \n"], id="indented-blank-line"),
    pytest.param(" \r\n\t \n   ", [" \r\n\t \n", "   "], id="trailing-spaces"),
    pytest.param("\n    \n  next", ["\n    \n", " ", " next"], id="following-word"),
    pytest.param("\r\t\r\n\tword", ["\r\t\r\n", "\tword"], id="mixed-crlf"),
    pytest.param("\n\u2003\n\u00a0x", ["\n\u2003\n", "\u00a0x"], id="unicode-indentation"),
    pytest.param("\n    text", ["\n", "   ", " text"], id="single-newline-boundary"),
    pytest.param(" \t\n \r\t\n\n  ", [" \t\n \r\t\n\n", "  "], id="multiple-newline-runs"),
    pytest.param("\n\t\n\t}", ["\n\t\n", "\t", "}"], id="following-symbol"),
]


@pytest.fixture(scope="module")
def tokenizer_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    runtime = Path(os.environ.get("TRTMC_RUNTIME_ROOT", root / "build"))
    binary = runtime / "test_qwen3_vl_moe_tokenizer"
    if binary.is_file():
        return binary
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None or shutil.which("pkg-config") is None:
        pytest.skip("native tokenizer regression requires C++ and ICU development files")
    flags = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "icu-uc"], text=True, capture_output=True
    )
    if flags.returncode:
        pytest.skip("native tokenizer regression requires ICU development files")
    binary = tmp_path_factory.mktemp("qwen3-vl-moe-tokenizer") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-I",
            str(root),
            str(Path(__file__).parent / "cpp/tokenizer_probe.cpp"),
            "-o",
            str(binary),
            *flags.stdout.split(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.fixture(scope="module")
def reference_tokenizer():
    split = pre_tokenizers.Split(Regex(PATTERN), behavior="isolated")
    reference = Tokenizer(models.BPE(unk_token="[UNK]"))
    reference.pre_tokenizer = pre_tokenizers.Sequence(
        [
            split,
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    reference.decoder = decoders.ByteLevel()
    reference.train_from_iterator(
        (TEXTS + [case.values[0] for case in NEWLINE_CASES]) * 20,
        trainers.BpeTrainer(
            vocab_size=2000,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
            show_progress=False,
        ),
    )
    return split, reference


def _probe(tokenizer_probe, reference, texts):
    completed = subprocess.run(
        [str(tokenizer_probe)],
        input=json.dumps({"tokenizer": reference.to_str(), "texts": texts}),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_native_unicode_regex_and_ids_match_reference(tokenizer_probe, reference_tokenizer):
    split, reference = reference_tokenizer
    results = _probe(tokenizer_probe, reference, TEXTS)
    assert len(results) == len(TEXTS)
    for text, actual in zip(TEXTS, results):
        assert actual["pieces"] == [piece for piece, _ in split.pre_tokenize_str(text)], repr(text)
        assert actual["ids"] == reference.encode(text, add_special_tokens=False).ids, repr(text)


@pytest.mark.parametrize("text,pieces", NEWLINE_CASES)
def test_newline_match_ends_at_last_newline(tokenizer_probe, reference_tokenizer, text, pieces):
    split, reference = reference_tokenizer
    assert [piece for piece, _ in split.pre_tokenize_str(text)] == pieces
    (actual,) = _probe(tokenizer_probe, reference, [text])
    assert actual["pieces"] == pieces
    assert actual["ids"] == reference.encode(text, add_special_tokens=False).ids
