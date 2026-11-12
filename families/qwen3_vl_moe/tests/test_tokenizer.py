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


def test_native_unicode_regex_and_ids_match_reference(tokenizer_probe):
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
        TEXTS * 20,
        trainers.BpeTrainer(
            vocab_size=2000,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
            show_progress=False,
        ),
    )
    completed = subprocess.run(
        [str(tokenizer_probe)],
        input=json.dumps({"tokenizer": reference.to_str(), "texts": TEXTS}),
        text=True,
        capture_output=True,
        check=True,
    )
    results = json.loads(completed.stdout)
    assert len(results) == len(TEXTS)
    for text, actual in zip(TEXTS, results):
        assert actual["pieces"] == [piece for piece, _ in split.pre_tokenize_str(text)], repr(text)
        assert actual["ids"] == reference.encode(text, add_special_tokens=False).ids, repr(text)
