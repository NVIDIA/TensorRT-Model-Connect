# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native tokenizer against the reference library, id for id.

``test_openai_gpt_text_runtime --encode`` (built with the family's CTests into
``TRTMC_NATIVE_BUILD_DIR``) encodes and decodes a corpus with the published
``tokenizer.json`` and with the synthetic one of the C++ unit test; ``tokenizers``
does the same, and every id and every decoded text must be equal. The corpus holds
fixed edge cases and seeded random text over the scripts and character classes the
normaliser treats differently. The same binary must refuse every tokenizer that
the builder's contract refuses.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
from pathlib import Path

import pytest

from families.openai_gpt.tests.synthetic import TOKENIZER, merge_variants, mutations

_TEST_DIR = Path(__file__).resolve().parent
_BINARY = "test_openai_gpt_text_runtime"
_CASES = [
    "The old harbor was quiet at dawn, and a single fishing boat drifted past the breakwater.",
    "Héllo naïve café — CRÈME BRÛLÉE",
    "Don't stop!! (Really?) \"quoted\" [brackets] {braces} <angles> a/b\\c",
    "tab\there\nnewline\r\nreturn nbsp line　ideographic",
    "日本語のテキスト 中文文本 한국어 텍스트",
    "İstanbul ß ẞ Σίσυφος ǅemal ﬁne ＦＵＬＬ",
    "x́̂y äọ é",
    "a😀b 👍🏽 ❤️ \U0001d165\U0001d166",
    "<unk> hello<unk>world <unk><unk> <UNK>",
    "\u0000\u0007\u001b\u007f\u0085­​‍﻿� end",
    "3.14159 $100 50% #1 @user e-mail",
    "",
    "   ",
    "supercalifragilisticexpialidocious" * 8,
    "ﷺ ﷻ ㎒ Ⅻ ½ ²³",
]
# Code point pools: ASCII, Latin with accents, combining marks, Greek, Cyrillic, Arabic,
# Devanagari, Thai, CJK, Hangul, compatibility forms, symbols and emoji, control and
# format characters, and the many kinds of whitespace.
_POOLS = [
    (0x20, 0x7E), (0xA0, 0x17F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
    (0x600, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1D00, 0x1DFF), (0x2000, 0x206F),
    (0x2100, 0x214F), (0x3000, 0x303F), (0x3040, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00),
    (0xF900, 0xFAFF), (0xFB00, 0xFB4F), (0xFE00, 0xFE6F), (0xFF00, 0xFFEF), (0x0, 0x1F),
    (0x1D160, 0x1D17F), (0x1F300, 0x1F6FF), (0xE0000, 0xE007F),
]


def _binary() -> Path:
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if not build:
        pytest.skip("the native tokenizer needs TRTMC_NATIVE_BUILD_DIR (a CMake build with tests)")
    binary = Path(build) / _BINARY
    assert binary.is_file(), f"{binary} is missing: build the target {_BINARY}"
    return binary


def _published_tokenizer() -> Path:
    from huggingface_hub import snapshot_download

    manifest = json.loads((_TEST_DIR / "manifests/openai-gpt.json").read_text(encoding="utf-8"))
    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    )
    return path / "tokenizer.json"


def _random_texts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        pieces = []
        for _ in range(rng.randint(1, 12)):
            lo, hi = rng.choice(_POOLS)
            word = "".join(
                chr(cp)
                for cp in (rng.randint(lo, hi) for _ in range(rng.randint(1, 10)))
                if not 0xD800 <= cp <= 0xDFFF
            )
            pieces.append(word if rng.random() < 0.9 else "<unk>")
        texts.append(rng.choice(["", " ", "  ", "\n"]).join(pieces))
    return texts


def _native(binary: Path, tokenizer: Path, requests: list[dict], tmp_path: Path) -> list[dict]:
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(request) + "\n" for request in requests), encoding="utf-8")
    completed = subprocess.run(
        [str(binary), "--encode", str(tokenizer), str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    answers = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(answers) == len(requests)
    return answers


def _compare(binary: Path, tokenizer_path: Path, texts: list[str], tmp_path: Path) -> int:
    from tokenizers import Tokenizer

    reference = Tokenizer.from_file(str(tokenizer_path))
    expected = [reference.encode(text).ids for text in texts]
    rng = random.Random(7)
    size = reference.get_vocab_size()
    id_paths = [ids for ids in expected if ids] + [
        [rng.randrange(size) for _ in range(rng.randint(1, 40))] for _ in range(500)
    ]
    requests = [{"encode": text} for text in texts] + [{"decode": ids} for ids in id_paths]
    answers = _native(binary, tokenizer_path, requests, tmp_path)
    for text, ids, answer in zip(texts, expected, answers):
        assert answer.get("ids") == ids, (text, ids, answer)
    for ids, answer in zip(id_paths, answers[len(texts):]):
        assert answer["text"] == reference.decode(ids, skip_special_tokens=True), ids
    return sum(len(ids) for ids in expected)


def test_published_tokenizer_matches_the_reference(tmp_path: Path) -> None:
    binary = _binary()
    tokenizer = _published_tokenizer()
    texts = _CASES + _random_texts(3000, seed=286)
    assert _compare(binary, tokenizer, texts, tmp_path) > 10000


def test_transformers_agrees_on_ordinary_text() -> None:
    """transformers 5.2.0 rebuilds this tokenizer with NFD, Lowercase and StripAccents
    instead of the file's BertNormalizer (README, "Tokenizer"); on text without control
    characters, CJK ideographs or spacing and enclosing marks both give the same ids."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = _published_tokenizer()
    auto = AutoTokenizer.from_pretrained(tokenizer.parent, local_files_only=True)
    reference = Tokenizer.from_file(str(tokenizer))
    rng = random.Random(5)
    ordinary = [chr(cp) for cp in range(0x20, 0x7F)] + [chr(cp) for cp in range(0xC0, 0x180)]
    texts = _CASES[:3] + [
        "".join(rng.choice(ordinary) for _ in range(rng.randint(1, 80))) for _ in range(500)
    ]
    for text in texts:
        assert auto(text)["input_ids"] == reference.encode(text).ids, text
    # And where the rebuilt normaliser differs: it keeps a control character in the word.
    assert auto("a\u0007b")["input_ids"] != reference.encode("a\u0007b").ids


def test_synthetic_tokenizer_matches_the_reference(tmp_path: Path) -> None:
    binary = _binary()
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(TOKENIZER), encoding="utf-8")
    words = ["hello", "world", "the", "thing", "sing", "aaaa", "a", "Héllo", "中", "x", "<unk>"]
    rng = random.Random(3)
    texts = [
        rng.choice(["", " ", ",", "!", ".", "\t"]).join(rng.choice(words) for _ in range(6))
        for _ in range(500)
    ]
    _compare(binary, path, _CASES + texts, tmp_path)


@pytest.mark.parametrize("name", sorted(merge_variants()))
def test_merge_list_variants_match_the_reference(name: str, tmp_path: Path) -> None:
    from tokenizers import Tokenizer

    variant = merge_variants()[name]
    # Read as BPE: without a model type, a merge list the reference cannot read loads another model.
    assert type(Tokenizer.from_str(json.dumps(variant)).model).__name__ == "BPE"
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(variant), encoding="utf-8")
    texts = ["xb", "hello xb world", "aaa xb the thing", "x b", "bx"]
    _compare(_binary(), path, _CASES + texts, tmp_path)


def test_cpp_unit_test_uses_the_synthetic_tokenizer() -> None:
    source = (_TEST_DIR / "cpp/test_openai_gpt_text_runtime.cpp").read_text(encoding="utf-8")
    literal = re.search(r'R"json\((.*?)\)json"', source, re.S)
    assert literal is not None
    assert json.loads(literal.group(1)) == TOKENIZER


@pytest.mark.parametrize("name", sorted(mutations()))
def test_runtime_refuses_what_the_builder_refuses(name: str, tmp_path: Path) -> None:
    binary = _binary()
    path = tmp_path / "tokenizer.json"
    # ASCII with escapes, so a lone surrogate reaches the parser as the escape it would read.
    path.write_text(json.dumps(mutations()[name]), encoding="utf-8")
    requests = tmp_path / "requests.jsonl"
    requests.write_text("", encoding="utf-8")
    completed = subprocess.run(
        [str(binary), "--encode", str(path), str(requests)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 3, (completed.returncode, completed.stderr)
    assert "tokenizer.json" in completed.stderr
