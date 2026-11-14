# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native coding and newline boundaries must match the independent tokenizer."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from families.qwen3_next.tests.tokenizer_contract import tokenizer_document

TEXTS = ['def add(a, b):\n    return a + b\n', "// UTF-8: café, 中文, 🙂\n\tprint('hello')\n", '<|im_start|>user\nExplain this code.\n<|im_end|>\n<|im_start|>assistant\n', "café = 'decomposed NFC'\n", '# العربية ١٢٣, हिन्दी, Ελληνικά\n', "foo'bar I'Spoon WE'RE I'll I'VE couldn't\n", "\t'quote'\n\t== 42\n", 'x = 1234567890\n\n    # whitespace  \n', 'foo\n    \n    bar', 'a\n  \n  b', 'x\n\t\n\ny', '\t \n  \r\n    end', 'a\n \n b']


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory):
    from tokenizers import Tokenizer, pre_tokenizers, trainers
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_document()))
    tokenizer.train_from_iterator(TEXTS * 3, trainer=trainers.BpeTrainer(
        vocab_size=2048, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|im_start|>", "<|im_end|>"], show_progress=False))
    path = tmp_path_factory.mktemp("qwen3-next-tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def native(path, text):
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not runtime:
        pytest.skip("requires the native Qwen3-Next tokenizer probe")
    probe = Path(runtime) / "qwen3_next_tokenizer_probe"
    assert probe.is_file()
    return subprocess.run([str(probe), str(path), text], text=True,
                          capture_output=True, timeout=30)


def compare(path, text):
    from tokenizers import Tokenizer
    result = native(path, text)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    upstream = Tokenizer.from_file(str(path))
    assert output["ids"] == upstream.encode(text, add_special_tokens=False).ids
    assert output["text"] == upstream.decode(output["ids"], skip_special_tokens=True)


@pytest.mark.parametrize("text", TEXTS)
def test_native_tokenizer(text, tokenizer_path):
    compare(tokenizer_path, text)


def test_pinned_checkpoint_tokenizer():
    location = os.environ.get("TRTMC_QWEN3_NEXT_TOKENIZER")
    checkpoint = os.environ.get("TRTMC_QWEN3_NEXT_CHECKPOINT")
    if not location and not checkpoint:
        pytest.skip("requires pinned checkpoint tokenizer metadata")
    path = Path(location) if location else Path(checkpoint) / "tokenizer.json"
    for text in TEXTS:
        compare(path, text)


def test_empty_bpe_affixes_preserve_the_supported_contract(tokenizer_path, tmp_path):
    document = json.loads(tokenizer_path.read_text())
    document["model"].update(continuing_subword_prefix="", end_of_word_suffix="")
    path = tmp_path / "empty-affixes.json"
    path.write_text(json.dumps(document))
    compare(path, "a\n  \n  b")


@pytest.mark.parametrize("change", ["regex", "behavior", "invert", "pre_tokenizer", "normalizer", "decoder", "post_processor", "continuing_subword_prefix", "end_of_word_suffix"])
def test_unsupported_tokenizer_contract_fails(tokenizer_path, tmp_path, change):
    document = json.loads(tokenizer_path.read_text())
    if change == "regex":
        document["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "unknown"
    elif change == "behavior":
        document["pre_tokenizer"]["pretokenizers"][0]["behavior"] = "Removed"
    elif change == "invert":
        document["pre_tokenizer"]["pretokenizers"][0]["invert"] = True
    elif change in {"continuing_subword_prefix", "end_of_word_suffix"}:
        document["model"][change] = "unsupported"
    else:
        document[change] = {"type": "Metaspace"}
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(document))
    result = native(path, "abc")
    assert result.returncode != 0
    assert "unsupported Qwen3-Next" in result.stderr
