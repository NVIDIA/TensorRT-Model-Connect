# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Flint token IDs must match the independent tokenizer implementation."""
import json
import os
from pathlib import Path
import subprocess

import pytest

PRETOKENIZER = {'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': '\\p{N}{1,3}'}, 'behavior': 'Isolated', 'invert': False}, {'type': 'Split', 'pattern': {'Regex': '[一-龥\u3040-ゟ゠-ヿ]+'}, 'behavior': 'Isolated', 'invert': False}, {'type': 'Split', 'pattern': {'Regex': "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, 'behavior': 'Isolated', 'invert': False}, {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True, 'use_regex': False}]}
DECODER = {'type': 'ByteLevel', 'add_prefix_space': True, 'trim_offsets': True, 'use_regex': True}
TEXTS = [
    "The capital of France is", "HelloWorld CamelCase fooBAR", "don't I'M we're They'll",
    "1234567 ١٢٣٤٥٦٧ Ⅷ ²", " שלום עולם!", "বাংলা, தமிழ்।", "العربية، ١٢٣", "हिन्दी।",
    "አማርኛ! 가 한글", "中文字符ひらがなカタカナ", "café e\u0301", "αβΓΔ Σσς",
    "a\r\n\tb\n\n", "<|user|>Hello<|end_of_text|>", "",
] + ["a" + chr(cp) + "b" for cp in [0x85, 0x1680, *range(0x2000, 0x200b), 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]]


@pytest.fixture(scope="module")
def probe():
    path = os.environ.get("TRTMC_FLINT_TOKENIZER_PROBE")
    if not path:
        pytest.skip("requires the family-owned native tokenizer probe")
    assert Path(path).is_file()
    return path


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    document = json.loads(Tokenizer(models.BPE()).to_str())
    document.update(pre_tokenizer=PRETOKENIZER, decoder=DECODER)
    tokenizer = Tokenizer.from_str(json.dumps(document))
    tokenizer.train_from_iterator(TEXTS * 3, trainer=trainers.BpeTrainer(
        vocab_size=2048, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|user|>", "<|end_of_text|>"], show_progress=False))
    path = tmp_path_factory.mktemp("flint-tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def native(probe, path, requests):
    return subprocess.run([probe, str(path)], input=json.dumps(requests),
                          text=True, capture_output=True, timeout=30)


def compare_tokenizers(probe, path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    result = native(probe, path, [{"text": text} for text in TEXTS])
    assert result.returncode == 0, result.stderr
    for text, output in zip(TEXTS, json.loads(result.stdout), strict=True):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert output["ids"] == ids, repr(text)
        assert output["text"] == tokenizer.decode(ids, skip_special_tokens=True), repr(text)


def test_unicode_regex_pipeline_matches_reference(probe, tokenizer_path):
    compare_tokenizers(probe, tokenizer_path)


def test_pinned_checkpoint_tokenizer(probe):
    path = os.environ.get("TRTMC_FLINT_TOKENIZER")
    if not path:
        pytest.skip("requires the pinned checkpoint tokenizer metadata")
    compare_tokenizers(probe, Path(path))


def test_generated_partial_bytes_decode_like_reference(probe, tokenizer_path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sequences = [[tokenizer.token_to_id(token)] for token in ("â", "ÿ", "ð")]
    sequences.append([tokenizer.token_to_id("<|end_of_text|>"), *sequences[0]])
    result = native(probe, tokenizer_path, [{"ids": ids} for ids in sequences])
    assert result.returncode == 0, result.stderr
    for ids, output in zip(sequences, json.loads(result.stdout), strict=True):
        assert output["text"] == tokenizer.decode(ids, skip_special_tokens=True)


@pytest.mark.parametrize("change", ["pre_tokenizer", "decoder", "normalizer", "post_processor", "prefix_space", "byte_regex", "byte_fallback", "dropout"])
def test_unsupported_tokenizer_configuration_rejected(probe, tokenizer_path, tmp_path, change):
    document = json.loads(tokenizer_path.read_text())
    if change == "pre_tokenizer":
        document[change] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    elif change == "decoder":
        document[change] = {"type": "Metaspace"}
    elif change == "normalizer":
        document[change] = {"type": "Lowercase"}
    elif change == "post_processor":
        document[change] = {"type": "TemplateProcessing"}
    elif change == "prefix_space":
        document["pre_tokenizer"]["pretokenizers"][-1]["add_prefix_space"] = True
    elif change == "byte_regex":
        document["pre_tokenizer"]["pretokenizers"][-1]["use_regex"] = True
    elif change == "byte_fallback":
        document["model"][change] = True
    else:
        document["model"][change] = 0.1
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(document))
    result = native(probe, path, [{"text": "Hello"}])
    assert result.returncode != 0
    assert "unsupported Flint" in result.stderr
