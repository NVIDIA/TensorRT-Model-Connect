# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free CPU checks of the minicpm_ocr family: identity, default task and the checkpoint contract
the builder enforces (no quantized checkpoints, the tokenizer pipeline the native runtime implements)."""

import json
from pathlib import Path

import pytest

from families.minicpm_ocr.support import describe
from families.minicpm_ocr.checkpoint_contract import SPLIT_PATTERN, require_unquantized, tokenizer_bytes
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

_CONFIG = {
    "model_type": "minicpm_ocr",
    "architectures": ["MiniCPMOcrForConditionalGeneration"],
    "text_config": {"model_type": "qwen3_5_text"},
    "vision_config": {"model_type": "minicpm_ocr_vision"},
}


def test_checkpoint_identity_selects_images_text_to_text() -> None:
    support = describe(ModelMetadata(dict(_CONFIG), {}))
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp16"


def test_resolution_owns_the_checkpoint_alone() -> None:
    family, support = resolve_family(ModelMetadata(dict(_CONFIG), {}))
    assert family == "minicpm_ocr"
    assert support.default_task == "images_text_to_text"


def test_architecture_alone_identifies_the_checkpoint() -> None:
    support = describe(ModelMetadata({"architectures": ["MiniCPMOcrForConditionalGeneration"]}, {}))
    assert support is not None and support.default_task == "images_text_to_text"


def test_related_identities_are_not_claimed() -> None:
    for config in (
        {},
        {"model_type": "minicpmv4_6", "architectures": ["MiniCPMV4_6ForConditionalGeneration"]},
        {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]},
        {"model_type": "qwen3_5_text"},
        {"model_type": "minicpm_ocr_vision"},
    ):
        assert describe(ModelMetadata(config, {}, ("model.pt",))) is None


def _tokenizer(**overrides) -> dict:
    tokenizer = {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT_PATTERN}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "byte_fallback": False, "ignore_merges": False, "vocab": {}, "merges": []},
    }
    tokenizer.update(overrides)
    return tokenizer


def _model_dir(tmp_path: Path, tokenizer: dict) -> Path:
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    return tmp_path


def test_the_checkpoint_tokenizer_pipeline_is_accepted(tmp_path: Path) -> None:
    data = tokenizer_bytes(_model_dir(tmp_path, _tokenizer()))
    assert json.loads(data)["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == SPLIT_PATTERN


@pytest.mark.parametrize("override", [
    {"normalizer": None},
    {"normalizer": {"type": "NFKC"}},
    {"decoder": {"type": "Metaspace"}},
    {"model": {"type": "BPE", "byte_fallback": True, "vocab": {}, "merges": []}},
    {"model": {"type": "BPE", "ignore_merges": True, "vocab": {}, "merges": []}},
    {"model": {"type": "WordPiece", "vocab": {}}},
    {"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True}},
], ids=["no-normalizer", "nfkc", "metaspace-decoder", "byte-fallback", "ignore-merges", "wordpiece",
        "bytelevel-regex"])
def test_other_tokenizer_pipelines_are_refused(tmp_path: Path, override: dict) -> None:
    with pytest.raises(ValueError, match="native tokenizer"):
        tokenizer_bytes(_model_dir(tmp_path, _tokenizer(**override)))


def test_a_different_split_pattern_is_refused(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    grouped_digits = SPLIT_PATTERN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")
    assert grouped_digits != SPLIT_PATTERN
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"] = {"Regex": grouped_digits}
    with pytest.raises(ValueError, match="native tokenizer"):
        tokenizer_bytes(_model_dir(tmp_path, tokenizer))


@pytest.mark.parametrize("config", [
    {**_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}},
    {**_CONFIG, "text_config": {"model_type": "qwen3_5_text", "quantization_config": {"quant_method": "gptq"}}},
], ids=["awq", "gptq-text"])
def test_quantized_derivatives_are_refused(config: dict) -> None:
    with pytest.raises(ValueError, match="quantized checkpoints are not supported"):
        require_unquantized(config)


def test_the_bf16_release_is_not_refused() -> None:
    require_unquantized(_CONFIG)
