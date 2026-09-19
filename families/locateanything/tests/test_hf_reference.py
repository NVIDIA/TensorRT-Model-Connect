# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

from families.locateanything import model
from families.locateanything.tests import hf_reference, vision_oracle


def test_manual_prompt_has_exact_locateanything_image_context() -> None:
    prompt = hf_reference.manual_chat_prompt("Point to: white vehicle.")
    assert prompt.count("<IMG_CONTEXT>") == 256
    assert prompt.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    assert "</img>Point to: white vehicle.<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_rope_theta_uses_the_checkpoint_text_config() -> None:
    assert hf_reference._rope_theta({"text_config": {"rope_theta": 1_000_000}}) == 1_000_000
    assert hf_reference._rope_theta({"text_config": {"rope_parameters": {"rope_theta": 42}}}) == 42


def test_image_preprocess_defaults_match_locateanything() -> None:
    parameters = inspect.signature(vision_oracle.preprocess_image_inputs_for_trt).parameters
    assert parameters["fixed_image_size"].default == 448
    assert parameters["patch_size"].default == 14
    assert parameters["image_mean"].default == (0.5, 0.5, 0.5)
    assert parameters["image_std"].default == (0.5, 0.5, 0.5)
    assert parameters["interpolation"].default == "bicubic"


def test_hf_reference_is_local_manual_and_slow() -> None:
    source = "\n".join(
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (hf_reference, vision_oracle)
    )
    assert "AutoProcessor" not in source
    assert "snapshot_download" not in source
    assert "hf_hub_download" not in source
    assert "DynamicCache" not in source
    assert "all_tied_weights_keys" not in source
    assert "get_expanded_tied_weights_keys" not in source
    assert 'generation_mode="slow"' in source
    assert "preprocess_image_inputs_for_trt(" in source
    assert "local_files_only=True" in source


def test_builder_uses_tokenizer_json_for_native_framing(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class Encoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    class LocalTokenizer:
        def encode(self, _text: str, add_special_tokens: bool = True) -> Encoding:
            return Encoding([1, 7, 2] if add_special_tokens else [7])

    class Tokenizer:
        @staticmethod
        def from_file(path: str) -> LocalTokenizer:
            calls.append(path)
            return LocalTokenizer()

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))

    assert model._tokenizer_runtime_contract(tmp_path) == {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": [1],
        "tokenizer_suffix_ids": [2],
    }
    assert calls == [str(tmp_path / "tokenizer.json")]
