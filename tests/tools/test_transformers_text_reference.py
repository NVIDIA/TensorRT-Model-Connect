# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_reference_module():
    path = REPO_ROOT / "tools" / "reference" / "transformers_text.py"
    spec = importlib.util.spec_from_file_location(
        "transformers_text_reference_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_internlm_reference_module():
    from tensorrt_model_connect.models.internlm.tests.tools import transformers_text

    return transformers_text


def _arguments() -> argparse.Namespace:
    return argparse.Namespace(
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        do_sample=False,
        apply_chat_template=False,
    )


def test_generation_settings_apply_explicit_hf_overrides() -> None:
    module = _load_reference_module()
    settings = module._generation_settings(
        _arguments(),
        {
            "generation": {"max_new_tokens": 64},
            "task_eval": {
                "hf_use_cache": False,
                "hf_generation_overrides": {
                    "no_repeat_ngram_size": 0,
                    "forced_bos_token_id": None,
                    "forced_eos_token_id": None,
                },
            },
        },
        {},
    )

    assert settings["generation_overrides"] == {
        "use_cache": False,
        "no_repeat_ngram_size": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
    }


def test_generation_settings_reject_non_mapping_hf_overrides() -> None:
    module = _load_reference_module()
    with pytest.raises(
        ValueError, match="task_eval.hf_generation_overrides must be a mapping"
    ):
        module._generation_settings(
            _arguments(),
            {"task_eval": {"hf_generation_overrides": ["not", "a", "mapping"]}},
            {},
        )


def test_generated_token_max_score_ids_preserve_exact_ties() -> None:
    torch = pytest.importorskip("torch")
    module = _load_reference_module()

    candidates = module._generated_token_max_score_ids(
        (
            torch.tensor([[1.0, 3.0, 3.0]]),
            torch.tensor([[4.0, 2.0, 4.0]]),
        )
    )

    assert candidates == [[1, 2], [0, 2]]


def test_input_token_ids_preserve_native_tokenizer_framing() -> None:
    torch = pytest.importorskip("torch")
    module = _load_reference_module()

    assert module._input_token_ids(torch.tensor([[2, 10, 11]])) == [2, 10, 11]


def test_internlm_reference_uses_pinned_independent_fast_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_reference_module()
    owner_module = _load_internlm_reference_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer_path = tmp_path / "official-tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": "{{ messages }}",
        "clean_up_tokenization_spaces": False,
    }
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        owner_module,
        "_resolve_cached_model_ref",
        lambda *_args, **_kwargs: model_dir,
    )
    monkeypatch.setattr(
        owner_module,
        "_resolve_reference_tokenizer_json",
        lambda _model_dir, *, local_files_only: tokenizer_path,
    )
    captured: dict[str, object] = {}
    tokenizer = SimpleNamespace(pad_token_id=3, eos_token="</s>")

    def fast_tokenizer(**kwargs):
        captured.update(kwargs)
        return tokenizer

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("InternLM must not use the broken slow tokenizer")
            )
        ),
        PreTrainedTokenizerFast=fast_tokenizer,
    )
    arguments = argparse.Namespace(
        model="internlm/internlm2-math-1_8b",
        model_revision="",
        family="internlm",
        trust_remote_code=True,
        local_files_only=True,
    )

    assert module._load_tokenizer(arguments, transformers) is tokenizer
    assert captured == {
        "tokenizer_file": str(tokenizer_path),
        "model_input_names": ["input_ids", "attention_mask"],
        **tokenizer_config,
    }


def test_internlm_reference_tokenizer_honors_online_cache_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_internlm_reference_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.model").write_bytes(b"official-source-model")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda path: (
            module._SOURCE_MODEL_SHA256
            if path.name == "tokenizer.model"
            else module._TOKENIZER_SHA256
        ),
    )

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(tokenizer_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    assert (
        module._resolve_reference_tokenizer_json(
            model_dir,
            local_files_only=False,
        )
        == tokenizer_path
    )
    assert captured["local_files_only"] is False
