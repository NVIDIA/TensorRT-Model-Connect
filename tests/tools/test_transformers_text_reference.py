# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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
