# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_xglm_build_and_reference_share_tokenizer_semantics() -> None:
    profiles = default_execution_profiles(
        family="xglm",
        runtime_strategy="xglm_decoder_kv_cache",
        reference_backend="hf_transformers",
    )

    assert profiles == {
        "build": "xglm_tokenizer",
        "runtime": "base",
        "reference": "xglm_tokenizer",
    }


def test_xglm_tokenizer_profile_is_fully_pinned() -> None:
    family_dir = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/xglm"
    )
    requirements = (
        family_dir / "python_profile_requirements/xglm_tokenizer.lock.txt"
    ).read_text(encoding="utf-8")

    assert "transformers==5.2.0" in requirements
    assert "huggingface-hub==1.22.0" in requirements
    assert "tokenizers==0.22.2" in requirements
    assert "sentencepiece==0.2.2" in requirements
