# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_olmo_build_and_reference_share_tokenizer_semantics() -> None:
    profiles = default_execution_profiles(
        family="olmo",
        runtime_strategy="olmo_decoder_kv_cache",
        reference_backend="hf_transformers",
    )

    assert profiles == {
        "build": "olmo_tokenizer",
        "runtime": "base",
        "reference": "olmo_tokenizer",
    }


def test_olmo_tokenizer_profile_is_fully_pinned() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/olmo/"
        "python_profile_requirements/olmo_tokenizer.lock.txt"
    ).read_text(encoding="utf-8")
    pins = [line for line in requirements.splitlines() if line and not line.startswith("#")]

    assert pins == [
        "transformers==5.2.0",
        "tokenizers==0.22.2",
        "huggingface-hub==1.22.0",
    ]
