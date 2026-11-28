# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.nemotron_bolt.support import describe


def _target():
    return {
        "model_type": "nemotron_h_omni",
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "llm_config": {
            "model_type": "nemotron_h_puzzle",
            "architectures": ["NemotronHPuzzleForCausalLM"],
            "hidden_size": 4096,
            "vocab_size": 131072,
            "block_configs": [{}] * 88,
        },
        "vision_config": {"model_type": "radio"},
    }


def test_bolt_identity_and_default_task():
    family, support = resolve_family(ModelMetadata(_target(), {}))
    assert family == "nemotron_bolt"
    assert support.default_task == "vision_language_generation"
    assert set(support.tasks) == {"text_generation", "vision_language_generation"}


def test_reject_other_omni_and_standalone_draft():
    assert (
        describe(ModelMetadata({**_target(), "sound_config": {"model_type": "audio"}}, {})) is None
    )
    assert (
        describe(ModelMetadata({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, {}))
        is None
    )
