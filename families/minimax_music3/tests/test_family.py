# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing and build-contract tests for MiniMax-Music3."""

from __future__ import annotations

import json

from tensorrt_model_connect.model_support import ModelMetadata

from families.minimax_music3 import model, support


PUBLISHED_CONFIG = {
    "architectures": ["MiniMaxMusic3ForConditionalGeneration"],
    "model_type": "minimax_music3",
}

PUBLISHED_INDEX = {
    "_blocks_class_name": "MiniMaxMusic3Blocks",
    "_class_name": "MiniMaxMusic3ModularPipeline",
    "condition_encoder": ["diffusers", "MiniMaxMusic3ConditionEncoder", {}],
    "language_model": ["transformers", "Qwen3ForCausalLM", {}],
    "rvq_depth_decoder": ["diffusers", "MiniMaxMusic3RVQDepthDecoder", {}],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler", {}],
    "tokenizer": ["transformers", "Qwen2Tokenizer", {}],
    "transformer": ["diffusers", "MiniMaxMusic3Transformer1DModel", {}],
    "vocoder": ["diffusers", "MiniMaxMusic3Vocoder", {}],
}


def test_support_claims_the_published_checkpoint() -> None:
    resolved = support.describe(ModelMetadata(config=PUBLISHED_CONFIG, model_index={}, files=()))

    assert resolved is not None
    assert resolved.tasks == ("audio_generation",)
    assert resolved.default_task == "audio_generation"


def test_support_does_not_claim_other_minimax_families() -> None:
    metadata = ModelMetadata(
        config={"model_type": "minimax_h3", "architectures": ["MiniMaxH3"]},
        model_index={},
        files=(),
    )

    assert support.describe(metadata) is None


def test_reads_every_published_modular_component(tmp_path) -> None:
    (tmp_path / model.MODULAR_INDEX_NAME).write_text(json.dumps(PUBLISHED_INDEX), encoding="utf-8")

    components = model.read_pipeline_components(tmp_path)

    assert set(components) == set(model.REQUIRED_COMPONENTS)
    assert components["transformer"] == ("diffusers", "MiniMaxMusic3Transformer1DModel")


def test_bundle_runtime_values_are_family_owned() -> None:
    from families.minimax_music3 import engines

    runtime = engines.bundle_config_overrides()

    assert runtime["sampling_rate"] == 44100
    assert runtime["output_channels"] == 2
    assert runtime["default_inference_steps"] == 30
