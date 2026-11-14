# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the FLUX.2-klein (Flux2KleinPipeline) build contract."""

import json
from pathlib import Path

import pytest

from families.flux.config import ModelConfig
from families.flux.model import (
    _FluxModel,
    _flux2_text_encoder_spec,
    _reject_unvalidated_klein_options,
)
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family

# text_encoder/config.json of black-forest-labs/FLUX.2-klein-base-4B (Qwen3-4B).
QWEN3_TEXT_ENCODER = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "rms_norm_eps": 1e-06,
}

# transformer/config.json of the same checkpoint.
KLEIN_TRANSFORMER = {
    "_class_name": "Flux2Transformer2DModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 7680,
    "mlp_ratio": 3.0,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}


def test_qwen3_text_encoder_spec_follows_flux2_klein_pipeline() -> None:
    spec = _flux2_text_encoder_spec(QWEN3_TEXT_ENCODER)

    assert spec["kind"] == "qwen3"
    assert spec["extract_layers"] == (9, 18, 27)
    assert len(spec["extract_layers"]) * spec["hidden_size"] == 7680
    assert (spec["num_heads"], spec["num_kv_heads"], spec["head_dim"]) == (32, 8, 128)
    assert spec["eps"] == pytest.approx(1e-6)
    assert spec["rope_theta"] == pytest.approx(1e6)


def test_mistral_text_encoder_spec_keeps_flux2_dev_contract() -> None:
    spec = _flux2_text_encoder_spec(
        {
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {"hidden_size": 5120, "num_hidden_layers": 40, "head_dim": 128},
        }
    )

    assert spec["kind"] == "mistral"
    assert spec["extract_layers"] == (10, 20, 30)
    assert len(spec["extract_layers"]) * spec["hidden_size"] == 15360
    assert spec["eps"] == pytest.approx(1e-5)


def test_klein_runtime_config_selects_qwen_template_and_true_cfg() -> None:
    config = ModelConfig(
        raw={
            "_transformer_config": KLEIN_TRANSFORMER,
            "_text_encoder_config": QWEN3_TEXT_ENCODER,
            "_scheduler_config": {"use_dynamic_shifting": True, "shift": 3.0},
            "_vae_config": {"latent_channels": 32},
        }
    )

    runtime = _FluxModel().get_diffusion_config(config)

    assert runtime["text_encoder_dim"] == 7680
    assert runtime["dit_dim"] == 3072 and runtime["dit_num_heads"] == 24
    assert runtime["guidance_embeds"] == 0
    assert runtime["true_cfg"] == 1
    assert runtime["text_template"] == "qwen3_chat"
    assert runtime["pad_token"] == "<|endoftext|>"
    assert (runtime["num_inference_steps"], runtime["guidance_scale"]) == (50, 4.0)


def test_flux2_dev_runtime_config_is_unchanged_by_klein_keys() -> None:
    config = ModelConfig(
        raw={
            "_transformer_config": {"_class_name": "Flux2Transformer2DModel"},
            "_text_encoder_config": {"text_config": {"hidden_size": 5120}},
        }
    )

    runtime = _FluxModel().get_diffusion_config(config)

    assert runtime["text_encoder_dim"] == 15360
    assert runtime["guidance_embeds"] == 1
    assert (runtime["num_inference_steps"], runtime["guidance_scale"]) == (28, 3.5)
    assert not {"true_cfg", "text_template", "pad_token"} & set(runtime)


@pytest.mark.parametrize(
    "options",
    (
        {"fp8_scales": {"x": {}}},
        {"nvfp4_checkpoint": "nvfp4"},
        {"parallel_enabled": True},
        {"max_batch_size": 2},
        {"is_distilled": True},
    ),
)
def test_klein_rejects_unvalidated_build_options(options: dict) -> None:
    arguments = {
        "fp8_scales": None,
        "nvfp4_checkpoint": None,
        "parallel_enabled": False,
        "max_batch_size": 1,
        "is_distilled": False,
    }
    arguments.update(options)
    with pytest.raises(NotImplementedError, match="FLUX.2-klein"):
        _reject_unvalidated_klein_options(**arguments)


def test_flux2_klein_pipeline_resolves_to_flux(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )

    family, support = resolve_family(load_model_metadata(tmp_path))

    assert family == "flux"
    assert support.default_task == "image_generation"


def test_load_weights_records_model_index_for_the_distilled_check(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline", "is_distilled": True}), encoding="utf-8"
    )
    config = ModelConfig()

    _FluxModel().load_weights(str(tmp_path), config)

    assert config.raw["_model_index"]["is_distilled"] is True
