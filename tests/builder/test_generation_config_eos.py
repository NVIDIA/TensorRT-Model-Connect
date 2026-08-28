# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for effective EOS defaults embedded in engine bundles."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_model_connect.engine_builder import build_bundle


def _build_bundled_config(
    tmp_path,
    model_config,
    generation_config=None,
    bundle_overrides=None,
):
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    if generation_config is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation_config))

    mock_plugin = MagicMock()
    mock_plugin.name = "example_family"
    mock_plugin.runtime_strategy = ""
    mock_plugin.load_weights.return_value = {}
    mock_plugin.build_engine.return_value = b"PLAN"

    del mock_plugin.build_vision_engine
    del mock_plugin.build_extra_engines
    del mock_plugin.embed_input
    del mock_plugin.get_vl_config
    del mock_plugin.get_segmentation_config
    del mock_plugin.get_audio_config
    if bundle_overrides is None:
        del mock_plugin.get_bundle_config_overrides
    else:
        mock_plugin.get_bundle_config_overrides.return_value = bundle_overrides
    del mock_plugin.tokenizer_json_bundle_override

    with patch(
        "tensorrt_model_connect.engine_builder.find_plugin",
        return_value=mock_plugin,
    ):
        with patch(
            "tensorrt_model_connect.engine_builder._get_trt_version",
            return_value="10.0",
        ):
            with patch(
                "tensorrt_model_connect.engine_builder._get_gpu_name",
                return_value="",
            ):
                with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                    with patch("tensorrt_model_connect.engine_builder.write_bundle") as write:
                        build_bundle(str(tmp_path), str(tmp_path / "output.bundle"))

    sections = write.call_args[0][2]
    config_section = next(section for section in sections if section.name == "config.json")
    return json.loads(config_section.data)


@pytest.mark.parametrize(
    ("model_eos", "generation_eos"),
    [
        (5, [5, 7]),
        ([5, 7], 3),
    ],
)
def test_generation_config_eos_overrides_model_config_in_bundle(
    tmp_path,
    model_eos,
    generation_eos,
):
    """generation_config.json supplies the effective scalar/list EOS defaults."""
    model_config = {
        "model_type": "example_decoder",
        "architectures": ["ExampleDecoderForCausalLM"],
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "eos_token_id": model_eos,
    }
    bundled_config = _build_bundled_config(
        tmp_path,
        model_config,
        {"eos_token_id": generation_eos},
    )
    assert bundled_config["eos_token_id"] == generation_eos


def test_nested_decoder_fields_are_canonicalized_at_bundle_root(tmp_path):
    """Runtime fields must not depend on recursive JSON key searches."""
    text_config = {
        "vocab_size": 101,
        "hidden_size": 72,
        "intermediate_size": 216,
        "num_hidden_layers": 3,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "head_dim": 12,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
    }
    model_config = {
        "model_type": "example_decoder",
        "architectures": ["ExampleVisionLanguageForConditionalGeneration"],
        "vision_config": {"hidden_size": 1024},
        "text_config": text_config,
    }

    bundled_config = _build_bundled_config(tmp_path, model_config)

    assert bundled_config["text_config"] == text_config
    assert bundled_config["vision_config"]["hidden_size"] == 1024
    for key, value in text_config.items():
        assert bundled_config[key] == value


def test_segformer_stage_fields_are_not_coerced_to_decoder_scalars(tmp_path):
    """SegFormer's hierarchical encoder dimensions remain model-owned."""
    model_config = {
        "model_type": "segformer",
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "num_attention_heads": [1, 2, 5, 8],
    }

    bundled_config = _build_bundled_config(tmp_path, model_config)

    assert bundled_config["depths"] == model_config["depths"]
    assert bundled_config["hidden_sizes"] == model_config["hidden_sizes"]
    assert bundled_config["num_attention_heads"] == model_config["num_attention_heads"]
    canonical_decoder_fields = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
        "max_position_embeddings",
    }
    assert canonical_decoder_fields.isdisjoint(bundled_config)


def test_model_owned_bundle_overrides_win_after_canonicalization(tmp_path):
    model_config = {
        "model_type": "example_decoder",
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }

    bundled_config = _build_bundled_config(
        tmp_path,
        model_config,
        bundle_overrides={"hidden_size": 96, "model_owned_value": "kept"},
    )

    assert bundled_config["hidden_size"] == 96
    assert bundled_config["model_owned_value"] == "kept"
