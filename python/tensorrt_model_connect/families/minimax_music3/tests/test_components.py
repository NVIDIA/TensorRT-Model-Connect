# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 component geometry reader."""

from __future__ import annotations

import importlib
import json

import pytest

components = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.components"
)

ComponentGeometryError = components.ComponentGeometryError

#: The published component configs at revision
#: fbdf52fbaaca799592917417eb05f1899f1255ec, trimmed to the read keys.
PUBLISHED = {
    "transformer": {
        "_class_name": "MiniMaxMusic3Transformer1DModel",
        "attention_head_dim": 64,
        "condition_dim": 2048,
        "ff_inner_dim": 8192,
        "fourier_embedding_dim": 256,
        "in_channels": 128,
        "num_attention_heads": 32,
        "num_layers": 36,
        "rotary_dim": 32,
    },
    "condition_encoder": {
        "_class_name": "MiniMaxMusic3ConditionEncoder",
        "condition_hidden_dim": 4096,
        "input_hop_length": 960,
        "input_sampling_rate": 24000,
        "num_condition_layers": 8,
        "out_dim": 2048,
        "output_hop_length": 512,
        "output_sampling_rate": 44100,
    },
    "rvq_depth_decoder": {
        "_class_name": "MiniMaxMusic3RVQDepthDecoder",
        "audio_vocab_size": 1024,
        "hidden_size": 4096,
        "intermediate_size": 6144,
        "max_position_embeddings": 16,
        "num_attention_heads": 16,
        "num_codebooks": 8,
        "num_layers": 4,
    },
    "vocoder": {
        "_class_name": "MiniMaxMusic3Vocoder",
        "decoder_hidden_dim": 1536,
        "decoder_input_dim": 1024,
        "latent_channels": 128,
        "sampling_rate": 44100,
        "upsampling_ratios": [8, 8, 4, 2],
    },
    "language_model": {
        "architectures": ["Qwen3ForCausalLM"],
        "head_dim": 128,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
    },
}


def _checkpoint(tmp_path, overrides: dict | None = None):
    for name, config in PUBLISHED.items():
        merged = dict(config)
        if overrides and name in overrides:
            merged.update(overrides[name])
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(merged), encoding="utf-8")
    return tmp_path


def test_published_pipeline_geometry(tmp_path) -> None:
    geometry = components.detect_pipeline_geometry(_checkpoint(tmp_path))

    assert geometry.transformer_layers == 36
    assert geometry.transformer_hidden == 2048  # 32 heads x 64
    assert geometry.transformer_in_channels == 128
    assert geometry.condition_dim == 2048
    assert geometry.condition_encoder_layers == 8
    assert geometry.language_model_hidden == 4096
    assert geometry.language_model_layers == 36
    assert geometry.depth_decoder_codebooks == 8
    assert geometry.depth_decoder_vocab == 1024
    assert geometry.vocoder_latent_channels == 128
    assert geometry.vocoder_upsample_factor == 512  # 8 * 8 * 4 * 2
    assert geometry.sampling_rate == 44100


def test_derived_rates(tmp_path) -> None:
    geometry = components.detect_pipeline_geometry(_checkpoint(tmp_path))

    assert geometry.latent_frames_per_second == pytest.approx(44100 / 512)
    assert geometry.max_audio_seconds == pytest.approx(360.0)  # 9000 / 25


def test_checkpoint_rate_differs_from_the_documented_one(tmp_path) -> None:
    """The model card says 32 kHz; every component config says 44100.

    The configs win, and this test exists so the discrepancy is visible rather
    than discovered by a failing acceptance check.
    """
    geometry = components.detect_pipeline_geometry(_checkpoint(tmp_path))

    assert geometry.sampling_rate != components.DOCUMENTED_SAMPLING_RATE


def test_missing_component_is_named(tmp_path) -> None:
    root = _checkpoint(tmp_path)
    (root / "vocoder" / "config.json").unlink()

    with pytest.raises(ComponentGeometryError, match="vocoder/config.json"):
        components.detect_pipeline_geometry(root)


def test_condition_encoder_width_must_match_the_transformer(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"condition_encoder": {"out_dim": 1024}})

    with pytest.raises(ComponentGeometryError, match="condition_encoder emits 1024"):
        components.detect_pipeline_geometry(root)


def test_vocoder_latent_channels_must_match_the_transformer(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"vocoder": {"latent_channels": 64}})

    with pytest.raises(ComponentGeometryError, match="vocoder consumes 64"):
        components.detect_pipeline_geometry(root)


def test_sampling_rates_must_agree(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"condition_encoder": {"output_sampling_rate": 32000}})

    with pytest.raises(ComponentGeometryError, match="targets 32000 Hz"):
        components.detect_pipeline_geometry(root)


def test_hop_must_match_the_upsample_factor(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"vocoder": {"upsampling_ratios": [8, 8, 4]}})

    with pytest.raises(ComponentGeometryError, match="disagrees with the vocoder"):
        components.detect_pipeline_geometry(root)


def test_depth_decoder_must_consume_the_language_model_width(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"rvq_depth_decoder": {"hidden_size": 2048}})

    with pytest.raises(ComponentGeometryError, match="rvq_depth_decoder consumes 2048"):
        components.detect_pipeline_geometry(root)


def test_non_integer_field_is_rejected(tmp_path) -> None:
    root = _checkpoint(tmp_path, {"transformer": {"num_attention_heads": "32"}})

    with pytest.raises(ComponentGeometryError, match="num_attention_heads"):
        components.detect_pipeline_geometry(root)
