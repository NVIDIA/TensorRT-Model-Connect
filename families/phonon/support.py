# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Phonon checkpoint identity, independent of optional dependencies."""

from tensorrt_model_connect.model_support import family_support

_identity = family_support(
    model_types=("gradium_phonon_tts",),
    architectures=("GradiumPhononTTSForConditionalGeneration",),
    tasks=("text_to_speech",),
    default_task="text_to_speech",
)


def _validate_fields(actual, expected, context, partial):
    if not isinstance(actual, dict):
        raise ValueError(f"unsupported Phonon {context}: expected an object")
    for name, value in expected.items():
        if partial and name not in actual:
            continue
        if actual.get(name) != value:
            raise ValueError(f"unsupported Phonon {context}: {name}={actual.get(name)!r}")


def _validate_architecture(config, *, partial=False):
    """Validate full build inputs or the fields available during discovery."""
    expected = {
        "model_type": "gradium_phonon_tts",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "latent_dim": 32,
        "audio_prompt_dim": 512,
        "hidden_act": "gelu",
        "attention_bias": False,
        "layer_norm_eps": 1e-5,
        "vocab_size": 4001,
    }
    _validate_fields(config, expected, "architecture", partial)
    flow = config.get("flow_head_config", {})
    expected_flow = {
        "flow_type": "lsd",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "latent_dim": 32,
        "condition_dim": 512,
        "hidden_act": "silu",
        "layer_norm_eps": 1e-6,
        "rms_norm_eps": 1e-5,
    }
    _validate_fields(flow, expected_flow, "flow head", partial)
    mimi = config.get("mimi", {})
    expected_codec = {
        "channels": 1,
        "sample_rate": 24000,
        "frame_rate": 12.5,
        "dimension": 512,
        "quantizer_dimension": 32,
        "quantizer_output_dimension": 512,
        "n_filters": 64,
        "n_residual_layers": 1,
        "ratios": [6, 5, 4],
        "kernel_size": 7,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "compress": 2,
        "transformer_d_model": 512,
        "transformer_num_heads": 8,
        "transformer_num_layers": 2,
        "transformer_context": 250,
        "transformer_max_period": 10000.0,
        "gelu": "erf",
    }
    _validate_fields(mimi, expected_codec, "audio decoder", partial)
    if (not partial or "rope_parameters" in config) and config.get("rope_parameters") != {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }:
        raise ValueError("unsupported Phonon rotary embeddings")
    _validate_fields(
        config.get("flow_lm", {}), {"time_rms_norm": "rms", "gelu": "erf"}, "flow LM", partial
    )


def describe(metadata):
    support = _identity(metadata)
    if support is None:
        return None
    try:
        _validate_architecture(metadata.config, partial=True)
    except ValueError:
        return None
    return support
