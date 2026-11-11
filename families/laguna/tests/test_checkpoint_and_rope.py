# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Laguna config contract, checkpoint mapping and RoPE tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.laguna.checkpoint import BF16, LagunaWeights, expected_tensors  # noqa: E402
from families.laguna.config import RopeSpec, parse_config  # noqa: E402
from families.laguna.rope import rope_tables  # noqa: E402


# The rope / attention / MoE fields of poolside/Laguna-XS.2 config.json
# (revision a397bde04501591d0ac3d74a5d9d0a4ec298ffb8), layer lists shortened.
XS2_ROPE = {
    "full_attention": {
        "rope_theta": 500000.0,
        "rope_type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_slow": 1.0,
        "beta_fast": 64.0,
        "attention_factor": 1.4158883083359672,
        "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    },
}


def tiny_raw_config(**overrides) -> dict:
    raw = {
        "model_type": "laguna",
        "architectures": ["LagunaForCausalLM"],
        "vocab_size": 48,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_attention_heads_per_layer": [2, 4, 4],
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "mlp_layer_types": ["dense", "sparse", "sparse"],
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 6,
        "shared_expert_intermediate_size": 5,
        "moe_routed_scaling_factor": 2.5,
        "sliding_window": 4,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "bos_token_id": 2,
        "eos_token_id": [2, 24],
        "pad_token_id": 9,
        "rope_parameters": XS2_ROPE,
    }
    raw.update(overrides)
    return raw


def test_config_reads_the_xs2_layout() -> None:
    cfg = parse_config(tiny_raw_config())
    assert cfg.heads_per_layer == (2, 4, 4)
    assert cfg.kv_width == 16
    assert cfg.eos_token_ids == (2, 24)
    assert cfg.rope["full_attention"] == RopeSpec(
        "yarn", 500000.0, 4, 64.0, 4096, 64.0, 1.0, 1.4158883083359672, True
    )
    assert cfg.rope["sliding_attention"].rotary_dim == 8
    assert cfg.routed_scaling_factor == 2.5


@pytest.mark.parametrize(
    "override",
    [
        {"gating": "per-element"},
        {"attention_bias": True},
        {"moe_apply_router_weight_on_input": True},
        {"num_attention_heads_per_layer": [3, 4, 4]},
        {"mlp_layer_types": ["dense", "sparse"]},
        {"hidden_act": "gelu"},
        {
            "rope_parameters": {
                **XS2_ROPE,
                "sliding_attention": {"rope_type": "dynamic", "rope_theta": 1.0},
            }
        },
        {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
        {"rope_parameters": {"full_attention": XS2_ROPE["full_attention"]}},
    ],
)
def test_config_rejects_what_the_family_cannot_represent(override: dict) -> None:
    with pytest.raises(ValueError):
        parse_config(tiny_raw_config(**override))


def test_top_level_rope_values_fill_missing_per_layer_values() -> None:
    nested = {
        "full_attention": {"rope_type": "default"},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    cfg = parse_config(
        tiny_raw_config(rope_parameters=nested, rope_theta=50000.0, partial_rotary_factor=0.5)
    )
    assert cfg.rope["full_attention"] == RopeSpec("default", 50000.0, 4)
    # Per-layer values win over the top-level ones.
    assert cfg.rope["sliding_attention"] == RopeSpec("default", 10000.0, 4)


def test_yarn_original_context_defaults_to_max_position_embeddings() -> None:
    full = dict(XS2_ROPE["full_attention"])
    del full["original_max_position_embeddings"]
    cfg = parse_config(tiny_raw_config(rope_parameters={**XS2_ROPE, "full_attention": full}))
    assert cfg.rope["full_attention"].original_max_position_embeddings == 64


def _random_checkpoint(directory: Path, raw: dict, seed: int = 0) -> dict[str, np.ndarray]:
    cfg = parse_config(raw)
    rng = np.random.default_rng(seed)
    tensors = {
        name: rng.standard_normal(shape).astype(ml_dtypes.bfloat16)
        for name, shape in expected_tensors(cfg).items()
    }
    safetensors_numpy.save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return tensors


def test_weights_keep_bf16_bits_and_stack_experts(tmp_path: Path) -> None:
    raw = tiny_raw_config()
    stored = _random_checkpoint(tmp_path, raw)
    weights = LagunaWeights(tmp_path, parse_config(raw), BF16).tensors
    for ours, theirs in (
        ("embed", "model.embed_tokens.weight"),
        ("lm_head", "lm_head.weight"),
        ("layer.1.q_proj", "model.layers.1.self_attn.q_proj.weight"),
        ("layer.1.g_proj", "model.layers.1.self_attn.g_proj.weight"),
        ("layer.0.mlp.down_proj", "model.layers.0.mlp.down_proj.weight"),
        ("layer.2.shared.up_proj", "model.layers.2.mlp.shared_expert.up_proj.weight"),
    ):
        assert weights[ours].dtype == BF16
        assert np.array_equal(weights[ours].view(np.uint16), stored[theirs].view(np.uint16)), ours
    stack = weights["layer.2.experts.down_proj"]
    assert stack.shape == (4, 16, 6)
    for expert in range(4):
        source = stored[f"model.layers.2.mlp.experts.{expert}.down_proj.weight"]
        assert np.array_equal(stack[expert].view(np.uint16), source.view(np.uint16))
    router = weights["layer.1.router"]
    assert router.dtype == np.float32
    assert np.array_equal(router, stored["model.layers.1.mlp.gate.weight"].astype(np.float32))
    bias = weights["layer.1.router_bias"]
    expected = stored["model.layers.1.mlp.experts.e_score_correction_bias"].astype(np.float32)
    assert bias.dtype == np.float32 and np.array_equal(bias, expected)


def test_fp32_storage_widens_exactly(tmp_path: Path) -> None:
    raw = tiny_raw_config()
    stored = _random_checkpoint(tmp_path, raw)
    weights = LagunaWeights(tmp_path, parse_config(raw), np.dtype(np.float32)).tensors
    expected = stored["model.layers.0.self_attn.o_proj.weight"].astype(np.float32)
    assert np.array_equal(weights["layer.0.o_proj"], expected)


def test_checkpoint_must_match_the_config(tmp_path: Path) -> None:
    raw = tiny_raw_config()
    stored = _random_checkpoint(tmp_path, raw)
    stored["model.layers.0.mlp.experts.0.gate_proj.weight"] = stored["model.norm.weight"]
    safetensors_numpy.save_file(stored, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="unexpected"):
        LagunaWeights(tmp_path, parse_config(raw), BF16)


def _yarn_reference(spec: RopeSpec, positions: int) -> tuple[np.ndarray, np.ndarray]:
    """Float64 YaRN (transformers ``_compute_yarn_parameters``, truncate=True)."""
    dim = spec.rotary_dim
    pos_freqs = spec.theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (spec.factor * pos_freqs)

    def correction(rotations: float) -> float:
        return (
            dim
            * math.log(spec.original_max_position_embeddings / (rotations * 2 * math.pi))
            / (2 * math.log(spec.theta))
        )

    low = max(math.floor(correction(spec.beta_fast)), 0)
    high = min(math.ceil(correction(spec.beta_slow)), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0.0, 1.0)
    keep = 1.0 - ramp
    inv_freq = interpolation * (1.0 - keep) + extrapolation * keep
    angles = np.arange(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angles) * spec.attention_factor, np.sin(angles) * spec.attention_factor


def test_partial_yarn_table_matches_the_reference_formula() -> None:
    spec = parse_config(
        tiny_raw_config(
            head_dim=128,
            num_attention_heads_per_layer=[48, 64, 64],
            num_attention_heads=48,
            num_key_value_heads=8,
        )
    ).rope["full_attention"]
    assert spec.rotary_dim == 64
    cos, sin = rope_tables(spec, 4097)
    ref_cos, ref_sin = _yarn_reference(spec, 4097)
    assert cos.shape == (4097, 32) and cos.dtype == np.float32
    # Float32 angles at positions up to 4096 differ from float64 by < 4e-4 rad.
    assert np.max(np.abs(cos - ref_cos)) < 1e-3
    assert np.max(np.abs(sin - ref_sin)) < 1e-3
    assert np.max(np.abs(cos[:64] - ref_cos[:64])) < 1e-5


def test_default_table_is_plain_rope() -> None:
    spec = RopeSpec("default", 10000.0, 8)
    cos, sin = rope_tables(spec, 5)
    inv_freq = 1.0 / 10000.0 ** (np.arange(0, 8, 2) / 8)
    angles = np.arange(5)[:, None] * inv_freq[None, :]
    assert np.allclose(cos, np.cos(angles), atol=1e-6)
    assert np.allclose(sin, np.sin(angles), atol=1e-6)
