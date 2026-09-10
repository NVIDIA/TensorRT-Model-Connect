# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the family-owned dense LFM2 builder."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.lfm2 import checkpoint_mapper
from families.lfm2.config import validate_dense_lfm2_config


FAMILY = Path(__file__).resolve().parent.parent
_MODEL_SOURCE = FAMILY / "model.py"


def _config(raw: dict) -> SimpleNamespace:
    return SimpleNamespace(raw=raw)


@pytest.mark.parametrize(
    (
        "hidden",
        "heads",
        "block_ff_dim",
        "auto_adjust",
        "num_layers",
        "attention_indices",
        "expected_mlp",
    ),
    [
        (1024, 16, 6656, True, 16, [2, 5, 8, 10, 12, 14], 4608),
        (1536, 24, 10240, True, 16, [2, 5, 8, 10, 12, 14], 6912),
        (2048, 32, 12288, True, 16, [2, 5, 8, 10, 12, 14], 8192),
        (2048, 32, 10752, False, 30, [2, 5, 9, 13, 17, 21, 24, 27], 10752),
    ],
)
def test_dense_config_covers_all_four_lfm2_sizes(
    hidden: int,
    heads: int,
    block_ff_dim: int,
    auto_adjust: bool,
    num_layers: int,
    attention_indices: list[int],
    expected_mlp: int,
) -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 65536,
        "hidden_size": hidden,
        "block_dim": hidden,
        "block_ff_dim": block_ff_dim,
        "block_auto_adjust_ff_dim": auto_adjust,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
        "block_use_swiglu": True,
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads,
        "num_heads": heads,
        "num_key_value_heads": 8,
        "full_attn_idxs": attention_indices,
        "conv_L_cache": 3,
        "conv_dim": hidden,
        "conv_dim_out": hidden,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 128000,
        "use_pos_enc": True,
    }
    if not auto_adjust:
        # The released 2.6B schema serializes both aliases at the same
        # pre-adjust value and disables adjustment.
        raw["intermediate_size"] = block_ff_dim

    parsed = validate_dense_lfm2_config(_config(raw))

    assert parsed.hidden_size == hidden
    assert parsed.head_dim == 64
    assert parsed.intermediate_size == expected_mlp
    assert parsed.num_attention_layers == len(attention_indices)
    assert parsed.num_conv_layers == num_layers - len(attention_indices)
    assert parsed.conv_l_cache == 3
    assert parsed.tie_word_embeddings is True
    assert parsed.default_cache_length == 32768
    assert all(parsed.layer_types[index] == "full_attention" for index in attention_indices)


def test_lfm2_350m_architecture_accounts_for_every_checkpoint_parameter() -> None:
    vocab = 65536
    hidden = 1024
    intermediate = 4608
    head_dim = 64
    conv_layers = 10
    attention_layers = 6

    tied_embedding_and_final_norm = vocab * hidden + hidden
    per_layer_norms_and_swiglu = 2 * hidden + 3 * hidden * intermediate
    conv_operator = 3 * hidden * hidden + hidden * 3 + hidden * hidden
    attention_operator = (
        hidden * hidden
        + hidden * (8 * head_dim)
        + hidden * (8 * head_dim)
        + hidden * hidden
        + 2 * head_dim
    )

    total = (
        tied_embedding_and_final_norm
        + 16 * per_layer_norms_and_swiglu
        + conv_layers * conv_operator
        + attention_layers * attention_operator
    )
    assert total == 354_483_968


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"num_experts": 8}, "MoE/VL"),
        ({"vision_config": {"hidden_size": 128}}, "MoE/VL"),
        ({"architectures": ["Lfm2ForConditionalGeneration"]}, "Lfm2ForCausalLM"),
        ({"layer_types": ["conv", "mamba"]}, "layer types"),
        (
            {"layer_types": ["full_attention", "full_attention"]},
            "at least one conv layer",
        ),
    ],
)
def test_config_rejects_out_of_scope_variants(update: dict, message: str) -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 8,
        "block_ff_dim": 12,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": 8,
        "conv_dim_out": 8,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "use_pos_enc": True,
    }
    raw.update(update)

    with pytest.raises(ValueError, match=message):
        validate_dense_lfm2_config(_config(raw))


def test_checkpoint_mapper_routes_conv_attention_and_tied_head(monkeypatch) -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 5,
        "hidden_size": 4,
        "block_ff_dim": 6,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": 4,
        "conv_dim_out": 4,
        "norm_eps": 1e-5,
        "rope_theta": 1000.0,
        "tie_embedding": True,
        "use_pos_enc": True,
    }

    def seq(*shape: int, start: int) -> np.ndarray:
        count = int(np.prod(shape))
        return np.arange(start, start + count, dtype=np.float32).reshape(shape)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": seq(5, 4, start=0),
        "model.embedding_norm.weight": seq(4, start=20),
        "model.layers.0.operator_norm.weight": seq(4, start=30),
        "model.layers.0.ffn_norm.weight": seq(4, start=40),
        "model.layers.0.conv.in_proj.weight": seq(12, 4, start=50),
        "model.layers.0.conv.conv.weight": seq(4, 1, 3, start=100),
        "model.layers.0.conv.out_proj.weight": seq(4, 4, start=120),
        "model.layers.0.feed_forward.w1.weight": seq(6, 4, start=140),
        "model.layers.0.feed_forward.w3.weight": seq(6, 4, start=170),
        "model.layers.0.feed_forward.w2.weight": seq(4, 6, start=200),
        "model.layers.1.operator_norm.weight": seq(4, start=230),
        "model.layers.1.ffn_norm.weight": seq(4, start=240),
        "model.layers.1.self_attn.q_proj.weight": seq(4, 4, start=250),
        "model.layers.1.self_attn.k_proj.weight": seq(2, 4, start=270),
        "model.layers.1.self_attn.v_proj.weight": seq(2, 4, start=280),
        "model.layers.1.self_attn.out_proj.weight": seq(4, 4, start=290),
        "model.layers.1.self_attn.q_layernorm.weight": seq(2, start=310),
        "model.layers.1.self_attn.k_layernorm.weight": seq(2, start=320),
        "model.layers.1.feed_forward.w1.weight": seq(6, 4, start=330),
        "model.layers.1.feed_forward.w3.weight": seq(6, 4, start=360),
        "model.layers.1.feed_forward.w2.weight": seq(4, 6, start=390),
    }
    monkeypatch.setattr(checkpoint_mapper, "_open_safetensors", lambda _path: object())
    monkeypatch.setattr(
        checkpoint_mapper,
        "_has_tensor",
        lambda _readers, name: name in tensors,
    )
    monkeypatch.setattr(
        checkpoint_mapper,
        "_load_tensor",
        lambda _readers, name: tensors[name],
    )

    weights = checkpoint_mapper.load_lfm2_weights(
        "/unused",
        _config(raw),
        precision="bf16",
    )

    assert weights["_layer_types"] == ["conv", "full_attention"]
    assert weights["_num_conv_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["layer.0.conv_in"].shape == (4, 12)
    assert weights["layer.0.conv_weight"].shape == (4, 3)
    assert weights["layer.1.w_q"].shape == (4, 4)
    assert weights["layer.1.w_k"].shape == (4, 2)
    assert weights["layer.1.q_norm"].shape == (2,)
    np.testing.assert_array_equal(
        weights["w_lm_head"],
        tensors["model.embed_tokens.weight"].T,
    )
    assert all(
        value.dtype == np.float32 for value in weights.values() if isinstance(value, np.ndarray)
    )


def test_mapper_rejects_biases_when_conv_bias_is_false(monkeypatch) -> None:
    present = {"model.layers.0.conv.conv.bias"}
    monkeypatch.setattr(
        checkpoint_mapper,
        "_has_tensor",
        lambda _readers, name: name in present,
    )

    with pytest.raises(ValueError, match="conv_bias=false"):
        checkpoint_mapper._store_conv_biases(
            checkpoint_mapper.WeightDict(),
            object(),
            "model.layers.0.conv",
            "layer.0",
            4,
            required=False,
        )


def test_mapper_requires_all_biases_when_conv_bias_is_true(monkeypatch) -> None:
    present = {
        "model.layers.0.conv.in_proj.bias",
        "model.layers.0.conv.conv.bias",
    }
    monkeypatch.setattr(
        checkpoint_mapper,
        "_has_tensor",
        lambda _readers, name: name in present,
    )

    with pytest.raises(ValueError, match="requires all conv bias tensors"):
        checkpoint_mapper._store_conv_biases(
            checkpoint_mapper.WeightDict(),
            object(),
            "model.layers.0.conv",
            "layer.0",
            4,
            required=True,
        )


def test_mapper_loads_all_biases_when_conv_bias_is_true(monkeypatch) -> None:
    shapes = {
        "model.layers.0.conv.in_proj.bias": (12,),
        "model.layers.0.conv.conv.bias": (4,),
        "model.layers.0.conv.out_proj.bias": (4,),
    }
    monkeypatch.setattr(
        checkpoint_mapper,
        "_has_tensor",
        lambda _readers, name: name in shapes,
    )
    monkeypatch.setattr(
        checkpoint_mapper,
        "_load_exact",
        lambda _readers, name, shape: np.full(shape, len(name), dtype=np.float32),
    )
    weights = checkpoint_mapper.WeightDict()

    checkpoint_mapper._store_conv_biases(
        weights,
        object(),
        "model.layers.0.conv",
        "layer.0",
        4,
        required=True,
    )

    assert weights["layer.0.conv_in_bias"].shape == (12,)
    assert weights["layer.0.conv_bias"].shape == (4,)
    assert weights["layer.0.conv_out_bias"].shape == (4,)


def test_model_source_uses_family_explicit_native_graph_contract() -> None:
    source = _MODEL_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "add_kv_cache_update" in source
    assert "KVCacheMode.LINEAR" in source
    assert "add_attention_v2" not in source
    assert "attention.key_value_lengths" not in source
    assert "add_active_prefix_causal_masks" in source
    assert "add_explicit_masked_grouped_query_attention" in source
    assert "add_rotary_embedding" in source
    assert "conv_state" in source and "present_conv" in source
    assert "16 << 30" in source

    sibling_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "families." in node.module:
                sibling_imports.append((node.lineno, node.module))
    assert sibling_imports == []


def test_implicit_cache_default_never_exceeds_a_shorter_model_limit() -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 8,
        "block_ff_dim": 12,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": 8,
        "conv_dim_out": 8,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 4096,
        "use_pos_enc": True,
    }

    assert validate_dense_lfm2_config(_config(raw)).default_cache_length == 4096


def test_exact_case_selection_does_not_expand_a_same_named_recipe(monkeypatch) -> None:
    from families.lfm2.tests import test_e2e

    monkeypatch.delenv("TRTMC_E2E", raising=False)
    manifest = {"name": "shared-recipe"}
    options = {"--e2e-testcase": ["shared-recipe"], "--e2e-model": [], "--e2e-models-file": None}
    config = SimpleNamespace(getoption=lambda key, default=None: options.get(key, default))
    test_e2e._require_selected("shared-recipe", manifest, config)
    with pytest.raises(pytest.skip.Exception):
        test_e2e._require_selected("shared-recipe-chat", manifest, config)
    options["--e2e-model"] = ["shared-recipe"]
    with pytest.raises(pytest.skip.Exception):
        test_e2e._require_selected("shared-recipe-chat", manifest, config)
    options["--e2e-testcase"] = []
    test_e2e._require_selected("shared-recipe", manifest, config)
    test_e2e._require_selected("shared-recipe-chat", manifest, config)
    with pytest.raises(pytest.skip.Exception):
        test_e2e._require_selected("different-case", {"name": "different-recipe"}, config)
