# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/static contracts for the family-owned dense LFM2 Python builder."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.lfm2.config import validate_dense_lfm2_config


checkpoint_mapper = importlib.import_module(
    "tensorrt_model_connect.families.lfm2.checkpoint_mapper"
)
debug_runner = importlib.import_module("tensorrt_model_connect.families.lfm2.debug_runner")


_ROOT = Path(__file__).resolve().parents[2]
_MODEL_SOURCE = _ROOT / "python" / "tensorrt_model_connect" / "families" / "lfm2" / "model.py"
_PLUGIN_SOURCE = _MODEL_SOURCE.with_name("plugin.py")


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


def test_config_accepts_explicit_layer_types_and_legacy_aliases() -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "block_dim": 8,
        "block_ff_dim": 12,
        "block_auto_adjust_ff_dim": False,
        "num_layers": 2,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_l_cache": 3,
        "conv_dim": 8,
        "conv_dim_out": 8,
        "block_norm_eps": 1e-6,
        "theta": 12345.0,
        "tie_embedding": False,
        "use_pos_enc": True,
    }

    parsed = validate_dense_lfm2_config(_config(raw))

    assert parsed.hidden_size == 8
    assert parsed.intermediate_size == 12
    assert parsed.num_hidden_layers == 2
    assert parsed.norm_eps == pytest.approx(1e-6)
    assert parsed.rope_theta == pytest.approx(12345.0)
    assert parsed.tie_word_embeddings is False


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


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"block_dim": 16}, "block_dim must equal hidden_size"),
        (
            {"intermediate_size": 13},
            "intermediate_size must equal the pre-adjust block_ff_dim",
        ),
        ({"num_layers": 3}, "num_layers must equal num_hidden_layers"),
        ({"num_heads": 4}, "num_heads must equal num_attention_heads"),
        ({"block_norm_eps": 1e-6}, "block_norm_eps must equal norm_eps"),
        ({"conv_l_cache": 4}, "conv_l_cache must equal conv_L_cache"),
        ({"theta": 500000.0}, "theta must equal rope_theta"),
        (
            {"tie_word_embeddings": True, "tie_embedding": False},
            "tie_embedding must equal tie_word_embeddings",
        ),
        ({"full_attn_idxs": [0]}, "full_attn_idxs must agree with layer_types"),
    ],
)
def test_config_rejects_contradictory_legacy_aliases(
    update: dict,
    message: str,
) -> None:
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 8,
        "block_dim": 8,
        "block_ff_dim": 12,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": 8,
        "conv_dim_out": 8,
        "norm_eps": 1e-5,
        "block_norm_eps": 1e-5,
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


def test_model_source_uses_shared_explicit_native_graph_contract() -> None:
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
            if "tensorrt_model_connect.families." in node.module:
                sibling_imports.append((node.lineno, node.module))
    assert sibling_imports == []


def test_debug_runner_exposes_the_family_owned_causal_contract() -> None:
    assert callable(debug_runner.load_engine_from_bundle)
    assert callable(debug_runner.load_config_from_bundle)
    assert callable(debug_runner.runner_from_bundle)
    source = Path(debug_runner.__file__).read_text(encoding="utf-8")
    assert 'expected = "lfm2_hybrid_conv_attention"' in source
    assert 'f"present_k_{index}", self._cache_k[index]' in source
    assert 'f"present_conv_{index}", self._present_conv[index]' in source


def test_plugin_uses_the_validated_implicit_cache_default() -> None:
    source = _PLUGIN_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "default_max_cache_length"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert "default_cache_length" in method_source
    assert "32768" not in _MODEL_SOURCE.read_text(encoding="utf-8")


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


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.trt
def test_tiny_real_trt_builder_emits_native_state_contract(precision: str) -> None:
    trt = pytest.importorskip("tensorrt")
    model = importlib.import_module("tensorrt_model_connect.families.lfm2.model")
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 128,
        "block_dim": 128,
        "block_ff_dim": 64,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": 128,
        "conv_dim_out": 128,
        "norm_eps": 1e-5,
        "block_norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 8,
        "use_pos_enc": True,
    }
    cfg = _config(raw)
    parsed = validate_dense_lfm2_config(cfg)
    zeros = np.zeros
    ones = np.ones
    embedding = zeros((32, 128), dtype=np.float32)
    embedding[1] = np.linspace(0.1, 1.0, 128, dtype=np.float32)
    embedding[2] = np.linspace(1.0, 0.1, 128, dtype=np.float32)
    identity = np.eye(128, dtype=np.float32)
    conv_in = np.concatenate((identity, identity, identity), axis=1)
    conv_weight = np.broadcast_to(
        np.array([0.25, 0.5, 1.0], dtype=np.float32),
        (128, 3),
    ).copy()
    final_gamma = np.linspace(0.17, 1.73, 128, dtype=np.float32)
    lm_head = zeros((128, 32), dtype=np.float32)
    lm_head[:32, :] = np.eye(32, dtype=np.float32)
    weights = checkpoint_mapper.WeightDict(
        {
            "embedding": embedding,
            "layer.0.operator_norm": ones((128,), dtype=np.float32),
            "layer.0.ffn_norm": ones((128,), dtype=np.float32),
            "layer.0.conv_in": conv_in,
            "layer.0.conv_weight": conv_weight,
            "layer.0.conv_out": identity,
            "layer.0.w1": zeros((128, 64), dtype=np.float32),
            "layer.0.w3": zeros((128, 64), dtype=np.float32),
            "layer.0.w2": zeros((64, 128), dtype=np.float32),
            "layer.1.operator_norm": ones((128,), dtype=np.float32),
            "layer.1.ffn_norm": ones((128,), dtype=np.float32),
            "layer.1.w_q": zeros((128, 128), dtype=np.float32),
            "layer.1.w_k": zeros((128, 64), dtype=np.float32),
            "layer.1.w_v": zeros((128, 64), dtype=np.float32),
            "layer.1.w_o": zeros((128, 128), dtype=np.float32),
            "layer.1.q_norm": ones((64,), dtype=np.float32),
            "layer.1.k_norm": ones((64,), dtype=np.float32),
            "layer.1.w1": zeros((128, 64), dtype=np.float32),
            "layer.1.w3": zeros((128, 64), dtype=np.float32),
            "layer.1.w2": zeros((64, 128), dtype=np.float32),
            "final_norm": final_gamma,
            "w_lm_head": lm_head,
            "_layer_types": list(parsed.layer_types),
            "_num_attention_layers": parsed.num_attention_layers,
            "_num_conv_layers": parsed.num_conv_layers,
            "_hidden_size": parsed.hidden_size,
            "_intermediate_size": parsed.intermediate_size,
            "_head_dim": parsed.head_dim,
            "_conv_l_cache": parsed.conv_l_cache,
        }
    )

    plan = model.build_lfm2_engine(
        cfg,
        weights,
        8,
        precision=precision,
        debug_layer_outputs=True,
    )
    assert plan
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    assert {
        "token_id",
        "position_id",
        "cache_write_indices",
        "key_value_lengths",
        "cache_k_0",
        "cache_v_0",
        "conv_state_0",
        "logits",
        "present_k_0",
        "present_v_0",
        "present_conv_0",
    } <= names
    assert tuple(engine.get_tensor_shape("cache_k_0")) == (1, 1, 8, 64)
    assert tuple(engine.get_tensor_shape("present_k_0")) == (1, 1, 8, 64)
    assert tuple(engine.get_tensor_shape("conv_state_0")) == (128, 3)
    assert tuple(engine.get_tensor_shape("present_conv_0")) == (128, 3)

    runner = debug_runner.Lfm2TrtRunner(
        engine_plan=plan,
        max_cache_length=8,
        num_conv_layers=1,
        num_attention_layers=1,
    )
    try:
        first = runner.step(1)
        second = runner.step(2)
        assert first["logits"].shape == (1, 32)
        assert second["logits"].shape == (1, 32)

        def rms(value: np.ndarray) -> np.ndarray:
            return value / np.sqrt(np.mean(value * value) + 1e-5)

        norm1 = rms(embedding[1])
        bx1 = norm1 * norm1
        hidden1 = embedding[1] + norm1 * bx1
        expected1 = (final_gamma * rms(hidden1))[:32]
        norm2 = rms(embedding[2])
        bx2 = norm2 * norm2
        conv2 = 0.5 * bx1 + bx2
        hidden2 = embedding[2] + norm2 * conv2
        expected2 = (final_gamma * rms(hidden2))[:32]
        tolerance = 0.04 if precision == "bf16" else 0.01
        np.testing.assert_allclose(first["logits"][0], expected1, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(second["logits"][0], expected2, rtol=tolerance, atol=tolerance)

        torch = pytest.importorskip("torch")
        torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        def pinned_hf_rms_norm(hidden: np.ndarray) -> np.ndarray:
            hidden_work = torch.from_numpy(hidden).to(torch_dtype)
            variance = hidden_work.float().pow(2).mean(-1, keepdim=True)
            normalized = (hidden_work.float() * torch.rsqrt(variance + 1e-5)).to(torch_dtype)
            gamma_work = torch.from_numpy(final_gamma).to(torch_dtype)
            return (gamma_work * normalized).float().cpu().numpy()

        def old_fp32_gamma_order(hidden: np.ndarray) -> np.ndarray:
            hidden_work = torch.from_numpy(hidden).to(torch_dtype)
            variance = hidden_work.float().pow(2).mean(-1, keepdim=True)
            normalized = hidden_work.float() * torch.rsqrt(variance + 1e-5)
            return (
                (normalized * torch.from_numpy(final_gamma).float())
                .to(torch_dtype)
                .float()
                .cpu()
                .numpy()
            )

        for result in (first, second):
            hidden = result["debug_hidden_1"][0]
            expected_hf = pinned_hf_rms_norm(hidden)
            old_order = old_fp32_gamma_order(hidden)
            assert np.any(expected_hf[:32].view(np.uint32) != old_order[:32].view(np.uint32))
            np.testing.assert_array_equal(result["logits"][0], expected_hf[:32])
        assert runner.cache_length == 2
        runner.reset()
        assert runner.cache_length == 0
    finally:
        runner.close()


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.trt
def test_nonzero_native_attention_matches_hf_two_step_oracle(precision: str) -> None:
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    model = importlib.import_module("tensorrt_model_connect.families.lfm2.model")
    hidden = 128
    head_dim = 64
    vocab = 128
    raw = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "block_dim": hidden,
        "block_ff_dim": 64,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_dim": hidden,
        "conv_dim_out": hidden,
        "norm_eps": 1e-5,
        "block_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 4,
        "use_pos_enc": True,
    }
    cfg = _config(raw)
    parsed = validate_dense_lfm2_config(cfg)
    zeros = np.zeros
    ones = np.ones
    identity = np.eye(hidden, dtype=np.float32)
    positions = np.arange(hidden, dtype=np.float32)
    embedding = zeros((vocab, hidden), dtype=np.float32)
    embedding[1] = 0.31 * np.sin(0.071 * positions) + 0.17 * np.cos(0.037 * positions)
    embedding[2] = 0.29 * np.cos(0.053 * positions) - 0.13 * np.sin(0.097 * positions)
    operator_gamma = np.linspace(0.61, 1.39, hidden, dtype=np.float32)
    q_gamma = np.linspace(0.73, 1.27, head_dim, dtype=np.float32)
    k_gamma = np.linspace(1.31, 0.69, head_dim, dtype=np.float32)
    final_gamma = np.linspace(0.83, 1.17, hidden, dtype=np.float32)
    k_projection = zeros((hidden, head_dim), dtype=np.float32)
    k_projection[:head_dim, :] = np.eye(head_dim, dtype=np.float32)
    v_projection = zeros((hidden, head_dim), dtype=np.float32)
    v_projection[head_dim:, :] = np.eye(head_dim, dtype=np.float32)
    weights = checkpoint_mapper.WeightDict(
        {
            "embedding": embedding,
            "layer.0.operator_norm": ones((hidden,), dtype=np.float32),
            "layer.0.ffn_norm": ones((hidden,), dtype=np.float32),
            "layer.0.conv_in": zeros((hidden, 3 * hidden), dtype=np.float32),
            "layer.0.conv_weight": zeros((hidden, 3), dtype=np.float32),
            "layer.0.conv_out": zeros((hidden, hidden), dtype=np.float32),
            "layer.0.w1": zeros((hidden, 64), dtype=np.float32),
            "layer.0.w3": zeros((hidden, 64), dtype=np.float32),
            "layer.0.w2": zeros((64, hidden), dtype=np.float32),
            "layer.1.operator_norm": operator_gamma,
            "layer.1.ffn_norm": ones((hidden,), dtype=np.float32),
            "layer.1.w_q": identity,
            "layer.1.w_k": k_projection,
            "layer.1.w_v": v_projection,
            "layer.1.w_o": identity,
            "layer.1.q_norm": q_gamma,
            "layer.1.k_norm": k_gamma,
            "layer.1.w1": zeros((hidden, 64), dtype=np.float32),
            "layer.1.w3": zeros((hidden, 64), dtype=np.float32),
            "layer.1.w2": zeros((64, hidden), dtype=np.float32),
            "final_norm": final_gamma,
            "w_lm_head": identity,
            "_layer_types": list(parsed.layer_types),
            "_num_attention_layers": parsed.num_attention_layers,
            "_num_conv_layers": parsed.num_conv_layers,
            "_hidden_size": parsed.hidden_size,
            "_intermediate_size": parsed.intermediate_size,
            "_head_dim": parsed.head_dim,
            "_conv_l_cache": parsed.conv_l_cache,
        }
    )
    plan = model.build_lfm2_engine(
        cfg,
        weights,
        4,
        precision=precision,
        debug_layer_outputs=True,
    )
    assert plan
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    assert runtime.deserialize_cuda_engine(plan) is not None
    runner = debug_runner.Lfm2TrtRunner(
        engine_plan=plan,
        max_cache_length=4,
        num_conv_layers=1,
        num_attention_layers=1,
    )
    try:
        actual = (runner.step(1), runner.step(2))
    finally:
        runner.close()

    work_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    torch_operator_gamma = torch.from_numpy(operator_gamma).to(work_dtype)
    torch_q_gamma = torch.from_numpy(q_gamma).to(work_dtype)
    torch_k_gamma = torch.from_numpy(k_gamma).to(work_dtype)
    torch_final_gamma = torch.from_numpy(final_gamma).to(work_dtype)
    torch_q_weight = torch.from_numpy(identity).to(work_dtype)
    torch_k_weight = torch.from_numpy(k_projection).to(work_dtype)
    torch_v_weight = torch.from_numpy(v_projection).to(work_dtype)
    torch_o_weight = torch.from_numpy(identity).to(work_dtype)
    torch_lm_head = torch.from_numpy(identity).to(work_dtype)
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    def rms_norm(value, gamma):
        variance = value.float().pow(2).mean(-1, keepdim=True)
        normalized = (value.float() * torch.rsqrt(variance + 1e-5)).to(work_dtype)
        return normalized * gamma

    def apply_rope(value, absolute_position: int):
        angle = float(absolute_position) * inv_freq
        cosine = torch.cat((angle.cos(), angle.cos())).to(work_dtype)
        sine = torch.cat((angle.sin(), angle.sin())).to(work_dtype)
        rotated = torch.cat((-value[..., head_dim // 2 :], value[..., : head_dim // 2]), dim=-1)
        return value * cosine + rotated * sine

    def oracle(
        *,
        scale_scores: bool = True,
        apply_position_rope: bool = True,
        apply_qk_norm: bool = True,
        retain_prefix_cache: bool = True,
        repeat_gqa_value: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        cached_keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []
        hidden_outputs: list[np.ndarray] = []
        logits_outputs: list[np.ndarray] = []
        for absolute_position, result in enumerate(actual):
            hidden_input = torch.from_numpy(result["debug_hidden_0"][0]).to(work_dtype)
            normalized = rms_norm(hidden_input, torch_operator_gamma)
            query = (normalized @ torch_q_weight).reshape(2, head_dim)
            key = (normalized @ torch_k_weight).reshape(1, head_dim)
            value = (normalized @ torch_v_weight).reshape(1, head_dim)
            if apply_qk_norm:
                query = rms_norm(query, torch_q_gamma)
                key = rms_norm(key, torch_k_gamma)
            rope_position = absolute_position if apply_position_rope else 0
            query = apply_rope(query, rope_position)
            key = apply_rope(key, rope_position)
            if not retain_prefix_cache:
                cached_keys.clear()
                cached_values.clear()
            cached_keys.append(key[0])
            cached_values.append(value[0])
            active_keys = torch.stack(cached_keys, dim=0)
            active_values = torch.stack(cached_values, dim=0)
            contexts: list[torch.Tensor] = []
            for query_head in range(2):
                if not repeat_gqa_value and query_head == 1:
                    contexts.append(torch.zeros((head_dim,), dtype=work_dtype))
                    continue
                scores = query[query_head] @ active_keys.transpose(0, 1)
                if scale_scores:
                    scores = scores * (head_dim**-0.5)
                probabilities = torch.softmax(scores.float(), dim=-1).to(work_dtype)
                contexts.append(probabilities @ active_values)
            context = torch.stack(contexts, dim=0).reshape(hidden)
            attention_output = context @ torch_o_weight
            hidden_output = hidden_input + attention_output
            logits = rms_norm(hidden_output, torch_final_gamma) @ torch_lm_head
            hidden_outputs.append(hidden_output.float().cpu().numpy())
            logits_outputs.append(logits.float().cpu().numpy())
        return hidden_outputs, logits_outputs

    expected_hidden, expected_logits = oracle()
    tolerance = 0.025 if precision == "fp16" else 0.07
    for index, result in enumerate(actual):
        np.testing.assert_allclose(
            result["debug_hidden_1"][0],
            expected_hidden[index],
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            result["logits"][0],
            expected_logits[index],
            rtol=tolerance,
            atol=tolerance,
        )

    expected_second = expected_hidden[1]
    negative_controls = (
        oracle(scale_scores=False)[0][1],
        oracle(apply_position_rope=False)[0][1],
        oracle(apply_qk_norm=False)[0][1],
        oracle(retain_prefix_cache=False)[0][1],
        oracle(repeat_gqa_value=False)[0][1],
    )
    for wrong_second in negative_controls:
        assert np.max(np.abs(expected_second - wrong_second)) > 0.01
