# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/static contracts for the family-owned LFM2-MoE Python builder."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.lfm2_moe.config import validate_lfm2_moe_config


checkpoint_mapper = importlib.import_module(
    "tensorrt_model_connect.families.lfm2_moe.checkpoint_mapper"
)
debug_runner = importlib.import_module("tensorrt_model_connect.families.lfm2_moe.debug_runner")


_ROOT = Path(__file__).resolve().parents[2]
_MODEL_SOURCE = _ROOT / "python" / "tensorrt_model_connect" / "families" / "lfm2_moe" / "model.py"
_PLUGIN_SOURCE = _MODEL_SOURCE.with_name("plugin.py")


def _config(raw: dict) -> SimpleNamespace:
    return SimpleNamespace(raw=raw)


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _lfm2_8b_a1b_raw() -> dict:
    layer_types = ["conv"] * 24
    for index in (2, 6, 10, 14, 18, 21):
        layer_types[index] = "full_attention"
    return {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": 65536,
        "hidden_size": 2048,
        "intermediate_size": 7168,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "layer_types": layer_types,
        "conv_L_cache": 3,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 128000,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1792,
        "num_dense_layers": 2,
        "use_expert_bias": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "bos_token_id": 1,
        "eos_token_id": 7,
        "pad_token_id": 0,
    }


def _tiny_moe_raw() -> dict:
    return {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "norm_eps": 1e-5,
        "rope_theta": 1000.0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 3,
        "num_dense_layers": 1,
        "use_expert_bias": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "use_pos_enc": True,
    }


def test_moe_config_accepts_the_pinned_8b_a1b_schema() -> None:
    parsed = validate_lfm2_moe_config(_config(_lfm2_8b_a1b_raw()))

    assert parsed.hidden_size == 2048
    assert parsed.head_dim == 64
    # HF Lfm2MoeConfig has no block_* schema: 7168 is final, never auto-adjusted.
    assert parsed.intermediate_size == 7168
    assert parsed.num_hidden_layers == 24
    assert parsed.num_attention_layers == 6
    assert parsed.num_conv_layers == 18
    assert tuple(
        index for index, kind in enumerate(parsed.layer_types) if kind == "full_attention"
    ) == (2, 6, 10, 14, 18, 21)
    assert parsed.num_experts == 32
    assert parsed.num_experts_per_tok == 4
    assert parsed.moe_intermediate_size == 1792
    assert parsed.num_dense_layers == 2
    assert parsed.num_moe_layers == 22
    assert not parsed.is_moe_layer(0)
    assert not parsed.is_moe_layer(1)
    assert parsed.is_moe_layer(2)
    assert parsed.conv_l_cache == 3
    assert parsed.conv_bias is False
    assert parsed.tie_word_embeddings is True
    assert parsed.default_cache_length == 32768
    assert (parsed.bos_token_id, parsed.eos_token_id, parsed.pad_token_id) == (1, 7, 0)

    overrides = parsed.bundle_overrides()
    # The C++ runtime contract stays MoE-free: same keys as the dense family.
    assert set(overrides) == {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "norm_eps",
        "rope_theta",
        "layer_types",
        "num_attention_layers",
        "num_conv_layers",
        "conv_L_cache",
        "conv_dim",
        "tie_word_embeddings",
        "native_kv_cache",
        "native_kv_contract_version",
    }
    assert overrides["num_attention_layers"] == 6
    assert overrides["num_conv_layers"] == 18
    assert overrides["conv_L_cache"] == 3
    assert overrides["conv_dim"] == 2048
    assert overrides["native_kv_cache"] is True
    assert overrides["native_kv_contract_version"] == 1


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"architectures": ["Lfm2ForCausalLM"]}, "belong to the lfm2 family"),
        ({"model_type": "lfm2"}, "model_type='lfm2_moe'"),
        ({"architectures": ["Lfm2MoeForConditionalGeneration"]}, "Lfm2MoeForCausalLM"),
        ({"vision_config": {"hidden_size": 128}}, "VL fields"),
        ({"use_expert_bias": False}, "use_expert_bias=true"),
        ({"norm_topk_prob": False}, "norm_topk_prob=true"),
        ({"routed_scaling_factor": 2.0}, "routed_scaling_factor=1.0"),
        ({"num_experts": 1}, "at least 2"),
        ({"num_experts_per_tok": 33}, "must not exceed num_experts"),
        ({"num_dense_layers": 24}, "at least one MoE layer"),
        ({"num_local_experts": 8}, "num_local_experts must equal num_experts"),
        ({"layer_types": ["conv", "mamba"] + ["conv"] * 22}, "layer types"),
        ({"full_attn_idxs": [0]}, "full_attn_idxs must agree with layer_types"),
    ],
)
def test_moe_config_rejects_out_of_scope_variants(update: dict, message: str) -> None:
    raw = _lfm2_8b_a1b_raw()
    raw.update(update)

    with pytest.raises(ValueError, match=message):
        validate_lfm2_moe_config(_config(raw))


@pytest.mark.parametrize(
    "missing_field",
    ["num_experts", "num_experts_per_tok", "moe_intermediate_size", "num_dense_layers"],
)
def test_moe_config_requires_every_expert_field(missing_field: str) -> None:
    raw = _lfm2_8b_a1b_raw()
    del raw[missing_field]

    with pytest.raises(ValueError, match="missing MoE fields"):
        validate_lfm2_moe_config(_config(raw))


def test_moe_config_keeps_legacy_alias_normalization() -> None:
    raw = _tiny_moe_raw()
    del raw["layer_types"]
    raw.update(
        {
            "block_dim": 4,
            "num_layers": 2,
            "num_heads": 2,
            "full_attn_idxs": [1],
            "conv_l_cache": 3,
            "block_norm_eps": 1e-5,
            "theta": 1000.0,
            "tie_embedding": False,
        }
    )

    parsed = validate_lfm2_moe_config(_config(raw))

    assert parsed.layer_types == ("conv", "full_attention")
    assert parsed.norm_eps == pytest.approx(1e-5)
    assert parsed.rope_theta == pytest.approx(1000.0)
    assert parsed.tie_word_embeddings is False


def test_moe_config_applies_dense_ffn_auto_adjust_only_with_block_schema() -> None:
    raw = _tiny_moe_raw()
    raw.update({"block_ff_dim": 6656, "intermediate_size": 6656, "block_auto_adjust_ff_dim": True})
    assert validate_lfm2_moe_config(_config(raw)).intermediate_size == 4608

    plain = _tiny_moe_raw()
    plain["intermediate_size"] = 6656
    assert validate_lfm2_moe_config(_config(plain)).intermediate_size == 6656


def test_checkpoint_mapper_routes_dense_moe_and_tied_head(monkeypatch) -> None:
    raw = _tiny_moe_raw()
    hidden = 4
    dense_inter = 6
    moe_inter = 3
    experts = 4

    def seq(*shape: int, start: int) -> np.ndarray:
        count = int(np.prod(shape))
        return np.arange(start, start + count, dtype=np.float32).reshape(shape)

    rng = np.random.default_rng(7)

    def rand(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=shape).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": seq(8, hidden, start=0),
        "model.embedding_norm.weight": seq(hidden, start=40),
        "model.layers.0.operator_norm.weight": seq(hidden, start=50),
        "model.layers.0.ffn_norm.weight": seq(hidden, start=60),
        "model.layers.0.conv.in_proj.weight": seq(3 * hidden, hidden, start=70),
        "model.layers.0.conv.conv.weight": seq(hidden, 1, 3, start=130),
        "model.layers.0.conv.out_proj.weight": seq(hidden, hidden, start=150),
        "model.layers.0.feed_forward.w1.weight": seq(dense_inter, hidden, start=170),
        "model.layers.0.feed_forward.w3.weight": seq(dense_inter, hidden, start=200),
        "model.layers.0.feed_forward.w2.weight": seq(hidden, dense_inter, start=230),
        "model.layers.1.operator_norm.weight": seq(hidden, start=260),
        "model.layers.1.ffn_norm.weight": seq(hidden, start=270),
        "model.layers.1.self_attn.q_proj.weight": seq(hidden, hidden, start=280),
        "model.layers.1.self_attn.k_proj.weight": seq(2, hidden, start=300),
        "model.layers.1.self_attn.v_proj.weight": seq(2, hidden, start=310),
        "model.layers.1.self_attn.out_proj.weight": seq(hidden, hidden, start=320),
        "model.layers.1.self_attn.q_layernorm.weight": seq(2, start=340),
        "model.layers.1.self_attn.k_layernorm.weight": seq(2, start=350),
        "model.layers.1.feed_forward.gate.weight": rand(experts, hidden),
        "model.layers.1.feed_forward.expert_bias": rand(experts),
    }
    for expert in range(experts):
        prefix = f"model.layers.1.feed_forward.experts.{expert}"
        tensors[f"{prefix}.w1.weight"] = rand(moe_inter, hidden)
        tensors[f"{prefix}.w3.weight"] = rand(moe_inter, hidden)
        tensors[f"{prefix}.w2.weight"] = rand(hidden, moe_inter)

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

    weights = checkpoint_mapper.load_lfm2_moe_weights(
        "/unused",
        _config(raw),
        precision="bf16",
    )

    assert weights["_layer_types"] == ["conv", "full_attention"]
    assert weights["_num_conv_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["layer.0.conv_in"].shape == (hidden, 3 * hidden)
    assert weights["layer.0.conv_weight"].shape == (hidden, 3)
    assert weights["layer.0.w1"].shape == (hidden, dense_inter)
    assert weights["layer.1.w_q"].shape == (hidden, hidden)
    assert weights["layer.1.q_norm"].shape == (2,)
    assert "layer.1.w1" not in weights
    assert "layer.0.moe_w13" not in weights
    assert weights["layer.1.moe_w13"].shape == (experts, hidden, 2 * moe_inter)
    assert weights["layer.1.moe_w2"].shape == (experts, moe_inter, hidden)
    assert weights["layer.1.router"].shape == (experts, hidden)
    assert weights["layer.1.expert_bias"].shape == (experts,)
    np.testing.assert_array_equal(
        weights["layer.1.router"],
        tensors["model.layers.1.feed_forward.gate.weight"],
    )
    np.testing.assert_array_equal(
        weights["layer.1.expert_bias"],
        tensors["model.layers.1.feed_forward.expert_bias"],
    )
    np.testing.assert_array_equal(
        weights["w_lm_head"],
        tensors["model.embed_tokens.weight"].T,
    )
    assert all(
        value.dtype == np.float32 for value in weights.values() if isinstance(value, np.ndarray)
    )

    # Verify the stacked orientation BY VALUE: pushing a known vector through
    # the stacked arrays must equal each expert's torch-free SwiGLU reference
    # computed from the raw HF tensors.
    probe = np.array([0.31, -1.17, 0.53, 2.09], dtype=np.float32)
    for expert in range(experts):
        prefix = f"model.layers.1.feed_forward.experts.{expert}"
        w1 = tensors[f"{prefix}.w1.weight"]
        w3 = tensors[f"{prefix}.w3.weight"]
        w2 = tensors[f"{prefix}.w2.weight"]
        reference = (_silu(w1 @ probe) * (w3 @ probe)) @ w2.T

        stacked = probe @ weights["layer.1.moe_w13"][expert]
        gate, up = stacked[:moe_inter], stacked[moe_inter:]
        actual = (_silu(gate) * up) @ weights["layer.1.moe_w2"][expert]

        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)


def test_mapper_metadata_cross_checks_the_validated_config() -> None:
    raw = _lfm2_8b_a1b_raw()
    parsed = validate_lfm2_moe_config(_config(raw))
    expected = {
        "_layer_types": list(parsed.layer_types),
        "_num_attention_layers": parsed.num_attention_layers,
        "_num_conv_layers": parsed.num_conv_layers,
        "_hidden_size": parsed.hidden_size,
        "_intermediate_size": parsed.intermediate_size,
        "_head_dim": parsed.head_dim,
        "_conv_l_cache": parsed.conv_l_cache,
        "_num_experts": parsed.num_experts,
        "_num_experts_per_tok": parsed.num_experts_per_tok,
        "_moe_intermediate_size": parsed.moe_intermediate_size,
        "_num_dense_layers": parsed.num_dense_layers,
    }

    mapper_source = (_MODEL_SOURCE.with_name("checkpoint_mapper.py")).read_text(encoding="utf-8")
    model_source = _MODEL_SOURCE.read_text(encoding="utf-8")
    for name in expected:
        assert f'weights["{name}"]' in mapper_source
        assert f'"{name}"' in model_source


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
    # Router contract: FP32 sigmoid scores, bias-selected top-k, unbiased
    # combine weights normalized with the pinned 1e-6 epsilon.
    assert "def _add_moe_ffn" in source
    assert "ActivationType.SIGMOID" in source
    assert "add_topk" in source
    assert "TopKOperation.MAX" in source
    assert "1e-6" in source

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
    assert 'expected = "lfm2_moe_hybrid_conv_attention"' in source
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


def _tiny_trt_fixture() -> tuple[dict, dict, np.ndarray, dict[str, np.ndarray]]:
    """Analytic 2-layer fixture: layer 0 dense conv, layer 1 MoE attention."""

    hidden = 64
    head_dim = 32
    vocab = 16
    experts = 4
    top_k = 2
    moe_inter = 32
    dense_inter = 48
    raw = {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": dense_inter,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "layer_types": ["conv", "full_attention"],
        "conv_L_cache": 3,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 8,
        "num_experts": experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": moe_inter,
        "num_dense_layers": 1,
        "use_expert_bias": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "use_pos_enc": True,
    }

    zeros = np.zeros
    ones = np.ones
    embedding = zeros((vocab, hidden), dtype=np.float32)
    embedding[1] = np.linspace(0.1, 1.0, hidden, dtype=np.float32)
    identity = np.eye(hidden, dtype=np.float32)
    conv_in = np.concatenate((identity, identity, identity), axis=1)
    conv_weight = np.broadcast_to(
        np.array([0.25, 0.5, 1.0], dtype=np.float32),
        (hidden, 3),
    ).copy()
    final_gamma = np.linspace(0.83, 1.17, hidden, dtype=np.float32)
    lm_head = zeros((hidden, vocab), dtype=np.float32)
    lm_head[:vocab, :] = np.eye(vocab, dtype=np.float32)

    def rms(value: np.ndarray) -> np.ndarray:
        return value / np.sqrt(np.mean(value * value) + 1e-5)

    # FP32 reference of the pre-MoE hidden state for token 1 (first step):
    # conv layer with [I, I, I] in-projection and empty state contributes
    # norm * (norm * norm); the zeroed dense FFN and zeroed attention add 0.
    norm1 = rms(embedding[1])
    hidden_pre_moe = embedding[1] + norm1 * (norm1 * norm1)
    moe_input = rms(hidden_pre_moe)

    rng = np.random.default_rng(20260820)
    w1 = rng.normal(0.0, 0.1, size=(experts, moe_inter, hidden)).astype(np.float32)
    w3 = rng.normal(0.0, 0.1, size=(experts, moe_inter, hidden)).astype(np.float32)
    w2 = rng.normal(0.0, 0.2, size=(experts, hidden, moe_inter)).astype(np.float32)
    stacked_w13 = np.concatenate(
        (np.transpose(w1, (0, 2, 1)), np.transpose(w3, (0, 2, 1))),
        axis=2,
    ).astype(np.float32)
    stacked_w2 = np.transpose(w2, (0, 2, 1)).astype(np.float32)

    # Engineer well-separated router logits at the FP32 reference input so
    # BF16/FP16 rounding cannot flip either the biased or unbiased ordering.
    target_logits = np.array([2.5, 2.0, 0.5, -1.0], dtype=np.float32)
    router = np.outer(target_logits, moe_input / float(moe_input @ moe_input)).astype(np.float32)
    expert_bias = np.array([-2.0, 0.0, 0.0, 2.0], dtype=np.float32)

    weights = {
        "embedding": embedding,
        "layer.0.operator_norm": ones((hidden,), dtype=np.float32),
        "layer.0.ffn_norm": ones((hidden,), dtype=np.float32),
        "layer.0.conv_in": conv_in,
        "layer.0.conv_weight": conv_weight,
        "layer.0.conv_out": identity,
        "layer.0.w1": zeros((hidden, dense_inter), dtype=np.float32),
        "layer.0.w3": zeros((hidden, dense_inter), dtype=np.float32),
        "layer.0.w2": zeros((dense_inter, hidden), dtype=np.float32),
        "layer.1.operator_norm": ones((hidden,), dtype=np.float32),
        "layer.1.ffn_norm": ones((hidden,), dtype=np.float32),
        "layer.1.w_q": zeros((hidden, hidden), dtype=np.float32),
        "layer.1.w_k": zeros((hidden, head_dim), dtype=np.float32),
        "layer.1.w_v": zeros((hidden, head_dim), dtype=np.float32),
        "layer.1.w_o": zeros((hidden, hidden), dtype=np.float32),
        "layer.1.q_norm": ones((head_dim,), dtype=np.float32),
        "layer.1.k_norm": ones((head_dim,), dtype=np.float32),
        "layer.1.moe_w13": stacked_w13,
        "layer.1.moe_w2": stacked_w2,
        "layer.1.router": router,
        "layer.1.expert_bias": expert_bias,
        "final_norm": final_gamma,
        "w_lm_head": lm_head,
    }
    analytic = {
        "hidden_pre_moe": hidden_pre_moe,
        "moe_input": moe_input,
        "router": router,
        "expert_bias": expert_bias,
        "stacked_w13": stacked_w13,
        "stacked_w2": stacked_w2,
        "final_gamma": final_gamma,
    }
    return raw, weights, embedding, analytic


def _reference_moe_output(
    analytic: dict[str, np.ndarray],
    *,
    top_k: int,
    moe_inter: int,
    select_with_bias: bool = True,
    combine_from_biased: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, set[int]]:
    """Closed-form router + expert math with switchable negative controls."""

    moe_input = analytic["moe_input"]
    logits = analytic["router"] @ moe_input
    scores = 1.0 / (1.0 + np.exp(-logits))
    selection_scores = scores + analytic["expert_bias"] if select_with_bias else scores
    selected = np.argsort(-selection_scores)[:top_k]
    base = (scores + analytic["expert_bias"])[selected] if combine_from_biased else scores[selected]
    combine = base / (base.sum() + 1e-6) if normalize else base
    output = np.zeros_like(moe_input)
    for weight, expert in zip(combine, selected):
        up_gate = moe_input @ analytic["stacked_w13"][expert]
        gate, up = up_gate[:moe_inter], up_gate[moe_inter:]
        output += weight * ((_silu(gate) * up) @ analytic["stacked_w2"][expert])
    return output, {int(expert) for expert in selected}


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.trt
def test_tiny_real_trt_builder_routes_experts_like_the_closed_form(precision: str) -> None:
    trt = pytest.importorskip("tensorrt")
    model = importlib.import_module("tensorrt_model_connect.families.lfm2_moe.model")
    top_k = 2
    moe_inter = 32
    raw, weight_values, embedding, analytic = _tiny_trt_fixture()
    cfg = _config(raw)
    parsed = validate_lfm2_moe_config(cfg)
    weights = checkpoint_mapper.WeightDict(weight_values)
    weights.update(
        {
            "_layer_types": list(parsed.layer_types),
            "_num_attention_layers": parsed.num_attention_layers,
            "_num_conv_layers": parsed.num_conv_layers,
            "_hidden_size": parsed.hidden_size,
            "_intermediate_size": parsed.intermediate_size,
            "_head_dim": parsed.head_dim,
            "_conv_l_cache": parsed.conv_l_cache,
            "_num_experts": parsed.num_experts,
            "_num_experts_per_tok": parsed.num_experts_per_tok,
            "_moe_intermediate_size": parsed.moe_intermediate_size,
            "_num_dense_layers": parsed.num_dense_layers,
        }
    )

    plan = model.build_lfm2_moe_engine(
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
    assert tuple(engine.get_tensor_shape("cache_k_0")) == (1, 1, 8, 32)
    assert tuple(engine.get_tensor_shape("conv_state_0")) == (64, 3)

    runner = debug_runner.Lfm2MoeTrtRunner(
        engine_plan=plan,
        max_cache_length=8,
        num_conv_layers=1,
        num_attention_layers=1,
    )
    try:
        first = runner.step(1)
    finally:
        runner.close()
    assert first["logits"].shape == (1, 16)

    def rms(value: np.ndarray) -> np.ndarray:
        return value / np.sqrt(np.mean(value * value) + 1e-5)

    moe_output, selected = _reference_moe_output(analytic, top_k=top_k, moe_inter=moe_inter)
    expected_hidden = analytic["hidden_pre_moe"] + moe_output
    expected_logits = (analytic["final_gamma"] * rms(expected_hidden))[:16]

    tolerance = 0.02 if precision == "fp16" else 0.05
    actual_hidden = first["debug_hidden_1"][0]
    np.testing.assert_allclose(actual_hidden, expected_hidden, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(first["logits"][0], expected_logits, rtol=tolerance, atol=tolerance)

    # Negative control (a): selection without expert_bias picks a different
    # expert set and moves the layer output detectably.
    unbiased_output, unbiased_selected = _reference_moe_output(
        analytic,
        top_k=top_k,
        moe_inter=moe_inter,
        select_with_bias=False,
    )
    assert unbiased_selected != selected
    wrong_hidden_a = analytic["hidden_pre_moe"] + unbiased_output
    assert np.max(np.abs(wrong_hidden_a - expected_hidden)) > 0.05
    assert np.max(np.abs(wrong_hidden_a - actual_hidden)) > 0.04

    # Negative control (b): combine weights taken from the BIASED scores
    # diverge from the engine, proving the bias touches selection only.
    biased_combine_output, _ = _reference_moe_output(
        analytic,
        top_k=top_k,
        moe_inter=moe_inter,
        combine_from_biased=True,
    )
    wrong_hidden_b = analytic["hidden_pre_moe"] + biased_combine_output
    assert np.max(np.abs(wrong_hidden_b - expected_hidden)) > 0.05
    assert np.max(np.abs(wrong_hidden_b - actual_hidden)) > 0.04

    # Negative control (c): skipping the +1e-6 top-k normalization diverges.
    unnormalized_output, _ = _reference_moe_output(
        analytic,
        top_k=top_k,
        moe_inter=moe_inter,
        normalize=False,
    )
    wrong_hidden_c = analytic["hidden_pre_moe"] + unnormalized_output
    assert np.max(np.abs(wrong_hidden_c - expected_hidden)) > 0.05
    assert np.max(np.abs(wrong_hidden_c - actual_hidden)) > 0.04
