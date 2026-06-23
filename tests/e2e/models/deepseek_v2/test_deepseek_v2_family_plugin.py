"""Branch-focused tests for the DeepSeek-V2 family plugin.

Trace: ARCH-FAM-001, UD-FAM-DEEPSEEK-V2
Intent: Validate DeepSeek-V2 MLA cache head_dim override and weight loading branches
Preconditions: ModelConfig with MLA-specific qk_nope_head_dim and qk_rope_head_dim fields is provided
Postconditions: Bundle config overrides compute correct K-cache head_dim and weights load with correct routing
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.deepseek_v2 as deepseek_v2
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch,
                     tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(deepseek_v2, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(
        deepseek_v2, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(deepseek_v2, "_load_tensor", _load)


def test_get_bundle_config_overrides_injects_k_cache_head_dim():
    """Intent: verify bundle head_dim override for MLA cache sizing.
    Preconditions: raw config defines qk_nope_head_dim and qk_rope_head_dim.
    Postconditions: returned head_dim equals the summed K per-head dimension.
    """
    cfg = ModelConfig(
        model_type="deepseek_v2",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=7,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        raw={"qk_nope_head_dim": 96, "qk_rope_head_dim": 32},
    )
    assert deepseek_v2.plugin.get_bundle_config_overrides(cfg) == {"head_dim": 128}


def test_load_weights_v2_lite_dense_and_moe_routing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: execute direct-Q path and dense/MoE routing schedule branches.
    Preconditions: q_lora_rank is None and only one layer qualifies as MoE by schedule.
    Postconditions: direct-Q keys, MoE keys, dense keys, and metadata/fallbacks are populated correctly.
    """
    raw = {
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 1,
        "v_head_dim": 2,
        "kv_lora_rank": 4,
        "q_lora_rank": None,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 2,
        "moe_intermediate_size": 5,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
    }
    cfg = ModelConfig(
        model_type="deepseek_v2",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=7,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": _seq(6, 8, start=0),
    }

    for layer_idx in range(3):
        hf = f"model.layers.{layer_idx}"
        tensors[f"{hf}.input_layernorm.weight"] = _seq(8, start=100 + 10 * layer_idx)
        tensors[f"{hf}.post_attention_layernorm.weight"] = _seq(
            8, start=130 + 10 * layer_idx
        )
        tensors[f"{hf}.self_attn.q_proj.weight"] = _seq(8, 8, start=200 + 50 * layer_idx)
        tensors[f"{hf}.self_attn.kv_a_proj_with_mqa.weight"] = _seq(
            5, 8, start=300 + 50 * layer_idx
        )
        tensors[f"{hf}.self_attn.kv_a_layernorm.weight"] = _seq(
            4, start=400 + 10 * layer_idx
        )
        tensors[f"{hf}.self_attn.kv_b_proj.weight"] = _seq(
            10, 4, start=450 + 50 * layer_idx
        )
        tensors[f"{hf}.self_attn.o_proj.weight"] = _seq(8, 4, start=550 + 50 * layer_idx)

    # Layer 0 dense MLP.
    tensors["model.layers.0.mlp.gate_proj.weight"] = _seq(7, 8, start=700)
    tensors["model.layers.0.mlp.up_proj.weight"] = _seq(7, 8, start=760)
    tensors["model.layers.0.mlp.down_proj.weight"] = _seq(8, 7, start=820)

    # Layer 1 MoE.
    tensors["model.layers.1.mlp.gate.weight"] = _seq(2, 8, start=900)
    for expert in range(2):
        ep = f"model.layers.1.mlp.experts.{expert}"
        tensors[f"{ep}.gate_proj.weight"] = _seq(5, 8, start=950 + expert * 100)
        tensors[f"{ep}.up_proj.weight"] = _seq(5, 8, start=980 + expert * 100)
        tensors[f"{ep}.down_proj.weight"] = _seq(8, 5, start=1010 + expert * 100)
    tensors["model.layers.1.mlp.shared_experts.gate_proj.weight"] = _seq(5, 8, start=1200)
    tensors["model.layers.1.mlp.shared_experts.up_proj.weight"] = _seq(5, 8, start=1260)
    tensors["model.layers.1.mlp.shared_experts.down_proj.weight"] = _seq(8, 5, start=1320)

    # Layer 2 dense MLP (not MoE because frequency filter excludes it).
    tensors["model.layers.2.mlp.gate_proj.weight"] = _seq(7, 8, start=1400)
    tensors["model.layers.2.mlp.up_proj.weight"] = _seq(7, 8, start=1460)
    tensors["model.layers.2.mlp.down_proj.weight"] = _seq(8, 7, start=1520)

    # model.norm.weight and lm_head.weight intentionally omitted (fallbacks).
    _patch_tensor_io(monkeypatch, tensors)

    weights = deepseek_v2.plugin.load_weights("/unused", cfg)

    # Direct-Q path present; LoRA-Q path absent.
    assert "layer.0.w_q" in weights
    assert "layer.0.w_q_a" not in weights
    assert "layer.0.w_q_b" not in weights

    # Layer 0 and 2 are dense.
    for idx in (0, 2):
        assert f"layer.{idx}.w_gate" in weights
        assert f"layer.{idx}.w_up" in weights
        assert f"layer.{idx}.w_down" in weights
        assert f"layer.{idx}.router" not in weights

    # Layer 1 is MoE.
    assert "layer.1.router" in weights
    assert "layer.1.w_gate" not in weights
    for expert in range(2):
        assert f"layer.1.expert.{expert}.w_gate" in weights
        assert f"layer.1.expert.{expert}.w_up" in weights
        assert f"layer.1.expert.{expert}.w_down" in weights
    assert "layer.1.shared.w_gate" in weights
    assert "layer.1.shared.w_up" in weights
    assert "layer.1.shared.w_down" in weights

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_out"], tensors["model.embed_tokens.weight"].T.astype(np.float32)
    )

    assert weights["_attention_size"] == 8
    assert weights["_kv_lora_rank"] == 4
    assert weights["_q_lora_rank"] is None
    assert weights["_n_routed_experts"] == 2
    assert weights["_n_shared_experts"] == 1
    assert weights["_first_k_dense_replace"] == 1
    assert weights["_moe_layer_freq"] == 2
    assert weights["_moe_intermediate_size"] == 5
    assert weights["_shared_intermediate_size"] == 5
    assert weights["_norm_topk_prob"] is True
    assert weights["_routed_scaling_factor"] == 1.5


def test_load_weights_q_lora_branch_with_present_final_and_lm_head(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: execute Q-LoRA projection branch and explicit final/lm-head branches.
    Preconditions: q_lora_rank > 0 with a single dense layer and explicit final/lm-head tensors.
    Postconditions: Q-LoRA keys are used instead of direct-Q key, and explicit final/lm-head tensors are consumed.
    """
    raw = {
        "qk_nope_head_dim": 3,
        "qk_rope_head_dim": 1,
        "v_head_dim": 2,
        "kv_lora_rank": 4,
        "q_lora_rank": 2,
        "n_routed_experts": 1,
        "n_shared_experts": 1,
        "first_k_dense_replace": 10,
    }
    cfg = ModelConfig(
        model_type="deepseek_v2",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=7,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": _seq(6, 8, start=0),
        "model.layers.0.input_layernorm.weight": _seq(8, start=100),
        "model.layers.0.post_attention_layernorm.weight": _seq(8, start=130),
        "model.layers.0.self_attn.q_a_proj.weight": _seq(2, 8, start=200),
        "model.layers.0.self_attn.q_a_layernorm.weight": _seq(2, start=220),
        "model.layers.0.self_attn.q_b_proj.weight": _seq(8, 2, start=230),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": _seq(5, 8, start=260),
        "model.layers.0.self_attn.kv_a_layernorm.weight": _seq(4, start=300),
        "model.layers.0.self_attn.kv_b_proj.weight": _seq(10, 4, start=320),
        "model.layers.0.self_attn.o_proj.weight": _seq(8, 4, start=360),
        "model.layers.0.mlp.gate_proj.weight": _seq(7, 8, start=400),
        "model.layers.0.mlp.up_proj.weight": _seq(7, 8, start=460),
        "model.layers.0.mlp.down_proj.weight": _seq(8, 7, start=520),
        "model.norm.weight": _seq(8, start=600),
        "lm_head.weight": _seq(6, 8, start=700),
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = deepseek_v2.plugin.load_weights("/unused", cfg)

    assert "layer.0.w_q" not in weights
    assert "layer.0.w_q_a" in weights
    assert "layer.0.q_a_norm" in weights
    assert "layer.0.w_q_b" in weights
    np.testing.assert_allclose(weights["final_norm"], tensors["model.norm.weight"])
    np.testing.assert_allclose(weights["w_out"], tensors["lm_head.weight"].T.astype(np.float32))


def test_matches_accepts_v2_and_v3_aliases():
    """Intent: verify model-type matching aliases.
    Preconditions: inputs include valid aliases and a non-matching family name.
    Postconditions: only deepseek_v2 and deepseek_v3 match.
    """
    assert deepseek_v2.plugin.matches("deepseek_v2")
    assert deepseek_v2.plugin.matches("DeepSeek_V3")
    assert not deepseek_v2.plugin.matches("deepseek_v1")
