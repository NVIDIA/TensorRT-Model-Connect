"""Branch-focused tests for the GPT-OSS family plugin.

Trace: ARCH-FAM-001, UD-FAM-GPT-OSS
Intent: Validate GPT-OSS family plugin weight loading via HuggingFace AutoModel state_dict path
Preconditions: Fake HF model loader returns synthetic state_dict with GPT-OSS weight naming
Postconditions: Plugin correctly maps HF weight keys to canonical names with expected transforms
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.gpt_oss as gpt_oss
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _install_fake_hf_loader(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, np.ndarray],
    call_log: list[dict[str, object]],
) -> None:
    class _FakeTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _FakeModel:
        def state_dict(self):
            return {k: _FakeTensor(v) for k, v in state.items()}

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_dir, torch_dtype=None, low_cpu_mem_usage=None):
            call_log.append(
                {
                    "model_dir": model_dir,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": low_cpu_mem_usage,
                }
            )
            return _FakeModel()

    fake_torch = types.SimpleNamespace(bfloat16="fake-bfloat16")
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_AutoModelForCausalLM)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_load_weights_keeps_kv_biases_compact_and_unpacks_experts(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: verify compact KV-bias and packed-expert de-interleave branches.
    Preconditions: q_dim != kv_dim and packed expert tensors are present.
    Postconditions: K/V biases stay at kv_dim and expert tensors split by even/odd layout.
    """
    cfg = ModelConfig(
        model_type="gpt_oss",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={"num_local_experts": 2, "num_experts_per_tok": 3},
    )

    state = {
        "model.embed_tokens.weight": _seq(6, 8, start=0),
        "model.layers.0.input_layernorm.weight": _seq(8, start=100),
        "model.layers.0.post_attention_layernorm.weight": _seq(8, start=200),
        "model.layers.0.self_attn.q_proj.weight": _seq(8, 8, start=300),
        "model.layers.0.self_attn.k_proj.weight": _seq(4, 8, start=400),
        "model.layers.0.self_attn.v_proj.weight": _seq(4, 8, start=500),
        "model.layers.0.self_attn.o_proj.weight": _seq(8, 8, start=600),
        "model.layers.0.self_attn.q_proj.bias": _seq(8, start=700),
        "model.layers.0.self_attn.k_proj.bias": np.array(
            [11.0, 12.0, 21.0, 22.0], dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.bias": np.array(
            [31.0, 32.0, 41.0, 42.0], dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.bias": _seq(8, start=710),
        "model.layers.0.self_attn.sinks": _seq(4, start=720),
        "model.layers.0.mlp.router.weight": _seq(2, 8, start=800),
        "model.layers.0.mlp.router.bias": _seq(2, start=820),
        "model.layers.0.mlp.experts.gate_up_proj": _seq(2, 8, 6, start=900),
        "model.layers.0.mlp.experts.gate_up_proj_bias": _seq(2, 6, start=1200),
        "model.layers.0.mlp.experts.down_proj": _seq(2, 3, 8, start=1400),
        "model.layers.0.mlp.experts.down_proj_bias": _seq(2, 8, start=1700),
        "model.norm.weight": _seq(8, start=1800),
        "lm_head.weight": _seq(6, 8, start=1900),
        "lm_head.bias": _seq(6, start=2000),
    }

    call_log: list[dict[str, object]] = []
    _install_fake_hf_loader(monkeypatch, state, call_log)

    weights = gpt_oss.plugin.load_weights("/fake/model", cfg)

    assert call_log[0]["model_dir"] == "/fake/model"
    assert call_log[0]["torch_dtype"] == "fake-bfloat16"
    assert call_log[0]["low_cpu_mem_usage"] is True

    np.testing.assert_allclose(
        weights["layer.0.k_bias"],
        state["model.layers.0.self_attn.k_proj.bias"])
    np.testing.assert_allclose(
        weights["layer.0.v_bias"],
        state["model.layers.0.self_attn.v_proj.bias"])

    gate_up = state["model.layers.0.mlp.experts.gate_up_proj"]
    gate_up_bias = state["model.layers.0.mlp.experts.gate_up_proj_bias"]
    np.testing.assert_allclose(
        weights["layer.0.expert.0.w_gate"], gate_up[0][:, ::2].astype(np.float32))
    np.testing.assert_allclose(
        weights["layer.0.expert.0.w_up"], gate_up[0][:, 1::2].astype(np.float32))
    np.testing.assert_allclose(
        weights["layer.0.expert.0.gate_bias"], gate_up_bias[0][::2].astype(np.float32))
    np.testing.assert_allclose(
        weights["layer.0.expert.0.up_bias"], gate_up_bias[0][1::2].astype(np.float32))

    assert "layer.0.sinks" in weights
    np.testing.assert_allclose(weights["final_norm"], state["model.norm.weight"])
    np.testing.assert_allclose(weights["w_out"], state["lm_head.weight"].T.astype(np.float32))
    np.testing.assert_allclose(weights["lm_head_bias"], state["lm_head.bias"])
    assert weights["_attention_size"] == 8
    assert weights["_num_experts"] == 2
    assert weights["_moe_intermediate_size"] == 3
    assert weights["_num_experts_per_tok"] == 3


def test_load_weights_no_kv_expansion_and_output_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: verify direct KV-bias path and final/lm-head fallback behavior.
    Preconditions: q_dim == kv_dim, with final norm and lm_head omitted from state dict.
    Postconditions: K/V biases remain unexpanded, final_norm becomes ones, and embedding-tied output is used.
    """
    cfg = ModelConfig(
        model_type="gpt_oss",
        vocab_size=6,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        raw={"num_local_experts": 1},
    )

    state = {
        "model.embed_tokens.weight": _seq(6, 8, start=0),
        "model.layers.0.input_layernorm.weight": _seq(8, start=100),
        "model.layers.0.post_attention_layernorm.weight": _seq(8, start=200),
        "model.layers.0.self_attn.q_proj.weight": _seq(8, 8, start=300),
        "model.layers.0.self_attn.k_proj.weight": _seq(8, 8, start=400),
        "model.layers.0.self_attn.v_proj.weight": _seq(8, 8, start=500),
        "model.layers.0.self_attn.o_proj.weight": _seq(8, 8, start=600),
        "model.layers.0.self_attn.q_proj.bias": _seq(8, start=700),
        "model.layers.0.self_attn.k_proj.bias": _seq(8, start=720),
        "model.layers.0.self_attn.v_proj.bias": _seq(8, start=740),
        "model.layers.0.self_attn.o_proj.bias": _seq(8, start=760),
        "model.layers.0.mlp.router.weight": _seq(1, 8, start=800),
        "model.layers.0.mlp.router.bias": _seq(1, start=810),
        "model.layers.0.mlp.experts.gate_up_proj": _seq(1, 8, 4, start=820),
        "model.layers.0.mlp.experts.gate_up_proj_bias": _seq(1, 4, start=860),
        "model.layers.0.mlp.experts.down_proj": _seq(1, 2, 8, start=880),
        "model.layers.0.mlp.experts.down_proj_bias": _seq(1, 8, start=920),
    }

    call_log: list[dict[str, object]] = []
    _install_fake_hf_loader(monkeypatch, state, call_log)

    weights = gpt_oss.plugin.load_weights("/fake/model", cfg)

    np.testing.assert_allclose(
        weights["layer.0.k_bias"], state["model.layers.0.self_attn.k_proj.bias"])
    np.testing.assert_allclose(
        weights["layer.0.v_bias"], state["model.layers.0.self_attn.v_proj.bias"])
    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_out"], state["model.embed_tokens.weight"].T.astype(np.float32))
    assert "lm_head_bias" not in weights
    assert "layer.0.sinks" not in weights
    assert weights["_num_experts_per_tok"] == 4


def test_matches_is_case_insensitive_exact():
    """Intent: verify model_type matching guard.
    Preconditions: candidate model types include exact, case variant, and non-matching names.
    Postconditions: only case-insensitive `gpt_oss` is accepted.
    """
    assert gpt_oss.plugin.matches("gpt_oss")
    assert gpt_oss.plugin.matches("GPT_OSS")
    assert not gpt_oss.plugin.matches("gpt-oss")
