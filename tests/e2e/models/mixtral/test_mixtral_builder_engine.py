"""Per-family engine tests for Mixtral (Mixture of Experts).

Intention:
    Validate the Mixtral family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    Mixtral uses standard RMSNorm + RoPE + GQA attention (no biases) but
    replaces the SwiGLU MLP with a router + N expert MLPs. The router uses
    standard top-k softmax to select top-2 experts per token. Each expert
    is a SwiGLU MLP with gate/up/down projections.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    get_config_dict() (to add num_local_experts and num_experts_per_tok),
    make_hf_tensors() (for MoE weight layout with router + per-expert MLPs),
    and expected_weight_keys() (for router + expert.{e}.w_gate/up/down keys).
    Uses a tiny model with 2 experts to keep engine build fast.

Trace: ARCH-FAM-001, UD-FAM-MIXTRAL-01
Intent: Validate the Mixtral MoE family plugin weight loading including router weights, per-expert SwiGLU MLP mapping, and MoE config fields (num_local_experts, num_experts_per_tok).
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Router and per-expert weight keys are present, expert MLP shapes are correct, and MoE-specific config fields are parsed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("safetensors not available", allow_module_level=True)

try:
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


# Number of experts kept tiny for fast engine builds.
_NUM_EXPERTS = 2
_NUM_EXPERTS_PER_TOK = 1


class MixtralPluginTester(FamilyPluginTester):
    """Tester for the Mixtral family plugin.

    Mixtral uses:
      - RMSNorm (no biases)
      - Standard RoPE + GQA attention (no biases)
      - MoE MLP: router [num_experts, hidden] + per-expert SwiGLU
        (w1/w3/w2 -> gate/up/down)
      - Standard top-k softmax routing with renormalized weights
    """

    plugin_module = "tensorrt_model_connect.families.mixtral"
    model_type = "mixtral"

    def get_config_dict(self) -> dict:
        """Mixtral config with MoE-specific fields."""
        d = super().get_config_dict()
        d["num_local_experts"] = _NUM_EXPERTS
        d["num_experts_per_tok"] = _NUM_EXPERTS_PER_TOK
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Mixtral's weight layout.

        Key differences from standard decoder:
          - No gate_proj/up_proj/down_proj at layer level
          - Instead: block_sparse_moe.gate.weight [num_experts, hidden]
          - Per-expert: block_sparse_moe.experts.{e}.w1.weight [inter, hidden] (gate)
          - Per-expert: block_sparse_moe.experts.{e}.w3.weight [inter, hidden] (up)
          - Per-expert: block_sparse_moe.experts.{e}.w2.weight [hidden, inter] (down)
          - RMSNorm only (no biases on norms or attention)
        """
        s = self.spec
        kv_hidden = s.num_key_value_heads * s.head_dim
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.embed_tokens.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            # RMSNorm (no bias)
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            # Q/K/V/O projections (no biases)
            t[f"{p}.self_attn.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.o_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            # Router
            t[f"{p}.block_sparse_moe.gate.weight"] = rand(
                _NUM_EXPERTS, s.hidden_size)
            # Per-expert SwiGLU
            for e in range(_NUM_EXPERTS):
                ep = f"{p}.block_sparse_moe.experts.{e}"
                t[f"{ep}.w1.weight"] = rand(s.intermediate_size, s.hidden_size)
                t[f"{ep}.w3.weight"] = rand(s.intermediate_size, s.hidden_size)
                t[f"{ep}.w2.weight"] = rand(s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Mixtral weight keys: standard attention + router + per-expert SwiGLU."""
        s = self.spec
        keys = {"embedding", "final_norm", "w_out"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.input_norm",
                f"{prefix}.post_attn_norm",
                f"{prefix}.router",
            })
            for e in range(_NUM_EXPERTS):
                keys.update({
                    f"{prefix}.expert.{e}.w_gate",
                    f"{prefix}.expert.{e}.w_up",
                    f"{prefix}.expert.{e}.w_down",
                })
        return keys


class TestMixtralEngine(FamilyPluginTestMixin):
    """Engine tests for Mixtral family plugin."""

    tester_class = MixtralPluginTester
