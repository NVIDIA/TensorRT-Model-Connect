"""Per-family engine tests for Nemotron-4.

Intention:
    Validate the Nemotron-4 family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    Nemotron-4 uses LayerNorm1P (gamma offset +1), 2-projection MLP
    (up_proj/down_proj with squared ReLU), GQA, partial RoPE, and optional
    LayerNorm biases. The plugin applies the +1 gamma offset during weight
    loading, so the test verifies that loaded norm weights differ from raw
    HF values by exactly +1.0.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    make_hf_tensors() to produce Nemotron's HF weight layout (up_proj/down_proj
    MLP, LayerNorm biases), expected_weight_keys() for fc1/fc2 MLP keys +
    norm betas, and get_config_dict() to add partial_rotary_factor.

Trace: ARCH-FAM-001, UD-FAM-NEMOTRON-01
Intent: Validate the Nemotron-4 family plugin weight loading including LayerNorm1P gamma offset, 2-projection MLP with squared ReLU, partial RoPE, and LayerNorm biases.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Norm gamma weights are offset by +1.0, fc1/fc2 MLP keys map correctly, norm biases are loaded, and partial RoPE config is parsed.
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


class NemotronPluginTester(FamilyPluginTester):
    """Tester for the Nemotron-4 family plugin.

    Nemotron-4 uses:
      - LayerNorm1P (gamma offset +1) with bias
      - 2-projection MLP (up_proj -> relu^2 -> down_proj)
      - GQA (grouped query attention)
      - Partial RoPE (partial_rotary_factor = 0.5)
    """

    plugin_module = "tensorrt_model_connect.families.nemotron"
    model_type = "nemotron"

    def get_config_dict(self) -> dict:
        d = super().get_config_dict()
        d["partial_rotary_factor"] = 0.5
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Nemotron's weight layout.

        Key differences from standard decoder:
          - MLP uses up_proj/down_proj (2-projection, no gate_proj)
          - LayerNorm with bias
          - Final norm with bias
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
            # LayerNorm with bias
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.input_layernorm.bias"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.bias"] = rand(s.hidden_size)
            # Q/K/V/O projections
            t[f"{p}.self_attn.q_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.o_proj.weight"] = rand(s.hidden_size, s.hidden_size)
            # 2-projection MLP: up_proj/down_proj (no gate_proj)
            t[f"{p}.mlp.up_proj.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.down_proj.weight"] = rand(s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["model.norm.bias"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Nemotron weight keys: fc1/fc2 MLP + norm betas."""
        s = self.spec
        keys = {"embedding", "final_norm", "final_norm_beta", "w_out"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.w_fc1",
                f"{prefix}.w_fc2",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                f"{prefix}.post_attn_norm",
                f"{prefix}.post_attn_norm_beta",
            })
        return keys


class TestNemotronEngine(FamilyPluginTestMixin):
    """Engine tests for Nemotron-4 family plugin."""

    tester_class = NemotronPluginTester
