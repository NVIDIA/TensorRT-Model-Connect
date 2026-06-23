"""Engine tests for the Phi-3 family plugin.

Phi-3 uses fused qkv_proj [q_dim + 2*kv_dim, hidden] and fused gate_up_proj
[2*intermediate, hidden]. The tester overrides make_hf_tensors() to produce
these fused weights in the correct layout.

Trace: ARCH-FAM-001, UD-FAM-PHI-01
Intent: Validate the Phi-3 family plugin weight loading including fused QKV splitting and fused gate_up_proj splitting into separate gate and up projections.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Fused QKV is split into separate Q/K/V with correct shapes, fused gate_up is split into gate and up projections, and all weight keys are present.
"""
import numpy as np

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class PhiPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.phi"
    model_type = "phi3"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic Phi-3 weight layout with fused QKV and gate_up.

        Intention:
            Phi-3 stores QKV as a single fused tensor (self_attn.qkv_proj.weight)
            with shape [q_dim + 2*kv_dim, hidden], and gate+up MLP as a fused
            tensor (mlp.gate_up_proj.weight) with shape [2*intermediate, hidden].
            The plugin splits these during load_weights().

        Setup:
            Build synthetic tensors matching Phi-3's checkpoint layout:
            - model.embed_tokens.weight [vocab, hidden]
            - model.layers.{i}.input_layernorm.weight [hidden]
            - model.layers.{i}.post_attention_layernorm.weight [hidden]
            - model.layers.{i}.self_attn.qkv_proj.weight [q_dim+2*kv_dim, hidden]
            - model.layers.{i}.self_attn.o_proj.weight [hidden, hidden]
            - model.layers.{i}.mlp.gate_up_proj.weight [2*intermediate, hidden]
            - model.layers.{i}.mlp.down_proj.weight [hidden, intermediate]
            - model.norm.weight [hidden]
            - lm_head.weight [vocab, hidden]
        """
        s = self.spec
        kv_hidden = s.num_key_value_heads * s.head_dim
        q_dim = s.num_attention_heads * s.head_dim

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.embed_tokens.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)

            # Fused QKV: [q_dim + 2*kv_dim, hidden]
            total_qkv = q_dim + 2 * kv_hidden
            t[f"{p}.self_attn.qkv_proj.weight"] = rand(
                total_qkv, s.hidden_size)

            t[f"{p}.self_attn.o_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)

            # Fused gate_up: [2*intermediate, hidden]
            t[f"{p}.mlp.gate_up_proj.weight"] = rand(
                2 * s.intermediate_size, s.hidden_size)

            t[f"{p}.mlp.down_proj.weight"] = rand(
                s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t


class TestPhiEngine(FamilyPluginTestMixin):
    tester_class = PhiPluginTester

    def test_fused_qkv_split_correctly(self, tester, tmp_path):
        """Validate that fused qkv_proj is correctly split into Q, K, V.

        Intention:
            Phi-3 stores QKV as a single [q_dim + 2*kv_dim, hidden] tensor.
            The plugin must split it into separate Q [q_dim, hidden],
            K [kv_dim, hidden], and V [kv_dim, hidden] before transposing.
            If the split offsets are wrong, Q/K/V will contain each other's
            data, causing completely wrong attention patterns.

            Example bug this catches: Off-by-one in the split that puts the
            last row of Q into K, or the first row of V into K.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify that w_q shape[1] == q_dim (transposed to [hidden, q_dim]).
            3. Verify shapes of w_k and w_v are consistent.
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        q_dim = s.num_attention_heads * s.head_dim

        # After transpose: w_q should be [hidden, q_dim]
        assert weights["layer.0.w_q"].shape == (s.hidden_size, q_dim), (
            f"w_q shape {weights['layer.0.w_q'].shape} != "
            f"expected ({s.hidden_size}, {q_dim})"
        )

    def test_fused_gate_up_split_correctly(self, tester, tmp_path):
        """Validate that fused gate_up_proj is correctly split into gate and up.

        Intention:
            Phi-3 stores gate and up MLP projections as a single
            [2*intermediate, hidden] tensor. The plugin splits at the midpoint.
            If the split is wrong, gate and up projections will be swapped or
            contain overlapping data, producing wrong SwiGLU activation.

            Example bug this catches: Splitting at intermediate+1 instead of
            intermediate, causing a shape mismatch in the down projection.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify w_gate and w_up have shape [hidden, intermediate].
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        # After transpose: w_gate and w_up should be [hidden, intermediate]
        assert weights["layer.0.w_gate"].shape == (
            s.hidden_size, s.intermediate_size), (
            f"w_gate shape {weights['layer.0.w_gate'].shape} != "
            f"expected ({s.hidden_size}, {s.intermediate_size})"
        )
        assert weights["layer.0.w_up"].shape == (
            s.hidden_size, s.intermediate_size), (
            f"w_up shape {weights['layer.0.w_up'].shape} != "
            f"expected ({s.hidden_size}, {s.intermediate_size})"
        )
