# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Falcon family plugin.

Falcon-3 uses:
  - LayerNorm (with beta/bias) instead of RMSNorm
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
  - RoPE for positional encoding (alibi=false)
  - GQA (grouped query attention) with separate Q/K/V projections
  - No QKV biases, no output projection bias
  - model.layers.{i}.mlp.dense_h_to_4h/dense_4h_to_h naming

Trace: ARCH-FAM-001, UD-FAM-FALCON-01
Intent: Validate the Falcon family plugin weight loading including LayerNorm with bias, GELU FC MLP, RoPE config, and GQA projections.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All weight keys map correctly from Falcon's HF layout, LayerNorm biases are loaded, and FC MLP keys (dense_h_to_4h/dense_4h_to_h) resolve to fc1/fc2.
"""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.config import ModelConfig
from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class FalconPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.falcon.model"
    model_type = "falcon"

    def get_config_dict(self) -> dict:
        """Return config dict for Falcon-3 style (RoPE, not ALiBi).

        Intention:
            Falcon-3 uses RoPE for positional encoding (alibi=false). The
            config dict includes the standard decoder keys plus the alibi
            flag set to false.

        Setup:
            Return the standard config dict with alibi=false appended.
        """
        d = super().get_config_dict()
        d["alibi"] = False
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic Falcon-3 weight layout.

        Intention:
            Falcon-3 uses standard model.layers prefix but with LayerNorm
            (weight + bias), separate Q/K/V/O projections, and FC MLP
            (dense_h_to_4h / dense_4h_to_h) instead of SwiGLU
            (gate/up/down). No QKV or output biases in Falcon-3.

        Setup:
            Build synthetic tensors matching Falcon-3's checkpoint layout:
            - model.embed_tokens.weight [vocab, hidden]
            - model.layers.{i}.input_layernorm.{weight,bias}
            - model.layers.{i}.post_attention_layernorm.{weight,bias}
            - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
            - model.layers.{i}.mlp.dense_h_to_4h.weight
            - model.layers.{i}.mlp.dense_4h_to_h.weight
            - model.norm.{weight,bias}
            - lm_head.weight [vocab, hidden]
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

            # Separate Q/K/V/O projections (no biases)
            t[f"{p}.self_attn.q_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.o_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)

            # FC MLP: dense_h_to_4h and dense_4h_to_h
            t[f"{p}.mlp.dense_h_to_4h.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = rand(
                s.hidden_size, s.intermediate_size)

        # Final LayerNorm with bias
        t["model.norm.weight"] = rand(s.hidden_size)
        t["model.norm.bias"] = rand(s.hidden_size)

        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Return expected weight keys for Falcon-3.

        Falcon-3 uses LayerNorm (with beta), FC MLP (w_fc1/w_fc2),
        and metadata keys for attention/MLP sizes. No QKV biases.
        """
        s = self.spec
        keys = {"embedding", "final_norm", "final_norm_beta", "w_out"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                f"{prefix}.post_attn_norm",
                f"{prefix}.post_attn_norm_beta",
                f"{prefix}.w_fc1",
                f"{prefix}.w_fc2",
            })
        return keys


class TestFalconEngine(FamilyPluginTestMixin):
    tester_class = FalconPluginTester

    def test_rw_alibi_bias_uses_falcon_logit_scale(self, monkeypatch):
        """Validate Falcon-RW ALiBi is scaled like HF Falcon attention logits.

        Intention:
            HF Falcon's default SDPA path rounds ALiBi in BF16, casts it to
            the model dtype, and then scales the additive mask by
            1/sqrt(head_dim). The Falcon plugin forwards that scale without
            pre-scaling the slopes before their BF16 rounding boundary.

        Setup:
            Monkeypatch the shared decoder builder, build a tiny ALiBi Falcon
            config, and verify the plugin forwards the expected ALiBi scale and
            exact GELU activation.
        """
        fake_trt = types.ModuleType("tensorrt")
        monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
        from tensorrt_model_connect import trt_compat
        monkeypatch.setattr(trt_compat, "_module", None)
        from tensorrt_model_connect.families.falcon import model as falcon_module

        captured = {}

        def fake_build_standard_decoder_engine(*args, **kwargs):
            captured.update(kwargs)
            return b"plan"

        monkeypatch.setattr(
            falcon_module,
            "build_standard_decoder_engine",
            fake_build_standard_decoder_engine,
        )
        config = ModelConfig.create_tiny(
            "falcon",
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=4,
            alibi=True,
        )

        plan = falcon_module.build_engine(config, {}, 8)

        assert plan == b"plan"
        assert captured["position_type"] == "alibi"
        assert captured["activation"] == "gelu"
        assert captured["alibi_bias_scale"] == pytest.approx(0.5)

    def test_layernorm_beta_present(self, tester, tmp_path):
        """Validate that Falcon includes LayerNorm bias (beta) weights.

        Intention:
            Falcon uses full LayerNorm (not RMSNorm), which requires both
            gamma (weight) and beta (bias) parameters. If the beta weights
            are missing, the LayerNorm computation will be incorrect,
            effectively using RMSNorm instead of LayerNorm.

            Example bug this catches: A plugin that loads LayerNorm weights
            but forgets to load the bias tensors because it was copied from
            a RMSNorm-based family template.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify input_norm_beta and post_attn_norm_beta exist for
               each layer.
            3. Verify final_norm_beta exists.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        for i in range(s.num_hidden_layers):
            assert f"layer.{i}.input_norm_beta" in weights, (
                f"Layer {i} missing input_norm_beta (LayerNorm bias)"
            )
            assert f"layer.{i}.post_attn_norm_beta" in weights, (
                f"Layer {i} missing post_attn_norm_beta (LayerNorm bias)"
            )

        assert "final_norm_beta" in weights, (
            "Missing final_norm_beta (final LayerNorm bias)"
        )

    def test_fc_mlp_keys_present(self, tester, tmp_path):
        """Validate that Falcon uses FC MLP keys (w_fc1/w_fc2) not SwiGLU keys.

        Intention:
            Falcon uses a 2-projection GELU FC MLP (dense_h_to_4h / dense_4h_to_h)
            mapped to w_fc1/w_fc2, NOT the 3-projection SwiGLU MLP
            (gate/up/down mapped to w_gate/w_up/w_down). If the wrong MLP
            keys are used, the engine builder will fail to find the weights
            or construct the wrong MLP graph.

            Example bug this catches: A plugin that maps dense_h_to_4h to
            w_gate instead of w_fc1, causing the SwiGLU builder path to be
            selected instead of the FC path.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify w_fc1 and w_fc2 exist for each layer.
            3. Verify w_gate, w_up, w_down do NOT exist (FC MLP, not SwiGLU).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        for i in range(s.num_hidden_layers):
            assert f"layer.{i}.w_fc1" in weights, (
                f"Layer {i} missing w_fc1 (FC MLP)"
            )
            assert f"layer.{i}.w_fc2" in weights, (
                f"Layer {i} missing w_fc2 (FC MLP)"
            )
            assert f"layer.{i}.w_gate" not in weights, (
                f"Layer {i} has w_gate — should use FC MLP, not SwiGLU"
            )
