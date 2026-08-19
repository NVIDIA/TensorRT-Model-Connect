# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for Phi-MoE (Mixture of Experts with SparseMixer).

Intention:
    Validate the Phi-MoE family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    Phi-MoE uses the standard decoder attention (RoPE + GQA) but replaces the
    SwiGLU MLP with a router + N expert MLPs. The router uses SparseMixer
    (not standard top-k softmax) to select top-2 experts per token. Each
    expert's weight is computed from an independent masked softmax over all
    logits, so the weights do NOT sum to 1.0.

    Key differences from standard Phi-3:
      - LayerNorm (with bias) instead of RMSNorm
      - Separate Q/K/V/O projections (not fused) with biases
      - MoE block: router + N experts, each a SwiGLU MLP
      - lm_head has bias

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    get_config_dict() (to add num_local_experts, num_experts_per_tok, and
    LayerNorm biases), make_hf_tensors() (for MoE weight layout with router +
    per-expert MLPs + biases on everything), and expected_weight_keys() (for
    router + expert.{e}.w_gate/up/down + bias keys + lm_head_bias).
    Uses a tiny model with 2 experts to keep engine build fast.
    Tier 2 is skipped because Phi-MoE uses a custom MoE decoder builder with
    SparseMixer routing rather than the standard single-engine builder.

Trace: ARCH-FAM-001, UD-FAM-PHIMOE-01
Intent: Validate the Phi-MoE family plugin weight loading including SparseMixer router, per-expert SwiGLU MLP mapping, LayerNorm biases, QKV biases, and lm_head bias.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Router and per-expert weight keys are present with biases, LayerNorm biases are loaded, and MoE-specific config fields are parsed correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip("safetensors.numpy", reason="safetensors not available")
pytest.importorskip("tensorrt_model_connect.config", reason="tensorrt_model_connect requires tensorrt")

from tensorrt_model_connect.models.phi_moe.tests._family_plugin_tester import (
    FamilyPluginTester,
)
from tensorrt_model_connect.models.phi_moe.tests._family_plugin_test_mixin import (
    FamilyPluginTestMixin,
)


# Number of experts kept tiny for fast engine builds.
_NUM_EXPERTS = 2
_NUM_EXPERTS_PER_TOK = 2


class PhiMoEPluginTester(FamilyPluginTester):
    """Tester for the Phi-MoE family plugin.

    Phi-MoE uses:
      - LayerNorm (with bias) instead of RMSNorm
      - Separate Q/K/V/O projections with biases
      - GQA (num_key_value_heads may differ from num_attention_heads)
      - RoPE positional encoding
      - MoE MLP: router [num_experts, hidden] + per-expert SwiGLU
        (w1/w3/w2 -> gate/up/down)
      - SparseMixer routing (top-2) with independent softmax per expert
      - lm_head weight + bias
    """

    plugin_module = "tensorrt_model_connect.models.phi_moe.model"
    model_type = "phimoe"

    def get_config_dict(self) -> dict:
        """Phi-MoE config with MoE-specific fields and LayerNorm biases."""
        d = super().get_config_dict()
        d["num_local_experts"] = _NUM_EXPERTS
        d["num_experts_per_tok"] = _NUM_EXPERTS_PER_TOK
        d["router_jitter_noise"] = 0.01
        return d

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Phi-MoE's weight layout.

        Key differences from standard decoder:
          - LayerNorm has bias: input_layernorm.bias, post_attention_layernorm.bias
          - Q/K/V/O projections have biases
          - No gate_proj/up_proj/down_proj at layer level
          - Instead: block_sparse_moe.gate.weight [num_experts, hidden]
          - Per-expert: block_sparse_moe.experts.{e}.w1.weight [inter, hidden] (gate)
          - Per-expert: block_sparse_moe.experts.{e}.w3.weight [inter, hidden] (up)
          - Per-expert: block_sparse_moe.experts.{e}.w2.weight [hidden, inter] (down)
          - model.norm has bias
          - lm_head has bias
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
            # Q/K/V/O projections with biases
            t[f"{p}.self_attn.q_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.q_proj.bias"] = rand(s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.k_proj.bias"] = rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.bias"] = rand(kv_hidden)
            t[f"{p}.self_attn.o_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.o_proj.bias"] = rand(s.hidden_size)
            # Router
            t[f"{p}.block_sparse_moe.gate.weight"] = rand(
                _NUM_EXPERTS, s.hidden_size)
            # Per-expert SwiGLU
            for e in range(_NUM_EXPERTS):
                ep = f"{p}.block_sparse_moe.experts.{e}"
                t[f"{ep}.w1.weight"] = rand(
                    s.intermediate_size, s.hidden_size)
                t[f"{ep}.w3.weight"] = rand(
                    s.intermediate_size, s.hidden_size)
                t[f"{ep}.w2.weight"] = rand(
                    s.hidden_size, s.intermediate_size)

        # Final norm with bias
        t["model.norm.weight"] = rand(s.hidden_size)
        t["model.norm.bias"] = rand(s.hidden_size)
        # LM head with bias
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        t["lm_head.bias"] = rand(s.vocab_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Phi-MoE weight keys: attention + biases + router + per-expert SwiGLU.

        Per-layer keys:
          w_q, w_k, w_v, w_o (transposed projections)
          q_bias, k_bias, v_bias, o_bias (attention biases)
          input_norm, input_norm_beta (LayerNorm with bias)
          post_attn_norm, post_attn_norm_beta
          router (transposed [hidden, num_experts])
          expert.{e}.w_gate, expert.{e}.w_up, expert.{e}.w_down

        Global keys:
          embedding, final_norm, final_norm_beta, w_out, lm_head_bias
        """
        s = self.spec
        keys = {
            "embedding", "final_norm", "final_norm_beta",
            "w_out", "lm_head_bias",
        }
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.w_q",
                f"{prefix}.w_k",
                f"{prefix}.w_v",
                f"{prefix}.w_o",
                f"{prefix}.q_bias",
                f"{prefix}.k_bias",
                f"{prefix}.v_bias",
                f"{prefix}.o_bias",
                f"{prefix}.input_norm",
                f"{prefix}.input_norm_beta",
                f"{prefix}.post_attn_norm",
                f"{prefix}.post_attn_norm_beta",
                f"{prefix}.router",
            })
            for e in range(_NUM_EXPERTS):
                keys.update({
                    f"{prefix}.expert.{e}.w_gate",
                    f"{prefix}.expert.{e}.w_up",
                    f"{prefix}.expert.{e}.w_down",
                })
        return keys


class TestPhiMoEEngine(FamilyPluginTestMixin):
    """Engine tests for Phi-MoE family plugin.

    Tier 0 and Tier 1 tests run via the mixin. Tier 2 (engine build) is
    skipped because Phi-MoE uses a custom MoE decoder builder with
    SparseMixer routing rather than the standard single-engine builder.
    """

    tester_class = PhiMoEPluginTester

    # --- Tier 2 skips ---
    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_build_engine_succeeds(self, tester, tmp_path):
        pass

    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_engine_io_tensor_names(self, tester, tmp_path):
        pass

    @pytest.mark.skip(
        reason="custom builder -- uses non-standard graph construction"
    )
    def test_engine_logits_output_shape(self, tester, tmp_path):
        pass

    # --- Phi-MoE-specific Tier 1 tests ---

    def test_layernorm_biases_present(self, tester, tmp_path):
        """Validate that LayerNorm biases are loaded for all layers.

        Intention:
            Phi-MoE uses LayerNorm (not RMSNorm), which has both weight and
            bias (beta) parameters. Standard RMSNorm-based families skip
            the bias. If the bias is missing, the LayerNorm output will be
            shifted incorrectly.

            Example bug this catches: A plugin that copies the RMSNorm loading
            pattern and skips the bias tensor.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each layer, verify input_norm_beta and post_attn_norm_beta.
            3. Verify final_norm_beta exists.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        for i in range(s.num_hidden_layers):
            pfx = f"layer.{i}"
            assert f"{pfx}.input_norm_beta" in weights, (
                f"Missing {pfx}.input_norm_beta (LayerNorm bias)"
            )
            assert f"{pfx}.post_attn_norm_beta" in weights, (
                f"Missing {pfx}.post_attn_norm_beta (LayerNorm bias)"
            )
        assert "final_norm_beta" in weights, (
            "Missing final_norm_beta (final LayerNorm bias)"
        )

    def test_attention_biases_present(self, tester, tmp_path):
        """Validate that Q/K/V/O biases are loaded for all layers.

        Intention:
            Phi-MoE has biases on all attention projections (Q, K, V, O).
            Missing biases cause numerical errors that are hard to diagnose.

            Example bug this catches: A plugin that loads attention weights
            but skips the bias tensors because most other families omit them.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each layer, verify q_bias, k_bias, v_bias, o_bias.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        for i in range(s.num_hidden_layers):
            pfx = f"layer.{i}"
            for proj in ("q_bias", "k_bias", "v_bias", "o_bias"):
                key = f"{pfx}.{proj}"
                assert key in weights, f"Missing attention bias: {key}"

    def test_router_shape_transposed(self, tester, tmp_path):
        """Validate that the router weight is transposed to [hidden, num_experts].

        Intention:
            HF stores the router as [num_experts, hidden], but the plugin
            transposes it to [hidden, num_experts] for matmul as the
            right-hand operand. If not transposed, the router will produce
            wrong expert selection logits.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify layer.0.router shape is [hidden, num_experts].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        router = weights["layer.0.router"]
        expected = (s.hidden_size, _NUM_EXPERTS)
        assert router.shape == expected, (
            f"Router shape {router.shape} != expected {expected} "
            f"(should be transposed from HF [num_experts, hidden])"
        )

    def test_per_expert_weights_shapes(self, tester, tmp_path):
        """Validate per-expert SwiGLU weight shapes.

        Intention:
            Each expert has gate, up, and down projections. After transpose:
              - w_gate: [hidden, intermediate] (from HF [intermediate, hidden])
              - w_up: [hidden, intermediate]
              - w_down: [intermediate, hidden] (from HF [hidden, intermediate])

            If any shape is wrong, the SwiGLU computation produces wrong outputs
            or crashes with a TRT shape mismatch.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each expert in layer 0, verify gate/up/down shapes.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        for e in range(_NUM_EXPERTS):
            pfx = f"layer.0.expert.{e}"
            w_gate = weights[f"{pfx}.w_gate"]
            w_up = weights[f"{pfx}.w_up"]
            w_down = weights[f"{pfx}.w_down"]
            # After transpose: [in, out]
            assert w_gate.shape == (s.hidden_size, s.intermediate_size), (
                f"Expert {e} w_gate shape {w_gate.shape} != "
                f"expected ({s.hidden_size}, {s.intermediate_size})"
            )
            assert w_up.shape == (s.hidden_size, s.intermediate_size), (
                f"Expert {e} w_up shape {w_up.shape} != "
                f"expected ({s.hidden_size}, {s.intermediate_size})"
            )
            assert w_down.shape == (s.intermediate_size, s.hidden_size), (
                f"Expert {e} w_down shape {w_down.shape} != "
                f"expected ({s.intermediate_size}, {s.hidden_size})"
            )

    def test_lm_head_bias_present(self, tester, tmp_path):
        """Validate that the LM head bias is loaded.

        Intention:
            Phi-MoE has a bias on the LM head (lm_head.bias). Most families
            do NOT have this, so it is easily overlooked. If missing, the
            logits will be shifted and token probabilities will be wrong.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify lm_head_bias exists and has shape [vocab_size].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        assert "lm_head_bias" in weights, (
            "Missing lm_head_bias (Phi-MoE has bias on lm_head)"
        )
        assert weights["lm_head_bias"].shape == (s.vocab_size,), (
            f"lm_head_bias shape {weights['lm_head_bias'].shape} != "
            f"expected ({s.vocab_size},)"
        )

    def test_gqa_kv_stays_compact(self, tester, tmp_path):
        """Validate that K/V projections stay at compact KV width.

        Intention:
            Phi-MoE uses GQA where num_key_value_heads may be less than
            num_attention_heads. The plugin must keep K/V projections compact
            and let TRT native attention consume num_kv_heads directly.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify w_k and w_v have shape [hidden, kv_dim].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        kv_dim = s.num_key_value_heads * s.head_dim
        w_k = weights["layer.0.w_k"]
        w_v = weights["layer.0.w_v"]
        assert w_k.shape == (s.hidden_size, kv_dim), (
            f"w_k shape {w_k.shape} != expected "
            f"({s.hidden_size}, {kv_dim})"
        )
        assert w_v.shape == (s.hidden_size, kv_dim), (
            f"w_v shape {w_v.shape} != expected "
            f"({s.hidden_size}, {kv_dim})"
        )
