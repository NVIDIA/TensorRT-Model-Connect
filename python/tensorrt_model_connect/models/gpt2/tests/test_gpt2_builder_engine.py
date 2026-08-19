# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the GPT-2 family plugin.

GPT-2 uses:
  - Learned absolute position embeddings (wpe)
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (c_fc/c_proj) with GELU activation
  - Fused QKV via a single c_attn Conv1D weight
  - Conv1D weights stored as [in, out] (NOT [out, in] like Linear)
  - Tied word embeddings (wte == lm_head)
  - HF config uses n_embd, n_head, n_layer, n_inner
  - h.{i} prefix (not model.layers.{i})

Trace: ARCH-FAM-001, UD-FAM-GPT2-01
Intent: Validate the GPT-2 family plugin weight loading including Conv1D transpose, fused QKV splitting, learned positions, tied embeddings, and non-standard HF config aliases.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Conv1D weights are transposed to [out, in], fused QKV is split correctly, position embeddings are loaded, and config aliases (n_embd, n_head) resolve properly.
"""
import json

import numpy as np

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class GPT2PluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.models.gpt2.model"
    model_type = "gpt2"

    def get_config_dict(self) -> dict:
        """Return config dict using GPT-2's non-standard key names.

        Intention:
            GPT-2 uses n_embd instead of hidden_size, n_head instead of
            num_attention_heads, n_layer instead of num_hidden_layers, and
            n_inner for intermediate_size. ModelConfig.from_json handles
            these aliases via fallback chains.

        Setup:
            Return a config dict with GPT-2-style key names.
        """
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "n_embd": s.hidden_size,
            "n_head": s.num_attention_heads,
            "n_layer": s.num_hidden_layers,
            "n_inner": s.intermediate_size,
            "layer_norm_epsilon": s.rms_norm_eps,
            "n_positions": s.max_position_embeddings,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic GPT-2 weight layout with Conv1D weights.

        Intention:
            GPT-2 uses Conv1D for all projections, which stores weights as
            [in_features, out_features] (opposite of Linear [out, in]).
            The c_attn weight is fused QKV [hidden, 3*hidden]. All layers
            use h.{i} prefix and ln_1/ln_2 for LayerNorm naming.

        Setup:
            Build synthetic tensors matching GPT-2's checkpoint layout:
            - wte.weight [vocab, hidden]
            - wpe.weight [max_pos, hidden]
            - h.{i}.ln_1.{weight,bias} [hidden]
            - h.{i}.ln_2.{weight,bias} [hidden]
            - h.{i}.attn.c_attn.{weight,bias} Conv1D [hidden, 3*hidden]
            - h.{i}.attn.c_proj.{weight,bias} Conv1D [hidden, hidden]
            - h.{i}.mlp.c_fc.{weight,bias} Conv1D [hidden, intermediate]
            - h.{i}.mlp.c_proj.{weight,bias} Conv1D [intermediate, hidden]
            - ln_f.{weight,bias} [hidden]
        """
        s = self.spec

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}

        # Token embedding
        t["wte.weight"] = rand(s.vocab_size, s.hidden_size)

        # Position embedding
        t["wpe.weight"] = rand(s.max_position_embeddings, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"h.{i}"

            # LayerNorm with bias
            t[f"{p}.ln_1.weight"] = rand(s.hidden_size)
            t[f"{p}.ln_1.bias"] = rand(s.hidden_size)
            t[f"{p}.ln_2.weight"] = rand(s.hidden_size)
            t[f"{p}.ln_2.bias"] = rand(s.hidden_size)

            # Fused QKV: Conv1D [hidden, 3*hidden]
            t[f"{p}.attn.c_attn.weight"] = rand(
                s.hidden_size, 3 * s.hidden_size)
            t[f"{p}.attn.c_attn.bias"] = rand(3 * s.hidden_size)

            # Output projection: Conv1D [hidden, hidden]
            t[f"{p}.attn.c_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.attn.c_proj.bias"] = rand(s.hidden_size)

            # MLP: Conv1D [hidden, intermediate] and [intermediate, hidden]
            t[f"{p}.mlp.c_fc.weight"] = rand(
                s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.c_fc.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.c_proj.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.c_proj.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["ln_f.weight"] = rand(s.hidden_size)
        t["ln_f.bias"] = rand(s.hidden_size)

        return t

    def expected_weight_keys(self) -> set[str]:
        """Return expected weight keys for GPT-2.

        GPT-2 uses position_embedding, LayerNorm (with beta), FC MLP
        (w_fc1/w_fc2), QKV biases, output bias, and metadata keys.
        """
        s = self.spec
        keys = {
            "embedding", "position_embedding",
            "final_norm", "final_norm_beta", "w_out",
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
                f"{prefix}.w_fc1",
                f"{prefix}.w_fc2",
                f"{prefix}.fc1_bias",
                f"{prefix}.fc2_bias",
            })
        return keys

    def expected_engine_input_names(self) -> set[str]:
        """Return expected engine inputs for GPT-2 (includes position_id for learned positions)."""
        s = self.spec
        names = {"token_id", "position_id", "attention_mask"}
        for i in range(s.num_hidden_layers):
            names.add(f"cache_k_{i}")
            names.add(f"cache_v_{i}")
        return names


class TestGPT2Engine(FamilyPluginTestMixin):
    tester_class = GPT2PluginTester

    def test_conv1d_weights_not_transposed(self, tester, tmp_path):
        """Validate that Conv1D weights are used directly without transposing.

        Intention:
            GPT-2 uses Conv1D instead of Linear for all projections. Conv1D
            stores weights as [in_features, out_features], which is already
            the layout needed by the engine builder (matmul right-hand operand).
            If the plugin accidentally transposes Conv1D weights (treating them
            like Linear [out, in]), the dimensions will be swapped and the
            matmul will produce wrong results.

            Example bug this catches: Applying _transpose_2d to Conv1D weights,
            which would flip them from the correct [in, out] to wrong [out, in].

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify w_q shape is [hidden, hidden] (Conv1D [in, out] used
               directly as [in, out]).
            3. Verify the values match the raw Conv1D weight slice (no transpose).
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        # w_q should be [hidden, hidden] from Conv1D [hidden, 3*hidden][:, :hidden]
        w_q = weights["layer.0.w_q"]
        assert w_q.shape == (s.hidden_size, s.hidden_size), (
            f"w_q shape {w_q.shape} != expected ({s.hidden_size}, {s.hidden_size})"
        )

        # Verify values match the raw c_attn slice (not transposed)
        raw_c_attn = raw["h.0.attn.c_attn.weight"]
        expected_q = raw_c_attn[:, :s.hidden_size].astype(np.float32)
        np.testing.assert_allclose(
            w_q, expected_q, atol=1e-6,
            err_msg="w_q values do not match raw Conv1D slice "
                    "(Conv1D should NOT be transposed)",
        )

    def test_position_embedding_loaded(self, tester, tmp_path):
        """Validate that GPT-2's learned position embedding (wpe) is loaded.

        Intention:
            GPT-2 uses learned absolute position embeddings (wpe.weight) with
            shape [max_positions, hidden]. Unlike RoPE-based models, these are
            added to the token embeddings during forward pass. If the position
            embedding is missing, the model will have no positional information.

            Example bug this catches: A plugin that doesn't load wpe.weight
            because the standard decoder template uses RoPE instead of learned
            positions.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify position_embedding exists and has shape
               [max_positions, hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        assert "position_embedding" in weights, (
            "Missing position_embedding (GPT-2's wpe)"
        )
        pos = weights["position_embedding"]
        assert pos.shape == (s.max_position_embeddings, s.hidden_size), (
            f"Position embedding shape {pos.shape} != "
            f"expected ({s.max_position_embeddings}, {s.hidden_size})"
        )

    def test_fused_qkv_bias_split(self, tester, tmp_path):
        """Validate that GPT-2's fused c_attn bias is correctly split into Q/K/V biases.

        Intention:
            GPT-2's c_attn stores a fused bias [3*hidden] that must be split
            into separate Q, K, V biases of size [hidden] each. If the split
            offsets are wrong, the attention computation will use wrong bias
            values for some heads.

            Example bug this catches: Splitting at [hidden, hidden, hidden]
            with wrong offsets, e.g., starting K bias at 0 instead of hidden.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify q_bias, k_bias, v_bias each have shape [hidden].
            3. Verify their values match the corresponding slices of the
               raw c_attn.bias.
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        raw_bias = raw["h.0.attn.c_attn.bias"]
        expected_q_bias = raw_bias[:s.hidden_size].astype(np.float32)
        expected_k_bias = raw_bias[s.hidden_size:2*s.hidden_size].astype(np.float32)
        expected_v_bias = raw_bias[2*s.hidden_size:].astype(np.float32)

        np.testing.assert_allclose(
            weights["layer.0.q_bias"], expected_q_bias, atol=1e-6,
            err_msg="q_bias does not match raw c_attn.bias[:hidden]",
        )
        np.testing.assert_allclose(
            weights["layer.0.k_bias"], expected_k_bias, atol=1e-6,
            err_msg="k_bias does not match raw c_attn.bias[hidden:2*hidden]",
        )
        np.testing.assert_allclose(
            weights["layer.0.v_bias"], expected_v_bias, atol=1e-6,
            err_msg="v_bias does not match raw c_attn.bias[2*hidden:]",
        )

    def test_transformer_prefixed_weights_load(self, tester, tmp_path):
        """Validate DistilGPT2-style transformer.* checkpoint keys.

        DistilGPT2 is a GPT2LMHeadModel checkpoint whose safetensors use
        transformer.wte.weight, transformer.h.* and transformer.ln_f.*. The
        plugin should handle that prefix without changing the shared loader.
        """
        from safetensors.numpy import save_file
        from tensorrt_model_connect.config import ModelConfig

        raw = tester.make_hf_tensors()
        prefixed = {f"transformer.{key}": value for key, value in raw.items()}
        tmp_path.joinpath("config.json").write_text(
            json.dumps(tester.get_config_dict()),
            encoding="utf-8",
        )
        save_file(prefixed, str(tmp_path / "model.safetensors"))

        config = ModelConfig.from_dir(tmp_path)
        weights = tester.get_plugin().load_weights(str(tmp_path), config)

        np.testing.assert_allclose(
            weights["embedding"],
            raw["wte.weight"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            weights["position_embedding"],
            raw["wpe.weight"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            weights["layer.0.w_q"],
            raw["h.0.attn.c_attn.weight"][:, :tester.spec.hidden_size],
            atol=1e-6,
        )
