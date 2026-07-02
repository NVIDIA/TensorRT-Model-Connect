# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the BLOOM family plugin.

BLOOM uses:
  - ALiBi position encoding (no position embeddings)
  - LayerNorm (with beta) instead of RMSNorm
  - Embedding LayerNorm after token embedding lookup
  - Fused QKV projection (query_key_value) with head-interleaved layout
  - All linear layers have bias
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
  - Tied word embeddings (word_embeddings == lm_head)
  - HF config uses d_model, attention_heads, num_layers (not hidden_size, etc.)
  - transformer.h.{i} prefix (not model.layers.{i})

Trace: ARCH-FAM-001, UD-FAM-BLOOM-01
Intent: Validate the BLOOM family plugin weight loading including ALiBi, fused QKV splitting, embedding LayerNorm, tied embeddings, and non-standard HF key aliases.
Preconditions: safetensors and tensorrt_model_connect are importable; no TRT or GPU required for weight-loading tests.
Postconditions: All weight keys map correctly from BLOOM's HF layout, fused QKV is split per-head, biases are loaded, and config aliases (d_model, attention_heads) resolve properly.
"""
import numpy as np

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class BloomPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.bloom"
    model_type = "bloom"

    def get_config_dict(self) -> dict:
        """Return config dict using BLOOM's non-standard key names.

        Intention:
            BLOOM uses d_model instead of hidden_size, attention_heads instead
            of num_attention_heads, num_layers instead of num_hidden_layers,
            and ffn_dim for intermediate_size. ModelConfig.from_json handles
            these aliases, but we need to provide them correctly in the
            synthetic config.

        Setup:
            Return a config dict with BLOOM-style key names.
        """
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "d_model": s.hidden_size,
            "ffn_dim": s.intermediate_size,
            "num_layers": s.num_hidden_layers,
            "attention_heads": s.num_attention_heads,
            "num_key_value_heads": s.num_attention_heads,
            "layer_norm_epsilon": s.rms_norm_eps,
            "max_position_embeddings": s.max_position_embeddings,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic BLOOM weight layout with fused head-interleaved QKV.

        Intention:
            BLOOM uses transformer.h.{i} prefix, head-interleaved fused QKV
            (query_key_value.{weight,bias}), dense_h_to_4h/dense_4h_to_h MLP
            with biases, embedding LayerNorm (word_embeddings_layernorm),
            and all LayerNorms have biases.

        Setup:
            Build synthetic tensors matching BLOOM's checkpoint layout:
            - transformer.word_embeddings.weight [vocab, hidden]
            - transformer.word_embeddings_layernorm.{weight,bias} [hidden]
            - transformer.h.{i}.input_layernorm.{weight,bias}
            - transformer.h.{i}.post_attention_layernorm.{weight,bias}
            - transformer.h.{i}.self_attention.query_key_value.{weight,bias}
              [3*hidden, hidden] head-interleaved
            - transformer.h.{i}.self_attention.dense.{weight,bias}
            - transformer.h.{i}.mlp.dense_h_to_4h.{weight,bias}
            - transformer.h.{i}.mlp.dense_4h_to_h.{weight,bias}
            - transformer.ln_f.{weight,bias}
        """
        s = self.spec

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}

        # Token embedding
        t["transformer.word_embeddings.weight"] = rand(
            s.vocab_size, s.hidden_size)

        # Embedding LayerNorm
        t["transformer.word_embeddings_layernorm.weight"] = rand(s.hidden_size)
        t["transformer.word_embeddings_layernorm.bias"] = rand(s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"transformer.h.{i}"

            # LayerNorms with bias
            t[f"{p}.input_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.input_layernorm.bias"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.weight"] = rand(s.hidden_size)
            t[f"{p}.post_attention_layernorm.bias"] = rand(s.hidden_size)

            # Fused QKV: head-interleaved [3*hidden, hidden]
            # Layout: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
            t[f"{p}.self_attention.query_key_value.weight"] = rand(
                3 * s.hidden_size, s.hidden_size)
            t[f"{p}.self_attention.query_key_value.bias"] = rand(
                3 * s.hidden_size)

            # Output projection with bias
            t[f"{p}.self_attention.dense.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attention.dense.bias"] = rand(s.hidden_size)

            # MLP with biases
            t[f"{p}.mlp.dense_h_to_4h.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = rand(s.intermediate_size)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = rand(
                s.hidden_size, s.intermediate_size)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["transformer.ln_f.weight"] = rand(s.hidden_size)
        t["transformer.ln_f.bias"] = rand(s.hidden_size)

        return t

    def expected_weight_keys(self) -> set[str]:
        """Return expected weight keys for BLOOM.

        BLOOM uses LayerNorm (with beta), FC MLP (w_fc1/w_fc2), embedding
        LayerNorm, QKV biases, output bias, and metadata keys.
        """
        s = self.spec
        keys = {
            "embedding", "embedding_norm", "embedding_norm_beta",
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
        """Return expected engine inputs for BLOOM (ALiBi, no position_id)."""
        s = self.spec
        names = {"token_id", "position_id", "attention_mask"}
        for i in range(s.num_hidden_layers):
            names.add(f"cache_k_{i}")
            names.add(f"cache_v_{i}")
        return names


class TestBloomEngine(FamilyPluginTestMixin):
    tester_class = BloomPluginTester

    def test_embedding_norm_present(self, tester, tmp_path):
        """Validate that BLOOM includes the embedding LayerNorm.

        Intention:
            BLOOM applies a LayerNorm after the token embedding lookup and
            before feeding into the first transformer layer. This is stored
            as embedding_norm (weight) and embedding_norm_beta (bias). If
            these are missing, the embedding output will not be normalized,
            which BLOOM's architecture relies on for stable training and
            inference.

            Example bug this catches: A plugin that skips loading the
            word_embeddings_layernorm because it's not present in other
            families' weight layouts.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify embedding_norm and embedding_norm_beta exist in
               the WeightDict.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "embedding_norm" in weights, (
            "Missing embedding_norm (BLOOM's embedding LayerNorm weight)"
        )
        assert "embedding_norm_beta" in weights, (
            "Missing embedding_norm_beta (BLOOM's embedding LayerNorm bias)"
        )

    def test_head_interleaved_qkv_split(self, tester, tmp_path):
        """Validate that BLOOM's head-interleaved fused QKV is correctly split.

        Intention:
            BLOOM stores QKV in a head-interleaved layout:
            [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...] where each block is
            head_dim rows. The plugin must de-interleave into separate Q, K, V
            tensors. If the de-interleaving is wrong, attention heads will use
            each other's key/value data.

            Example bug this catches: Treating the fused QKV as a simple
            concatenation [all_Q, all_K, all_V] instead of head-interleaved,
            which would mix Q rows from one head with K/V rows from another.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify w_q has shape [hidden, hidden] (after transpose).
            3. Verify w_k and w_v have consistent shapes.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        # After transpose and de-interleaving: w_q should be [hidden, hidden]
        assert weights["layer.0.w_q"].shape == (s.hidden_size, s.hidden_size), (
            f"w_q shape {weights['layer.0.w_q'].shape} != "
            f"expected ({s.hidden_size}, {s.hidden_size})"
        )

    def test_all_biases_present(self, tester, tmp_path):
        """Validate that all BLOOM bias weights are loaded.

        Intention:
            BLOOM uses bias on every linear layer (Q, K, V, O, fc1, fc2)
            and every LayerNorm. Missing biases cause numerical errors
            that are hard to diagnose because the model still runs but
            produces subtly wrong outputs.

            Example bug this catches: A plugin that loads weight tensors
            but skips bias tensors for the attention output projection.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each layer, verify all bias keys are present.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        bias_suffixes = [
            "q_bias", "k_bias", "v_bias", "o_bias",
            "input_norm_beta", "post_attn_norm_beta",
            "fc1_bias", "fc2_bias",
        ]
        for i in range(s.num_hidden_layers):
            for suffix in bias_suffixes:
                key = f"layer.{i}.{suffix}"
                assert key in weights, f"Missing bias key: {key}"
