# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the OPT family plugin.

OPT uses a different HF weight layout from standard decoders:
  - model.decoder.embed_tokens.weight (not model.embed_tokens.weight)
  - model.decoder.embed_positions.weight (learned positions with offset=2)
  - model.decoder.layers.{i}.self_attn_layer_norm (not input_layernorm)
  - model.decoder.layers.{i}.self_attn.{q,k,v}_proj + out_proj (with biases)
  - model.decoder.layers.{i}.fc1/fc2 (not mlp.gate/up/down)
  - model.decoder.final_layer_norm
  - LayerNorm everywhere (with beta/bias)
  - ReLU activation in MLP (2-proj fc1/fc2, not SwiGLU gate/up/down)

Trace: ARCH-FAM-001, UD-FAM-OPT-01
Intent: Validate the OPT family plugin weight loading including model.decoder.* prefix mapping, learned positions with offset, fc1/fc2 MLP, and biases on all linear layers and LayerNorms.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All weight keys map correctly from OPT's non-standard HF layout, position embeddings include the 2-position offset, and biases are loaded for all projections and norms.
"""
import numpy as np

from tensorrt_model_connect.models.opt.tests._family_plugin_tester import (
    FamilyPluginTester,
)
from tensorrt_model_connect.models.opt.tests._family_plugin_test_mixin import (
    FamilyPluginTestMixin,
)


class OPTPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.models.opt.model"
    model_type = "opt"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic OPT weight layout.

        Intention:
            OPT uses model.decoder.* prefix, self_attn_layer_norm/final_layer_norm
            naming, fc1/fc2 for MLP, out_proj for attention output, learned
            position embeddings with a 2-position offset, and biases on all
            linear layers and LayerNorms.

        Setup:
            Build synthetic tensors matching OPT's checkpoint layout:
            - model.decoder.embed_tokens.weight [vocab, hidden]
            - model.decoder.embed_positions.weight [max_pos+2, hidden]
            - model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
            - model.decoder.layers.{i}.final_layer_norm.{weight,bias}
            - model.decoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
            - model.decoder.layers.{i}.self_attn.out_proj.{weight,bias}
            - model.decoder.layers.{i}.fc1.{weight,bias}
            - model.decoder.layers.{i}.fc2.{weight,bias}
            - model.decoder.final_layer_norm.{weight,bias}
            - lm_head.weight [vocab, hidden]
        """
        s = self.spec

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.decoder.embed_tokens.weight"] = rand(
            s.vocab_size, s.hidden_size)

        # Position embedding with offset=2: [max_pos+2, hidden]
        t["model.decoder.embed_positions.weight"] = rand(
            s.max_position_embeddings + 2, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.decoder.layers.{i}"

            # LayerNorm 1 (self_attn_layer_norm)
            t[f"{p}.self_attn_layer_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.self_attn_layer_norm.bias"] = rand(s.hidden_size)

            # LayerNorm 2 (final_layer_norm)
            t[f"{p}.final_layer_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.final_layer_norm.bias"] = rand(s.hidden_size)

            # Q/K/V/O projections with biases
            kv_hidden = s.num_key_value_heads * s.head_dim
            t[f"{p}.self_attn.q_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.q_proj.bias"] = rand(s.hidden_size)
            t[f"{p}.self_attn.k_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.k_proj.bias"] = rand(kv_hidden)
            t[f"{p}.self_attn.v_proj.weight"] = rand(
                kv_hidden, s.hidden_size)
            t[f"{p}.self_attn.v_proj.bias"] = rand(kv_hidden)
            t[f"{p}.self_attn.out_proj.weight"] = rand(
                s.hidden_size, s.hidden_size)
            t[f"{p}.self_attn.out_proj.bias"] = rand(s.hidden_size)

            # MLP: fc1 [intermediate, hidden], fc2 [hidden, intermediate]
            t[f"{p}.fc1.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.fc1.bias"] = rand(s.intermediate_size)
            t[f"{p}.fc2.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.fc2.bias"] = rand(s.hidden_size)

        # Final LayerNorm
        t["model.decoder.final_layer_norm.weight"] = rand(s.hidden_size)
        t["model.decoder.final_layer_norm.bias"] = rand(s.hidden_size)

        # LM head
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Return expected weight keys for OPT.

        OPT uses LayerNorm (with beta), fc1/fc2 MLP, position embeddings,
        QKV biases, output bias, and metadata keys for attention/MLP sizes.
        """
        s = self.spec
        keys = {
            "embedding", "position_embedding", "final_norm", "final_norm_beta",
            "w_out",
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
        """Return expected engine inputs for OPT (includes position_id for learned positions)."""
        s = self.spec
        names = {"token_id", "position_id", "attention_mask"}
        for i in range(s.num_hidden_layers):
            names.add(f"cache_k_{i}")
            names.add(f"cache_v_{i}")
        return names


class TestOPTEngine(FamilyPluginTestMixin):
    tester_class = OPTPluginTester

    def test_position_embedding_offset_absorbed(self, tester, tmp_path):
        """Validate that OPT's position offset=2 is absorbed by slicing.

        Intention:
            OPT adds an offset of 2 to position IDs, meaning the raw position
            embedding table has 2 extra rows at the start that correspond to
            padding positions. The plugin absorbs this by slicing the table
            from index 2 onwards. If the slice is missing, position 0 at
            inference time will use the padding embedding, causing wrong
            positional information for every token.

            Example bug this catches: Loading the raw position embedding
            without slicing, so the first real position uses padding data.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify position_embedding shape is [max_pos, hidden] (not
               max_pos+2).
            3. Verify position_embedding values match raw[2:].
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec

        pos = weights["position_embedding"]
        raw_pos = raw["model.decoder.embed_positions.weight"]

        # Shape should be max_pos (without the 2 offset rows)
        assert pos.shape[0] == s.max_position_embeddings, (
            f"Position embedding shape {pos.shape[0]} != "
            f"expected {s.max_position_embeddings} (offset not absorbed)"
        )
        np.testing.assert_allclose(
            pos[:10],
            raw_pos[2:12].astype(np.float32),
            atol=1e-6,
            err_msg="Position embedding values do not match raw[2:]",
        )
