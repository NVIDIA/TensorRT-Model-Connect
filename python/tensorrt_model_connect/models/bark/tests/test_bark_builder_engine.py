# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for Bark (text-to-audio).

Intention:
    Validate the Bark family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness.

    Bark is a multi-stage text-to-audio model with three sub-models:
      1. Semantic: text tokens -> semantic tokens (GPT decoder with learned positions)
      2. Coarse: semantic tokens -> coarse audio codes (GPT decoder)
      3. Fine: coarse codes -> fine audio codes (iterative refinement, no KV cache)

    Each sub-model's weights are loaded, mapped, and prefixed (semantic.*,
    coarse.*, fine.*) into a single WeightDict.

    HF Bark uses fused QKV via att_proj.weight [3H, H], which the plugin
    splits into separate Q/K/V. Bark uses LayerNorm (weight + optional bias),
    learned positional embeddings, and GELU MLP (in_proj / out_proj).

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    get_config_dict() (to add bark-specific sub-model config), make_hf_tensors()
    (for bark's multi-prefix weight layout with fused QKV), and
    expected_weight_keys() (for semantic.*/coarse.*/fine.* keys).
    Tier 2 is skipped because bark uses a custom multi-engine builder.

Trace: ARCH-FAM-001, UD-FAM-BARK-01
Intent: Validate the Bark family plugin weight loading, fused QKV splitting, and multi-prefix key mapping for semantic/coarse/fine sub-models.
Preconditions: safetensors and tensorrt_model_connect are importable; no TRT or GPU required for weight-loading tests.
Postconditions: All weight keys are present with correct shapes, fused QKV is split into separate Q/K/V, and multi-prefix layout matches expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip("safetensors.numpy", reason="safetensors not available")
pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig
from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


# Bark sub-model dimensions kept tiny for fast tests.
_SEM_LAYERS = 2
_SEM_HIDDEN = 16
_SEM_HEADS = 4
_SEM_MAX_POS = 32
_SEM_VOCAB = 32
_SEM_OUTPUT_VOCAB = 24

_COARSE_LAYERS = 2
_COARSE_HIDDEN = 16
_COARSE_HEADS = 4
_COARSE_MAX_POS = 32
_COARSE_VOCAB = 32
_COARSE_OUTPUT_VOCAB = 24

_FINE_LAYERS = 2
_FINE_HIDDEN = 16
_FINE_HEADS = 4
_FINE_MAX_POS = 32
_FINE_N_EMBED_TABLES = 8
_FINE_N_LM_HEADS = 7
_FINE_CODEBOOK_SIZE = 16


def _make_bark_tp_weights(
    *,
    hidden: int = _SEM_HIDDEN,
    num_layers: int = _SEM_LAYERS,
    mlp_size: int | None = None,
) -> WeightDict:
    rng = np.random.RandomState(123)
    mlp_size = mlp_size or hidden * 4

    def rand(*shape: int) -> np.ndarray:
        return rng.randn(*shape).astype(np.float32)

    weights = WeightDict(
        {
            "_attention_size": hidden,
            "metadata": "kept",
            "embedding": rand(_SEM_VOCAB, hidden),
            "position_embedding": rand(_SEM_MAX_POS, hidden),
            "final_norm": rand(hidden),
            "final_norm_beta": rand(hidden),
            "w_out": rand(hidden, _SEM_OUTPUT_VOCAB),
            "extra_tensor": rand(3, 3),
        }
    )
    for i in range(num_layers):
        lp = f"layer.{i}"
        for key in ("w_q", "w_k", "w_v"):
            weights[f"{lp}.{key}"] = rand(hidden, hidden)
        for key in ("q_bias", "k_bias", "v_bias"):
            weights[f"{lp}.{key}"] = rand(hidden)
        weights[f"{lp}.w_o"] = rand(hidden, hidden)
        weights[f"{lp}.o_bias"] = rand(hidden)
        weights[f"{lp}.w_fc1"] = rand(hidden, mlp_size)
        weights[f"{lp}.fc1_bias"] = rand(mlp_size)
        weights[f"{lp}.w_fc2"] = rand(mlp_size, hidden)
        weights[f"{lp}.fc2_bias"] = rand(hidden)
        for key in (
            "input_norm",
            "input_norm_beta",
            "post_attn_norm",
            "post_attn_norm_beta",
        ):
            weights[f"{lp}.{key}"] = rand(hidden)
    return weights


def _bark_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.models.bark.decoder_tp_builder",
        reason="TensorRT is required for Bark TP builder tests",
    )


class BarkPluginTester(FamilyPluginTester):
    """Tester for the Bark family plugin.

    Bark uses:
      - Three sub-models: semantic, coarse, fine
      - Fused QKV via att_proj.weight [3H, H]
      - LayerNorm (weight + optional bias, bark-small has no bias)
      - Learned positional embeddings (position_embeds_layer.weight)
      - GELU MLP (in_proj / out_proj, not SwiGLU)
      - Fine model has multiple embedding tables and LM heads
      - HF prefix: semantic.*, coarse_acoustics.*, fine_acoustics.*
    """

    plugin_module = "tensorrt_model_connect.models.bark.model"
    model_type = "bark"
    spec = TinyModelSpec(
        vocab_size=_SEM_VOCAB,
        hidden_size=_SEM_HIDDEN,
        num_hidden_layers=_SEM_LAYERS,
        num_attention_heads=_SEM_HEADS,
        num_key_value_heads=_SEM_HEADS,
    )

    def get_config_dict(self) -> dict:
        """Bark config with sub-model configs nested in HF style."""
        return {
            "model_type": "bark",
            "vocab_size": _SEM_VOCAB,
            "hidden_size": _SEM_HIDDEN,
            "num_hidden_layers": _SEM_LAYERS,
            "num_attention_heads": _SEM_HEADS,
            "semantic_config": {
                "block_size": _SEM_MAX_POS,
                "input_vocab_size": _SEM_VOCAB,
                "output_vocab_size": _SEM_OUTPUT_VOCAB,
                "num_layers": _SEM_LAYERS,
                "num_heads": _SEM_HEADS,
                "hidden_size": _SEM_HIDDEN,
            },
            "coarse_acoustics_config": {
                "block_size": _COARSE_MAX_POS,
                "input_vocab_size": _COARSE_VOCAB,
                "output_vocab_size": _COARSE_OUTPUT_VOCAB,
                "num_layers": _COARSE_LAYERS,
                "num_heads": _COARSE_HEADS,
                "hidden_size": _COARSE_HIDDEN,
            },
            "fine_acoustics_config": {
                "block_size": _FINE_MAX_POS,
                "input_vocab_size": _FINE_CODEBOOK_SIZE,
                "output_vocab_size": _FINE_CODEBOOK_SIZE,
                "num_layers": _FINE_LAYERS,
                "num_heads": _FINE_HEADS,
                "hidden_size": _FINE_HIDDEN,
                "n_codes_total": _FINE_N_EMBED_TABLES,
                "n_codes_given": 1,
            },
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Bark's weight layout.

        Bark has three sub-models with different prefixes:
          - semantic.input_embeds_layer.weight, semantic.layers.{i}.*, etc.
          - coarse_acoustics.input_embeds_layer.weight, etc.
          - fine_acoustics.input_embeds_layers.{i}.weight (multiple embeds),
            fine_acoustics.lm_heads.{j}.weight (multiple LM heads)

        Each sub-model's layers use:
          - layernorm_1/2.weight [hidden] (no bias for bark-small)
          - attn.att_proj.weight [3*H, H] (fused QKV)
          - attn.out_proj.weight [H, H]
          - mlp.in_proj.weight [4*H, H]
          - mlp.out_proj.weight [H, 4*H]
        """
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}

        # --- Semantic sub-model ---
        h = _SEM_HIDDEN
        inter = h * 4
        t["semantic.input_embeds_layer.weight"] = rand(_SEM_VOCAB, h)
        t["semantic.position_embeds_layer.weight"] = rand(_SEM_MAX_POS, h)
        for i in range(_SEM_LAYERS):
            p = f"semantic.layers.{i}"
            t[f"{p}.layernorm_1.weight"] = rand(h)
            t[f"{p}.layernorm_2.weight"] = rand(h)
            t[f"{p}.attn.att_proj.weight"] = rand(3 * h, h)
            t[f"{p}.attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.mlp.in_proj.weight"] = rand(inter, h)
            t[f"{p}.mlp.out_proj.weight"] = rand(h, inter)
        t["semantic.layernorm_final.weight"] = rand(h)
        t["semantic.lm_head.weight"] = rand(_SEM_OUTPUT_VOCAB, h)

        # --- Coarse sub-model ---
        h = _COARSE_HIDDEN
        inter = h * 4
        t["coarse_acoustics.input_embeds_layer.weight"] = rand(_COARSE_VOCAB, h)
        t["coarse_acoustics.position_embeds_layer.weight"] = rand(_COARSE_MAX_POS, h)
        for i in range(_COARSE_LAYERS):
            p = f"coarse_acoustics.layers.{i}"
            t[f"{p}.layernorm_1.weight"] = rand(h)
            t[f"{p}.layernorm_2.weight"] = rand(h)
            t[f"{p}.attn.att_proj.weight"] = rand(3 * h, h)
            t[f"{p}.attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.mlp.in_proj.weight"] = rand(inter, h)
            t[f"{p}.mlp.out_proj.weight"] = rand(h, inter)
        t["coarse_acoustics.layernorm_final.weight"] = rand(h)
        t["coarse_acoustics.lm_head.weight"] = rand(_COARSE_OUTPUT_VOCAB, h)

        # --- Fine sub-model ---
        h = _FINE_HIDDEN
        inter = h * 4
        for i in range(_FINE_N_EMBED_TABLES):
            t[f"fine_acoustics.input_embeds_layers.{i}.weight"] = rand(_FINE_CODEBOOK_SIZE, h)
        t["fine_acoustics.position_embeds_layer.weight"] = rand(_FINE_MAX_POS, h)
        for i in range(_FINE_LAYERS):
            p = f"fine_acoustics.layers.{i}"
            t[f"{p}.layernorm_1.weight"] = rand(h)
            t[f"{p}.layernorm_2.weight"] = rand(h)
            t[f"{p}.attn.att_proj.weight"] = rand(3 * h, h)
            t[f"{p}.attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.mlp.in_proj.weight"] = rand(inter, h)
            t[f"{p}.mlp.out_proj.weight"] = rand(h, inter)
        t["fine_acoustics.layernorm_final.weight"] = rand(h)
        for j in range(_FINE_N_LM_HEADS):
            t[f"fine_acoustics.lm_heads.{j}.weight"] = rand(_FINE_CODEBOOK_SIZE, h)

        return t

    def expected_weight_keys(self) -> set[str]:
        """Bark weight keys: semantic.*/coarse.*/fine.* per sub-model.

        Each semantic/coarse sub-model produces:
          {prefix}.embedding, {prefix}.position_embedding,
          {prefix}.layer.{i}.input_norm, {prefix}.layer.{i}.input_norm_beta,
          {prefix}.layer.{i}.post_attn_norm, {prefix}.layer.{i}.post_attn_norm_beta,
          {prefix}.layer.{i}.w_q, w_k, w_v, w_o, w_fc1, w_fc2,
          {prefix}.final_norm, {prefix}.final_norm_beta,
          {prefix}.w_out

        Fine sub-model produces:
          fine.embedding_{0..7}, fine.position_embedding,
          fine.layer.{i}.* (same per-layer keys),
          fine.final_norm, fine.final_norm_beta,
          fine.w_lm_head_{0..6}
        """
        keys: set[str] = set()

        # Semantic sub-model
        for prefix in ("semantic", "coarse"):
            n_layers = _SEM_LAYERS if prefix == "semantic" else _COARSE_LAYERS
            keys.update(
                {
                    f"{prefix}.embedding",
                    f"{prefix}.position_embedding",
                    f"{prefix}.final_norm",
                    f"{prefix}.final_norm_beta",
                    f"{prefix}.w_out",
                }
            )
            for i in range(n_layers):
                lp = f"{prefix}.layer.{i}"
                keys.update(
                    {
                        f"{lp}.input_norm",
                        f"{lp}.input_norm_beta",
                        f"{lp}.post_attn_norm",
                        f"{lp}.post_attn_norm_beta",
                        f"{lp}.w_q",
                        f"{lp}.w_k",
                        f"{lp}.w_v",
                        f"{lp}.w_o",
                        f"{lp}.w_fc1",
                        f"{lp}.w_fc2",
                    }
                )

        # Fine sub-model
        for i in range(_FINE_N_EMBED_TABLES):
            keys.add(f"fine.embedding_{i}")
        keys.add("fine.position_embedding")
        keys.add("fine.final_norm")
        keys.add("fine.final_norm_beta")
        for j in range(_FINE_N_LM_HEADS):
            keys.add(f"fine.w_lm_head_{j}")
        for i in range(_FINE_LAYERS):
            lp = f"fine.layer.{i}"
            keys.update(
                {
                    f"{lp}.input_norm",
                    f"{lp}.input_norm_beta",
                    f"{lp}.post_attn_norm",
                    f"{lp}.post_attn_norm_beta",
                    f"{lp}.w_q",
                    f"{lp}.w_k",
                    f"{lp}.w_v",
                    f"{lp}.w_o",
                    f"{lp}.w_fc1",
                    f"{lp}.w_fc2",
                }
            )

        return keys


class TestBarkEngine(FamilyPluginTestMixin):
    """Engine tests for Bark family plugin.

    Tier 0 and Tier 1 tests run via the mixin. Tier 2 (engine build) is
    skipped because Bark uses a custom multi-engine builder (semantic +
    coarse + fine + codec engines) rather than the standard single-engine
    builder.
    """

    tester_class = BarkPluginTester

    # --- Tier 2 skips ---
    @pytest.mark.skip(reason="custom builder -- uses non-standard graph construction")
    def test_build_engine_succeeds(self, tester, tmp_path):
        pass

    @pytest.mark.skip(reason="custom builder -- uses non-standard graph construction")
    def test_engine_io_tensor_names(self, tester, tmp_path):
        pass

    @pytest.mark.skip(reason="custom builder -- uses non-standard graph construction")
    def test_engine_logits_output_shape(self, tester, tmp_path):
        pass

    # --- Bark-specific Tier 1 tests ---

    def test_fused_qkv_correctly_split(self, tester, tmp_path):
        """Validate that Bark's fused att_proj [3H, H] is correctly split into Q/K/V.

        Intention:
            Bark stores Q, K, V in a single att_proj.weight tensor with shape
            [3*hidden, hidden]. The plugin must split this into three separate
            [hidden, hidden] tensors and transpose each to [in, out] layout.

            Example bug this catches: A plugin that splits at the wrong offset
            (e.g., taking the first 2H rows for Q instead of H rows).

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify semantic.layer.0.w_q has shape [hidden, hidden].
            3. Verify w_k and w_v have the same shape.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        h = _SEM_HIDDEN
        for proj in ("w_q", "w_k", "w_v"):
            key = f"semantic.layer.0.{proj}"
            assert key in weights, f"Missing key: {key}"
            assert weights[key].shape == (h, h), (
                f"{key} shape {weights[key].shape} != expected ({h}, {h})"
            )

    def test_fine_multiple_embedding_tables(self, tester, tmp_path):
        """Validate that Bark's fine model loads all 8 embedding tables.

        Intention:
            The fine model has 8 separate embedding tables (one per codebook).
            The plugin must load all of them as fine.embedding_{0..7}.

            Example bug this catches: A plugin that only loads the first
            embedding table and misses the remaining 7.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify all fine.embedding_{0..7} keys exist.
            3. Verify each has shape [codebook_size, hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for i in range(_FINE_N_EMBED_TABLES):
            key = f"fine.embedding_{i}"
            assert key in weights, f"Missing key: {key}"
            assert weights[key].shape == (_FINE_CODEBOOK_SIZE, _FINE_HIDDEN), (
                f"{key} shape {weights[key].shape} != "
                f"expected ({_FINE_CODEBOOK_SIZE}, {_FINE_HIDDEN})"
            )

    def test_fine_multiple_lm_heads(self, tester, tmp_path):
        """Validate that Bark's fine model loads all 7 LM heads (transposed).

        Intention:
            The fine model has 7 LM heads (codebooks 1-7). Each is transposed
            from HF [codebook_size, hidden] to [hidden, codebook_size] for
            matmul as the right-hand operand.

            Example bug this catches: A plugin that forgets to transpose the
            LM heads, leaving them in [codebook_size, hidden] layout.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify all fine.w_lm_head_{0..6} keys exist.
            3. Verify each has shape [hidden, codebook_size] (transposed).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for j in range(_FINE_N_LM_HEADS):
            key = f"fine.w_lm_head_{j}"
            assert key in weights, f"Missing key: {key}"
            assert weights[key].shape == (_FINE_HIDDEN, _FINE_CODEBOOK_SIZE), (
                f"{key} shape {weights[key].shape} != "
                f"expected ({_FINE_HIDDEN}, {_FINE_CODEBOOK_SIZE})"
            )

    def test_semantic_embedding_not_shared_with_lm_head(self, tester, tmp_path):
        """Validate that semantic embedding and w_out are independent.

        Intention:
            Bark's semantic model has a separate lm_head.weight with a
            different output vocab size from the input embedding. They
            should not be tied.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify semantic.embedding and semantic.w_out have different
               shapes (different vocab dimensions).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        embed = weights["semantic.embedding"]
        w_out = weights["semantic.w_out"]
        # Embedding is [vocab, hidden], w_out is [hidden, output_vocab]
        assert embed.shape[0] == _SEM_VOCAB
        assert w_out.shape[1] == _SEM_OUTPUT_VOCAB
        assert embed.shape[0] != w_out.shape[1], (
            "Semantic embedding and lm_head should have different vocab sizes"
        )

    def test_position_embeddings_present(self, tester, tmp_path):
        """Validate that learned position embeddings are loaded for each sub-model.

        Intention:
            Bark uses learned positional embeddings (not RoPE/sinusoidal).
            Each sub-model must have its position_embedding loaded.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify position embeddings exist for semantic, coarse, and fine.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for prefix, max_pos, hidden in [
            ("semantic", _SEM_MAX_POS, _SEM_HIDDEN),
            ("coarse", _COARSE_MAX_POS, _COARSE_HIDDEN),
            ("fine", _FINE_MAX_POS, _FINE_HIDDEN),
        ]:
            key = f"{prefix}.position_embedding"
            assert key in weights, f"Missing key: {key}"
            assert weights[key].shape == (max_pos, hidden), (
                f"{key} shape {weights[key].shape} != expected ({max_pos}, {hidden})"
            )

    # --- Override mixin tests that assume standard single-model layout ---

    def test_load_weights_embedding_shape(self, tester, tmp_path):
        """Override: Bark has prefixed embeddings, not a top-level 'embedding'.

        Verify semantic.embedding has shape [vocab, hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "semantic.embedding" in weights, "Missing semantic.embedding key"
        assert weights["semantic.embedding"].shape == (_SEM_VOCAB, _SEM_HIDDEN), (
            f"semantic.embedding shape {weights['semantic.embedding'].shape} "
            f"!= expected ({_SEM_VOCAB}, {_SEM_HIDDEN})"
        )

    def test_load_weights_projections_transposed(self, tester, tmp_path):
        """Override: Bark uses prefixed layer keys, not layer.0.w_q.

        Verify semantic.layer.0.w_q has shape[0] == hidden (transposed).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        w_q = weights["semantic.layer.0.w_q"]
        assert w_q.shape[0] == _SEM_HIDDEN, (
            f"w_q shape[0] = {w_q.shape[0]}, expected {_SEM_HIDDEN} "
            f"(projection should be transposed from HF [out, in] to [in, out])"
        )

    def test_bark_tp_build_rejects_single_device_parallel_config(self):
        decoder_tp_builder = _bark_tp_builder_module()

        with pytest.raises(ValueError, match="enabled parallel config"):
            decoder_tp_builder.build_bark_tp_decoder_engine(
                object(),
                _make_bark_tp_weights(),
                max_cache_length=4,
                sub_model="semantic",
                sub_cfg={
                    "hidden_size": _SEM_HIDDEN,
                    "num_heads": _SEM_HEADS,
                    "num_layers": _SEM_LAYERS,
                    "vocab_size": _SEM_VOCAB,
                    "output_vocab": _SEM_OUTPUT_VOCAB,
                    "intermediate_size": _SEM_HIDDEN * 4,
                },
                parallel_config=ParallelConfig(),
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"hidden": _SEM_HIDDEN + 1}, "hidden_size"),
            ({"num_heads": _SEM_HEADS + 1}, "num_heads"),
            ({"mlp_size": _SEM_HIDDEN * 4 + 1}, "mlp_size"),
        ],
    )
    def test_bark_tp_validation_rejects_non_divisible_dimensions(
        self,
        kwargs,
        message,
    ):
        decoder_tp_builder = _bark_tp_builder_module()
        params = {
            "sub_model": "semantic",
            "hidden": _SEM_HIDDEN,
            "num_heads": _SEM_HEADS,
            "mlp_size": _SEM_HIDDEN * 4,
            "parallel": ParallelConfig(
                mode="tensor_parallel",
                tp_size=2,
                rank=0,
            ),
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_bark_tp(**params)

    @pytest.mark.parametrize(
        ("key", "shape", "message"),
        [
            ("layer.0.w_q", (_SEM_HIDDEN, _SEM_HIDDEN - 1), "last dimension"),
            ("layer.0.w_o", (_SEM_HIDDEN - 1, _SEM_HIDDEN), "first dimension"),
        ],
    )
    def test_bark_tp_sharding_rejects_unshardable_weight_shapes(
        self,
        key,
        shape,
        message,
    ):
        decoder_tp_builder = _bark_tp_builder_module()
        weights = _make_bark_tp_weights()
        weights[key] = np.zeros(shape, dtype=np.float32)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder.shard_bark_decoder_weights(
                weights,
                sub_model="semantic",
                sub_cfg={
                    "hidden_size": _SEM_HIDDEN,
                    "num_heads": _SEM_HEADS,
                    "num_layers": _SEM_LAYERS,
                    "intermediate_size": _SEM_HIDDEN * 4,
                },
                parallel_config=ParallelConfig(
                    mode="tensor_parallel",
                    tp_size=2,
                    rank=0,
                ),
            )

    def test_bark_tp_shards_rank_local_decoder_weights(self):
        decoder_tp_builder = _bark_tp_builder_module()
        weights = _make_bark_tp_weights()
        shard = decoder_tp_builder.shard_bark_decoder_weights(
            weights,
            sub_model="semantic",
            sub_cfg={
                "hidden_size": _SEM_HIDDEN,
                "num_heads": _SEM_HEADS,
                "num_layers": _SEM_LAYERS,
                "intermediate_size": _SEM_HIDDEN * 4,
            },
            parallel_config=ParallelConfig(
                mode="tensor_parallel",
                tp_size=2,
                rank=1,
            ),
        )

        assert "_attention_size" not in shard
        assert shard["metadata"] == "kept"
        assert shard["embedding"] is weights["embedding"]
        assert shard["final_norm"] is weights["final_norm"]
        np.testing.assert_array_equal(
            shard["extra_tensor"],
            weights["extra_tensor"],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_q"],
            weights["layer.0.w_q"][:, _SEM_HIDDEN // 2 :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.q_bias"],
            weights["layer.0.q_bias"][_SEM_HIDDEN // 2 :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_o"],
            weights["layer.0.w_o"][_SEM_HIDDEN // 2 :, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc1"],
            weights["layer.0.w_fc1"][:, (_SEM_HIDDEN * 4) // 2 :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.fc1_bias"],
            weights["layer.0.fc1_bias"][(_SEM_HIDDEN * 4) // 2 :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc2"],
            weights["layer.0.w_fc2"][(_SEM_HIDDEN * 4) // 2 :, :],
        )
        assert shard["layer.0.o_bias"] is weights["layer.0.o_bias"]
