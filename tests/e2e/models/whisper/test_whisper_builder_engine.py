# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for Whisper (encoder-decoder ASR).

Intention:
    Validate the Whisper family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness.

    Whisper is an encoder-decoder transformer for automatic speech recognition:
      - Encoder: Conv1d stem -> learned positional encoding -> N self-attention
        layers -> encoder output
      - Decoder: autoregressive text generation with causal self-attention
        (KV cache) + cross-attention to encoder output + GELU MLP

    Uses LayerNorm (not RMSNorm), GELU activation, learned positional
    embeddings, and Q/K/V/O projections with biases (though k_proj.bias
    may be absent). Cross-attention has per-layer K/V projections baked
    into the decoder TRT graph.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    get_config_dict() (for encoder/decoder-specific fields), make_hf_tensors()
    (for encoder conv stem + encoder/decoder layers), and
    expected_weight_keys() (for enc_layer.*/layer.*/cross_* keys).
    Tier 2 is skipped because Whisper uses a custom dual-engine builder
    (encoder + decoder) rather than the standard single-engine builder.

Trace: ARCH-FAM-001, UD-FAM-WHISPER-01
Intent: Validate the Whisper family plugin weight loading for encoder-decoder ASR including conv stem, encoder/decoder layers, cross-attention, learned positions, and dual-engine key layout.
Preconditions: safetensors and tensorrt_model_connect are importable; no TRT or GPU required for weight-loading tests.
Postconditions: All encoder (enc_layer.*), decoder (layer.*), and cross-attention (cross_*) weight keys are present with correct shapes for the dual-engine architecture.
"""

from __future__ import annotations

import importlib

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


# Whisper dimensions kept tiny for fast tests.
_HIDDEN = 16
_ENC_LAYERS = 2
_DEC_LAYERS = 2
_ENC_HEADS = 4
_DEC_HEADS = 4
_ENC_FFN = 32
_DEC_FFN = 32
_VOCAB = 32
_NUM_MEL_BINS = 8
_MAX_SOURCE_POSITIONS = 16
_MAX_TARGET_POSITIONS = 16


def _make_whisper_tp_weights(
    *,
    hidden: int = _HIDDEN,
    dec_layers: int = _DEC_LAYERS,
    dec_heads: int = _DEC_HEADS,
    dec_ffn: int = _DEC_FFN,
) -> WeightDict:
    rng = np.random.RandomState(123)

    def rand(*shape: int) -> np.ndarray:
        return rng.randn(*shape).astype(np.float32)

    weights = WeightDict({
        "_dec_layers": dec_layers,
        "_dec_heads": dec_heads,
        "_dec_ffn": dec_ffn,
        "_max_source_positions": _MAX_SOURCE_POSITIONS,
        "dec_embedding": rand(_VOCAB, hidden),
        "dec_pos_embedding": rand(_MAX_TARGET_POSITIONS, hidden),
        "final_norm": rand(hidden),
        "final_norm_beta": rand(hidden),
        "w_out": rand(hidden, _VOCAB),
    })
    for i in range(dec_layers):
        pfx = f"layer.{i}"
        for key in (
            "w_q", "w_k", "w_v",
            "cross_w_q", "cross_w_k", "cross_w_v",
        ):
            weights[f"{pfx}.{key}"] = rand(hidden, hidden)
        for key in ("q_bias", "k_bias", "v_bias"):
            weights[f"{pfx}.{key}"] = rand(hidden)
        for key in ("cross_b_q", "cross_b_k", "cross_b_v"):
            weights[f"{pfx}.{key}"] = rand(hidden)
        for key in ("w_o", "cross_w_o"):
            weights[f"{pfx}.{key}"] = rand(hidden, hidden)
        weights[f"{pfx}.w_fc1"] = rand(hidden, dec_ffn)
        weights[f"{pfx}.fc1_bias"] = rand(dec_ffn)
        weights[f"{pfx}.w_fc2"] = rand(dec_ffn, hidden)
        weights[f"{pfx}.o_bias"] = rand(hidden)
        weights[f"{pfx}.cross_b_o"] = rand(hidden)
        weights[f"{pfx}.fc2_bias"] = rand(hidden)
        for key in (
            "input_norm", "input_norm_beta",
            "cross_attn_norm", "cross_attn_norm_beta",
            "post_attn_norm", "post_attn_norm_beta",
        ):
            weights[f"{pfx}.{key}"] = rand(hidden)
    return weights


def _whisper_tp_builder_module():
    return importlib.import_module(
        "tensorrt_model_connect.families.whisper.decoder_tp_builder"
    )


@pytest.mark.parametrize(
    ("fp32_layers", "expected_precision"),
    [([], "fp16"), ([0], "fp32")],
)
def test_whisper_encoder_selector_routes_precision(
    monkeypatch: pytest.MonkeyPatch,
    fp32_layers: list[int],
    expected_precision: str,
) -> None:
    plugin_module = pytest.importorskip(
        "tensorrt_model_connect.families.whisper.plugin"
    )
    observed = {}

    def fake_build_encoder(config, weights, *, precision, verbose):
        del config, weights, verbose
        observed["precision"] = precision
        return b"encoder-plan"

    monkeypatch.setattr(
        plugin_module, "_build_whisper_encoder", fake_build_encoder
    )
    config = type("Config", (), {"raw": {"_fp32_layers": fp32_layers}})()

    plan = plugin_module.plugin.build_vision_engine(
        "unused", config, WeightDict(), precision="fp16"
    )

    assert plan == b"encoder-plan"
    assert observed["precision"] == expected_precision


def test_whisper_encoder_selector_rejects_unknown_index() -> None:
    plugin_module = pytest.importorskip(
        "tensorrt_model_connect.families.whisper.plugin"
    )
    config = type("Config", (), {"raw": {"_fp32_layers": [1]}})()

    with pytest.raises(ValueError, match="supports only selector 0"):
        plugin_module.plugin.build_vision_engine(
            "unused", config, WeightDict(), precision="fp16"
        )


class WhisperPluginTester(FamilyPluginTester):
    """Tester for the Whisper family plugin.

    Whisper uses:
      - Encoder with Conv1d stem + learned positional embeddings
      - Decoder with causal self-attention + cross-attention to encoder
      - LayerNorm (with bias) everywhere
      - GELU activation in MLP
      - Q/K/V/O projections with biases (k_proj.bias may be absent)
      - Cross-attention with per-layer K/V projections
      - HF prefix: model.encoder.layers.{i}.* + model.decoder.layers.{i}.*
    """

    plugin_module = "tensorrt_model_connect.families.whisper.plugin"
    model_type = "whisper"
    spec = TinyModelSpec(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_DEC_FFN,
        num_hidden_layers=_DEC_LAYERS,
        num_attention_heads=_DEC_HEADS,
        num_key_value_heads=_DEC_HEADS,
    )

    def get_config_dict(self) -> dict:
        """Whisper config with encoder/decoder-specific fields."""
        return {
            "model_type": "whisper",
            "vocab_size": _VOCAB,
            "d_model": _HIDDEN,
            "encoder_layers": _ENC_LAYERS,
            "decoder_layers": _DEC_LAYERS,
            "encoder_attention_heads": _ENC_HEADS,
            "decoder_attention_heads": _DEC_HEADS,
            "encoder_ffn_dim": _ENC_FFN,
            "decoder_ffn_dim": _DEC_FFN,
            "num_mel_bins": _NUM_MEL_BINS,
            "max_source_positions": _MAX_SOURCE_POSITIONS,
            "max_target_positions": _MAX_TARGET_POSITIONS,
            "layer_norm_eps": 1e-5,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Whisper's weight layout.

        Encoder weights:
          - model.encoder.conv1.weight [hidden, mel_bins, 3]
          - model.encoder.conv1.bias [hidden]
          - model.encoder.conv2.weight [hidden, hidden, 3]
          - model.encoder.conv2.bias [hidden]
          - model.encoder.embed_positions.weight [max_source, hidden]
          - model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
          - model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}
          - model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
          - model.encoder.layers.{i}.fc1.{weight,bias}
          - model.encoder.layers.{i}.fc2.{weight,bias}
          - model.encoder.layers.{i}.final_layer_norm.{weight,bias}
          - model.encoder.layer_norm.{weight,bias}

        Decoder weights:
          - model.decoder.embed_tokens.weight [vocab, hidden]
          - model.decoder.embed_positions.weight [max_target, hidden]
          - model.decoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
          - model.decoder.layers.{i}.self_attn.out_proj.{weight,bias}
          - model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
          - model.decoder.layers.{i}.encoder_attn.{q,k,v}_proj.{weight,bias}
          - model.decoder.layers.{i}.encoder_attn.out_proj.{weight,bias}
          - model.decoder.layers.{i}.encoder_attn_layer_norm.{weight,bias}
          - model.decoder.layers.{i}.fc1.{weight,bias}
          - model.decoder.layers.{i}.fc2.{weight,bias}
          - model.decoder.layers.{i}.final_layer_norm.{weight,bias}
          - model.decoder.layer_norm.{weight,bias}
          - proj_out.weight [vocab, hidden]  (or tied to embed_tokens)
        """
        h = _HIDDEN
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}

        # --- Encoder ---
        # Conv stem
        t["model.encoder.conv1.weight"] = rand(h, _NUM_MEL_BINS, 3)
        t["model.encoder.conv1.bias"] = rand(h)
        t["model.encoder.conv2.weight"] = rand(h, h, 3)
        t["model.encoder.conv2.bias"] = rand(h)

        # Learned positional embedding
        t["model.encoder.embed_positions.weight"] = rand(
            _MAX_SOURCE_POSITIONS, h)

        # Encoder layers
        for i in range(_ENC_LAYERS):
            p = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{p}.self_attn.{proj}_proj.weight"] = rand(h, h)
                t[f"{p}.self_attn.{proj}_proj.bias"] = rand(h)
            t[f"{p}.self_attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.self_attn.out_proj.bias"] = rand(h)
            t[f"{p}.self_attn_layer_norm.weight"] = rand(h)
            t[f"{p}.self_attn_layer_norm.bias"] = rand(h)
            t[f"{p}.fc1.weight"] = rand(_ENC_FFN, h)
            t[f"{p}.fc1.bias"] = rand(_ENC_FFN)
            t[f"{p}.fc2.weight"] = rand(h, _ENC_FFN)
            t[f"{p}.fc2.bias"] = rand(h)
            t[f"{p}.final_layer_norm.weight"] = rand(h)
            t[f"{p}.final_layer_norm.bias"] = rand(h)

        # Encoder final norm
        t["model.encoder.layer_norm.weight"] = rand(h)
        t["model.encoder.layer_norm.bias"] = rand(h)

        # --- Decoder ---
        t["model.decoder.embed_tokens.weight"] = rand(_VOCAB, h)
        t["model.decoder.embed_positions.weight"] = rand(
            _MAX_TARGET_POSITIONS, h)

        for i in range(_DEC_LAYERS):
            p = f"model.decoder.layers.{i}"
            # Self-attention
            for proj in ("q", "k", "v"):
                t[f"{p}.self_attn.{proj}_proj.weight"] = rand(h, h)
                t[f"{p}.self_attn.{proj}_proj.bias"] = rand(h)
            t[f"{p}.self_attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.self_attn.out_proj.bias"] = rand(h)
            t[f"{p}.self_attn_layer_norm.weight"] = rand(h)
            t[f"{p}.self_attn_layer_norm.bias"] = rand(h)
            # Cross-attention
            for proj in ("q", "k", "v"):
                t[f"{p}.encoder_attn.{proj}_proj.weight"] = rand(h, h)
                t[f"{p}.encoder_attn.{proj}_proj.bias"] = rand(h)
            t[f"{p}.encoder_attn.out_proj.weight"] = rand(h, h)
            t[f"{p}.encoder_attn.out_proj.bias"] = rand(h)
            t[f"{p}.encoder_attn_layer_norm.weight"] = rand(h)
            t[f"{p}.encoder_attn_layer_norm.bias"] = rand(h)
            # MLP
            t[f"{p}.fc1.weight"] = rand(_DEC_FFN, h)
            t[f"{p}.fc1.bias"] = rand(_DEC_FFN)
            t[f"{p}.fc2.weight"] = rand(h, _DEC_FFN)
            t[f"{p}.fc2.bias"] = rand(h)
            t[f"{p}.final_layer_norm.weight"] = rand(h)
            t[f"{p}.final_layer_norm.bias"] = rand(h)

        # Decoder final norm
        t["model.decoder.layer_norm.weight"] = rand(h)
        t["model.decoder.layer_norm.bias"] = rand(h)

        # LM head (proj_out)
        t["proj_out.weight"] = rand(_VOCAB, h)

        return t

    def expected_weight_keys(self) -> set[str]:
        """Whisper weight keys: encoder + decoder + cross-attention.

        Encoder keys: enc_conv1/2_weight/bias, enc_pos_embedding,
          enc_layer.{i}.w_q/k/v/o, b_q/k/v/o, attn_norm, attn_norm_beta,
          w_fc1/2, b_fc1/2, ffn_norm, ffn_norm_beta,
          enc_final_norm, enc_final_norm_beta

        Decoder keys (layer.{i}.*): w_q/k/v/o, q/k/v/o_bias,
          input_norm, input_norm_beta,
          cross_w_q/k/v/o, cross_b_q/k/v/o,
          cross_attn_norm, cross_attn_norm_beta,
          w_fc1/2, fc1/2_bias,
          post_attn_norm, post_attn_norm_beta

        Global: dec_embedding, dec_pos_embedding,
          final_norm, final_norm_beta, w_out
        """
        keys: set[str] = set()

        # Encoder conv stem
        keys.update({
            "enc_conv1_weight", "enc_conv1_bias",
            "enc_conv2_weight", "enc_conv2_bias",
            "enc_pos_embedding",
        })

        # Encoder layers
        for i in range(_ENC_LAYERS):
            pfx = f"enc_layer.{i}"
            for proj in ("q", "k", "v"):
                keys.add(f"{pfx}.w_{proj}")
                keys.add(f"{pfx}.b_{proj}")
            keys.update({
                f"{pfx}.w_o", f"{pfx}.b_o",
                f"{pfx}.attn_norm", f"{pfx}.attn_norm_beta",
                f"{pfx}.w_fc1", f"{pfx}.b_fc1",
                f"{pfx}.w_fc2", f"{pfx}.b_fc2",
                f"{pfx}.ffn_norm", f"{pfx}.ffn_norm_beta",
            })

        keys.update({
            "enc_final_norm", "enc_final_norm_beta",
        })

        # Decoder embeddings
        keys.update({
            "dec_embedding", "dec_pos_embedding",
        })

        # Decoder layers
        for i in range(_DEC_LAYERS):
            pfx = f"layer.{i}"
            # Self-attention
            for proj in ("q", "k", "v"):
                keys.add(f"{pfx}.w_{proj}")
                keys.add(f"{pfx}.{proj}_bias")
            keys.update({
                f"{pfx}.w_o", f"{pfx}.o_bias",
                f"{pfx}.input_norm", f"{pfx}.input_norm_beta",
            })
            # Cross-attention
            for proj in ("q", "k", "v"):
                keys.add(f"{pfx}.cross_w_{proj}")
                keys.add(f"{pfx}.cross_b_{proj}")
            keys.update({
                f"{pfx}.cross_w_o", f"{pfx}.cross_b_o",
                f"{pfx}.cross_attn_norm", f"{pfx}.cross_attn_norm_beta",
            })
            # MLP
            keys.update({
                f"{pfx}.w_fc1", f"{pfx}.fc1_bias",
                f"{pfx}.w_fc2", f"{pfx}.fc2_bias",
                f"{pfx}.post_attn_norm", f"{pfx}.post_attn_norm_beta",
            })

        # Global
        keys.update({
            "final_norm", "final_norm_beta", "w_out",
        })

        return keys


class TestWhisperEngine(FamilyPluginTestMixin):
    """Engine tests for Whisper family plugin.

    Tier 0 and Tier 1 tests run via the mixin. Tier 2 (engine build) is
    skipped because Whisper uses a custom dual-engine builder (encoder +
    decoder) rather than the standard single-engine builder.
    """

    tester_class = WhisperPluginTester

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

    # --- Whisper-specific Tier 1 tests ---

    def test_encoder_conv_stem_loaded(self, tester, tmp_path):
        """Validate that encoder Conv1d stem weights are loaded correctly.

        Intention:
            Whisper's encoder begins with two Conv1d layers that process the
            mel spectrogram. If these weights are missing or have wrong shapes,
            the encoder will produce garbage features.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify enc_conv1/2_weight and enc_conv1/2_bias exist.
            3. Verify conv1 weight shape: [hidden, mel_bins, 3].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "enc_conv1_weight" in weights
        assert "enc_conv1_bias" in weights
        assert "enc_conv2_weight" in weights
        assert "enc_conv2_bias" in weights
        assert weights["enc_conv1_weight"].shape == (
            _HIDDEN, _NUM_MEL_BINS, 3), (
            f"enc_conv1_weight shape {weights['enc_conv1_weight'].shape} != "
            f"expected ({_HIDDEN}, {_NUM_MEL_BINS}, 3)"
        )

    def test_cross_attention_weights_present(self, tester, tmp_path):
        """Validate that cross-attention Q/K/V/O weights exist for all decoder layers.

        Intention:
            Whisper's decoder has cross-attention to the encoder output at every
            layer. If cross-attention weights are missing, the decoder cannot
            attend to the encoder features and will produce text unrelated to
            the audio input.

        Setup:
            1. Create synthetic model directory and load weights.
            2. For each decoder layer, verify cross_w_q/k/v/o and
               cross_b_q/k/v/o exist.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for i in range(_DEC_LAYERS):
            pfx = f"layer.{i}"
            for proj in ("q", "k", "v"):
                w_key = f"{pfx}.cross_w_{proj}"
                b_key = f"{pfx}.cross_b_{proj}"
                assert w_key in weights, f"Missing cross-attn key: {w_key}"
                assert b_key in weights, f"Missing cross-attn key: {b_key}"
            assert f"{pfx}.cross_w_o" in weights
            assert f"{pfx}.cross_b_o" in weights

    def test_decoder_embedding_shape(self, tester, tmp_path):
        """Validate that decoder embedding has correct shape.

        Intention:
            Whisper uses dec_embedding (not the standard 'embedding' key)
            for the decoder token embedding lookup.

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify dec_embedding shape is [vocab, hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "dec_embedding" in weights, "Missing dec_embedding key"
        assert weights["dec_embedding"].shape == (_VOCAB, _HIDDEN), (
            f"dec_embedding shape {weights['dec_embedding'].shape} != "
            f"expected ({_VOCAB}, {_HIDDEN})"
        )

    def test_encoder_positional_embedding_shape(self, tester, tmp_path):
        """Validate encoder learned positional embedding shape.

        Intention:
            Whisper uses learned (not sinusoidal) positional embeddings for the
            encoder. The shape must be [max_source_positions, hidden].

        Setup:
            1. Create synthetic model directory and load weights.
            2. Verify enc_pos_embedding shape.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "enc_pos_embedding" in weights, (
            "Missing enc_pos_embedding key"
        )
        assert weights["enc_pos_embedding"].shape == (
            _MAX_SOURCE_POSITIONS, _HIDDEN), (
            f"enc_pos_embedding shape {weights['enc_pos_embedding'].shape} != "
            f"expected ({_MAX_SOURCE_POSITIONS}, {_HIDDEN})"
        )

    # --- Override mixin tests that assume standard single-model layout ---

    def test_load_weights_embedding_shape(self, tester, tmp_path):
        """Override: Whisper uses dec_embedding, not 'embedding'.

        Verify dec_embedding has shape [vocab, hidden].
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert "dec_embedding" in weights, "Missing dec_embedding key"
        assert weights["dec_embedding"].shape == (_VOCAB, _HIDDEN), (
            f"dec_embedding shape {weights['dec_embedding'].shape} != "
            f"expected ({_VOCAB}, {_HIDDEN})"
        )

    def test_load_weights_projections_transposed(self, tester, tmp_path):
        """Override: Whisper uses layer.{i}.w_q for decoder projections.

        Verify layer.0.w_q has shape[0] == hidden (transposed from [out, in] to [in, out]).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        w_q = weights["layer.0.w_q"]
        assert w_q.shape[0] == _HIDDEN, (
            f"w_q shape[0] = {w_q.shape[0]}, expected {_HIDDEN} "
            f"(projection should be transposed from HF [out, in] to [in, out])"
        )

    def test_whisper_tp_build_rejects_single_device_mode(self):
        decoder_tp_builder = _whisper_tp_builder_module()

        with pytest.raises(ValueError, match="requires tensor_parallel mode"):
            decoder_tp_builder.build_whisper_tp_decoder_engine(
                object(),
                WeightDict(),
                max_cache_length=4,
                parallel_config=ParallelConfig(),
            )

    def test_whisper_tp_validation_ignores_single_device_mode(self):
        decoder_tp_builder = _whisper_tp_builder_module()

        decoder_tp_builder._validate_whisper_tp(
            WeightDict(),
            hidden=_HIDDEN,
            num_heads=_DEC_HEADS,
            ffn_dim=_DEC_FFN,
            parallel=ParallelConfig(),
        )

    @pytest.mark.parametrize(
        ("parallel", "overrides", "message"),
        [
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=-1),
                {},
                "concrete rank",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"hidden": _HIDDEN + 1},
                "hidden size divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"num_heads": _DEC_HEADS + 1},
                "decoder_attention_heads divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"ffn_dim": _DEC_FFN + 1},
                "decoder_ffn_dim divisible",
            ),
        ],
    )
    def test_whisper_tp_validation_rejects_bad_config_dimensions(
        self,
        parallel,
        overrides,
        message,
    ):
        decoder_tp_builder = _whisper_tp_builder_module()
        kwargs = {
            "hidden": _HIDDEN,
            "num_heads": _DEC_HEADS,
            "ffn_dim": _DEC_FFN,
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_whisper_tp(
                _make_whisper_tp_weights(),
                parallel=parallel,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("key", "shape", "message"),
        [
            ("layer.0.w_q", (_HIDDEN, _HIDDEN - 1), "output dim"),
            ("layer.0.w_o", (_HIDDEN - 1, _HIDDEN), "input dim"),
            ("layer.0.w_fc1", (_HIDDEN, _DEC_FFN - 1), "w_fc1 output dim"),
        ],
    )
    def test_whisper_tp_validation_rejects_unshardable_weight_shapes(
        self,
        key,
        shape,
        message,
    ):
        decoder_tp_builder = _whisper_tp_builder_module()
        weights = _make_whisper_tp_weights()
        weights[key] = np.zeros(shape, dtype=np.float32)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_whisper_tp(
                weights,
                hidden=_HIDDEN,
                num_heads=_DEC_HEADS,
                ffn_dim=_DEC_FFN,
                parallel=ParallelConfig(
                    mode="tensor_parallel",
                    tp_size=2,
                    rank=0,
                ),
            )

    def test_whisper_tp_sharding_returns_original_for_single_device_mode(self):
        decoder_tp_builder = _whisper_tp_builder_module()
        weights = _make_whisper_tp_weights()
        assert decoder_tp_builder.shard_whisper_decoder_weights(
            weights,
            parallel=ParallelConfig(),
        ) is weights

    def test_whisper_tp_shards_rank_local_decoder_weights(self):
        decoder_tp_builder = _whisper_tp_builder_module()
        weights = _make_whisper_tp_weights()
        shard = decoder_tp_builder.shard_whisper_decoder_weights(
            weights,
            parallel=ParallelConfig(
                mode="tensor_parallel",
                tp_size=2,
                rank=1,
            ),
        )

        assert isinstance(shard, WeightDict)
        assert shard["_tensor_parallel_size"] == 2
        assert shard["_tensor_parallel_rank"] == 1
        assert shard["_dec_layers"] == _DEC_LAYERS

        np.testing.assert_array_equal(
            shard["layer.0.w_q"],
            weights["layer.0.w_q"][:, _HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.cross_w_v"],
            weights["layer.0.cross_w_v"][:, _HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.q_bias"],
            weights["layer.0.q_bias"][_HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.cross_b_k"],
            weights["layer.0.cross_b_k"][_HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_o"],
            weights["layer.0.w_o"][_HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.cross_w_o"],
            weights["layer.0.cross_w_o"][_HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc1"],
            weights["layer.0.w_fc1"][:, _DEC_FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.fc1_bias"],
            weights["layer.0.fc1_bias"][_DEC_FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc2"],
            weights["layer.0.w_fc2"][_DEC_FFN // 2:, :],
        )
        assert shard["final_norm"] is weights["final_norm"]
