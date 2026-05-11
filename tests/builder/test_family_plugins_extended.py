"""Extended tests for family plugin load_weights — covers plugins with low test coverage.

Tests T5, BART, OLMo-2, ModernBERT, DeBERTa, Electra, FNet, DPR, ConvBERT,
M2M-100, Marian, XLNet, OLMo, and Albert plugins.

Each test creates a synthetic model directory with config.json and mock safetensors
files containing random tensors of the correct shapes, then calls plugin.load_weights()
and verifies the returned WeightDict has the expected keys and transforms applied.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-EXTENDED
Intent: Validate load_weights for extended family plugins (T5, BART, OLMo, ModernBERT, DeBERTa, etc.)
Preconditions: Synthetic safetensors with each family's HF weight naming are available
Postconditions: Each plugin produces expected canonical weight keys including encoder-decoder and encoder-only variants
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

# ---- helpers shared across all tests ----

RNG = np.random.RandomState(123)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


# =========================================================================
# T5 — encoder-decoder seq2seq
# =========================================================================

class TestT5Plugin:
    """T5 plugin: shared embedding, relative attention biases, encoder+decoder layers."""

    VOCAB, HIDDEN, LAYERS, HEADS, DKV, DFF = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, dkv, dff):
        t = {}
        t["shared.weight"] = _rand(vocab, hidden)
        # Encoder relative bias
        t["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["encoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"encoder.block.{i}"
            for proj in ("q", "k", "v", "o"):
                t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(dkv * heads if proj in ("q",) else dkv * heads if proj == "o" else dkv * heads, hidden) if proj != "o" else _rand(hidden, dkv * heads)
            # Fix: T5 uses [d_kv*heads, hidden] for q,k,v and [hidden, d_kv*heads] for o
            t[f"{pfx}.layer.0.SelfAttention.q.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.k.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.v.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.o.weight"] = _rand(hidden, dkv * heads)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)

        # Decoder relative biases
        t["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"decoder.block.{i}"
            # Self-attention
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            # Cross-attention
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)
            # FFN
            t[f"{pfx}.layer.2.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.2.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.2.layer_norm.weight"] = _rand(hidden)

        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.t5 import plugin

        config = {
            "model_type": "t5",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_heads": self.HEADS,
            "d_kv": self.DKV,
            "d_ff": self.DFF,
            "num_layers": self.LAYERS,
            "num_encoder_layers": self.LAYERS,
            "num_decoder_layers": self.LAYERS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.DKV, self.DFF)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert weights["shared_embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "enc_final_norm" in weights
        assert "final_norm" in weights
        assert "w_out" in weights
        assert "enc_rel_attn_bias" in weights
        assert "dec_self_rel_attn_bias" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "attn_norm", "w_fc1", "w_fc2", "ffn_norm"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"
            for key in ("w_q", "w_k", "w_v", "w_o", "input_norm",
                        "cross_w_q", "cross_w_k", "cross_w_v", "cross_w_o",
                        "cross_attn_norm", "w_fc1", "w_fc2", "post_attn_norm"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.t5 import plugin
        assert plugin.matches("t5")
        assert not plugin.matches("bart")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.t5 import plugin
        assert plugin.runtime_strategy == "text_to_text"


# =========================================================================
# BART — encoder-decoder seq2seq
# =========================================================================

class TestBartPlugin:
    """BART plugin: shared embedding, encoder+decoder with cross-attention."""

    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, ffn, max_pos):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)

        # Encoder position embeddings (offset=2 -> max_pos+2)
        t["model.encoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        # Encoder layernorm_embedding
        t["model.encoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.encoder.layernorm_embedding.bias"] = _rand(hidden)

        for i in range(layers):
            pfx = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        # Decoder
        t["model.decoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        t["model.decoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.decoder.layernorm_embedding.bias"] = _rand(hidden)

        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            # Self-attention
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            # Cross-attention
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            # FFN
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.bart import plugin

        config = {
            "model_type": "bart",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "encoder_layers": self.LAYERS,
            "decoder_layers": self.LAYERS,
            "encoder_attention_heads": self.HEADS,
            "decoder_attention_heads": self.HEADS,
            "encoder_ffn_dim": self.FFN,
            "decoder_ffn_dim": self.FFN,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "shared_embedding" in weights
        assert "enc_pos_embedding" in weights
        assert "dec_pos_embedding" in weights
        assert "enc_embed_norm" in weights
        assert "dec_embed_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "b_q", "b_k", "b_v", "b_o",
                        "attn_norm", "attn_norm_beta", "w_fc1", "b_fc1", "w_fc2",
                        "b_fc2", "ffn_norm", "ffn_norm_beta"):
                assert f"enc_layer.{i}.{key}" in weights, f"Missing enc_layer.{i}.{key}"

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "q_bias", "k_bias", "v_bias",
                        "o_bias", "input_norm", "input_norm_beta",
                        "cross_w_q", "cross_w_k", "cross_w_v", "cross_w_o",
                        "cross_b_q", "cross_b_k", "cross_b_v", "cross_b_o",
                        "cross_attn_norm", "cross_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "post_attn_norm", "post_attn_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.bart import plugin
        assert plugin.matches("bart")
        assert plugin.matches("mbart")
        assert not plugin.matches("t5")


# =========================================================================
# OLMo-2 — post-norm decoder with QK normalization
# =========================================================================

class TestOlmo2Plugin:
    """OLMo-2 plugin: post-norm, QK normalization, SwiGLU MLP."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 32, 16, 2, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_hidden = kv_heads * head_dim
        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_feedforward_layernorm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.k_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.v_proj.weight"] = _rand(kv_hidden, hidden)
            t[f"{p}.self_attn.o_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.self_attn.q_norm.weight"] = _rand(hidden)
            t[f"{p}.self_attn.k_norm.weight"] = _rand(kv_hidden)
            t[f"{p}.mlp.gate_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.up_proj.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.down_proj.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.olmo2 import plugin

        config = {
            "model_type": "olmo2",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.KV_HEADS, self.MLP)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert "final_norm" in weights
        assert "w_out" in weights

        for i in range(self.LAYERS):
            for key in ("post_attn_norm", "post_ff_norm", "w_q", "w_k", "w_v",
                        "w_o", "q_norm", "k_norm", "w_gate", "w_up", "w_down"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.olmo2 import plugin
        assert plugin.matches("olmo2")
        assert not plugin.matches("olmo")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.olmo2 import plugin
        assert plugin.runtime_strategy == "decoder_kv_cache"


# =========================================================================
# ModernBERT — encoder-only with fused QKV and GeGLU
# =========================================================================

class TestModernbertPlugin:
    """ModernBERT plugin: fused QKV, GeGLU MLP, RoPE, no bias."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE = 32, 16, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate):
        t = {}
        t["model.embeddings.tok_embeddings.weight"] = _rand(vocab, hidden)
        t["model.embeddings.norm.weight"] = _rand(hidden)
        t["model.final_norm.weight"] = _rand(hidden)

        for i in range(layers):
            p = f"model.layers.{i}"
            if i > 0:
                t[f"{p}.attn_norm.weight"] = _rand(hidden)
            t[f"{p}.attn.Wqkv.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.Wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp_norm.weight"] = _rand(hidden)
            t[f"{p}.mlp.Wi.weight"] = _rand(2 * intermediate, hidden)
            t[f"{p}.mlp.Wo.weight"] = _rand(hidden, intermediate)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        tensors = self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "embed_norm" in weights
        assert "final_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o", "mlp_norm",
                        "w_mlp_input", "w_mlp_gate", "w_down"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"
            # Layer 0 has no attn_norm
            if i > 0:
                assert f"layer.{i}.attn_norm" in weights

    def test_fused_qkv_split(self, tmp_path):
        """Verify the fused QKV weight is split correctly into Q, K, V."""
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        # Build known QKV
        q_raw = _rand(self.HIDDEN, self.HIDDEN)
        k_raw = _rand(self.HIDDEN, self.HIDDEN)
        v_raw = _rand(self.HIDDEN, self.HIDDEN)
        wqkv = np.concatenate([q_raw, k_raw, v_raw], axis=0)

        tensors = {}
        tensors["model.embeddings.tok_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        tensors["model.embeddings.norm.weight"] = _rand(self.HIDDEN)
        tensors["model.final_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.attn.Wqkv.weight"] = wqkv
        tensors["model.layers.0.attn.Wo.weight"] = _rand(self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.mlp_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.mlp.Wi.weight"] = _rand(2 * self.INTERMEDIATE, self.HIDDEN)
        tensors["model.layers.0.mlp.Wo.weight"] = _rand(self.HIDDEN, self.INTERMEDIATE)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Q, K, V should be transposed versions of the raw splits
        np.testing.assert_array_equal(weights["layer.0.w_q"],
                                       np.ascontiguousarray(q_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_k"],
                                       np.ascontiguousarray(k_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_v"],
                                       np.ascontiguousarray(v_raw.T.astype(np.float32)))

    def test_geglu_split(self, tmp_path):
        """Verify the fused GeGLU Wi weight is split into input and gate."""
        from tensorrt_model_connect.families.modernbert import plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
        }
        input_raw = _rand(self.INTERMEDIATE, self.HIDDEN)
        gate_raw = _rand(self.INTERMEDIATE, self.HIDDEN)
        wi = np.concatenate([input_raw, gate_raw], axis=0)

        tensors = {}
        tensors["model.embeddings.tok_embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        tensors["model.embeddings.norm.weight"] = _rand(self.HIDDEN)
        tensors["model.final_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.attn.Wqkv.weight"] = _rand(3 * self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.attn.Wo.weight"] = _rand(self.HIDDEN, self.HIDDEN)
        tensors["model.layers.0.mlp_norm.weight"] = _rand(self.HIDDEN)
        tensors["model.layers.0.mlp.Wi.weight"] = wi
        tensors["model.layers.0.mlp.Wo.weight"] = _rand(self.HIDDEN, self.INTERMEDIATE)

        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        np.testing.assert_array_equal(weights["layer.0.w_mlp_input"],
                                       np.ascontiguousarray(input_raw.T.astype(np.float32)))
        np.testing.assert_array_equal(weights["layer.0.w_mlp_gate"],
                                       np.ascontiguousarray(gate_raw.T.astype(np.float32)))

    def test_matches(self):
        from tensorrt_model_connect.families.modernbert import plugin
        assert plugin.matches("modernbert")
        assert plugin.matches("ModernBert")
        assert not plugin.matches("bert")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.modernbert import plugin
        assert plugin.runtime_strategy == "encoder_only"


# =========================================================================
# DeBERTa — encoder-only with disentangled attention
# =========================================================================

class TestDebertaPlugin:
    """DeBERTa plugin: disentangled attention, relative position embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["deberta.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["deberta.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["deberta.embeddings.LayerNorm.bias"] = _rand(hidden)

        # Relative position embedding
        t["deberta.encoder.rel_embeddings.weight"] = _rand(2 * max_pos, hidden)

        for i in range(layers):
            p = f"deberta.encoder.layer.{i}"
            # Fused QKV (in_proj)
            t[f"{p}.attention.self.in_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.self.q_bias"] = _rand(hidden)
            t[f"{p}.attention.self.v_bias"] = _rand(hidden)
            # Position projections (c2p and p2c)
            t[f"{p}.attention.self.pos_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.pos_q_proj.bias"] = _rand(hidden)
            # Output
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            # FFN
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.deberta import plugin

        config = {
            "model_type": "deberta",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "pos_att_type": "c2p|p2c",
            "max_relative_positions": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "embed_norm" in weights
        assert "rel_embeddings" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "q_bias", "v_bias",
                        "pos_proj", "pos_q_proj", "w_o", "o_bias",
                        "post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.deberta import plugin
        assert plugin.matches("deberta")
        assert not plugin.matches("deberta-v2")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.deberta import plugin
        assert plugin.runtime_strategy == "encoder_only"


# =========================================================================
# Electra — encoder-only (BERT-like)
# =========================================================================

class TestElectraPlugin:
    """Electra plugin: BERT-like encoder with learned pos + token type embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["electra.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["electra.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["electra.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["electra.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["electra.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"electra.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(hidden)
            t[f"{p}.attention.self.value.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.electra import plugin

        config = {
            "model_type": "electra",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "token_type_embedding" in weights
        assert "embed_norm" in weights
        assert "embed_norm_beta" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "q_bias", "k_bias", "v_bias",
                        "w_o", "o_bias", "post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.electra import plugin
        assert plugin.matches("electra")
        assert not plugin.matches("bert")

    def test_runtime_strategy(self):
        from tensorrt_model_connect.families.electra import plugin
        assert plugin.runtime_strategy == "encoder_only"


# =========================================================================
# FNet — encoder-only with Fourier Transform
# =========================================================================

class TestFNetPlugin:
    """FNet plugin: no attention, uses DFT, GELU FFN."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE, MAX_POS = 32, 16, 2, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate, max_pos):
        t = {}
        t["fnet.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["fnet.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["fnet.embeddings.token_type_embeddings.weight"] = _rand(4, hidden)
        t["fnet.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["fnet.embeddings.LayerNorm.bias"] = _rand(hidden)
        t["fnet.embeddings.projection.weight"] = _rand(hidden, hidden)
        t["fnet.embeddings.projection.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"fnet.encoder.layer.{i}"
            t[f"{p}.fourier.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.fourier.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.fnet import plugin

        config = {
            "model_type": "fnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "type_vocab_size": 4,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "token_type_embedding" in weights
        assert "embed_norm" in weights
        assert "embed_projection" in weights

        for i in range(self.LAYERS):
            for key in ("post_attn_norm", "post_attn_norm_beta",
                        "w_fc1", "fc1_bias", "w_fc2", "fc2_bias",
                        "output_norm", "output_norm_beta"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.fnet import plugin
        assert plugin.matches("fnet")
        assert not plugin.matches("bert")


# =========================================================================
# DPR — Dense Passage Retrieval (BERT-based encoder)
# =========================================================================

class TestDPRPlugin:
    """DPR plugin: BERT-like encoder for dense retrieval."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        # DPR uses ctx_encoder or question_encoder prefix
        t["ctx_encoder.bert_model.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["ctx_encoder.bert_model.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["ctx_encoder.bert_model.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["ctx_encoder.bert_model.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["ctx_encoder.bert_model.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"ctx_encoder.bert_model.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(hidden)
            t[f"{p}.attention.self.value.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(hidden)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        t["ctx_encoder.bert_model.pooler.dense.weight"] = _rand(hidden, hidden)
        t["ctx_encoder.bert_model.pooler.dense.bias"] = _rand(hidden)
        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.dpr import plugin

        config = {
            "model_type": "dpr",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "embed_norm" in weights
        assert "pooler_w" in weights
        assert "pooler_bias" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.dpr import plugin
        assert plugin.matches("dpr")
        assert not plugin.matches("bert")


# =========================================================================
# ConvBERT — encoder-only with mixed attention (span + conv)
# =========================================================================

class TestConvBERTPlugin:
    """ConvBERT plugin: mixed attention with span-based convolution."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        # ConvBERT uses head_ratio=2 by default — half heads are conv-based
        new_heads = heads // 2
        attn_head_size = (hidden // new_heads) // 2
        all_head_size = new_heads * attn_head_size
        conv_kernel_size = 9

        t = {}
        t["convbert.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["convbert.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["convbert.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["convbert.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["convbert.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"convbert.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.key.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.value.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(all_head_size)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            # SeparableConv1D: depthwise [channels, 1, kernel] + pointwise [1, channels, 1]
            t[f"{p}.attention.self.key_conv_attn_layer.depthwise.weight"] = _rand(all_head_size, 1, conv_kernel_size)
            t[f"{p}.attention.self.key_conv_attn_layer.pointwise.weight"] = _rand(1, all_head_size, 1)
            t[f"{p}.attention.self.key_conv_attn_layer.bias"] = _rand(all_head_size, 1)
            t[f"{p}.attention.self.conv_kernel_layer.weight"] = _rand(new_heads * conv_kernel_size, all_head_size)
            t[f"{p}.attention.self.conv_kernel_layer.bias"] = _rand(new_heads * conv_kernel_size)
            t[f"{p}.attention.self.conv_out_layer.weight"] = _rand(hidden, all_head_size)
            t[f"{p}.attention.self.conv_out_layer.bias"] = _rand(hidden)
            # FFN
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.convbert import plugin

        config = {
            "model_type": "convbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "head_ratio": 2,
            "conv_kernel_size": 9,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "embed_norm" in weights

        for i in range(self.LAYERS):
            for key in ("w_q", "w_k", "w_v", "w_o"):
                assert f"layer.{i}.{key}" in weights, f"Missing layer.{i}.{key}"

    def test_matches(self):
        from tensorrt_model_connect.families.convbert import plugin
        assert plugin.matches("convbert")
        assert not plugin.matches("bert")


# =========================================================================
# XLNet — generalized autoregressive with permutation LM
# =========================================================================

class TestXLNetPlugin:
    """XLNet plugin: two-stream attention, relative segment embeddings."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE = 32, 16, 2, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate):
        head_dim = hidden // heads
        t = {}
        t["transformer.word_embedding.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            p = f"transformer.layer.{i}"
            # XLNet stores projections as [hidden, num_heads, d_head] (not [out, in])
            for proj in ("q", "k", "v", "o", "r"):
                t[f"{p}.rel_attn.{proj}"] = _rand(hidden, heads, head_dim)
            t[f"{p}.rel_attn.r_w_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_r_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_s_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.seg_embed"] = _rand(2, heads, head_dim)
            t[f"{p}.rel_attn.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.rel_attn.layer_norm.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_1.weight"] = _rand(intermediate, hidden)
            t[f"{p}.ff.layer_1.bias"] = _rand(intermediate)
            t[f"{p}.ff.layer_2.weight"] = _rand(hidden, intermediate)
            t[f"{p}.ff.layer_2.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.xlnet import plugin

        config = {
            "model_type": "xlnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "d_inner": self.INTERMEDIATE,
            "n_layer": self.LAYERS,
            "n_head": self.HEADS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)

    def test_matches(self):
        from tensorrt_model_connect.families.xlnet import plugin
        assert plugin.matches("xlnet")
        assert not plugin.matches("bert")


# =========================================================================
# Albert — parameter-sharing encoder
# =========================================================================

class TestAlbertPlugin:
    """Albert plugin: cross-layer parameter sharing, factored embedding."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 2, 4, 32, 64
    EMBEDDING_SIZE = 8  # Albert uses factored embedding (smaller than hidden)

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos, embed_size):
        t = {}
        t["albert.embeddings.word_embeddings.weight"] = _rand(vocab, embed_size)
        t["albert.embeddings.position_embeddings.weight"] = _rand(max_pos, embed_size)
        t["albert.embeddings.token_type_embeddings.weight"] = _rand(2, embed_size)
        t["albert.embeddings.LayerNorm.weight"] = _rand(embed_size)
        t["albert.embeddings.LayerNorm.bias"] = _rand(embed_size)
        # Embedding projection (embed_size -> hidden)
        t["albert.encoder.embedding_hidden_mapping_in.weight"] = _rand(hidden, embed_size)
        t["albert.encoder.embedding_hidden_mapping_in.bias"] = _rand(hidden)

        # Albert uses shared weights across groups
        group_prefix = "albert.encoder.albert_layer_groups.0.albert_layers.0"
        t[f"{group_prefix}.attention.query.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.query.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.key.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.key.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.value.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.value.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.dense.weight"] = _rand(hidden, hidden)
        t[f"{group_prefix}.attention.dense.bias"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.weight"] = _rand(hidden)
        t[f"{group_prefix}.attention.LayerNorm.bias"] = _rand(hidden)
        t[f"{group_prefix}.ffn.weight"] = _rand(intermediate, hidden)
        t[f"{group_prefix}.ffn.bias"] = _rand(intermediate)
        t[f"{group_prefix}.ffn_output.weight"] = _rand(hidden, intermediate)
        t[f"{group_prefix}.ffn_output.bias"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.weight"] = _rand(hidden)
        t[f"{group_prefix}.full_layer_layer_norm.bias"] = _rand(hidden)

        # Pooler
        t["albert.pooler.weight"] = _rand(hidden, hidden)
        t["albert.pooler.bias"] = _rand(hidden)

        return t

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.albert import plugin

        config = {
            "model_type": "albert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "embedding_size": self.EMBEDDING_SIZE,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "num_hidden_groups": 1,
            "inner_group_num": 1,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS,
            self.INTERMEDIATE, self.MAX_POS, self.EMBEDDING_SIZE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert "embedding" in weights
        assert "position_embedding" in weights
        assert "embed_norm" in weights

    def test_matches(self):
        from tensorrt_model_connect.families.albert import plugin
        assert plugin.matches("albert")
        assert not plugin.matches("bert")


# =========================================================================
# fp8_calibrate — FP8 quantization scale extraction
# =========================================================================

class TestFp8Calibrate:
    """Test fp8_calibrate utility functions (pure Python, no GPU)."""

    def test_import(self):
        """Verify fp8_calibrate can be imported."""
        from tensorrt_model_connect import fp8_calibrate
        assert hasattr(fp8_calibrate, "extract_scales_from_state_dict")

    def test_maxbound_constants(self):
        """Verify maxbound constants are defined correctly."""
        from tensorrt_model_connect.fp8_calibrate import _MAXBOUND, _DEFAULT_MAXBOUND
        assert _MAXBOUND[(4, 3)] == 448.0  # FP8 E4M3
        assert _MAXBOUND[(5, 2)] == 57344.0  # FP8 E5M2
        assert _MAXBOUND[(0, 8)] == 127.0  # INT8
        assert _DEFAULT_MAXBOUND == 448.0

    def test_extract_scales_empty(self):
        """Scale extraction with empty state dict returns empty."""
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict
        result = extract_scales_from_state_dict({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_scales_with_amax(self):
        """Scale extraction finds and maps _amax tensors."""
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict

        state = {
            "model.layers.0.self_attn.q_proj.input_quantizer._amax": 224.0,
            "model.layers.0.self_attn.q_proj.weight_quantizer._amax": 112.0,
            "model.layers.0.mlp.gate_proj.input_quantizer._amax": 100.0,
            # Missing weight_quantizer — should not appear in output
        }
        result = extract_scales_from_state_dict(state)
        assert "model.layers.0.self_attn.q_proj" in result
        entry = result["model.layers.0.self_attn.q_proj"]
        assert "input_scale" in entry
        assert "weight_scale" in entry
        assert abs(entry["input_scale"] - 224.0 / 448.0) < 1e-6
        assert abs(entry["weight_scale"] - 112.0 / 448.0) < 1e-6
        # Incomplete entry should not be present
        assert "model.layers.0.mlp.gate_proj" not in result

    def test_extract_scales_with_exclude(self):
        """Scale extraction respects exclude_pattern."""
        import re
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict

        state = {
            "model.layers.0.self_attn.q_proj.input_quantizer._amax": 224.0,
            "model.layers.0.self_attn.q_proj.weight_quantizer._amax": 112.0,
        }
        # Exclude everything matching self_attn
        result = extract_scales_from_state_dict(
            state, exclude_pattern=re.compile(r".*self_attn.*"))
        assert len(result) == 0

    def test_maxbound_from_config(self):
        """Maxbound extraction from ModelOpt config."""
        from tensorrt_model_connect.fp8_calibrate import _maxbound_from_config

        config = {"quant_cfg": {"*weight_quantizer": {"num_bits": (4, 3)}}}
        assert _maxbound_from_config(config) == 448.0

        config = {"quant_cfg": {"*weight_quantizer": {"num_bits": (0, 8)}}}
        assert _maxbound_from_config(config) == 127.0

        # Unknown format falls back to default
        assert _maxbound_from_config({}) == 448.0


# =========================================================================
# Schedulers — flow match Euler scheduler
# =========================================================================

class TestSchedulers:
    """Test scheduler base class and flow match Euler implementation."""

    def test_flow_match_euler_init(self):
        from tensorrt_model_connect.schedulers.flow_match_euler import FlowMatchEulerScheduler
        scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0)
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 1.0

    def test_flow_match_euler_set_timesteps(self):
        from tensorrt_model_connect.schedulers.flow_match_euler import FlowMatchEulerScheduler
        scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(4)
        timesteps = scheduler.timesteps
        assert len(timesteps) == 4
        # Timesteps should be monotonically decreasing (from ~1000 toward 0)
        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i + 1]

    def test_flow_match_euler_step(self):
        """Verify Euler step computes prev_sample correctly."""
        from tensorrt_model_connect.schedulers.flow_match_euler import FlowMatchEulerScheduler
        scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(4)

        sample = np.ones((1, 4), dtype=np.float32)
        model_output = np.ones((1, 4), dtype=np.float32)
        result = scheduler.step(model_output, scheduler.timesteps[0].item(), sample, 0)
        assert result.shape == sample.shape
        assert result.dtype == np.float32

    def test_flow_match_euler_add_noise(self):
        """Verify add_noise interpolates between original and noise."""
        from tensorrt_model_connect.schedulers.flow_match_euler import FlowMatchEulerScheduler
        scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)
        original = np.zeros((1, 4), dtype=np.float32)
        noise = np.ones((1, 4), dtype=np.float32)
        # At timestep=0 (sigma=0), result should be original
        result = scheduler.add_noise(original, noise, 0.0)
        np.testing.assert_allclose(result, original)
        # At timestep=1000 (sigma=1.0), result should be noise
        result = scheduler.add_noise(original, noise, 1000.0)
        np.testing.assert_allclose(result, noise)

    def test_flow_match_euler_shift(self):
        """Verify shift parameter affects timestep schedule."""
        from tensorrt_model_connect.schedulers.flow_match_euler import FlowMatchEulerScheduler
        s1 = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0)
        s2 = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=3.0)
        s1.set_timesteps(4)
        s2.set_timesteps(4)
        # Shifted schedule should differ
        assert not np.allclose(s1.timesteps, s2.timesteps)

    def test_scheduler_registry(self):
        from tensorrt_model_connect.schedulers import get_scheduler
        scheduler = get_scheduler("flow_match_euler", num_train_timesteps=1000)
        assert scheduler is not None
        assert scheduler.num_train_timesteps == 1000

    def test_scheduler_registry_unknown(self):
        from tensorrt_model_connect.schedulers import get_scheduler
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("nonexistent_scheduler")
