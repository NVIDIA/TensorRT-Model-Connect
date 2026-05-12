"""DPR (Dense Passage Retrieval) family plugin -- BERT-based dual encoder.

DPR uses a BERT backbone with the prefix 'ctx_encoder.bert_model.*' for context
encoders and 'question_encoder.bert_model.*' for question encoders. The
architecture is identical to BERT -- the only difference is the weight key prefix.

DPR outputs a pooled [CLS] embedding for passage retrieval. Uses the embedding
runtime strategy with mean-pool + L2-normalize in the C++ runtime.

Trace: ARCH-ENCODER, UD-DPR-WEIGHTS
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...encoder_builder import build_encoder_engine


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias, handling legacy gamma/beta naming."""
    if _has_tensor(readers, f"{prefix}.weight"):
        w = _load_tensor(readers, f"{prefix}.weight")
        b = _load_tensor(readers, f"{prefix}.bias")
    else:
        w = _load_tensor(readers, f"{prefix}.gamma")
        b = _load_tensor(readers, f"{prefix}.beta")
    return w.astype(np.float32), b.astype(np.float32)


def _detect_dpr_prefix(readers) -> str:
    """Detect the DPR weight prefix.

    DPR context encoders use 'ctx_encoder.bert_model', question encoders
    use 'question_encoder.bert_model'. Fall back to 'bert' for compatibility.
    """
    for prefix in (
        "ctx_encoder.bert_model",
        "question_encoder.bert_model",
        "bert",
    ):
        if _has_tensor(readers, f"{prefix}.embeddings.word_embeddings.weight"):
            return prefix
    if _has_tensor(readers, "embeddings.word_embeddings.weight"):
        return ""
    return "ctx_encoder.bert_model"


def _bpfx(root, key):
    """Join root prefix with key, handling empty root."""
    return f"{root}.{key}" if root else key


class DprPlugin:
    name = "dpr"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "dpr"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        _num_heads = config.num_attention_heads
        _intermediate = config.intermediate_size
        max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 2)

        root = _detect_dpr_prefix(readers)

        weights = WeightDict()

        embedding = _load_tensor(
            readers, _bpfx(root, "embeddings.word_embeddings.weight"))
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        pos_embed = _load_tensor(
            readers, _bpfx(root, "embeddings.position_embeddings.weight"))
        assert pos_embed.shape == (max_pos, hidden), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {hidden})")
        weights["position_embedding"] = pos_embed.astype(np.float32)

        tt_key = _bpfx(root, "embeddings.token_type_embeddings.weight")
        if _has_tensor(readers, tt_key):
            tt_embed = _load_tensor(readers, tt_key)
            assert tt_embed.shape == (type_vocab_size, hidden), (
                f"Token type embedding shape {tt_embed.shape} "
                f"!= ({type_vocab_size}, {hidden})")
            weights["token_type_embedding"] = tt_embed.astype(np.float32)
        else:
            weights["token_type_embedding"] = np.zeros(
                (type_vocab_size, hidden), dtype=np.float32)

        embed_ln_w, embed_ln_b = _load_ln(
            readers, _bpfx(root, "embeddings.LayerNorm"))
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = _bpfx(root, f"encoder.layer.{layer_idx}")

            q_w = _load_tensor(
                readers, f"{hf_prefix}.attention.self.query.weight")
            k_w = _load_tensor(
                readers, f"{hf_prefix}.attention.self.key.weight")
            v_w = _load_tensor(
                readers, f"{hf_prefix}.attention.self.value.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(
                q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(
                k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(
                v_w.T.astype(np.float32))

            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.query.bias"
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.key.bias"
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.value.bias"
            ).astype(np.float32)

            o_w = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(
                o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.bias"
            ).astype(np.float32)

            attn_ln_w, attn_ln_b = _load_ln(
                readers, f"{hf_prefix}.attention.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

            fc1_w = _load_tensor(
                readers, f"{hf_prefix}.intermediate.dense.weight")
            fc1_b = _load_tensor(
                readers, f"{hf_prefix}.intermediate.dense.bias")
            fc2_w = _load_tensor(
                readers, f"{hf_prefix}.output.dense.weight")
            fc2_b = _load_tensor(
                readers, f"{hf_prefix}.output.dense.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(
                fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(
                fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            out_ln_w, out_ln_b = _load_ln(
                readers, f"{hf_prefix}.output.LayerNorm")
            weights[f"{prefix}.output_norm"] = out_ln_w
            weights[f"{prefix}.output_norm_beta"] = out_ln_b

        pooler_key = _bpfx(root, "pooler.dense.weight")
        if _has_tensor(readers, pooler_key):
            pooler_w = _load_tensor(readers, pooler_key)
            pooler_b = _load_tensor(
                readers, _bpfx(root, "pooler.dense.bias"))
            weights["pooler_w"] = np.ascontiguousarray(
                pooler_w.T.astype(np.float32))
            weights["pooler_bias"] = pooler_b.astype(np.float32)

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        return build_encoder_engine(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = DprPlugin()
