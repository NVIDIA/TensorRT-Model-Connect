"""RoBERTa / XLM-RoBERTa family plugin — encoder-only bidirectional transformer.

Architecturally identical to BERT:
  - Learned absolute position embeddings
  - Token type embeddings (present but unused — all zeros)
  - LayerNorm (with bias) instead of RMSNorm
  - 2-projection MLP (fc1/fc2) with GELU activation
  - POST-norm (residual then LayerNorm), not pre-norm
  - Bidirectional attention (no causal mask)

Key differences from BERT:
  - Weight prefix is "roberta." instead of "bert."
  - Some XLM-RoBERTa checkpoints use "model.roberta." prefix
  - Token type embeddings exist but are unused (all zeros at inference)
  - Vocab size differs (50265 for RoBERTa, ~250K for XLM-RoBERTa)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .encoder_builder import build_encoder_engine


def _detect_prefix(readers) -> str:
    """Detect the weight prefix used in the checkpoint.

    Returns "roberta" or "model.roberta" depending on which prefix is found.
    XLM-RoBERTa checkpoints sometimes nest under "model.roberta.*".
    """
    if _has_tensor(readers, "model.roberta.embeddings.word_embeddings.weight"):
        return "model.roberta"
    return "roberta"


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias, handling legacy gamma/beta naming."""
    if _has_tensor(readers, f"{prefix}.weight"):
        w = _load_tensor(readers, f"{prefix}.weight")
        b = _load_tensor(readers, f"{prefix}.bias")
    else:
        w = _load_tensor(readers, f"{prefix}.gamma")
        b = _load_tensor(readers, f"{prefix}.beta")
    return w.astype(np.float32), b.astype(np.float32)


class RobertaPlugin:
    name = "roberta"
    runtime_strategy = "roberta_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("roberta", "xlm-roberta", "camembert")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        _intermediate = config.intermediate_size
        _max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 1)

        hf_root = _detect_prefix(readers)

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(
            readers, f"{hf_root}.embeddings.word_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (learned absolute).
        # RoBERTa has padding_idx=1, so position IDs start at 2.
        # The position embedding table has (max_pos, hidden) rows where
        # rows 0 and 1 are padding-related. Slice starting at offset 2
        # so the encoder builder can use positions [0, 1, ..., N-1].
        pos_embed_raw = _load_tensor(
            readers, f"{hf_root}.embeddings.position_embeddings.weight")
        pad_idx = config.raw.get("pad_token_id", 1)
        pos_offset = pad_idx + 1  # RoBERTa positions start at padding_idx + 1
        pos_embed = pos_embed_raw[pos_offset:].astype(np.float32)
        weights["position_embedding"] = pos_embed

        # Token type embedding — present but unused (all zeros at inference).
        # Load if available; otherwise synthesize zeros.
        tt_key = f"{hf_root}.embeddings.token_type_embeddings.weight"
        if _has_tensor(readers, tt_key):
            tt_embed = _load_tensor(readers, tt_key)
            assert tt_embed.shape == (type_vocab_size, hidden), (
                f"Token type embedding shape {tt_embed.shape} "
                f"!= ({type_vocab_size}, {hidden})")
            weights["token_type_embedding"] = tt_embed.astype(np.float32)
        else:
            weights["token_type_embedding"] = np.zeros(
                (type_vocab_size, hidden), dtype=np.float32)

        # Embedding LayerNorm
        embed_ln_w, embed_ln_b = _load_ln(
            readers, f"{hf_root}.embeddings.LayerNorm")
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{hf_root}.encoder.layer.{layer_idx}"

            # Q, K, V projections — HF stores [out, in], transpose to [in, out]
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

            # QKV biases
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.query.bias",
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.key.bias",
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.value.bias",
            ).astype(np.float32)

            # Output projection
            o_w = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(
                o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.bias",
            ).astype(np.float32)

            # Post-attention LayerNorm (handles legacy gamma/beta)
            attn_ln_w, attn_ln_b = _load_ln(
                readers, f"{hf_prefix}.attention.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

            # FFN: intermediate.dense -> output.dense
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

            # Output LayerNorm (handles legacy gamma/beta)
            out_ln_w, out_ln_b = _load_ln(
                readers, f"{hf_prefix}.output.LayerNorm")
            weights[f"{prefix}.output_norm"] = out_ln_w
            weights[f"{prefix}.output_norm_beta"] = out_ln_b

        # Pooler (optional — used for [CLS] representation)
        pooler_key = f"{hf_root}.pooler.dense.weight"
        if _has_tensor(readers, pooler_key):
            pooler_w = _load_tensor(readers, pooler_key)
            pooler_b = _load_tensor(readers, f"{hf_root}.pooler.dense.bias")
            weights["pooler_w"] = np.ascontiguousarray(
                pooler_w.T.astype(np.float32))
            weights["pooler_bias"] = pooler_b.astype(np.float32)

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        public_module = sys.modules.get(__package__)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="RoBERTa tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("RoBERTa tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_encoder_engine
            builder = getattr(
                public_module, "build_tp_encoder_engine", build_tp_encoder_engine)
            return builder(
                config, weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        builder = getattr(public_module, "build_encoder_engine", build_encoder_engine)
        return builder(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = RobertaPlugin()
