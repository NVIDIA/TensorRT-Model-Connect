"""XLNet family plugin -- encoder-only bidirectional transformer with relative positional encoding.

XLNet uses:
  - Sinusoidal relative positional encoding (Transformer-XL style)
  - Segment-relative encoding (learned seg_embed)
  - Relative attention with content (ac), position (bd), and segment (ef) scores
  - POST-norm (residual then LayerNorm) -- same flow as BERT but with relative attention
  - GELU activation in FFN
  - Weight shapes: q/k/v/o/r are [d_model, n_head, d_head] (not [out, in])
  - Additional per-layer biases: r_w_bias, r_r_bias, r_s_bias
  - For inference: content-stream only (no query stream), bidirectional attention

Trace IDs: ARCH-XLNET, UD-XLNET-001
"""

from __future__ import annotations

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


def _detect_xlnet_prefix(readers) -> str:
    """Detect weight prefix: transformer (standard HF .bin) or empty (stripped)."""
    if _has_tensor(readers, "transformer.word_embedding.weight"):
        return "transformer"
    if _has_tensor(readers, "word_embedding.weight"):
        return ""
    return "transformer"


def _pfx(root, key):
    """Join root prefix with key, handling empty root."""
    return f"{root}.{key}" if root else key


class XlnetPlugin:
    name = "xlnet"
    runtime_strategy = "xlnet_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "xlnet"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        d_head = config.raw.get("d_head", hidden // num_heads)

        root = _detect_xlnet_prefix(readers)
        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, _pfx(root, "word_embedding.weight"))
        assert embedding.shape == (vocab, hidden)
        weights["embedding"] = embedding.astype(np.float32)

        # mask_emb
        mask_key = _pfx(root, "mask_emb")
        if _has_tensor(readers, mask_key):
            weights["mask_emb"] = _load_tensor(readers, mask_key).astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = _pfx(root, f"layer.{layer_idx}")

            for proj in ["q", "k", "v", "o", "r"]:
                w = _load_tensor(readers, f"{hf_prefix}.rel_attn.{proj}")
                w_flat = w.reshape(hidden, num_heads * d_head)
                weights[f"{prefix}.w_{proj}"] = w_flat.astype(np.float32)

            for bias_name in ["r_w_bias", "r_r_bias", "r_s_bias"]:
                b = _load_tensor(readers, f"{hf_prefix}.rel_attn.{bias_name}")
                weights[f"{prefix}.{bias_name}"] = b.astype(np.float32)

            seg = _load_tensor(readers, f"{hf_prefix}.rel_attn.seg_embed")
            weights[f"{prefix}.seg_embed"] = seg.astype(np.float32)

            weights[f"{prefix}.attn_norm"] = _load_tensor(
                readers, f"{hf_prefix}.rel_attn.layer_norm.weight").astype(np.float32)
            weights[f"{prefix}.attn_norm_beta"] = _load_tensor(
                readers, f"{hf_prefix}.rel_attn.layer_norm.bias").astype(np.float32)

            weights[f"{prefix}.ff_norm"] = _load_tensor(
                readers, f"{hf_prefix}.ff.layer_norm.weight").astype(np.float32)
            weights[f"{prefix}.ff_norm_beta"] = _load_tensor(
                readers, f"{hf_prefix}.ff.layer_norm.bias").astype(np.float32)

            fc1_w = _load_tensor(readers, f"{hf_prefix}.ff.layer_1.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.ff.layer_1.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.ff.layer_2.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.ff.layer_2.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="XLNet tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("XLNet tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_xlnet_engine
            return build_tp_xlnet_engine(
                config, weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        from .xlnet_builder import build_xlnet_engine
        return build_xlnet_engine(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = XlnetPlugin()
