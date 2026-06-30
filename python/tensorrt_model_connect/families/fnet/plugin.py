"""FNet family plugin -- encoder-only model with Fourier Transform instead of attention.

FNet replaces self-attention with a 2D Discrete Fourier Transform (DFT):
  - No Q/K/V projections, no attention weights
  - Each layer applies FFT2D along (seq_len, hidden_size) dims, takes real part
  - POST-norm (residual then LayerNorm after Fourier/FFN)
  - Embedding: word + position + token_type -> LayerNorm -> Linear projection
  - FFN: fc1 -> gelu_new -> fc2 (same as BERT)
  - 2D DFT implemented via pre-computed DFT matrices (matrix multiplication)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias."""
    w = _load_tensor(readers, f"{prefix}.weight")
    b = _load_tensor(readers, f"{prefix}.bias")
    return w.astype(np.float32), b.astype(np.float32)


class FNetPlugin:
    name = "fnet"
    runtime_strategy = "fnet_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "fnet"

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        _intermediate = config.intermediate_size
        max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 4)

        # Detect prefix: "fnet" or ""
        if _has_tensor(readers, "fnet.embeddings.word_embeddings.weight"):
            root = "fnet"
        elif _has_tensor(readers, "embeddings.word_embeddings.weight"):
            root = ""
        else:
            root = "fnet"

        def _pfx(key):
            return f"{root}.{key}" if root else key

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, _pfx("embeddings.word_embeddings.weight"))
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (learned absolute)
        pos_embed = _load_tensor(readers, _pfx("embeddings.position_embeddings.weight"))
        assert pos_embed.shape == (max_pos, hidden), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {hidden})"
        )
        weights["position_embedding"] = pos_embed.astype(np.float32)

        # Token type embedding
        tt_embed = _load_tensor(readers, _pfx("embeddings.token_type_embeddings.weight"))
        assert tt_embed.shape == (type_vocab_size, hidden), (
            f"Token type embedding shape {tt_embed.shape} != ({type_vocab_size}, {hidden})"
        )
        weights["token_type_embedding"] = tt_embed.astype(np.float32)

        # Embedding LayerNorm
        embed_ln_w, embed_ln_b = _load_ln(readers, _pfx("embeddings.LayerNorm"))
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        # Embedding projection (FNet has a linear projection after LayerNorm)
        proj_w = _load_tensor(readers, _pfx("embeddings.projection.weight"))
        proj_b = _load_tensor(readers, _pfx("embeddings.projection.bias"))
        weights["embed_projection"] = np.ascontiguousarray(proj_w.T.astype(np.float32))
        weights["embed_projection_bias"] = proj_b.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = _pfx(f"encoder.layer.{layer_idx}")

            # Post-Fourier LayerNorm
            fourier_ln_w, fourier_ln_b = _load_ln(readers, f"{hf_prefix}.fourier.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = fourier_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = fourier_ln_b

            # FFN: intermediate.dense -> output.dense
            fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm
            out_ln_w, out_ln_b = _load_ln(readers, f"{hf_prefix}.output.LayerNorm")
            weights[f"{prefix}.output_norm"] = out_ln_w
            weights[f"{prefix}.output_norm_beta"] = out_ln_b

        # Pooler (optional)
        pooler_key = _pfx("pooler.dense.weight")
        if _has_tensor(readers, pooler_key):
            pooler_w = _load_tensor(readers, pooler_key)
            pooler_b = _load_tensor(readers, _pfx("pooler.dense.bias"))
            weights["pooler_w"] = np.ascontiguousarray(pooler_w.T.astype(np.float32))
            weights["pooler_bias"] = pooler_b.astype(np.float32)

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(parallel, feature="FNet tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("FNet tensor-parallel builds do not support quantization")
            from .model.parallel import build_tp_fnet_encoder_engine

            return build_tp_fnet_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        from .model.model import build_fnet_encoder_engine

        return build_fnet_encoder_engine(
            config, weights, max_seq_length=max_cache_length, verbose=verbose
        )


plugin = FNetPlugin()
