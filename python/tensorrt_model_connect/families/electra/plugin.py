"""Electra family plugin -- encoder-only discriminator/generator transformer.

ELECTRA has the same architecture as BERT:
  - Learned absolute position embeddings
  - Token type embeddings (segment A/B)
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (fc1/fc2) with GELU activation
  - POST-norm (residual then LayerNorm), not pre-norm
  - Bidirectional attention (no causal mask)
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
from .model.model import build_encoder_engine


def _load_ln(readers, prefix):
    if _has_tensor(readers, f"{prefix}.weight"):
        w = _load_tensor(readers, f"{prefix}.weight")
        b = _load_tensor(readers, f"{prefix}.bias")
    else:
        w = _load_tensor(readers, f"{prefix}.gamma")
        b = _load_tensor(readers, f"{prefix}.beta")
    return w.astype(np.float32), b.astype(np.float32)


def _detect_prefix(readers) -> str:
    if _has_tensor(readers, "electra.embeddings.word_embeddings.weight"):
        return "electra"
    if _has_tensor(readers, "embeddings.word_embeddings.weight"):
        return ""
    return "electra"


def _bpfx(root, key):
    return f"{root}.{key}" if root else key


class ElectraPlugin:
    name = "electra"
    runtime_strategy = "electra_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "electra"

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
        _num_heads = config.num_attention_heads
        _intermediate = config.intermediate_size
        _max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        embedding_size = config.raw.get("embedding_size", hidden)

        root = _detect_prefix(readers)
        weights = WeightDict()

        embedding = _load_tensor(readers, _bpfx(root, "embeddings.word_embeddings.weight"))
        assert embedding.shape == (vocab, embedding_size)

        if embedding_size != hidden:
            proj_w = _load_tensor(readers, _bpfx(root, "embeddings_project.weight"))
            proj_b = _load_tensor(readers, _bpfx(root, "embeddings_project.bias"))
            embedding = embedding.astype(np.float32) @ proj_w.T.astype(np.float32) + proj_b.astype(
                np.float32
            )

        weights["embedding"] = embedding.astype(np.float32)

        pos_embed = _load_tensor(readers, _bpfx(root, "embeddings.position_embeddings.weight"))
        if embedding_size != hidden and pos_embed.shape[1] == embedding_size:
            proj_w = _load_tensor(readers, _bpfx(root, "embeddings_project.weight"))
            proj_b = _load_tensor(readers, _bpfx(root, "embeddings_project.bias"))
            pos_embed = pos_embed.astype(np.float32) @ proj_w.T.astype(np.float32) + proj_b.astype(
                np.float32
            )
        weights["position_embedding"] = pos_embed.astype(np.float32)

        tt_key = _bpfx(root, "embeddings.token_type_embeddings.weight")
        if _has_tensor(readers, tt_key):
            tt_embed = _load_tensor(readers, tt_key)
            assert tt_embed.shape[0] == type_vocab_size
            if embedding_size != hidden and tt_embed.shape[1] == embedding_size:
                proj_w = _load_tensor(readers, _bpfx(root, "embeddings_project.weight"))
                proj_b = _load_tensor(readers, _bpfx(root, "embeddings_project.bias"))
                tt_embed = tt_embed.astype(np.float32) @ proj_w.T.astype(
                    np.float32
                ) + proj_b.astype(np.float32)
            weights["token_type_embedding"] = tt_embed.astype(np.float32)
        else:
            weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

        embed_ln_w, embed_ln_b = _load_ln(readers, _bpfx(root, "embeddings.LayerNorm"))
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = _bpfx(root, f"encoder.layer.{layer_idx}")

            q_w = _load_tensor(readers, f"{hf_prefix}.attention.self.query.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attention.self.key.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attention.self.value.weight")
            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.query.bias"
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.key.bias"
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.value.bias"
            ).astype(np.float32)

            o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.bias"
            ).astype(np.float32)

            attn_ln_w, attn_ln_b = _load_ln(readers, f"{hf_prefix}.attention.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

            fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")
            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            out_ln_w, out_ln_b = _load_ln(readers, f"{hf_prefix}.output.LayerNorm")
            weights[f"{prefix}.output_norm"] = out_ln_w
            weights[f"{prefix}.output_norm_beta"] = out_ln_b

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
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="ELECTRA tensor-parallel builds"
            )
            if quant_ctx is not None:
                raise ValueError("ELECTRA tensor-parallel builds do not support quantization")
            from .model.parallel import build_tp_encoder_engine

            return build_tp_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_encoder_engine(
            config, weights, max_seq_length=max_cache_length, verbose=verbose
        )


plugin = ElectraPlugin()
