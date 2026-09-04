# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DistilBERT family plugin — encoder-only bidirectional transformer.

DistilBERT is a distilled version of BERT with:
  - 6 layers (vs BERT's 12), 768 hidden, 12 heads (from config)
  - Learned absolute position embeddings
  - NO token type embeddings (no segment A/B)
  - NO pooler layer
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection FFN (lin1/lin2) with GELU activation
  - POST-norm (residual then LayerNorm), not pre-norm
  - Bidirectional attention (no causal mask)

Weight naming:
  - Embeddings: distilbert.embeddings.word_embeddings, position_embeddings, LayerNorm
  - Attention: distilbert.transformer.layer.N.attention.{q_lin,k_lin,v_lin,out_lin}
  - FFN: distilbert.transformer.layer.N.ffn.{lin1,lin2}
  - Norms: distilbert.transformer.layer.N.{sa_layer_norm,output_layer_norm}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
)
from .parallel import normalize_parallel_config
from .encoder_builder import build_encoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _DistilBertModel:
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
        max_pos = config.max_position_embeddings

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, "distilbert.embeddings.word_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (learned absolute)
        pos_embed = _load_tensor(readers, "distilbert.embeddings.position_embeddings.weight")
        assert pos_embed.shape == (max_pos, hidden), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {hidden})"
        )
        weights["position_embedding"] = pos_embed.astype(np.float32)

        # DistilBERT has no token_type_embeddings. The encoder builder expects
        # one, so provide a zero table that acts as an identity under addition.
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

        # Embedding LayerNorm
        embed_ln_w = _load_tensor(readers, "distilbert.embeddings.LayerNorm.weight")
        embed_ln_b = _load_tensor(readers, "distilbert.embeddings.LayerNorm.bias")
        weights["embed_norm"] = embed_ln_w.astype(np.float32)
        weights["embed_norm_beta"] = embed_ln_b.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"distilbert.transformer.layer.{layer_idx}"

            # Q, K, V projections — HF stores [out, in], transpose to [in, out]
            q_w = _load_tensor(readers, f"{hf_prefix}.attention.q_lin.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attention.k_lin.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attention.v_lin.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # QKV biases
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.q_lin.bias"
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.k_lin.bias"
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.v_lin.bias"
            ).astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attention.out_lin.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.out_lin.bias"
            ).astype(np.float32)

            # Post-attention LayerNorm (sa_layer_norm)
            sa_ln_w = _load_tensor(readers, f"{hf_prefix}.sa_layer_norm.weight")
            sa_ln_b = _load_tensor(readers, f"{hf_prefix}.sa_layer_norm.bias")
            weights[f"{prefix}.post_attn_norm"] = sa_ln_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = sa_ln_b.astype(np.float32)

            # FFN: lin1 -> GELU -> lin2
            fc1_w = _load_tensor(readers, f"{hf_prefix}.ffn.lin1.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.ffn.lin1.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.ffn.lin2.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.ffn.lin2.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm (output_layer_norm)
            out_ln_w = _load_tensor(readers, f"{hf_prefix}.output_layer_norm.weight")
            out_ln_b = _load_tensor(readers, f"{hf_prefix}.output_layer_norm.bias")
            weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
            weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

        # DistilBERT has no pooler layer.

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
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("DistilBERT tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_encoder_engine

            return build_tp_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_encoder_engine(
            config, weights, max_seq_length=max_cache_length, precision=precision, verbose=verbose
        )


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _tokenizer_runtime_contract(model_dir: Path) -> dict[str, object]:
    """Resolve this family's exact native-tokenizer framing."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=True,
    )
    default_ids = list(tokenizer.encode("hello"))
    plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    if default_ids == plain_ids:
        prefix_ids, suffix_ids = [], []
    elif not plain_ids:
        prefix_ids, suffix_ids = default_ids, []
    else:
        frame = next(
            (
                start
                for start in range(len(default_ids) - len(plain_ids) + 1)
                if default_ids[start : start + len(plain_ids)] == plain_ids
            ),
            None,
        )
        if frame is None:
            raise RuntimeError("tokenizer special-token framing is not a prefix/suffix")
        prefix_ids = default_ids[:frame]
        suffix_ids = default_ids[frame + len(plain_ids) :]
    return {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": prefix_ids,
        "tokenizer_suffix_ids": suffix_ids,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.image_height is not None:
        raise NotImplementedError("distilbert does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("distilbert does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("distilbert does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("distilbert does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("distilbert task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "distilbert":
        raise ValueError(
            f"DistilBERT builder requires model_type='distilbert', got {config.model_type!r}"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("DistilBERT precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("DistilBERT max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("DistilBERT has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("DistilBERT does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _DistilBertModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="distilbert", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            **_tokenizer_runtime_contract(model_dir),
            "tensor_parallel_size": parallel.tp_size,
        },
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
