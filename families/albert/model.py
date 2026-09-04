# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ALBERT family plugin -- encoder-only with cross-layer parameter sharing.

ALBERT (A Lite BERT) uses:
  - Embedding factorization: small embedding_size (128) projected to hidden_size (768)
  - Cross-layer parameter sharing: single set of transformer weights reused N times
  - Learned absolute position embeddings
  - Token type embeddings (segment A/B)
  - LayerNorm (with beta) -- POST-norm architecture
  - 2-projection MLP (ffn/ffn_output) with gelu_new activation
  - Bidirectional attention (no causal mask)
  - Pooler dense layer for [CLS] token representation
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


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias."""
    w = _load_tensor(readers, f"{prefix}.weight")
    b = _load_tensor(readers, f"{prefix}.bias")
    return w.astype(np.float32), b.astype(np.float32)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _AlbertModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        _hidden = config.hidden_size
        embedding_size = config.raw.get("embedding_size", 128)
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        _num_heads = config.num_attention_heads
        _intermediate = config.intermediate_size
        max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        num_hidden_groups = config.raw.get("num_hidden_groups", 1)
        inner_group_num = config.raw.get("inner_group_num", 1)

        weights = WeightDict()

        # Word embedding: [vocab, embedding_size]
        embedding = _load_tensor(readers, "albert.embeddings.word_embeddings.weight")
        assert embedding.shape == (vocab, embedding_size), (
            f"Embedding shape {embedding.shape} != ({vocab}, {embedding_size})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding: [max_pos, embedding_size]
        pos_embed = _load_tensor(readers, "albert.embeddings.position_embeddings.weight")
        assert pos_embed.shape == (max_pos, embedding_size), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {embedding_size})"
        )
        weights["position_embedding"] = pos_embed.astype(np.float32)

        # Token type embedding: [type_vocab_size, embedding_size]
        tt_embed = _load_tensor(readers, "albert.embeddings.token_type_embeddings.weight")
        assert tt_embed.shape == (type_vocab_size, embedding_size), (
            f"Token type embedding shape {tt_embed.shape} != ({type_vocab_size}, {embedding_size})"
        )
        weights["token_type_embedding"] = tt_embed.astype(np.float32)

        # Embedding LayerNorm (over embedding_size dim)
        embed_ln_w, embed_ln_b = _load_ln(readers, "albert.embeddings.LayerNorm")
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        # Embedding projection: embedding_size -> hidden_size
        proj_w = _load_tensor(readers, "albert.encoder.embedding_hidden_mapping_in.weight")
        proj_b = _load_tensor(readers, "albert.encoder.embedding_hidden_mapping_in.bias")
        # proj_w is [hidden, embedding_size] in HF -- transpose to [embedding_size, hidden]
        weights["embed_projection"] = np.ascontiguousarray(proj_w.T.astype(np.float32))
        weights["embed_projection_bias"] = proj_b.astype(np.float32)

        # ALBERT cross-layer parameter sharing:
        # Only one set of transformer weights per group.
        # Replicate for all num_layers so the encoder_builder sees per-layer weights.
        layers_per_group = num_layers // num_hidden_groups

        for group_idx in range(num_hidden_groups):
            for inner_idx in range(inner_group_num):
                hf_prefix = (
                    f"albert.encoder.albert_layer_groups.{group_idx}.albert_layers.{inner_idx}"
                )

                # Load shared weights once
                q_w = _load_tensor(readers, f"{hf_prefix}.attention.query.weight")
                k_w = _load_tensor(readers, f"{hf_prefix}.attention.key.weight")
                v_w = _load_tensor(readers, f"{hf_prefix}.attention.value.weight")
                o_w = _load_tensor(readers, f"{hf_prefix}.attention.dense.weight")

                q_b = _load_tensor(readers, f"{hf_prefix}.attention.query.bias")
                k_b = _load_tensor(readers, f"{hf_prefix}.attention.key.bias")
                v_b = _load_tensor(readers, f"{hf_prefix}.attention.value.bias")
                o_b = _load_tensor(readers, f"{hf_prefix}.attention.dense.bias")

                attn_ln_w, attn_ln_b = _load_ln(readers, f"{hf_prefix}.attention.LayerNorm")

                fc1_w = _load_tensor(readers, f"{hf_prefix}.ffn.weight")
                fc1_b = _load_tensor(readers, f"{hf_prefix}.ffn.bias")
                fc2_w = _load_tensor(readers, f"{hf_prefix}.ffn_output.weight")
                fc2_b = _load_tensor(readers, f"{hf_prefix}.ffn_output.bias")

                out_ln_w, out_ln_b = _load_ln(readers, f"{hf_prefix}.full_layer_layer_norm")

                group_start = group_idx * layers_per_group

                for offset in range(layers_per_group):
                    layer_idx = group_start + offset
                    prefix = f"layer.{layer_idx}"

                    weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
                    weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
                    weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))
                    weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))

                    weights[f"{prefix}.q_bias"] = q_b.astype(np.float32)
                    weights[f"{prefix}.k_bias"] = k_b.astype(np.float32)
                    weights[f"{prefix}.v_bias"] = v_b.astype(np.float32)
                    weights[f"{prefix}.o_bias"] = o_b.astype(np.float32)

                    weights[f"{prefix}.post_attn_norm"] = attn_ln_w
                    weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

                    weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
                    weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
                    weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
                    weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

                    weights[f"{prefix}.output_norm"] = out_ln_w
                    weights[f"{prefix}.output_norm_beta"] = out_ln_b

        # Pooler
        pooler_w = _load_tensor(readers, "albert.pooler.weight")
        pooler_b = _load_tensor(readers, "albert.pooler.bias")
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
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("ALBERT tensor-parallel builds do not support quantization")
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
        raise NotImplementedError("albert does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("albert does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("albert does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("albert does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("albert task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "albert":
        raise ValueError(f"ALBERT builder requires model_type='albert', got {config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("ALBERT precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("ALBERT max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("ALBERT has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("ALBERT does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _AlbertModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="albert", task=request.task, backend="trt")
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
