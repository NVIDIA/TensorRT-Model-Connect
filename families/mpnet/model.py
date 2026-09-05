# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPNet family plugin — encoder-only bidirectional transformer.

MPNet (Masked and Permuted Pre-training) shares the same encoder
architecture as BERT but with different weight naming:
  - No token type embeddings (no segment A/B)
  - QKV projections use .attn.q/.attn.k/.attn.v (not .self.query etc.)
  - Post-attention LayerNorm is at .attention.LayerNorm (not .attention.output.LayerNorm)
  - Weight prefix may be absent or "mpnet."

Detection: model_type == "mpnet"
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
    _has_tensor,
)
from .parallel import normalize_parallel_config
from .encoder_builder import build_encoder_engine


def _detect_prefix(readers) -> str:
    """Detect weight prefix: '' (sentence-transformers) or 'mpnet.'."""
    if _has_tensor(readers, "mpnet.embeddings.word_embeddings.weight"):
        return "mpnet"
    if _has_tensor(readers, "embeddings.word_embeddings.weight"):
        return ""
    return "mpnet"


def _pfx(root, key):
    """Join root prefix with key, handling empty root."""
    return f"{root}.{key}" if root else key


def _compute_relative_position_bias(
    seq_length: int,
    num_buckets: int,
    num_heads: int,
    bias_table: np.ndarray,
) -> np.ndarray:
    """Pre-compute relative position bias matrix [num_heads, seq_len, seq_len].

    Uses the T5-style bucketing scheme (bidirectional):
    - Half the buckets for positive relative positions, half for negative
    - Exact buckets for small relative distances, log-spaced for larger ones
    """
    half_buckets = num_buckets // 2
    max_distance = 128  # T5/MPNet default

    # Relative positions: query_pos - key_pos
    context_position = np.arange(seq_length)[:, None]
    memory_position = np.arange(seq_length)[None, :]
    relative_position = memory_position - context_position  # [seq_len, seq_len]

    # Bidirectional bucketing — mirrors HF's _relative_position_bucket.
    # HF computes: n = -relative_position; offset = (n < 0) * half_buckets.
    # So: positive relative_position → n < 0 → offset = half_buckets
    #     zero/negative rel_pos → n >= 0 → offset = 0
    n = -relative_position  # negate to match HF convention
    buckets = np.zeros_like(n, dtype=np.int32)
    pos_mask = n < 0  # positive relative_position gets offset
    buckets[pos_mask] = half_buckets
    n_abs = np.abs(n)

    max_exact = half_buckets // 2
    is_small = n_abs < max_exact

    val_if_large = max_exact + (
        np.log(n_abs.astype(np.float64) / max_exact + 1e-12)
        / np.log(max_distance / max_exact)
        * (half_buckets - max_exact)
    ).astype(np.int32)
    val_if_large = np.minimum(val_if_large, half_buckets - 1)

    buckets += np.where(is_small, n_abs, val_if_large)

    # Look up bias: bias_table[bucket, head] -> [seq_len, seq_len, num_heads]
    bias = bias_table[buckets]  # [seq_len, seq_len, num_heads]
    # Transpose to [num_heads, seq_len, seq_len]
    return bias.transpose(2, 0, 1).astype(np.float32)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _MPNetModel:
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
        _max_pos = config.max_position_embeddings

        root = _detect_prefix(readers)

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, _pfx(root, "embeddings.word_embeddings.weight"))
        assert embedding.shape == (vocab, hidden)
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding — MPNet uses padding_idx=1, positions start at 2
        pos_embed_raw = _load_tensor(readers, _pfx(root, "embeddings.position_embeddings.weight"))
        pad_idx = config.raw.get("pad_token_id", 1)
        pos_offset = pad_idx + 1
        pos_embed = pos_embed_raw[pos_offset:].astype(np.float32)
        weights["position_embedding"] = pos_embed

        # No token type embedding — synthesize zeros matching type_vocab_size
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

        # Embedding LayerNorm
        ln_w = _load_tensor(readers, _pfx(root, "embeddings.LayerNorm.weight"))
        ln_b = _load_tensor(readers, _pfx(root, "embeddings.LayerNorm.bias"))
        weights["embed_norm"] = ln_w.astype(np.float32)
        weights["embed_norm_beta"] = ln_b.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf = _pfx(root, f"encoder.layer.{layer_idx}")

            # Q, K, V — MPNet uses .attention.attn.{q,k,v}
            q_w = _load_tensor(readers, f"{hf}.attention.attn.q.weight")
            k_w = _load_tensor(readers, f"{hf}.attention.attn.k.weight")
            v_w = _load_tensor(readers, f"{hf}.attention.attn.v.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf}.attention.attn.q.bias"
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf}.attention.attn.k.bias"
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf}.attention.attn.v.bias"
            ).astype(np.float32)

            # Output projection — .attention.attn.o
            o_w = _load_tensor(readers, f"{hf}.attention.attn.o.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf}.attention.attn.o.bias"
            ).astype(np.float32)

            # Post-attention LayerNorm — .attention.LayerNorm
            attn_ln_w = _load_tensor(readers, f"{hf}.attention.LayerNorm.weight")
            attn_ln_b = _load_tensor(readers, f"{hf}.attention.LayerNorm.bias")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b.astype(np.float32)

            # FFN: intermediate.dense -> output.dense
            fc1_w = _load_tensor(readers, f"{hf}.intermediate.dense.weight")
            fc1_b = _load_tensor(readers, f"{hf}.intermediate.dense.bias")
            fc2_w = _load_tensor(readers, f"{hf}.output.dense.weight")
            fc2_b = _load_tensor(readers, f"{hf}.output.dense.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm — .output.LayerNorm
            out_ln_w = _load_tensor(readers, f"{hf}.output.LayerNorm.weight")
            out_ln_b = _load_tensor(readers, f"{hf}.output.LayerNorm.bias")
            weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
            weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

        # Relative attention bias — shared across all layers
        # Shape: [num_buckets, num_heads] -> pre-compute [num_heads, seq_len, seq_len]
        rel_bias_key = _pfx(root, "encoder.relative_attention_bias.weight")
        if _has_tensor(readers, rel_bias_key):
            rel_bias_w = _load_tensor(readers, rel_bias_key).astype(np.float32)
            num_buckets = rel_bias_w.shape[0]
            _num_attn_heads = rel_bias_w.shape[1]
            weights["_relative_attention_bias"] = rel_bias_w
            weights["_relative_attention_num_buckets"] = num_buckets

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
        # Pre-compute relative position bias if present
        if "_relative_attention_bias" in weights:
            bias_table = weights.pop("_relative_attention_bias")
            num_buckets = weights.pop("_relative_attention_num_buckets")
            num_heads = bias_table.shape[1]
            bias_matrix = _compute_relative_position_bias(
                max_cache_length, num_buckets, num_heads, bias_table
            )
            weights["relative_position_bias"] = bias_matrix

        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("MPNet tensor-parallel builds do not support quantization")
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
    if request.dynamic_kv_cache:
        raise NotImplementedError("mpnet does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("mpnet does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("mpnet does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("mpnet does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("mpnet does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("mpnet task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "mpnet":
        raise ValueError(f"MPNet builder requires model_type='mpnet', got {config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("MPNet precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("MPNet max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("MPNet has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("MPNet does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _MPNetModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="mpnet", task=request.task, backend=request.backend)
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
