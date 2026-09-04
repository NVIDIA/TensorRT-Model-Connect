# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_torch_checkpoint,
    _load_tensor,
    _has_tensor,
)
from .parallel import normalize_parallel_config


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


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _XLNetModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_torch_checkpoint(model_dir_path)

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
                readers, f"{hf_prefix}.rel_attn.layer_norm.weight"
            ).astype(np.float32)
            weights[f"{prefix}.attn_norm_beta"] = _load_tensor(
                readers, f"{hf_prefix}.rel_attn.layer_norm.bias"
            ).astype(np.float32)

            weights[f"{prefix}.ff_norm"] = _load_tensor(
                readers, f"{hf_prefix}.ff.layer_norm.weight"
            ).astype(np.float32)
            weights[f"{prefix}.ff_norm_beta"] = _load_tensor(
                readers, f"{hf_prefix}.ff.layer_norm.bias"
            ).astype(np.float32)

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
                raise ValueError("XLNet tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_xlnet_engine

            return build_tp_xlnet_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        from .xlnet_builder import build_xlnet_engine

        return build_xlnet_engine(
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
        raise NotImplementedError("xlnet does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("xlnet does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("xlnet does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("xlnet does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("xlnet task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "xlnet":
        raise ValueError(f"XLNet builder requires model_type='xlnet', got {config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("XLNet precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("XLNet max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("XLNet has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("XLNet does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _XLNetModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="xlnet", task=request.task, backend="trt")
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
