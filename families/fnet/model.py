# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias."""
    w = _load_tensor(readers, f"{prefix}.weight")
    b = _load_tensor(readers, f"{prefix}.bias")
    return w.astype(np.float32), b.astype(np.float32)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _FNetModel:
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
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("FNet tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_fnet_encoder_engine

            return build_tp_fnet_encoder_engine(
                config,
                weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel,
            )

        from .fnet_encoder_builder import build_fnet_encoder_engine

        return build_fnet_encoder_engine(
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
        raise NotImplementedError("fnet does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("fnet does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("fnet does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("fnet does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("fnet does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("fnet task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "fnet":
        raise ValueError(f"FNet builder requires model_type='fnet', got {config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("FNet precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("FNet max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("FNet has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("FNet does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _FNetModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="fnet", task=request.task, backend=request.backend)
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
            "pad_token_id": config.pad_token_id,
        },
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
