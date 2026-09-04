# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from .encoder_builder import build_encoder_engine


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


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _ElectraModel:
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
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("ELECTRA tensor-parallel builds do not support quantization")
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
        raise NotImplementedError("electra does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("electra does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("electra does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("electra does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("electra task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "electra":
        raise ValueError(
            f"ELECTRA builder requires model_type='electra', got {config.model_type!r}"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("ELECTRA precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("ELECTRA max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("ELECTRA has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("ELECTRA does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _ElectraModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="electra", task=request.task, backend="trt")
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
