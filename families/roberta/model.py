# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import json
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
    """Detect the weight prefix used in the checkpoint.

    Returns "roberta" or "model.roberta" depending on which prefix is found.
    XLM-RoBERTa checkpoints sometimes nest under "model.roberta.*".
    """
    if _has_tensor(readers, "model.roberta.embeddings.word_embeddings.weight"):
        return "model.roberta"
    return "roberta"


def _load_ln(readers, prefix):
    """Load LayerNorm weight+bias, handling gamma/beta naming."""
    if _has_tensor(readers, f"{prefix}.weight"):
        w = _load_tensor(readers, f"{prefix}.weight")
        b = _load_tensor(readers, f"{prefix}.bias")
    else:
        w = _load_tensor(readers, f"{prefix}.gamma")
        b = _load_tensor(readers, f"{prefix}.beta")
    return w.astype(np.float32), b.astype(np.float32)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _RobertaModel:
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
        type_vocab_size = config.raw.get("type_vocab_size", 1)

        hf_root = _detect_prefix(readers)

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, f"{hf_root}.embeddings.word_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (learned absolute).
        # RoBERTa has padding_idx=1, so position IDs start at 2.
        # The position embedding table has (max_pos, hidden) rows where
        # rows 0 and 1 are padding-related. Slice starting at offset 2
        # so the encoder builder can use positions [0, 1, ..., N-1].
        pos_embed_raw = _load_tensor(readers, f"{hf_root}.embeddings.position_embeddings.weight")
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
                f"Token type embedding shape {tt_embed.shape} != ({type_vocab_size}, {hidden})"
            )
            weights["token_type_embedding"] = tt_embed.astype(np.float32)
        else:
            weights["token_type_embedding"] = np.zeros((type_vocab_size, hidden), dtype=np.float32)

        # Embedding LayerNorm
        embed_ln_w, embed_ln_b = _load_ln(readers, f"{hf_root}.embeddings.LayerNorm")
        weights["embed_norm"] = embed_ln_w
        weights["embed_norm_beta"] = embed_ln_b

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{hf_root}.encoder.layer.{layer_idx}"

            # Q, K, V projections — HF stores [out, in], transpose to [in, out]
            q_w = _load_tensor(readers, f"{hf_prefix}.attention.self.query.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attention.self.key.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attention.self.value.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # QKV biases
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers,
                f"{hf_prefix}.attention.self.query.bias",
            ).astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers,
                f"{hf_prefix}.attention.self.key.bias",
            ).astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers,
                f"{hf_prefix}.attention.self.value.bias",
            ).astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers,
                f"{hf_prefix}.attention.output.dense.bias",
            ).astype(np.float32)

            # Post-attention LayerNorm (handles gamma/beta)
            attn_ln_w, attn_ln_b = _load_ln(readers, f"{hf_prefix}.attention.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

            # FFN: intermediate.dense -> output.dense
            fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm (handles gamma/beta)
            out_ln_w, out_ln_b = _load_ln(readers, f"{hf_prefix}.output.LayerNorm")
            weights[f"{prefix}.output_norm"] = out_ln_w
            weights[f"{prefix}.output_norm_beta"] = out_ln_b

        # Pooler (optional — used for [CLS] representation)
        pooler_key = f"{hf_root}.pooler.dense.weight"
        if _has_tensor(readers, pooler_key):
            pooler_w = _load_tensor(readers, pooler_key)
            pooler_b = _load_tensor(readers, f"{hf_root}.pooler.dense.bias")
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
                raise ValueError("RoBERTa tensor-parallel builds do not support quantization")
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


def _native_tokenizer_json(model_dir: Path) -> bytes:
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"RoBERTa checkpoint has no tokenizer.json: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid RoBERTa tokenizer.json at {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("RoBERTa tokenizer.json model must be an object")
    model = payload["model"]
    vocab = model.get("vocab")
    if isinstance(vocab, list):
        model_type = "Unigram"
    elif isinstance(vocab, dict):
        model_type = "BPE"
    else:
        raise ValueError("RoBERTa tokenizer.json model.vocab must be an array or object")
    declared_type = model.get("type")
    if declared_type is None:
        model["type"] = model_type
    elif declared_type != model_type:
        raise ValueError("RoBERTa tokenizer.json model.type does not match model.vocab")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.image_height is not None:
        raise NotImplementedError("roberta does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("roberta does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("roberta does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("roberta does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task not in {"encoding", "embedding", "reranking"}:
        raise ValueError("roberta task must be encoding, embedding, or reranking")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"roberta", "xlm-roberta", "camembert"}:
        raise ValueError(
            f"RoBERTa builder requires model_type='roberta', got {config.model_type!r}"
        )
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("RoBERTa precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 512),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("RoBERTa max_sequence_length exceeds checkpoint capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("RoBERTa has no family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("RoBERTa does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _RobertaModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="roberta", task=request.task, backend="trt")
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
    writer.add_bytes("tokenizer.json", _native_tokenizer_json(model_dir))
    for filename in _BUNDLE_FILES:
        if filename == "tokenizer.json":
            continue
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
