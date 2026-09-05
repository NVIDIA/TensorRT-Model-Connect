# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLOOM family plugin — ALiBi positions + fused QKV + embedding LayerNorm.

BLOOM (BigScience) uses:
  - ALiBi position encoding (no position embeddings)
  - LayerNorm (with beta) instead of RMSNorm
  - Embedding LayerNorm after token embedding lookup
  - Fused QKV projection (query_key_value) — standard Linear layout
  - All linear layers have bias
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
  - Tied word embeddings (word_embeddings == lm_head)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _BloomModel:
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
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads

        weights = WeightDict()

        # Token embedding
        embed_key = "word_embeddings.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "transformer.word_embeddings.weight"
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Embedding LayerNorm
        emb_ln_key_prefix = "word_embeddings_layernorm"
        if not _has_tensor(readers, f"{emb_ln_key_prefix}.weight"):
            emb_ln_key_prefix = "transformer.word_embeddings_layernorm"
        weights["embedding_norm"] = _load_tensor(readers, f"{emb_ln_key_prefix}.weight").astype(
            np.float32
        )
        weights["embedding_norm_beta"] = _load_tensor(readers, f"{emb_ln_key_prefix}.bias").astype(
            np.float32
        )

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"
            # Some BLOOM checkpoints use "h.{i}" without "transformer." prefix
            if not _has_tensor(readers, f"{hf_prefix}.input_layernorm.weight"):
                hf_prefix = f"h.{layer_idx}"

            # Input LayerNorm
            ln1_w = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            ln1_b = _load_tensor(readers, f"{hf_prefix}.input_layernorm.bias")
            weights[f"{prefix}.input_norm"] = ln1_w.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_b.astype(np.float32)

            # Post-attention LayerNorm
            ln2_w = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            ln2_b = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_b.astype(np.float32)

            # Fused QKV: [3*hidden, hidden] — standard Linear layout
            qkv_w = _load_tensor(readers, f"{hf_prefix}.self_attention.query_key_value.weight")
            qkv_b = _load_tensor(readers, f"{hf_prefix}.self_attention.query_key_value.bias")

            # BLOOM stores QKV per-head interleaved (like GPT-NeoX):
            # For each head h, rows [h*3*hd : h*3*hd+hd] are Q,
            # [h*3*hd+hd : h*3*hd+2*hd] are K, [h*3*hd+2*hd : h*3*hd+3*hd] are V.
            q_parts, k_parts, v_parts = [], [], []
            qb_parts, kb_parts, vb_parts = [], [], []
            for h in range(num_heads):
                base = h * 3 * head_dim
                q_parts.append(qkv_w[base : base + head_dim])
                k_parts.append(qkv_w[base + head_dim : base + 2 * head_dim])
                v_parts.append(qkv_w[base + 2 * head_dim : base + 3 * head_dim])
                qb_parts.append(qkv_b[base : base + head_dim])
                kb_parts.append(qkv_b[base + head_dim : base + 2 * head_dim])
                vb_parts.append(qkv_b[base + 2 * head_dim : base + 3 * head_dim])

            q_w = np.concatenate(q_parts, axis=0)
            k_w = np.concatenate(k_parts, axis=0)
            v_w = np.concatenate(v_parts, axis=0)

            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            weights[f"{prefix}.q_bias"] = np.concatenate(qb_parts).astype(np.float32)
            weights[f"{prefix}.k_bias"] = np.concatenate(kb_parts).astype(np.float32)
            weights[f"{prefix}.v_bias"] = np.concatenate(vb_parts).astype(np.float32)

            # Output projection (with bias)
            o_w = _load_tensor(readers, f"{hf_prefix}.self_attention.dense.weight")
            o_b = _load_tensor(readers, f"{hf_prefix}.self_attention.dense.bias")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")
            weights[f"{prefix}.o_bias"] = o_b.astype(np.float32)

            # MLP: dense_h_to_4h (fc1) and dense_4h_to_h (fc2)
            fc1_w = _load_tensor(readers, f"{hf_prefix}.mlp.dense_h_to_4h.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.mlp.dense_h_to_4h.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.mlp.dense_4h_to_h.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.mlp.dense_4h_to_h.bias")

            if mlp_size == 0:
                mlp_size = fc1_w.shape[0]

            # Linear: [out, in] -> transpose to [in, out]
            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_w, "fc1")
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_w, "fc2")
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

        # Final LayerNorm
        fn_w_key = "transformer.ln_f.weight"
        if not _has_tensor(readers, fn_w_key):
            fn_w_key = "ln_f.weight"
        fn_b_key = fn_w_key.replace(".weight", ".bias")
        weights["final_norm"] = _load_tensor(readers, fn_w_key).astype(np.float32)
        weights["final_norm_beta"] = _load_tensor(readers, fn_b_key).astype(np.float32)

        # LM head — BLOOM ties word_embeddings and lm_head
        if _has_tensor(readers, "lm_head.weight"):
            lm_head = _load_tensor(readers, "lm_head.weight")
            weights["w_out"] = _transpose_2d(lm_head, "lm_head")
        else:
            # Tied: reuse embedding [vocab, hidden] -> transpose to [hidden, vocab]
            weights["w_out"] = np.ascontiguousarray(embedding.T.astype(np.float32))

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

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
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="alibi",
                activation="gelu",
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="alibi",
            activation="gelu",
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
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


def _runtime_config(model_dir: Path, config: ModelConfig, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Bloom bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("bloom does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("bloom does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("bloom does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("bloom does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("bloom does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("bloom supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "bloom":
        raise ValueError(f"Bloom does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Bloom precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Bloom max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Bloom has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Bloom does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _BloomModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config)

    writer.set_header(family="bloom", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"

    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
