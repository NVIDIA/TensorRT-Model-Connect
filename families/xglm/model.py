# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM family plugin — sinusoidal positions + GELU FC MLP.

XGLM (facebook/xglm-564M) uses:
  - Sinusoidal position embeddings (computed, not learned, with offset=2)
  - LayerNorm (with beta)
  - 2-projection MLP (fc1/fc2) with GELU activation
  - Separate Q/K/V/O projections with biases
  - Separate lm_head (not tied despite what config says)
  - Config uses d_model, ffn_dim, attention_heads, num_layers
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
    _open_torch_checkpoint,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .default_decoder import build_standard_decoder_engine


def _make_sinusoidal_position_embedding(
    num_positions: int,
    embedding_dim: int,
    padding_idx: int = 1,
) -> np.ndarray:
    """Create sinusoidal position embedding table matching HF XGLMSinusoidal."""
    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    positions = np.arange(num_positions, dtype=np.float32)
    emb = positions[:, None] * emb[None, :]
    table = np.zeros((num_positions, embedding_dim), dtype=np.float32)
    table[:, :half_dim] = np.sin(emb)
    table[:, half_dim:] = np.cos(emb)
    table[padding_idx] = 0.0
    return table


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _XGLMModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_torch_checkpoint(model_dir_path)

        # XGLM uses d_model, ffn_dim, attention_heads, num_layers
        hidden = config.hidden_size  # from d_model
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers  # from num_layers
        num_heads = config.num_attention_heads  # from attention_heads

        weights = WeightDict()

        # Token embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape[0] == vocab
        weights["embedding"] = embedding.astype(np.float32)

        # XGLM uses scale_embedding: embed * sqrt(hidden_size)
        scale = config.raw.get("scale_embedding", False)
        if scale:
            weights["embedding"] = weights["embedding"] * np.sqrt(hidden).astype(np.float32)

        # Sinusoidal position embedding (computed, not stored in checkpoint).
        # XGLM uses padding_idx=1 and offset=2 (positions 0,1 unused).
        max_pos = config.max_position_embeddings
        pos_table = _make_sinusoidal_position_embedding(max_pos + 2, hidden, padding_idx=1)
        # Offset: XGLM adds 2 to position indices, so position 0 maps to row 2
        weights["position_embedding"] = pos_table[2:].astype(np.float32)

        attention_size = num_heads * (hidden // num_heads)
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm 1 (pre-attention)
            ln1_w = _load_tensor(readers, f"{hf_prefix}.self_attn_layer_norm.weight")
            ln1_b = _load_tensor(readers, f"{hf_prefix}.self_attn_layer_norm.bias")
            weights[f"{prefix}.input_norm"] = ln1_w.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_b.astype(np.float32)

            # LayerNorm 2 (pre-MLP)
            ln2_w = _load_tensor(readers, f"{hf_prefix}.final_layer_norm.weight")
            ln2_b = _load_tensor(readers, f"{hf_prefix}.final_layer_norm.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_b.astype(np.float32)

            # Q/K/V projections
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.out_proj.weight")

            weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

            # QKV biases
            q_bias = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.bias")
            k_bias = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.bias")
            v_bias = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.bias")
            weights[f"{prefix}.q_bias"] = q_bias.astype(np.float32)
            weights[f"{prefix}.k_bias"] = k_bias.astype(np.float32)
            weights[f"{prefix}.v_bias"] = v_bias.astype(np.float32)

            # Output projection bias
            o_bias_key = f"{hf_prefix}.self_attn.out_proj.bias"
            if _has_tensor(readers, o_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(readers, o_bias_key).astype(np.float32)

            # MLP: fc1 and fc2
            fc1_raw = _load_tensor(readers, f"{hf_prefix}.fc1.weight")
            fc2_raw = _load_tensor(readers, f"{hf_prefix}.fc2.weight")
            if mlp_size == 0:
                mlp_size = fc1_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_raw, "fc1")
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_raw, "fc2")

            # MLP biases
            fc1_bias = _load_tensor(readers, f"{hf_prefix}.fc1.bias")
            fc2_bias = _load_tensor(readers, f"{hf_prefix}.fc2.bias")
            weights[f"{prefix}.fc1_bias"] = fc1_bias.astype(np.float32)
            weights[f"{prefix}.fc2_bias"] = fc2_bias.astype(np.float32)

        # Final LayerNorm
        final_ln_w_key = "model.layer_norm.weight"
        final_ln_b_key = "model.layer_norm.bias"
        if _has_tensor(readers, final_ln_w_key):
            weights["final_norm"] = _load_tensor(readers, final_ln_w_key).astype(np.float32)
            if _has_tensor(readers, final_ln_b_key):
                weights["final_norm_beta"] = _load_tensor(readers, final_ln_b_key).astype(
                    np.float32
                )
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

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
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("XGLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError("XGLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="learned",
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
            position_type="learned",
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
    """Build one XGLM bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("xglm does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("xglm does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("xglm does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("xglm does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("xglm does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("xglm supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "xglm":
        raise ValueError(f"XGLM does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("XGLM precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("XGLM max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("XGLM has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("XGLM does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _XGLMModel()
    config.raw["_model_dir"] = str(model_dir)
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="xglm", task=request.task, backend=request.backend)
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
    tokenizer_override = None
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            writer.add_bytes(filename, tokenizer_override)
        elif path.is_file():
            writer.add_bytes(filename, path.read_bytes())
